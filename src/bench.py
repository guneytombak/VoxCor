"""
src/bench.py

Lightweight, self-contained benchmarking utilities for measuring
GPU memory, CPU/RAM usage, and wall-clock time across experiment stages
(feature extraction, registration, etc.).

Design goals:
  - Zero overhead when not actively measuring (no background threads).
  - Minimal import footprint (stdlib + torch + optional psutil).
  - Composable: use as context managers, decorators, or explicit start/stop.
  - Hierarchical: nest benchmarks to get per-stage AND total breakdowns.
  - JSON-serialisable reports for easy logging.

Usage examples::

    from src.bench import Bench, BenchSuite

    # ── Single stage ────────────────────────────────────────
    with Bench("fit", device="cuda") as b:
        model.fit(data)
    print(b.report())
    # {'tag': 'fit', 'sec': 12.3, 'gpu_peak_mb': 4200.5, 'ram_delta_mb': 120.3, ...}

    # ── Nested stages ───────────────────────────────────────
    suite = BenchSuite("vit3d_pipeline", device="cuda")
    with suite.stage("fit"):
        model.fit(data)
    with suite.stage("transform"):
        feats = model.transform(data)
    with suite.stage("save"):
        model.save_pt("out.pt")
    print(suite.summary())       # per-stage + total
    suite.save_json("bench.json")

    # ── Decorator ───────────────────────────────────────────
    @Bench.wrap("my_func", device="cuda")
    def my_func():
        ...
"""

from __future__ import annotations

import gc
import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from functools import wraps
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch

# ─────────────────────────────────────────────────────────────
# Optional: psutil for RAM measurement (graceful fallback)
# ─────────────────────────────────────────────────────────────

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


def _get_ram_mb() -> float:
    """
    Return the current process resident memory usage in megabytes.

    The value corresponds to RSS memory reported by ``psutil`` for the
    current Python process. If ``psutil`` is not installed, the function
    returns ``-1.0`` so that benchmarking code can continue without an
    optional dependency.

    Returns
    -------
    float
        Current process RSS memory in MB, or ``-1.0`` when unavailable.
    """
    if not _HAS_PSUTIL:
        return -1.0
    return psutil.Process().memory_info().rss / 1024**2


# ─────────────────────────────────────────────────────────────
# Bench — single-stage benchmark
# ─────────────────────────────────────────────────────────────

@dataclass
class BenchResult:
    """
    Container for one completed benchmark measurement.

    The result stores wall-clock time, CUDA memory statistics, and process
    RAM usage before and after a measured code block. All memory values are
    expressed in megabytes. CUDA-related fields are set to ``-1.0`` when the
    benchmark did not run on a CUDA device. RAM-related fields are set to
    ``-1.0`` when ``psutil`` is unavailable.

    Attributes
    ----------
    tag :
        Human-readable label identifying the measured stage.
    sec :
        Wall-clock duration in seconds.
    gpu_peak_alloc_mb :
        Peak CUDA memory allocated during the measured block.
    gpu_peak_reserved_mb :
        Peak CUDA memory reserved by the CUDA caching allocator.
    gpu_alloc_before_mb :
        CUDA memory allocated immediately before the measured block.
    gpu_alloc_after_mb :
        CUDA memory allocated immediately after the measured block.
    gpu_alloc_delta_mb :
        Difference between CUDA memory allocated after and before the block.
    ram_before_mb :
        Process RSS memory before the measured block.
    ram_after_mb :
        Process RSS memory after the measured block.
    ram_delta_mb :
        Difference between process RSS memory after and before the block.
    """

    tag: str = ""

    # Wall-clock
    sec: float = 0.0

    # GPU (all in MB; -1 if CUDA not used)
    gpu_peak_alloc_mb: float = -1.0
    gpu_peak_reserved_mb: float = -1.0
    gpu_alloc_before_mb: float = -1.0
    gpu_alloc_after_mb: float = -1.0
    gpu_alloc_delta_mb: float = -1.0

    # RAM (all in MB; -1 if psutil unavailable)
    ram_before_mb: float = -1.0
    ram_after_mb: float = -1.0
    ram_delta_mb: float = -1.0

    def as_dict(self) -> Dict[str, Any]:
        """
        Return the benchmark result as a JSON-friendly dictionary.

        Floating-point values are rounded to four decimal places for readability
        while preserving all fields in the dataclass.

        Returns
        -------
        dict
            Dictionary representation of the benchmark result.
        """
        d = asdict(self)
        # Round floats for readability
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}

    def short(self) -> str:
        """
        Return a compact one-line human-readable summary.

        The summary includes the tag, wall-clock time, CUDA peak allocation and
        allocation delta when available, and RAM delta when available.

        Returns
        -------
        str
            Compact textual summary of the benchmark result.
        """
        parts = [f"{self.tag}" if self.tag else "bench"]
        parts.append(f"{self.sec:.3f}s")
        if self.gpu_peak_alloc_mb >= 0:
            parts.append(f"gpu_peak={self.gpu_peak_alloc_mb:.0f}MB")
            parts.append(f"gpu_Δ={self.gpu_alloc_delta_mb:+.0f}MB")
        if self.ram_delta_mb >= 0:
            parts.append(f"ram_Δ={self.ram_delta_mb:+.0f}MB")
        return "  ".join(parts)

    def __repr__(self) -> str:
        return f"BenchResult({self.short()})"


class Bench:
    """
    Context manager and decorator for measuring one execution stage.

    ``Bench`` measures wall-clock time unconditionally. When a CUDA device is
    used, it also records CUDA memory allocated before and after the measured
    block, peak allocated memory, and peak reserved memory. When ``psutil`` is
    installed, it additionally records process RAM usage before and after the
    block.

    The class is intended for lightweight instrumentation of expensive
    pipeline stages such as feature extraction, projection fitting,
    registration, saving, or evaluation. It performs no background monitoring;
    measurements are taken only at entry and exit.

    Parameters
    ----------
    tag :
        Human-readable label for the measured stage.
    device :
        Device used for CUDA memory accounting. If ``None``, CUDA is used when
        available; otherwise CPU-only timing is performed.
    sync :
        If ``True``, synchronize the CUDA device before starting and after
        finishing the measured block. This gives more accurate GPU timing.
    gc_before :
        If ``True``, run Python garbage collection and clear the CUDA cache
        before the measured block to reduce measurement noise.
    reset_peak :
        If ``True``, reset CUDA peak memory statistics before the measured
        block so that peak values refer only to this measurement.

    Attributes
    ----------
    result :
        A :class:`BenchResult` containing the most recent completed
        measurement.
    """

    def __init__(
        self,
        tag: str = "",
        *,
        device: Optional[Union[str, torch.device]] = None,
        sync: bool = True,
        gc_before: bool = True,
        reset_peak: bool = True,
    ):
        self.tag = tag
        self._sync = sync
        self._gc_before = gc_before
        self._reset_peak = reset_peak

        # Resolve device
        if device is None:
            self._use_cuda = torch.cuda.is_available()
            self._device = torch.device("cuda") if self._use_cuda else torch.device("cpu")
        else:
            self._device = torch.device(device)
            self._use_cuda = self._device.type == "cuda"

        self.result: BenchResult = BenchResult(tag=tag)

    # ── Context manager ─────────────────────────────────────

    def __enter__(self) -> "Bench":
        """
        Start the benchmark measurement.

        Performs optional garbage collection, optional CUDA cache clearing,
        optional CUDA synchronization, and records the initial RAM and CUDA memory
        state.

        Returns
        -------
        Bench
            The active benchmark object.
        """
        if self._gc_before:
            gc.collect()
            if self._use_cuda:
                torch.cuda.empty_cache()

        if self._use_cuda and self._sync:
            torch.cuda.synchronize(self._device)

        # Snapshot before
        self._ram_before = _get_ram_mb()
        if self._use_cuda:
            if self._reset_peak:
                torch.cuda.reset_peak_memory_stats(self._device)
            self._gpu_alloc_before = torch.cuda.memory_allocated(self._device) / 1024**2
        else:
            self._gpu_alloc_before = -1.0

        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *_exc) -> None:
        """
        Finish the benchmark measurement and store the result.

        Synchronizes CUDA if requested, records elapsed wall-clock time, collects
        final RAM and CUDA memory values, and writes the completed measurement to
        ``self.result``.

        Parameters
        ----------
        *_exc :
            Exception information passed by the context manager protocol. The
            benchmark does not suppress exceptions.
        """
        if self._use_cuda and self._sync:
            torch.cuda.synchronize(self._device)

        elapsed = time.perf_counter() - self._t0

        # Snapshot after
        ram_after = _get_ram_mb()
        if self._use_cuda:
            gpu_alloc_after  = torch.cuda.memory_allocated(self._device) / 1024**2
            gpu_peak_alloc   = torch.cuda.max_memory_allocated(self._device) / 1024**2
            gpu_peak_reserve = torch.cuda.max_memory_reserved(self._device) / 1024**2
        else:
            gpu_alloc_after = gpu_peak_alloc = gpu_peak_reserve = -1.0

        self.result = BenchResult(
            tag=self.tag,
            sec=elapsed,
            gpu_peak_alloc_mb=gpu_peak_alloc,
            gpu_peak_reserved_mb=gpu_peak_reserve,
            gpu_alloc_before_mb=self._gpu_alloc_before,
            gpu_alloc_after_mb=gpu_alloc_after,
            gpu_alloc_delta_mb=(
                gpu_alloc_after - self._gpu_alloc_before
                if self._gpu_alloc_before >= 0 else -1.0
            ),
            ram_before_mb=self._ram_before,
            ram_after_mb=ram_after,
            ram_delta_mb=(
                ram_after - self._ram_before
                if self._ram_before >= 0 else -1.0
            ),
        )

    # ── Convenience accessors ────────────────────────────────

    def report(self) -> Dict[str, Any]:
        """
        Return the completed benchmark result as a JSON-friendly dictionary.

        Returns
        -------
        dict
            Dictionary representation of ``self.result``.
        """
        return self.result.as_dict()

    # ── Decorator factory ────────────────────────────────────

    @staticmethod
    def wrap(tag: str = "", **bench_kwargs):
        """
        Create a decorator that benchmarks each call to a function.

        The wrapped function behaves normally and returns its original return
        value. After each call, the completed :class:`BenchResult` is attached to
        the wrapper as ``wrapper.bench_result``.

        Parameters
        ----------
        tag :
            Benchmark label. If empty, the wrapped function's name is used.
        **bench_kwargs :
            Additional keyword arguments forwarded to :class:`Bench`, such as
            ``device``, ``sync``, ``gc_before``, or ``reset_peak``.

        Returns
        -------
        Callable
            Decorator that wraps a function with benchmarking instrumentation.

        Examples
        --------
        >>> @Bench.wrap("extract_features", device="cuda")
        ... def extract_features(x):
        ...     return model(x)
        >>> y = extract_features(x)
        >>> print(extract_features.bench_result)
        """
        def decorator(fn):
            @wraps(fn)
            def wrapper(*args, **kwargs):
                _tag = tag or fn.__name__
                b = Bench(_tag, **bench_kwargs)
                with b:
                    result = fn(*args, **kwargs)
                wrapper.bench_result = b.result
                return result
            wrapper.bench_result = None
            return wrapper
        return decorator


# ─────────────────────────────────────────────────────────────
# BenchSuite — multi-stage benchmark collector
# ─────────────────────────────────────────────────────────────

class BenchSuite:
    """
    Collector for multiple named benchmark stages.

    ``BenchSuite`` manages a sequence of :class:`Bench` measurements and
    provides JSON-serialisable summaries, readable text summaries, and
    save/load helpers. It is useful for measuring multi-stage pipelines where
    individual stages and the overall runtime should be reported together.

    A suite can optionally measure a total runtime using :meth:`total`. The
    total measurement includes gaps between stages, whereas ``stages_sum_sec``
    only sums the explicitly measured stages.

    Parameters
    ----------
    name :
        Human-readable suite name.
    device :
        Default device forwarded to each stage unless overridden.
    **kwargs :
        Default keyword arguments forwarded to each :class:`Bench`, such as
        ``sync``, ``gc_before``, or ``reset_peak``.

    Attributes
    ----------
    name :
        Name of the benchmark suite.
    stages :
        Completed per-stage benchmark results.
    total_result :
        Optional completed total benchmark result.
    """

    def __init__(
        self,
        name: str = "bench",
        *,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ):
        self.name = name
        self._device = device
        self._kwargs = kwargs
        self._stages: List[BenchResult] = []
        self._total_bench: Optional[Bench] = None

    @contextmanager
    def stage(self, tag: str, **override_kwargs):
        """
        Benchmark one named stage and append its result to the suite.

        Suite-level benchmark options are used by default. Any keyword arguments
        passed here override the suite defaults for this stage only.

        Parameters
        ----------
        tag :
            Human-readable stage name.
        **override_kwargs :
            Stage-specific keyword arguments forwarded to :class:`Bench`.

        Yields
        ------
        Bench
            Active benchmark object for the stage.

        Examples
        --------
        >>> suite = BenchSuite("pipeline", device="cuda")
        >>> with suite.stage("fit"):
        ...     model.fit(data)
        """
        kw = {**self._kwargs}
        if self._device is not None and "device" not in override_kwargs:
            kw["device"] = self._device
        kw.update(override_kwargs)

        b = Bench(tag, **kw)
        with b:
            yield b
        self._stages.append(b.result)

    @contextmanager
    def total(self):
        """
        Benchmark the total runtime of a complete suite block.

        This context manager is optional. Unlike the sum of individual stage
        times, the total includes any work or idle time between nested
        ``suite.stage(...)`` blocks.

        Yields
        ------
        None

        Examples
        --------
        >>> suite = BenchSuite("pipeline", device="cuda")
        >>> with suite.total():
        ...     with suite.stage("fit"):
        ...         model.fit(data)
        ...     with suite.stage("transform"):
        ...         model.transform(data)
        """
        kw = {**self._kwargs, "gc_before": False, "reset_peak": False}
        if self._device is not None:
            kw["device"] = self._device
        b = Bench(f"{self.name}__total", **kw)
        with b:
            yield
        self._total_bench = b

    # ── Results ──────────────────────────────────────────────

    @property
    def stages(self) -> List[BenchResult]:
        """
        Return a copy of the completed stage results.

        Returns
        -------
        list of BenchResult
            Per-stage benchmark results collected so far.
        """
        return list(self._stages)

    @property
    def total_result(self) -> Optional[BenchResult]:
        """
        Return the optional total benchmark result.

        Returns
        -------
        BenchResult or None
            Total benchmark result if :meth:`total` has been used, otherwise
            ``None``.
        """
        return self._total_bench.result if self._total_bench else None

    def summary(self) -> Dict[str, Any]:
        """
        Return a JSON-serialisable summary of the benchmark suite.

        The summary contains the suite name, all completed stages, the sum of
        per-stage runtimes, and the optional total measurement.

        Returns
        -------
        dict
            JSON-friendly benchmark summary.
        """
        stages = [r.as_dict() for r in self._stages]
        total  = self._total_bench.result.as_dict() if self._total_bench else None
        return {
            "name": self.name,
            "stages": stages,
            "stages_sum_sec": round(sum(r.sec for r in self._stages), 4),
            "total": total,
        }

    def summary_str(self, indent: int = 2) -> str:
        """
        Return a readable table summarizing all benchmark stages.

        Parameters
        ----------
        indent :
            Number of leading spaces reserved for table rows. Currently kept for
            API compatibility; the table formatting uses a fixed layout.

        Returns
        -------
        str
            Pretty-printed benchmark table.
        """
        lines = [f"BenchSuite: {self.name}", "─" * 72]
        col = f"  {'stage':<28s}  {'sec':>8s}  {'gpu_peak':>10s}  {'gpu_Δ':>9s}  {'ram_Δ':>9s}"
        lines.append(col)
        lines.append("─" * 72)

        for r in self._stages:
            gpu_p = f"{r.gpu_peak_alloc_mb:.0f}MB" if r.gpu_peak_alloc_mb >= 0 else "n/a"
            gpu_d = f"{r.gpu_alloc_delta_mb:+.0f}MB" if r.gpu_alloc_delta_mb != -1 else "n/a"
            ram_d = f"{r.ram_delta_mb:+.0f}MB" if r.ram_delta_mb != -1 else "n/a"
            lines.append(f"  {r.tag:<28s}  {r.sec:>8.3f}  {gpu_p:>10s}  {gpu_d:>9s}  {ram_d:>9s}")

        lines.append("─" * 72)
        sum_sec = sum(r.sec for r in self._stages)
        lines.append(f"  {'(stages sum)':<28s}  {sum_sec:>8.3f}")

        if self._total_bench:
            r = self._total_bench.result
            gpu_p = f"{r.gpu_peak_alloc_mb:.0f}MB" if r.gpu_peak_alloc_mb >= 0 else "n/a"
            gpu_d = f"{r.gpu_alloc_delta_mb:+.0f}MB" if r.gpu_alloc_delta_mb != -1 else "n/a"
            ram_d = f"{r.ram_delta_mb:+.0f}MB" if r.ram_delta_mb != -1 else "n/a"
            lines.append(f"  {'TOTAL':<28s}  {r.sec:>8.3f}  {gpu_p:>10s}  {gpu_d:>9s}  {ram_d:>9s}")

        return "\n".join(lines)

    def save_json(self, path: Union[str, Path]) -> None:
        """
        Save the benchmark suite summary to a JSON file.

        Parent directories are created automatically.

        Parameters
        ----------
        path :
            Output JSON path.
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w") as f:
            json.dump(self.summary(), f, indent=2)

    @classmethod
    def load_json(
        cls,
        path: Union[str, Path],
        *,
        name: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ) -> "BenchSuite":
        """
        Load a benchmark suite from a previously saved JSON file.

        Previously recorded stages are restored so that new stages can be appended
        in resumed or fault-tolerant runs. Unknown fields in old checkpoint files
        are ignored for forward/backward compatibility. If the file does not
        exist, an empty suite is returned.

        Parameters
        ----------
        path :
            Path to a JSON file produced by :meth:`save_json`.
        name :
            Optional suite-name override. If omitted, the stored name is used.
        device :
            Default device for newly added stages after loading.
        **kwargs :
            Default keyword arguments forwarded to :class:`Bench` for newly added
            stages.

        Returns
        -------
        BenchSuite
            Restored or newly initialized benchmark suite.
        """
        p = Path(path)
        if not p.exists():
            return cls(name=name or "bench", device=device, **kwargs)

        with p.open("r") as f:
            data = json.load(f)

        suite_name = name if name is not None else data.get("name", "bench")
        suite = cls(suite_name, device=device, **kwargs)

        known_fields = set(BenchResult.__dataclass_fields__)   # type: ignore[attr-defined]
        for stage_dict in data.get("stages", []):
            # Filter to known fields so old checkpoints with extra keys don't break.
            filtered = {k: v for k, v in stage_dict.items() if k in known_fields}
            suite._stages.append(BenchResult(**filtered))

        return suite

    def __repr__(self) -> str:
        n = len(self._stages)
        return f"BenchSuite({self.name!r}, {n} stages)"


# ─────────────────────────────────────────────────────────────
# Snapshot — instant point-in-time memory reading (no timing)
# ─────────────────────────────────────────────────────────────

def memory_snapshot(
    tag: str = "",
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, Any]:
    """
    Take a point-in-time memory snapshot without timing a code block.

    This helper is useful for lightweight logging at arbitrary points in a
    pipeline. It records current CUDA allocated/reserved memory and current
    process RSS memory when available. Unlike :class:`Bench`, it does not
    reset peak statistics, synchronize CUDA, or measure elapsed time.

    Parameters
    ----------
    tag :
        Human-readable label attached to the snapshot.
    device :
        Device used for CUDA memory accounting. If ``None``, CUDA is used when
        available.

    Returns
    -------
    dict
        JSON-friendly memory snapshot containing ``tag``, CUDA memory fields,
        and ``ram_mb``.
    """
    snap: Dict[str, Any] = {"tag": tag}

    if device is None:
        use_cuda = torch.cuda.is_available()
        dev = torch.device("cuda") if use_cuda else torch.device("cpu")
    else:
        dev = torch.device(device)
        use_cuda = dev.type == "cuda"

    if use_cuda:
        snap["gpu_alloc_mb"]    = round(torch.cuda.memory_allocated(dev) / 1024**2, 2)
        snap["gpu_reserved_mb"] = round(torch.cuda.memory_reserved(dev) / 1024**2, 2)
        snap["gpu_peak_alloc_mb"] = round(torch.cuda.max_memory_allocated(dev) / 1024**2, 2)
    else:
        snap["gpu_alloc_mb"] = snap["gpu_reserved_mb"] = snap["gpu_peak_alloc_mb"] = -1.0

    snap["ram_mb"] = round(_get_ram_mb(), 2)
    return snap


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────

__all__ = [
    "Bench",
    "BenchResult",
    "BenchSuite",
    "memory_snapshot",
]
