from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional, Union

import torch

from .utils import clone_batch_shallow, resolve_preferred_device


class BaseMaskGenerator(ABC):
    """
    Project-native base class for generated masks.

    Main API
    --------
    - fit(batch)
    - transform(batch) -> List[torch.Tensor]
    - fit_transform(batch) -> List[torch.Tensor]
    - __call__(batch) -> List[torch.Tensor]
    - return_batch(batch, inplace=False) -> batch with "genmsks"

    Notes
    -----
    Subclasses may be logically stateless. In that case, `fit()` can be a no-op.
    """

    MASKGEN_NAME = "base"

    def __init__(self, preferred_device: Optional[Union[str, torch.device]] = None) -> None:
        self.preferred_device = None if preferred_device is None else str(torch.device(preferred_device))
        self.device = resolve_preferred_device(preferred_device)
        self.fitted = False

    @property
    def expr(self) -> str:
        return f"{self.MASKGEN_NAME}()"

    @property
    def summary(self) -> Dict[str, Any]:
        return {
            "type": self.MASKGEN_NAME,
            "expr": self.expr,
            "preferred_device": self.preferred_device,
            "device": str(self.device),
            "fitted": self.fitted,
            "config": self.get_config(),
        }

    def get_config(self) -> Dict[str, Any]:
        """
        Return a JSON-serializable configuration dict.
        Subclasses should override when they hold configs.
        """
        return {}

    def _refresh_runtime_device(self) -> torch.device:
        self.device = resolve_preferred_device(self.preferred_device)
        return self.device

    def fit(self, batch: Dict[str, Any]) -> "BaseMaskGenerator":
        """
        Default fit hook. Subclasses may override.
        """
        self._refresh_runtime_device()
        self.fitted = True
        return self

    @abstractmethod
    def transform(self, batch: Dict[str, Any]) -> List[torch.Tensor]:
        raise NotImplementedError

    def fit_transform(self, batch: Dict[str, Any]) -> List[torch.Tensor]:
        self.fit(batch)
        return self.transform(batch)

    def __call__(self, batch: Dict[str, Any]) -> List[torch.Tensor]:
        return self.transform(batch)

    def return_batch(self, batch: Dict[str, Any], inplace: bool = False) -> Dict[str, Any]:
        """
        Return batch with generated masks under key "genmsks".

        Parameters
        ----------
        batch:
            Project batch dict.
        inplace:
            If False, shallow-copy the top-level batch structure first.

        Returns
        -------
        Dict[str, Any]
            Batch containing:
                batch["genmsks"] = List[torch.Tensor]
        """
        out = batch if inplace else clone_batch_shallow(batch)
        out["genmsks"] = self.transform(batch)
        return out

    def state_dict(self) -> Dict[str, Any]:
        return {
            "type": self.MASKGEN_NAME,
            "preferred_device": self.preferred_device,
            "fitted": self.fitted,
            "config": self.get_config(),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> "BaseMaskGenerator":
        self.preferred_device = state.get("preferred_device", self.preferred_device)
        self._refresh_runtime_device()
        self.fitted = bool(state.get("fitted", self.fitted))
        return self