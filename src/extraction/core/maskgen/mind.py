from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import torch

from src.model.cnn import MINDModel

from .base import BaseMaskGenerator
from .defaults import MINDMaskGeneratorConfig, get_default_mind_maskgen_config
from .utils import binary_dilate3d, fill_holes_3d


class MINDMaskGenerator(BaseMaskGenerator):
    """
    Generate foreground masks from MIND features.

    Rule
    ----
    For MIND feature volume mf of shape (D,H,W,C):

        fg = ~(all(mf > msk_th, dim=-1))

    Then:
    - fill enclosed holes
    - optionally dilate foreground

    Notes
    -----
    This generator is logically stateless with respect to data fitting.
    Therefore:
    - fit() is a no-op except runtime-device/model refresh
    - fitted is always True
    """

    MASKGEN_NAME = "mind"

    def __init__(
        self,
        config: Optional[Union[MINDMaskGeneratorConfig, Dict[str, Any]]] = None,
        default_config_name: str = "default",
        preferred_device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        super().__init__(preferred_device=preferred_device)

        if config is None:
            self.config = get_default_mind_maskgen_config(default_config_name)
        elif isinstance(config, MINDMaskGeneratorConfig):
            self.config = config
        elif isinstance(config, dict):
            self.config = MINDMaskGeneratorConfig.from_dict(config)
        else:
            raise TypeError(f"Unsupported config type: {type(config)}")

        self.mind_model = self._build_mind_model()
        self.fitted = True

    def _build_mind_model(self) -> MINDModel:
        return MINDModel(
            radius=int(self.config.mind_r),
            dilation=int(self.config.mind_d),
            use_mask=False,
        )

    def _ensure_runtime_ready(self) -> None:
        self._refresh_runtime_device()

    def fit(self, batch: Dict[str, Any]) -> "MINDMaskGenerator":
        """
        No-op fit for compatibility with project-native fit/transform API.
        """
        self._ensure_runtime_ready()
        self.mind_model = self._build_mind_model()
        self.fitted = True
        return self

    def get_config(self) -> Dict[str, Any]:
        return self.config.to_dict()

    @property
    def expr(self) -> str:
        return (
            "mindmask("
            f"th={self.config.msk_th},"
            f"r={self.config.mind_r},"
            f"d={self.config.mind_d},"
            f"conn={self.config.connectivity},"
            f"dil={self.config.final_dilate}"
            ")"
        )

    def _features_to_mask(self, mf: torch.Tensor) -> torch.Tensor:
        if not isinstance(mf, torch.Tensor):
            raise TypeError(f"MIND features must be torch.Tensor, got {type(mf)}")
        if mf.ndim != 4:
            raise ValueError(
                f"Expected MIND feature tensor of shape (D,H,W,C), got {tuple(mf.shape)}"
            )

        mask = ~torch.all((mf > float(self.config.msk_th)), dim=-1).to(torch.bool)
        return mask

    def _postprocess_mask(self, mask: torch.Tensor) -> torch.Tensor:
        mask = fill_holes_3d(
            mask,
            connectivity=int(self.config.connectivity),
            max_iters=self.config.max_iters,
        )

        if int(self.config.final_dilate) > 0:
            mask = binary_dilate3d(mask, kernel_size=int(self.config.final_dilate))

        return mask

    def transform(self, batch: Dict[str, Any]) -> List[torch.Tensor]:
        self._ensure_runtime_ready()

        mind_features = self.mind_model.extract(batch)
        if not isinstance(mind_features, list):
            raise TypeError(
                f"MINDModel.extract(batch) must return a list, got {type(mind_features)}"
            )

        masks: List[torch.Tensor] = []
        for mf in mind_features:
            raw = self._features_to_mask(mf)
            proc = self._postprocess_mask(raw)
            masks.append(proc)

        return masks

    def state_dict(self) -> Dict[str, Any]:
        state = super().state_dict()
        state.update(
            {
                "class_name": self.__class__.__name__,
                "config": self.get_config(),
                "expr": self.expr,
                "summary": self.summary,
            }
        )
        return state

    def load_state_dict(self, state: Dict[str, Any]) -> "MINDMaskGenerator":
        super().load_state_dict(state)

        cfg = state.get("config", None)
        self.config = MINDMaskGeneratorConfig.from_dict(cfg)
        self.mind_model = self._build_mind_model()

        # By design, always logically fitted.
        self.fitted = True
        return self