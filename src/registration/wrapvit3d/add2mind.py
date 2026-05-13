import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional, List
from copy import deepcopy

from src.model.cnn.mind import MINDModel
from src.extraction.preprocess import PreprocessPipeline
from .utils import (BaseViT3DWrapper, AutoSelector, select_indices_from_feature_pack, 
                    MultiAxisFeaturePack, WrappedMultiAxisFeaturePack, __MULTIAXIS_FEATURE_PACK_MAIN_FEATURE_NAMES__)

class BasicAdd2MINDViT3DWrapper(BaseViT3DWrapper):
    def __init__(self, model, select:str="[:]", norm:str="none", mult:float=1.0, 
                 mind_params:Dict[str, Any] = {"radius": 1, "dilation": 2, "use_mask": False}):
        
        super(BasicAdd2MINDViT3DWrapper, self).__init__(model=model)
        self.mind = MINDModel(**mind_params)
        self.select = AutoSelector(select)
        self.norm = norm
        self.mult = mult

    @torch.inference_mode()
    def transform(self,
        batch: Dict[str, Any],
        local_pp: Optional[PreprocessPipeline] = None,
    ) -> List[MultiAxisFeaturePack]:

        if local_pp is not None:
            raise NotImplementedError("Local preprocess pipeline is not supported in BasicAdd2MINDViT3DWrapper yet.")
        
        list_of_mafps : List[MultiAxisFeaturePack] = self.model.transform(batch, local_pp=local_pp)
        list_of_mind_features : List[torch.Tensor] = self.compute_mind_features(batch)

        list_of_mafps = select_indices_from_feature_pack(list_of_mafps, self.select)
        
        for i in range(len(list_of_mafps)):

            if self.norm == "l2":
                for feature_name in __MULTIAXIS_FEATURE_PACK_MAIN_FEATURE_NAMES__:
                    raw_feature = getattr(list_of_mafps[i], feature_name)
                    if raw_feature is not None:
                        normed_feature = deepcopy(raw_feature)
                        normed_feature.data = F.normalize(raw_feature.data, p=2, dim=-1)
                        setattr(list_of_mafps[i], feature_name, normed_feature)

            if self.mult != 1.0:
                for feature_name in __MULTIAXIS_FEATURE_PACK_MAIN_FEATURE_NAMES__:
                    raw_feature = getattr(list_of_mafps[i], feature_name)
                    if raw_feature is not None:
                        multed_feature = deepcopy(raw_feature)
                        multed_feature.data = raw_feature.data * self.mult
                        setattr(list_of_mafps[i], feature_name, multed_feature)

        vids_from_mafps = [mafp.vid for mafp in list_of_mafps]
        vids_from_batch = batch["vids"]
        assert vids_from_mafps == vids_from_batch, f"VIDs from MAFPs do not match VIDs from batch " +\
            f"MAFPs: {vids_from_mafps}, Batch: {vids_from_batch}"

        for i in range(len(list_of_mafps)):
            list_of_mafps[i] = WrappedMultiAxisFeaturePack(
                original=list_of_mafps[i],
                addition=list_of_mind_features[i]
            )

        return list_of_mafps

    def compute_mind_features(self, batch: Dict[str, Any]) -> List[torch.Tensor]:

        vols = batch["vols"]  # [(D, H, W)]
        masks = batch["msks"]  # [(D, H, W)]

        mind_features = []

        for vol, mask in zip(vols, masks):

            vol_torch = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0).to(self.mind.device)  # (1, 1, D, H, W)
            if mask is None:
                mask_torch = None
            else:
                mask_torch = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(self.mind.device)  # (1, 1, D, H, W)
            mind_feat = self.mind(vol_torch, mask_torch).squeeze(0).permute(1, 2, 3, 0).cpu()  # (D, H, W, C)
            mind_features.append(mind_feat)

        return mind_features