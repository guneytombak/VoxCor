from typing import Dict, Any
from .utils import BaseViT3DWrapper

MODEL_REGISTRY: Dict[str, str] = {
    "basicadd2mind": "src.registration.wrapvit3d.add2mind:BasicAdd2MINDViT3DWrapper"
}

def get_vit3d_wrapper(wrapper_config: Dict[str, Any], model) -> BaseViT3DWrapper:
    wrapper_params = wrapper_config.copy()
    wrapper_name = wrapper_params.pop("name")
    if wrapper_name not in MODEL_REGISTRY:
        raise ValueError(f"Wrapper {wrapper_name} not found in MODEL_REGISTRY. " +\
            f"Available wrappers: {list(MODEL_REGISTRY.keys())}")
    
    wrapper_class_path = MODEL_REGISTRY[wrapper_name]
    module_path, class_name = wrapper_class_path.rsplit(":", 1)
    module = __import__(module_path, fromlist=[class_name])
    wrapper_class = getattr(module, class_name)
    
    return wrapper_class(model=model, **wrapper_params)
    
    