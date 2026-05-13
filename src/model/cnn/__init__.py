"""
CNN-based volumetric feature extractors.

Re-exports :class:`MINDModel` and provides :func:`get_cnn_model`, a
narrow factory that constructs CNN models by short name. For full
coverage of the CNN family (``"mind"``, ``"anatomix"``, ``"anamind"``)
use the top-level :func:`src.model.get_model` factory instead.
"""

from .mind import MINDModel

def get_cnn_model(name: str, **kwargs) -> MINDModel:
    """Construct a CNN model by short name.

    Parameters
    ----------
    name
        Currently only ``"mind"`` is supported here.
    **kwargs
        Forwarded to the model constructor.

    Returns
    -------
    MINDModel

    Raises
    ------
    ValueError
        If *name* is unknown.

    See Also
    --------
    src.model.get_model
        Broader factory covering ``"anatomix"`` and ``"anamind"`` as well.
    """
    name = name.lower()
    if name == "mind":
        return MINDModel(**kwargs)
    else:
        raise ValueError(f"Unknown CNN model '{name}'. Supported: 'mind'.")