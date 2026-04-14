def build_loss(use_wiou: bool = False) -> str:
    """
    Placeholder loss selector.
    Extend this module when customizing Ultralytics trainer internals.
    """
    return "wiou" if use_wiou else "ciou"
