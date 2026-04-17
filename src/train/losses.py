from __future__ import annotations

from typing import Any

import torch
from ultralytics.utils import metrics

_ORIGINAL_BBOX_IOU = metrics.bbox_iou
_WIOU_PATCH_APPLIED = False


def _wiou_from_iou(iou: torch.Tensor, gamma: float = 1.9, delta: float = 3.0) -> torch.Tensor:
    """
    Dynamic focusing re-weighting over IoU to emulate Wise-IoU behavior.
    Returned value keeps bbox_iou contract (higher is better, <= 1.0).
    """
    iou_detached = iou.detach().clamp(0.0, 1.0)
    focus = (delta * torch.pow(iou_detached, gamma)).clamp(0.0, 10.0)
    penalty = (1.0 - iou).clamp(min=1e-6)
    wiou_score = 1.0 - penalty * focus
    return wiou_score.clamp(-1.0, 1.0)


def _bbox_iou_with_wiou(*args: Any, **kwargs: Any) -> torch.Tensor:
    # Ultralytics loss path requests CIoU=True for bbox regression.
    # We intercept only that case and replace CIoU score with WIoU-style score.
    if not kwargs.get("CIoU", False):
        return _ORIGINAL_BBOX_IOU(*args, **kwargs)

    patched_kwargs = dict(kwargs)
    patched_kwargs["CIoU"] = False
    patched_kwargs["DIoU"] = False
    patched_kwargs["GIoU"] = False
    base_iou = _ORIGINAL_BBOX_IOU(*args, **patched_kwargs)
    return _wiou_from_iou(base_iou)


def apply_wiou_patch(enable: bool) -> str:
    global _WIOU_PATCH_APPLIED
    if enable and not _WIOU_PATCH_APPLIED:
        metrics.bbox_iou = _bbox_iou_with_wiou
        _WIOU_PATCH_APPLIED = True
    elif not enable and _WIOU_PATCH_APPLIED:
        metrics.bbox_iou = _ORIGINAL_BBOX_IOU
        _WIOU_PATCH_APPLIED = False
    return "wiou" if enable else "ciou"
