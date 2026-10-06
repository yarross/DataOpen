"""Loss for UiNet: scale-range assignment (FCOS style), focal BCE on classes, GIoU on boxes."""

from __future__ import annotations

import torch

from .model import STRIDES, decode_boxes, grid_centers

LEVEL_MAX_SIDE = (40.0, 120.0, 1e9)  # which stride handles a box, by its longer side in input pixels
LEVEL_MIN_SIDE = (0.0, 24.0, 80.0)  # (with overlap, so a box near a boundary has positives on both levels)


def giou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ix = (torch.min(a[:, 2], b[:, 2]) - torch.max(a[:, 0], b[:, 0])).clamp(min=0)
    iy = (torch.min(a[:, 3], b[:, 3]) - torch.max(a[:, 1], b[:, 1])).clamp(min=0)
    inter = ix * iy
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = aa + ab - inter + 1e-6
    iou = inter / union
    ex = torch.max(a[:, 2], b[:, 2]) - torch.min(a[:, 0], b[:, 0])
    ey = torch.max(a[:, 3], b[:, 3]) - torch.min(a[:, 1], b[:, 1])
    enc = ex * ey + 1e-6
    return iou - (enc - union) / enc


def assign(gt_boxes: torch.Tensor, gt_cls: torch.Tensor, h: int, w: int, stride: int, level: int, n_cls: int):
    """Per cell of one level: target class (-1 none) and target box. gt_boxes (n,4) xyxy."""
    dev = gt_boxes.device
    cls_t = torch.full((h, w), -1, dtype=torch.long, device=dev)
    box_t = torch.zeros((h, w, 4), device=dev)
    if len(gt_boxes) == 0:
        return cls_t, box_t
    c = grid_centers(h, w, stride, dev)
    ww, hh = gt_boxes[:, 2] - gt_boxes[:, 0], gt_boxes[:, 3] - gt_boxes[:, 1]
    side = torch.max(ww, hh)
    use = (side >= LEVEL_MIN_SIDE[level]) & (side < LEVEL_MAX_SIDE[level])
    area = ww * hh
    order = torch.argsort(-area)  # large first, so small boxes overwrite where they overlap
    for i in order.tolist():
        if not bool(use[i]):
            continue
        x0, y0, x1, y1 = gt_boxes[i].tolist()
        gx, gy = (x0 + x1) / 2, (y0 + y1) / 2
        rx, ry = max(1.5 * stride, 0.2 * (x1 - x0)), max(1.5 * stride, 0.2 * (y1 - y0))
        inside = (c[..., 0] > x0) & (c[..., 0] < x1) & (c[..., 1] > y0) & (c[..., 1] < y1)
        near = ((c[..., 0] - gx).abs() < rx) & ((c[..., 1] - gy).abs() < ry)
        pos = inside & near
        if not bool(pos.any()):  # tiny box: the cell that holds its centre
            ix, iy = int(min(max(gx // stride, 0), w - 1)), int(min(max(gy // stride, 0), h - 1))
            pos = torch.zeros((h, w), dtype=torch.bool, device=dev)
            pos[iy, ix] = True
        cls_t[pos] = int(gt_cls[i])
        box_t[pos] = gt_boxes[i]
    return cls_t, box_t


def focal_bce(logits: torch.Tensor, target: torch.Tensor, gamma: float = 2.0, alpha: float = 0.25) -> torch.Tensor:
    p = torch.sigmoid(logits)
    ce = torch.nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
    pt = p * target + (1 - p) * (1 - target)
    a = alpha * target + (1 - alpha) * (1 - target)
    return a * (1 - pt) ** gamma * ce


def ui_loss(outs: list[torch.Tensor], gts: list[tuple[torch.Tensor, torch.Tensor]], n_cls: int, w_box: float = 2.0):
    """outs: per level (B, n_cls+4, H, W); gts: per image (boxes (n,4) xyxy in input pixels, classes (n,))."""
    cls_loss, box_loss, n_pos = 0.0, 0.0, 0
    for lv, (raw, stride) in enumerate(zip(outs, STRIDES)):
        b, _, h, w = raw.shape
        logits = raw[:, :n_cls].permute(0, 2, 3, 1)
        boxes = decode_boxes(raw[:, n_cls:], stride)
        target = torch.zeros_like(logits)
        for i in range(b):
            ct, bt = assign(gts[i][0], gts[i][1], h, w, stride, lv, n_cls)
            pos = ct >= 0
            if bool(pos.any()):
                target[i][pos, ct[pos]] = 1.0
                box_loss = box_loss + (1.0 - giou(boxes[i][pos], bt[pos])).sum()
                n_pos += int(pos.sum())
        cls_loss = cls_loss + focal_bce(logits, target).sum()
    n = max(n_pos, 1)
    return (cls_loss + w_box * box_loss) / n, {"cls": float(cls_loss.detach()) / n, "box": float(box_loss.detach()) / n, "pos": n_pos}
