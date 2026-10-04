"""Losses and label assignment (PyTorch).

Two heads are trained on the same features: a ONE-TO-MANY head (each object supervises its top-k anchors: dense, stable
gradients) and a ONE-TO-ONE head (top-1 anchor per object: learns not to duplicate, so inference needs no NMS), as in YOLOv10.

Per positive anchor: IoU-aware classification (task-aligned targets), CIoU box, OKS keypoint loss whose per-point weights come
from the SCHEMA (the aim point counts 3x), a calibrated keypoint-score loss (target = the point's own similarity), a visibility
loss, and for the aim point a stride-4 Gaussian heatmap + sub-cell offset loss.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .config import TrainConfig
from .layout import HeadLayout


def box_iou_pairwise(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """a, b broadcastable (..., 4) xyxy -> IoU (...)."""
    lt = torch.maximum(a[..., :2], b[..., :2])
    rb = torch.minimum(a[..., 2:], b[..., 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    aa = (a[..., 2] - a[..., 0]).clamp(min=0) * (a[..., 3] - a[..., 1]).clamp(min=0)
    ab = (b[..., 2] - b[..., 0]).clamp(min=0) * (b[..., 3] - b[..., 1]).clamp(min=0)
    return inter / (aa + ab - inter + eps)


def ciou(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """(N, 4) xyxy each -> CIoU (N,)."""
    iou = box_iou_pairwise(pred, gt)
    cw = torch.maximum(pred[:, 2], gt[:, 2]) - torch.minimum(pred[:, 0], gt[:, 0])
    ch = torch.maximum(pred[:, 3], gt[:, 3]) - torch.minimum(pred[:, 1], gt[:, 1])
    c2 = cw ** 2 + ch ** 2 + eps
    rho2 = (((pred[:, 0] + pred[:, 2]) - (gt[:, 0] + gt[:, 2])) ** 2 + ((pred[:, 1] + pred[:, 3]) - (gt[:, 1] + gt[:, 3])) ** 2) / 4
    wp, hp = (pred[:, 2] - pred[:, 0]).clamp(min=eps), (pred[:, 3] - pred[:, 1]).clamp(min=eps)
    wg, hg = (gt[:, 2] - gt[:, 0]).clamp(min=eps), (gt[:, 3] - gt[:, 1]).clamp(min=eps)
    v = (4 / math.pi ** 2) * (torch.atan(wg / hg) - torch.atan(wp / hp)) ** 2
    with torch.no_grad():
        alpha = v / (v - iou + (1 + eps))
    return iou - (rho2 / c2 + v * alpha)


@torch.no_grad()
def task_aligned_assign(scores: torch.Tensor, boxes: torch.Tensor, anchors: torch.Tensor, stride: torch.Tensor,
                        gt_boxes: torch.Tensor, gt_cls: torch.Tensor, gt_valid: torch.Tensor, topk: int, alpha: float,
                        beta: float, eps: float = 1e-9):
    """scores (B,A,nc) probabilities, boxes (B,A,4), anchors (A,2), stride (A,), gt_* (B,M,...).
    Returns fg (B,A) bool, gt_idx (B,A) long, target_scores (B,A,nc)."""
    B, A, nc = scores.shape
    M = gt_boxes.shape[1]
    if M == 0:
        return (torch.zeros(B, A, dtype=torch.bool, device=scores.device), torch.zeros(B, A, dtype=torch.long, device=scores.device),
                torch.zeros(B, A, nc, device=scores.device))
    ax, ay = anchors[:, 0][None, None], anchors[:, 1][None, None]
    x1, y1, x2, y2 = (gt_boxes[..., i][..., None] for i in range(4))
    inside = torch.stack([ax - x1, ay - y1, x2 - ax, y2 - ay], dim=-1).amin(-1) > 0
    gcx, gcy = (x1 + x2) / 2, (y1 + y2) / 2
    near = torch.maximum((ax - gcx).abs(), (ay - gcy).abs()) < 1.5 * stride[None, None]       # tiny objects: centre sampling
    mask_in = (inside | near) & gt_valid[..., None]
    ious = box_iou_pairwise(gt_boxes[:, :, None, :], boxes[:, None, :, :])                    # (B, M, A)
    cls_idx = gt_cls.clamp(min=0)[..., None].expand(B, M, A)
    s = scores.permute(0, 2, 1).gather(1, cls_idx)                                           # (B, M, A)
    metric = (s.pow(alpha) * ious.pow(beta)) * mask_in
    k = min(topk, A)
    topv, topi = metric.topk(k, dim=-1)
    in_topk = torch.zeros_like(metric, dtype=torch.bool).scatter_(-1, topi, topv > 0)
    pos = in_topk & mask_in
    multi = pos.sum(1) > 1
    if multi.any():                                                                          # an anchor serves ONE object
        best = ious.argmax(1)
        one = F.one_hot(best, M).permute(0, 2, 1).bool()
        pos = torch.where(multi[:, None, :], one & mask_in, pos)
    fg = pos.any(1)
    gt_idx = pos.float().argmax(1)
    align = metric * pos
    iou_pos = (ious * pos).amax(-1, keepdim=True)
    norm = (align * iou_pos / (align.amax(-1, keepdim=True) + eps)).amax(1)                  # (B, A)
    tcls = gt_cls.gather(1, gt_idx).clamp(min=0)
    target = torch.zeros(B, A, nc, device=scores.device)
    target.scatter_(2, tcls[..., None], (norm * fg)[..., None])
    return fg, gt_idx, target


class DetectionLoss:
    def __init__(self, cfg: TrainConfig, layout: HeadLayout, sigmas, weights, primary: tuple = ()) -> None:
        self.cfg, self.lay = cfg, layout
        self.sigmas = torch.tensor(sigmas, dtype=torch.float32)
        self.weights = torch.tensor(weights, dtype=torch.float32)
        self.primary = tuple(primary)
        self._anchor_cache: dict = {}

    # ---- flatten / decode ----
    def anchors(self, shapes, device):
        key = (tuple(shapes), str(device))
        if key not in self._anchor_cache:
            pts, strides = [], []
            for (h, w), s in zip(shapes, self.lay.strides):
                gy, gx = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
                pts.append(torch.stack([(gx.reshape(-1) + 0.5) * s, (gy.reshape(-1) + 0.5) * s], dim=1).float())
                strides.append(torch.full((h * w,), float(s), device=device))
            self._anchor_cache[key] = (torch.cat(pts), torch.cat(strides))
        return self._anchor_cache[key]

    def decode(self, outs: list[torch.Tensor]):
        lay = self.lay
        shapes = [(o.shape[2], o.shape[3]) for o in outs]
        anchors, stride = self.anchors(shapes, outs[0].device)
        flat = torch.cat([o.flatten(2) for o in outs], dim=2).permute(0, 2, 1)                 # (B, A, C)
        s = lay.slices
        cls = flat[..., s["cls"]]
        d = torch.exp(flat[..., s["box"]].clamp(-6, 6)) * stride[None, :, None]
        box = torch.stack([anchors[None, :, 0] - d[..., 0], anchors[None, :, 1] - d[..., 1],
                           anchors[None, :, 0] + d[..., 2], anchors[None, :, 1] + d[..., 3]], dim=-1)
        K = lay.n_kpt
        off = flat[..., s["kp"]].reshape(*flat.shape[:2], K, 2) * lay.offset_scale * stride[None, :, None, None]
        kxy = anchors[None, :, None, :] + off
        return cls, box, kxy, flat[..., s["kp_score"]], flat[..., s["kp_vis"]], anchors, stride

    # ---- one head ----
    def head_loss(self, outs: list[torch.Tensor], tg: dict, topk: int):
        c = self.cfg
        cls, box, kxy, kscore, kvis, anchors, stride = self.decode(outs)
        gt_box, gt_cls, gt_kp, valid = tg["boxes"], tg["cls"], tg["kpts"], tg["valid"]
        fg, gt_idx, tscore = task_aligned_assign(cls.detach().sigmoid(), box.detach(), anchors, stride, gt_box, gt_cls, valid,
                                                 topk, c.tal_alpha, c.tal_beta)
        tss = tscore.sum().clamp(min=1.0)
        l_cls = F.binary_cross_entropy_with_logits(cls, tscore, reduction="sum") / tss
        zero = cls.sum() * 0.0
        l_box = l_kp = l_l1 = l_ks = l_vis = zero
        if fg.any():
            bi, ai = fg.nonzero(as_tuple=True)
            gi = gt_idx[bi, ai]
            w = tscore[bi, ai].sum(-1)
            tb = gt_box[bi, gi]
            l_box = ((1.0 - ciou(box[bi, ai], tb)) * w).sum() / tss
            tk = gt_kp[bi, gi]                                                                # (n, K, 3)
            lab = (tk[..., 2] > 0).float()
            area = ((tb[:, 2] - tb[:, 0]) * (tb[:, 3] - tb[:, 1])).clamp(min=1.0)
            k2 = (2.0 * self.sigmas.to(cls.device)) ** 2
            d2 = ((kxy[bi, ai] - tk[..., :2]) ** 2).sum(-1)
            e = d2 / (2.0 * area[:, None] * k2[None] + 1e-9)
            sim = torch.exp(-e)
            wk = self.weights.to(cls.device)[None] * lab
            wsum = wk.sum(-1).clamp(min=1e-6)
            l_kp = (((1.0 - sim) * wk).sum(-1) / wsum * w).sum() / tss
            # OKS saturates: for a tight sigma (the aim point's) the gradient vanishes a few pixels away from the target, so
            # a point that starts far off never converges. A distance term in units of person height always has a gradient.
            hgt = (tb[:, 3] - tb[:, 1]).clamp(min=1.0)
            dist = torch.sqrt(d2 + 1e-6) / hgt[:, None]
            l_l1 = ((F.smooth_l1_loss(dist, torch.zeros_like(dist), reduction="none", beta=0.02) * wk).sum(-1) / wsum * w).sum() / tss
            ks_t = sim.detach()
            l_ks = (F.binary_cross_entropy_with_logits(kscore[bi, ai], ks_t, reduction="none") * lab).sum(-1) / lab.sum(-1).clamp(min=1)
            l_ks = (l_ks * w).sum() / tss
            vt = (tk[..., 2] == 2).float()
            l_vis = (F.binary_cross_entropy_with_logits(kvis[bi, ai], vt, reduction="none") * lab).sum(-1) / lab.sum(-1).clamp(min=1)
            l_vis = (l_vis * w).sum() / tss
        total = c.w_cls * l_cls + c.w_box * l_box + c.w_kp * l_kp + c.w_kp_l1 * l_l1 + c.w_kp_score * l_ks + c.w_vis * l_vis
        return total, {"cls": l_cls.detach(), "box": l_box.detach(), "kp": l_kp.detach(), "kp_l1": l_l1.detach(),
                       "kp_score": l_ks.detach(), "vis": l_vis.detach()}

    # ---- aim heatmap ----
    def heat_loss(self, ref: torch.Tensor, tg: dict) -> torch.Tensor:
        """CenterNet focal loss on the stride-4 heatmap + L1 on the sub-cell offset at the centre cell, primary keypoints only.
        Vectorized over all (image, object, primary point) triples."""
        P = len(self.primary)
        B, _, H, W = ref.shape
        stride = self.lay.refine_stride
        dev = ref.device
        gt_kp, gt_box, valid = tg["kpts"], tg["boxes"], tg["valid"]
        bi, mi = torch.nonzero(valid, as_tuple=True)
        hm_t = torch.zeros(B * P, H * W, device=dev)
        off_t = torch.zeros(B, 2 * P, H, W, device=dev)
        off_m = torch.zeros(B, P, H, W, device=dev)
        pos_w = torch.zeros(B, P, H, W, device=dev)
        if len(bi):
            ent_b, ent_p, ent_x, ent_y, ent_v, ent_h = [], [], [], [], [], []
            for pi, kp in enumerate(self.primary):
                k = gt_kp[bi, mi, kp]                                                 # (N, 3)
                ok = k[:, 2] > 0
                ent_b.append(bi[ok])
                ent_p.append(torch.full_like(bi[ok], pi))
                ent_x.append(k[ok, 0])
                ent_y.append(k[ok, 1])
                ent_v.append(k[ok, 2])
                ent_h.append((gt_box[bi, mi, 3] - gt_box[bi, mi, 1])[ok])
            eb, ep, ex, ey, ev, eh = (torch.cat(t) for t in (ent_b, ent_p, ent_x, ent_y, ent_v, ent_h))
            if len(eb):
                cx, cy = ex / stride - 0.5, ey / stride - 0.5                          # continuous cell coordinates
                ix = torch.round(cx).clamp(0, W - 1).long()
                iy = torch.round(cy).clamp(0, H - 1).long()
                sig = (0.04 * eh / stride).clamp(min=1.0)
                wt = torch.where(ev == 2, 1.0, 0.6)                                   # an occluded aim point is a softer target
                gy, gx = torch.meshgrid(torch.arange(H, device=dev), torch.arange(W, device=dev), indexing="ij")
                gx, gy = gx.reshape(-1).float(), gy.reshape(-1).float()
                for lo in range(0, len(eb), 256):                                       # chunked: N x H x W floats
                    sl = slice(lo, lo + 256)
                    # the Gaussian is centred on the INTEGER cell: its peak is exactly 1 there, the sub-cell remainder is what
                    # the offset channels learn
                    g = torch.exp(-((gx[None] - ix[sl, None]) ** 2 + (gy[None] - iy[sl, None]) ** 2) / (2 * sig[sl, None] ** 2))
                    hm_t.scatter_reduce_(0, (eb[sl] * P + ep[sl])[:, None].expand(-1, H * W), g * wt[sl, None], reduce="amax")
                off_t[eb, ep, iy, ix] = cx - ix
                off_t[eb, P + ep, iy, ix] = cy - iy
                off_m[eb, ep, iy, ix] = 1.0
                pos_w[eb, ep, iy, ix] = wt
        hm_t = hm_t.view(B, P, H, W)
        p = ref[:, :P].sigmoid().clamp(1e-4, 1 - 1e-4)
        l_pos = -(((1 - p) ** 2) * torch.log(p) * pos_w).sum()
        l_neg = -(((1 - hm_t) ** 4) * (p ** 2) * torch.log(1 - p) * (1 - off_m)).sum()
        n_pos = off_m.sum().clamp(min=1.0)
        l_off = ((ref[:, P:2 * P] - off_t[:, :P]).abs() * off_m).sum() + ((ref[:, 2 * P:3 * P] - off_t[:, P:]).abs() * off_m).sum()
        return (l_pos + l_neg) / n_pos + l_off / n_pos

    def __call__(self, out: dict, tg: dict):
        c = self.cfg
        t_m, i_m = self.head_loss(out["o2m"], tg, c.tal_topk_o2m)
        t_o, i_o = self.head_loss(out["o2o"], tg, c.tal_topk_o2o)
        total = t_m + c.w_o2o * t_o
        items = {f"m_{k}": v for k, v in i_m.items()} | {f"o_{k}": v for k, v in i_o.items()}
        if out.get("refine") is not None and self.primary:
            h = self.heat_loss(out["refine"], tg)
            total = total + c.w_heat * h
            items["heat"] = h.detach()
        items["total"] = total.detach()
        return total, items
