"""Deriving one keypoint set from another: the layer that makes the target schema a *data* decision.

Two keypoint sets exist in the pipeline:

    capture schema   what the game mod reports (e.g. HUMAN_13: bones resolved by the mod's bone maps)
    target schema    what the dataset is labeled with and the baseline model is judged on (e.g. SHOOTER_12)

A `SchemaMapping` turns the first into the second with small rules ("neck = mean(l_shoulder, r_shoulder)",
"head_top = extend_m(neck, head, 0.12)") evaluated on 3D world points, validity masks and visibility flags. The SAME rule
language maps a baseline model's own keypoints (COCO-17, say) onto the target schema in 2D, so ground truth and
predictions pass through identical definitions of a derived point.

Rule language (arguments are source keypoint names; numbers may come from `{params}` placeholders):

    A leading @ refers to an already derived TARGET point instead of a source one: extend(@neck, @head_center, 0.6).

    copy(a)                  the point a
    mean(a, b, ...)          mean of the points
    weighted(a:w, b:w, ...)  weighted mean (weights are normalized)
    lerp(a, b, t)            a + (b - a) * t
    extend(a, b, k)          b + (b - a) * k           past b, k times the length |a->b| (works in 2D and 3D)
    extend_m(a, b, d)        b + unit(b - a) * d       past b by d METERS (3D only)
    ... ; vis=min|mean|majority    how visibility / confidence of the parents combines (default min)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Mapping, Optional, Sequence

import numpy as np

from .models import EntityState, FrameSnapshot
from .schema import SkeletonSchema


class RuleError(ValueError):
    pass


_CALL = re.compile(r"^\s*([a-z_]+)\s*\((.*)\)\s*(?:;\s*vis\s*=\s*([a-z]+))?\s*$", re.S)
_OPS = {"copy": (1, 1), "mean": (1, 99), "weighted": (1, 99), "lerp": (2, 2), "extend": (2, 2), "extend_m": (2, 2)}


@dataclass(frozen=True)
class Rule:
    op: str
    args: tuple[str, ...]
    weights: tuple[float, ...]       # per arg (mean = equal, weighted = given, others unused)
    num: float = 0.0                 # t / k / d
    vis: str = "min"

    @property
    def needs_3d(self) -> bool:
        return self.op == "extend_m"

    @staticmethod
    def parse(expr: str, params: Optional[Mapping[str, float]] = None) -> "Rule":
        try:
            text = expr.format_map({k: repr(float(v)) for k, v in (params or {}).items()})
        except (KeyError, IndexError, ValueError) as e:
            raise RuleError(f"rule {expr!r}: unknown parameter {e}") from e
        m = _CALL.match(text)
        if not m:
            raise RuleError(f"rule {expr!r}: expected op(arg, ...) [; vis=min|mean|majority]")
        op, body, vis = m.group(1), m.group(2), m.group(3) or "min"
        if op not in _OPS:
            raise RuleError(f"rule {expr!r}: unknown op {op!r}; known: {sorted(_OPS)}")
        if vis not in ("min", "mean", "majority"):
            raise RuleError(f"rule {expr!r}: vis must be min, mean or majority")
        parts = [p.strip() for p in body.split(",") if p.strip()]
        num = 0.0
        if op in ("lerp", "extend", "extend_m"):
            if len(parts) != 3:
                raise RuleError(f"rule {expr!r}: {op}(a, b, number)")
            try:
                num = float(parts.pop())
            except ValueError as e:
                raise RuleError(f"rule {expr!r}: the last argument must be a number") from e
        lo, hi = _OPS[op]
        if not lo <= len(parts) <= hi:
            raise RuleError(f"rule {expr!r}: wrong number of arguments for {op}")
        names, weights = [], []
        for p in parts:
            if op == "weighted":
                if ":" not in p:
                    raise RuleError(f"rule {expr!r}: weighted() arguments look like name:weight")
                n, w = p.split(":", 1)
                names.append(n.strip())
                try:
                    weights.append(float(w))
                except ValueError as e:
                    raise RuleError(f"rule {expr!r}: bad weight {w!r}") from e
            else:
                names.append(p)
                weights.append(1.0)
        if op == "weighted" and (any(w < 0 for w in weights) or sum(weights) <= 0):
            raise RuleError(f"rule {expr!r}: weights must be non-negative with a positive sum")
        return Rule(op, tuple(names), tuple(weights), num, vis)


class SchemaMapping:
    """source keypoint set -> target keypoint set by rules, for 3D skeletons and for 2D model outputs."""

    def __init__(self, source: SkeletonSchema, target: SkeletonSchema, rules: Mapping[str, Rule]) -> None:
        self.source, self.target, self.rules = source, target, dict(rules)
        missing = [k for k in target.keypoints if k not in self.rules]
        if missing:
            raise RuleError(f"{source.name} -> {target.name}: no rule for target keypoints {missing}")
        for name, r in self.rules.items():
            if name not in target.keypoints:
                raise RuleError(f"rule for {name!r}, which is not a keypoint of {target.name}")
            bad = [a for a in r.args if (a[1:] not in target.keypoints if a.startswith("@") else a not in source.keypoints)]
            if bad:
                raise RuleError(f"{target.name}.{name}: unknown keypoint(s) {bad} (source {source.name}; @name = target point)")
        # (is_target, index) per argument; targets are evaluated in dependency order
        self._ordered = [self.rules[k] for k in target.keypoints]
        self._refs = [[(a.startswith("@"), (target if a.startswith("@") else source).index(a.lstrip("@")))
                       for a in r.args] for r in self._ordered]
        self._order = self._toposort()

    def _toposort(self) -> list[int]:
        done: list[int] = []
        state: dict[int, int] = {}

        def visit(j: int) -> None:
            if state.get(j) == 2:
                return
            if state.get(j) == 1:
                raise RuleError(f"{self.target.name}: rules depend on each other in a cycle at {self.target.keypoints[j]!r}")
            state[j] = 1
            for is_t, i in self._refs[j]:
                if is_t:
                    visit(i)
            state[j] = 2
            done.append(j)

        for j in range(len(self._ordered)):
            visit(j)
        return done

    @property
    def requires_3d(self) -> bool:
        return any(r.needs_3d for r in self._ordered)

    # ---- one rule on arrays ----
    @staticmethod
    def _eval(rule: Rule, pts: np.ndarray) -> np.ndarray:
        w = np.asarray(rule.weights, dtype=np.float64)
        if rule.op in ("copy", "mean", "weighted"):
            return (pts * w[:, None]).sum(axis=0) / w.sum()
        a, b = pts[0], pts[1]
        if rule.op == "lerp":
            return a + (b - a) * rule.num
        if rule.op == "extend":
            return b + (b - a) * rule.num
        d = b - a                                         # extend_m
        n = float(np.linalg.norm(d))
        return b + (d / n) * rule.num if n > 1e-9 else b

    def points(self, p: np.ndarray, valid: Optional[np.ndarray] = None, aux: Optional[np.ndarray] = None,
               aux_is_flags: bool = True) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """p: (M, D) source points, valid: (M,) bool, aux: (M,) visibility flags (ints) or confidences (floats).
        Returns the (K, D) target points, the (K,) validity (all parents valid) and the combined aux (or None)."""
        p = np.asarray(p, dtype=np.float64)
        if p.ndim != 2 or p.shape[0] != self.source.num_keypoints:
            raise ValueError(f"expected {self.source.num_keypoints} {self.source.name} points, got shape {p.shape}")
        if p.shape[1] == 2 and self.requires_3d:
            raise RuleError(f"{self.source.name} -> {self.target.name} uses extend_m(): 3D points only")
        valid = np.ones(len(p), dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
        k = self.target.num_keypoints
        out = np.zeros((k, p.shape[1]))
        ok = np.zeros(k, dtype=bool)
        aux_out = None if aux is None else np.zeros(k, dtype=np.float64)
        aux_in = None if aux is None else np.asarray(aux, dtype=np.float64)
        for j in self._order:
            rule, refs = self._ordered[j], self._refs[j]
            if not all((ok[i] if t else valid[i]) for t, i in refs):
                continue
            pts = np.stack([out[i] if t else p[i] for t, i in refs])
            if not np.isfinite(pts).all():
                continue
            out[j] = self._eval(rule, pts)
            ok[j] = True
            if aux_in is not None:
                v = np.array([aux_out[i] if t else aux_in[i] for t, i in refs])
                w = np.asarray(rule.weights, dtype=np.float64)
                if rule.vis == "min":
                    aux_out[j] = v.min()
                elif rule.vis == "mean" or not aux_is_flags:
                    aux_out[j] = float((v * w).sum() / w.sum())
                else:                                     # majority of the weight sees it
                    aux_out[j] = 2.0 if float(w[v >= 2].sum() / w.sum()) >= 0.5 else 1.0
        return out, ok, aux_out

    # ---- 3D: ground truth ----
    def convert_entity(self, e: EntityState, extra_meta: Optional[Mapping[str, object]] = None) -> EntityState:
        sk, valid, vis = self.points(e.skeleton_world, e.joint_valid, e.engine_visibility, aux_is_flags=True)
        meta = {**(extra_meta or {}), **e.meta}
        return replace(e, skeleton_world=sk, joint_valid=valid,
                       engine_visibility=None if vis is None else np.rint(vis).astype(int), meta=meta)

    def convert_snapshot(self, snap: FrameSnapshot, meta_by_entity: Optional[Mapping[int, Mapping[str, object]]] = None
                         ) -> FrameSnapshot:
        ents = [self.convert_entity(e, (meta_by_entity or {}).get(e.entity_id)) for e in snap.entities]
        return replace(snap, entities=ents)

    # ---- 2D: a model's keypoints (x, y, conf) -> target schema ----
    def apply2d(self, kp: np.ndarray) -> np.ndarray:
        """(M, 3) model keypoints -> (K, 3): derived points are computed like the ground truth, confidence = parents'."""
        kp = np.asarray(kp, dtype=np.float64)
        xy, ok, conf = self.points(kp[:, :2], None, kp[:, 2], aux_is_flags=False)
        out = np.zeros((self.target.num_keypoints, 3))
        out[:, :2] = xy
        out[:, 2] = np.where(ok, conf, 0.0)
        return out

    def describe(self) -> list[str]:
        return [f"{k} = {r.op}({', '.join(r.args)}" + (f", {r.num}" if r.op in ('lerp', 'extend', 'extend_m') else "") + ")"
                for k, r in zip(self.target.keypoints, self._ordered)]


def parse_rules(exprs: Mapping[str, str], params: Optional[Mapping[str, float]] = None) -> dict[str, Rule]:
    return {k: Rule.parse(v, params) for k, v in exprs.items()}


def identity_mapping(schema: SkeletonSchema) -> SchemaMapping:
    return SchemaMapping(schema, schema, {k: Rule("copy", (k,), (1.0,)) for k in schema.keypoints})


def names_of(rule: Rule) -> Sequence[str]:
    return rule.args
