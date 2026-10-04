"""Target schemas as data: a TOML file defines the keypoints, their OKS sigmas/weights, the primary (aim) point, the
classes and the rules that derive every point from what the game mod reports. See docs/SCHEMAS.md."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

from .derive import Rule, RuleError, SchemaMapping, identity_mapping, parse_rules
from .schema import COCO_17, HUMAN_13, SkeletonSchema

try:
    import tomllib
except ModuleNotFoundError:          # Python 3.10
    import tomli as tomllib          # type: ignore[no-redef]

BUILTIN_DIR = Path(__file__).resolve().parents[1] / "schemas"
KNOWN_SOURCES = {"human13": HUMAN_13, "coco17": COCO_17}


class SchemaFileError(ValueError):
    pass


@dataclass
class SchemaBundle:
    """A target schema plus everything needed to produce it: rules from the capture schema, rules per model type."""

    schema: SkeletonSchema
    source: str
    rules: dict[str, str]
    params: dict[str, float] = field(default_factory=dict)
    models: dict[str, dict[str, str]] = field(default_factory=dict)

    def mapping(self, capture: SkeletonSchema, params: Optional[Mapping[str, float]] = None) -> SchemaMapping:
        """Capture schema (what the adapter reports) -> this target. `params` overrides the rule constants."""
        if capture.name == self.schema.name:
            return identity_mapping(self.schema)
        if capture.name != self.source:
            raise SchemaFileError(f"schema {self.schema.name} is written against the {self.source!r} capture schema, but the "
                                  f"game reports {capture.name!r}; add `[schema.models.{capture.name}]`-style rules or a new "
                                  f"schema file")
        try:
            return SchemaMapping(capture, self.schema, parse_rules(self.rules, {**self.params, **(params or {})}))
        except RuleError as e:
            raise SchemaFileError(str(e)) from e

    def model_mapping(self, model: str) -> SchemaMapping:
        """A baseline model's keypoint set (`coco17`) -> this target, in 2D."""
        src = KNOWN_SOURCES.get(model)
        if src is None or model not in self.models:
            raise SchemaFileError(f"schema {self.schema.name} has no mapping for model keypoints {model!r}; "
                                  f"available: {sorted(self.models) or 'none'}")
        try:
            return SchemaMapping(src, self.schema, parse_rules(self.models[model], self.params))
        except RuleError as e:
            raise SchemaFileError(str(e)) from e

    def model_names(self) -> list[str]:
        return sorted(self.models)


def _tup(v: Any) -> tuple:
    return tuple(tuple(x) if isinstance(x, list) else x for x in v)


def bundle_from_dict(d: Mapping[str, Any]) -> SchemaBundle:
    s = d.get("schema")
    if not isinstance(s, dict):
        raise SchemaFileError("missing [schema] table")
    try:
        names = tuple(s["keypoints"])
        points = s.get("points", {})
        missing = [n for n in names if n not in points]
        if missing:
            raise SchemaFileError(f"no [schema.points.<name>] table for {missing}")
        extra = [n for n in points if n not in names]
        if extra:
            raise SchemaFileError(f"[schema.points.*] for unknown keypoints {extra}")
        schema = SkeletonSchema(
            name=str(s["name"]), keypoints=names, edges=_tup(s.get("edges", [])), flip_pairs=_tup(s.get("flip_pairs", [])),
            sigmas=tuple(float(points[n].get("sigma", 0.07)) for n in names),
            weights=tuple(float(points[n].get("weight", 1.0)) for n in names),
            derived=tuple(n for n in names if points[n].get("derived", False)),
            primary=tuple(s.get("primary", [])),
            groups=tuple((g, tuple(m)) for g, m in s.get("groups", {}).items()),
            roles=tuple((r, n) for r, n in s.get("roles", {}).items()),
            classes=tuple(s.get("classes", ["person"])), class_key=str(s.get("class_key", "")))
        rules = {n: str(points[n]["rule"]) for n in names if "rule" in points[n]}
    except KeyError as e:
        raise SchemaFileError(f"missing key {e} in the schema file") from e
    except ValueError as e:
        raise SchemaFileError(str(e)) from e
    models = {m: {k: str(v) for k, v in r.items()} for m, r in s.get("models", {}).items()}
    return SchemaBundle(schema, str(s.get("source", "")), rules,
                        {k: float(v) for k, v in s.get("params", {}).items()}, models)


def load_schema(spec: str) -> SchemaBundle:
    """`spec` is a built-in name ("shooter12") or a path to a .toml file."""
    path = Path(spec)
    if not path.suffix:
        path = BUILTIN_DIR / f"{spec}.toml"
    if not path.is_file():
        builtins = sorted(p.stem for p in BUILTIN_DIR.glob("*.toml"))
        raise SchemaFileError(f"schema {spec!r} not found (built-in: {builtins}; or give a path to a .toml file)")
    try:
        return bundle_from_dict(tomllib.loads(path.read_text(encoding="utf-8")))
    except tomllib.TOMLDecodeError as e:
        raise SchemaFileError(f"{path}: {e}") from e


def resolve_target(spec: Optional[str], capture: SkeletonSchema, params: Optional[Mapping[str, float]] = None
                   ) -> tuple[Optional[SchemaBundle], Optional[SchemaMapping]]:
    """(bundle, capture->target mapping), or (None, None) when no target schema was asked for."""
    if not spec:
        return None, None
    bundle = load_schema(spec)
    return bundle, bundle.mapping(capture, params)


def builtin_schemas() -> list[str]:
    return sorted(p.stem for p in BUILTIN_DIR.glob("*.toml"))


__all__ = ["SchemaBundle", "SchemaFileError", "load_schema", "resolve_target", "bundle_from_dict", "builtin_schemas",
           "Rule"]
