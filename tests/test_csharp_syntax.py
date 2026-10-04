"""C# is not compiled in this repo's CI (no .NET SDK), so at least guarantee the sources PARSE and stay within
C# 7.3 (Unity 2020 / BepInEx 5). Type errors can only be found by building: `dataopen install --game <game>`."""
import re
from pathlib import Path

import pytest

ts = pytest.importorskip("tree_sitter")
cs = pytest.importorskip("tree_sitter_c_sharp")

ROOT = Path(__file__).resolve().parents[1] / "adapters"
FILES = sorted(ROOT.rglob("*.cs"))


def parser():
    return ts.Parser(ts.Language(cs.language()))


def error_lines(src: bytes):
    out = []

    def walk(n):
        if n.type == "ERROR" or n.is_missing:
            out.append(n.start_point[0] + 1)
        for c in n.children:
            walk(c)

    tree = parser().parse(src)
    if tree.root_node.has_error:
        walk(tree.root_node)
    return out


def test_there_are_csharp_sources():
    assert len(FILES) >= 8


@pytest.mark.parametrize("path", FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_parses_without_syntax_errors(path):
    assert error_lines(path.read_bytes()) == [], f"syntax errors in {path} at lines {error_lines(path.read_bytes())}"


NEWER_THAN_73 = [
    (r"\?\?=", "null-coalescing assignment (C# 8)"),
    (r"\bis\s+not\b", "'is not' pattern (C# 9)"),
    (r"\busing\s+var\b", "using declaration (C# 8)"),
    (r"\bnew\(\)", "target-typed new (C# 9)"),
    (r"\brecord\s+(class|struct)?\s*\w+\s*\(", "records (C# 9)"),
    (r"\bswitch\s*\{", "switch expression (C# 8)"),
    (r"\.\.\s*\^|\^\d", "ranges/indices (C# 8)"),
    (r"\bstatic\s+\(", "static local function/lambda (C# 8/9)"),
]


@pytest.mark.parametrize("path", FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_stays_within_csharp_73(path):
    text = re.sub(r'"(\\.|[^"\\])*"|//.*', "", path.read_text())
    for pat, what in NEWER_THAN_73:
        assert not re.search(pat, text), f"{path.name}: uses {what}; Unity 2020 compiles C# 7.3"


def test_the_checker_actually_detects_errors():
    assert error_lines(b"class A { void M() { int x = ; } ") != []
    assert error_lines(b"class A { void M() { int x = 1; } }") == []


def test_msbuild_and_nuget_files_are_well_formed_xml():
    """A `--` inside an XML comment made both plugin projects unloadable (MSB4025) and the C# CI never reached the
    compiler. XML well-formedness needs no .NET, so check it here."""
    import xml.etree.ElementTree as ET
    root = Path(__file__).resolve().parents[1] / "adapters" / "unity"
    files = sorted([*root.glob("**/*.csproj"), *root.glob("**/*.props"), *root.glob("**/NuGet.config")])
    assert any(f.suffix == ".csproj" for f in files)
    for f in files:
        try:
            ET.parse(f)
        except ET.ParseError as e:
            raise AssertionError(f"{f.relative_to(root)} is not well-formed XML: {e}") from e
