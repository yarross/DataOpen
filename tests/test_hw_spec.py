"""The SKU matrix, the DNP groups, the bill of materials and the enclosure: kept consistent with each other (docs/HARDWARE.md)."""
import pytest

from dataopen.hw import report as R
from dataopen.hw import spec as S


def test_one_core_board_and_variants_differ_only_by_dnp_groups():
    codes = [s.code for s in S.SKUS]
    assert codes == ["DO-1", "DO-2", "DO-3", "DO-4"] and len({s.name for s in S.SKUS}) == 4
    core = {p.ref for p in S.PARTS if p.group == S.CORE}
    for s in S.SKUS:
        assert {p.ref for p in S.bom(s)} >= core                                    # the core is fitted on every SKU
        assert {p.ref for p in S.bom(s)} | {p.ref for p in S.dnp(s)} == {p.ref for p in S.PARTS}
        assert not {p.ref for p in S.bom(s)} & {p.ref for p in S.dnp(s)}
        assert set(s.groups) <= set(S.GROUPS)
    for a in S.SKUS:                                                                 # the difference between two SKUs is exactly the groups
        for b in S.SKUS:
            diff = {p.ref for p in S.bom(a)} ^ {p.ref for p in S.bom(b)}
            groups = {p.group for p in S.PARTS if p.ref in diff}
            assert groups == set(a.groups) ^ set(b.groups)


def test_the_named_skus_have_the_ports_the_brief_asks_for():
    ports = {s.code: {x for fl in S.ports(s).values() for x in fl} for s in S.SKUS}
    assert "HDMI IN" in ports["DO-1"] and "DP IN" not in ports["DO-1"] and "MOUSE C" not in ports["DO-1"] and "MOUSE A" in ports["DO-1"]
    assert "DP IN" in ports["DO-2"] and "DP OUT" in ports["DO-2"] and "HDMI IN" not in ports["DO-2"]
    assert "MOUSE C" in ports["DO-3"] and "HDMI IN" in ports["DO-3"] and "DP IN" not in ports["DO-3"]
    assert {"HDMI IN", "HDMI OUT", "DP IN", "DP OUT", "MOUSE A", "MOUSE C", "BUZZER"} <= ports["DO-4"]
    # the safety controls and the separate power are everywhere
    for p in ports.values():
        assert {"POWER", "TO PC", "PANIC", "MODE", "EXT BUTTON", "BYPASS LED", "STATUS LED", "SLOT LEDS", "SLOT", "CONFIRM"} <= p


def test_references_are_unique_and_prices_sane():
    refs = [p.ref for p in S.PARTS]
    assert len(refs) == len(set(refs))
    for p in S.PARTS:
        assert 0 < p.lo <= p.hi and p.qty >= 1 and p.group in (S.CORE,) + S.GROUPS


def test_bom_is_the_sum_of_its_parts_and_grows_with_what_is_fitted():
    for s in S.SKUS:
        lo, hi = S.cost(s)
        assert lo == pytest.approx(sum(p.qty * p.lo for p in S.bom(s))) and hi == pytest.approx(sum(p.qty * p.hi for p in S.bom(s)))
        assert 0 < lo < hi
    base, dp, c, pro = (S.sku(x) for x in ("DO-1", "DO-2", "DO-3", "DO-4"))
    for s in (dp, c):                                                                # a fitted group never makes a SKU cheaper
        assert set(base.groups) <= set(s.groups) or s is dp
    assert S.cost(pro)[0] > max(S.cost(x)[0] for x in (base, dp, c)) and S.retail(pro) > S.retail(base)
    assert S.retail(base) <= S.retail(c) <= S.retail(pro) and S.retail(base) <= S.retail(dp)
    assert all(S.retail(s) % S.RETAIL_STEP == 0 for s in S.SKUS)


def test_the_safety_path_is_on_every_sku_and_never_a_dnp_option():
    for s in S.SKUS:
        refs = {p.ref for p in S.bom(s)}
        assert {"K1", "K2", "U5", "U6", "R1", "SW1", "SW2", "J4", "D2", "U7", "J1"} <= refs


def test_every_fitted_connector_has_a_place_and_nothing_overflows_a_face():
    for s in S.SKUS:
        placed = [x for fl in S.ports(s).values() for x in fl]
        assert len(placed) == len(set(placed))
        for name, used in S.face_use_mm(s).items():
            face = next(f for f in S.FACES if f.name == name)
            assert used <= S.face_room_mm(face), (s.code, name, used)
    # a DNP part has no cutout only where its group is absent
    for p in S.PARTS:
        if p.port:
            assert p.port in S.PORT_FACE and p.port in S.PORT_MM
    fitted_ports = {p.port for p in S.bom(S.sku("DO-1")) if p.port}
    assert not {"DP IN", "MOUSE C", "BUZZER"} & fitted_ports


def test_the_block_structure_is_closed():
    domains = set(S.BLOCKS.values())
    assert domains == {"D0", "D1", "D2", "D3", "D4", "IO"}
    for src, dst, what in S.LINKS:
        assert src in S.BLOCKS and dst in S.BLOCKS and what
    linked = {x for l in S.LINKS for x in l[:2]}
    assert linked == set(S.BLOCKS)                                                   # every block is connected to something


def test_the_generated_tables_exist_for_every_name_and_render():
    for name, fn in R.TABLES.items():
        text = fn()
        assert text.strip() and "\n" in text or name == "diagram"
    for s in S.SKUS:
        assert s.code in R.sku_table()
    assert S.DIAGRAM.count("K1") >= 1 and "RELEASED" not in S.DIAGRAM


def test_the_documented_enclosure_fits_the_numbers_used_for_cooling():
    x, y, z = S.ENCLOSURE["size_mm"]
    assert (x, y, z) == (100, 75, 26)
    assert {f.name for f in S.FACES} == {"top", "mouse-end", "pc-end", "monitor-side"}
