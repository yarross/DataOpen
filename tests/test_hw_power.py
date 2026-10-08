"""Power domains, the budget per SKU and the passive-cooling estimate."""
import pytest

from dataopen.hw import power as W
from dataopen.hw import spec as S


def test_the_supply_is_a_separate_usb_c_and_the_pc_gives_nothing():
    assert (W.SUPPLY_V, W.SUPPLY_A) == (5.0, 3.0)
    assert W.PC_SENSE_UA <= 10 and W.MOUSE_PC_MA <= 500
    j = {p.ref: p for p in S.PARTS}
    # the PC cable is data only (no power connector toward the PC)
    assert j["J1"].port == "POWER" and j["J3"].port == "TO PC"
    assert "USB 2.0 data only" in j["J3"].name


def test_every_load_sits_in_a_known_domain_and_d0_is_software_free():
    assert set(W.DOMAINS) == {"D0", "D1", "D2", "D3", "D4"}
    assert all(l.domain in W.DOMAINS for l in W.LOADS)
    assert "no software" in W.DOMAINS["D0"][1] and "always" in W.DOMAINS["D0"][1]
    assert "ON only while" in W.DOMAINS["D3"][1] or "load-switched" in W.DOMAINS["D3"][0] + W.DOMAINS["D3"][1]
    assert W.SEQUENCE[0].startswith("VEXT") and W.SEQUENCE[1] == "D0"                # the safety domain is first


def test_the_peak_fits_the_supply_with_the_stated_margin_on_every_sku():
    for s in S.SKUS:
        b = W.budget(s)
        assert b["peak"] <= W.SUPPLY_W * W.DERATE and b["margin_peak_w"] > 0, (s.code, b)
        assert b["mouse_only"] < b["video"] < b["peak"]


def test_video_load_is_one_input_at_a_time():
    pro, base, dp = S.sku("DO-4"), S.sku("DO-1"), S.sku("DO-2")
    v = [l for l, w in W.loads(pro, "video") if l.domain == "D3"]
    assert len(v) == 1                                                               # HDMI and DP are not both powered
    assert [l for l, w in W.loads(base, "mouse_only") if l.domain == "D3"] == []     # without the detector the video domain is off
    assert W.load_w(dp, "video") > W.load_w(base, "video")                           # the DP branch costs more than the HDMI one


def test_thermal_estimate_follows_its_formula():
    x, y, z = S.ENCLOSURE["size_mm"]
    area = 2 * (x * y + x * z + y * z) / 1e6
    assert W.case_area_m2() == pytest.approx(area)
    assert W.r_th_k_per_w() == pytest.approx(1 / (W.H_W_M2K * area))
    for s in S.SKUS:
        t = W.thermal(s)
        assert t["mouse_only"]["dt_k"] == pytest.approx(W.input_w(s, "mouse_only") * W.r_th_k_per_w(), abs=0.06)
        assert t["mouse_only"]["ok"] is (t["mouse_only"]["dt_k"] <= W.TOUCH_DT_K)
    assert W.sustained_cap_w() == pytest.approx(W.TOUCH_DT_K / W.r_th_k_per_w())


def test_the_honest_thermal_finding_is_what_the_numbers_say():
    """Mouse assistance alone fits the compact case; the UI detector running all the time does NOT (docs/HARDWARE.md, section 4)."""
    for s in S.SKUS:
        assert W.thermal(s)["mouse_only"]["ok"]
        assert not W.thermal(s)["video"]["ok"]
        assert W.area_needed_m2(s) > W.case_area_m2()
    assert W.sustained_cap_w() < min(W.input_w(s, "video") for s in S.SKUS)


def test_unknown_mode_is_refused():
    with pytest.raises(ValueError):
        W.loads(S.SKUS[0], "turbo")
