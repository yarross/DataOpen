"""Video timings, link budgets and the time at which each scan line exists.

The monitor path of the device never waits for any of this: it is a passthrough. These numbers are about the CAPTURE tap: how much of
the frame must have been transmitted before the detector's input exists. That is physics (the pixels arrive over the frame period),
not an implementation choice, and it is the honest floor of the analysis latency."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class VideoMode:
    name: str
    h_active: int
    v_active: int
    h_total: int
    v_total: int
    pclk_hz: int
    source: str = "cta"  # "cta" (CTA-861 table) or "cvt-rb" (computed, reduced blanking v1)

    @property
    def refresh_hz(self) -> float:
        return self.pclk_hz / (self.h_total * self.v_total)

    @property
    def line_time_us(self) -> float:
        return self.h_total / self.pclk_hz * 1e6

    @property
    def frame_time_us(self) -> float:
        return self.h_total * self.v_total / self.pclk_hz * 1e6

    @property
    def active_time_us(self) -> float:
        return self.v_active * self.line_time_us

    def row_done_us(self, y: int) -> float:
        """Time after the start of the active video at which scan line `y` (0-based) has fully arrived."""
        return (y + 1) * self.line_time_us


# CTA-861 modes by VIC (the subset a PC GPU actually uses): (h_active, v_active, h_total, v_total, pclk_hz)
CTA = {
    1: (640, 480, 800, 525, 25_175_000),
    4: (1280, 720, 1650, 750, 74_250_000),
    16: (1920, 1080, 2200, 1125, 148_500_000),
    19: (1280, 720, 1980, 750, 74_250_000),
    31: (1920, 1080, 2640, 1125, 148_500_000),
    32: (1920, 1080, 2750, 1125, 74_250_000),
    34: (1920, 1080, 2200, 1125, 74_250_000),
    47: (1280, 720, 1650, 750, 148_500_000),  # 720p120
    63: (1920, 1080, 2200, 1125, 297_000_000),  # 1080p120
    64: (1920, 1080, 2640, 1125, 297_000_000),  # 1080p100
    93: (3840, 2160, 5500, 2250, 297_000_000),  # 2160p24
    95: (3840, 2160, 4400, 2250, 297_000_000),  # 2160p30
    97: (3840, 2160, 4400, 2250, 594_000_000),  # 2160p60
}
CTA_NAMES = {
    1: "640x480@60",
    4: "1280x720@60",
    16: "1920x1080@60",
    19: "1280x720@50",
    31: "1920x1080@50",
    32: "1920x1080@24",
    34: "1920x1080@30",
    47: "1280x720@120",
    63: "1920x1080@120",
    64: "1920x1080@100",
    93: "3840x2160@24",
    95: "3840x2160@30",
    97: "3840x2160@60",
}


def cta_mode(vic: int) -> VideoMode:
    h, v, ht, vt, pc = CTA[vic]
    return VideoMode(CTA_NAMES[vic], h, v, ht, vt, pc, "cta")


def _vsync_lines(h: int, v: int) -> int:
    r = h / v
    if abs(r - 4 / 3) < 0.02:
        return 4
    if abs(r - 16 / 9) < 0.02:
        return 5
    if abs(r - 16 / 10) < 0.02:
        return 6
    if abs(r - 5 / 4) < 0.02:
        return 7
    return 10


def cvt_rb(h_active: int, v_active: int, refresh_hz: float) -> VideoMode:
    """CVT reduced blanking v1 (what high-refresh monitors report as their timing): 160 pixels of horizontal blanking, at least 460 us of
    vertical blanking, pixel clock rounded down to 0.25 MHz."""
    h_period_est = ((1e6 / refresh_hz) - 460.0) / v_active
    vbi = math.floor(460.0 / h_period_est) + 1
    vbi = max(vbi, 3 + _vsync_lines(h_active, v_active) + 6)
    h_total, v_total = h_active + 160, v_active + vbi
    pclk = int(h_total * v_total * refresh_hz // 250_000) * 250_000
    return VideoMode(f"{h_active}x{v_active}@{refresh_hz:g}", h_active, v_active, h_total, v_total, pclk, "cvt-rb")


def mode(name: str) -> VideoMode:
    """'1920x1080@144' -> the CTA mode if one exists at that rate, else CVT-RB."""
    wh, hz = name.lower().split("@")
    w, h = map(int, wh.split("x"))
    hz_f = float(hz)
    for vic, (ha, va, ht, vt, pc) in CTA.items():
        if (ha, va) == (w, h) and abs(pc / (ht * vt) - hz_f) < 0.6 and name.lower() == CTA_NAMES[vic]:
            return cta_mode(vic)
    return cvt_rb(w, h, hz_f)


# ---- link budgets: what a tap built from commodity parts can follow --------------------------------------------------------
HDMI_TMDS_MAX_CHAR_HZ = 600_000_000  # HDMI 2.0 (scrambled TMDS): 3 lanes x 6 Gb/s = 18 Gb/s
HDMI_NO_SCRAMBLING_MAX_CHAR_HZ = 340_000_000
DP_RATES_GBPS = {"RBR": 1.62, "HBR": 2.7, "HBR2": 5.4, "HBR3": 8.1}


def hdmi_char_rate_hz(m: VideoMode, bpc: int = 8, ycbcr420: bool = False) -> float:
    r = m.pclk_hz * (bpc / 8)
    return r / 2 if ycbcr420 else r


def dp_payload_gbps(m: VideoMode, bpc: int = 8) -> float:
    return m.pclk_hz * 3 * bpc / 1e9


def dp_link_capacity_gbps(rate: str = "HBR2", lanes: int = 4) -> float:
    return DP_RATES_GBPS[rate] * lanes * 0.8  # 8b/10b


def tap_supported(m: VideoMode, interface: str, bpc: int = 8, dp_rate: str = "HBR2") -> tuple[bool, str]:
    """Can a TMDS (HDMI 2.0 class) or DP 1.2 (HBR2) class tap follow this mode? Beyond that the link is FRL / HBR3 / DSC, which these parts do not decode."""  # noqa: E501
    if interface == "hdmi":
        r = hdmi_char_rate_hz(m, bpc)
        if r <= HDMI_TMDS_MAX_CHAR_HZ:
            return True, f"TMDS {r / 1e6:.0f} MHz <= 600 MHz"
        return False, f"TMDS {r / 1e6:.0f} MHz > 600 MHz (needs FRL / 4:2:0 / DSC)"
    if interface == "dp":
        need, cap = dp_payload_gbps(m, bpc), dp_link_capacity_gbps(dp_rate)
        if need <= cap:
            return True, f"{need:.1f} Gb/s <= {cap:.2f} Gb/s ({dp_rate} x4)"
        return False, f"{need:.1f} Gb/s > {cap:.2f} Gb/s ({dp_rate} x4): HBR3/DSC territory"
    raise ValueError(interface)


@dataclass(frozen=True)
class Roi:
    x: int
    y: int
    w: int
    h: int


def last_row_needed(roi: Roi) -> int:
    return roi.y + roi.h - 1


def data_age_us(m: VideoMode, roi: Roi, tail_us: float = 0.0) -> dict:
    """Timeline of one captured frame, microseconds after the start of the active video. `ready` is when the ROI's last line has arrived
    plus the processing that cannot start earlier; `age_of_content` is how old the ROI's middle was by then."""
    mid = roi.y + roi.h / 2
    t_ready = m.row_done_us(last_row_needed(roi)) + tail_us
    return {
        "line_time_us": m.line_time_us,
        "frame_time_us": m.frame_time_us,
        "roi_first_row_us": m.row_done_us(roi.y - 1) if roi.y else 0.0,
        "last_row_us": m.row_done_us(last_row_needed(roi)),
        "ready_us": t_ready,
        "mid_row_us": (mid) * m.line_time_us,
        "age_of_content_us": t_ready - mid * m.line_time_us,
    }
