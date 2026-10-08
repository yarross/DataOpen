"""Power domains, the power budget per SKU and the passive-cooling estimate (docs/HARDWARE.md section 4 and 7).

Every wattage here is an ESTIMATE written down so that the requirements (the supply, the thermal cap) are checkable against each other.
None of it has been measured on a board.
"""
from __future__ import annotations

from dataclasses import dataclass

from . import spec as S

SUPPLY_V, SUPPLY_A = 5.0, 3.0
SUPPLY_W = SUPPLY_V * SUPPLY_A
DERATE = 0.8                               # nothing is asked of the supply beyond 80 % of its rating
CONVERSION = 0.85                          # bucks and load switches: input power = load / this
PC_SENSE_UA = 10                           # what the adapter may take from the PC's VBUS: a sense divider, nothing else
MOUSE_PC_MA = 500                          # in bypass the PC powers the mouse exactly as with a direct cable
HOLD_UP_MS = 10                            # logic stays alive this long after VEXT is lost; the relay coils do not (they release at once)

# thermal: natural convection + radiation from the case. DT = P / (h * A)
H_W_M2K = 10.0
TOUCH_DT_K = 22.0                          # 47 C surface in a 25 C room: acceptable for a device that is touched
MODES = ("mouse_only", "video", "peak")

# domains: name -> what is in it and when it is powered
DOMAINS = {
    "D0": ("safety: supervisor, window watchdog, panic loop, MODE switch, gates, coil drivers, BYPASS LED", "always, from VEXT only; no software"),
    "D1": ("bridge MCU and its flash", "from VEXT after D0 is good; starts the bridge, which runs without the SoM"),
    "D2": ("compute module, BLE, (Wi-Fi), eMMC", "from VEXT; 20-40 s to boot (an assumption); the bridge passes the mouse through meanwhile"),
    "D3": ("video: splitter / DP repeater, capture bridges, video relay coils", "load-switched; ON only while the UI detector uses the screen"),
    "D4": ("RGB, slot LEDs, buzzer", "from VEXT"),
}
SEQUENCE = ("VEXT in range", "D0", "D1", "D2", "D3 (only on request)", "D4")


@dataclass(frozen=True)
class Load:
    name: str
    domain: str
    mouse_only: float                      # W, mouse assisted, no video
    video: float                           # W, UI detector running
    peak: float                            # W, short peak with the SoM power governor at its cap
    group: str = S.CORE                    # fitted only where this group is fitted
    note: str = ""


LOADS = (
    Load("SoM + eMMC", "D2", 1.8, 3.2, 4.5, note="peak is the CAP the power governor must enforce (the part could take more)"),
    Load("bridge MCU + flash", "D1", 0.45, 0.45, 0.6),
    Load("BLE", "D2", 0.10, 0.10, 0.25),
    Load("safety domain logic", "D0", 0.05, 0.05, 0.05),
    Load("K1 + K2 coils (engaged = coil on)", "D0", 0.30, 0.30, 0.30),
    Load("mouse (adapter rail)", "D1", 0.50, 0.50, 0.50, note="500 mA limit; PC rail is used instead while released"),
    Load("LEDs", "D4", 0.15, 0.15, 0.20),
    Load("HDMI splitter + capture + relay coils", "D3", 0.0, 2.65, 2.65, S.G_HDMI, "splitter 0.9, capture 1.0, 5 relay coils 0.75"),
    Load("DP repeater + capture + relay coils", "D3", 0.0, 2.90, 2.90, S.G_DP, "repeater 0.7, capture 1.1, 6 relay coils 1.1"),
    Load("Type-C mouse port", "D1", 0.0, 0.0, 0.05, S.G_USBC_MOUSE),
    Load("Wi-Fi", "D2", 0.0, 0.20, 0.50, S.G_WIFI),
    Load("buzzer", "D4", 0.0, 0.0, 0.05, S.G_AUDIO),
)


def _active_video_load(s: S.Sku, mode: str) -> list[Load]:
    """With both HDMI and DP fitted only the input in use is powered (D3 is one switched rail): count the larger, not the sum."""
    video = [l for l in LOADS if l.domain == "D3" and l.group in s.groups]
    if mode == "mouse_only" or not video:
        return []
    return [max(video, key=lambda l: getattr(l, mode))]


def loads(s: S.Sku, mode: str) -> list[tuple[Load, float]]:
    if mode not in MODES:
        raise ValueError(mode)
    out = [(l, getattr(l, mode)) for l in LOADS if l.domain != "D3" and (l.group == S.CORE or l.group in s.groups)]
    out += [(l, getattr(l, mode)) for l in _active_video_load(s, mode)]
    return out


def load_w(s: S.Sku, mode: str) -> float:
    return sum(w for _, w in loads(s, mode))


def input_w(s: S.Sku, mode: str) -> float:
    """What the USB-C supply has to deliver."""
    return load_w(s, mode) / CONVERSION


def budget(s: S.Sku) -> dict:
    return {m: round(input_w(s, m), 2) for m in MODES} | {"limit_w": SUPPLY_W * DERATE,
                                                          "margin_peak_w": round(SUPPLY_W * DERATE - input_w(s, "peak"), 2)}


def case_area_m2() -> float:
    x, y, z = S.ENCLOSURE["size_mm"]
    return 2 * (x * y + x * z + y * z) / 1e6


def r_th_k_per_w() -> float:
    return 1.0 / (H_W_M2K * case_area_m2())


def delta_t_k(s: S.Sku, mode: str) -> float:
    """Case rise above the room for a power that is dissipated for hours (the peak is a minute, so it is judged on 'video')."""
    return (input_w(s, "video" if mode == "peak" else mode)) * r_th_k_per_w()


def sustained_cap_w() -> float:
    return TOUCH_DT_K / r_th_k_per_w()


def thermal(s: S.Sku) -> dict:
    return {m: {"dt_k": round(delta_t_k(s, m), 1), "ok": delta_t_k(s, m) <= TOUCH_DT_K} for m in ("mouse_only", "video")} | {
        "cap_w": round(sustained_cap_w(), 1), "r_th_k_per_w": round(r_th_k_per_w(), 2)}


def area_needed_m2(s: S.Sku, mode: str = "video") -> float:
    """Case surface that would hold `mode` at the touch limit with this h."""
    return input_w(s, mode) / (H_W_M2K * TOUCH_DT_K)
