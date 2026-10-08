"""SKUs through DNP groups on one core board, the bill of materials, the enclosure and the block structure.

All prices are ORDER-OF-MAGNITUDE guides in US dollars at roughly 1000 units, written from memory, not quotes. Part families named in
`note` are CANDIDATES to be checked against their datasheets; none of them has been evaluated.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# --------------------------------------------------------------------------------------------------------------------------- DNP groups
CORE = "CORE"
G_HDMI, G_DP, G_USBC_MOUSE, G_AUDIO, G_WIFI = "G_HDMI", "G_DP", "G_USBC_MOUSE", "G_AUDIO", "G_WIFI"
GROUPS = (G_HDMI, G_DP, G_USBC_MOUSE, G_AUDIO, G_WIFI)
RETAIL_FACTOR = 2.3                        # BOM -> shelf price: margin, channel, certification, support, returns (a guess, not a quote)
RETAIL_STEP = 10


@dataclass(frozen=True)
class Part:
    ref: str
    name: str
    group: str                             # CORE (always fitted) or a DNP group
    qty: int
    lo: float                              # unit price guide, USD
    hi: float
    note: str = ""
    port: Optional[str] = None             # a connector or control that appears on the enclosure
    candidate: bool = False                # a family named from memory, to be verified


P = Part
PARTS: tuple[Part, ...] = (
    # ---- compute and bridge
    P("U1", "Compute module (SoM), RK3588-class, 4 GB / 32 GB eMMC, board-to-board", CORE, 1, 55, 95, "same class on every SKU (docs/VIDEO.md); HDMI RX, NPU", candidate=True),
    P("U2", "Bridge MCU, Cortex-M7 600 MHz, two USB HS (i.MX RT1062 class)", CORE, 1, 11, 16, "docs/BRIDGE.md reference part; RP2040 is the budget idea there", candidate=True),
    P("U3", "QSPI flash 16 MB for the bridge firmware (two banks)", CORE, 1, 1.5, 2.5),
    P("U4", "Secure element (ATECC608-class), for the key store that is a file today", CORE, 1, 0.8, 1.4, "firmware support not written", candidate=True),
    P("M1", "BLE 5 module (PWA control link)", CORE, 1, 3, 5, "GATT server on the SoM is not written", candidate=True),
    # ---- the fail-safe path
    P("K1", "Signal relay DPDT, normally closed, USB 2.0 HS capable (mouse D+/D-)", CORE, 1, 2.5, 4.5, "released = mouse straight to the PC", candidate=True),
    P("K2", "Relay SPDT, normally closed (mouse VBUS: PC rail when released, adapter rail when engaged)", CORE, 1, 1.5, 3),
    P("U5", "Voltage supervisor + window watchdog (own reset, no MCU needed)", CORE, 1, 1.5, 3, "100 ms window as in docs/BRIDGE.md", candidate=True),
    P("U6", "AND gates and coil drivers of the safety domain (discrete logic + MOSFETs)", CORE, 1, 1.0, 2.0, "no processor in this path"),
    P("R1", "Panic loop RC (about 3 s), pull-downs, flyback diodes", CORE, 1, 0.3, 0.6),
    # ---- power
    P("U7", "USB-C input protection, e-fuse / load switch with over-current limit", CORE, 1, 1.0, 2.0, candidate=True),
    P("U8", "Bucks and LDOs: 3V3 always-on, 3V3 IO, 1V8, 1V2", CORE, 1, 5, 9),
    P("U9", "Mouse VBUS load switch, 500 mA limit", CORE, 1, 0.7, 1.2),
    P("C1", "Bulk and hold-up capacitors", CORE, 1, 1.5, 3),
    # ---- connectors and controls that are on every SKU
    P("J1", "USB-C power input receptacle", CORE, 1, 0.5, 0.9, port="POWER"),
    P("J2", "USB-A receptacle for the mouse", CORE, 1, 0.4, 0.8, port="MOUSE A"),
    P("J3", "USB-C receptacle 'TO PC' (USB 2.0 data only)", CORE, 1, 0.5, 0.9, port="TO PC"),
    P("J4", "3.5 mm jack for an external big button, with shorting contact", CORE, 1, 0.4, 0.8, "open loop = bypass", port="EXT BUTTON"),
    P("SW1", "PANIC: large pushbutton, one NC and one NO contact", CORE, 1, 3, 6, port="PANIC"),
    P("SW2", "MODE: mechanical slide switch ASSIST / BYPASS", CORE, 1, 1, 2, port="MODE"),
    P("SW3", "SLOT tactile button", CORE, 1, 0.2, 0.4, port="SLOT"),
    P("SW4", "CONFIRM tactile button", CORE, 1, 0.2, 0.4, port="CONFIRM"),
    P("D1", "RGB status LED + driver", CORE, 1, 0.4, 0.8, port="STATUS LED"),
    P("D2", "BYPASS amber LED, driven by the safety domain (no firmware)", CORE, 1, 0.1, 0.2, port="BYPASS LED"),
    P("D3", "Slot LEDs, 4 pieces", CORE, 4, 0.1, 0.2, port="SLOT LEDS"),
    P("SW5", "RECOVERY: recessed pushbutton (a paper clip), holds the SoM boot-ROM pin while power is applied", CORE, 1, 0.1, 0.2, port="RECOVERY"),
    P("E4", "ESD and series resistors for the service data lines of the POWER port (to the SoM USB OTG; unused in normal operation)", CORE, 1, 0.3, 0.6),
    P("TP1", "Pad field for the factory jig (pogo pins); the debug port is fused off at the end of provisioning", CORE, 1, 0.3, 0.8),
    # ---- board, enclosure, box
    P("PCB", "6-layer board, about 90 x 60 mm", CORE, 1, 8, 14),
    P("MSC", "Passives, ESD arrays, connectors of the SoM, fuses", CORE, 1, 6, 10),
    P("ASM", "SMT assembly and handling", CORE, 1, 12, 22),
    P("TST", "Programming and functional test", CORE, 1, 2, 4),
    P("ENC", "Extruded aluminium case, end plates, light pipes, thermal pad, feet", CORE, 1, 9, 16),
    P("BOX", "In the box: 5 V / 3 A USB-C supply, two cables, printed guide, packaging", CORE, 1, 7, 13),
    # ---- G_HDMI
    P("J5", "HDMI receptacle IN (from the graphics card)", G_HDMI, 1, 0.5, 1.0, port="HDMI IN"),
    P("J6", "HDMI receptacle OUT (to the monitor)", G_HDMI, 1, 0.5, 1.0, port="HDMI OUT"),
    P("U10", "TMDS splitter 1:2 (LT86102 class)", G_HDMI, 1, 7, 12, "docs/VIDEO.md candidate", candidate=True),
    P("U11", "HDMI to MIPI CSI-2 bridge (LT6911UXC class)", G_HDMI, 1, 8, 14, "docs/VIDEO.md candidate", candidate=True),
    P("K3", "RF relay DPDT, normally closed, one per TMDS pair (4 pieces)", G_HDMI, 4, 6, 11, "signal integrity at 3.4 Gbit/s per lane NOT verified", candidate=True),
    P("K4", "Signal relay 4-pole, normally closed (DDC, CEC, HPD, +5V)", G_HDMI, 1, 2, 4),
    P("E1", "ESD arrays and common-mode parts for HDMI", G_HDMI, 1, 1.5, 3),
    # ---- G_DP
    P("J7", "DisplayPort receptacle IN", G_DP, 1, 0.7, 1.2, port="DP IN"),
    P("J8", "DisplayPort receptacle OUT", G_DP, 1, 0.7, 1.2, port="DP OUT"),
    P("U12", "DisplayPort repeater / redriver for the monitor branch", G_DP, 1, 5, 9, "docs/VIDEO.md: DP has no passive splitter", candidate=True),
    P("U13", "DisplayPort to MIPI CSI-2 bridge (LT7911 class)", G_DP, 1, 9, 16, candidate=True),
    P("K5", "RF relay NC, one per DP lane and one for AUX (5 pieces)", G_DP, 5, 6, 11, "HBR (2.7 Gbit/s per lane) only; NOT verified", candidate=True),
    P("E2", "ESD arrays and AUX level shifting for DisplayPort", G_DP, 1, 1.5, 3),
    # ---- G_USBC_MOUSE
    P("J9", "USB-C receptacle for a USB-C mouse (USB 2.0, Rp on CC)", G_USBC_MOUSE, 1, 0.5, 0.9, port="MOUSE C"),
    P("U14", "Second mouse VBUS load switch (the two ports never both powered)", G_USBC_MOUSE, 1, 0.7, 1.2),
    P("E3", "CC network and ESD for the mouse Type-C port", G_USBC_MOUSE, 1, 0.3, 0.7),
    # ---- G_AUDIO, G_WIFI
    P("BZ1", "Piezo buzzer and driver, quiet", G_AUDIO, 1, 0.4, 0.9, port="BUZZER"),
    P("M2", "Wi-Fi module for the SoftAP link (the iPhone route, docs/PWA.md)", G_WIFI, 1, 4, 7, "software for it not written", candidate=True),
)


@dataclass(frozen=True)
class Sku:
    code: str
    name: str
    groups: tuple[str, ...]                # the DNP groups that are FITTED on this SKU (CORE is always fitted)
    audience: str
    video: str
    mouse: str

    def fits(self, part: Part) -> bool:
        return part.group == CORE or part.group in self.groups


SKUS: tuple[Sku, ...] = (
    Sku("DO-1", "Base", (G_HDMI,), "один компьютер с HDMI, обычная USB-A мышь, самый доступный вариант", "HDMI", "USB-A"),
    Sku("DO-2", "DP", (G_DP,), "компьютер с DisplayPort, обычная USB-A мышь", "DisplayPort", "USB-A"),
    Sku("DO-3", "Type-C", (G_HDMI, G_USBC_MOUSE), "HDMI; в том числе мыши и трекболы с USB-C кабелем", "HDMI", "USB-A + USB-C"),
    Sku("DO-4", "Pro", (G_HDMI, G_DP, G_USBC_MOUSE, G_AUDIO, G_WIFI), "клиники и мастерские, несколько рабочих мест, все порты, звук и Wi-Fi", "HDMI + DisplayPort",
        "USB-A + USB-C"),
)


def sku(code: str) -> Sku:
    for s in SKUS:
        if s.code == code or s.name.lower() == code.lower():
            return s
    raise KeyError(code)


def bom(s: Sku) -> list[Part]:
    return [p for p in PARTS if s.fits(p)]


def dnp(s: Sku) -> list[Part]:
    return [p for p in PARTS if not s.fits(p)]


def cost(s: Sku) -> tuple[float, float]:
    items = bom(s)
    return sum(p.qty * p.lo for p in items), sum(p.qty * p.hi for p in items)


def retail(s: Sku) -> int:
    lo, hi = cost(s)
    return int(round((lo + hi) / 2 * RETAIL_FACTOR / RETAIL_STEP)) * RETAIL_STEP


# ----------------------------------------------------------------------------------------------------------------------------- enclosure
@dataclass(frozen=True)
class Face:
    name: str
    length_mm: float                       # the room along the face
    where: str
    rows: int = 1                          # how many rows of controls fit across it (the top is a surface, the ends are a line)


FACES = (Face("top", 100, "верх: органы управления и индикация", rows=3), Face("mouse-end", 75, "торец со стороны мыши"),
         Face("pc-end", 75, "торец со стороны ПК"), Face("monitor-side", 100, "боковая грань к монитору"),
         Face("bottom", 100, "низ: утопленная кнопка RECOVERY, наклейка с серийным номером и QR", rows=2))
PORT_FACE = {"POWER": "monitor-side", "TO PC": "pc-end", "HDMI IN": "pc-end", "DP IN": "pc-end", "HDMI OUT": "monitor-side", "DP OUT": "monitor-side",
             "MOUSE A": "mouse-end", "MOUSE C": "mouse-end", "EXT BUTTON": "mouse-end", "PANIC": "top", "MODE": "top", "SLOT": "top",
             "CONFIRM": "top", "RECOVERY": "bottom", "STATUS LED": "top", "BYPASS LED": "top", "SLOT LEDS": "top", "BUZZER": "top"}
PORT_MM = {"POWER": 10, "TO PC": 10, "HDMI IN": 16, "DP IN": 19, "HDMI OUT": 16, "DP OUT": 19, "MOUSE A": 16, "MOUSE C": 10, "EXT BUTTON": 8,
           "PANIC": 34, "MODE": 14, "SLOT": 8, "CONFIRM": 8, "RECOVERY": 6, "STATUS LED": 6, "BYPASS LED": 4, "SLOT LEDS": 24, "BUZZER": 3}
GAP_MM = 6.0
ENCLOSURE = {"size_mm": (100, 75, 26), "material": "алюминиевый экструдированный профиль 6063 + торцевые крышки из PC/ABS, светопроводы из PC",
             "ingress": "IP20 (не герметичен)", "mass_g": "ориентир 220-280", "cooling": "пассивное: корпус это радиатор, термопрокладка от SoM"}


def ports(s: Sku) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {f.name: [] for f in FACES}
    for p in bom(s):
        if p.port:
            out[PORT_FACE[p.port]].append(p.port)
    return out


def face_room_mm(f: Face) -> float:
    return f.length_mm * f.rows


def face_use_mm(s: Sku) -> dict[str, float]:
    pl = ports(s)
    return {f.name: sum(PORT_MM[x] + GAP_MM for x in pl[f.name]) + GAP_MM for f in FACES}


# ---------------------------------------------------------------------------------------------------------------------- blocks and links
# domain: D0 safety (VEXT only, no software), D1 bridge MCU, D2 compute, D3 video (power-gated), D4 indication, IO connectors
BLOCKS = {
    "VEXT": "D0", "PWR_IN": "D0", "SUPERVISOR": "D0", "WDG": "D0", "PANIC_LOOP": "D0", "MODE_SW": "D0", "GATE": "D0", "COIL_DRV": "D0",
    "K1": "D0", "K2": "D0", "KV": "D0", "BYPASS_LED": "D0",
    "MCU": "D1", "SOM": "D2", "BLE": "D2", "VIDEO_TAP": "D3", "RGB": "D4", "SLOT_LEDS": "D4", "BUZZER": "D4",
    "MOUSE": "IO", "PC": "IO", "GPU": "IO", "MONITOR": "IO", "BTN_SOFT": "IO", "PHONE": "IO",
}
# (source, destination, what). Only GATE may drive COIL_DRV; MCU and SOM reach the relays only as INPUTS of the gate.
LINKS = (
    ("PWR_IN", "VEXT", "5 V from the separate USB-C supply"), ("VEXT", "SUPERVISOR", "VEXT_OK threshold"), ("VEXT", "GATE", "supply of the gate"),
    ("SUPERVISOR", "GATE", "VEXT_OK"), ("WDG", "GATE", "WDG_OK"), ("PANIC_LOOP", "GATE", "PANIC_LOOP_OK"), ("MODE_SW", "GATE", "MODE_ASSIST"),
    ("MCU", "WDG", "kick"), ("MCU", "GATE", "ENGAGE request"), ("MCU", "GATE", "VIDEO_EN request"),
    ("GATE", "COIL_DRV", "mouse / video coil enable"), ("COIL_DRV", "K1", "coil"), ("COIL_DRV", "K2", "coil"), ("COIL_DRV", "KV", "coil"),
    ("GATE", "BYPASS_LED", "NOT mouse path engaged"),
    ("MOUSE", "K1", "D+/D-"), ("K1", "PC", "D+/D- direct when released"), ("K1", "MCU", "USB host side when engaged"), ("MCU", "PC", "USB device side"),
    ("PC", "K2", "VBUS sense / pass-through when released"), ("K2", "MOUSE", "VBUS"),
    ("GPU", "KV", "TMDS / DP"), ("KV", "MONITOR", "direct when released"), ("KV", "VIDEO_TAP", "through the splitter when engaged"),
    ("VIDEO_TAP", "MONITOR", "monitor branch"), ("VIDEO_TAP", "SOM", "MIPI CSI-2"),
    ("MCU", "SOM", "BridgeLink SPI + DRDY"), ("SOM", "MCU", "ISP: BOOT / RESET / UART (the SoM re-flashes the MCU)"),
    ("PWR_IN", "SOM", "service USB data lines (recovery only; unused in normal operation)"), ("SOM", "BLE", "HCI"), ("PHONE", "BLE", "GATT / CtlLink"), ("SOM", "RGB", "pattern"),
    ("SOM", "SLOT_LEDS", "pattern"), ("SOM", "BUZZER", "pattern"), ("BTN_SOFT", "SOM", "SLOT / CONFIRM"),
)

DIAGRAM = r"""
   MOUSE (USB-A / USB-C)                                                         PC
      │ D+/D-, VBUS                                                               ▲  USB 2.0 data (TO PC)
      ▼                                                                           │
 ┌──────────┐  released (no power, Panic, hang, MODE)  ────────────────────────► direct
 │ K1 K2    │──────────────────────────────────────────────────────────────────────┘
 │ NC relays│  engaged ──► ┌───────────────┐  USB host    ┌───────────┐  USB device
 └────▲─────┘              │  BRIDGE MCU   │◄────────────►│ (as PC    │──────────► PC
      │ coil                │ (RT1062 class)│              │  sees it) │
      │                     └──┬─────────┬──┘              └───────────┘
 ┌────┴──────────────────┐     │ kick    │ BridgeLink SPI + DRDY
 │ SAFETY DOMAIN D0      │◄────┘ ENGAGE/VIDEO_EN requests only
 │ VEXT_OK · WDG_OK ·    │                      │
 │ PANIC_LOOP_OK · MODE  │ ──► BYPASS LED       ▼
 │ (gates, no processor) │              ┌───────────────┐  BLE      ┌───────┐
 └────┬──────────────────┘              │  COMPUTE (SoM)│◄─────────►│ PHONE │ (PWA)
      │ coil                            │ RK3588 class  │           └───────┘
 ┌────▼─────┐                           └───▲───────▲───┘
 │ KV relays│ released: GPU ─► MONITOR      │ MIPI  │ RGB / 4 slot LEDs / buzzer / SLOT / CONFIRM
 │ NC, RF   │ engaged:  GPU ─► splitter ─┬──┘       ▼
 └──────────┘                           └─► MONITOR  (indication, panel)
 GPU (HDMI / DP) ─► KV                      capture (HDMI->MIPI, DP->MIPI)

 POWER: separate USB-C 5 V / 3 A ─► e-fuse ─► D0 always on (3V3_AON) ─► D1 ─► D2 ─► D3 (only when video is used) ─► D4
        PC VBUS: sensed through a divider (< 10 µA) and passed to the mouse ONLY through K2 when released. The adapter takes nothing from the PC.
""".strip("\n")
