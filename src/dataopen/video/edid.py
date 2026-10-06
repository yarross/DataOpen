"""EDID: parsing (what the monitor can display, so what the GPU will choose and what the tap must follow), a builder for tests, and the
passthrough rules. The rule of the device is simple and has no exceptions: the GPU sees the monitor's EDID **byte for byte**, and the
monitor's presence (hot plug) is the GPU's presence signal. The device never edits, merges, filters or synthesizes an EDID."""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Optional

from .timing import CTA, CTA_NAMES, VideoMode, cta_mode

HEADER = bytes([0, 255, 255, 255, 255, 255, 255, 0])


class EdidError(ValueError):
    pass


def checksum_ok(block: bytes) -> bool:
    return len(block) == 128 and sum(block) % 256 == 0


def _fix_checksum(block: bytearray) -> bytes:
    block[127] = (-sum(block[:127])) % 256
    return bytes(block)


@dataclass
class EdidInfo:
    manufacturer: str
    product_code: int
    version: tuple[int, int]
    n_ext: int
    modes: list[VideoMode] = field(default_factory=list)  # preferred first
    preferred: Optional[VideoMode] = None
    hdmi: bool = False
    max_tmds_hz: Optional[int] = None  # from the HDMI / HDMI Forum vendor blocks, if present
    warnings: list[str] = field(default_factory=list)


def _dtd(d: bytes) -> Optional[VideoMode]:
    pclk = struct.unpack_from("<H", d, 0)[0] * 10_000
    if pclk == 0:
        return None
    ha = d[2] | ((d[4] >> 4) << 8)
    hb = d[3] | ((d[4] & 15) << 8)
    va = d[5] | ((d[7] >> 4) << 8)
    vb = d[6] | ((d[7] & 15) << 8)
    return VideoMode(f"{ha}x{va}", ha, va, ha + hb, va + vb, pclk, "edid")


def parse(edid: bytes) -> EdidInfo:
    if len(edid) < 128 or len(edid) % 128:
        raise EdidError("EDID length must be a multiple of 128")
    base = edid[:128]
    if base[:8] != HEADER:
        raise EdidError("bad EDID header")
    if not checksum_ok(base):
        raise EdidError("bad base block checksum")
    mid = struct.unpack_from(">H", base, 8)[0]
    man = "".join(chr(64 + ((mid >> s) & 31)) for s in (10, 5, 0))
    info = EdidInfo(man, struct.unpack_from("<H", base, 10)[0], (base[18], base[19]), base[126])
    if len(edid) != 128 * (1 + base[126]):
        info.warnings.append("length does not match the extension count")
    for off in (54, 72, 90, 108):
        m = _dtd(base[off : off + 18])
        if m is not None:
            info.modes.append(m)
    for k in range(base[126]):
        blk = edid[128 * (k + 1) : 128 * (k + 2)]
        if len(blk) < 128 or not checksum_ok(blk):
            info.warnings.append(f"extension {k + 1}: bad or missing checksum")
            continue
        if blk[0] != 2:
            continue
        dtd_off = blk[2]
        i = 4
        while i < dtd_off and i < 127:
            tag, n = blk[i] >> 5, blk[i] & 31
            data = blk[i + 1 : i + 1 + n]
            if tag == 2:
                for b in data:
                    vic = b & 0x7F
                    if vic in CTA:
                        info.modes.append(cta_mode(vic))
            elif tag == 3 and n >= 3:
                oui = bytes(data[:3])
                if oui == bytes([0x03, 0x0C, 0x00]):
                    info.hdmi = True
                    if n >= 7 and data[6]:
                        info.max_tmds_hz = max(info.max_tmds_hz or 0, data[6] * 5_000_000)
                elif oui == bytes([0xD8, 0x5D, 0xC4]) and n >= 7:
                    info.hdmi = True
                    info.max_tmds_hz = max(info.max_tmds_hz or 0, data[5] * 5_000_000)
            i += 1 + n
        if dtd_off >= 4:
            j = dtd_off
            while j + 18 <= 127:
                m = _dtd(blk[j : j + 18])
                if m is None:
                    break
                info.modes.append(m)
                j += 18
    first_dtd = _dtd(base[54:72])
    info.preferred = first_dtd
    return info


def build(
    modes: list[VideoMode],
    preferred: int = 0,
    manufacturer: str = "ACM",
    product: int = 0x1234,
    hdmi_max_tmds_mhz: Optional[int] = None,
    vics: tuple[int, ...] = (),
) -> bytes:
    """A valid EDID 1.4 with CTA-861 extension (for tests and for sanity-checking the rules; not for programming a device)."""
    base = bytearray(128)
    base[:8] = HEADER
    mid = sum((ord(c) - 64) << s for c, s in zip(manufacturer, (10, 5, 0)))
    struct.pack_into(">H", base, 8, mid)
    struct.pack_into("<H", base, 10, product)
    base[18], base[19] = 1, 4
    base[20] = 0xA5
    ordered = [modes[preferred]] + [m for i, m in enumerate(modes) if i != preferred]
    for k, m in enumerate(ordered[:4]):
        if m.pclk_hz // 10_000 > 0xFFFF:
            raise EdidError("pixel clock above 655.35 MHz does not fit a base-EDID DTD (DisplayID timing; not modelled)")
        hb, vb = m.h_total - m.h_active, m.v_total - m.v_active
        d = bytearray(18)
        struct.pack_into("<H", d, 0, m.pclk_hz // 10_000)
        d[2], d[3] = m.h_active & 255, hb & 255
        d[4] = ((m.h_active >> 8) << 4) | (hb >> 8)
        d[5], d[6] = m.v_active & 255, vb & 255
        d[7] = ((m.v_active >> 8) << 4) | (vb >> 8)
        base[54 + 18 * k : 72 + 18 * k] = d
    n_ext = 1 if (vics or hdmi_max_tmds_mhz) else 0
    base[126] = n_ext
    out = _fix_checksum(base)
    if n_ext:
        blk = bytearray(128)
        blk[0], blk[1] = 2, 3
        body = b""
        if vics:
            body += bytes([(2 << 5) | len(vics)]) + bytes(vics)
        if hdmi_max_tmds_mhz:
            body += bytes([(3 << 5) | 7, 0x03, 0x0C, 0x00, 0x10, 0x00, 0x00, hdmi_max_tmds_mhz // 5])
        blk[4 : 4 + len(body)] = body
        blk[2] = 4 + len(body)
        out += _fix_checksum(blk)
    return out


@dataclass
class PassthroughState:
    hpd_to_gpu: bool = False
    edid_to_gpu: Optional[bytes] = None
    events: list[str] = field(default_factory=list)


class EdidProxy:
    """The EDID/HPD rules, as a model the tests hold the hardware design to.

    * monitor present  -> the GPU is shown HPD high and reads exactly the monitor's EDID bytes;
    * monitor replaced -> HPD low for >= 100 ms (the HDMI/DP re-read pulse), then the new monitor's EDID, again byte for byte;
    * monitor absent   -> HPD low. Nothing is invented (no "safe" EDID): the GPU sees what it would see without the device.
    DDC/CI, SCDC and HDCP traffic is forwarded to the monitor, never answered by the device."""

    HPD_PULSE_US = 100_000

    def __init__(self) -> None:
        self.state = PassthroughState()
        self._monitor_edid: Optional[bytes] = None
        self._pulse_until = 0

    def monitor(self, t_us: int, edid: Optional[bytes]) -> None:
        if edid is None:
            self._monitor_edid = None
            self.state.events.append(f"{t_us}: monitor gone, HPD low")
            return
        if self._monitor_edid is not None and edid != self._monitor_edid:
            self._pulse_until = t_us + self.HPD_PULSE_US
            self.state.events.append(f"{t_us}: monitor replaced, HPD pulse")
        self._monitor_edid = bytes(edid)

    def tick(self, t_us: int) -> PassthroughState:
        present = self._monitor_edid is not None and t_us >= self._pulse_until
        self.state.hpd_to_gpu = present
        self.state.edid_to_gpu = self._monitor_edid if present else None
        return self.state

    def read(self, t_us: int, offset: int, n: int) -> Optional[bytes]:
        """A DDC read of the EDID by the GPU: the monitor's bytes, or None (NACK) while HPD is low."""
        st = self.tick(t_us)
        return None if st.edid_to_gpu is None else st.edid_to_gpu[offset : offset + n]


def required_tap_modes(info: EdidInfo) -> list[VideoMode]:
    """The modes the GPU may legitimately choose (all that the EDID lists): the tap has to follow every one of them, because the GPU, not
    the device, decides."""
    seen, out = set(), []
    for m in info.modes:
        k = (m.h_active, m.v_active, m.pclk_hz)
        if k not in seen:
            seen.add(k)
            out.append(m)
    return out


__all__ = ["EdidInfo", "EdidError", "EdidProxy", "parse", "build", "required_tap_modes", "checksum_ok", "CTA_NAMES"]
