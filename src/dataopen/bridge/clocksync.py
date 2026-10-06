"""Mapping the module's clock to the bridge's, from BridgeLink TSYNC exchanges (NTP-style, minimum round trip wins).

The module sends m_t (its time) in a TSYNC frame; the bridge answers (m_t, bridge_rx, bridge_tx); the module notes m_recv when the reply
arrives. offset = bridge - module = ((bridge_rx - m_t) + (bridge_tx - m_recv)) / 2, trustworthy to about half the round trip; the
exchange with the smallest round trip is the one kept. Frame timestamps in the module's clock become bridge time with `to_bridge`."""

from __future__ import annotations

from typing import Optional


class ClockSync:
    def __init__(self, keep: int = 16) -> None:
        self.keep = keep
        self._samples: list[tuple[float, float]] = []  # (rtt_us, offset_us)

    def add(self, m_send: int, bridge_rx: int, bridge_tx: int, m_recv: int) -> None:
        rtt = (m_recv - m_send) - (bridge_tx - bridge_rx)
        off = ((bridge_rx - m_send) + (bridge_tx - m_recv)) / 2
        self._samples.append((rtt, off))
        self._samples = self._samples[-self.keep :]

    @property
    def offset_us(self) -> Optional[float]:
        if not self._samples:
            return None
        return min(self._samples)[1]

    @property
    def uncertainty_us(self) -> Optional[float]:
        if not self._samples:
            return None
        return min(self._samples)[0] / 2

    def to_bridge(self, t_module_us: int) -> int:
        if self.offset_us is None:
            raise RuntimeError("no TSYNC exchange yet")
        return int(round(t_module_us + self.offset_us))
