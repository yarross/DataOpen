"""Frame-level filtering: negatives, empty/ambiguous frames, near-duplicates."""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from typing import Optional

import numpy as np

from .annotation import BuildResult, Verdict
from .models import FrameKind, FrameSnapshot, FrameSpec


class IFrameValidator(ABC):
    @abstractmethod
    def check(self, spec: FrameSpec, snap: FrameSnapshot, built: BuildResult) -> Optional[str]:
        """Return a reject reason, or None to accept."""


class NegativeFrameValidator(IFrameValidator):
    """A negative frame must contain no person pixels at all."""

    def check(self, spec, snap, built):
        if spec.kind is FrameKind.NEGATIVE and any(v is not Verdict.ABSENT for v in built.verdicts.values()):
            return "negative_contains_person"
        return None


class PositiveFrameValidator(IFrameValidator):
    """Positive frame needs >= 1 accepted person; optionally no unlabeled-but-present persons
    (a person that is visible but unlabeled teaches the detector that persons are background)."""

    def __init__(self, strict_unlabeled: bool = True) -> None:
        self.strict = strict_unlabeled

    def check(self, spec, snap, built):
        if spec.kind is not FrameKind.POSITIVE:
            return None
        if not built.annotations:
            return "positive_without_annotations"
        if self.strict and any(v.name.startswith("IGNORE") for v in built.verdicts.values()):
            return "unlabeled_person_present"
        return None


def dhash(thumb: np.ndarray) -> int:
    """64-bit difference hash from an (8, 9) grayscale thumbnail."""
    bits = (thumb[:, 1:] > thumb[:, :-1]).reshape(-1)
    return int("".join("1" if b else "0" for b in bits), 2)


class DuplicateFilter(IFrameValidator):
    """Rejects (a) frames whose coarse scene state (camera + skeletons, quantized) was already
    seen, and optionally (b) frames whose thumbnail hash is within `hamming` bits of a recent one.

    (b) is OFF by default: a tiny thumbnail barely changes when persons are small, and flat
    scenes (fog, sky) hash identically, so it false-positives on legitimately different frames.
    Enable it with hamming=0 to catch *stale GPU readbacks* (state moved, pixels did not).
    Sliding window keeps both checks O(window)."""

    def __init__(self, window: int = 512, hamming: int = 0, pos_quant: float = 0.05,
                 image_hash: bool = False) -> None:
        self.hamming, self.q, self.image_hash = hamming, pos_quant, image_hash
        self._sigs: set[bytes] = set()
        self._hashes: deque[int] = deque(maxlen=window)
        self._sig_order: deque[bytes] = deque(maxlen=window)

    def _signature(self, snap: FrameSnapshot) -> bytes:
        parts = [np.round(snap.camera.world_to_camera / self.q).astype(np.int64).tobytes()]
        for e in snap.entities:
            parts.append(np.round(e.skeleton_world / self.q).astype(np.int64).tobytes())
        return b"".join(parts)

    def check(self, spec, snap, built):
        sig = self._signature(snap)
        if sig in self._sigs:
            return "duplicate_state"
        h = dhash(snap.thumbnail) if (self.image_hash and snap.thumbnail is not None) else None
        if h is not None and any(bin(h ^ o).count("1") <= self.hamming for o in self._hashes):
            return "duplicate_image"
        if len(self._sig_order) == self._sig_order.maxlen:
            self._sigs.discard(self._sig_order[0])
        self._sig_order.append(sig)
        self._sigs.add(sig)
        if h is not None:
            self._hashes.append(h)
        return None


def default_validators() -> list[IFrameValidator]:
    return [NegativeFrameValidator(), PositiveFrameValidator(), DuplicateFilter()]
