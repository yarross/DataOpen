"""How far the profile has got: a plain-language number for the phone and the three answers behind it.

There is no 'convergence' in the engine: it learns passively and every statistic carries a sample count `n`. Progress here is exactly that
and nothing more: the mean over the statistics the assistance layers need of min(1, n / N_TARGET). It says "enough observations have been
seen", NOT that the numbers are statistically settled, and it is not validated on people (docs/PWA.md section 0).
"""
from __future__ import annotations

from dataclasses import dataclass

from ..assist.params import AscParams
from ..assist.tremor import TremorParams
from .profile import ProfileView

N_TARGET = 30
ASC_STATS = ("d_brake", "v_max", "overshoot", "t_motor")
TREMOR_STATS = ("jitter_amp", "jitter_hz")


@dataclass(frozen=True)
class Progress:
    fill: int                   # 0..100
    asc_ready: bool             # the movement-assistance layer would be ON with this profile
    tremor: str                 # 'collecting' | 'not_needed' (enough data, a tremor too small to filter) | 'ready'
    per_stat: dict

    @property
    def tremor_ready(self) -> bool:
        return self.tremor == "ready"


def profile_progress(view: ProfileView | None) -> Progress:
    if view is None:
        return Progress(0, False, "collecting", {})
    names = ASC_STATS + TREMOR_STATS
    per = {n: view.stat(n).n for n in names}
    fill = sum(min(1.0, per[n] / N_TARGET) for n in names) / len(names)
    asc_ready = AscParams.from_view(view).enabled
    if TremorParams.from_view(view).enabled:
        tremor = "ready"
    elif all(view.confident(n) for n in TREMOR_STATS) and view.deg_per_count > 0:
        tremor = "not_needed"
    else:
        tremor = "collecting"
    return Progress(int(round(100 * fill)), asc_ready, tremor, per)
