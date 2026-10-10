"""Safe defaults of the scene path (docs/LATENCY.md, "Умолчания v1 и деградация").

The scene path decides how early the help knows its target. The bridge uses a scene until `scene_ttl_ms` (100 ms) after its capture and
then not at all, so a pipeline whose worst case sits near that cliff makes the help come and go. These numbers keep the pipeline well
inside the cliff and say what the module does when it is not: it stops sending scenes (the help is off, predictably) instead of sending
stale ones. Every value here is a decision with a derivation in `latency/budget.py`, pinned by `tests/test_scene_policy.py`.

All of it is SIMULATION: no RK3588, no NPU, no board. The inference times are a documented host measurement (CPU) and a target (NPU)."""

from __future__ import annotations

from dataclasses import dataclass

# inference of the UI network, ms (p50, p95). CPU: a documented host measurement (docs/UIDET.md, ONNX Runtime, 4 threads). NPU: a TARGET,
# nothing was run on an NPU.
INFER_MS = {"cpu": (11.0, 16.0), "npu": (3.0, 4.0)}
# multiply-adds of the reference network (`UiNet`, 640x640) as the device's own static counter (`updates.models`) counts them for the exported
# file. docs/UIDET.md quotes "about 1.0 GMAC" for the same network: that count was taken another way, so the CALIBRATION below uses the counter
# that is applied to every model on the device, and the two figures are not mixed.
REF_MACS = 2_022_092_800
ACTIVE_BACKEND = "cpu"            # the backend this build runs the detector on (becomes "npu" when the RKNN path exists)
INFER_BUDGET_MS = 19.0            # largest p95 inference the budget allows (derived: latency.budget.max_infer_p95_ms, rounded down)


@dataclass(frozen=True)
class ScenePolicy:
    latest_only: bool = True               # take the newest frame, let go of the older ones (no FIFO of 4)
    max_age_ms: float = 35.0               # a frame older than this when it is TAKEN is dropped unseen
    confirm_hits: int = 2                  # frames a target must be seen in (kills one-frame phantoms); see docs/LATENCY.md
    publish_max_age_ms: float = 70.0       # a scene older than this when it is READY is late: it is not sent
    recover_max_age_ms: float = 50.0       # to switch back on the scenes have to be this fresh (hysteresis)
    late_trip: int = 3                     # this many late scenes in a row switch the scene help off
    recover_n: int = 15                    # this many fresh scenes in a row (and `hold_ms`) switch it back on
    hold_ms: int = 2000                    # stays off at least this long
    infer_window: int = 50                 # detections the inference p95 is taken over
    infer_budget_ms: float = INFER_BUDGET_MS


V1 = ScenePolicy()


def est_infer_ms(macs: int, backend: str = ACTIVE_BACKEND) -> tuple[float, float]:
    """(p50, p95) inference estimate for a model of `macs` multiply-adds: the reference network's time scaled by the work. A ROUGH linear
    estimate from ONE network (memory, operator mix and the quantised NPU path are not in it); the runtime guard in `SceneHealth` is what
    catches an estimate that is wrong."""
    p50, p95 = INFER_MS[backend]
    f = max(macs, 0) / REF_MACS
    return p50 * f, p95 * f


def fits_budget(macs: int, backend: str = ACTIVE_BACKEND) -> bool:
    return est_infer_ms(macs, backend)[1] <= V1.infer_budget_ms
