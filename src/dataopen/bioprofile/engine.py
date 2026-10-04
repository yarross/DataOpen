"""The BioProfile engine: an online state machine over a 1 ms grid of mouse motion plus target observations.

Inputs (both optional in any order, timestamps non-decreasing, microseconds):
    on_mouse(t_us, dx, dy)      raw relative counts (event driven: silence = the hand is still)
    on_target(t_us, obs|None)   a detector frame: the target as angles from the crosshair, or None when nothing was detected
                                (the None frames are what advance time while no target is visible)
Output: `snapshot()` -> ProfileState (rolling median + sigma of the key metrics, error rates, fatigue), `episodes` (a log), and
`subscribe(fn)` callbacks fired after each completed episode.

State machine (details and thresholds in docs/BIOPROFILE.md):

    IDLE --target appears, hand still--> ARMED --first movement--> MOVING --speed dies--> SETTLE --still / timeout--> (TRACK | IDLE)
      |                                    '--no movement for t_max--> lapse                  '--new movement--> MOVING (a correction)
      '--target appears, hand already moving, target far--> CENSORED (no reaction time can be measured; nothing is learned)
      '--target near and hand following it--------------------------------------------> TRACK --target gone / window full--> IDLE

Scenario labels per episode: WIDE_FLICK (distance >= d_flick and a fast movement), MICRO_TRACK (tracking window), with the modifiers
SURPRISE (the hand was still when the target appeared) and LOW_VIS. Error scenarios: overshoot, undershoot, wrong initial direction,
miss (settled outside the target), lapse (no reaction), plus anticipation (a "reaction" faster than a human can react).
The engine never produces any input; it only measures.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from .fatigue import DriftTracker
from .profile import COUNTS, RATES, STATS, ProfileState, Stat
from .rolling import RollingMedianSigma
from .types import Episode, Scenario, State, TargetObs


@dataclass
class EngineConfig:
    deg_per_count: float = 0.02          # mouse counts -> degrees (from the player's sensitivity; REQUIRED to be right)
    latency_comp_us: int = 2083          # added to T_motor: half a 240 Hz frame (the target appears between two frames)
    window: int = 32                     # rolling window for every metric
    min_n: int = 8                       # values before a rolling statistic counts as ready
    # movement / stillness
    v_still: float = 8.0                 # deg/s: below this the hand counts as still
    v_on: float = 12.0                   # deg/s: absolute floor of the onset threshold
    onset_k: float = 5.0                 # onset threshold = max(v_on, baseline mean + onset_k * baseline sigma)
    onset_bins: int = 6                  # ms the speed must stay above the threshold
    v_settle: float = 10.0               # deg/s: below max(v_settle, 0.12 * peak) the first movement is over
    v_corr: float = 4.0                  # deg/s: a corrective submovement is this slow: its own start / end thresholds
    settle_ms: int = 25
    correction_wait_ms: int = 350        # after a movement ends, how long a corrective movement may still start
    still_final_ms: int = 160            # a pause between two submovements of one flick is up to ~150 ms
    max_corrections: int = 1             # corrective submovements attributed to the flick; later movement is tracking
    min_correction_deg: float = 0.75     # a new movement while still this far from the target is a correction, else tracking begins
    # reaction
    t_min_ms: float = 100.0              # faster than this is not a reaction to the target (anticipation)
    t_max_ms: float = 1200.0             # no movement within this: lapse
    # geometry
    d_flick: float = 10.0                # deg: distances from here are wide flicks
    v_flick_min: float = 120.0           # deg/s: and the peak must be at least this fast
    r_track: float = 5.0                 # deg: closer than this the player is tracking, not flicking
    absent_ms: int = 120                 # target not seen for this long = gone (a missed detector frame is not "gone")
    # tracking
    track_enter_ms: int = 150
    track_min_ms: int = 400
    track_max_ms: int = 2000
    v_track_min: float = 12.0            # deg/s: the hand must be moving to count as following
    jitter_band_hz: tuple = (3.0, 16.0)  # micro-corrections / physiological tremor live here
    # visibility and error thresholds
    lowvis_thr: float = 0.5
    overshoot_err: float = 0.15
    undershoot_err: float = 0.20
    direction_err_deg: float = 35.0
    rate_alpha: float = 0.04             # EWMA weight of one episode in the error rates
    # session
    block_s: float = 120.0
    rest_gap_s: float = 300.0
    profile_id: int = 0


@dataclass
class _Ep:
    t0_ms: int
    lowvis: bool
    surprise: bool
    d0: float = 0.0
    e0: np.ndarray = field(default_factory=lambda: np.zeros(2))
    radius: float = 0.5
    base_mu: float = 0.0
    base_sd: float = 0.0
    thr: float = 0.0
    onset_ms: int = 0
    view_onset: np.ndarray = field(default_factory=lambda: np.zeros(2))
    peak: float = 0.0
    peak_ms: int = 0
    main_end_ms: Optional[int] = None
    still_since: Optional[int] = None
    n_corr: int = 0
    err_pause: Optional[float] = None
    out: Episode = field(default_factory=lambda: Episode("flick"))
    start_hist: int = 0


class BioProfileEngine:
    def __init__(self, cfg: Optional[EngineConfig] = None, profile: Optional[ProfileState] = None) -> None:
        self.cfg = cfg or EngineConfig()
        c = self.cfg
        self.stats = {s.name: RollingMedianSigma(c.window, c.min_n, sigma_floor=0.0) for s in STATS}
        self.counts = {k: 0 for k in COUNTS}
        self.rates = {k: 0.0 for k in RATES}
        self.drift = DriftTracker(c.block_s)
        self.episodes: list[Episode] = []
        self._subs: list[Callable[[ProfileState], None]] = []
        self.state = State.IDLE
        self.profile_id = c.profile_id
        self.generation = 0
        # time grid
        self._bin: Optional[int] = None
        self._dx = self._dy = 0.0
        self._view = np.zeros(2)
        self._hist: deque = deque(maxlen=4000)       # (ms, view_x, view_y, speed_smoothed, speed_raw)
        self._last_ms: Optional[int] = None
        self._sess_t0_ms: Optional[int] = None
        self._last_activity_ms: Optional[int] = None
        self._d5: deque = deque(maxlen=5)
        # target
        self._tgt: Optional[np.ndarray] = None       # target in world angles (e_obs + view at the observation)
        self._tgt_obs: Optional[TargetObs] = None
        self._last_seen_ms: Optional[int] = None
        self._present = False
        self._tgt_hist: list[tuple[int, float, float]] = []
        self._ep: Optional[_Ep] = None
        self._track_t0: Optional[int] = None
        self._track_ep: Optional[_Ep] = None
        self._track_enter: Optional[int] = None
        self.dropped_samples = 0
        self._armed_run = 0
        self._corr_run = 0
        self._seed_n: dict[str, int] = {}               # sample counts carried over from a stored profile
        if profile is not None:
            self._seed(profile)

    # ------------------------------------------------------------------ public
    def subscribe(self, fn: Callable[[ProfileState], None]) -> None:
        self._subs.append(fn)

    def on_mouse(self, t_us: int, dx: float, dy: float) -> None:
        if not (math.isfinite(dx) and math.isfinite(dy)):
            self.dropped_samples += 1
            return
        ms = self._clock(t_us)
        if ms is None:
            return
        self._advance_to(ms)
        k = self.cfg.deg_per_count
        self._dx += dx * k
        self._dy += dy * k

    def on_target(self, t_us: int, obs: Optional[TargetObs]) -> None:
        ms = self._clock(t_us)
        if ms is None:
            return
        self._advance_to(ms)
        if obs is None:
            return
        if not (math.isfinite(obs.x_deg) and math.isfinite(obs.y_deg)):
            self.dropped_samples += 1
            return
        view_now = self._view + np.array([self._dx, self._dy])
        world = np.array([obs.x_deg, obs.y_deg]) + view_now
        appeared = not self._present
        self._tgt, self._tgt_obs, self._last_seen_ms, self._present = world, obs, ms, True
        if self._ep is not None or self._track_ep is not None or self.state != State.IDLE:
            self._tgt_hist.append((ms, float(world[0]), float(world[1])))
        if appeared:
            self._on_appear(ms, obs, np.array([obs.x_deg, obs.y_deg]))

    def advance(self, t_us: int) -> None:
        ms = self._clock(t_us)
        if ms is not None:
            self._advance_to(ms)

    def snapshot(self, clean: bool = False) -> ProfileState:
        c = self.cfg
        st = ProfileState(profile_id=self.profile_id, generation=self.generation, deg_per_count=c.deg_per_count,
                          latency_comp_us=c.latency_comp_us, clean=clean)
        for s in STATS:
            r = self.stats[s.name]
            if len(r):
                med, sig = r.stats()
                st.stats[s.name] = Stat(med, sig, min(r.n_total + self._seed_n.get(s.name, 0), 255))
        st.counts = dict(self.counts)
        st.rates = dict(self.rates)
        st.fatigue = self.drift.result()
        return st

    # ------------------------------------------------------------------ seeding
    def _seed(self, p: ProfileState) -> None:
        self.profile_id, self.generation = p.profile_id, p.generation
        for name, s in p.stats.items():
            if s.valid and name in self.stats:
                self.stats[name].seed(s.median, s.sigma if math.isfinite(s.sigma) else 0.0, k=min(max(s.n, 3), 8))
                self._seed_n[name] = s.n
        self.counts = dict(p.counts)
        self.rates = dict(p.rates)

    # ------------------------------------------------------------------ time grid
    def _clock(self, t_us: int) -> Optional[int]:
        try:
            ms = int(t_us) // 1000
        except (TypeError, ValueError, OverflowError):
            self.dropped_samples += 1
            return None
        if self._last_ms is not None and ms < self._last_ms - 5:      # out of order by more than a few ms: ignore
            self.dropped_samples += 1
            return None
        ms = max(ms, self._last_ms or ms)
        self._last_ms = ms
        return ms

    def _advance_to(self, ms: int) -> None:
        if self._bin is None:
            self._bin = ms
            self._sess_t0_ms = ms
            return
        gap = ms - self._bin
        if gap <= 0:
            return
        if gap > 2000:                                  # a long silence: the hand was resting; do not iterate millions of ms
            rest_s = gap / 1000.0
            self._abort_episode()
            self._hist.clear()
            self._d5.clear()
            self._view = self._view + np.array([self._dx, self._dy])
            self._dx = self._dy = 0.0
            self._bin = ms
            self._present = False
            if rest_s >= self.cfg.rest_gap_s:
                self._new_session(ms)
            return
        while self._bin < ms:
            self._finalize_bin(self._bin)
            self._bin += 1

    def _new_session(self, ms: int) -> None:
        self.drift.reset()
        self._sess_t0_ms = ms

    def _finalize_bin(self, ms: int) -> None:
        dx, dy, self._dx, self._dy = self._dx, self._dy, 0.0, 0.0
        self._view = self._view + np.array([dx, dy])
        self._d5.append((dx, dy))
        sx, sy = sum(d[0] for d in self._d5), sum(d[1] for d in self._d5)
        sp = math.hypot(sx, sy) / (len(self._d5) * 0.001)
        raw = math.hypot(dx, dy) * 1000.0
        self._hist.append((ms, float(self._view[0]), float(self._view[1]), sp, raw))
        if raw > 0:
            self._last_activity_ms = ms
        if self._present and self._last_seen_ms is not None and ms - self._last_seen_ms >= self.cfg.absent_ms:
            self._present = False
            self._on_gone(ms)
        self._step(ms, sp)

    # ------------------------------------------------------------------ helpers
    def _e(self) -> np.ndarray:
        """Dead-reckoned error vector: the last seen target minus where the view is now."""
        return (self._tgt - self._view) if self._tgt is not None else np.zeros(2)

    def _speeds(self, ms_from: int, ms_to: int) -> np.ndarray:
        return np.array([h[3] for h in self._hist if ms_from <= h[0] < ms_to])

    # ------------------------------------------------------------------ appearance
    def _on_appear(self, ms: int, obs: TargetObs, e0: np.ndarray) -> None:
        c = self.cfg
        if self._ep is not None or self.state in (State.MOVING, State.SETTLE):
            return
        if self.state == State.TRACK:
            return
        base = self._speeds(ms - 200, ms)
        mu = float(base.mean()) if len(base) else 0.0
        sd = float(base.std()) if len(base) else 0.0
        pre_moving = mu > c.v_still or (len(base) and float(base.max()) > 3 * c.v_still)
        d0 = float(np.hypot(*e0))
        lowvis = bool(obs.degraded or obs.vis < c.lowvis_thr)
        ep = _Ep(ms, lowvis, surprise=not pre_moving, d0=d0, e0=e0.copy(), radius=obs.radius_deg, base_mu=mu, base_sd=sd)
        ep.out = Episode("flick", t_appear_us=ms * 1000, surprise=ep.surprise, lowvis=lowvis, d0_deg=d0)
        if pre_moving and d0 < c.r_track:
            self._begin_track(ms, ep)
        elif pre_moving:
            ep.out.kind = "censored"
            self._ep = ep
            self.state = State.IDLE                     # nothing to learn; waits for the target to go
            self._finish(ep.out)
            self._ep = ep
            ep.still_since = -1                         # marker: censored, waiting for the target to disappear
        else:
            ep.thr = max(c.v_on, mu + c.onset_k * sd)
            self._ep, self.state = ep, State.ARMED
            self._armed_run = 0

    def _on_gone(self, ms: int) -> None:
        ep = self._ep
        if self.state == State.TRACK:
            self._finish_track(ms)
        elif ep is not None and ep.still_since == -1:    # censored episode over
            self._ep = None
        elif ep is not None and self.state == State.ARMED:
            self._ep, self.state = None, State.IDLE
        elif ep is not None and self.state in (State.MOVING, State.SETTLE):
            self._end_movement(ms, gone=True)
        self._tgt_hist = []

    def _abort_episode(self) -> None:
        self._ep, self._track_ep, self.state = None, None, State.IDLE
        self._tgt_hist = []

    # ------------------------------------------------------------------ the state machine
    def _step(self, ms: int, sp: float) -> None:
        c, ep = self.cfg, self._ep
        if self.state == State.IDLE:
            self._maybe_enter_track(ms, sp)
            return
        if self.state == State.ARMED and ep is not None:
            self._armed_run = self._armed_run + 1 if sp > ep.thr else 0
            if self._armed_run >= c.onset_bins:
                self._onset(ms, ep)
            elif ms - ep.t0_ms > c.t_max_ms:
                self._lapse(ms, ep)
            return
        if self.state == State.MOVING and ep is not None:
            if sp > ep.peak:
                ep.peak, ep.peak_ms = sp, ms
            lim = max(c.v_settle if ep.n_corr == 0 else c.v_corr, 0.12 * ep.peak)
            if sp < lim:
                if ep.still_since is None:
                    ep.still_since = ms
                if ms - ep.still_since >= c.settle_ms:
                    self._movement_paused(ms, ep)
            else:
                ep.still_since = None
            if ms - ep.onset_ms > 1500:
                self._movement_paused(ms, ep)
            return
        if self.state == State.SETTLE and ep is not None:
            thr = max(c.v_corr, ep.base_mu + 3.0 * ep.base_sd)
            self._corr_run = self._corr_run + 1 if sp > thr else 0
            if self._corr_run >= 5:                           # a (slow) new movement: a correction, or tracking beginning
                self._corr_run = 0
                if self._tgt is not None and ep.n_corr < c.max_corrections \
                        and float(np.hypot(*self._e())) >= min(c.min_correction_deg, ep.radius):
                    ep.n_corr += 1
                    ep.still_since, ep.peak = None, sp
                    self.state = State.MOVING
                else:
                    self._end_movement(ms)                    # already on target: this movement is the start of tracking
                return
            if ep.still_since is None:
                ep.still_since = ms
            waited = ms - (ep.main_end_ms or ms)
            if (ms - ep.still_since >= c.still_final_ms and sp < c.v_settle) or waited >= c.correction_wait_ms:
                self._end_movement(ms)
            return
        if self.state == State.TRACK:
            if self._track_t0 is not None and ms - self._track_t0 >= c.track_max_ms:
                self._finish_track(ms)

    # ---- reaction onset
    def _onset(self, ms: int, ep: _Ep) -> None:
        c = self.cfg
        low = max(3.0, ep.base_mu + 2.0 * ep.base_sd)
        onset = ms
        for h in reversed(self._hist):                   # walk back to where the speed first left the noise floor
            if h[3] <= low:
                break
            onset = h[0]
        onset = self._refine_onset(onset, ms)
        ep.onset_ms = onset
        i = next((k for k, h in enumerate(self._hist) if h[0] == onset), len(self._hist) - 1)
        ep.view_onset = np.array([self._hist[i][1], self._hist[i][2]])
        ep.start_hist = onset
        e_on = self._e() + (self._view - ep.view_onset) * 0 + (self._view - ep.view_onset)   # error as it was at the onset
        ep.d0 = float(np.hypot(*e_on)) if np.hypot(*e_on) > 0 else ep.d0
        ep.e0 = e_on
        t_motor = (onset - ep.t0_ms) + c.latency_comp_us / 1000.0      # the target appeared, on average, half a frame before it was seen
        ep.out.t_motor_ms = t_motor
        if t_motor < c.t_min_ms:
            ep.out.anticipation = True
            ep.out.t_motor_ms = None
        ep.peak, ep.peak_ms, ep.still_since = 0.0, ms, None
        self.state = State.MOVING

    def _refine_onset(self, onset: int, detected: int) -> int:
        """A threshold crossing is late by the time the speed needs to climb to the threshold. Speed rises about quadratically from a
        movement's start, so sqrt(speed) is linear in time: extrapolating that line back to zero gives the start (the 5 ms smoothing
        delays each speed sample by 2 ms, compensated). Falls back to the plain crossing when the ramp is too short or not rising."""
        pts = [(h[0] - 2.0, math.sqrt(h[3])) for h in self._hist if onset <= h[0] <= detected]
        if len(pts) < 4:
            return onset
        t = np.array([p[0] for p in pts])
        y = np.array([p[1] for p in pts])
        slope, icpt = np.polyfit(t, y, 1)
        if slope <= 0:
            return onset
        t0 = -icpt / slope
        return int(round(min(max(t0, onset - 12.0), float(onset))))

    def _lapse(self, ms: int, ep: _Ep) -> None:
        ep.out.kind = "miss"
        ep.out.scenarios = [Scenario.SURPRISE.value] + ([Scenario.LOW_VIS.value] if ep.lowvis else [])
        self._rate("lapse", 1.0)
        self.counts["surprise"] = min(self.counts["surprise"] + 1, 65535)
        self._ep, self.state = None, State.IDLE
        self._finish(ep.out)

    # ---- main movement metrics
    def _movement_paused(self, ms: int, ep: _Ep) -> None:
        c = self.cfg
        if ep.main_end_ms is None:                        # first pause: the ballistic phase is over, measure it
            ep.main_end_ms = ms - c.settle_ms
            seg = [h for h in self._hist if ep.onset_ms <= h[0] <= ep.main_end_ms]
            if len(seg) >= 5:
                self._measure_main(ep, seg)
        if self._tgt is not None:
            ep.err_pause = float(np.hypot(*self._e()))      # where the movement stopped, before the target moves on
        ep.still_since, self.state = ms - c.settle_ms, State.SETTLE

    def _measure_main(self, ep: _Ep, seg: list) -> None:
        c = self.cfg
        pos = np.array([[h[1], h[2]] for h in seg]) - ep.view_onset[None]
        sp = np.array([h[3] for h in seg])
        d0 = ep.d0
        if d0 < 1e-6:
            return
        u = ep.e0 / d0
        proj = pos @ u
        a1 = float(proj[-1])
        k = int(np.argmax(sp))
        ep.peak = float(sp[k])
        ep.out.d0_deg = d0
        if d0 >= c.d_flick and ep.peak >= c.v_flick_min:
            ep.out.v_max = ep.peak
            remaining = float(np.hypot(*(ep.e0 - pos[k])))
            ep.out.d_brake_frac = min(remaining / d0, 1.5)
            ep.out.overshoot = max(0.0, (float(proj.max()) - d0) / d0)
            ep.out.scenarios = [Scenario.WIDE_FLICK.value]
            moved = np.hypot(pos[:, 0], pos[:, 1])
            j = int(np.argmax(moved >= max(1.0, 0.1 * d0))) if (moved >= max(1.0, 0.1 * d0)).any() else None
            if j is not None:
                a = math.atan2(pos[j, 1], pos[j, 0]) - math.atan2(u[1], u[0])
                ep.out.heading_dev_deg = abs(math.degrees(math.atan2(math.sin(a), math.cos(a))))
            ep.out.undershoot = a1 < (1.0 - c.undershoot_err) * d0
            ep.out.main_known = True

    def _end_movement(self, ms: int, gone: bool = False) -> None:
        c, ep = self.cfg, self._ep
        if ep is None:
            self.state = State.IDLE
            return
        o = ep.out
        o.n_corrections = ep.n_corr
        if ep.err_pause is not None:
            o.err_final_deg = ep.err_pause
        elif self._tgt is not None:
            o.err_final_deg = float(np.hypot(*self._e()))
        if ep.surprise:
            sc = [Scenario.SURPRISE.value] + ([Scenario.LOW_VIS.value] if ep.lowvis else [])
            o.scenarios = sc + [s for s in o.scenarios if s not in sc]
        self._ep = None
        near = (not gone) and self._present and self._tgt is not None and float(np.hypot(*self._e())) < c.r_track
        self.state = State.IDLE
        self._learn_flick(o, ep)
        self._finish(o)
        if near:
            self._begin_track(ms, _Ep(ms, ep.lowvis, False, radius=ep.radius))

    def _learn_flick(self, o: Episode, ep: _Ep) -> None:
        c = self.cfg
        known = o.main_known
        if ep.surprise:
            self.counts["surprise"] = min(self.counts["surprise"] + 1, 65535)
            if ep.lowvis:
                self.counts["lowvis"] = min(self.counts["lowvis"] + 1, 65535)
            if o.anticipation:
                self.counts["anticipation"] = min(self.counts["anticipation"] + 1, 65535)
            elif o.t_motor_ms is not None:
                self.stats["t_motor_lowvis" if ep.lowvis else "t_motor"].add(o.t_motor_ms)
            self._rate("lapse", 0.0)
        if o.err_final_deg is not None:
            self.stats["err_final"].add(o.err_final_deg)
            self._rate("miss", 1.0 if o.err_final_deg > max(ep.radius, 0.1) else 0.0)
        if known:
            self.counts["flick"] = min(self.counts["flick"] + 1, 65535)
            self.stats["v_max"].add(o.v_max)
            self.stats["d_brake"].add(o.d_brake_frac)
            self.stats["overshoot"].add(o.overshoot)
            self._rate("overshoot", 1.0 if o.overshoot > c.overshoot_err else 0.0)
            self._rate("undershoot", 1.0 if o.undershoot else 0.0)
            if o.heading_dev_deg is not None:
                self._rate("direction", 1.0 if o.heading_dev_deg > c.direction_err_deg else 0.0)
        t_s = ((ep.t0_ms - (self._sess_t0_ms or ep.t0_ms)) / 1000.0)
        self.drift.add(t_s, o.t_motor_ms if (ep.surprise and not ep.lowvis) else None, o.err_final_deg if known else None)

    def _rate(self, name: str, hit: float) -> None:
        a = self.cfg.rate_alpha
        self.rates[name] += a * (hit - self.rates[name])

    # ---- tracking
    def _maybe_enter_track(self, ms: int, sp: float) -> None:
        c = self.cfg
        if not (self._present and self._tgt is not None and float(np.hypot(*self._e())) < c.r_track and sp > c.v_track_min):
            self._track_enter = None
            return
        if self._track_enter is None:
            self._track_enter = ms
        if ms - self._track_enter >= c.track_enter_ms and self._ep is None:
            obs = self._tgt_obs
            lowvis = bool(obs and (obs.degraded or obs.vis < c.lowvis_thr))
            self._begin_track(ms - c.track_enter_ms, _Ep(ms, lowvis, False, radius=obs.radius_deg if obs else 0.5))

    def _begin_track(self, ms: int, ep: _Ep) -> None:
        self.state, self._track_t0, self._track_ep, self._track_enter = State.TRACK, ms, ep, None
        self._tgt_hist = [(m, x, y) for m, x, y in self._tgt_hist if m >= ms - 300]
        if self._tgt is not None and not self._tgt_hist:
            self._tgt_hist = [(ms, float(self._tgt[0]), float(self._tgt[1]))]

    def _finish_track(self, ms: int) -> None:
        c, ep, t0 = self.cfg, self._track_ep, self._track_t0
        self.state, self._track_ep, self._track_t0 = State.IDLE, None, None
        hist, self._tgt_hist = self._tgt_hist, []
        if ep is None or t0 is None or ms - t0 < c.track_min_ms:
            return
        o = Episode("track", t_appear_us=t0 * 1000, lowvis=ep.lowvis, scenarios=[Scenario.MICRO_TRACK.value])
        seg = [h for h in self._hist if t0 <= h[0] <= ms]
        if len(seg) < c.track_min_ms * 0.8 or len(hist) < 10:
            return
        res = measure_tracking(np.array([[h[0], h[1], h[2]] for h in seg]), np.array(hist, dtype=float), c)
        o.phase_lag_ms, o.jitter_hz, o.jitter_amp_deg = res.get("lag"), res.get("hz"), res.get("amp")
        if ep.lowvis:
            o.scenarios.append(Scenario.LOW_VIS.value)
        self.counts["track"] = min(self.counts["track"] + 1, 65535)
        if o.phase_lag_ms is not None:
            self.stats["phase_lag"].add(o.phase_lag_ms)
        if o.jitter_hz is not None and o.jitter_amp_deg is not None:
            self.stats["jitter_hz"].add(o.jitter_hz)
            self.stats["jitter_amp"].add(o.jitter_amp_deg)
        self._finish(o)

    # ---- completion
    def _finish(self, o: Episode) -> None:
        self.episodes.append(o)
        if len(self.episodes) > 5000:
            del self.episodes[:1000]
        if self._subs:
            snap = self.snapshot()
            for fn in self._subs:
                fn(snap)


def measure_tracking(view: np.ndarray, tgt: np.ndarray, c: EngineConfig) -> dict:
    """view: (N, 3) [ms, x, y] on a 1 ms grid; tgt: (M, 3) target world angles at observation times (noisy, irregular).
    Returns phase lag (ms), jitter frequency (Hz) and amplitude (deg); entries are missing when they cannot be measured honestly
    (a static target has no phase lag)."""
    t = view[:, 0]
    out: dict = {}
    tt = tgt[:, 0]
    keep = np.concatenate([[True], np.diff(tt) > 0])
    tt, tx, ty = tt[keep], tgt[keep, 1], tgt[keep, 2]
    if len(tt) < 10:
        return out
    step = 5
    grid = np.arange(max(t[0], tt[0]), min(t[-1], tt[-1]), step, dtype=float)
    if len(grid) < 40:
        return out
    k = 5                                              # smooth the noisy observations over ~5 samples before differentiating
    ker = np.ones(k) / k
    sx = np.convolve(np.pad(tx, k // 2, mode="edge"), ker, "valid")        # edge-padded: zero padding would fake a spike at both ends
    sy = np.convolve(np.pad(ty, k // 2, mode="edge"), ker, "valid")
    gx, gy = np.interp(grid, tt, sx), np.interp(grid, tt, sy)
    vx, vy = np.interp(grid, t, view[:, 1]), np.interp(grid, t, view[:, 2])
    h = 5                                              # velocity = central difference over 2*h grid steps (50 ms)
    tv = np.stack([gx[2 * h:] - gx[:-2 * h], gy[2 * h:] - gy[:-2 * h]], axis=1) / (2 * h * step / 1000.0)
    mv = np.stack([vx[2 * h:] - vx[:-2 * h], vy[2 * h:] - vy[:-2 * h]], axis=1) / (2 * h * step / 1000.0)
    lag_ms = 0.0
    if float(np.sqrt((tv ** 2).sum(axis=1).mean())) >= 4.0:           # the target must actually move
        scores: list[float] = []
        max_lag = min(60, len(tv) // 3)
        for L in range(0, max_lag + 1):
            a, b = tv[: len(tv) - L], mv[L:]
            den = math.sqrt((a ** 2).sum() * (b ** 2).sum()) + 1e-12
            scores.append(float((a * b).sum()) / den)
        L = int(np.argmax(scores))
        if scores[L] >= 0.5:
            frac = 0.0
            if 0 < L < len(scores) - 1:
                d = scores[L - 1] - 2 * scores[L] + scores[L + 1]
                if d < 0:
                    frac = 0.5 * (scores[L - 1] - scores[L + 1]) / d
            lag_ms = (L + frac) * step
            out["lag"] = lag_ms
    # jitter: the mouse path minus the (lag-shifted) target path, then the spectrum of what is left in the 3-16 Hz band
    n = len(t)
    if n < 300:
        return out
    rx = view[:, 1] - np.interp(t - lag_ms, tt, sx)
    ry = view[:, 2] - np.interp(t - lag_ms, tt, sy)
    x = np.arange(n, dtype=float)
    res = []
    for r in (rx, ry):
        a, b = np.polyfit(x, r, 2)[:2] if n > 3 else (0.0, 0.0)
        res.append(r - np.polyval(np.polyfit(x, r, 2), x))        # remove the offset and slow drift of the residual
    w = np.hanning(n)
    norm = 2.0 / (n * float((w ** 2).sum()))
    freqs = np.fft.rfftfreq(n, d=0.001)
    pw = sum(np.abs(np.fft.rfft(r * w)) ** 2 for r in res) * norm
    lo, hi = c.jitter_band_hz
    band = (freqs >= lo) & (freqs <= hi)
    tot = float(pw[band].sum())
    var_total = float(pw[1:].sum())
    if tot <= 0 or var_total <= 0:
        return out
    out["hz"] = float((freqs[band] * pw[band]).sum() / tot)       # power-weighted centre of the band
    out["amp"] = math.sqrt(2.0 * tot)                              # peak amplitude of a sinusoid with that power
    return out
