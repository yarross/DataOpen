"""Constants of the model and their personalization from a BioProfile.

`AscConfig` holds the tunable constants (initial values, tuned on the closed-loop simulator, NOT on people). `AscParams` is what the
tick uses: everything derived from the profile once, so the 1 kHz path does no profile lookups. All speeds are counts per millisecond.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from ..bioprofile.profile import ProfileView


@dataclass(frozen=True)
class AscConfig:
    # guard
    v_on_deg_s: float = 12.0             # movement starts when the input speed stays above this (same as BioProfile's onset floor)
    v_still_deg_s: float = 8.0
    on_ms: int = 6
    still_ms: int = 150
    t_min_ms: float = 100.0              # no reaction is faster than this
    t_sigma_k: float = 2.0               # T_lo = T_motor - k * sigma
    # geometry
    ov_zone_gain: float = 0.5            # zone grows with the person's overshoot rate
    r_min_px: float = 8.0
    # strength. Braking slope: from how often / how far the person overshoots and misses. Hold well: from how large their tremor is
    # compared with the object (K_hold = (radius / tremor_hold_div) / tremor amplitude, in pixels).
    px_per_count: float = 1.0            # the pointer's baseline gain at K = 1 (px per HID count): the adapter's / OS's business
    s_lo: float = 0.12
    s_brake_hi: float = 2.0              # K ~ 0.33 at the top of the slope for the most overshoot-prone person
    min_need: float = 0.12               # below this the person needs no braking slope (their hold well may still apply)
    s_cap: float = 9.0
    k_floor: float = 0.1
    w_overshoot: float = 0.4
    w_miss: float = 0.3
    w_ov_med: float = 0.3
    ov_med_ref: float = 0.2
    rate_ref: float = 0.5
    tremor_hold_div: float = 3.0
    fatigue_relief: float = 0.5          # phi: how much fatigue weakens the assistance (0 disables)
    fatigue_z_full: float = 4.0
    # shape: a broad shallow slope (braking) plus a narrow deep well around the object (hold still / tremor)
    deep_radius_mult: float = 3.0
    back_gain: float = 1.5               # zone behind the object reaches back_gain * the person's typical overshoot distance
    # approach / recede
    lambda_speed: float = 0.5
    mu_recede: float = 0.8
    c0: float = 0.2
    c1: float = 0.8
    leave_frac: float = 0.6              # v_leave = leave_frac * V_ref
    away_ms: float = 250.0
    away_ramp_ms: float = 150.0
    max_open_ms: float = 3000.0
    max_open_ramp_ms: float = 500.0
    # smoothing
    lead_base_ms: float = 25.0           # look ahead along the cursor's approach (the smoothing lags by about this) ...
    lead_ov_gain_ms: float = 120.0       # ... further for people who overshoot more: lead = base + gain * typical overshoot
    vp_tau_ms: float = 6.0
    f_attack_hz: float = 14.0
    f_release_hz: float = 7.0
    slew_per_s: float = 6.0
    v_tau_ms: float = 8.0
    min_confidence_n: int = 12
    gap_us: int = 20_000                 # a longer gap between reports resets the velocity estimate
    ramp_min_ms: float = 20.0
    ramp_max_ms: float = 80.0


@dataclass(frozen=True)
class AscParams:
    enabled: bool = False
    v_on: float = 0.0                    # counts/ms
    v_still: float = 0.0
    t_lo_us: int = 0
    ramp_us: int = 0
    f_b: float = 0.5
    ov_rate: float = 0.0
    ov_med: float = 0.0                  # typical overshoot as a fraction of the movement distance
    s_brake: float = 0.0                 # slope strength S at the top of the braking zone (fatigue already applied)
    tremor_counts: float = 0.0           # tremor amplitude in HID counts
    hold_scale: float = 1.0              # 1 - phi * fatigue level, applies to the hold well
    v_ref: float = 1.0                   # counts/ms
    v_leave: float = 1.0
    need: float = 0.0
    fatigue: float = 0.0                 # 0..1
    cfg: AscConfig = field(default_factory=AscConfig)

    @staticmethod
    def disabled(cfg: Optional[AscConfig] = None) -> "AscParams":
        return AscParams(cfg=cfg or AscConfig())

    @staticmethod
    def from_view(view: ProfileView, cfg: Optional[AscConfig] = None, deg_per_count: Optional[float] = None) -> "AscParams":
        """Personalize from a BioProfile. Without enough evidence (or an unknown sensitivity) assistance stays OFF (K = 1.0)."""
        cfg = cfg or AscConfig()
        dpc = deg_per_count if deg_per_count else view.deg_per_count
        need_metrics = ("d_brake", "v_max", "overshoot", "t_motor")
        if not dpc or dpc <= 0 or not all(view.confident(m, cfg.min_confidence_n) for m in need_metrics):
            return AscParams.disabled(cfg)
        t = view.stat("t_motor")
        sig_t = t.sigma if math.isfinite(t.sigma) else 0.0
        t_lo = max(cfg.t_min_ms, t.median - cfg.t_sigma_k * sig_t)
        ramp = min(max(sig_t, cfg.ramp_min_ms), cfg.ramp_max_ms)
        f_b = min(max(view.stat("d_brake").median, 0.15), 0.9)
        v_ref = max(view.stat("v_max").median, 1.0) / dpc / 1000.0
        rates = view.error_rates
        jit = view.stat("jitter_amp")
        tremor_counts = (jit.median if jit.valid else 0.0) / dpc
        ov_med = min(max(view.stat("overshoot").median, 0.0), 1.0)
        need = (cfg.w_overshoot * min(1.0, rates["overshoot"] / cfg.rate_ref) + cfg.w_miss * min(1.0, rates["miss"] / cfg.rate_ref)
                + cfg.w_ov_med * min(1.0, ov_med / cfg.ov_med_ref))
        need = min(max(need, 0.0), 1.0)
        fat = view.fatigue
        level = 0.0
        if fat.valid and math.isfinite(fat.z_t) and math.isfinite(fat.z_err):
            level = min(max(min(fat.z_t, fat.z_err) / cfg.fatigue_z_full, 0.0), 1.0)
        scale = 1.0 - cfg.fatigue_relief * level
        s_brake = (cfg.s_lo + (cfg.s_brake_hi - cfg.s_lo) * need) * scale if need >= cfg.min_need else 0.0
        return AscParams(True, cfg.v_on_deg_s / dpc / 1000.0, cfg.v_still_deg_s / dpc / 1000.0, int(t_lo * 1000), int(ramp * 1000), f_b,
                         rates["overshoot"], ov_med, s_brake, tremor_counts, scale, v_ref, cfg.leave_frac * v_ref, need, level, cfg)
