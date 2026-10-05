/* Adaptive Sensitivity Correction: fixed-point (Q16.16) core. Mirrors dataopen/assist/fixed.py operation for operation.
 * Pure integer C99, no libm, no heap, no globals: one asc_state_t per pointing device, one asc_params_t per person (from their
 * BioProfile, computed outside). asc_tick() is called once per HID report and returns the coefficient K (<= 1.0) and the scaled delta.
 * Guarantees: K in [k_floor, 1.0]; K == 1.0 exactly while the guard is closed; |out| <= |raw| with the same sign; raw == 0 => out == 0.
 * Right shifts of negative values are arithmetic (gcc/clang), as in the Python golden model. */
#ifndef ASC_CORE_H
#define ASC_CORE_H
#include <stdint.h>

#define ASC_ONE 65536

enum { ASC_LOCKED = 0, ASC_WAIT = 1, ASC_OPEN = 2 };
enum { ASC_OK = 0, ASC_NO_PROFILE = 1, ASC_NO_OBJECT = 2, ASC_REASON_LOCKED = 3, ASC_STIMULUS_LOCK = 4, ASC_WAIT_OBJECT = 5 };

typedef struct {
    int32_t enabled, v_on, v_still, on_us, still_us, t_lo_us, ramp_us, f_b, ov_rate, ov_med, s_brake, tremor_px, hold_scale, v_ref,
        v_leave, ov_zone_gain, r_min_px, deep_mult, back_gain, hold_div, lam, mu, c0, inv_c, k_floor, s_cap, lead_base, lead_gain,
        away_us, away_ramp_us, open_us, open_ramp_us, w_att, w_rel, slew, v_tau, vp_tau, gap_us;
} asc_params_t;

typedef struct {
    int32_t k, kd, vx, vy, vpx, vpy, pxp, pyp, d0, zone, obj_id, on_us, still_us, away_us, carry_x, carry_y, s;
    int32_t guard, have_p, has_last;
    int64_t last_t, t_move, t_open;
} asc_state_t;

typedef struct {
    int32_t id, x, y, radius;       /* Q16.16 pixels (x, y, radius) */
    int32_t has_appear;
    int64_t t_appear_us;
} asc_obj_t;

typedef struct { int32_t k, dx, dy, guard, reason, s; } asc_out_t;

void asc_reset(asc_state_t *st);
/* obj == NULL: no object of interest this tick. px, py: cursor position (Q16.16 px). Returns 0. */
int asc_tick(const asc_params_t *p, asc_state_t *st, int64_t t_us, int32_t dx, int32_t dy, int32_t px, int32_t py,
             const asc_obj_t *obj, asc_out_t *out);
#endif
