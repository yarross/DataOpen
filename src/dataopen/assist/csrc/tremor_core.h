/* Tremor suppression: fixed-point core mirroring dataopen/assist/tremor_fixed.py operation for operation.
 * One-sided: per axis and per report |out| <= |in| with the same sign or zero; zero input gives zero output; at most trim_cap counts are
 * removed from one report. Knows nothing about objects or the screen: only dx, dy and the person's parameters.
 * Filter states are Q24 (int64), the rest Q16.16. Right shifts of negative values are arithmetic (gcc/clang). */
#ifndef TREMOR_CORE_H
#define TREMOR_CORE_H
#include <stdint.h>

typedef struct {
    int32_t enabled, a_lp, a_band, a_e, s_max, trim_cap, v_t, lp_weight, eps, r_lo, inv_r, big_lo, inv_big, reset_us, reset_ms;
} tremor_params_t; /* a_lp, a_band, a_e are Q24 (they fit in int32: 2^24 = 16.7M); the rest Q16.16 or plain integers */

typedef struct {
    int64_t l1[2], l2[2], b1[2], b2[2], e_band, e_lp, last_t;
    int32_t carry[2], zero_ms, s, r, has_last;
} tremor_state_t;

void tremor_reset(tremor_state_t *st);
int tremor_tick(const tremor_params_t *p, tremor_state_t *st, int64_t t_us, int32_t dx, int32_t dy, int32_t *ox, int32_t *oy);
#endif
