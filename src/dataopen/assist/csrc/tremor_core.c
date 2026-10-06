#include "tremor_core.h"
#include <string.h>

#define ONE ((int64_t)65536)
#define HALF (ONE >> 1)
#define HALF24 ((int64_t)1 << 23)

static int64_t mulq(int64_t a, int64_t b) { return (a * b + HALF) >> 16; }
static int64_t mulq24(int64_t a, int64_t b) { return (a * b + HALF24) >> 24; }
static int64_t divq(int64_t a, int64_t b) { return (a * 65536) / b; }
static int64_t clampi(int64_t x, int64_t lo, int64_t hi) { return x < lo ? lo : (x > hi ? hi : x); }
static int64_t maxi(int64_t a, int64_t b) { return a > b ? a : b; }
static int64_t absi(int64_t a) { return a < 0 ? -a : a; }
static int64_t sq24(int64_t v) { int64_t w = v >> 8; return ((w * w) >> 16) * 256; }

static int64_t smoothstep(int64_t x) {
    x = clampi(x, 0, ONE);
    int64_t x3 = mulq(mulq(x, x), x);
    int64_t inner = mulq(x, 6 * x - 15 * ONE) + 10 * ONE;
    return mulq(x3, inner);
}

void tremor_reset(tremor_state_t *st) { memset(st, 0, sizeof *st); }

static void forget(tremor_state_t *st) {
    memset(st->l1, 0, sizeof st->l1); memset(st->l2, 0, sizeof st->l2);
    memset(st->b1, 0, sizeof st->b1); memset(st->b2, 0, sizeof st->b2);
    st->e_band = st->e_lp = 0; st->zero_ms = 0; st->s = st->r = 0;
}

static void step(const tremor_params_t *p, tremor_state_t *st, int64_t dx, int64_t dy, int64_t hp[2]) {
    int64_t bsq = 0, lsq = 0;
    int64_t xs[2];
    xs[0] = dx; xs[1] = dy;
    for (int i = 0; i < 2; i++) {
        int64_t x24 = xs[i] * 16777216;
        st->l1[i] += mulq24(p->a_lp, x24 - st->l1[i]);
        st->l2[i] += mulq24(p->a_lp, st->l1[i] - st->l2[i]);
        hp[i] = x24 - st->l2[i];
        st->b1[i] += mulq24(p->a_band, hp[i] - st->b1[i]);
        st->b2[i] += mulq24(p->a_band, st->b1[i] - st->b2[i]);
        bsq += sq24(st->b2[i]);
        lsq += sq24(st->l2[i]);
    }
    st->e_band += mulq24(p->a_e, bsq - st->e_band);
    st->e_lp += mulq24(p->a_e, lsq - st->e_lp);
    int64_t eb = st->e_band >> 8, el = st->e_lp >> 8;
    st->r = (int32_t)divq(eb, eb + mulq(p->lp_weight, el) + p->eps);
    st->s = (int32_t)mulq(p->s_max, smoothstep(mulq(st->r - p->r_lo, p->inv_r)));
}

int tremor_tick(const tremor_params_t *p, tremor_state_t *st, int64_t t_us, int32_t dx, int32_t dy, int32_t *ox, int32_t *oy) {
    int64_t gap = 0, hp[2];
    if (st->has_last) {
        int64_t dt = t_us - st->last_t;
        if (dt > p->reset_us) forget(st);
        else if (dt > 1500) { gap = (dt + 500) / 1000 - 1; if (gap > p->reset_ms) gap = p->reset_ms; }
    }
    st->last_t = t_us; st->has_last = 1;
    *ox = dx; *oy = dy;
    if (!p->enabled) return 0;
    for (int64_t k = 0; k < gap; k++) step(p, st, 0, 0, hp);
    st->zero_ms += (int32_t)gap;
    if (dx == 0 && dy == 0) {
        st->zero_ms += 1;
        step(p, st, 0, 0, hp);
        if (st->zero_ms >= p->reset_ms) forget(st);
        st->carry[0] = st->carry[1] = 0;
        *ox = *oy = 0;
        return 0;
    }
    st->zero_ms = 0;
    step(p, st, dx, dy, hp);
    int64_t big = divq(maxi(absi(dx), absi(dy)) * 65536, p->v_t);
    int64_t s_eff = mulq(st->s, ONE - smoothstep(mulq(big - p->big_lo, p->inv_big)));
    for (int i = 0; i < 2; i++) {
        int64_t x = i == 0 ? dx : dy;
        int32_t *o = i == 0 ? ox : oy;
        if (x == 0) { *o = 0; continue; }
        int64_t xq = x * 65536;
        int64_t y = xq - mulq(s_eff, hp[i] >> 8);
        int64_t lo = x > 0 ? 0 : xq, hi = x > 0 ? xq : 0;
        y = clampi(y, lo, hi);
        y = clampi(y, xq - (int64_t)p->trim_cap * 65536, xq + (int64_t)p->trim_cap * 65536);
        y = clampi(y, lo, hi);
        int64_t v = y + st->carry[i];
        int64_t out = ((absi(v) + HALF) >> 16) * (v >= 0 ? 1 : -1);
        if (absi(out) > absi(x)) out = x;
        else if (out != 0 && (out > 0) != (x > 0)) out = 0;
        st->carry[i] = (int32_t)(v - out * 65536);
        *o = (int32_t)out;
    }
    return 0;
}
