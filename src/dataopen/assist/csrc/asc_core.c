#include "asc_core.h"
#include <string.h>

#define ONE ((int64_t)ASC_ONE)
#define HALF (ONE >> 1)
#define COORD_MAX (((int64_t)1 << 30) - 1)

static int64_t mulq(int64_t a, int64_t b) { return (a * b + HALF) >> 16; }
static int64_t sdiv(int64_t a, int64_t b) { return a / b; } /* C99: truncation toward zero */
static int64_t divq(int64_t a, int64_t b) { return (a * 65536) / b; }
static int64_t clampi(int64_t x, int64_t lo, int64_t hi) { return x < lo ? lo : (x > hi ? hi : x); }
static int64_t maxi(int64_t a, int64_t b) { return a > b ? a : b; }
static int64_t mini(int64_t a, int64_t b) { return a < b ? a : b; }
static int64_t absi(int64_t a) { return a < 0 ? -a : a; }

static int64_t isqrt64(int64_t n) {
    if (n <= 0) return 0;
    uint64_t x = (uint64_t)n, r = 0, bit = (uint64_t)1 << 62;
    while (bit > x) bit >>= 2;
    while (bit) {
        if (x >= r + bit) { x -= r + bit; r = (r >> 1) + bit; } else { r >>= 1; }
        bit >>= 2;
    }
    return (int64_t)r;
}

static int64_t smoothstep(int64_t x) {
    x = clampi(x, 0, ONE);
    int64_t x3 = mulq(mulq(x, x), x);
    int64_t inner = mulq(x, 6 * x - 15 * ONE) + 10 * ONE;
    return mulq(x3, inner);
}

static int64_t scale_us(int64_t x, int64_t dt_us) {
    int64_t n = x * dt_us;
    return n >= 0 ? (n + 500000) / 1000000 : -((-n + 500000) / 1000000);
}

void asc_reset(asc_state_t *st) {
    memset(st, 0, sizeof *st);
    st->k = ASC_ONE;
    st->d0 = -1;
    st->obj_id = -1;
    st->guard = ASC_LOCKED;
}

static void lock(asc_state_t *st) {
    st->guard = ASC_LOCKED;
    st->d0 = -1; st->zone = 0; st->obj_id = -1; st->away_us = 0; st->on_us = 0;
    st->carry_x = st->carry_y = 0;
}

static int64_t s_hold(const asc_params_t *p, int64_t radius) {
    if (p->tremor_px < 66) return 0;
    int64_t r_floor = divq(p->r_min_px, p->deep_mult);
    int64_t k_hold = clampi(divq(maxi(radius, r_floor), mulq(p->hold_div, p->tremor_px)), p->k_floor, ONE);
    return mulq(divq(ONE, k_hold) - ONE, p->hold_scale);
}

static int64_t resistance(const asc_params_t *p, asc_state_t *st, int64_t t_us, int64_t dt_us, int64_t pxq, int64_t pyq,
                          const asc_obj_t *obj, int64_t speed) {
    int64_t rq = obj->radius;
    int64_t dxq = clampi((int64_t)obj->x - pxq, -COORD_MAX, COORD_MAX);
    int64_t dyq = clampi((int64_t)obj->y - pyq, -COORD_MAX, COORD_MAX);
    int64_t dist = isqrt64(dxq * dxq + dyq * dyq);
    int64_t d = maxi(0, dist - rq);
    int64_t v_rad = dist > 66 ? sdiv((int64_t)st->vpx * dxq + (int64_t)st->vpy * dyq, dist) : 0;
    int64_t lead = p->lead_base + mulq(p->lead_gain, p->ov_med);
    int64_t d_look = maxi(0, d - mulq(maxi(v_rad, 0), lead));
    if (st->d0 < 0 || obj->id != st->obj_id) {
        st->obj_id = obj->id;
        st->d0 = (int32_t)d;
        int64_t r_min = maxi(2 * rq, p->r_min_px);
        int64_t raw_zone = mulq(mulq(d, p->f_b), ONE + mulq(p->ov_zone_gain, p->ov_rate));
        st->zone = (int32_t)clampi(raw_zone, r_min, maxi(d, r_min));
        st->away_us = 0;
    }
    int64_t cs = 0;
    if (speed > 66 && dist > 66)
        cs = clampi(sdiv((int64_t)st->vx * dxq + (int64_t)st->vy * dyq, mulq(speed, dist)), -ONE, ONE);
    int64_t a_rec = smoothstep(mulq(-cs - p->c0, p->inv_c));
    int64_t zone = st->zone + mulq(a_rec, maxi(0, mulq(p->back_gain, mulq(p->ov_med, st->d0)) - st->zone));
    int64_t g = smoothstep(ONE - divq(d_look, zone));
    int64_t r_deep = maxi(mulq(p->deep_mult, rq), p->r_min_px);
    int64_t g_deep = smoothstep(ONE - divq(d_look, r_deep));
    int64_t frac = clampi(divq(d_look, st->zone), 0, ONE);
    int64_t v_ref = mulq(p->v_ref, isqrt64(frac * 65536));
    int64_t e = clampi(divq(maxi(0, speed - v_ref), p->v_ref), 0, ONE);
    int64_t sh = s_hold(p, rq);
    int64_t s_brake_g = mulq(p->s_brake, g);
    int64_t s_hold_g = mulq(sh, g_deep);
    int64_t s_pos = mulq(s_brake_g, ONE + mulq(p->lam, e)) + s_hold_g;
    int64_t leave = smoothstep(divq(maxi(0, speed - mulq(45875, p->v_leave)), mulq(19661, p->v_leave)));
    if (a_rec > HALF) st->away_us += (int32_t)dt_us;
    else if (cs > 0) st->away_us = 0;
    int64_t persist = smoothstep(((int64_t)(st->away_us - p->away_us) * 65536) / p->away_ramp_us);
    int64_t age = t_us - st->t_open;
    int64_t timeout = smoothstep(((age - p->open_us) * 65536) / p->open_ramp_us);
    int64_t relief = maxi(persist, timeout);
    int64_t s_rec = mulq(mulq(mulq(p->mu, s_brake_g + s_hold_g), a_rec), mulq(ONE - leave, ONE - relief));
    int64_t s_all = mulq(mini(s_pos + s_rec, p->s_cap), ONE - timeout);
    int64_t ramp = smoothstep((age * 65536) / p->ramp_us);
    return mulq(s_all, ramp);
}

static void apply(asc_state_t *st, int64_t k, int32_t dx, int32_t dy, int32_t *ox, int32_t *oy) {
    *ox = *oy = 0;
    if (dx == 0 && dy == 0) { st->carry_x = st->carry_y = 0; return; }
    for (int axis = 0; axis < 2; axis++) {
        int64_t raw = axis == 0 ? dx : dy;
        if (raw == 0) continue;
        int64_t carry = axis == 0 ? st->carry_x : st->carry_y;
        int64_t v = k * raw + carry;
        int64_t o = ((absi(v) + HALF) >> 16) * (v >= 0 ? 1 : -1);
        if (absi(o) > absi(raw)) o = raw;
        else if (o != 0 && (o > 0) != (raw > 0)) o = 0;
        int64_t nc = v - o * 65536;
        if (axis == 0) { st->carry_x = (int32_t)nc; *ox = (int32_t)o; } else { st->carry_y = (int32_t)nc; *oy = (int32_t)o; }
    }
}

int asc_tick(const asc_params_t *p, asc_state_t *st, int64_t t_us, int32_t dx, int32_t dy, int32_t px, int32_t py,
             const asc_obj_t *obj, asc_out_t *out) {
    int64_t dt_us = st->has_last ? t_us - st->last_t : 1000;
    if (st->has_last && !(dt_us > 0 && dt_us <= p->gap_us)) {
        st->vx = st->vy = st->vpx = st->vpy = 0;
        st->on_us = 0;
        dt_us = 1000;
    }
    st->last_t = t_us;
    st->has_last = 1;
    int64_t dt_q = (dt_us * 65536) / 1000;
    int64_t a_v = divq(dt_q, p->v_tau + dt_q);
    st->vx += (int32_t)mulq(a_v, sdiv((int64_t)dx * 4294967296LL, dt_q) - st->vx);
    st->vy += (int32_t)mulq(a_v, sdiv((int64_t)dy * 4294967296LL, dt_q) - st->vy);
    int64_t speed = isqrt64((int64_t)st->vx * st->vx + (int64_t)st->vy * st->vy);
    if (st->have_p) {
        int64_t a_p = divq(dt_q, p->vp_tau + dt_q);
        st->vpx += (int32_t)mulq(a_p, sdiv(((int64_t)px - st->pxp) * 65536, dt_q) - st->vpx);
        st->vpy += (int32_t)mulq(a_p, sdiv(((int64_t)py - st->pyp) * 65536, dt_q) - st->vpy);
    }
    st->pxp = px; st->pyp = py; st->have_p = 1;
    int reason = ASC_OK;
    if (p->enabled) {
        st->on_us = speed >= p->v_on ? (int32_t)(st->on_us + dt_us) : 0;
        st->still_us = speed < p->v_still ? (int32_t)(st->still_us + dt_us) : 0;
    }
    if (!p->enabled) {
        lock(st);
        reason = ASC_NO_PROFILE;
    } else if (st->guard == ASC_LOCKED) {
        if (st->on_us >= p->on_us) {
            st->guard = ASC_WAIT; st->t_move = t_us - st->on_us; st->d0 = -1; st->obj_id = -1;
        }
    } else if (st->still_us >= p->still_us) {
        lock(st);
    }
    if (p->enabled && st->guard == ASC_WAIT) {
        if (obj == NULL) reason = ASC_WAIT_OBJECT;
        else if (obj->has_appear && st->t_move < obj->t_appear_us + p->t_lo_us) reason = ASC_STIMULUS_LOCK;
        else { st->guard = ASC_OPEN; st->t_open = t_us; }
    }
    if (p->enabled && st->guard == ASC_LOCKED) reason = ASC_REASON_LOCKED;
    int64_t s_tgt = 0;
    if (st->guard == ASC_OPEN && obj != NULL) s_tgt = resistance(p, st, t_us, dt_us, px, py, obj, speed);
    else if (st->guard == ASC_OPEN) reason = ASC_NO_OBJECT;
    int64_t k_tgt = maxi(divq(ONE, ONE + s_tgt), p->k_floor);
    int64_t k_out;
    if (st->guard != ASC_OPEN) {
        st->k = ASC_ONE; st->kd = 0; st->s = 0;
        k_out = ONE;
    } else {
        int64_t w = k_tgt < st->k ? p->w_att : p->w_rel;
        int64_t term = mulq(mulq(w, w), k_tgt - st->k) - mulq(2 * w, st->kd);
        st->kd = (int32_t)clampi(st->kd + scale_us(term, dt_us), -p->slew, p->slew);
        st->k = (int32_t)clampi(st->k + scale_us(st->kd, dt_us), p->k_floor, ONE);
        st->s = (int32_t)(divq(ONE, st->k) - ONE);
        k_out = st->k;
    }
    int32_t ox, oy;
    apply(st, k_out, dx, dy, &ox, &oy);
    out->k = (int32_t)k_out; out->dx = ox; out->dy = oy; out->guard = st->guard; out->reason = reason; out->s = st->s;
    return 0;
}
