#include "bridge.h"
#include <string.h>

#define ONE ((int64_t)65536)
#define COORD ((int64_t)1 << 30)

typedef char asc_params_is_38_int32[sizeof(asc_params_t) == 38 * sizeof(int32_t) ? 1 : -1];
typedef char tremor_params_is_15_int32[sizeof(tremor_params_t) == 15 * sizeof(int32_t) ? 1 : -1];

static int64_t absl(int64_t a) { return a < 0 ? -a : a; }
static int64_t clampl(int64_t x, int64_t lo, int64_t hi) { return x < lo ? lo : (x > hi ? hi : x); }
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
static uint32_t rd32(const uint8_t *p) { return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24); }
static void wr16(uint8_t *p, uint32_t v) { p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); }
static void wr32(uint8_t *p, uint32_t v) { wr16(p, v); wr16(p + 2, v >> 16); }
static int16_t sat16(int64_t v) { return (int16_t)clampl(v, -32768, 32767); }

/* ---- parameter validation: the module is untrusted; every denominator and every range the cores rely on is checked ---- */
static const int32_t ASC_RANGE[38][2] = {
    {0, 1}, {1, 1 << 24}, {1, 1 << 24}, {1000, 1000000}, {1000, 5000000}, {0, 3000000}, {1000, 5000000}, {0, 2 * 65536}, {0, 65536},
    {0, 8 * 65536}, {0, 20 * 65536}, {0, 1 << 28}, {0, 65536}, {256, 1 << 28}, {64, 1 << 28}, {0, 4 * 65536}, {4096, 1 << 28},
    {1 << 14, 64 * 65536}, {0, 16 * 65536}, {1 << 14, 64 * 65536}, {0, 8 * 65536}, {0, 8 * 65536}, {-65536, 65536}, {0, 1 << 24},
    {6553, 65536}, {0, 20 * 65536}, {0, 1 << 28}, {0, 1 << 28}, {0, 10000000}, {1000, 10000000}, {0, 60000000}, {1000, 60000000},
    {0, 1 << 26}, {0, 1 << 26}, {0, 1 << 26}, {0, 1 << 24}, {0, 1 << 24}, {1000, 1000000}};
static const int32_t TRM_RANGE[15][2] = {
    {0, 1}, {0, 1 << 24}, {0, 1 << 24}, {0, 1 << 19}, {0, 65536}, {0, 64}, {1024, 1 << 28}, {0, 16 * 65536}, {1, 1 << 20},
    {-65536, 65536}, {0, 1 << 24}, {0, 1 << 28}, {0, 1 << 24}, {1000, 10000000}, {1, 2000}};

static int in_ranges(const int32_t *v, const int32_t (*r)[2], int n) {
    for (int i = 0; i < n; i++) if (v[i] < r[i][0] || v[i] > r[i][1]) return 0;
    return 1;
}
int bridge_asc_params_valid(const asc_params_t *p) { return in_ranges((const int32_t *)p, ASC_RANGE, 38); }
int bridge_tremor_params_valid(const tremor_params_t *p) { return in_ranges((const int32_t *)p, TRM_RANGE, 15); }

static void asc_defaults(asc_params_t *p) {
    memset(p, 0, sizeof *p);
    p->on_us = 6000; p->still_us = 150000; p->ramp_us = 20000; p->v_ref = 65536; p->v_leave = 65536; p->inv_c = 65536;
    p->away_ramp_us = 1; p->open_ramp_us = 1; p->gap_us = 20000;
}
static void trm_defaults(tremor_params_t *p) {
    memset(p, 0, sizeof *p);
    p->eps = 164; p->v_t = 65536; p->reset_us = 300000; p->reset_ms = 300;
}

size_t bridge_sizeof(void) { return sizeof(bridge_t); }

void bridge_cfg_defaults(bridge_cfg_t *c) {
    c->scope = IMG_SCOPE_TRANSPARENT; c->chord_mask = 0x18; c->chord_ms = 3000; c->panic_long_ms = 3000; c->rearm_ms = 2000;
    c->engage_hold_ms = 2000; c->probe_ms = 3000; c->pc_cfg_ms = 3000; c->soft_hold_ms = 5000; c->auto_engage = 1;
    c->param_ttl_ms = 5000; c->scene_ttl_ms = 100; c->link_ttl_ms = 500; c->slow_poll_us = 4000; c->split_poll_us = 900;
    c->lockin_ratio_q16 = 2621; c->lockin_min_counts = 250; c->lockin_hold_ms = 500; c->budget_us = 300; c->overrun_limit = 3;
    c->status_period_ms = 100;
    c->domain_counts = 511;      /* the cores' arithmetic is exact for |counts| <= 511 per axis per tick; beyond that the motion is ballistic */
}

void bridge_init(bridge_t *b, const bridge_cfg_t *c) {
    memset(b, 0, sizeof *b);
    b->cfg = *c;
    fs_cfg_t fc;
    fs_cfg_defaults(&fc);
    fc.panic_long_ms = c->panic_long_ms; fc.rearm_ms = c->rearm_ms; fc.engage_hold_ms = c->engage_hold_ms; fc.probe_ms = c->probe_ms;
    fc.pc_cfg_ms = c->pc_cfg_ms; fc.soft_hold_ms = c->soft_hold_ms; fc.auto_engage = c->auto_engage;
    fs_init(&b->fs, &fc);
    img_init(&b->img);
    px_reset(&b->px);
    asc_defaults(&b->asc_p);
    trm_defaults(&b->trm_p);
    asc_reset(&b->asc_s);
    tremor_reset(&b->trm_s);
    b->ppc_q = (int32_t)ONE;
    b->speed = BR_SPEED_FS;
    b->poll_us = 1000;
    b->gain_x = b->gain_y = (int32_t)ONE;
    b->last_k = (int32_t)ONE;
    b->pos[0].t = -1;
    b->pos_n = 1;
    b->pos_head = 0;
    b->lb_t = -1;
    b->chain_dirty = 1;
}

void bridge_boot(bridge_t *b, int64_t t, int cause, uint16_t crashes) {
    fs_boot(&b->fs, t, cause, crashes);
    b->crashes = b->fs.crashes;
}
int bridge_nv_take(bridge_t *b, uint16_t *crashes) {
    int d = b->fs.nv_dirty;
    b->fs.nv_dirty = 0;
    *crashes = b->fs.crashes;
    return d;
}

/* ---- housekeeping shared by all events ---- */
static uint32_t conditions(const bridge_t *b, int64_t t) {
    uint32_t c = 0;
    int64_t ttl = (int64_t)b->cfg.param_ttl_ms * 1000;
    if (!((b->have_asc && t - b->t_asc <= ttl) || (b->have_trm && t - b->t_trm <= ttl))) c |= FSC_STALE_PARAMS;
    if (!b->link_seen || t - b->t_link_rx > (int64_t)b->cfg.link_ttl_ms * 1000) c |= FSC_STALE_LINK;
    if (b->mode == BR_MODE_SLOW) c |= FSC_SLOW;
    return c;
}

static void idle_reset(bridge_t *b) {
    b->chain_dirty = 1;
    b->active = 0;
    b->win_open = 0;
    b->win_in_x = b->win_in_y = b->win_out_x = b->win_out_y = 0;
    b->gain_x = b->gain_y = (int32_t)ONE;
    b->gcarry_x = b->gcarry_y = 0;
}

static void refresh(bridge_t *b, int64_t t) {
    int prev_hw = b->fs.hw;
    fs_poll(&b->fs, t, conditions(b, t));
    if (b->fs.hw != prev_hw) {
        if (b->fs.hw == HW_SEL_PROBE) { img_init(&b->img); px_reset(&b->px); b->img_code = 0; }
        if (b->fs.hw == HW_SEL_BYPASS) { img_init(&b->img); px_reset(&b->px); b->buttons = 0; b->t_chord0 = 0; b->chord_active = 0; b->chord_fired = 0; }
        idle_reset(b);
    }
}

/* ---- position bookkeeping ---- */
static void pos_push(bridge_t *b, int64_t t) {
    b->pos_head = (uint16_t)((b->pos_head + 1) % BR_POS_RING);
    b->pos[b->pos_head].t = t; b->pos[b->pos_head].cx = b->cum_x; b->pos[b->pos_head].cy = b->cum_y;
    if (b->pos_n < BR_POS_RING) b->pos_n++;
}
static int pos_at(const bridge_t *b, int64_t t_cap, int64_t *cx, int64_t *cy) {
    uint16_t i = b->pos_head;
    for (uint16_t k = 0; k < b->pos_n; k++) {
        if (b->pos[i].t <= t_cap) { *cx = b->pos[i].cx; *cy = b->pos[i].cy; return 1; }
        i = (uint16_t)((i + BR_POS_RING - 1) % BR_POS_RING);
    }
    return 0;
}
static void cum_add(bridge_t *b, int64_t t, int32_t ox, int32_t oy) {
    if (!ox && !oy) return;
    b->cum_x += ox; b->cum_y += oy;
    pos_push(b, t);
}

static int64_t unwrap32(int64_t now, uint32_t lo) { return now + (int32_t)(lo - (uint32_t)now); }

static int select_obj(bridge_t *b, int64_t t, int64_t pxq, int64_t pyq, asc_obj_t *o) {
    if (b->n_obj == 0 || t - b->sc_t_cap > (int64_t)b->cfg.scene_ttl_ms * 1000) return 0;
    int64_t mx = (b->cum_x - b->sc_cum_x) * b->ppc_q, my = (b->cum_y - b->sc_cum_y) * b->ppc_q;
    int best = -1;
    int64_t best_score = 0, bx = 0, by = 0;
    for (int i = 0; i < b->n_obj; i++) {
        int64_t rx = clampl(b->obj[i].x - mx, -COORD, COORD), ry = clampl(b->obj[i].y - my, -COORD, COORD);
        int64_t score = isqrt64(rx * rx + ry * ry) - clampl(b->obj[i].radius, 0, COORD / 4);
        if (best < 0 || score < best_score) { best = i; best_score = score; bx = rx; by = ry; }
    }
    o->id = b->obj[best].id;
    o->x = (int32_t)clampl(pxq + bx, -2147483647, 2147483647);
    o->y = (int32_t)clampl(pyq + by, -2147483647, 2147483647);
    o->radius = (int32_t)clampl(b->obj[best].radius, 0, COORD / 4);
    o->has_appear = b->obj[best].flags & 1;
    o->t_appear_us = unwrap32(t, b->obj[best].t_appear_lo);
    return 1;
}

/* ---- lock-in monitor: the help must never hold the pointer in a trap.
 * It compares the NET motion (vector sums) the person asked for with what the pointer got over the last 500 ms. Oscillation (tremor
 * the filters rightly remove) has almost no net motion, so it can not trip it; a sustained push that goes nowhere can. */
static void lock_clear(bridge_t *b) { memset(b->lb_in, 0, sizeof b->lb_in); memset(b->lb_out, 0, sizeof b->lb_out); }
static int lock_update(bridge_t *b, int64_t t, int32_t dx, int32_t dy, int32_t fx, int32_t fy) {
    int64_t bk = t / ((int64_t)BR_LOCK_BUCKET_MS * 1000);
    if (bk != b->lb_t) {
        int64_t steps = bk - b->lb_t;
        if (b->lb_t < 0 || steps >= BR_LOCK_BUCKETS || steps < 0) lock_clear(b);
        else for (int64_t s = 1; s <= steps; s++) { int i = (int)((b->lb_t + s) % BR_LOCK_BUCKETS); b->lb_in[i][0] = b->lb_in[i][1] = 0; b->lb_out[i][0] = b->lb_out[i][1] = 0; }
        b->lb_t = bk;
    }
    int i = (int)(bk % BR_LOCK_BUCKETS);
    b->lb_in[i][0] += dx; b->lb_in[i][1] += dy;
    b->lb_out[i][0] += fx; b->lb_out[i][1] += fy;
    int64_t ix = 0, iy = 0, ox = 0, oy = 0;
    for (int k = 0; k < BR_LOCK_BUCKETS; k++) { ix += b->lb_in[k][0]; iy += b->lb_in[k][1]; ox += b->lb_out[k][0]; oy += b->lb_out[k][1]; }
    int64_t net_in = absl(ix) + absl(iy), net_out = absl(ox) + absl(oy);
    if (net_in >= b->cfg.lockin_min_counts && net_out * 65536 < (int64_t)b->cfg.lockin_ratio_q16 * net_in) {
        b->lockin_trips++;
        b->lock_hold_until = t + (int64_t)b->cfg.lockin_hold_ms * 1000;
        b->chain_dirty = 1;
        lock_clear(b);
        return 1;
    }
    return 0;
}

static int chain_on(const bridge_t *b, int64_t t) {
    return b->fs.state == FS_ASSIST && t >= b->lock_hold_until && (b->have_asc || b->have_trm);
}

static int32_t clamp_axis(int32_t in, int32_t out, int *bad) {
    if (in == 0) { if (out != 0) *bad = 1; return 0; }
    if (out != 0 && (out > 0) != (in > 0)) { *bad = 1; return 0; }
    if (absl(out) > absl(in)) { *bad = 1; return in; }
    return out;
}

/* One tick of the correction chain: ASC on the person's own motion, then tremor suppression on its output, then the final clamp. */
static void chain_tick(bridge_t *b, int64_t t, int32_t dx, int32_t dy, int32_t *ox, int32_t *oy) {
    *ox = dx; *oy = dy;
    b->last_k = (int32_t)ONE; b->last_guard = ASC_LOCKED;
    if (!chain_on(b, t)) { b->chain_dirty = 1; return; }
    if (b->chain_dirty) { asc_reset(&b->asc_s); tremor_reset(&b->trm_s); b->chain_dirty = 0; }
    if (absl(dx) > b->cfg.domain_counts || absl(dy) > b->cfg.domain_counts) return;   /* a ballistic flick needs no help (and is outside the cores' exact range) */
    int64_t curx = b->cum_x * b->ppc_q, cury = b->cum_y * b->ppc_q;
    int64_t pxq = curx - b->base_q_x, pyq = cury - b->base_q_y;
    if (absl(pxq) > COORD || absl(pyq) > COORD) {      /* keep the cursor inside the core's coordinate range; restart its velocity estimate */
        b->base_q_x = curx; b->base_q_y = cury; pxq = pyq = 0; b->asc_s.have_p = 0;
    }
    asc_obj_t obj;
    int have_obj = select_obj(b, t, pxq, pyq, &obj);
    asc_out_t ao;
    asc_tick(&b->asc_p, &b->asc_s, t, dx, dy, (int32_t)pxq, (int32_t)pyq, have_obj ? &obj : 0, &ao);
    int32_t fx, fy;
    tremor_tick(&b->trm_p, &b->trm_s, t, ao.dx, ao.dy, &fx, &fy);
#ifdef BRIDGE_TESTING
    if (b->inject == 1 && (dx || dy)) { fx = dx * 2; fy = dy * 2; b->inject = 0; }
    else if (b->inject == 2) { fx = fy = 0; }                       /* a stuck chain: the lock-in monitor must release it */
#endif
    int bad = 0;
    fx = clamp_axis(dx, fx, &bad);
    fy = clamp_axis(dy, fy, &bad);
    if (bad) {
        b->invariant_viol++;
        if (++b->inv_pending >= 3) { b->inv_pending = 0; fs_fault(&b->fs, t, FSF_INVARIANT); }
    }
    b->last_k = ao.k; b->last_guard = ao.guard;
    if ((dx || dy) && lock_update(b, t, dx, dy, fx, fy)) { fx = dx; fy = dy; b->last_k = (int32_t)ONE; }
    *ox = fx; *oy = fy;
}

/* ---- telemetry ---- */
static void tel_push(bridge_t *b, int64_t t, int32_t rx, int32_t ry, int32_t ox, int32_t oy, int split) {
    if (!rx && !ry && !ox && !oy) return;
    if (b->tel_head - b->tel_tail >= BR_TEL_RING) {         /* drop the oldest; the new oldest is marked: samples were lost before it */
        b->tel_tail++; b->telem_dropped++;
        b->tel[b->tel_tail % BR_TEL_RING].flags |= 0x80;
    }
    br_tel_t *s = &b->tel[b->tel_head % BR_TEL_RING];
    s->t_us = (uint32_t)t; s->raw_dx = sat16(rx); s->raw_dy = sat16(ry); s->out_dx = sat16(ox); s->out_dy = sat16(oy);
    s->k_q15 = (uint16_t)clampl(b->last_k >> 1, 0, 32768);
    s->buttons = (uint8_t)b->buttons;
    s->flags = (uint8_t)((b->last_guard & 3) | ((b->fs.state == FS_ASSIST) << 2) | (split << 3) | ((t < b->lock_hold_until) << 4));
    b->tel_head++;
}

/* ---- the 1 ms grid ---- */
static void catch_up(bridge_t *b, int64_t t) {
    int n = 0;
    while (b->active && t - b->last_tick_t >= 1500) {
        if (n++ >= 64) { idle_reset(b); return; }
        b->last_tick_t += 1000;
        int32_t ox, oy;
        chain_tick(b, b->last_tick_t, 0, 0, &ox, &oy);
    }
}

static void split_close(bridge_t *b, int64_t t) {
    int n = 0;
    while (b->win_open && t >= b->win_end) {
        if (n++ >= 64) { idle_reset(b); return; }
        int32_t ox, oy;
        chain_tick(b, b->win_end, b->win_in_x, b->win_in_y, &ox, &oy);
        if (b->win_in_x) b->gain_x = (int32_t)clampl((int64_t)ox * 65536 / b->win_in_x, 0, ONE); else b->gcarry_x = 0;
        if (b->win_in_y) b->gain_y = (int32_t)clampl((int64_t)oy * 65536 / b->win_in_y, 0, ONE); else b->gcarry_y = 0;
        tel_push(b, b->win_end, b->win_in_x, b->win_in_y, b->win_out_x, b->win_out_y, 1);
        b->win_in_x = b->win_in_y = b->win_out_x = b->win_out_y = 0;
        b->win_end += 1000;
    }
}

static int32_t apply_gain(int32_t in, int32_t g, int32_t *carry) {
    if (in == 0) return 0;
    int64_t v = (int64_t)in * g + *carry;
    int64_t o = ((absl(v) + 32768) >> 16) * (v >= 0 ? 1 : -1);
    if (absl(o) > absl(in)) o = in;
    else if (o != 0 && (o > 0) != (in > 0)) o = 0;
    int64_t nc = v - o * 65536;
    *carry = (int32_t)clampl(nc, -65535, 65535);
    return (int32_t)o;
}

/* ---- host side image ---- */
void bridge_set_speed(bridge_t *b, int speed) { b->speed = speed; }
void bridge_dev_present(bridge_t *b, int64_t t, int present) {
    fs_device(&b->fs, t, present);
    refresh(b, t);
}
int bridge_img_device(bridge_t *b, const uint8_t *d, uint16_t len) { return img_set_device(&b->img, d, len); }
int bridge_img_config(bridge_t *b, const uint8_t *c, uint16_t len) { return img_set_config(&b->img, c, len); }
int bridge_img_n_if(const bridge_t *b) { return b->img.n_if; }
uint16_t bridge_img_rd_wanted(const bridge_t *b, int i) { return img_rd_wanted(&b->img, i); }
int bridge_img_report_desc(bridge_t *b, int i, const uint8_t *rd, uint16_t len) { return img_set_report_desc(&b->img, i, rd, len); }

static void compute_mode(bridge_t *b) {
    int64_t best = 0;
    for (int i = 0; i < b->img.n_ep; i++) {
        const img_ep_t *e = &b->img.ep[i];
        if (!(e->addr & 0x80) || b->img.ifc[e->iface].role != ROLE_MOUSE_MOTION) continue;
        int iv = e->interval < 1 ? 1 : e->interval;
        int64_t us = b->speed == BR_SPEED_HS ? (int64_t)125 << (iv > 16 ? 15 : iv - 1) : (int64_t)iv * 1000;
        if (best == 0 || us < best) best = us;
    }
    b->poll_us = best ? best : 1000;
    b->mode = b->poll_us > b->cfg.slow_poll_us ? BR_MODE_SLOW : (b->poll_us < b->cfg.split_poll_us ? BR_MODE_SPLIT : BR_MODE_DIRECT);
}

int bridge_img_done(bridge_t *b, int64_t t) {
    int code = img_finalize(&b->img, b->cfg.scope);
    b->img_code = code;
    if (code == IMG_OK) compute_mode(b);
    fs_image(&b->fs, t, code);
    refresh(b, t);
    return code;
}

/* ---- device side (the PC) ---- */
int bridge_pc_setup(bridge_t *b, const usb_setup_t *s, const uint8_t **data, uint16_t *len) { return px_classify(&b->img, s, data, len); }
void bridge_pc_forwarded_ok(bridge_t *b, int64_t t, const usb_setup_t *s) {
    px_forwarded_ok(&b->img, &b->px, s);
    if (s->bmRequestType == 0x00 && s->bRequest == 9 && s->wValue != 0) fs_pc_configured(&b->fs, t);
    refresh(b, t);
}
void bridge_pc_bus_reset(bridge_t *b, int64_t t) {
    px_reset(&b->px);
    fs_pc_reset(&b->fs, t);
    refresh(b, t);
}
void bridge_usb_error(bridge_t *b, int64_t t) {
    b->usb_errors++;
    if (b->usb_errors % 8 == 0) { fs_fault(&b->fs, t, FSF_USB_ERRORS); refresh(b, t); }
}

/* ---- reports ---- */
static int motion_if(const bridge_t *b, uint8_t ep, int *ii) {
    const img_ep_t *e = img_find_ep(&b->img, ep);
    if (!e || !(ep & 0x80) || (e->attr & 3) != 3) return 0;
    if (b->img.ifc[e->iface].role != ROLE_MOUSE_MOTION) return 0;
    *ii = e->iface;
    return 1;
}

static void chord_check(bridge_t *b, int64_t t) {
    uint32_t mask = (uint32_t)b->cfg.chord_mask;
    int held = mask != 0 && (b->buttons & mask) == mask;
    if (!held) { b->chord_active = 0; b->chord_fired = 0; return; }
    if (!b->chord_active) { b->chord_active = 1; b->t_chord0 = t; }
    if (!b->chord_fired && t - b->t_chord0 >= (int64_t)b->cfg.chord_ms * 1000) {
        b->chord_fired = 1;
        fs_chord(&b->fs, t);
        refresh(b, t);
    }
}

int bridge_mouse_in(bridge_t *b, int64_t t, uint8_t ep, const uint8_t *in, uint16_t len, uint8_t *out) {
    if (out != in && len) memcpy(out, in, len);
    int ii;
    if (!motion_if(b, ep, &ii)) { b->other_reports++; return 0; }
    const hid_map_t *map = px_map(&b->img, &b->px, ii);
    const hid_motion_t *m = hid_find(map, in, len);
    if (!m) { b->short_reports++; return 0; }
    b->motion_reports++;
    int32_t dx, dy;
    hid_xy_get(m, in, &dx, &dy);
    b->buttons = hid_buttons(m, in, len);
    refresh(b, t);
    chord_check(b, t);
    if (dx || dy) { b->last_motion_t = t; if (!b->active) { b->active = 1; b->last_tick_t = t - 1000; b->win_open = 0; } }
    int32_t ox = dx, oy = dy;
    if (!chain_on(b, t)) {
        b->win_open = 0; b->gain_x = b->gain_y = (int32_t)ONE; b->gcarry_x = b->gcarry_y = 0;
        b->chain_dirty = 1;
        tel_push(b, t, dx, dy, dx, dy, 0);
    } else if (b->mode == BR_MODE_SPLIT) {
        split_close(b, t);
        if ((dx || dy) && !b->win_open) { b->win_open = 1; b->win_end = t + 1000; }
        if (b->win_open) { b->win_in_x += dx; b->win_in_y += dy; }
        ox = apply_gain(dx, b->gain_x, &b->gcarry_x);
        oy = apply_gain(dy, b->gain_y, &b->gcarry_y);
        int bad = 0;
        ox = clamp_axis(dx, ox, &bad); oy = clamp_axis(dy, oy, &bad);
        if (b->win_open) { b->win_out_x += ox; b->win_out_y += oy; }
    } else {
        catch_up(b, t);
        if (dx || dy) {
            chain_tick(b, t, dx, dy, &ox, &oy);
            b->last_tick_t = t;
        } else {
            ox = dx; oy = dy;                             /* a button-only report: nothing to correct */
        }
        tel_push(b, t, dx, dy, ox, oy, 0);
    }
    cum_add(b, t, ox, oy);
    if (ox == dx && oy == dy) return 0;
    hid_xy_set(m, out, ox, oy);
    return 1;
}

int bridge_merge(bridge_t *b, uint8_t ep, uint8_t *a, const uint8_t *bb, uint16_t len) {
    int ii;
    if (!motion_if(b, ep, &ii)) return 0;
    const hid_motion_t *m = hid_find(px_map(&b->img, &b->px, ii), a, len);
    if (!m || !hid_find(px_map(&b->img, &b->px, ii), bb, len)) return 0;
    return hid_merge(m, a, bb, len);
}

void bridge_poll(bridge_t *b, int64_t t) {
    refresh(b, t);
    chord_check(b, t);
    if (b->active && t - b->last_motion_t > 1000000) { idle_reset(b); return; }
    if (!b->active) return;
    if (b->mode == BR_MODE_SPLIT && b->fs.state == FS_ASSIST) split_close(b, t);
    else catch_up(b, t);
}

void bridge_panic(bridge_t *b, int64_t t, int pressed) { fs_panic(&b->fs, t, pressed); refresh(b, t); }
void bridge_fatal(bridge_t *b, int64_t t) { fs_fault(&b->fs, t, FSF_FATAL); refresh(b, t); }
void bridge_note_cost(bridge_t *b, int64_t t, uint32_t us) {
    b->tick_n++; b->tick_sum_us += us;
    if (us > b->tick_max_us) b->tick_max_us = us;
    if (us > (uint32_t)b->cfg.budget_us) {
        b->overruns++;
        if (++b->overrun_streak >= (uint32_t)b->cfg.overrun_limit) { b->overrun_streak = 0; fs_fault(&b->fs, t, FSF_OVERRUN); refresh(b, t); }
    } else {
        b->overrun_streak = 0;
    }
}

/* ---- link ---- */
#define ASC_BLOB (16 + 152 + 4)
#define TRM_BLOB (16 + 60 + 4)

static void reject(bridge_t *b) { b->params_rejected++; }

static void blob_asc(bridge_t *b, int64_t t, const uint8_t *p, int n) {
    if (n != ASC_BLOB) { reject(b); return; }
    uint32_t crc = rd32(p + n - 4);
    if (crc != link_crc32(p, (uint32_t)n - 4)) { reject(b); return; }
    uint32_t gen = rd32(p), pid = rd32(p + 4), ppc = rd32(p + 8);
    asc_params_t q;
    memcpy(&q, p + 16, sizeof q);
    if (!bridge_asc_params_valid(&q) || ppc < 4096u || ppc > 16u * 65536u) { reject(b); return; }
    if (b->have_asc) {
        int32_t d = (int32_t)(gen - b->gen_asc);
        int64_t ttl = (int64_t)b->cfg.param_ttl_ms * 1000;
        if (d == 0 && crc == b->crc_asc) { b->t_asc = t; return; }
        if (d == 0 || (d < 0 && t - b->t_asc <= ttl)) { reject(b); return; }      /* replay, or a conflicting blob of the same generation */
    }
    b->asc_p = q; b->ppc_q = (int32_t)ppc; b->gen_asc = gen; b->pid_asc = pid; b->crc_asc = crc; b->have_asc = 1; b->t_asc = t;
}

static void blob_trm(bridge_t *b, int64_t t, const uint8_t *p, int n) {
    if (n != TRM_BLOB) { reject(b); return; }
    uint32_t crc = rd32(p + n - 4);
    if (crc != link_crc32(p, (uint32_t)n - 4)) { reject(b); return; }
    uint32_t gen = rd32(p), pid = rd32(p + 4);
    tremor_params_t q;
    memcpy(&q, p + 16, sizeof q);
    if (!bridge_tremor_params_valid(&q)) { reject(b); return; }
    if (b->have_trm) {
        int32_t d = (int32_t)(gen - b->gen_trm);
        int64_t ttl = (int64_t)b->cfg.param_ttl_ms * 1000;
        if (d == 0 && crc == b->crc_trm) { b->t_trm = t; return; }
        if (d == 0 || (d < 0 && t - b->t_trm <= ttl)) { reject(b); return; }
    }
    b->trm_p = q; b->gen_trm = gen; b->pid_trm = pid; b->crc_trm = crc; b->have_trm = 1; b->t_trm = t;
}

static void scene_rx(bridge_t *b, int64_t t, const link_frame_t *f) {
    if (f->len < 8) { reject(b); return; }
    int n = f->payload[4];
    if (n > BR_MAX_OBJ || f->len != 8 + 20 * n) { reject(b); return; }
    int64_t t_cap = unwrap32(t, rd32(f->payload));
    if (t_cap > t) t_cap = t;
    int64_t cx, cy;
    if (!pos_at(b, t_cap, &cx, &cy)) { b->n_obj = 0; return; }       /* capture older than the position history: unusable */
    for (int i = 0; i < n; i++) {
        const uint8_t *q = f->payload + 8 + 20 * i;
        b->obj[i].id = (int32_t)(q[0] | (q[1] << 8));
        b->obj[i].flags = q[2];
        b->obj[i].x = (int32_t)rd32(q + 4); b->obj[i].y = (int32_t)rd32(q + 8); b->obj[i].radius = (int32_t)rd32(q + 12);
        b->obj[i].t_appear_lo = rd32(q + 16);
    }
    b->n_obj = n; b->sc_t_cap = t_cap; b->sc_cum_x = cx; b->sc_cum_y = cy;
}

void bridge_link_rx(bridge_t *b, int64_t t, const uint8_t frame[LINK_FRAME]) {
    link_frame_t f;
    if (link_unpack(frame, &f) != LINK_OK) { b->link_rx_bad++; return; }
    b->link_rx_ok++; b->t_link_rx = t; b->link_seen = 1; b->rx_seq = f.seq;
    switch (f.kind) {
    case LK_PARAMS_ASC: case LK_PARAMS_TREMOR: {
        link_blob_t *bl = f.kind == LK_PARAMS_ASC ? &b->blob_asc : &b->blob_trm;
        int n = link_blob_feed(bl, &f, t, 100000);
        if (n < 0) reject(b);
        else if (n > 0) { if (f.kind == LK_PARAMS_ASC) blob_asc(b, t, bl->buf, n); else blob_trm(b, t, bl->buf, n); }
        break; }
    case LK_SCENE: scene_rx(b, t, &f); break;
    case LK_CMD:
        if (f.len >= 1 && f.payload[0] <= 2) fs_cmd(&b->fs, t, f.payload[0]); else reject(b);
        break;
    case LK_TSYNC:
        if (f.len >= 8) { b->tsync_m = (uint64_t)rd32(f.payload) | ((uint64_t)rd32(f.payload + 4) << 32); b->t_tsync_rx = t; b->tsync_pending = 1; }
        break;
    case LK_HELLO: case LK_NOP: break;
    default: reject(b); break;
    }
    refresh(b, t);
}

void bridge_status(const bridge_t *b, int64_t t, bridge_status_t *s) {
    memset(s, 0, sizeof *s);
    int64_t ttl = (int64_t)b->cfg.param_ttl_ms * 1000;
    s->state = b->fs.state; s->reason = b->fs.reason; s->hw = b->fs.hw; s->attach = fs_attach(&b->fs);
    s->latch_soft = b->fs.latch_soft; s->latch_hw = b->fs.latch_hw; s->mode = b->mode; s->guard = b->last_guard; s->k_q16 = b->last_k;
    s->gen_asc = (int32_t)b->gen_asc; s->gen_tremor = (int32_t)b->gen_trm;
    s->params_asc_ok = b->have_asc && t - b->t_asc <= ttl; s->params_trm_ok = b->have_trm && t - b->t_trm <= ttl;
    s->link_ok = b->link_seen && t - b->t_link_rx <= (int64_t)b->cfg.link_ttl_ms * 1000;
    s->scene_n = b->n_obj; s->scene_age_ms = b->n_obj ? (int32_t)((t - b->sc_t_cap) / 1000) : -1;
    s->lockin_hold = t < b->lock_hold_until; s->chord_active = b->chord_active; s->healthy = b->fs.healthy;
    s->img_code = b->img_code; s->n_motion_if = b->img.n_motion;
    s->motion_reports = (int32_t)b->motion_reports; s->other_reports = (int32_t)b->other_reports; s->short_reports = (int32_t)b->short_reports;
    s->invariant_viol = (int32_t)b->invariant_viol; s->lockin_trips = (int32_t)b->lockin_trips; s->overruns = (int32_t)b->overruns;
    s->params_rejected = (int32_t)b->params_rejected; s->link_rx_ok = (int32_t)b->link_rx_ok; s->link_rx_bad = (int32_t)b->link_rx_bad;
    s->telem_dropped = (int32_t)b->telem_dropped; s->telem_pending = (int32_t)(b->tel_head - b->tel_tail);
    s->tick_max_us = (int32_t)b->tick_max_us; s->tick_avg_us = b->tick_n ? (int32_t)(b->tick_sum_us / b->tick_n) : 0;
    s->usb_errors = (int32_t)b->usb_errors; s->crashes = b->fs.crashes; s->wants_kick = fs_wants_kick(&b->fs);
    s->vid = b->img.have_dev ? b->img.dev[8] | (b->img.dev[9] << 8) : 0; s->pid = b->img.have_dev ? b->img.dev[10] | (b->img.dev[11] << 8) : 0;
    s->buttons = (int32_t)b->buttons; s->active = b->active; s->cum_x = (int32_t)b->cum_x; s->cum_y = (int32_t)b->cum_y;
}

void bridge_link_tx(bridge_t *b, int64_t t, uint8_t out[LINK_FRAME]) {
    link_frame_t f;
    memset(&f, 0, sizeof f);
    f.seq = ++b->tx_seq; f.ack = b->rx_seq; f.frag_cnt = 1;
    if (b->tsync_pending) {
        b->tsync_pending = 0;
        f.kind = LK_TSYNC_REPLY; f.len = 24;
        wr32(f.payload, (uint32_t)b->tsync_m); wr32(f.payload + 4, (uint32_t)(b->tsync_m >> 32));
        wr32(f.payload + 8, (uint32_t)b->t_tsync_rx); wr32(f.payload + 12, (uint32_t)((uint64_t)b->t_tsync_rx >> 32));
        wr32(f.payload + 16, (uint32_t)t); wr32(f.payload + 20, (uint32_t)((uint64_t)t >> 32));
    } else if (t >= b->t_next_status) {
        bridge_status_t s;
        bridge_status(b, t, &s);
        b->t_next_status = t + (int64_t)b->cfg.status_period_ms * 1000;
        uint8_t *p = f.payload;
        f.kind = LK_STATUS; f.len = 72;
        p[0] = (uint8_t)s.state; p[1] = (uint8_t)s.reason; p[2] = (uint8_t)s.hw; p[3] = (uint8_t)s.mode;
        wr16(p + 4, (uint32_t)(s.latch_soft | (s.latch_hw ? 2 : 0) | (s.attach << 2) | (s.params_asc_ok << 3) | (s.params_trm_ok << 4) | (s.link_ok << 5) |
                               ((s.scene_n > 0) << 6) | (s.lockin_hold << 7) | (s.chord_active << 8) | (s.healthy << 9)));
        p[6] = (uint8_t)(s.img_code < 0 ? -s.img_code : 0); p[7] = (uint8_t)s.n_motion_if;
        wr32(p + 8, (uint32_t)s.gen_asc); wr32(p + 12, (uint32_t)s.gen_tremor); wr32(p + 16, (uint32_t)s.motion_reports);
        wr32(p + 20, (uint32_t)s.other_reports); wr32(p + 24, (uint32_t)s.invariant_viol); wr32(p + 28, (uint32_t)s.lockin_trips);
        wr32(p + 32, (uint32_t)s.overruns); wr32(p + 36, (uint32_t)s.params_rejected); wr32(p + 40, (uint32_t)s.link_rx_ok);
        wr32(p + 44, (uint32_t)s.link_rx_bad); wr32(p + 48, (uint32_t)s.telem_dropped);
        wr16(p + 52, (uint32_t)s.tick_max_us); wr16(p + 54, (uint32_t)s.tick_avg_us); wr16(p + 56, (uint32_t)(s.scene_age_ms < 0 ? 0xFFFF : s.scene_age_ms));
        wr16(p + 58, (uint32_t)s.usb_errors); wr16(p + 60, (uint32_t)s.vid); wr16(p + 62, (uint32_t)s.pid); wr16(p + 64, (uint32_t)s.crashes);
        p[66] = (uint8_t)s.guard; wr32(p + 68, (uint32_t)t);
    } else if (b->tel_head != b->tel_tail) {
        f.kind = LK_TELEM;
        uint32_t n = b->tel_head - b->tel_tail;
        if (n > 6) n = 6;
        f.payload[0] = (uint8_t)n; wr16(f.payload + 2, b->telem_dropped);
        for (uint32_t i = 0; i < n; i++) {
            const br_tel_t *s = &b->tel[(b->tel_tail + i) % BR_TEL_RING];
            uint8_t *q = f.payload + 4 + 16 * i;
            wr32(q, s->t_us); wr16(q + 4, (uint16_t)s->raw_dx); wr16(q + 6, (uint16_t)s->raw_dy); wr16(q + 8, (uint16_t)s->out_dx);
            wr16(q + 10, (uint16_t)s->out_dy); wr16(q + 12, s->k_q15); q[14] = s->buttons; q[15] = s->flags;
        }
        b->tel_tail += n;
        f.len = (uint16_t)(4 + 16 * n);
    } else {
        f.kind = LK_NOP;
    }
    link_pack(out, &f);
}

int bridge_hw_select(const bridge_t *b) { return fs_hw_select(&b->fs); }
int bridge_attach(const bridge_t *b) { return fs_attach(&b->fs); }
int bridge_wants_kick(const bridge_t *b) { return fs_wants_kick(&b->fs); }

#ifdef BRIDGE_TESTING
void bridge_param_range(int tremor, int i, int32_t *lo, int32_t *hi) {
    const int32_t (*r)[2] = tremor ? TRM_RANGE : ASC_RANGE;
    *lo = r[i][0]; *hi = r[i][1];
}
void bridge_debug_inject(bridge_t *b, int mode) { b->inject = (int8_t)mode; }
#endif

size_t bridge_status_sizeof(void) { return sizeof(bridge_status_t); }
size_t bridge_cfg_sizeof(void) { return sizeof(bridge_cfg_t); }
