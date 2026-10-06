/* Assistive HID bridge core: a transparent USB proxy for ONE mouse that edits only the X/Y fields of the mouse's own reports, in
 * place, and only ever makes them smaller (|out| <= |in| per axis, same sign or zero, zero in -> zero out). Sans-IO: the port
 * (USB host + device stacks, SPI DMA, GPIO, timer) feeds events in and collects results; nothing here blocks, allocates or touches
 * hardware. Single context: the port serialises the calls (one task, or ISRs masked around them).
 *
 * The compute module (SPI master) supplies the person's parameters (ASC and tremor blobs from their BioProfile) and a scene of
 * nearby objects of interest; it receives telemetry of raw/corrected motion. The 1 kHz path never waits for it. */
#ifndef BRIDGE_H
#define BRIDGE_H
#include <stddef.h>
#include <stdint.h>
#include "asc_core.h"
#include "tremor_core.h"
#include "hid_desc.h"
#include "usb_image.h"
#include "usb_proxy.h"
#include "link.h"
#include "failsafe.h"

#define BR_TEL_RING 2048
#define BR_POS_RING 256
#define BR_MAX_OBJ 5
#define BR_LOCK_BUCKETS 20
#define BR_LOCK_BUCKET_MS 25

enum { BR_MODE_DIRECT = 0, BR_MODE_SPLIT = 1, BR_MODE_SLOW = 2 };
enum { BR_SPEED_LS = 0, BR_SPEED_FS = 1, BR_SPEED_HS = 2 };

typedef struct {
    int32_t scope, chord_mask, chord_ms, panic_long_ms, rearm_ms, engage_hold_ms, probe_ms, pc_cfg_ms, soft_hold_ms, auto_engage,
        param_ttl_ms, scene_ttl_ms, link_ttl_ms, slow_poll_us, split_poll_us, lockin_ratio_q16, lockin_min_counts, lockin_hold_ms,
        budget_us, overrun_limit, status_period_ms, domain_counts;
} bridge_cfg_t;

typedef struct { uint32_t t_us; int16_t raw_dx, raw_dy, out_dx, out_dy; uint16_t k_q15; uint8_t buttons, flags; } br_tel_t;
typedef struct { int32_t id, flags; int64_t x, y, radius; uint32_t t_appear_lo; } br_obj_t;
typedef struct { int64_t t, cx, cy; } br_pos_t;

typedef struct {
    int32_t state, reason, hw, attach, latch_soft, latch_hw, mode, guard, k_q16, gen_asc, gen_tremor, params_asc_ok, params_trm_ok,
        link_ok, scene_n, scene_age_ms, lockin_hold, chord_active, healthy, img_code, n_motion_if, motion_reports, other_reports,
        short_reports, invariant_viol, lockin_trips, overruns, params_rejected, link_rx_ok, link_rx_bad, telem_dropped, telem_pending,
        tick_max_us, tick_avg_us, usb_errors, crashes, wants_kick, vid, pid, buttons, active, cum_x, cum_y;
} bridge_status_t;

typedef struct bridge {
    bridge_cfg_t cfg;
    fs_t fs;
    usb_image_t img;
    px_state_t px;
    int32_t speed, mode, img_code;
    int64_t poll_us;
    /* chain */
    asc_params_t asc_p;
    asc_state_t asc_s;
    tremor_params_t trm_p;
    tremor_state_t trm_s;
    int32_t ppc_q;
    uint32_t gen_asc, gen_trm, pid_asc, pid_trm, crc_asc, crc_trm;
    int8_t have_asc, have_trm;
    int64_t t_asc, t_trm, t_link_rx;
    int8_t chain_dirty, active, link_seen;
    int64_t last_tick_t, last_motion_t;
    int32_t last_k, last_guard;
    /* position bookkeeping (what the PC received) */
    int64_t cum_x, cum_y, base_q_x, base_q_y;
    br_pos_t pos[BR_POS_RING];
    uint16_t pos_n, pos_head;
    /* scene */
    br_obj_t obj[BR_MAX_OBJ];
    int32_t n_obj;
    int64_t sc_t_cap, sc_cum_x, sc_cum_y;
    /* split mode */
    int8_t win_open;
    int64_t win_end;
    int32_t win_in_x, win_in_y, win_out_x, win_out_y, gain_x, gain_y, gcarry_x, gcarry_y;
    /* lock-in monitor */
    int64_t lock_hold_until;
    int32_t lb_in[BR_LOCK_BUCKETS][2], lb_out[BR_LOCK_BUCKETS][2];   /* net (vector) motion per bucket */
    int64_t lb_t;
    /* buttons / chord */
    uint32_t buttons;
    int64_t t_chord0;
    int8_t chord_fired, chord_active;
    /* telemetry */
    br_tel_t tel[BR_TEL_RING];
    uint32_t tel_head, tel_tail;
    uint32_t telem_dropped;
    /* link */
    uint16_t tx_seq, rx_seq;
    int64_t t_next_status, t_tsync_rx;
    uint64_t tsync_m;
    int8_t tsync_pending;
    link_blob_t blob_asc, blob_trm;
    /* stats */
    uint32_t motion_reports, other_reports, short_reports, invariant_viol, lockin_trips, overruns, params_rejected, link_rx_ok,
        link_rx_bad, usb_errors, overrun_streak, inv_pending, tick_n;
    uint32_t tick_max_us;
    uint64_t tick_sum_us;
    uint16_t crashes;
    int8_t inject;
} bridge_t;

size_t bridge_sizeof(void);
size_t bridge_status_sizeof(void);
size_t bridge_cfg_sizeof(void);
void bridge_cfg_defaults(bridge_cfg_t *c);
void bridge_init(bridge_t *b, const bridge_cfg_t *c);
void bridge_boot(bridge_t *b, int64_t t, int reset_cause, uint16_t crashes);
int bridge_nv_take(bridge_t *b, uint16_t *crashes);           /* 1 if the persisted crash counter must be written */

/* host side (the mouse) */
void bridge_set_speed(bridge_t *b, int speed);
void bridge_dev_present(bridge_t *b, int64_t t, int present);
int bridge_img_device(bridge_t *b, const uint8_t *d, uint16_t len);
int bridge_img_config(bridge_t *b, const uint8_t *c, uint16_t len);
int bridge_img_n_if(const bridge_t *b);
uint16_t bridge_img_rd_wanted(const bridge_t *b, int i);
int bridge_img_report_desc(bridge_t *b, int i, const uint8_t *rd, uint16_t len);
int bridge_img_done(bridge_t *b, int64_t t);                   /* finalize the image; 0 = the bridge may engage */
/* device side (the PC) */
int bridge_pc_setup(bridge_t *b, const usb_setup_t *s, const uint8_t **data, uint16_t *len);
void bridge_pc_forwarded_ok(bridge_t *b, int64_t t, const usb_setup_t *s);
void bridge_pc_bus_reset(bridge_t *b, int64_t t);
void bridge_usb_error(bridge_t *b, int64_t t);                 /* a forwarded request timed out / the host port reported an error */
/* reports: copies in -> out (same length) and edits X/Y of motion reports of MOUSE_MOTION interfaces. Returns 1 if X/Y was changed. */
int bridge_mouse_in(bridge_t *b, int64_t t, uint8_t ep_addr, const uint8_t *in, uint16_t len, uint8_t *out);
int bridge_merge(bridge_t *b, uint8_t ep_addr, uint8_t *a, const uint8_t *bb, uint16_t len);
void bridge_poll(bridge_t *b, int64_t t);                      /* call every ~1 ms (timer) */
/* safety inputs */
void bridge_panic(bridge_t *b, int64_t t, int pressed);
void bridge_note_cost(bridge_t *b, int64_t t, uint32_t us);   /* time the port spent in one report path */
void bridge_fatal(bridge_t *b, int64_t t);
/* link */
void bridge_link_rx(bridge_t *b, int64_t t, const uint8_t frame[LINK_FRAME]);
void bridge_link_tx(bridge_t *b, int64_t t, uint8_t frame[LINK_FRAME]);
/* outputs */
int bridge_hw_select(const bridge_t *b);
int bridge_attach(const bridge_t *b);
int bridge_wants_kick(const bridge_t *b);
void bridge_status(const bridge_t *b, int64_t t, bridge_status_t *s);
/* validators (also used by the tests) */
int bridge_asc_params_valid(const asc_params_t *p);
int bridge_tremor_params_valid(const tremor_params_t *p);
#ifdef BRIDGE_TESTING
void bridge_param_range(int tremor, int i, int32_t *lo, int32_t *hi);   /* the accepted range of parameter i */
void bridge_debug_inject(bridge_t *b, int mode);               /* 1: the next chain output doubles (an invariant violation); 2: the chain outputs zero (a stuck chain) */
#endif
#endif
