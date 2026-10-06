#include "failsafe.h"
#include <string.h>

void fs_cfg_defaults(fs_cfg_t *c) {
    c->panic_long_ms = 3000; c->rearm_ms = 2000; c->engage_hold_ms = 2000; c->probe_ms = 3000; c->pc_cfg_ms = 3000; c->retry_ms = 2000;
    c->soft_hold_ms = 5000; c->auto_engage = 1; c->max_attempts = 3; c->crash_limit = 3; c->clean_ms = 60000;
}

void fs_init(fs_t *f, const fs_cfg_t *c) {
    memset(f, 0, sizeof *f);
    f->c = *c;
    f->hw = HW_SEL_BYPASS; f->state = FS_HW_BYPASS; f->reason = FSR_POWER_ON; f->cmd_assist = 1; f->healthy = 1;
}

void fs_boot(fs_t *f, int64_t t, int cause, uint16_t crashes) {
    (void)t;
    f->crashes = crashes;
    if (cause == RESET_WATCHDOG || cause == RESET_FAULT) { f->crashes++; f->nv_dirty = 1; }
    if (f->crashes >= f->c.crash_limit) f->latch_hw = FSR_CRASHLOOP;
}

static void to_bypass(fs_t *f, int64_t t, int reason, int latch, int retry) {
    f->hw = HW_SEL_BYPASS;
    f->img_ok = f->pc_cfg = 0;
    f->was_engaged = 0;
    if (latch) f->latch_hw = (uint8_t)reason;
    if (retry) f->t_retry_at = t + f->c.retry_ms;
    f->reason = (uint8_t)reason;
}

void fs_device(fs_t *f, int64_t t, int present) {
    if (present == f->present) return;
    f->present = (uint8_t)present;
    if (present) { f->t_present = t; return; }
    /* a different mouse is a new story: forget attempts and device-specific latches (never the panic/crashloop ones) */
    f->attempts = 0;
    if (f->latch_hw == FSR_IMAGE || f->latch_hw == FSR_ENGAGE_FAILED || f->latch_hw == FSR_USB_ERRORS) f->latch_hw = 0;
    if (f->hw != HW_SEL_BYPASS) to_bypass(f, t, FSR_NO_DEVICE, 0, 0);
    f->reason = FSR_NO_DEVICE;
}

void fs_image(fs_t *f, int64_t t, int code) {
    if (f->hw != HW_SEL_PROBE) return;
    if (code != 0) { to_bypass(f, t, FSR_IMAGE, 1, 0); return; }
    f->img_ok = 1;
}

void fs_pc_configured(fs_t *f, int64_t t) {
    (void)t;
    if (f->hw == HW_SEL_ENGAGED) { f->pc_cfg = 1; f->t_engaged = t; }
}

void fs_pc_reset(fs_t *f, int64_t t) {
    (void)t;
    f->pc_cfg = 0;                      /* the PC will configure again; we stay attached and wait */
    if (f->hw == HW_SEL_ENGAGED) f->t_attach = t;
}

static void rearm(fs_t *f, int64_t t) {
    f->latch_soft = 0; f->latch_hw = 0; f->attempts = 0; f->t_retry_at = t; f->t_hold_until = 0;
    f->crashes = 0; f->nv_dirty = 1; f->healthy = 1;
}

void fs_panic(fs_t *f, int64_t t, int pressed) {
    if (pressed && !f->panic_down) {
        f->panic_down = 1; f->t_panic_down = t; f->press_done = 0;
        f->press_started_latched = (uint8_t)(f->latch_soft || f->latch_hw);
        if (!f->press_started_latched) { f->latch_soft = 1; f->latch_soft_reason = FSR_PANIC; }   /* instant, same call */
    } else if (!pressed) {
        f->panic_down = 0;
    }
}

void fs_chord(fs_t *f, int64_t t) {
    (void)t;
    if (f->latch_hw) return;           /* a hard latch needs the physical button */
    if (f->latch_soft) f->latch_soft = 0;
    else { f->latch_soft = 1; f->latch_soft_reason = FSR_CHORD; }
}

void fs_cmd(fs_t *f, int64_t t, int cmd) {
    if (cmd == 0) f->cmd_assist = 0;
    else if (cmd == 1) f->cmd_assist = 1;           /* raising is only a request: latches and faults still win */
    else if (cmd == 2) to_bypass(f, t, FSR_CMD_BYPASS, 1, 0);
}

void fs_fault(fs_t *f, int64_t t, int fault) {
    switch (fault) {
    case FSF_FATAL: f->healthy = 0; to_bypass(f, t, FSR_FATAL, 1, 0); break;
    case FSF_USB_ERRORS: if (f->hw != HW_SEL_BYPASS) { f->attempts++; to_bypass(f, t, FSR_USB_ERRORS, 0, 1); } break;
    case FSF_INVARIANT: f->t_hold_until = t + (int64_t)f->c.soft_hold_ms * 1000; f->hold_reason = FSR_INVARIANT; break;
    case FSF_OVERRUN: f->t_hold_until = t + (int64_t)f->c.soft_hold_ms * 1000; f->hold_reason = FSR_OVERRUN; break;
    default: break;
    }
}

void fs_poll(fs_t *f, int64_t t, uint32_t conds) {
    if (f->panic_down) {
        int64_t held = (t - f->t_panic_down) / 1000;
        if (!f->press_started_latched && !f->latch_hw && held >= f->c.panic_long_ms) {
            to_bypass(f, t, FSR_PANIC_LONG, 1, 0);
            f->press_started_latched = 1; f->press_done = 1;          /* the long press has done its job; it does not also rearm */
        } else if (f->press_started_latched && !f->press_done && held >= f->c.rearm_ms) {
            if (f->latch_hw != FSR_CRASHLOOP || held >= 5000) { rearm(f, t); f->press_done = 1; }
        }
    }
    if (f->attempts >= f->c.max_attempts && !f->latch_hw && f->hw == HW_SEL_BYPASS) f->latch_hw = FSR_ENGAGE_FAILED;
    switch (f->hw) {
    case HW_SEL_BYPASS:
        if (f->present && !f->latch_hw && f->c.auto_engage && t >= f->t_retry_at && t - f->t_present >= (int64_t)f->c.engage_hold_ms * 1000) {
            f->hw = HW_SEL_PROBE; f->t_probe = t; f->img_ok = f->pc_cfg = 0;
        }
        break;
    case HW_SEL_PROBE:
        if (f->img_ok) { f->hw = HW_SEL_ENGAGED; f->t_attach = t; f->pc_cfg = 0; }
        else if (t - f->t_probe > (int64_t)f->c.probe_ms * 1000) { f->attempts++; to_bypass(f, t, FSR_IMAGE_TIMEOUT, 0, 1); }
        break;
    default:
        if (!f->pc_cfg && t - f->t_attach > (int64_t)f->c.pc_cfg_ms * 1000) { f->attempts++; to_bypass(f, t, FSR_PC_TIMEOUT, 0, 1); }
        if (f->pc_cfg && !f->was_engaged) { f->was_engaged = 1; f->t_engaged = t; }
        if (f->pc_cfg && f->crashes && t - f->t_engaged > (int64_t)f->c.clean_ms * 1000) { f->crashes = 0; f->nv_dirty = 1; }
        break;
    }
    /* derive the visible state */
    if (f->hw == HW_SEL_BYPASS) {
        f->state = FS_HW_BYPASS;
        if (f->latch_hw) f->reason = f->latch_hw;
        else if (!f->present) f->reason = FSR_NO_DEVICE;
        else if (f->reason != FSR_PC_TIMEOUT && f->reason != FSR_IMAGE_TIMEOUT && f->reason != FSR_USB_ERRORS) f->reason = FSR_SETTLING;
    } else if (f->hw == HW_SEL_PROBE) {
        f->state = FS_PROBE; f->reason = FSR_PROBING;
    } else if (!f->pc_cfg) {
        f->state = FS_PASSTHRU; f->reason = FSR_WAIT_PC;
    } else if (f->latch_soft) {
        f->state = FS_PASSTHRU; f->reason = f->latch_soft_reason;
    } else if (f->t_hold_until > t) {
        f->state = FS_PASSTHRU; f->reason = f->hold_reason;
    } else if (!f->cmd_assist) {
        f->state = FS_PASSTHRU; f->reason = FSR_CMD_PASSTHRU;
    } else if (conds & FSC_STALE_LINK) {
        f->state = FS_PASSTHRU; f->reason = FSR_STALE_LINK;
    } else if (conds & FSC_STALE_PARAMS) {
        f->state = FS_PASSTHRU; f->reason = FSR_STALE_PARAMS;
    } else if (conds & FSC_SLOW) {
        f->state = FS_PASSTHRU; f->reason = FSR_SLOW_MOUSE;
    } else {
        f->state = FS_ASSIST; f->reason = FSR_NONE;
    }
}

int fs_wants_kick(const fs_t *f) { return f->healthy; }
int fs_hw_select(const fs_t *f) { return f->healthy ? f->hw : HW_SEL_BYPASS; }
int fs_attach(const fs_t *f) { return f->hw == HW_SEL_ENGAGED && f->healthy; }
