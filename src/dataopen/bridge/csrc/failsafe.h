/* Fail-safe state machine of the assistive HID bridge. Principle: leaving the help is easy and fast, entering it is deliberate.
 * Nothing here depends on the compute module. Pure C99, sans-IO: events and a poll in, state out.
 *
 *   HW_BYPASS  the mouse is wired straight to the PC by the analog switch (no firmware involved)
 *   PROBE      the mouse sits on the bridge's host port, the bridge's device port is NOT attached to the PC yet
 *   PASSTHRU   the proxy runs, X/Y is not touched (out == in bit for bit)
 *   ASSIST     the proxy runs and the correction chain edits X/Y
 * Soft exits (panic press, mouse chord, module command, stale data, invariant/overrun trouble) go to PASSTHRU within the same call:
 * no USB re-enumeration, no gap in the pointer. Hard exits (firmware fault, unsupported device, USB errors, long press, power loss)
 * go to HW_BYPASS: the PC enumerates the real mouse again (0.2 - 2 s without input). */
#ifndef FAILSAFE_H
#define FAILSAFE_H
#include <stdint.h>

enum { FS_HW_BYPASS = 0, FS_PROBE = 1, FS_PASSTHRU = 2, FS_ASSIST = 3 };
enum { HW_SEL_BYPASS = 0, HW_SEL_PROBE = 1, HW_SEL_ENGAGED = 2 };
enum { FSR_NONE = 0, FSR_POWER_ON, FSR_NO_DEVICE, FSR_SETTLING, FSR_PROBING, FSR_IMAGE, FSR_PC_TIMEOUT, FSR_USB_ERRORS, FSR_PANIC,
       FSR_PANIC_LONG, FSR_CHORD, FSR_CMD_PASSTHRU, FSR_CMD_BYPASS, FSR_STALE_PARAMS, FSR_STALE_LINK, FSR_INVARIANT, FSR_OVERRUN,
       FSR_SLOW_MOUSE, FSR_CRASHLOOP, FSR_FATAL, FSR_ENGAGE_FAILED, FSR_WAIT_PC, FSR_IMAGE_TIMEOUT, FSR_COUNT };
enum { FSC_STALE_PARAMS = 1, FSC_STALE_LINK = 2, FSC_SLOW = 4 };           /* conditions from the bridge */
enum { RESET_POWER = 0, RESET_WATCHDOG = 1, RESET_FAULT = 2, RESET_SOFT = 3 };
enum { FSF_FATAL = 1, FSF_USB_ERRORS = 2, FSF_INVARIANT = 3, FSF_OVERRUN = 4 };

typedef struct {
    int32_t panic_long_ms, rearm_ms, engage_hold_ms, probe_ms, pc_cfg_ms, retry_ms, soft_hold_ms, auto_engage, max_attempts, crash_limit,
        clean_ms;
} fs_cfg_t;

typedef struct {
    fs_cfg_t c;
    uint8_t hw, state, reason, present, img_ok, pc_cfg, attempts, latch_soft, latch_soft_reason, cmd_assist, latch_hw, panic_down,
        press_started_latched, press_done, healthy, nv_dirty, hold_reason, was_engaged;
    uint16_t crashes;
    int64_t t_present, t_probe, t_attach, t_retry_at, t_hold_until, t_panic_down, t_engaged;
} fs_t;

void fs_cfg_defaults(fs_cfg_t *c);
void fs_init(fs_t *f, const fs_cfg_t *c);
/* Power-up. `crashes` is the persisted counter; a watchdog/fault reset adds one. */
void fs_boot(fs_t *f, int64_t t, int reset_cause, uint16_t crashes);
void fs_device(fs_t *f, int64_t t, int present);          /* a mouse appeared / disappeared at the bridge's host side */
void fs_image(fs_t *f, int64_t t, int code);             /* result of building the USB image: 0 ok, else not supported */
void fs_pc_configured(fs_t *f, int64_t t);               /* the PC sent SET_CONFIGURATION to the bridge's device port */
void fs_pc_reset(fs_t *f, int64_t t);                    /* bus reset seen by the bridge's device port */
void fs_panic(fs_t *f, int64_t t, int pressed);          /* the Panic input (level) */
void fs_chord(fs_t *f, int64_t t);                       /* the mouse-button chord was held long enough: toggles the soft latch */
void fs_cmd(fs_t *f, int64_t t, int cmd);                /* 0 passthru, 1 assist, 2 bypass (the module may only ask for less help) */
void fs_fault(fs_t *f, int64_t t, int fault);
void fs_poll(fs_t *f, int64_t t, uint32_t conds);
int fs_wants_kick(const fs_t *f);                        /* may the main loop kick the external watchdog? */
int fs_hw_select(const fs_t *f);                         /* HW_SEL_*: what the firmware asks of the analog switch */
int fs_attach(const fs_t *f);                            /* pull up the device port's D+/D-? */
#endif
