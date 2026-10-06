/* HID report descriptor reader and in-place X/Y editor for the assistive HID bridge.
 * Pure C99, no heap, no libm. Everything is bounded by HID_DESC_MAX and the table sizes below; arbitrary (hostile, truncated,
 * random) descriptor bytes can only produce an error code or a self-consistent map, never an out-of-bounds access or a long loop.
 * The editor touches exactly the bits of the X and Y fields of a report: every other bit (buttons, wheel, vendor fields,
 * the report id) is left as it was. */
#ifndef HID_DESC_H
#define HID_DESC_H
#include <stdint.h>

#define HID_DESC_MAX 1024
#define HID_MAX_RIDS 8       /* distinct report ids with a motion field */
#define HID_MAX_SLOTS 16     /* distinct input report ids seen */
#define HID_MAX_APPS 8

enum { HID_OK = 0, HID_E_BAD = -1, HID_E_NOXY = -2, HID_E_TOO_BIG = -3, HID_E_LIMIT = -4 };

/* top-level application collection kinds (bit set in hid_map_t.kinds) */
enum { HID_K_MOUSE = 1, HID_K_KEYBOARD = 2, HID_K_CONSUMER = 4, HID_K_VENDOR = 8, HID_K_DIGITIZER = 16, HID_K_OTHER = 32 };

typedef struct {
    uint8_t report_id;       /* 0: the interface has no report ids */
    uint8_t size_x, size_y;  /* bits, 1..16 */
    uint16_t off_x, off_y;   /* bit offset from the first byte of the report (the id byte counts) */
    uint16_t len;            /* bytes of the whole input report, id included */
    uint16_t btn_off;        /* first button bit (valid if btn_n > 0) */
    uint8_t btn_n;           /* number of button bits in the first button field (0..32) */
    uint8_t _pad;
} hid_motion_t;

typedef struct {
    hid_motion_t m[HID_MAX_RIDS];
    uint8_t n;               /* motion reports found */
    uint8_t uses_ids;        /* 1: every input report starts with its id byte */
    uint8_t kinds;           /* bit set of HID_K_* over the top-level application collections */
    uint8_t n_app;
    uint32_t app[HID_MAX_APPS];   /* (usage page << 16) | usage of each top-level application collection */
} hid_map_t;

/* Parses the descriptor. Returns HID_OK, or HID_E_NOXY when it parsed but has no relative X/Y mouse report (map still lists the
 * application collections so the caller can classify the interface), or another negative code. */
int hid_parse(const uint8_t *desc, uint16_t len, hid_map_t *map);
/* The fixed boot-protocol mouse layout: byte 0 buttons, 1 = X (i8), 2 = Y (i8). */
void hid_boot_map(hid_map_t *map);
/* Finds the motion entry that a report belongs to (by id, or the only one if there are no ids); NULL if it is not a (long enough)
 * motion report. */
const hid_motion_t *hid_find(const hid_map_t *map, const uint8_t *report, uint16_t len);
void hid_xy_get(const hid_motion_t *m, const uint8_t *report, int32_t *dx, int32_t *dy);
/* Writes dx, dy into the X/Y bits only. Values are clamped to the field range. */
void hid_xy_set(const hid_motion_t *m, uint8_t *report, int32_t dx, int32_t dy);
/* The button bits of the report (up to 32), 0 if the report has none. */
uint32_t hid_buttons(const hid_motion_t *m, const uint8_t *report, uint16_t len);
/* Bit masks of the X and Y fields over a report buffer of `len` bytes (for tests and for the merge below). */
void hid_xy_mask(const hid_motion_t *m, uint8_t *mask, uint16_t len);
/* Adds the motion of report b to report a if everything outside X/Y is identical and the sums fit the fields. Returns 1 if merged. */
int hid_merge(const hid_motion_t *m, uint8_t *a, const uint8_t *b, uint16_t len);
#endif
