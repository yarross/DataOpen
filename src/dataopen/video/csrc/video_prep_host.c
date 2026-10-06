/* Host-only helpers (not part of the firmware): whole-frame calls and timing of the streaming core, for tests and `dataopen video bench`. */
#include <time.h>
#include "video_prep.h"

static uint64_t now_ns(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec; }

size_t vp_sizeof(void) { return sizeof(vp_t); }

/* Feeds every source line of a frame (stride in bytes) and reports the time of the whole frame and of the LAST needed line alone. */
int vp_frame(vp_t *s, const uint8_t *src, uint32_t stride, uint8_t *out, uint64_t *ns_total, uint64_t *ns_last) {
    uint64_t t0 = now_ns(), tl = t0;
    vp_begin(s, out);
    int last = vp_last_row(s);
    for (int y = s->c.crop_y; y <= last; y++) {
        if (y == last) tl = now_ns();
        vp_line(s, y, src + (size_t)y * stride);
    }
    uint64_t t1 = now_ns();
    if (ns_total) *ns_total = t1 - t0;
    if (ns_last) *ns_last = t1 - tl;
    return vp_complete(s) ? 0 : -1;
}
