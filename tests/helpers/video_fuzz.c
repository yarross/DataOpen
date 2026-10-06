/* Random configurations and random lines through the streaming frame-preparation core, under ASan + UBSan. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "video_prep.h"

static uint64_t s_ = 88172645463325252ull;
static uint32_t rnd(void) { s_ ^= s_ << 13; s_ ^= s_ >> 7; s_ ^= s_ << 17; return (uint32_t)(s_ >> 11); }
static void fail(const char *m) { fprintf(stderr, "FUZZ FAIL: %s\n", m); abort(); }

int main(int argc, char **argv) {
    long n = argc > 1 ? atol(argv[1]) : 300;
    if (argc > 2) s_ += (uint64_t)atoll(argv[2]) * 2654435761u;
    static vp_t vp;
    static uint8_t src[4200 * 4 * 8], out[VP_MAX_OUT * VP_MAX_OUT * 3];
    long ok = 0, rejected = 0;
    for (long it = 0; it < n; it++) {
        vp_cfg_t c;
        c.src_w = (uint16_t)(1 + rnd() % 4096); c.src_h = (uint16_t)(1 + rnd() % 300);
        c.crop_w = (uint16_t)(1 + rnd() % (rnd() % 8 == 0 ? 5000 : c.src_w)); c.crop_h = (uint16_t)(1 + rnd() % (c.src_h + (rnd() % 8 == 0 ? 50 : 0)));
        c.crop_x = (uint16_t)(rnd() % (c.src_w + 3)); c.crop_y = (uint16_t)(rnd() % (c.src_h + 3));
        c.out_w = (uint16_t)(1 + rnd() % (rnd() % 10 == 0 ? 1200 : 700)); c.out_h = (uint16_t)(1 + rnd() % (rnd() % 10 == 0 ? 1200 : 700));
        c.fmt = (uint8_t)(rnd() % 5); c.limited = (uint8_t)(rnd() & 1); c.matrix = (uint8_t)(rnd() & 1); c.pad = 114;
        int r = vp_init(&vp, &c);
        if (r != VP_E_OK) { rejected++; continue; }
        if (c.fmt > VP_YUYV) fail("accepted a bad format");
        for (size_t i = 0; i < sizeof src; i++) src[i] = (uint8_t)rnd();
        vp_begin(&vp, out);
        size_t bpp = c.fmt <= VP_BGR24 ? 3 : 2;
        for (int y = 0; y < c.src_h; y++) {
            vp_line(&vp, y, src + (size_t)(rnd() % 4) * 100 * 0 + 0);   /* every line is a full-width buffer: src_w * bpp <= 4096*3 */
            (void)bpp;
        }
        if (!vp_complete(&vp)) fail("incomplete after all lines");
        int32_t g[8]; vp_geometry(&vp, g);
        if (g[0] < 1 || g[0] > c.out_w || g[1] < 1 || g[1] > c.out_h || g[2] + g[0] > c.out_w || g[3] + g[1] > c.out_h) fail("geometry outside the output");
        ok++;
    }
    printf("fuzz ok (%ld configs run, %ld rejected)\n", ok, rejected);
    return 0;
}
