/* Timing of the bridge core on the host: nanoseconds per mouse report through bridge_mouse_in() in each mode, and per link frame.
 * Built by `dataopen bridge bench` together with seeds.h (not part of the shared library). Host numbers, not Cortex-M numbers. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "bridge.h"
#include "seeds.h"

static uint64_t rng_s = 88172645463325252ull;
static uint32_t rnd(void) { rng_s ^= rng_s << 13; rng_s ^= rng_s >> 7; rng_s ^= rng_s << 17; return (uint32_t)(rng_s >> 11); }
static double now_ns(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec; }
static void put32(uint8_t *p, uint32_t v) { p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); p[2] = (uint8_t)(v >> 16); p[3] = (uint8_t)(v >> 24); }
static void send_frame(bridge_t *b, int64_t t, link_frame_t *f) { uint8_t raw[LINK_FRAME]; link_pack(raw, f); bridge_link_rx(b, t, raw); }

static void send_blob(bridge_t *b, int64_t t, uint8_t kind, const uint8_t *blob, int n) {
    int frags = (n + LINK_PAYLOAD - 1) / LINK_PAYLOAD;
    for (int i = 0; i < frags; i++) {
        link_frame_t f; memset(&f, 0, sizeof f);
        f.kind = kind; f.flags = 1; f.frag_idx = (uint8_t)i; f.frag_cnt = (uint8_t)frags;
        f.len = (uint16_t)(n - i * LINK_PAYLOAD > LINK_PAYLOAD ? LINK_PAYLOAD : n - i * LINK_PAYLOAD);
        memcpy(f.payload, blob + i * LINK_PAYLOAD, f.len);
        send_frame(b, t, &f);
    }
}

static void put_xy(uint8_t *rep, int32_t dx, int32_t dy) {
    hid_motion_t m; memset(&m, 0, sizeof m);
    m.size_x = m.size_y = 12; m.off_x = 24; m.off_y = 36; m.len = 8;
    memset(rep, 0, 8); rep[0] = 1;
    hid_xy_set(&m, rep, dx, dy);
}

static void setup(bridge_t *b, int64_t *t, int speed, int panic) {
    bridge_cfg_t c;
    bridge_cfg_defaults(&c);
    bridge_init(b, &c);
    bridge_set_speed(b, speed);
    bridge_boot(b, *t, RESET_POWER, 0);
    bridge_dev_present(b, *t, 1);
    for (int i = 0; i < 4000 && b->fs.hw != HW_SEL_PROBE; i++) { *t += 1000; bridge_poll(b, *t); }
    bridge_img_device(b, SEED_DEV, 18);
    bridge_img_config(b, SEED_CFG, sizeof SEED_CFG);
    for (int i = 0; i < bridge_img_n_if(b); i++) if (bridge_img_rd_wanted(b, i)) bridge_img_report_desc(b, i, SEED_RD_LOGI, sizeof SEED_RD_LOGI);
    bridge_img_done(b, *t);
    usb_setup_t s = {0x00, 9, 1, 0, 0};
    bridge_pc_forwarded_ok(b, *t, &s);
    send_blob(b, *t, LK_PARAMS_ASC, SEED_ASC_BLOB, ASC_BLOB_LEN);
    send_blob(b, *t, LK_PARAMS_TREMOR, SEED_TRM_BLOB, TRM_BLOB_LEN);
    link_frame_t f; memset(&f, 0, sizeof f); f.kind = LK_HELLO; send_frame(b, *t, &f);
    if (panic) bridge_panic(b, *t, 1);
}

static void scene(bridge_t *b, int64_t t) {
    link_frame_t f; memset(&f, 0, sizeof f);
    f.kind = LK_SCENE; f.len = 28;
    put32(f.payload, (uint32_t)t); f.payload[4] = 1;
    uint8_t *q = f.payload + 8;
    q[0] = 1; put32(q + 4, (uint32_t)(300 * 65536)); put32(q + 8, (uint32_t)(40 * 65536)); put32(q + 12, (uint32_t)(30 * 65536));
    send_frame(b, t, &f);
}

static int cmp_d(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return x < y ? -1 : x > y; }

static void keepalive(bridge_t *b, int64_t t, long i, int sub) {
    link_frame_t f; memset(&f, 0, sizeof f); f.kind = LK_HELLO;
    if (i % (100L * sub) == 0) send_frame(b, t, &f);                       /* the module's heartbeat (every 100 ms) and parameter refresh (every 1 s) */
    if (i % (1000L * sub) == 0) { send_blob(b, t, LK_PARAMS_ASC, SEED_ASC_BLOB, ASC_BLOB_LEN); send_blob(b, t, LK_PARAMS_TREMOR, SEED_TRM_BLOB, TRM_BLOB_LEN); }
}

static void bench(const char *name, int speed, int panic, long n) {
    static bridge_t b;
    int64_t t = 1000000;
    setup(&b, &t, speed, panic);
    uint8_t rep[8], out[8];
    double sum = 0;
    double *all = malloc(sizeof(double) * (size_t)n);
    long calls = 0;
    int sub = speed == BR_SPEED_HS ? 8 : 1;
    for (long i = 0; i < n; i++) {
        t += 1000 / sub;
        keepalive(&b, t, i, sub);
        if (i % (33L * sub) == 0) scene(&b, t);
        int32_t dx = (int32_t)(rnd() % 21) - 8, dy = (int32_t)(rnd() % 11) - 5;
        put_xy(rep, dx, dy);
        double t0 = now_ns();
        bridge_poll(&b, t);
        bridge_mouse_in(&b, t, 0x81, rep, 8, out);
        double d = now_ns() - t0;
        sum += d; all[calls++] = d;
    }
    qsort(all, (size_t)calls, sizeof(double), cmp_d);
    bridge_status_t s; bridge_status(&b, t, &s);
    printf("%-30s mean %6.1f  p50 %6.1f  p99 %7.1f  p99.9 %8.1f  max %9.0f ns/report   state=%d mode=%d\n", name, sum / (double)calls, all[calls / 2],
           all[calls * 99 / 100], all[calls * 999 / 1000], all[calls - 1], s.state, s.mode);
    free(all);
}

int main(int argc, char **argv) {
    long n = argc > 1 ? atol(argv[1]) : 300000;
    bench("ASSIST, 1 kHz mouse (direct)", BR_SPEED_FS, 0, n);
    bench("ASSIST, 8 kHz mouse (split)", BR_SPEED_HS, 0, n);
    bench("PASSTHRU (panic latched)", BR_SPEED_FS, 1, n);
    {
        static bridge_t b; int64_t t = 1000000; setup(&b, &t, BR_SPEED_FS, 0);
        bridge_status_t s; bridge_status(&b, t, &s);
        if (s.state != FS_ASSIST) { fprintf(stderr, "bench setup did not reach ASSIST (state %d reason %d)\n", s.state, s.reason); return 1; }
    }
    {
        static bridge_t b; int64_t t = 1000000; setup(&b, &t, BR_SPEED_FS, 0);
        link_frame_t f; memset(&f, 0, sizeof f); f.kind = LK_HELLO;
        uint8_t raw[LINK_FRAME]; link_pack(raw, &f);
        double t0 = now_ns();
        for (long i = 0; i < n; i++) { t += 5000; bridge_link_rx(&b, t, raw); }
        printf("%-34s %8.1f ns/frame\n", "link frame (CRC32 + dispatch)", (now_ns() - t0) / (double)n);
        uint8_t tx[LINK_FRAME];
        t0 = now_ns();
        for (long i = 0; i < n; i++) { t += 5000; bridge_link_tx(&b, t, tx); }
        printf("%-34s %8.1f ns/frame\n", "link frame out (CRC32)", (now_ns() - t0) / (double)n);
    }
    printf("sizeof(bridge_t) = %zu bytes (RAM of one bridge instance)\n", bridge_sizeof());
    return 0;
}
