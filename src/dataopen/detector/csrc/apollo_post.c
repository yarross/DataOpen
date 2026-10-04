/* Host post-processing for the NPU: dense per-level outputs -> KeypointArray (no NMS). Plain C99, no dependencies.
 * Port of dataopen/detector/postprocess.py (decode_dense) + structs.py (pack); a test compiles this file and checks it
 * against the Python reference on random tensors. Threading and the NPU runtime calls are the integrator's business.
 *
 *   level tensor l: NCHW float32, channels [cls n_cls | box 4 | kp offsets 2K | kp score K | kp vis K], batch 1
 *   refine tensor : [heatmap logits P | offset x P | offset y P] at refine_stride, or NULL
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "apollo_detection.h"

typedef struct {
    int n_cls, n_kpt, n_levels;
    int strides[4];
    float offset_scale;
    int refine_stride, n_primary, primary[4];
    int max_det;
    float conf_thr, refine_radius_px;
} ApolloCfg;

typedef struct { float score; int cls, level, idx; } Cand;

static float sigm(float x) { if (x > 30.f) x = 30.f; if (x < -30.f) x = -30.f; return 1.f / (1.f + expf(-x)); }
static float clampf(float x, float lo, float hi) { return x < lo ? lo : (x > hi ? hi : x); }
static int cmp_cand(const void* a, const void* b) {
    const Cand *x = a, *y = b;
    if (x->score != y->score) return x->score < y->score ? 1 : -1;
    if (x->level != y->level) return x->level - y->level;       /* stable: earlier anchor first, like numpy's stable argsort */
    return x->idx - y->idx;
}
static int16_t q16(float v) { long r = lroundf(v * 16.f); return (int16_t)(r < -32768 ? -32768 : (r > 32767 ? 32767 : r)); }
static uint16_t qu16(float v) { long r = lroundf(v * 16.f); return (uint16_t)(r < 0 ? 0 : (r > 65535 ? 65535 : r)); }
static uint8_t q8(float v) { long r = lroundf(v * 255.f); return (uint8_t)(r < 0 ? 0 : (r > 255 ? 255 : r)); }

static void refine_aim(float* kx, float* ky, const float* refine, int rh, int rw, const ApolloCfg* c) {
    int P = c->n_primary, plane = rh * rw;
    for (int pi = 0; pi < P; pi++) {
        int kp = c->primary[pi];
        float x = kx[kp], y = ky[kp];
        int cx = (int)(x / c->refine_stride), cy = (int)(y / c->refine_stride);
        int r = (int)ceilf(c->refine_radius_px / c->refine_stride);
        if (r < 1) r = 1;
        int x0 = cx - r < 0 ? 0 : cx - r, x1 = cx + r + 1 > rw ? rw : cx + r + 1;
        int y0 = cy - r < 0 ? 0 : cy - r, y1 = cy + r + 1 > rh ? rh : cy + r + 1;
        if (x1 <= x0 || y1 <= y0) continue;
        const float *hm = refine + pi * plane, *ox = refine + (P + pi) * plane, *oy = refine + (2 * P + pi) * plane;
        float mx = -1e30f;
        for (int yy = y0; yy < y1; yy++) for (int xx = x0; xx < x1; xx++) if (hm[yy * rw + xx] > mx) mx = hm[yy * rw + xx];
        float sum = 0.f, px = 0.f, py = 0.f;
        for (int yy = y0; yy < y1; yy++) for (int xx = x0; xx < x1; xx++) {
            float e = expf(hm[yy * rw + xx] - mx);
            sum += e;
            px += e * ((xx + 0.5f + ox[yy * rw + xx]) * c->refine_stride);
            py += e * ((yy + 0.5f + oy[yy * rw + xx]) * c->refine_stride);
        }
        px /= sum; py /= sum;
        if (hypotf(px - x, py - y) <= c->refine_radius_px) { kx[kp] = px; ky[kp] = py; }
    }
}

/* Returns the number of detections written (<= max_det), or -1 on bad arguments. `out` is zeroed first. */
int apollo_decode(const ApolloCfg* c, const float* const* levels, const int* H, const int* W, const float* refine, int rh, int rw,
                  KeypointArray* out) {
    if (!c || !levels || !out || c->n_kpt != APOLLO_NUM_KEYPOINTS || c->max_det > APOLLO_MAX_DETECTIONS) return -1;
    const int K = c->n_kpt, NC = c->n_cls, C = NC + 4 + 4 * K;
    size_t total = 0;
    for (int l = 0; l < c->n_levels; l++) total += (size_t)H[l] * W[l];
    Cand* cand = malloc(total * sizeof(Cand));
    if (!cand) return -1;
    size_t n = 0;
    for (int l = 0; l < c->n_levels; l++) {
        int plane = H[l] * W[l];
        for (int i = 0; i < plane; i++) {
            float best = -1.f; int bc = 0;
            for (int k = 0; k < NC; k++) {
                float s = sigm(levels[l][(size_t)k * plane + i]);
                if (s > best) { best = s; bc = k; }
            }
            if (best >= c->conf_thr) { cand[n].score = best; cand[n].cls = bc; cand[n].level = l; cand[n].idx = i; n++; }
        }
    }
    qsort(cand, n, sizeof(Cand), cmp_cand);
    memset(out, 0, sizeof(*out));
    out->magic = APOLLO_MAGIC; out->version = APOLLO_VERSION; out->n_keypoints = (uint8_t)K; out->max_detections = APOLLO_MAX_DETECTIONS;
    int count = (int)(n < (size_t)c->max_det ? n : (size_t)c->max_det);
    for (int d = 0; d < count; d++) {
        const Cand* a = &cand[d];
        const int l = a->level, plane = H[l] * W[l], i = a->idx, s = c->strides[l];
        const float* t = levels[l];
        float ax = (float)(i % W[l] + 0.5f) * s, ay = (float)(i / W[l] + 0.5f) * s;
        float bl = expf(clampf(t[(size_t)(NC + 0) * plane + i], -6.f, 6.f)) * s, bt = expf(clampf(t[(size_t)(NC + 1) * plane + i], -6.f, 6.f)) * s;
        float br = expf(clampf(t[(size_t)(NC + 2) * plane + i], -6.f, 6.f)) * s, bb = expf(clampf(t[(size_t)(NC + 3) * plane + i], -6.f, 6.f)) * s;
        float kx[APOLLO_NUM_KEYPOINTS], ky[APOLLO_NUM_KEYPOINTS];
        for (int k = 0; k < K; k++) {
            kx[k] = ax + t[(size_t)(NC + 4 + 2 * k) * plane + i] * c->offset_scale * s;
            ky[k] = ay + t[(size_t)(NC + 4 + 2 * k + 1) * plane + i] * c->offset_scale * s;
        }
        if (refine && c->refine_stride && c->n_primary) refine_aim(kx, ky, refine, rh, rw, c);
        ApolloDetection* o = &out->detections[d];
        o->class_id = (uint8_t)a->cls;
        o->confidence = q8(a->score);
        o->bbox[0] = qu16(fmaxf(ax - bl, 0.f)); o->bbox[1] = qu16(fmaxf(ay - bt, 0.f));
        o->bbox[2] = qu16(fmaxf((ax + br) - (ax - bl), 0.f)); o->bbox[3] = qu16(fmaxf((ay + bb) - (ay - bt), 0.f));
        uint16_t mask = 0;
        for (int k = 0; k < K; k++) {
            o->keypoints[k][0] = q16(kx[k]); o->keypoints[k][1] = q16(ky[k]);
            o->kp_confidence[k] = q8(sigm(t[(size_t)(NC + 4 + 2 * K + k) * plane + i]));
            if (sigm(t[(size_t)(NC + 4 + 3 * K + k) * plane + i]) > 0.5f) mask |= (uint16_t)(1u << k);
        }
        o->vis_mask = mask;
    }
    out->detection_count = (uint8_t)count;
    free(cand);
    (void)C;
    return count;
}
