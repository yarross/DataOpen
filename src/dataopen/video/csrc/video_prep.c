#include "video_prep.h"
#include <string.h>

static uint8_t clip8(int32_t v) { return (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v)); }

int vp_init(vp_t *s, const vp_cfg_t *c) {
    if (c->crop_w == 0 || c->crop_h == 0 || c->out_w == 0 || c->out_h == 0 || c->out_w > VP_MAX_OUT || c->out_h > VP_MAX_OUT ||
        c->crop_w > VP_MAX_CROP_W || (uint32_t)c->crop_x + c->crop_w > c->src_w || (uint32_t)c->crop_y + c->crop_h > c->src_h || c->fmt > VP_YUYV)
        return VP_E_ARGS;
    s->c = *c;
    /* r = min(out_w / crop_w, out_h / crop_h, 1); content = round(crop * r) (ties up, as round() on positives) */
    uint32_t cw = c->crop_w, ch = c->crop_h, ow = c->out_w, oh = c->out_h;
    uint32_t nw, nh;
    if (cw <= ow && ch <= oh) { nw = cw; nh = ch; }            /* never upscale: placed 1:1 */
    else if ((uint64_t)ow * ch <= (uint64_t)oh * cw) {          /* width limits */
        nw = ow; nh = (uint32_t)(((uint64_t)ch * ow * 2 + cw) / (2 * (uint64_t)cw));
    } else {
        nh = oh; nw = (uint32_t)(((uint64_t)cw * oh * 2 + ch) / (2 * (uint64_t)ch));
    }
    if (nw < 1) nw = 1;
    if (nh < 1) nh = 1;
    if (nw > ow) nw = ow;
    if (nh > oh) nh = oh;
    s->content_w = (uint16_t)nw; s->content_h = (uint16_t)nh;
    s->px = (uint16_t)((ow - nw) / 2); s->py = (uint16_t)((oh - nh) / 2);
    s->step_x = (uint32_t)(((uint64_t)cw << 16) / nw);
    s->step_y = (uint32_t)(((uint64_t)ch << 16) / nh);
    if (s->step_x < 65536 || s->step_y < 65536) return VP_E_UPSCALE;
    memset(s->wsum_x, 0, sizeof s->wsum_x);
    for (uint32_t x = 0; x < cw; x++) {
        uint64_t x0 = (uint64_t)x << 16, x1 = x0 + 65536;
        uint32_t k0 = (uint32_t)(x0 / s->step_x);
        if (k0 >= nw) k0 = nw - 1;
        uint64_t edge = (uint64_t)(k0 + 1) * s->step_x;
        s->cbin[x] = (uint16_t)k0;
        if (k0 + 1 < nw && edge < x1) {
            s->cw0[x] = (uint32_t)(edge - x0); s->cw1[x] = (uint32_t)(x1 - edge);
            s->wsum_x[k0] += s->cw0[x]; s->wsum_x[k0 + 1] += s->cw1[x];
        } else {
            s->cw0[x] = 65536; s->cw1[x] = 0;
            s->wsum_x[k0] += 65536;
        }
    }
    s->out = 0;
    return VP_E_OK;
}

void vp_begin(vp_t *s, uint8_t *out) {
    s->out = out;
    memset(out, s->c.pad, (size_t)s->c.out_w * s->c.out_h * 3);
    memset(s->acc, 0, sizeof s->acc);
    s->wy_sum[0] = s->wy_sum[1] = 0;
    s->cur_j = 0; s->rows_done = 0;
}

int vp_last_row(const vp_t *s) { return s->c.crop_y + s->c.crop_h - 1; }
int vp_complete(const vp_t *s) { return s->rows_done >= s->content_h; }

void vp_geometry(const vp_t *s, int32_t g[8]) {
    g[0] = s->content_w; g[1] = s->content_h; g[2] = s->px; g[3] = s->py; g[4] = (int32_t)s->step_x; g[5] = (int32_t)s->step_y;
    g[6] = s->c.crop_w; g[7] = s->c.crop_h;
}

/* source line -> RGB for the crop columns */
static void convert(vp_t *s, const uint8_t *src) {
    const vp_cfg_t *c = &s->c;
    uint8_t *d = s->line;
    uint32_t x0 = c->crop_x, n = c->crop_w;
    if (c->fmt == VP_RGB24 || c->fmt == VP_BGR24) {
        const uint8_t *p = src + (size_t)x0 * 3;
        for (uint32_t i = 0; i < n; i++, p += 3, d += 3) {
            uint8_t r = c->fmt == VP_RGB24 ? p[0] : p[2], g = p[1], b = c->fmt == VP_RGB24 ? p[2] : p[0];
            if (c->limited) { r = clip8((((int32_t)r - 16) * 298 + 128) >> 8); g = clip8((((int32_t)g - 16) * 298 + 128) >> 8); b = clip8((((int32_t)b - 16) * 298 + 128) >> 8); }
            d[0] = r; d[1] = g; d[2] = b;
        }
        return;
    }
    /* packed 4:2:2: 4 bytes per two pixels; chroma is replicated across the pair (what an ISP does without interpolation) */
    int yoff = c->fmt == VP_UYVY ? 1 : 0, uoff = c->fmt == VP_UYVY ? 0 : 1;      /* byte positions inside a pair of pixels: [U Y0 V Y1] or [Y0 U Y1 V] */
    int32_t kr, kgu, kgv, kb, ys;
    if (c->limited) {
        ys = 298;
        if (c->matrix == VP_BT709) { kr = 459; kgu = 55; kgv = 136; kb = 541; } else { kr = 409; kgu = 100; kgv = 208; kb = 516; }
    } else {
        ys = 256;
        if (c->matrix == VP_BT709) { kr = 403; kgu = 48; kgv = 120; kb = 475; } else { kr = 359; kgu = 88; kgv = 183; kb = 454; }
    }
    for (uint32_t i = 0; i < n; i++, d += 3) {
        uint32_t x = x0 + i;
        const uint8_t *pair = src + (size_t)(x >> 1) * 4;
        int32_t y = pair[yoff + ((x & 1) ? 2 : 0)], u = pair[uoff] - 128, v = pair[uoff + 2] - 128;
        int32_t yy = (c->limited ? y - 16 : y) * ys;
        d[0] = clip8((yy + kr * v + 128) >> 8);
        d[1] = clip8((yy - kgu * u - kgv * v + 128) >> 8);
        d[2] = clip8((yy + kb * u + 128) >> 8);
    }
}

static void emit(vp_t *s) {
    uint32_t j = s->cur_j;
    uint8_t *row = s->out + ((size_t)(s->py + j) * s->c.out_w + s->px) * 3;
    for (uint32_t ox = 0; ox < s->content_w; ox++) {
        uint64_t w = (uint64_t)s->wsum_x[ox] * s->wy_sum[0];
        for (uint32_t ch = 0; ch < 3; ch++) {
            uint64_t v = w ? (s->acc[0][ox * 3 + ch] + w / 2) / w : 0;
            row[ox * 3 + ch] = (uint8_t)(v > 255 ? 255 : v);
        }
    }
    memcpy(s->acc[0], s->acc[1], sizeof s->acc[0]);
    memset(s->acc[1], 0, sizeof s->acc[1]);
    s->wy_sum[0] = s->wy_sum[1]; s->wy_sum[1] = 0;
    s->cur_j++; s->rows_done++;
}

void vp_line(vp_t *s, int y, const uint8_t *src) {
    const vp_cfg_t *c = &s->c;
    if (y < c->crop_y || y >= c->crop_y + c->crop_h || s->out == 0 || s->rows_done >= s->content_h) return;
    convert(s, src);
    uint32_t ry = (uint32_t)(y - c->crop_y);
    /* horizontal: weighted sums per output column for this line */
    memset(s->hl, 0, (size_t)s->content_w * 3 * sizeof(uint32_t));
    const uint8_t *p = s->line;
    for (uint32_t x = 0; x < c->crop_w; x++, p += 3) {
        uint32_t k = s->cbin[x], w0 = s->cw0[x], w1 = s->cw1[x];
        s->hl[k * 3] += p[0] * w0; s->hl[k * 3 + 1] += p[1] * w0; s->hl[k * 3 + 2] += p[2] * w0;
        if (w1) { s->hl[(k + 1) * 3] += p[0] * w1; s->hl[(k + 1) * 3 + 1] += p[1] * w1; s->hl[(k + 1) * 3 + 2] += p[2] * w1; }
    }
    /* vertical: this line belongs to output row j0 (and maybe j0 + 1) */
    uint64_t y0 = (uint64_t)ry << 16, y1 = y0 + 65536;
    uint32_t j0 = (uint32_t)(y0 / s->step_y);
    if (j0 >= s->content_h) j0 = s->content_h - 1u;
    uint64_t edge = (uint64_t)(j0 + 1) * s->step_y;
    uint32_t wa = 65536, wb = 0;
    if (j0 + 1 < s->content_h && edge < y1) { wa = (uint32_t)(edge - y0); wb = (uint32_t)(y1 - edge); }
    uint32_t slot = j0 - s->cur_j;                     /* 0 or 1 */
    if (slot > 1) return;                              /* out of sequence: ignore (the caller feeds ascending lines) */
    for (uint32_t i = 0; i < (uint32_t)s->content_w * 3; i++) {
        s->acc[slot][i] += (uint64_t)s->hl[i] * wa;
        if (wb) s->acc[slot + 1][i] += (uint64_t)s->hl[i] * wb;
    }
    s->wy_sum[slot] += wa;
    if (wb) s->wy_sum[slot + 1] += wb;
    /* row cur_j is complete when its last contributing line has been fed */
    int last_line = (ry + 1 == c->crop_h);
    while (s->rows_done < s->content_h && (last_line || (uint64_t)(s->cur_j + 1) * s->step_y <= y1)) {
        if (s->wy_sum[0] == 0) break;
        emit(s);
        if (!last_line) break;
    }
}
