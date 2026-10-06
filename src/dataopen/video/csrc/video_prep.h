/* Streaming frame preparation for the detector: colour conversion, crop, area downscale and letterbox, one source line at a time.
 * Pure C99, no heap, no libm, constant memory (a few KB of accumulators, never a frame buffer): the output row is finished the moment the
 * last source line it depends on has been fed in, so what remains to do after the last needed line arrives is the work of ONE line.
 * Geometry is the detector's letterbox (centred, pad value 114, scale = min(out/crop) and never above 1: no upscaling).
 * Each output pixel is the exact area average of its footprint, normalised by the weights actually accumulated, so a constant input gives
 * exactly that constant. */
#ifndef VIDEO_PREP_H
#define VIDEO_PREP_H
#include <stdint.h>

#define VP_MAX_CROP_W 4096
#define VP_MAX_OUT 1024

enum { VP_RGB24 = 0, VP_BGR24 = 1, VP_UYVY = 2, VP_YUYV = 3 };
enum { VP_BT709 = 0, VP_BT601 = 1 };
enum { VP_E_OK = 0, VP_E_ARGS = -1, VP_E_UPSCALE = -2 };

typedef struct {
    uint16_t src_w, src_h, crop_x, crop_y, crop_w, crop_h, out_w, out_h;
    uint8_t fmt;       /* VP_RGB24 .. VP_YUYV */
    uint8_t limited;   /* 1: the source is limited range (16..235; chroma 16..240): expand to full range */
    uint8_t matrix;    /* VP_BT709 / VP_BT601 (YUV only) */
    uint8_t pad;       /* letterbox fill, 114 for the detector */
} vp_cfg_t;

typedef struct {
    vp_cfg_t c;
    uint16_t content_w, content_h, px, py;      /* where the scaled crop sits inside the output */
    uint32_t step_x, step_y;                    /* source pixels per output pixel, Q16 (>= 1.0) */
    uint16_t cur_j;                             /* the earliest output row not yet written */
    uint16_t rows_done;
    uint32_t wy_sum[2];
    uint32_t wsum_x[VP_MAX_OUT];
    uint16_t cbin[VP_MAX_CROP_W];
    uint32_t cw0[VP_MAX_CROP_W], cw1[VP_MAX_CROP_W];
    uint32_t hl[VP_MAX_OUT * 3];
    uint64_t acc[2][VP_MAX_OUT * 3];
    uint8_t line[VP_MAX_CROP_W * 3];
    uint8_t *out;
} vp_t;

int vp_init(vp_t *s, const vp_cfg_t *c);              /* VP_E_OK or an error; computes the geometry and the column tables */
void vp_begin(vp_t *s, uint8_t *out_rgb);             /* out_w * out_h * 3 bytes; fills the letterbox padding */
void vp_line(vp_t *s, int y, const uint8_t *src_line);   /* feed source line y (any order is NOT supported: ascending). Lines outside the crop are ignored. */
int vp_complete(const vp_t *s);                       /* every output row written */
int vp_last_row(const vp_t *s);                       /* the last source line the crop needs */
void vp_geometry(const vp_t *s, int32_t g[8]);        /* content_w, content_h, px, py, step_x, step_y, crop_w, crop_h */
#endif
