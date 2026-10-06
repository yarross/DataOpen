#include "link.h"
#include <string.h>

uint32_t link_crc32(const uint8_t *p, uint32_t n) {
    static const uint32_t T[16] = {0x00000000u, 0x1DB71064u, 0x3B6E20C8u, 0x26D930ACu, 0x76DC4190u, 0x6B6B51F4u, 0x4DB26158u, 0x5005713Cu,
                                   0xEDB88320u, 0xF00F9344u, 0xD6D6A3E8u, 0xCB61B38Cu, 0x9B64C2B0u, 0x86D3D2D4u, 0xA00AE278u, 0xBDBDF21Cu};
    uint32_t c = 0xFFFFFFFFu;
    for (uint32_t i = 0; i < n; i++) {
        c ^= p[i];
        c = (c >> 4) ^ T[c & 15u];
        c = (c >> 4) ^ T[c & 15u];
    }
    return ~c;
}

void link_pack(uint8_t out[LINK_FRAME], const link_frame_t *f) {
    memset(out, 0, LINK_FRAME);
    out[0] = LINK_SYNC; out[1] = LINK_VER; out[2] = f->kind; out[3] = f->flags;
    out[4] = (uint8_t)f->seq; out[5] = (uint8_t)(f->seq >> 8); out[6] = (uint8_t)f->ack; out[7] = (uint8_t)(f->ack >> 8);
    uint16_t n = f->len > LINK_PAYLOAD ? LINK_PAYLOAD : f->len;
    out[8] = (uint8_t)n; out[9] = (uint8_t)(n >> 8); out[10] = f->frag_idx; out[11] = f->frag_cnt;
    memcpy(out + 12, f->payload, n);
    uint32_t c = link_crc32(out, 124);
    out[124] = (uint8_t)c; out[125] = (uint8_t)(c >> 8); out[126] = (uint8_t)(c >> 16); out[127] = (uint8_t)(c >> 24);
}

int link_unpack(const uint8_t in[LINK_FRAME], link_frame_t *f) {
    if (in[0] != LINK_SYNC) return LINK_E_SYNC;
    uint32_t c = (uint32_t)in[124] | ((uint32_t)in[125] << 8) | ((uint32_t)in[126] << 16) | ((uint32_t)in[127] << 24);
    if (c != link_crc32(in, 124)) return LINK_E_CRC;
    if (in[1] != LINK_VER) return LINK_E_VER;
    uint16_t n = (uint16_t)(in[8] | (in[9] << 8));
    if (n > LINK_PAYLOAD) return LINK_E_LEN;
    memset(f, 0, sizeof *f);
    f->kind = in[2]; f->flags = in[3]; f->seq = (uint16_t)(in[4] | (in[5] << 8)); f->ack = (uint16_t)(in[6] | (in[7] << 8));
    f->len = n; f->frag_idx = in[10]; f->frag_cnt = in[11];
    memcpy(f->payload, in + 12, n);
    return LINK_OK;
}

void link_blob_reset(link_blob_t *b) { memset(b, 0, sizeof *b); }

int link_blob_feed(link_blob_t *b, const link_frame_t *f, int64_t t_us, int64_t timeout_us) {
    uint8_t serial = f->flags & 15u;
    if (f->frag_cnt < 1 || f->frag_cnt > 3 || f->frag_idx >= f->frag_cnt) return LINK_E_FRAG;
    if (f->frag_idx + 1 < f->frag_cnt && f->len != LINK_PAYLOAD) return LINK_E_FRAG;     /* only the last fragment may be short */
    if (b->active && (serial != b->serial || f->frag_cnt != b->cnt || t_us - b->t_start > timeout_us)) link_blob_reset(b);
    if (!b->active) { link_blob_reset(b); b->active = 1; b->serial = serial; b->cnt = f->frag_cnt; b->t_start = t_us; }
    memcpy(b->buf + (uint32_t)f->frag_idx * LINK_PAYLOAD, f->payload, f->len);
    if (f->frag_idx + 1 == f->frag_cnt) b->last_len = f->len;
    b->mask |= (uint8_t)(1u << f->frag_idx);
    if (b->mask != (uint8_t)((1u << b->cnt) - 1u)) return 0;
    b->active = 0;                                   /* complete: the bytes stay in buf until the next fragment arrives */
    b->mask = 0;
    return (b->cnt - 1) * LINK_PAYLOAD + b->last_len;
}
