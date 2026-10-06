#include "hid_desc.h"
#include <string.h>

#define MAX_USAGES 32
#define MAX_COLL 8
#define MAX_STACK 4

typedef struct { uint32_t page; int32_t lmin, lmax; uint32_t rsize, rcount; uint8_t rid; } glob_t;

typedef struct {
    uint32_t bits;                       /* running bit offset of this input report */
    uint8_t rid, used;
    uint8_t have_x, have_y, bad_xy;      /* bad_xy: X or Y exists but is absolute / unsigned / too wide */
    uint8_t size_x, size_y;
    uint16_t off_x, off_y, btn_off;
    uint8_t btn_n;
} slot_t;

static int32_t rd_s(const uint8_t *p, int n) {
    uint32_t v = 0;
    for (int i = 0; i < n; i++) v |= (uint32_t)p[i] << (8 * i);
    if (n == 1) return (int8_t)v;
    if (n == 2) return (int16_t)v;
    return (int32_t)v;
}
static uint32_t rd_u(const uint8_t *p, int n) {
    uint32_t v = 0;
    for (int i = 0; i < n; i++) v |= (uint32_t)p[i] << (8 * i);
    return v;
}

static slot_t *slot_for(slot_t *s, int *n, uint8_t rid) {
    for (int i = 0; i < *n; i++) if (s[i].rid == rid) return &s[i];
    if (*n >= HID_MAX_SLOTS) return 0;
    slot_t *t = &s[(*n)++];
    memset(t, 0, sizeof *t);
    t->rid = rid;
    t->used = 1;
    t->bits = rid ? 8u : 0u;
    return t;
}

static void note_app(hid_map_t *m, uint32_t page, uint32_t usage) {
    if (m->n_app < HID_MAX_APPS) m->app[m->n_app++] = (page << 16) | (usage & 0xFFFF);
    if (page == 1 && (usage == 2 || usage == 1)) m->kinds |= HID_K_MOUSE;
    else if (page == 1 && (usage == 6 || usage == 7)) m->kinds |= HID_K_KEYBOARD;
    else if (page == 0x0C) m->kinds |= HID_K_CONSUMER;
    else if (page >= 0xFF00) m->kinds |= HID_K_VENDOR;
    else if (page == 0x0D) m->kinds |= HID_K_DIGITIZER;
    else m->kinds |= HID_K_OTHER;
}

int hid_parse(const uint8_t *d, uint16_t len, hid_map_t *m) {
    memset(m, 0, sizeof *m);
    if (len == 0) return HID_E_BAD;
    if (len > HID_DESC_MAX) return HID_E_TOO_BIG;
    glob_t g, stack[MAX_STACK];
    memset(&g, 0, sizeof g);
    int sp = 0;
    uint32_t us[MAX_USAGES];
    int n_us = 0, have_range = 0;
    uint32_t umin = 0, umax = 0;
    uint32_t coll[MAX_COLL];
    int depth = 0;
    slot_t slots[HID_MAX_SLOTS];
    int n_slots = 0;
    uint16_t i = 0;
    while (i < len) {
        uint8_t pre = d[i];
        int size, type, tag;
        if (pre == 0xFE) {                               /* long item: skip */
            if (i + 2 >= len) return HID_E_BAD;
            uint16_t l = d[i + 1];
            if ((uint32_t)i + 3 + l > len) return HID_E_BAD;
            i = (uint16_t)(i + 3 + l);
            continue;
        }
        size = pre & 3; if (size == 3) size = 4;
        type = (pre >> 2) & 3;
        tag = pre >> 4;
        if ((uint32_t)i + 1u + (uint32_t)size > len) return HID_E_BAD;
        const uint8_t *p = d + i + 1;
        uint32_t uv = rd_u(p, size);
        int32_t sv = rd_s(p, size);
        i = (uint16_t)(i + 1 + size);
        if (type == 1) {                                  /* global */
            switch (tag) {
            case 0: g.page = uv; break;
            case 1: g.lmin = sv; break;
            case 2: g.lmax = sv; break;
            case 7: g.rsize = uv; break;
            case 8: if (uv == 0 || uv > 255) return HID_E_BAD; g.rid = (uint8_t)uv; m->uses_ids = 1; break;
            case 9: g.rcount = uv; break;
            case 10: if (sp >= MAX_STACK) return HID_E_LIMIT; stack[sp++] = g; break;
            case 11: if (sp <= 0) return HID_E_BAD; g = stack[--sp]; break;
            default: break;
            }
        } else if (type == 2) {                           /* local */
            if (tag == 0) {
                if (n_us < MAX_USAGES) us[n_us++] = size == 4 ? uv : ((g.page << 16) | (uv & 0xFFFF));
            } else if (tag == 1) { umin = size == 4 ? uv : ((g.page << 16) | (uv & 0xFFFF)); have_range |= 1; }
            else if (tag == 2) { umax = size == 4 ? uv : ((g.page << 16) | (uv & 0xFFFF)); have_range |= 2; }
        } else if (type == 0) {                           /* main */
            if (tag == 10) {                              /* collection */
                if (depth >= MAX_COLL) return HID_E_LIMIT;
                uint32_t u0 = n_us > 0 ? us[0] : 0;
                coll[depth++] = u0;
                if (depth == 1 && uv == 1) note_app(m, u0 >> 16, u0 & 0xFFFF);
                else if (depth == 1 && n_us > 0) { /* a top-level non-application collection: not an application */ }
            } else if (tag == 12) {                       /* end collection */
                if (depth <= 0) return HID_E_BAD;
                depth--;
            } else if (tag == 8) {                        /* input */
                if (g.rsize > 32 || g.rcount > 4096) return HID_E_LIMIT;
                slot_t *s = slot_for(slots, &n_slots, g.rid);
                if (!s) return HID_E_LIMIT;
                uint32_t total = g.rsize * g.rcount;
                if ((uint32_t)s->bits + total > 65535u) return HID_E_LIMIT;
                int in_mouse = 0;
                for (int c = 0; c < depth; c++) if (coll[c] == 0x00010002u || coll[c] == 0x00010001u) in_mouse = 1;
                int constant = uv & 1, variable = (uv >> 1) & 1, relative = (uv >> 2) & 1;
                if (!constant && variable && in_mouse && g.rsize >= 1) {
                    uint32_t cap = g.rcount < 64 ? g.rcount : 64;
                    for (uint32_t k = 0; k < cap; k++) {
                        uint32_t u = 0;
                        if (n_us > 0) u = us[k < (uint32_t)n_us ? k : (uint32_t)n_us - 1];
                        else if (have_range == 3) u = umin + k <= umax ? umin + k : umax;
                        uint32_t off = s->bits + k * g.rsize;
                        if ((u >> 16) == 9) {
                            if (s->btn_n == 0) { s->btn_off = (uint16_t)off; }
                            if (s->btn_n < 32 && g.rsize == 1 && off == (uint32_t)s->btn_off + s->btn_n) s->btn_n++;
                        } else if (u == 0x00010030u || u == 0x00010031u) {
                            int isx = u == 0x00010030u;
                            if (!relative || g.lmin >= 0 || g.rsize > 16) { s->bad_xy = 1; continue; }
                            if (isx) { s->have_x = 1; s->off_x = (uint16_t)off; s->size_x = (uint8_t)g.rsize; }
                            else { s->have_y = 1; s->off_y = (uint16_t)off; s->size_y = (uint8_t)g.rsize; }
                        }
                    }
                }
                s->bits += total;
            }
            n_us = 0; have_range = 0;                     /* local items end with every main item */
        }
    }
    if (depth != 0) return HID_E_BAD;
    for (int k = 0; k < n_slots; k++) {
        slot_t *s = &slots[k];
        if (!(s->have_x && s->have_y) || m->n >= HID_MAX_RIDS) continue;
        hid_motion_t *o = &m->m[m->n++];
        o->report_id = s->rid; o->size_x = s->size_x; o->size_y = s->size_y; o->off_x = s->off_x; o->off_y = s->off_y;
        o->len = (uint16_t)((s->bits + 7) / 8); o->btn_off = s->btn_off; o->btn_n = s->btn_n;
    }
    return m->n ? HID_OK : HID_E_NOXY;
}

void hid_boot_map(hid_map_t *m) {
    memset(m, 0, sizeof *m);
    m->n = 1; m->kinds = HID_K_MOUSE;
    hid_motion_t *o = &m->m[0];
    o->size_x = o->size_y = 8; o->off_x = 8; o->off_y = 16; o->len = 3; o->btn_off = 0; o->btn_n = 3;
}

const hid_motion_t *hid_find(const hid_map_t *m, const uint8_t *r, uint16_t len) {
    if (len == 0) return 0;
    for (int i = 0; i < m->n; i++) {
        const hid_motion_t *o = &m->m[i];
        if (m->uses_ids ? r[0] != o->report_id : o->report_id != 0) continue;
        uint16_t need_x = (uint16_t)((o->off_x + o->size_x + 7) / 8), need_y = (uint16_t)((o->off_y + o->size_y + 7) / 8);
        if (len < need_x || len < need_y) return 0;
        return o;
    }
    return 0;
}

static uint32_t get_bits(const uint8_t *b, uint32_t off, uint32_t size) {
    uint32_t i = off >> 3, sh = off & 7u, nb = (sh + size + 7u) >> 3;
    uint64_t v = 0;
    for (uint32_t k = 0; k < nb; k++) v |= (uint64_t)b[i + k] << (8u * k);
    return (uint32_t)((v >> sh) & (size >= 32 ? 0xFFFFFFFFu : ((1u << size) - 1u)));
}
static void put_bits(uint8_t *b, uint32_t off, uint32_t size, uint32_t val) {
    uint32_t i = off >> 3, sh = off & 7u, nb = (sh + size + 7u) >> 3;
    uint64_t mask = (size >= 32 ? 0xFFFFFFFFull : ((1ull << size) - 1ull)) << sh, v = 0;
    for (uint32_t k = 0; k < nb; k++) v |= (uint64_t)b[i + k] << (8u * k);
    v = (v & ~mask) | (((uint64_t)val << sh) & mask);
    for (uint32_t k = 0; k < nb; k++) b[i + k] = (uint8_t)(v >> (8u * k));
}
static int32_t sext(uint32_t v, uint32_t size) {
    uint32_t sign = 1u << (size - 1);
    return (int32_t)((v ^ sign) - sign);
}

void hid_xy_get(const hid_motion_t *m, const uint8_t *r, int32_t *dx, int32_t *dy) {
    *dx = sext(get_bits(r, m->off_x, m->size_x), m->size_x);
    *dy = sext(get_bits(r, m->off_y, m->size_y), m->size_y);
}

static int32_t clamp_field(int32_t v, uint32_t size) {
    int32_t hi = (int32_t)((1u << (size - 1)) - 1), lo = -hi - 1;
    return v < lo ? lo : (v > hi ? hi : v);
}

void hid_xy_set(const hid_motion_t *m, uint8_t *r, int32_t dx, int32_t dy) {
    put_bits(r, m->off_x, m->size_x, (uint32_t)clamp_field(dx, m->size_x) & ((1u << m->size_x) - 1u));
    put_bits(r, m->off_y, m->size_y, (uint32_t)clamp_field(dy, m->size_y) & ((1u << m->size_y) - 1u));
}

uint32_t hid_buttons(const hid_motion_t *m, const uint8_t *r, uint16_t len) {
    if (m->btn_n == 0 || (uint32_t)m->btn_off + m->btn_n > (uint32_t)len * 8u) return 0;
    return get_bits(r, m->btn_off, m->btn_n);
}

void hid_xy_mask(const hid_motion_t *m, uint8_t *mask, uint16_t len) {
    memset(mask, 0, len);
    for (uint32_t k = 0; k < m->size_x; k++) if (((m->off_x + k) >> 3) < len) mask[(m->off_x + k) >> 3] |= (uint8_t)(1u << ((m->off_x + k) & 7));
    for (uint32_t k = 0; k < m->size_y; k++) if (((m->off_y + k) >> 3) < len) mask[(m->off_y + k) >> 3] |= (uint8_t)(1u << ((m->off_y + k) & 7));
}

int hid_merge(const hid_motion_t *m, uint8_t *a, const uint8_t *b, uint16_t len) {
    uint8_t mask[64];
    if (len > sizeof mask) return 0;
    hid_xy_mask(m, mask, len);
    for (uint16_t i = 0; i < len; i++) if ((a[i] ^ b[i]) & (uint8_t)~mask[i]) return 0;
    int32_t ax, ay, bx, by;
    hid_xy_get(m, a, &ax, &ay);
    hid_xy_get(m, b, &bx, &by);
    int64_t sx = (int64_t)ax + bx, sy = (int64_t)ay + by;
    if (sx != clamp_field((int32_t)sx, m->size_x) || sy != clamp_field((int32_t)sy, m->size_y)) return 0;
    hid_xy_set(m, a, (int32_t)sx, (int32_t)sy);
    return 1;
}
