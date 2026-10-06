#include "usb_proxy.h"
#include <string.h>

void px_reset(px_state_t *st) {
    memset(st, 0, sizeof *st);
    for (int i = 0; i < IMG_MAX_IF; i++) st->proto[i] = 1;
    hid_boot_map(&st->boot);
}

static uint16_t lim(uint16_t n, uint16_t w) { return n < w ? n : w; }

int px_classify(const usb_image_t *im, const usb_setup_t *s, const uint8_t **data, uint16_t *len) {
    if (s->bmRequestType == 0x00 && s->bRequest == 5) return PX_LOCAL;                  /* SET_ADDRESS */
    if (!im->have_dev || !im->have_cfg || s->bRequest != 6) return PX_FORWARD;          /* only GET_DESCRIPTOR is ever cached */
    uint8_t type = (uint8_t)(s->wValue >> 8), idx = (uint8_t)(s->wValue & 0xFF);
    if (s->bmRequestType == 0x80) {
        if (type == 1 && idx == 0) { *data = im->dev; *len = lim(18, s->wLength); return PX_SERVE; }
        if (type == 2 && idx == 0) { *data = im->cfg; *len = lim(im->cfg_len, s->wLength); return PX_SERVE; }
    } else if (s->bmRequestType == 0x81 && idx == 0) {
        int i = img_find_if(im, (uint8_t)(s->wIndex & 0xFF));
        if (i < 0) return PX_FORWARD;
        const img_if_t *f = &im->ifc[i];
        if (type == 0x22 && f->have_rd) { *data = f->rd; *len = lim(f->rd_len, s->wLength); return PX_SERVE; }
        if (type == 0x21 && f->hid_off) { *data = im->cfg + f->hid_off; *len = lim(im->cfg[f->hid_off], s->wLength); return PX_SERVE; }
    }
    return PX_FORWARD;
}

void px_forwarded_ok(const usb_image_t *im, px_state_t *st, const usb_setup_t *s) {
    if (s->bmRequestType == 0x00 && s->bRequest == 9) {                                /* SET_CONFIGURATION */
        st->config = (uint8_t)s->wValue;
        for (int i = 0; i < IMG_MAX_IF; i++) st->proto[i] = 1;
    } else if (s->bmRequestType == 0x21 && s->bRequest == 0x0B) {                     /* HID SET_PROTOCOL */
        int i = img_find_if(im, (uint8_t)(s->wIndex & 0xFF));
        if (i >= 0) st->proto[i] = s->wValue ? 1 : 0;
    }
}

const hid_map_t *px_map(const usb_image_t *im, const px_state_t *st, int i) {
    const img_if_t *f = &im->ifc[i];
    if (st->proto[i] == 0 && f->sub == 1 && f->proto == 2) return &st->boot;
    return &f->map;
}
