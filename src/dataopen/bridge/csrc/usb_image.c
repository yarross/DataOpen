#include "usb_image.h"
#include <string.h>

void img_init(usb_image_t *im) { memset(im, 0, sizeof *im); }

int img_set_device(usb_image_t *im, const uint8_t *d, uint16_t len) {
    if (len < 18 || d[0] != 18 || d[1] != 1) return IMG_E_BAD_DEVICE;
    memcpy(im->dev, d, 18);
    im->have_dev = 1;
    return d[17] == 1 ? IMG_OK : IMG_E_MULTI_CONFIG;
}

int img_set_config(usb_image_t *im, const uint8_t *c, uint16_t len) {
    if (len < 9 || c[0] != 9 || c[1] != 2) return IMG_E_BAD_CONFIG;
    uint16_t total = (uint16_t)(c[2] | (c[3] << 8));
    if (total > IMG_CFG_MAX) return IMG_E_TOO_BIG;
    if (total < 9 || total > len) return IMG_E_BAD_CONFIG;
    memcpy(im->cfg, c, total);
    im->cfg_len = total;
    im->n_if = im->n_ep = 0;
    int cur = -1;
    uint16_t i = 0;
    int err = IMG_OK;
    while (i + 2 <= total) {
        uint8_t bl = c[i], bt = c[i + 1];
        if (bl < 2 || (uint32_t)i + bl > total) return IMG_E_BAD_CONFIG;
        if (bt == 4) {                                       /* interface */
            if (bl < 9) return IMG_E_BAD_CONFIG;
            if (c[i + 3] != 0) { err = IMG_E_ALT_SETTING; cur = -1; }      /* alternate settings can not be mirrored by this proxy */
            else {
                if (im->n_if >= IMG_MAX_IF) return IMG_E_TOO_MANY;
                img_if_t *f = &im->ifc[im->n_if];
                memset(f, 0, sizeof *f);
                f->num = c[i + 2]; f->cls = c[i + 5]; f->sub = c[i + 6]; f->proto = c[i + 7];
                cur = im->n_if++;
            }
        } else if (bt == 5 && cur >= 0) {                    /* endpoint */
            if (bl < 7) return IMG_E_BAD_CONFIG;
            if (im->n_ep >= IMG_MAX_EP) return IMG_E_TOO_MANY;
            img_ep_t *e = &im->ep[im->n_ep++];
            e->addr = c[i + 2]; e->attr = c[i + 3]; e->mps = (uint16_t)(c[i + 4] | (c[i + 5] << 8)); e->interval = c[i + 6];
            e->iface = (uint8_t)cur;
            im->ifc[cur].n_ep++;
        } else if (bt == 0x21 && cur >= 0 && im->ifc[cur].cls == 3) {   /* HID class descriptor */
            if (bl >= 9) {
                im->ifc[cur].hid_off = i;
                for (int k = 0; k < c[i + 5] && 6 + 3 * k + 2 < bl; k++)
                    if (c[i + 6 + 3 * k] == 0x22) im->ifc[cur].rd_wanted = (uint16_t)(c[i + 7 + 3 * k] | (c[i + 8 + 3 * k] << 8));
            }
        }
        i = (uint16_t)(i + bl);
    }
    im->have_cfg = 1;
    im->cfg_err = (int8_t)err;
    return err;
}

uint16_t img_rd_wanted(const usb_image_t *im, int i) { return i >= 0 && i < im->n_if ? im->ifc[i].rd_wanted : 0; }

int img_set_report_desc(usb_image_t *im, int i, const uint8_t *rd, uint16_t len) {
    if (i < 0 || i >= im->n_if) return IMG_E_BAD_CONFIG;
    img_if_t *f = &im->ifc[i];
    if (len > HID_DESC_MAX) { f->parse_err = HID_E_TOO_BIG; return IMG_E_TOO_BIG; }
    memcpy(f->rd, rd, len);
    f->rd_len = len;
    f->have_rd = 1;
    f->parse_err = (int8_t)hid_parse(f->rd, len, &f->map);
    f->parsed = 1;
    return IMG_OK;
}

int img_finalize(usb_image_t *im, int scope) {
    if (!im->have_dev || !im->have_cfg || im->n_if == 0) return IMG_E_BAD_CONFIG;
    if (im->dev[17] != 1) return IMG_E_MULTI_CONFIG;
    if (im->cfg_err) return im->cfg_err;
    for (int i = 0; i < im->n_ep; i++) {
        uint8_t type = im->ep[i].attr & 3;
        if (type == 1 || type == 2) return IMG_E_UNSUPPORTED_EP;   /* isochronous, bulk */
    }
    im->n_motion = 0;
    for (int i = 0; i < im->n_if; i++) {
        img_if_t *f = &im->ifc[i];
        f->role = ROLE_PASSTHROUGH;
        if (f->cls != 3) continue;
        if (f->rd_wanted && !f->have_rd) return IMG_E_MISSING_RD;
        if (f->have_rd && f->parse_err == HID_OK && (f->map.kinds & HID_K_MOUSE)) { f->role = ROLE_MOUSE_MOTION; im->n_motion++; }
        if (scope == IMG_SCOPE_MOUSE_ONLY) {
            int kbd = (f->have_rd && (f->map.kinds & (HID_K_KEYBOARD | HID_K_DIGITIZER))) || (f->sub == 1 && f->proto == 1);
            if (kbd) return IMG_E_NOT_MOUSE_ONLY;
        }
    }
    if (im->n_motion == 0) return IMG_E_NO_MOTION;
    return IMG_OK;
}

int img_find_if(const usb_image_t *im, uint8_t ifnum) {
    for (int i = 0; i < im->n_if; i++) if (im->ifc[i].num == ifnum) return i;
    return -1;
}
const img_ep_t *img_find_ep(const usb_image_t *im, uint8_t addr) {
    for (int i = 0; i < im->n_ep; i++) if (im->ep[i].addr == addr) return &im->ep[i];
    return 0;
}
