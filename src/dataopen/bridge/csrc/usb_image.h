/* The bridge's picture of the attached mouse: device descriptor, the (single) configuration descriptor, the report descriptor
 * of every HID interface, the endpoint table. Everything the PC sees at enumeration is served from here (byte for byte what the
 * mouse reported); everything else is forwarded to the mouse live (see usb_proxy.h).
 * Interfaces whose reports carry relative X/Y get the role MOUSE_MOTION: the bridge edits those reports. All others (receiver
 * HID++ channels, keyboard interfaces, vendor configuration, ...) are PASSTHROUGH: never parsed beyond classification, never touched. */
#ifndef USB_IMAGE_H
#define USB_IMAGE_H
#include <stdint.h>
#include "hid_desc.h"

#define IMG_MAX_IF 6
#define IMG_MAX_EP 12
#define IMG_CFG_MAX 1024

enum { IMG_OK = 0, IMG_E_BAD_DEVICE = -1, IMG_E_BAD_CONFIG = -2, IMG_E_TOO_BIG = -3, IMG_E_MULTI_CONFIG = -4, IMG_E_UNSUPPORTED_EP = -5,
       IMG_E_ALT_SETTING = -6, IMG_E_NO_MOTION = -7, IMG_E_NOT_MOUSE_ONLY = -8, IMG_E_MISSING_RD = -9, IMG_E_TOO_MANY = -10 };
enum { IMG_SCOPE_TRANSPARENT = 0, IMG_SCOPE_MOUSE_ONLY = 1 };
enum { ROLE_PASSTHROUGH = 0, ROLE_MOUSE_MOTION = 1 };

typedef struct {
    uint8_t addr, attr, interval, iface;   /* iface: index into usb_image_t.ifc */
    uint16_t mps;
} img_ep_t;

typedef struct {
    uint8_t num, cls, sub, proto, role, n_ep, have_rd, parsed;
    uint16_t hid_off;                      /* offset of the HID class descriptor in cfg (0 = none) */
    uint16_t rd_wanted, rd_len;
    int8_t parse_err;
    uint8_t rd[HID_DESC_MAX];
    hid_map_t map;
} img_if_t;

typedef struct {
    uint8_t dev[18];
    uint8_t cfg[IMG_CFG_MAX];
    uint16_t cfg_len;
    uint8_t have_dev, have_cfg, n_if, n_ep, n_motion;
    int8_t cfg_err;                        /* a problem found while parsing the configuration (alternate settings) */
    img_if_t ifc[IMG_MAX_IF];
    img_ep_t ep[IMG_MAX_EP];
} usb_image_t;

void img_init(usb_image_t *im);
int img_set_device(usb_image_t *im, const uint8_t *d, uint16_t len);
/* Parses the configuration descriptor: interfaces, endpoints, HID class descriptors. */
int img_set_config(usb_image_t *im, const uint8_t *c, uint16_t len);
/* How many report-descriptor bytes the HID class descriptor of interface index i announces (0 = not HID). */
uint16_t img_rd_wanted(const usb_image_t *im, int i);
int img_set_report_desc(usb_image_t *im, int i, const uint8_t *rd, uint16_t len);
/* Checks the whole picture, assigns roles. Returns IMG_OK or the first reason the bridge must not engage. */
int img_finalize(usb_image_t *im, int scope);
int img_find_if(const usb_image_t *im, uint8_t ifnum);          /* index or -1 */
const img_ep_t *img_find_ep(const usb_image_t *im, uint8_t addr);
#endif
