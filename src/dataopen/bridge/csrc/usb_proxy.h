/* Control-transfer routing of the transparent proxy, as pure functions over the cached image.
 * The PC gets descriptors (device, configuration 0, HID class and report descriptors) from the cache, byte for byte what the mouse
 * reported; every OTHER control request (strings in every language, BOS, MS OS, HID get/set report, idle, protocol, vendor and class
 * requests, set configuration/interface, feature requests, ...) is forwarded to the mouse live, its answer or STALL mirrored.
 * Only SET_ADDRESS is local (it addresses the bridge's own device port). Interrupt IN reports of MOUSE_MOTION interfaces are edited
 * by bridge.c; all other endpoints are copied through untouched by the port. */
#ifndef USB_PROXY_H
#define USB_PROXY_H
#include <stdint.h>
#include "usb_image.h"

typedef struct { uint8_t bmRequestType, bRequest; uint16_t wValue, wIndex, wLength; } usb_setup_t;
enum { PX_LOCAL = 0, PX_SERVE = 1, PX_FORWARD = 2 };

typedef struct {
    uint8_t proto[IMG_MAX_IF];     /* HID protocol per interface: 1 = report (default), 0 = boot */
    uint8_t config;
    hid_map_t boot;
} px_state_t;

void px_reset(px_state_t *st);
/* PX_SERVE fills data and len (already limited to wLength); PX_FORWARD / PX_LOCAL leave them untouched. */
int px_classify(const usb_image_t *im, const usb_setup_t *s, const uint8_t **data, uint16_t *len);
/* After a forwarded request completed successfully: track SET_PROTOCOL / SET_CONFIGURATION. */
void px_forwarded_ok(const usb_image_t *im, px_state_t *st, const usb_setup_t *s);
/* The report layout currently in force for interface index i (boot layout after SET_PROTOCOL(0) on a boot-mouse interface). */
const hid_map_t *px_map(const usb_image_t *im, const px_state_t *st, int i);
#endif
