/* BridgeLink v1: fixed 128-byte SPI frames between the bridge (slave) and the compute module (master). The hot 1 kHz path never
 * waits on this link. Layout:
 *   0 sync 0xB7 | 1 ver | 2 kind | 3 flags (low 4 bits: blob serial) | 4-5 seq | 6-7 ack | 8-9 len | 10 frag_idx | 11 frag_cnt
 *   12..123 payload (112 bytes) | 124..127 crc32 (zlib) over bytes 0..123. All integers little-endian. */
#ifndef LINK_H
#define LINK_H
#include <stdint.h>

#define LINK_FRAME 128
#define LINK_PAYLOAD 112
#define LINK_SYNC 0xB7
#define LINK_VER 1
#define LINK_BLOB_MAX 336          /* 3 fragments */

enum { LK_NOP = 0, LK_TELEM = 1, LK_STATUS = 2, LK_TSYNC_REPLY = 3,
       LK_HELLO = 0x10, LK_PARAMS_ASC = 0x11, LK_PARAMS_TREMOR = 0x12, LK_SCENE = 0x13, LK_CMD = 0x14, LK_TSYNC = 0x15 };
enum { LINK_OK = 0, LINK_E_SYNC = -1, LINK_E_VER = -2, LINK_E_LEN = -3, LINK_E_CRC = -4, LINK_E_FRAG = -5 };

typedef struct { uint8_t kind, flags, frag_idx, frag_cnt; uint16_t seq, ack, len; uint8_t payload[LINK_PAYLOAD]; } link_frame_t;

uint32_t link_crc32(const uint8_t *p, uint32_t n);
void link_pack(uint8_t out[LINK_FRAME], const link_frame_t *f);
int link_unpack(const uint8_t in[LINK_FRAME], link_frame_t *f);

/* Reassembly of a blob sent in up to 3 fragments (any order, duplicates ignored, a different serial starts afresh). */
typedef struct { uint8_t buf[LINK_BLOB_MAX]; uint8_t serial, cnt, mask, active; uint16_t last_len; int64_t t_start; } link_blob_t;
void link_blob_reset(link_blob_t *b);
/* Returns total length once complete (the blob is then in b->buf and the reassembler is cleared), 0 while incomplete, <0 on error. */
int link_blob_feed(link_blob_t *b, const link_frame_t *f, int64_t t_us, int64_t timeout_us);
#endif
