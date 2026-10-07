// CtlLink v1 in the browser: messages, chunks, the fixed status/info layouts. Mirrors src/dataopen/ctl/protocol.py; the two are tied
// together by pwa/tests/golden.json (bytes both sides must produce and accept) and by the generated constants.js.
import { VER, HDR, MAX_BODY, MAX_MSG, CHUNK_MIN, CHUNK_MAX, REASM_TIMEOUT_MS, STATUS_SIZE, INFO_SIZE, REASONS, MODE_UNKNOWN } from './constants.js';

const TABLE = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c >>> 0;
  }
  return t;
})();

export function crc32(bytes) {
  let c = 0xffffffff;
  for (let i = 0; i < bytes.length; i++) c = TABLE[(c ^ bytes[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

const enc = new TextEncoder();
const dec = new TextDecoder('utf-8', { fatal: true });

export const toHex = (u8) => Array.from(u8, (b) => b.toString(16).padStart(2, '0')).join('');
export const fromHex = (s) => Uint8Array.from(s.match(/../g) || [], (h) => parseInt(h, 16));

export function packMessage(type, req = 0, body = new Uint8Array(0), flags = 0) {
  if (body.length > MAX_BODY) throw new RangeError('message body too long');
  const out = new Uint8Array(HDR + body.length + 4);
  const dv = new DataView(out.buffer);
  dv.setUint8(0, VER);
  dv.setUint8(1, type & 0xff);
  dv.setUint16(2, req & 0xffff, true);
  dv.setUint16(4, body.length, true);
  dv.setUint16(6, flags & 0xffff, true);
  out.set(body, HDR);
  dv.setUint32(HDR + body.length, crc32(out.subarray(0, HDR + body.length)), true);
  return out;
}

export function packJson(type, req = 0, obj = {}) {
  return packMessage(type, req, enc.encode(JSON.stringify(obj)));
}

// null for anything that is not a valid message (size, version, length or CRC wrong)
export function unpackMessage(raw) {
  if (raw.length < HDR + 4 || raw.length > MAX_MSG) return null;
  const dv = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
  const n = dv.getUint16(4, true);
  if (dv.getUint8(0) !== VER || n > MAX_BODY || raw.length !== HDR + n + 4) return null;
  if (dv.getUint32(HDR + n, true) !== crc32(raw.subarray(0, HDR + n))) return null;
  return { type: dv.getUint8(1), req: dv.getUint16(2, true), flags: dv.getUint16(6, true), body: raw.slice(HDR, HDR + n) };
}

export function bodyJson(msg) {
  const v = msg.body.length ? JSON.parse(dec.decode(msg.body)) : {};
  if (v === null || typeof v !== 'object' || Array.isArray(v)) throw new TypeError('a JSON object was expected');
  return v;
}

// -> { chunks: Uint8Array[], next }   (`size` bytes per chunk at most, header included)
export function chunkMessage(raw, size, seq0 = 0) {
  if (!(size >= CHUNK_MIN && size <= CHUNK_MAX)) throw new RangeError(`chunk size must be ${CHUNK_MIN}..${CHUNK_MAX}`);
  const room = size - 1;
  const parts = [];
  for (let i = 0; i < raw.length; i += room) parts.push(raw.subarray(i, i + room));
  if (!parts.length) parts.push(new Uint8Array(0));
  let seq = seq0 & 0x3f;
  const chunks = parts.map((p, i) => {
    const c = new Uint8Array(1 + p.length);
    c[0] = (i === 0 ? 0x80 : 0) | (i === parts.length - 1 ? 0x40 : 0) | seq;
    c.set(p, 1);
    seq = (seq + 1) & 0x3f;
    return c;
  });
  return { chunks, next: seq };
}

// Chunks in, whole (not yet CRC-checked) messages out. Bounded memory, bounded wait.
export class Reassembler {
  constructor(timeoutMs = REASM_TIMEOUT_MS) {
    this.timeoutMs = timeoutMs;
    this.dropped = 0;
    this.reset();
  }
  reset() { this.parts = []; this.len = 0; this.active = false; this.expect = 0; this.last = 0; }
  feed(chunk, nowMs = 0) {
    if (!chunk.length) { this.dropped++; return null; }
    if (this.active && nowMs - this.last > this.timeoutMs) { this.reset(); this.dropped++; }
    const h = chunk[0], first = !!(h & 0x80), last = !!(h & 0x40), seq = h & 0x3f;
    if (first) {
      if (this.active) this.dropped++;
      this.parts = [];
      this.len = 0;
      this.active = true;
    } else if (!this.active || seq !== this.expect) {
      this.reset();
      this.dropped++;
      return null;
    }
    this.parts.push(chunk.subarray(1));
    this.len += chunk.length - 1;
    this.expect = (seq + 1) & 0x3f;
    this.last = nowMs;
    if (this.len > MAX_MSG) { this.reset(); this.dropped++; return null; }
    if (!last) return null;
    const out = new Uint8Array(this.len);
    let o = 0;
    for (const p of this.parts) { out.set(p, o); o += p.length; }
    this.reset();
    return out;
  }
}

// ---- status (the 20-byte characteristic) and info (16 bytes)
export function unpackStatus(b) {
  if (b.length !== STATUS_SIZE || b[0] !== VER) return null;
  const dv = new DataView(b.buffer, b.byteOffset, b.byteLength);
  const s = {
    bridge: dv.getUint8(1), mode: dv.getUint8(2), reason: dv.getUint8(3), flags: dv.getUint16(4, true), fill: dv.getUint8(6),
    ready: dv.getUint8(7), stateRev: dv.getUint16(8, true), manifestRev: dv.getUint16(10, true), uptimeMin: dv.getUint16(12, true),
    trialLeftS: dv.getUint16(14, true), strength: dv.getUint8(16), tremor: dv.getUint8(17), slot: dv.getUint8(18), slotMask: dv.getUint8(19),
  };
  s.reasonName = s.reason < REASONS.length ? REASONS[s.reason] : `R${s.reason}`;
  return s;
}

export function packStatus(s) {
  const out = new Uint8Array(STATUS_SIZE);
  const dv = new DataView(out.buffer);
  dv.setUint8(0, VER);
  dv.setUint8(1, s.bridge ?? 0);
  dv.setUint8(2, s.mode ?? MODE_UNKNOWN);
  dv.setUint8(3, s.reason ?? 0);
  dv.setUint16(4, s.flags ?? 0, true);
  dv.setUint8(6, s.fill ?? 0);
  dv.setUint8(7, s.ready ?? 0);
  dv.setUint16(8, s.stateRev ?? 0, true);
  dv.setUint16(10, s.manifestRev ?? 0, true);
  dv.setUint16(12, Math.min(s.uptimeMin ?? 0, 0xffff), true);
  dv.setUint16(14, Math.min(s.trialLeftS ?? 0, 0xffff), true);
  dv.setUint8(16, s.strength ?? 255);
  dv.setUint8(17, s.tremor ?? 255);
  dv.setUint8(18, s.slot ?? 0);
  dv.setUint8(19, s.slotMask ?? 0);
  return out;
}

export function unpackInfo(b) {
  if (b.length !== INFO_SIZE || b[0] !== VER) return null;
  const dv = new DataView(b.buffer, b.byteOffset, b.byteLength);
  return { caps: b[1], deviceId: toHex(b.subarray(2, 6)), manifestHash: toHex(b.subarray(6, 10)), manifestRev: dv.getUint16(10, true), chunkHint: dv.getUint16(12, true) };
}
