import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { crc32, packMessage, unpackMessage, packJson, bodyJson, chunkMessage, Reassembler, unpackStatus, packStatus, unpackInfo, toHex, fromHex } from '../js/link.js';
import * as C from '../js/constants.js';

const golden = JSON.parse(readFileSync(new URL('./golden.json', import.meta.url)));

test('crc32 matches the shared vectors', () => {
  for (const v of golden.crc32) assert.equal(crc32(fromHex(v.data)), v.crc);
  assert.equal(crc32(new TextEncoder().encode('123456789')), 0xcbf43926);
});

test('messages: the browser parses what Python wrote and writes the same bytes', () => {
  for (const m of golden.messages) {
    const raw = fromHex(m.hex);
    const msg = unpackMessage(raw);
    assert.ok(msg, m.name);
    assert.equal(msg.type, m.type);
    assert.equal(msg.req, m.req);
    assert.equal(toHex(msg.body), m.body);
    assert.equal(toHex(packMessage(msg.type, msg.req, msg.body, msg.flags)), m.hex, m.name);
  }
});

test('JSON bodies are compact UTF-8 identical to Python', () => {
  const byName = Object.fromEntries(golden.messages.map((m) => [m.name, m]));
  assert.equal(toHex(packJson(C.T.HELLO, 1, { v: 1, chunk: 100, lang: 'ru' })), byName.hello.hex);
  assert.equal(toHex(packJson(C.T.SET, 7, { key: 'assist.strength', value: 7 })), byName.set.hex);
  assert.equal(toHex(packJson(C.T.EVENT, 0, { msg: 'Помощь выключена' })), byName.utf8.hex);
  assert.deepEqual(bodyJson(unpackMessage(fromHex(byName.utf8.hex))), { msg: 'Помощь выключена' });
});

test('chunks: identical bytes at 20, 100 and 244', () => {
  for (const m of golden.messages) {
    for (const size of [20, 100, 244]) {
      const { chunks } = chunkMessage(fromHex(m.hex), size, 5);
      assert.deepEqual(chunks.map(toHex), m.chunks[String(size)], `${m.name}@${size}`);
      const r = new Reassembler();
      let out = null;
      for (const c of chunks) out = r.feed(c, 0);
      assert.equal(toHex(out), m.hex);
    }
  }
});

test('every single bit flip of a message is rejected', () => {
  const raw = packJson(C.T.SET, 7, { key: 'assist.strength', value: 7 });
  for (let i = 0; i < raw.length * 8; i++) {
    const b = raw.slice();
    b[i >> 3] ^= 1 << (i & 7);
    assert.equal(unpackMessage(b), null, `bit ${i}`);
  }
  for (let n = 0; n < raw.length; n++) assert.equal(unpackMessage(raw.subarray(0, n)), null);
});

test('the safety messages fit one 20-byte chunk', () => {
  for (const t of [C.T.STOP, C.T.HARD_BYPASS]) {
    const { chunks } = chunkMessage(packMessage(t, 0xffff), 20);
    assert.equal(chunks.length, 1);
    assert.ok(chunks[0].length <= 20);
  }
});

test('size limits', () => {
  assert.throws(() => chunkMessage(new Uint8Array(3), 19));
  assert.throws(() => chunkMessage(new Uint8Array(3), 245));
  assert.throws(() => packMessage(C.T.DATA, 0, new Uint8Array(C.MAX_BODY + 1)));
});

test('the reassembler never wedges', () => {
  const a = new Uint8Array(400).map((_, i) => i * 3), b = new Uint8Array(300).map((_, i) => i * 5);
  const ca = chunkMessage(packMessage(C.T.DATA, 1, a), 40, 0), cb = chunkMessage(packMessage(C.T.DATA, 2, b), 40, ca.next);
  let r = new Reassembler();
  for (const c of [...ca.chunks.slice(0, 2), ...ca.chunks.slice(3)]) assert.equal(r.feed(c, 0), null);   // a lost chunk
  let out = null;
  for (const c of cb.chunks) out = r.feed(c, 0);
  assert.equal(unpackMessage(out).req, 2);
  r = new Reassembler(1000);                                                                               // a stalled message times out
  r.feed(ca.chunks[0], 0);
  assert.equal(r.feed(ca.chunks[1], 5000), null);
  assert.equal(r.active, false);
  r = new Reassembler();                                                                                   // random bytes
  let seed = 12345;
  const rnd = () => (seed = (seed * 1103515245 + 12345) & 0x7fffffff) >> 8;
  for (let i = 0; i < 20000; i++) {
    const c = Uint8Array.from({ length: rnd() % 40 }, () => rnd() & 255);
    const o = r.feed(c, i);
    if (o) unpackMessage(o);
    assert.ok(r.len <= C.MAX_MSG);
  }
});

test('status layout: Python bytes in, the same fields out, and back', () => {
  for (const s of golden.status) {
    const u = unpackStatus(fromHex(s.hex));
    assert.equal(u.bridge, s.bridge);
    assert.equal(u.mode, s.mode);
    assert.equal(u.reason, s.reason);
    assert.equal(u.flags, s.flags);
    assert.equal(u.fill, s.fill);
    assert.equal(u.ready, s.ready);
    assert.equal(u.stateRev, s.state_rev);
    assert.equal(u.manifestRev, s.manifest_rev);
    assert.equal(u.uptimeS, s.uptime_s);
    assert.equal(u.trialLeftS, s.trial_left_s);
    assert.equal(u.strength, s.strength);
    assert.equal(u.tremor, s.tremor);
    assert.equal(toHex(packStatus(u)), s.hex);
  }
  assert.equal(unpackStatus(new Uint8Array(19)), null);
  assert.equal(C.STATUS_SIZE, 20);
});

test('info layout', () => {
  const i = golden.info[0];
  const u = unpackInfo(fromHex(i.hex));
  assert.deepEqual([u.caps, u.deviceId, u.manifestHash, u.manifestRev, u.chunkHint], [i.caps, i.device_id, i.manifest_hash, i.manifest_rev, i.chunk_hint]);
});

test('the reason table is the bridge enum', () => {
  assert.equal(unpackStatus(packStatus({ reason: C.REASONS.indexOf('PANIC') })).reasonName, 'PANIC');
  assert.equal(unpackStatus(packStatus({ reason: 250 })).reasonName, 'R250');
});
