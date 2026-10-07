// The Session against a scripted fake transport: the write queue, the safety action's priority, the chunk-size fallback, timeouts, the
// manifest cache, missed events. No device and no network involved.
import test from 'node:test';
import assert from 'node:assert/strict';
import { Session, SessionError } from '../js/session.js';
import { T, VER, ERR } from '../js/constants.js';
import { packJson, packMessage, unpackMessage, bodyJson, chunkMessage, Reassembler, packStatus, crc32, toHex } from '../js/link.js';
import { Store } from '../js/store.js';

// A pretend device: reassembles what the phone writes, answers through scripted handlers, can refuse long chunks.
class FakeTransport {
  constructor({ maxWrite = 244, writeDelayMs = 0, manifest = { schema: 1, rev: 1, title: { ru: 'а', en: 'a' }, pages: [] }, chunkReply = 100 } = {}) {
    this.maxWrite = maxWrite;
    this.writeDelayMs = writeDelayMs;
    this.written = [];
    this.messages = [];
    this.reasm = new Reassembler();
    this.seq = 0;
    this.manifest = manifest;
    this.manifestBytes = new TextEncoder().encode(JSON.stringify(manifest));
    this.chunkReply = chunkReply;
    this.stateRev = 1;
    this.state = { 'assist.on': false, 'assist.strength': 5 };
    this.silent = new Set();
    this.onChunk = this.onStatus = this.onClose = () => {};
    this.closed = false;
    this.opened = 0;
  }
  async open() {
    this.opened++;
    const info = new Uint8Array(16);
    info[0] = VER; info.set([1, 2, 3, 4], 2); info.set([0xde, 0xad, 0xbe, 0xef], 6); info[10] = 1; info[12] = 244;
    return { info, status: packStatus({ bridge: 1, mode: 2, stateRev: this.stateRev }) };
  }
  close() { this.closed = true; }
  async write(chunk) {
    if (chunk.length > this.maxWrite) throw new Error('value too long');
    if (this.writeDelayMs) await new Promise((r) => setTimeout(r, this.writeDelayMs));
    this.written.push(chunk);
    const raw = this.reasm.feed(chunk, 0);
    if (!raw) return;
    const m = unpackMessage(raw);
    this.messages.push(m);
    if (!this.silent.has(m.type)) queueMicrotask(() => this.answer(m));
  }
  send(raw, size = this.chunkReply) {
    const { chunks, next } = chunkMessage(raw, size, this.seq);
    this.seq = next;
    for (const c of chunks) this.onChunk(c);
  }
  answer(m) {
    const j = m.body.length && m.body[0] === 0x7b ? bodyJson(m) : {};
    switch (m.type) {
      case T.HELLO: this.send(packJson(T.HELLO_R, m.req, { v: 1, chunk: Math.min(j.chunk, 244), device: '01020304', fw: 'x', manifest_rev: 1, manifest_hash: crc32(this.manifestBytes).toString(16).padStart(8, '0'), trial_s: 20 })); break;
      case T.GET:
        if (j.what === 'manifest') this.send(packMessage(T.DATA, m.req, this.manifestBytes));
        else if (j.what === 'state') this.send(packJson(T.DATA, m.req, { rev: this.stateRev, state: this.state }));
        break;
      case T.SET: this.send(packJson(T.ACK, m.req, { ok: true })); break;
      case T.STOP: case T.HARD_BYPASS: this.send(packJson(T.ACK, m.req, { ok: true })); break;
      case T.PING: this.send(packMessage(T.PONG, m.req, m.body)); break;
      case T.ACT: this.send(packJson(T.ERR, m.req, { code: ERR.PHYSICAL, key: 'err.physical', detail: 'press' })); break;
      default: this.send(packJson(T.ERR, m.req, { code: ERR.UNSUPPORTED, key: 'err.unsupported', detail: '' }));
    }
  }
}

const timers = { setTimeout, clearTimeout };

test('connect: hello at the smallest size, then the negotiated one; manifest and state arrive', async () => {
  const tr = new FakeTransport();
  const s = new Session(tr, { chunk: 180, timers });
  await s.connect();
  assert.equal(s.conn, 'connected');
  assert.equal(s.hello.chunk, 180);
  assert.equal(s.chunk, 180);
  assert.ok(tr.written[0].length <= 20 || tr.written.length > 1);          // the HELLO was cut into 20-byte chunks
  assert.ok(tr.written.slice(0, 3).every((c) => c.length <= 20));
  assert.equal(s.manifest.rev, 1);
  assert.equal(s.state['assist.strength'], 5);
});

test('the manifest is cached by hash: a second connection does not download it', async () => {
  const store = new Store(null);
  const tr = new FakeTransport();
  const s1 = new Session(tr, { store, timers });
  await s1.connect();
  const gets1 = tr.messages.filter((m) => m.type === T.GET && bodyJson(m).what === 'manifest').length;
  const s2 = new Session(tr, { store, timers });
  await s2.connect();
  const gets2 = tr.messages.filter((m) => m.type === T.GET && bodyJson(m).what === 'manifest').length;
  assert.equal(gets1, 1);
  assert.equal(gets2, 1);
  assert.deepEqual(s2.manifest, tr.manifest);
  tr.manifest = { ...tr.manifest, rev: 2 };                                  // a changed manifest hashes differently and is fetched again
  tr.manifestBytes = new TextEncoder().encode(JSON.stringify(tr.manifest));
  const s3 = new Session(tr, { store, timers });
  await s3.connect();
  assert.equal(tr.messages.filter((m) => m.type === T.GET && bodyJson(m).what === 'manifest').length, 2);
  assert.equal(s3.manifest.rev, 2);
});

test('a chunk the link refuses makes the whole message go again at 20 bytes', async () => {
  const tr = new FakeTransport({ maxWrite: 20 });
  const s = new Session(tr, { chunk: 180, timers });
  await s.connect();
  assert.equal(s.chunk, 20);
  const r = await s.set('assist.strength', 7);
  assert.equal(r.json.ok, true);
  assert.ok(tr.written.every((c) => c.length <= 20));
});

test('the safety action jumps the queue at chunk level and cuts off a half-sent message', async () => {
  const tr = new FakeTransport({ writeDelayMs: 2 });
  const s = new Session(tr, { chunk: 40, timers });
  await s.connect();
  tr.written.length = 0;
  const big = new Uint8Array(2000).map((_, i) => i);
  const bundle = s.putBundle(big).catch((e) => e);
  await new Promise((r) => setTimeout(r, 15));                                // let a few chunks go out, not all
  assert.ok(tr.written.length >= 1 && tr.written.length < 50);
  const stopped = await s.stop();
  assert.equal(stopped.json.ok, true);
  const outcome = await bundle;
  assert.ok(outcome instanceof SessionError && outcome.key === 'err.interrupted');
  assert.ok(tr.messages.some((m) => m.type === T.STOP));
  assert.ok(!tr.messages.some((m) => m.type === T.BUNDLE_PUT));               // the cut-off message never completed on the device
});

test('stop waits behind nothing when the queue is idle, and works before any hello', async () => {
  const tr = new FakeTransport();
  const s = new Session(tr, { timers });
  s.tr.onChunk = (c) => s._onChunk(c);
  const r = await s.stop();
  assert.equal(r.json.ok, true);
  const rr = await s.hardBypass();
  assert.equal(rr.json.ok, true);
});

test('a silent device times out with a plain error and nothing stays pending', async () => {
  const tr = new FakeTransport();
  const s = new Session(tr, { timeoutMs: 50, timers });
  await s.connect();
  tr.silent.add(T.SET);
  await assert.rejects(s.set('assist.on', true), (e) => e instanceof SessionError && e.key === 'err.timeout');
  assert.equal(s.pending.size, 0);
});

test('device errors become keyed SessionErrors', async () => {
  const tr = new FakeTransport();
  const s = new Session(tr, { timers });
  await s.connect();
  await assert.rejects(s.act('pairing.forget', true), (e) => e.code === ERR.PHYSICAL && e.key === 'err.physical');
});

test('losing the connection rejects what is waiting and says so', async () => {
  const tr = new FakeTransport();
  const s = new Session(tr, { timers });
  await s.connect();
  tr.silent.add(T.SET);
  const states = [];
  s.on('conn', (c) => states.push(c));
  const p = s.set('assist.on', true).catch((e) => e);
  tr.onClose();
  const e = await p;
  assert.ok(e instanceof SessionError && e.key === 'err.disconnected');
  assert.equal(s.conn, 'lost');
  assert.deepEqual(states, ['lost']);
});

test('events update the state; a status with a newer revision triggers a full refresh', async () => {
  const tr = new FakeTransport();
  const s = new Session(tr, { timers });
  await s.connect();
  tr.send(packJson(T.EVENT, 0, { rev: 2, state: { 'assist.on': true } }));
  assert.equal(s.state['assist.on'], true);
  assert.equal(s.rev, 2);
  tr.state = { 'assist.on': true, 'assist.strength': 9 };
  tr.stateRev = 5;
  tr.onStatus(packStatus({ bridge: 1, mode: 3, stateRev: 5 }));              // events 3 and 4 were never seen
  await new Promise((r) => setTimeout(r, 20));
  assert.equal(s.state['assist.strength'], 9);
  assert.equal(s.rev, 5);
});

test('garbage from the device is counted and ignored', async () => {
  const tr = new FakeTransport();
  const s = new Session(tr, { timers });
  await s.connect();
  const before = s.bad;
  tr.onChunk(Uint8Array.from([0xc0 | 1, 1, 2, 3]));
  tr.onStatus(new Uint8Array(7));
  assert.equal(s.bad, before + 2);
  assert.equal(s.conn, 'connected');
});

test('a failed open leaves the session idle and rethrows', async () => {
  const tr = new FakeTransport();
  tr.open = async () => { throw new Error('User cancelled the requestDevice() chooser.'); };
  const s = new Session(tr, { timers });
  await assert.rejects(s.connect(), /cancelled/);
  assert.equal(s.conn, 'idle');
});

test('requests carry distinct, never-zero ids and wrap', async () => {
  const tr = new FakeTransport();
  const s = new Session(tr, { timers });
  await s.connect();
  s.reqId = 0xfffe;
  const ids = [];
  for (let i = 0; i < 4; i++) { await s.ping(); ids.push(tr.messages.at(-1).req); }
  assert.deepEqual(ids, [0xffff, 1, 2, 3]);
});

test('toHex is exported for tests', () => assert.equal(toHex(Uint8Array.from([1, 255])), '01ff'));
