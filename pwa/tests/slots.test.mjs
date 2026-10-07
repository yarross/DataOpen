// The phone client's side of slots and updates: the picker model, the text control, the firmware upload against a scripted device.
// The page stays dumb: it names the slots the device announces and carries opaque bytes; it decides nothing.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { slotsModel } from '../js/shell.js';
import { buildView, walk } from '../js/view.js';
import { Session, SessionError } from '../js/session.js';
import { T, VER, ERR, FW_CHUNK_MAX, SLOT_COUNT } from '../js/constants.js';
import { packJson, packMessage, unpackMessage, bodyJson, chunkMessage, Reassembler, packStatus, crc32 } from '../js/link.js';
import { STRINGS, errKeyFor, t } from '../js/i18n.js';

const manifest = JSON.parse(readFileSync(new URL('./manifest.json', import.meta.url)));
const status = (o = {}) => ({ bridge: 1, mode: 2, slot: 0, slotMask: 0b0101, ...o });

// ------------------------------------------------------------------------------------------------------------ the picker
test('four slots, named by the device, the active one marked, "has a profile" from the status mask', () => {
  const state = { 'slot.0.name': 'Работа', 'slot.1.name': '', 'slot.2.name': 'Браузер' };
  const m = slotsModel(status({ slot: 2 }), state, 'connected', 'ru');
  assert.equal(SLOT_COUNT, 4);
  assert.deepEqual(m.items.map((i) => [i.k, i.title, i.has, i.active]),
    [[0, '1 · Работа', true, false], [1, 'Слот 2', false, false], [2, '3 · Браузер', true, true], [3, 'Слот 4', false, false]]);
  assert.equal(m.current, 'Слот 3 · Браузер');
  assert.deepEqual(m.items.map((i) => i.hint), ['есть профиль', 'пусто', 'есть профиль', 'пусто']);
  assert.equal(slotsModel(status({ slot: 1 }), {}, 'connected', 'en').current, 'Slot 2');
});

test('no picker without a connection or a status, and hostile names stay plain strings', () => {
  assert.equal(slotsModel(status(), {}, 'lost', 'en'), null);
  assert.equal(slotsModel(null, {}, 'connected', 'en'), null);
  const m = slotsModel(status(), { 'slot.0.name': '<img src=x onerror=alert(1)>', 'slot.1.name': 42 }, 'connected', 'en');
  assert.equal(m.items[0].name, '<img src=x onerror=alert(1)>');         // text, drawn with textContent
  assert.equal(m.items[1].name, '');                                      // a number is not a name
  assert.equal(slotsModel(status({ slot: 250 }), {}, 'connected', 'en').current, 'Slot 4');   // an odd status never throws
});

test('every word of the picker exists in both languages', () => {
  for (const k of ['slot.title', 'slot.default', 'slot.current', 'slot.has', 'slot.empty', 'text.save', 'slot.cleared', 'fw.progress', 'fw.staged', 'fw.applying']) {
    assert.ok(STRINGS.ru[k] && STRINGS.en[k], k);
  }
  assert.equal(t('en', 'fw.progress', { pct: 40 }), 'Sending the update: 40 %');
});

// ------------------------------------------------------------------------------------------------------------ the manifest
test('the default manifest offers a slot name, clearing a slot, a copy of all slots and the update controls', () => {
  const v = buildView(manifest, { 'slot.name': 'Работа', 'fw.state': 'staged', 'fw.version': 3 }, 'en');
  const all = [...walk(v.pages.flatMap((p) => p.nodes))];
  const byId = Object.fromEntries(all.map((n) => [n.id, n]));
  assert.deepEqual([byId.slot_name.kind, byId.slot_name.value, byId.slot_name.maxLen, byId.slot_name.known], ['text', 'Работа', 24, true]);
  assert.deepEqual([byId.slot_clear.kind, byId.slot_clear.confirm, byId.slot_clear.danger], ['action', 'two-step', true]);
  assert.deepEqual([byId.export_all.op, byId.export_all.scope, byId.export.scope], ['bundle_get', 'all', 'active']);
  assert.deepEqual([byId.fw_put.op, byId.fw_put.accept], ['fw_put', '.dofw']);
  assert.equal(byId.fw_state.text, 'An update is loaded and waiting to be applied');
  assert.equal(byId.fw_version.text, '3');
  assert.equal(byId.fw_apply.confirm, 'two-step');
});

test('a text control with odd fields is clamped, not trusted', () => {
  const m = { pages: [{ id: 'p', title: { en: 'p' }, controls: [{ id: 'a', type: 'text', key: 'k', label: { en: 'a' }, maxlen: 100000 }, { id: 'b', type: 'text', key: 'z', label: { en: 'b' } }] }] };
  const [a, b] = buildView(m, { k: 5 }, 'en').pages[0].nodes;
  assert.deepEqual([a.maxLen, a.value, a.known], [40, '', false]);
  assert.deepEqual([b.maxLen, b.known], [24, false]);
});

// ------------------------------------------------------------------------------------------------------------ the update upload
class FwDevice {
  constructor(opts = {}) {
    this.reasm = new Reassembler();
    this.seq = 0;
    this.got = [];
    this.up = null;
    this.opts = opts;
    this.dropNextChunkReply = opts.dropReplies ?? 0;
    this.errors = opts.errors ?? [];
    this.messages = [];
    this.onChunk = this.onStatus = this.onClose = () => {};
    this.hello = { v: 1, chunk: 244, device: '01020304', fw: 'x', manifest_rev: 1, manifest_hash: '00000000', trial_s: 20 };
  }
  async open() {
    const info = new Uint8Array(16);
    info[0] = VER;
    return { info, status: packStatus({ bridge: 1, mode: 2 }) };
  }
  close() {}
  async write(chunk) {
    const raw = this.reasm.feed(chunk, 0);
    if (!raw) return;
    const m = unpackMessage(raw);
    this.messages.push(m);
    queueMicrotask(() => this.answer(m));
  }
  send(raw) {
    const { chunks, next } = chunkMessage(raw, 244, this.seq);
    this.seq = next;
    for (const c of chunks) this.onChunk(c);
  }
  ok(m, extra = {}) { this.send(packJson(T.ACK, m.req, { ok: true, ...extra })); }
  fail(m, detail) { this.send(packJson(T.ERR, m.req, { code: ERR.FW_REJECTED, key: 'err.fw_rejected', detail })); }
  answer(m) {
    const j = m.body.length && m.body[0] === 0x7b ? bodyJson(m) : {};
    if (m.type === T.HELLO) this.send(packJson(T.HELLO_R, m.req, this.hello));
    else if (m.type === T.GET && j.what === 'manifest') this.send(packMessage(T.DATA, m.req, new TextEncoder().encode(JSON.stringify({ schema: 1, rev: 1, title: { en: 'a', ru: 'а' }, pages: [] }))));
    else if (m.type === T.GET && j.what === 'state') this.send(packJson(T.DATA, m.req, { rev: 1, state: {} }));
    else if (m.type === T.GET && j.what === 'firmware') this.send(packJson(T.DATA, m.req, { supported: true, upload: this.up ? { next: this.up.next, size: this.up.size } : null }));
    else if (m.type === T.FW_BEGIN) { this.up = { size: j.size, next: 0, buf: [] }; this.ok(m, { next: 0 }); }
    else if (m.type === T.FW_CHUNK) {
      if (this.opts.swallow) return;                                                    // the device hears nothing
      if (this.errors.length) return this.fail(m, this.errors.shift());
      const off = new DataView(m.body.buffer, m.body.byteOffset).getUint32(0, true);
      const data = m.body.subarray(4);
      this.got.push([off, data.length]);
      if (off === this.up.next) { this.up.next += data.length; this.up.buf.push(data.slice()); }
      if (this.dropNextChunkReply > 0) { this.dropNextChunkReply--; return; }          // the acknowledgement never arrives
      this.ok(m, { next: this.up.next });
    } else if (m.type === T.FW_END) this.ok(m, { staged: 2 });
    else if (m.type === T.SET) this.ok(m);
    else this.send(packJson(T.ERR, m.req, { code: ERR.UNSUPPORTED, key: 'err.unsupported', detail: '' }));
  }
}

const image = (n) => Uint8Array.from({ length: n }, (_, i) => (i * 31 + 7) & 0xff);
const timers = { setTimeout, clearTimeout };
const connect = async (dev, extra = {}) => { const s = new Session(dev, { timers, timeoutMs: 400, ...extra }); await s.connect(); return s; };

test('an update is cut into numbered pieces, in order, with progress, and ends with FW_END', async () => {
  const dev = new FwDevice();
  const s = await connect(dev);
  const img = image(FW_CHUNK_MAX * 2 + 1234);
  const seen = [];
  const r = await s.putFirmware(img, (p) => seen.push(p));
  assert.equal(r.json.staged, 2);
  assert.deepEqual(dev.got, [[0, FW_CHUNK_MAX], [FW_CHUNK_MAX, FW_CHUNK_MAX], [FW_CHUNK_MAX * 2, 1234]]);
  assert.deepEqual(Buffer.concat(dev.up.buf), Buffer.from(img));                       // the device got exactly the bytes, nothing added
  assert.equal(seen[0], 0);
  assert.equal(seen.at(-1), 1);
  assert.deepEqual(seen, [...seen].sort((a, b) => a - b));
  const begin = dev.messages.find((m) => m.type === T.FW_BEGIN);
  assert.deepEqual(bodyJson(begin), { size: img.length, name: 'update' });              // no hash computed here: the page has no cryptography
});

test('a lost acknowledgement is asked about and the upload goes on from where the device is', async () => {
  const dev = new FwDevice({ dropReplies: 1 });
  const s = await connect(dev);
  const img = image(FW_CHUNK_MAX + 500);
  const r = await s.putFirmware(img, () => {}, { pieceTimeoutMs: 60 });
  assert.equal(r.json.staged, 2);
  assert.deepEqual(dev.got, [[0, FW_CHUNK_MAX], [FW_CHUNK_MAX, 500]]);                  // piece 1 sent, no answer, asked, then piece 2: nothing sent twice
  assert.deepEqual(Buffer.concat(dev.up.buf), Buffer.from(img));
});

test('an out-of-step answer resynchronises from the device offset; any other refusal ends the upload with its reason', async () => {
  const dev = new FwDevice({ errors: ['sequence'] });
  const s = await connect(dev);
  assert.equal((await s.putFirmware(image(3000))).json.staged, 2);
  const dev2 = new FwDevice({ errors: ['too_large'] });
  const s2 = await connect(dev2);
  await assert.rejects(() => s2.putFirmware(image(3000)), (e) => e instanceof SessionError && e.key === 'err.fw_rejected.too_large');
});

test('it gives up after a few silent pieces instead of waiting for ever', async () => {
  const dev = new FwDevice({ swallow: true });
  const s = await connect(dev, { timeoutMs: 30 });
  await assert.rejects(() => s.putFirmware(image(100), () => {}, { retries: 2, pieceTimeoutMs: 30 }), (e) => e instanceof SessionError);
});

test('selecting a slot and asking for all slots are plain requests; the scope goes through unchanged', async () => {
  const dev = new FwDevice();
  const s = await connect(dev);
  await s.selectSlot(2);
  assert.deepEqual(bodyJson(dev.messages.at(-1)), { key: 'slot.active', value: 2 });
  assert.equal(dev.messages.at(-1).type, T.SET);
  dev.answer = (m) => dev.send(packMessage(T.DATA, m.req, new Uint8Array([1, 2, 3])));
  await s.getBundle('self', 'all');
  assert.deepEqual(bodyJson(dev.messages.at(-1)), { what: 'bundle', for: 'self', scope: 'all' });
  await s.getBundle('self');
  assert.equal(bodyJson(dev.messages.at(-1)).scope, 'active');
});

test('the new device errors have words and the detail picks the right ones', () => {
  assert.equal(errKeyFor(ERR.FW_REJECTED, 'signature'), 'err.fw_rejected.signature');
  assert.equal(errKeyFor(ERR.FW_REJECTED, 'what'), 'err.fw_rejected');
  assert.equal(errKeyFor(ERR.BUSY, 'trial'), 'err.busy.trial');
  assert.equal(errKeyFor(ERR.BUSY, 'calibrating'), 'err.busy');
  assert.equal(errKeyFor(ERR.PHYSICAL, 'slot:2'), 'err.physical.slot');
  assert.equal(errKeyFor(ERR.PHYSICAL, 'fw.apply'), 'err.physical.fw.apply');
  assert.equal(errKeyFor(ERR.PHYSICAL, 'fw.rollback'), 'err.physical.fw.rollback');
  for (const k of Object.keys(STRINGS.en).filter((x) => x.startsWith('err.fw_rejected.') || x.startsWith('err.physical.'))) assert.ok(STRINGS.ru[k], k);
});
