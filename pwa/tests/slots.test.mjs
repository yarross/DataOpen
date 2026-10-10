// The phone client's side of slots and updates: the picker model, the text control, the firmware upload against a scripted device.
// The page stays dumb: it names the slots the device announces and carries opaque bytes; it decides nothing.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { slotsModel } from '../js/shell.js';
import { buildView, walk } from '../js/view.js';
import { Session, SessionError } from '../js/session.js';
import { T, VER, ERR, FW_CHUNK_MAX, PKG_CHUNK_MAX, SLOT_COUNT } from '../js/constants.js';
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
test('the default manifest offers a slot name, clearing a slot and the update controls, and nothing that takes data out', () => {
  const v = buildView(manifest, { 'slot.name': 'Работа', 'fw.state': 'staged', 'fw.version': 3 }, 'en');
  const all = [...walk(v.pages.flatMap((p) => p.nodes))];
  const byId = Object.fromEntries(all.map((n) => [n.id, n]));
  assert.deepEqual([byId.slot_name.kind, byId.slot_name.value, byId.slot_name.maxLen, byId.slot_name.known], ['text', 'Работа', 24, true]);
  assert.deepEqual([byId.slot_clear.kind, byId.slot_clear.confirm, byId.slot_clear.danger], ['action', 'two-step', true]);
  assert.ok(!byId.export && !byId.export_all && !byId.export_other);                 // no copy, no 'for another device' (docs/RESIDENCY.md)
  assert.deepEqual(all.filter((n) => n.kind === 'file').map((n) => n.op).sort(), ['bundle_put', 'card_get', 'fw_put', 'pkg_put']);
  assert.deepEqual([byId.fw_put.op, byId.fw_put.accept], ['fw_put', '.dofw']);
  assert.equal(byId.fw_state.text, 'Loaded, waiting to be applied');
  assert.equal(byId.fw_version.text, '3');
  assert.equal(byId.fw_apply.confirm, 'two-step');
  // channel B, apart from the system controls: its own file op, its own actions
  assert.deepEqual([byId.pkg_put.op, byId.pkg_put.accept], ['pkg_put', '.dopk']);
  assert.deepEqual([byId.pkg_apply.confirm, byId.pkg_discard.confirm, byId.model_revert.confirm], ['two-step', 'none', 'two-step']);
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
  fail(m, detail) {
    const pkg = m.type === T.PKG_CHUNK || m.type === T.PKG_BEGIN || m.type === T.PKG_END;
    this.send(packJson(T.ERR, m.req, { code: pkg ? ERR.PKG_REJECTED : ERR.FW_REJECTED, key: pkg ? 'err.pkg_rejected' : 'err.fw_rejected', detail }));
  }
  answer(m) {
    const j = m.body.length && m.body[0] === 0x7b ? bodyJson(m) : {};
    if (m.type === T.HELLO) this.send(packJson(T.HELLO_R, m.req, this.hello));
    else if (m.type === T.GET && j.what === 'manifest') this.send(packMessage(T.DATA, m.req, new TextEncoder().encode(JSON.stringify({ schema: 1, rev: 1, title: { en: 'a', ru: 'а' }, pages: [] }))));
    else if (m.type === T.GET && j.what === 'state') this.send(packJson(T.DATA, m.req, { rev: 1, state: {} }));
    else if (m.type === T.GET && j.what === 'firmware') this.send(packJson(T.DATA, m.req, { supported: true, upload: this.up ? { next: this.up.next, size: this.up.size } : null }));
    else if (m.type === T.GET && j.what === 'packages') this.send(packJson(T.DATA, m.req, { supported: true, upload: this.up ? { next: this.up.next, size: this.up.size } : null }));
    else if (m.type === T.FW_BEGIN || m.type === T.PKG_BEGIN) { this.up = { size: j.size, next: 0, buf: [] }; this.ok(m, { next: 0 }); }
    else if (m.type === T.FW_CHUNK || m.type === T.PKG_CHUNK) {
      if (this.opts.swallow) return;                                                    // the device hears nothing
      if (this.errors.length) return this.fail(m, this.errors.shift());
      const off = new DataView(m.body.buffer, m.body.byteOffset).getUint32(0, true);
      const data = m.body.subarray(4);
      this.got.push([off, data.length]);
      if (off === this.up.next) { this.up.next += data.length; this.up.buf.push(data.slice()); }
      if (this.dropNextChunkReply > 0) { this.dropNextChunkReply--; return; }          // the acknowledgement never arrives
      this.ok(m, { next: this.up.next });
    } else if (m.type === T.FW_END) this.ok(m, { staged: 2 });
    else if (m.type === T.PKG_END) this.ok(m, { pending: { kinds: ['tuning'], from: 'AAAA-BBBB', button: 'trust' } });
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

test('selecting a slot is a plain request; asking the device for a copy of a slot is not a request at all', async () => {
  const dev = new FwDevice();
  const s = await connect(dev);
  await s.selectSlot(2);
  assert.deepEqual(bodyJson(dev.messages.at(-1)), { key: 'slot.active', value: 2 });
  assert.equal(dev.messages.at(-1).type, T.SET);
  const sent = dev.messages.length;
  assert.equal(typeof s.getBundle, 'undefined');
  for (const what of ['bundle', 'profile', 'model', 'weights', 'export']) {
    await assert.rejects(s.request(T.GET, { what, for: 'self', scope: 'all' }), (e) => e.code === ERR.RESIDENT);
  }
  assert.equal(dev.messages.length, sent);                                         // nothing was written to the device for any of them
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

// ---------------------------------------------------------------------------------------------------------------- channel B: packages
test('a package goes in its own channel, in numbered pieces, and the begin message carries a size and nothing else', async () => {
  const dev = new FwDevice();
  const s = await connect(dev);
  const pkg = image(PKG_CHUNK_MAX * 2 + 700);
  const seen = [];
  const r = await s.putPackage(pkg, (p) => seen.push(p));
  assert.deepEqual(r.json.pending.kinds, ['tuning']);
  assert.deepEqual(dev.got, [[0, PKG_CHUNK_MAX], [PKG_CHUNK_MAX, PKG_CHUNK_MAX], [PKG_CHUNK_MAX * 2, 700]]);
  assert.deepEqual(Buffer.concat(dev.up.buf), Buffer.from(pkg));
  assert.deepEqual(bodyJson(dev.messages.find((m) => m.type === T.PKG_BEGIN)), { size: pkg.length });
  assert.ok(!dev.messages.some((m) => m.type === T.FW_BEGIN || m.type === T.FW_CHUNK || m.type === T.FW_END));   // never through the system channel
  assert.equal(seen.at(-1), 1);
});

test('a package refused by its header ends the upload at once with the reason in words, and an out-of-step answer resynchronises', async () => {
  const dev = new FwDevice({ errors: ['wrong_device'] });
  const s = await connect(dev);
  await assert.rejects(() => s.putPackage(image(PKG_CHUNK_MAX * 3)), (e) => e instanceof SessionError && e.key === 'err.pkg_rejected.wrong_device' && e.code === ERR.PKG_REJECTED);
  assert.deepEqual(dev.got, []);                                                       // refused with the first piece: nothing else was sent
  const dev2 = new FwDevice({ errors: ['sequence'] });
  const s2 = await connect(dev2);
  assert.deepEqual((await s2.putPackage(image(3000))).json.pending.kinds, ['tuning']);
});

test('a lost acknowledgement of a package piece is asked about, not sent twice', async () => {
  const dev = new FwDevice({ dropReplies: 1 });
  const s = await connect(dev);
  const pkg = image(PKG_CHUNK_MAX + 500);
  await s.putPackage(pkg, () => {}, { pieceTimeoutMs: 60 });
  assert.deepEqual(dev.got, [[0, PKG_CHUNK_MAX], [PKG_CHUNK_MAX, 500]]);
  assert.deepEqual(Buffer.concat(dev.up.buf), Buffer.from(pkg));
});
