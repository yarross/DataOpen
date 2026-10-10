// Channel B (packages for the slots) through the browser client's own code, against the REAL gateway and bridge core (`dataopen ctl serve-sim`).
// Skipped unless DATAOPEN_SIM_URL points at one (tests/test_pwa.py starts it). The packages come from a test sender the dev server makes up.
import test from 'node:test';
import assert from 'node:assert/strict';
import { Session, SessionError } from '../js/session.js';
import { WsTransport } from '../js/transport/ws.js';
import { Store } from '../js/store.js';
import { ERR } from '../js/constants.js';

const URL_ = process.env.DATAOPEN_SIM_URL;
const skip = !URL_;
const base = URL_ || 'http://127.0.0.1:0/';
const wsUrl = base.replace(/^http/, 'ws') + 'ws';
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
async function until(fn, ms = 6000, what = 'condition') {
  const end = Date.now() + ms;
  for (;;) {
    const v = await fn();
    if (v) return v;
    if (Date.now() > end) throw new Error(`timed out waiting for ${what}`);
    await sleep(25);
  }
}
async function open() {
  const s = new Session(new WsTransport(wsUrl), { store: new Store(null), timeoutMs: 8000, chunk: 180 });
  await s.connect();
  return s;
}
const make = async (kind, seq, slot) => new Uint8Array(await (await fetch(`${base}sim/package/${kind}/${seq}${slot === undefined ? '' : '/' + slot}`)).arrayBuffer());
const press = () => fetch(base + 'sim/button');
const reason = (key) => (e) => e instanceof SessionError && e.code === ERR.PKG_REJECTED && e.key === `err.pkg_rejected.${key}`;
let seq = 100 + (Date.now() % 1000) * 10;                         // the sender's counter only grows, and the device may have seen earlier runs
const next = () => ++seq;

test('a forged, a misaddressed and a tampered package are refused, the first two with the very first piece', { skip }, async () => {
  const s = await open();
  const seen = [];
  await assert.rejects(async () => s.putPackage(await make('forged', next()), (p) => seen.push(p)), reason('bad_signature'));
  assert.deepEqual(seen, [0]);                                      // the header was judged with the first piece: nothing else was sent
  seen.length = 0;
  await assert.rejects(async () => s.putPackage(await make('other', next()), (p) => seen.push(p)), reason('wrong_device'));
  assert.deepEqual(seen, [0]);
  await assert.rejects(async () => s.putPackage(await make('tampered', next())), reason('tampered'));
  const info = await s.getPackages();
  assert.equal(info.pending, null);
  assert.equal(info.upload, null);                                   // and nothing half-received is left
  s.close();
});

test('a package from a stranger waits, asks for the button on apply, and then changes the slot', { skip }, async () => {
  const s = await open();
  const n = next();
  const r = await s.putPackage(await make('tuning', n, 2));
  assert.deepEqual(r.json.pending.kinds, ['tuning', 'meta']);
  assert.equal(r.json.pending.slot, 2);
  assert.equal(r.json.pending.button, 'trust');
  await until(() => s.state['pkg.state'] === 'pending', 3000, 'the pending state');
  assert.equal(s.state['pkg.kinds'], 'tuning,meta');
  assert.notEqual((await s.getSlots()).slots[2].name, 'Из приложения');           // received is not applied
  await assert.rejects(() => s.act('pkg.apply', true), (e) => e.code === ERR.PHYSICAL && String(e.detail).startsWith('trust:'));
  assert.equal((await s.getPackages()).pending.id, r.json.pending.id);            // the refusal applied nothing and kept the package
  await press();
  await s.act('pkg.apply', true);
  await until(() => s.state['pkg.state'] === 'none', 3000, 'no pending package');
  assert.equal((await s.getSlots()).slots[2].name, 'Из приложения');
  await assert.rejects(async () => s.putPackage(await make('tuning', n, 2)), reason('replay'));      // the same sender, the same number
  s.close();
});

test('weights need the button even from a known sender, are kept per slot, and can be swapped back', { skip }, async () => {
  const s = await open();
  await s.selectSlot(0);
  await s.putPackage(await make('model', next(), 0));
  await assert.rejects(() => s.act('pkg.apply', true), (e) => e.code === ERR.PHYSICAL && String(e.detail).startsWith('model:'));
  await press();
  await s.act('pkg.apply', true);
  await until(() => s.state['model.state'] === 'ok', 3000, 'the model state');
  assert.equal(s.state['model.name'], 'icons');
  const v1 = s.state['model.version'];
  await assert.rejects(() => s.act('pkg.revert', true), reason('no_previous'));          // only one generation so far
  await s.putPackage(await make('model', next(), 0));
  await press();
  await s.act('pkg.apply', true);
  await until(() => s.state['model.version'] > v1, 3000, 'the new model');
  const v2 = s.state['model.version'];
  await s.act('pkg.revert', true);
  await until(() => s.state['model.version'] === v1, 3000, 'the previous model back');
  await s.act('pkg.revert', true);
  await until(() => s.state['model.version'] === v2, 3000, 'and forward again');
  const info = await s.getPackages();
  assert.equal(info.models[0].state, 'ok');
  assert.equal(info.models[0].previous, true);
  s.close();
});

test('the two channels do not stand in for each other', { skip }, async () => {
  const s = await open();
  const image = new Uint8Array(await (await fetch(`${base}sim/image/9`)).arrayBuffer());
  await assert.rejects(() => s.putPackage(image), reason('wrong_channel'));            // a system image sent as a package
  const pkg = await make('tuning', next(), 1);
  await assert.rejects(() => s.putFirmware(pkg), (e) => e instanceof SessionError && e.code === ERR.FW_REJECTED && e.key === 'err.fw_rejected.wrong_channel');
  assert.equal((await s.getFirmware()).upload, null);
  assert.equal((await s.getPackages()).upload, null);
  s.close();
});

test('a loaded package can be dropped without a trace', { skip }, async () => {
  const s = await open();
  await s.putPackage(await make('tuning', next(), 3));
  await until(() => s.state['pkg.state'] === 'pending', 3000, 'pending');
  await s.act('pkg.discard', false);
  await until(() => s.state['pkg.state'] === 'none', 3000, 'none');
  assert.equal((await s.getPackages()).pending, null);
  s.close();
});

test('a model the device will not take is refused with its own reason, nothing is kept, nobody becomes trusted', { skip }, async () => {
  const s = await open();
  for (const [kind, key] of [['ops', 'model_ops'], ['files', 'model_files'], ['io', 'model_io'], ['heavy', 'model_cost'], ['pose', 'not_ui_model']]) {
    await assert.rejects(async () => s.putPackage(await make(kind, next(), 0)), reason(key), kind);
    const info = await s.getPackages();
    assert.equal(info.pending, null, kind);
    assert.equal(info.upload, null, kind);
  }
  s.close();
});

test('a model can be taken out of the slot by itself, with no button, and the number of who brought it is shown', { skip }, async () => {
  const s = await open();
  await s.selectSlot(0);
  await s.putPackage(await make('model', next(), 0));
  await press();
  await s.act('pkg.apply', true);
  await until(() => s.state['model.state'] === 'ok', 3000, 'the model state');
  assert.match(s.state['model.from'], /^[0-9A-Z]{4}(-[0-9A-Z]{4}){3}$/);
  await assert.rejects(() => s.act('model.clear', false), (e) => e.code === ERR.NOT_ALLOWED);          // two steps: the first only asks
  await s.act('model.clear', true);
  await until(() => s.state['model.state'] === 'none', 3000, 'no model');
  assert.equal(s.state['model.from'], '');
  assert.equal((await s.getPackages()).models[0].previous, false);                                  // the previous copy went with it
  await assert.rejects(() => s.act('model.clear', true), reason('no_model'));
  s.close();
});

test('forgetting the trusted senders needs the button, and the next package asks again', { skip }, async () => {
  const s = await open();
  await assert.rejects(() => s.act('trust.clear', true), (e) => e.code === ERR.PHYSICAL && e.detail === 'trust.clear');
  await press();
  await s.act('trust.clear', true);
  await until(() => s.state['trusted.count'] === 0, 3000, 'no trusted sender');
  await s.putPackage(await make('tuning', next(), 3));
  await assert.rejects(() => s.act('pkg.apply', true), (e) => e.code === ERR.PHYSICAL && String(e.detail).startsWith('trust:'));
  await s.act('pkg.discard', false);
  s.close();
});
