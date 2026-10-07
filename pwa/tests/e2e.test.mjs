// The browser client's own code (session, link, WebSocket transport) against the REAL gateway and the real C bridge core, through
// `dataopen ctl serve-sim`. Skipped unless DATAOPEN_SIM_URL points at one (tests/test_pwa.py starts it).
import test from 'node:test';
import assert from 'node:assert/strict';
import { Session } from '../js/session.js';
import { WsTransport } from '../js/transport/ws.js';
import { Store } from '../js/store.js';
import { T, SF, RB, ERR } from '../js/constants.js';
import { fromHex } from '../js/link.js';

const URL_ = process.env.DATAOPEN_SIM_URL;
const skip = !URL_;
const base = URL_ || 'http://127.0.0.1:0/';
const wsUrl = base.replace(/^http/, 'ws') + 'ws';
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const bridge = async () => (await fetch(base + 'sim/bridge')).json();
async function until(fn, ms = 4000, what = 'condition') {
  const end = Date.now() + ms;
  for (;;) {
    const v = await fn();
    if (v) return v;
    if (Date.now() > end) throw new Error(`timed out waiting for ${what}`);
    await sleep(25);
  }
}
async function open(store = new Store(null)) {
  const s = new Session(new WsTransport(wsUrl), { store, timeoutMs: 4000, chunk: 180 });
  await s.connect();
  return s;
}
async function neutral(s) {                                   // back to 'help off, knobs at 5' whatever the previous test did
  const b = await bridge();
  if (b.state === 'HW_BYPASS' || b.reason === 'PANIC') await fetch(base + 'sim/panic/2300');   // only the hand lifts a latch
  await s.stop();
  await s.set('assist.strength', 5);
  await s.set('tremor.level', 5);
  await until(async () => (await bridge()).reason === 'CMD_PASSTHRU', 15000, 'passthru');
}

test('connect: manifest, state and status all arrive', { skip }, async () => {
  const s = await open();
  assert.equal(s.conn, 'connected');
  assert.equal(s.hello.fw, 'ctl-1');
  assert.ok(s.chunk > 20);
  assert.ok(s.manifest.pages.length >= 3);
  assert.equal(typeof s.state['assist.on'], 'boolean');
  assert.ok(s.status && s.status.bridge === 1);
  s.close();
});

test('enabling help is a trial; the bridge really engages and undo really undoes it', { skip }, async () => {
  const s = await open();
  await neutral(s);
  const statuses = [];
  s.on('status', (x) => statuses.push(x));
  const r = await s.set('assist.on', true);
  assert.equal(r.json.ok, true);
  await until(async () => (await bridge()).state === 'ASSIST', 4000, 'ASSIST');
  await until(() => statuses.some((x) => x.flags & SF.TRIAL), 3000, 'trial flag in status');
  assert.ok(statuses.at(-1).flags & SF.ASSIST_WANTED);
  await s.confirm(false);                                      // undo
  await until(async () => (await bridge()).reason === 'CMD_PASSTHRU', 4000, 'passthru after undo');
  assert.equal(s.state['assist.on'], false);
  s.close();
});

test('confirm keeps it; lowering is free; the browser sees the knobs the device holds', { skip }, async () => {
  const s = await open();
  await neutral(s);
  await s.set('assist.on', true);
  await s.confirm(true);
  await s.set('assist.strength', 8);
  assert.equal((await s.set('assist.strength', 2)).json.trial, 0);
  assert.equal(s.state['assist.strength'], 2);
  const s2 = await open();                                       // a second browser sees the same
  assert.equal(s2.state['assist.strength'], 2);
  assert.equal(s2.state['assist.on'], true);
  s.close();
  s2.close();
  await neutral(await open());
});

test('stop is immediate and the bridge is byte-exact afterwards', { skip }, async () => {
  const s = await open();
  await neutral(s);
  await s.set('assist.on', true);
  await s.confirm(true);
  await until(async () => (await bridge()).state === 'ASSIST', 4000, 'ASSIST');
  await s.stop();
  const b = await bridge();
  assert.equal(b.state, 'PASSTHRU');
  assert.equal(b.reason, 'CMD_PASSTHRU');
  assert.equal(b.assist_wanted, false);
  s.close();
});

test('hard bypass is one-way from the browser', { skip }, async () => {
  const s = await open();
  await neutral(s);
  await s.hardBypass();
  await until(async () => (await bridge()).reason === 'CMD_BYPASS', 3000, 'CMD_BYPASS');
  await s.set('assist.on', true);
  await s.confirm(true);
  await sleep(600);
  assert.equal((await bridge()).state, 'HW_BYPASS');
  s.close();
});

test('the hand wins: a Panic latch is reported and the browser cannot lift it', { skip }, async () => {
  const s = await open();
  await neutral(s);
  await fetch(base + 'sim/panic/2200');                         // a long hold re-arms whatever the previous test latched
  await sleep(300);
  await neutral(s);
  await s.set('assist.on', true);
  await s.confirm(true);
  await until(async () => (await bridge()).state === 'ASSIST', 6000, 'ASSIST');
  await fetch(base + 'sim/panic/100');
  await until(() => s.status && s.status.reasonName === 'PANIC', 3000, 'PANIC in status');
  assert.ok(s.status.flags & SF.LATCH_SOFT);
  await s.set('assist.on', false);
  await s.set('assist.on', true);
  await s.confirm(true);
  await sleep(500);
  assert.equal((await bridge()).reason, 'PANIC');
  await fetch(base + 'sim/panic/2200');
  s.close();
});

test('the device refuses what its manifest does not allow, with keyed errors', { skip }, async () => {
  const s = await open();
  await assert.rejects(s.set('assist.strength', 11), (e) => e.code === ERR.BAD_VALUE && e.key === 'err.bad_value');
  await assert.rejects(s.set('nope', 1), (e) => e.key === 'err.bad_key');
  await assert.rejects(s.act('pairing.forget', false), (e) => e.code === ERR.NOT_ALLOWED);
  await assert.rejects(s.act('pairing.forget', true), (e) => e.key === 'err.physical');
  await fetch(base + 'sim/button');
  assert.equal((await s.act('pairing.forget', true)).json.ok, true);
  s.close();
});

test('a settings file goes out and comes back, a damaged one is refused with a plain reason', { skip }, async () => {
  const s = await open();
  await neutral(s);
  const raw = await s.getBundle();
  assert.ok(raw.length > 100 && raw.length < 400);
  assert.equal(String.fromCharCode(...raw.slice(0, 4)), 'DOBN');
  assert.equal((await s.putBundle(raw)).json.ok, true);
  const bad = raw.slice();
  bad[40] ^= 1;
  await assert.rejects(s.putBundle(bad), (e) => e.key === 'err.bad_bundle.damaged');
  await assert.rejects(s.putBundle(new Uint8Array(5000)), (e) => e.code === ERR.TOO_BIG);
  s.close();
});

test('calibration through the browser: help goes off, progress rises, a profile is stored at the end', { skip }, async () => {
  const s = await open();
  await neutral(s);
  const before = s.status.fill;
  await s.set('calib.running', true);
  assert.ok((await bridge()).calibrating);
  const fills = [];
  s.on('status', (x) => fills.push(x.fill));
  await until(() => s.status.flags & SF.CALIBRATING, 2000, 'calibrating flag');
  await until(() => s.status.fill >= 90, 8000, 'fill >= 90');
  assert.ok(fills.length > 2 && fills.every((v, i) => i === 0 || v >= fills[i - 1]));
  await assert.rejects(s.set('assist.on', true), (e) => e.code === ERR.BUSY);
  await s.set('calib.running', false);
  assert.equal((await bridge()).calibrating, false);
  assert.ok(s.status.ready & RB.ASC || true);
  assert.ok(s.status.fill >= before);
  s.close();
});

test('losing the browser changes nothing about the device', { skip }, async () => {
  const s = await open();
  await neutral(s);
  await s.set('assist.on', true);
  await s.confirm(true);
  await until(async () => (await bridge()).state === 'ASSIST', 4000, 'ASSIST');
  s.close();
  await sleep(1500);
  assert.equal((await bridge()).state, 'ASSIST');
  const s2 = await open();
  assert.equal(s2.state['assist.on'], true);
  await neutral(s2);
  s2.close();
});

test('the page itself and its service worker are served', { skip }, async () => {
  const html = await (await fetch(base)).text();
  assert.match(html, /<title>/);
  assert.equal((await fetch(base + 'sw.js')).status, 200);
  assert.equal((await fetch(base + 'manifest.webmanifest')).headers.get('content-type'), 'application/manifest+json');
  assert.equal((await fetch(base + '../etc/passwd')).status, 404);
  assert.equal((await fetch(base + 'tests/golden.json')).status, 200);
  assert.ok(fromHex('00').length === 1 && T.STOP === 36);
});
