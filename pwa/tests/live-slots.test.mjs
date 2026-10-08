// Slots, the panel's buttons and the two-bank update through the browser client's own code, against the REAL gateway and bridge core
// (`dataopen ctl serve-sim`). Skipped unless DATAOPEN_SIM_URL points at one (tests/test_pwa.py starts it).
import test from 'node:test';
import assert from 'node:assert/strict';
import { Session } from '../js/session.js';
import { WsTransport } from '../js/transport/ws.js';
import { Store } from '../js/store.js';
import { SF, ERR } from '../js/constants.js';

const URL_ = process.env.DATAOPEN_SIM_URL;
const skip = !URL_;
const base = URL_ || 'http://127.0.0.1:0/';
const wsUrl = base.replace(/^http/, 'ws') + 'ws';
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const bridge = async () => (await fetch(base + 'sim/bridge')).json();
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
  const s = new Session(new WsTransport(wsUrl), { store: new Store(null), timeoutMs: 6000, chunk: 180 });
  await s.connect();
  return s;
}
const press = (name, ms) => fetch(`${base}sim/press/${name}/${ms ?? 100}`);

test('the device shows the serial number of its label and a card carrying the manufacturer\'s chain', { skip }, async () => {
  const s = await open();
  assert.match(s.state['device.serial'], /^DO[1-9]-\d{4}-\d{5}-[0-9A-Z]$/);
  const card = await s.getIdentity();
  assert.equal(typeof card.device.att.serial, 'string');
  assert.equal(card.device.att.serial, s.state['device.serial']);
  assert.deepEqual(Object.keys(card.device).sort(), ['att', 'sig']);           // public parts only: no key of the device ever appears
  assert.ok(!JSON.stringify(card).includes('storage'));
  s.close();
});

test('the phone picks a slot; the device says so in the status, the state and its LEDs', { skip }, async () => {
  const s = await open();
  await s.selectSlot(0);
  await s.set('slot.name', 'Работа');
  await until(() => s.state['slot.0.name'] === 'Работа', 3000, 'the name in the state');
  await s.selectSlot(1);
  await until(() => s.status.slot === 1 && s.state['slot.active'] === 1, 3000, 'slot 1 in status and state');
  const b = await bridge();
  assert.equal(b.slot, 1);
  assert.deepEqual(b.leds, [0, 1, 0, 0]);                               // the LED of the active slot, and only that one
  assert.equal(b.assist_wanted, false);                                  // switching a slot never turned assistance on
  assert.ok(s.status.slotMask & 1);                                      // slot 0 holds a profile
  await assert.rejects(s.selectSlot(4), (e) => e.code === ERR.BAD_VALUE);
  await assert.rejects(s.selectSlot('1'), (e) => e.code === ERR.BAD_VALUE);
  await s.selectSlot(0);
  assert.equal((await bridge()).slot, 0);
  assert.equal((await s.getSlots()).slots[0].name, 'Работа');
  s.close();
});

test('the SLOT button on the device moves the slot and the phone hears about it', { skip }, async () => {
  const s = await open();
  await s.selectSlot(0);
  await until(() => s.status.slot === 0, 2000, 'slot 0');
  await press('slot');
  await until(() => s.status.slot !== 0, 3000, 'the phone sees the button');
  const b = await bridge();
  assert.equal(b.slot, s.status.slot);
  assert.equal(b.leds.indexOf(1), b.slot);
  await s.selectSlot(0);
  s.close();
});

test('calibrating into another slot leaves the first profile alone; clearing a slot needs the button and touches only that slot', { skip }, async () => {
  const s = await open();
  await s.stop();
  await s.selectSlot(2);
  await until(() => s.status.slot === 2, 2000, 'slot 2');
  const before = await bridge();
  await s.set('calib.running', true);
  await until(() => s.status.fill >= 90, 9000, 'fill >= 90');
  await assert.rejects(s.selectSlot(0), (e) => e.code === ERR.BUSY);      // not in the middle of learning
  await s.set('calib.running', false);
  await until(async () => (await bridge()).slots[2] === true, 3000, 'a profile in slot 2');
  const mid = await bridge();
  assert.equal(mid.slots[0], before.slots[0]);
  assert.equal(mid.slot, 2);
  await assert.rejects(s.act('slot.clear', true), (e) => e.code === ERR.PHYSICAL && e.key === 'err.physical.slot' && e.params.id === '3');
  await press('confirm');                                               // the hand on the device
  assert.equal((await s.act('slot.clear', true)).json.ok, true);
  const after = await bridge();
  assert.deepEqual([after.slots[2], after.slots[0]], [false, before.slots[0]]);
  assert.equal(after.assist_wanted, false);
  await s.selectSlot(0);
  s.close();
});

test('an update goes in by pieces, waits for the button, proves itself, and can be taken back', { skip }, async () => {
  const s = await open();
  assert.equal(s.state['fw.state'], 'current');
  assert.equal(s.state['fw.version'], 1);
  const image = new Uint8Array(await (await fetch(`${base}sim/image/2/1`)).arrayBuffer());
  const seen = [];
  assert.equal((await s.putFirmware(image, (p) => seen.push(p))).json.staged, 2);
  assert.equal(seen.at(-1), 1);
  await until(() => s.state['fw.state'] === 'staged', 3000, 'staged');
  assert.equal(s.state['fw.version'], 1);                                // still the old one running
  await assert.rejects(s.act('fw.apply', false), (e) => e.code === ERR.NOT_ALLOWED);
  await assert.rejects(s.act('fw.apply', true), (e) => e.code === ERR.PHYSICAL && e.key === 'err.physical.fw.apply');
  await press('confirm');
  await s.act('fw.apply', true);
  await until(() => s.conn === 'lost', 4000, 'the device restarting drops the link');
  const s2 = await until(async () => { try { return await open(); } catch { return null; } }, 8000, 'the device back');
  assert.equal(s2.state['fw.version'], 2);
  assert.equal(s2.state['fw.state'], 'trial');                           // on probation until it has proved itself
  await until(() => s2.state['fw.state'] === 'current', 15000, 'confirmed by itself');
  assert.equal((await bridge()).fw.version, 2);
  // a refused image says why, and nothing changes
  await assert.rejects(s2.putFirmware(new Uint8Array(500).fill(7)), (e) => e.key === 'err.fw_rejected.damaged');
  const same = new Uint8Array(await (await fetch(`${base}sim/image/2/1`)).arrayBuffer());
  await assert.rejects(s2.putFirmware(same), (e) => e.key === 'err.fw_rejected.same');
  // and the person can go back to what ran before (the button again)
  await assert.rejects(s2.act('fw.rollback', true), (e) => e.code === ERR.PHYSICAL && e.key === 'err.physical.fw.rollback');
  await press('confirm');
  await s2.act('fw.rollback', true);
  await until(() => s2.conn === 'lost', 4000, 'the device restarting again');
  const s3 = await until(async () => { try { return await open(); } catch { return null; } }, 8000, 'the device back again');
  assert.equal(s3.state['fw.version'], 1);
  assert.equal((await bridge()).fw.running, 'A');
  s3.close();
});

test('the safety shell never depends on slots or updates: stop still works at once', { skip }, async () => {
  const s = await open();
  await s.stop();
  await until(async () => (await bridge()).reason === 'CMD_PASSTHRU', 5000, 'passthru');
  assert.ok(!(s.status.flags & SF.ASSIST_WANTED));
  s.close();
});
