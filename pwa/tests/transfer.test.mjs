// Files come IN, never out (docs/RESIDENCY.md): a settings file made by a sender (not by a device) goes to a device through the browser
// client's own code: a real gateway with the real bridge core behind it, a session, and a file in between that the client only ever
// carries. Needs DATAOPEN_SIM_URL (A, with a profile) and DATAOPEN_SIM_URL_B (B, without one); tests/test_pwa.py starts both.
import test from 'node:test';
import assert from 'node:assert/strict';
import { Session } from '../js/session.js';
import { WsTransport } from '../js/transport/ws.js';
import { Store } from '../js/store.js';
import { ERR } from '../js/constants.js';

const A = process.env.DATAOPEN_SIM_URL, B = process.env.DATAOPEN_SIM_URL_B;
const skip = !(A && B);
const open = async (base) => {
  const s = new Session(new WsTransport((base || 'http://x/').replace(/^http/, 'ws') + 'ws'), { store: new Store(null), timeoutMs: 4000, chunk: 180 });
  await s.connect();
  return s;
};
const press = (base) => fetch(base + 'sim/button');
const file = async (base, kind, seq) => new Uint8Array(await (await fetch(`${base}sim/bundle/${kind}/${seq}`)).arrayBuffer());
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const until = async (fn, what) => { for (let i = 0; i < 200; i++) { if (await fn()) return; await sleep(25); } throw new Error(`timed out waiting for ${what}`); };

test('a file from a sender: the button on the device, the file opaque in between, replay and other devices refused', { skip }, async () => {
  const a = await open(A), b = await open(B);
  await b.stop();
  assert.equal(b.state['profile.fill'], 0);
  const raw = await file(B, 'profile', 1);
  assert.equal(String.fromCharCode(...raw.slice(0, 4)), 'DOBS');
  assert.ok(!String.fromCharCode(...raw).includes('BIOP'));

  // 1. the receiver does not know the sender: it asks for ITS button and names the sender
  await assert.rejects(b.putBundle(raw), (e) => e.code === ERR.PHYSICAL && e.key === 'err.physical.trust' && typeof e.params.id === 'string');
  assert.equal(b.state['profile.fill'], 0);
  assert.equal(b.state['trusted.count'], 0);

  // 2. a damaged copy is refused without spending the press; the real one goes through
  await press(B);
  const bad = raw.slice();
  bad[180] ^= 4;
  await assert.rejects(b.putBundle(bad), (e) => e.key === 'err.bad_signature');
  assert.equal((await b.putBundle(raw)).json.ok, true);
  assert.equal(b.state['trusted.count'], 1);
  assert.ok(b.state['profile.fill'] > 0);

  // 3. the same file again, and a file made for another device, are refused
  await assert.rejects(b.putBundle(raw), (e) => e.key === 'err.replay');
  await press(A);
  await assert.rejects(a.putBundle(raw), (e) => e.key === 'err.wrong_device');

  // 4. a new file from the now-trusted sender needs no button
  assert.equal((await b.putBundle(await file(B, 'profile', 2))).json.ok, true);
  await b.stop();
  a.close();
  b.close();
});

test('erasing takes the profile away; after a factory reset a file made for the old owner no longer works', { skip }, async () => {
  const b = await open(B);
  const old = await file(B, 'profile', 3);
  await assert.rejects(b.act('erase.profile', true), (e) => e.key === 'err.physical.erase');
  await press(B);
  await b.act('erase.profile', true);
  assert.equal(b.state['profile.fill'], 0);
  assert.equal(b.state['trusted.count'], 0);
  await assert.rejects(b.putBundle(old), (e) => e.key === 'err.physical.trust');   // the owner is the same, the senders are forgotten
  await press(B);
  assert.equal((await b.putBundle(old)).json.ok, true);
  await press(B);
  const idBefore = b.state['device.id'];
  await b.act('factory.reset', true);
  assert.notEqual((await b.getIdentity()).id, idBefore);
  await assert.rejects(b.putBundle(old), (e) => e.key === 'err.wrong_device');   // after a factory reset it is a different device
  b.close();
});

test('one file for ALL slots puts every slot in by number, names included, and only that device opens it', { skip }, async () => {
  const a = await open(A), b = await open(B);
  await b.stop();
  const raw = await file(B, 'slots', 4);
  assert.equal(String.fromCharCode(...raw.slice(0, 4)), 'DOBS');
  assert.ok(!String.fromCharCode(...raw).includes('BIOP') && !String.fromCharCode(...raw).includes('Браузер'));   // nothing readable in it
  await press(A);
  await assert.rejects(a.putBundle(raw), (e) => e.key === 'err.wrong_device');      // not on another device
  await press(B);                                                                    // B has just been reset: the sender is new to it
  assert.equal((await b.putBundle(raw)).json.ok, true);
  await until(() => b.state['slot.0.name'] === 'Работа' && b.state['slot.2.name'] === 'Браузер', 'the names on B');
  assert.equal(b.status.slotMask & 0b1111, 0b0101);                                  // slots 0 and 2 hold a profile on B, 1 and 3 do not
  await b.selectSlot(2);
  await until(() => b.status.slot === 2, 'B on slot 2');
  assert.ok(b.status.fill > 0 || b.state['profile.fill'] > 0);
  await b.selectSlot(0);
  a.close();
  b.close();
});
