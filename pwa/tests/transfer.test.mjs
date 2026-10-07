// A profile moves from device A to device B through the browser client's own code: two real gateways (with the real bridge core behind each),
// two sessions, and a file in between that the client only ever carries. Needs DATAOPEN_SIM_URL (A, with a profile) and DATAOPEN_SIM_URL_B
// (B, without one); tests/test_pwa.py starts both.
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

test('A to B: the button on each side, the file opaque in between, replay and strangers refused', { skip }, async () => {
  const a = await open(A), b = await open(B);
  const cardB = await b.getIdentity();
  assert.equal(b.state['profile.fill'], 0);
  assert.ok(a.state['profile.fill'] > 0);

  // 1. a file for ANOTHER device needs the button on the source, and the message names who it is for
  await assert.rejects(a.getBundle(cardB), (e) => e.code === ERR.PHYSICAL && e.key === 'err.physical.export' && e.params.id === cardB.id);
  await press(A);
  const raw = await a.getBundle(cardB);
  assert.equal(String.fromCharCode(...raw.slice(0, 4)), 'DOBS');
  assert.ok(!String.fromCharCode(...raw).includes('BIOP'));

  // 2. the receiver does not know the sender: it asks for ITS button and names the sender
  await assert.rejects(b.putBundle(raw), (e) => e.code === ERR.PHYSICAL && e.key === 'err.physical.trust' && e.params.id === a.state['device.id']);
  assert.equal(b.state['profile.fill'], 0);
  assert.equal(b.state['trusted.count'], 0);

  // 3. a damaged copy is refused without spending the press; the real one goes through
  await press(B);
  const bad = raw.slice();
  bad[180] ^= 4;
  await assert.rejects(b.putBundle(bad), (e) => e.key === 'err.bad_signature');
  assert.equal((await b.putBundle(raw)).json.ok, true);
  assert.equal(b.state['trusted.count'], 1);
  assert.equal(b.state['profile.fill'], a.state['profile.fill']);

  // 4. the same file again, and the file on any other device, are refused
  await assert.rejects(b.putBundle(raw), (e) => e.key === 'err.replay');
  await press(A);
  await assert.rejects(a.putBundle(raw), (e) => e.key === 'err.wrong_device');

  // 5. a new file from the now-trusted sender needs no button
  await press(A);
  const raw2 = await a.getBundle(cardB);
  assert.equal((await b.putBundle(raw2)).json.ok, true);
  await b.stop();
  a.close();
  b.close();
});

test('erasing on B takes the profile away, and the old file no longer works there', { skip }, async () => {
  const b = await open(B);
  const copy = await b.getBundle('self');
  await assert.rejects(b.act('erase.profile', true), (e) => e.key === 'err.physical.erase');
  await press(B);
  await b.act('erase.profile', true);
  assert.equal(b.state['profile.fill'], 0);
  assert.equal(b.state['trusted.count'], 0);
  assert.equal((await b.putBundle(copy)).json.ok, true);                 // its own copy still opens: the identity survives a data erase
  await press(B);
  const idBefore = b.state['device.id'];
  await b.act('factory.reset', true);
  assert.notEqual((await b.getIdentity()).id, idBefore);
  await assert.rejects(b.putBundle(copy), (e) => e.key === 'err.wrong_device');   // after a factory reset it is a different device
  b.close();
});
