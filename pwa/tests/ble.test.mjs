// BleTransport against a fake Web Bluetooth that has the real API's shape (requestDevice, getDevices, gatt, characteristics, DataView
// values, characteristicvaluechanged events). With DATAOPEN_SIM_URL set the fake is wired to the real gateway through the simulator's
// WebSocket, so the Session runs over the BLE code path end to end.
import test from 'node:test';
import assert from 'node:assert/strict';
import { BleTransport } from '../js/transport/ble.js';
import { Session } from '../js/session.js';
import { Store } from '../js/store.js';
import { UUID, T, SF } from '../js/constants.js';
import { packStatus, unpackStatus } from '../js/link.js';

class FakeChar extends EventTarget {
  constructor(uuid, o = {}) { super(); Object.assign(this, { uuid, value: null, notifying: false, writes: [] }, o); }
  async readValue() { return this.value; }
  async startNotifications() { this.notifying = true; return this; }
  push(u8, offset = 3) {                                     // a DataView over a bigger buffer, as the browser gives them
    const buf = new Uint8Array(u8.length + offset + 5);
    buf.set(u8, offset);
    this.value = new DataView(buf.buffer, offset, u8.length);
    const ev = new Event('characteristicvaluechanged');
    Object.defineProperty(ev, 'target', { value: this });
    this.dispatchEvent(ev);
  }
}

function fakeBluetooth({ maxWrite = 244, withoutResponse = true, known = true, gate = null } = {}) {
  const log = { requested: [], connects: 0 };
  const info = new FakeChar(UUID.info), status = new FakeChar(UUID.status), cin = new FakeChar(UUID.ctlIn), cout = new FakeChar(UUID.ctlOut);
  info.value = new DataView(Uint8Array.from([1, 3, 1, 2, 3, 4, 0xde, 0xad, 0xbe, 0xef, 1, 0, 244, 0, 0, 0]).buffer);
  status.value = new DataView(packStatus({ bridge: 1, mode: 2 }).buffer);
  const write = async (c) => { if (c.length > maxWrite) throw new DOMException('value too long', 'NetworkError'); cin.writes.push(c.slice()); gate?.(c); };
  if (withoutResponse) cin.writeValueWithoutResponse = write; else cin.writeValue = write;
  const chars = { [UUID.info]: info, [UUID.status]: status, [UUID.ctlIn]: cin, [UUID.ctlOut]: cout };
  const device = Object.assign(new EventTarget(), { id: 'dev-1', name: 'DataOpen-0001' });
  device.gatt = {
    connected: false,
    async connect() {
      log.connects++;
      device.gatt.connected = true;
      return { async getPrimaryService(u) { assert.equal(u, UUID.service); return { async getCharacteristic(c) { assert.ok(chars[c], `unexpected characteristic ${c}`); return chars[c]; } }; } };
    },
    disconnect() { device.gatt.connected = false; device.dispatchEvent(new Event('gattserverdisconnected')); },
  };
  const bt = { async requestDevice(o) { log.requested.push(o); return device; } };
  if (known !== null) bt.getDevices = async () => (known ? [device] : []);
  return { bt, device, chars, log };
}

test('first time: the chooser is asked for the device service only, with a user gesture', async () => {
  const f = fakeBluetooth({ known: false });
  const tr = new BleTransport({ bluetooth: f.bt, store: new Store(null) });
  await assert.rejects(tr.open({ interactive: false }), /needs-gesture/);
  assert.equal(f.log.requested.length, 0);
  const r = await tr.open({ interactive: true });
  assert.deepEqual(f.log.requested, [{ filters: [{ services: [UUID.service] }] }]);
  assert.equal(r.info.length, 16);
  assert.equal(unpackStatus(r.status).mode, 2);
  assert.ok(f.chars[UUID.ctlOut].notifying && f.chars[UUID.status].notifying);
});

test('afterwards the remembered device is reused with no chooser at all', async () => {
  const f = fakeBluetooth({ known: true });
  const store = new Store(null);
  await new BleTransport({ bluetooth: f.bt, store }).open({ interactive: true });
  const again = new BleTransport({ bluetooth: f.bt, store });
  await again.open({ interactive: false });
  assert.equal(f.log.requested.length, 1);
  assert.equal(f.log.connects, 2);
});

test('a browser without getDevices still works (it just asks each time)', async () => {
  const f = fakeBluetooth({ known: null });
  const tr = new BleTransport({ bluetooth: f.bt, store: new Store(null) });
  await assert.rejects(tr.open({ interactive: false }), /needs-gesture/);
  await tr.open({ interactive: true });
  assert.equal(f.log.requested.length, 1);
});

test('values are copied out of the browser DataView (offset buffers), status and chunks reach the right callbacks', async () => {
  const f = fakeBluetooth();
  const tr = new BleTransport({ bluetooth: f.bt, store: new Store(null) });
  const chunks = [], statuses = [];
  tr.onChunk = (c) => chunks.push(c);
  tr.onStatus = (s) => statuses.push(s);
  await tr.open();
  f.chars[UUID.ctlOut].push(Uint8Array.from([0xc0, 1, 2, 3]));
  f.chars[UUID.status].push(packStatus({ bridge: 1, mode: 3, flags: SF.TRIAL }));
  assert.deepEqual([...chunks[0]], [0xc0, 1, 2, 3]);
  assert.equal(chunks[0].buffer.byteLength, 4);
  assert.equal(unpackStatus(statuses[0]).mode, 3);
});

test('writes go without response when the characteristic allows it, with response otherwise', async () => {
  for (const withoutResponse of [true, false]) {
    const f = fakeBluetooth({ withoutResponse });
    const tr = new BleTransport({ bluetooth: f.bt, store: new Store(null) });
    await tr.open();
    await tr.write(Uint8Array.from([0x80 | 0x40, 9]));
    assert.deepEqual([...f.chars[UUID.ctlIn].writes[0]], [0xc0, 9]);
  }
});

test('a drop of the link is reported once and writes then fail loudly', async () => {
  const f = fakeBluetooth();
  const tr = new BleTransport({ bluetooth: f.bt, store: new Store(null) });
  let closed = 0;
  tr.onClose = () => closed++;
  await tr.open();
  f.device.gatt.disconnect();
  assert.equal(closed, 1);
  await assert.rejects(tr.write(Uint8Array.from([1])), /not connected/);
  await tr.open();                                                                   // and it can reconnect to the same device
  assert.equal(f.log.connects, 2);
  f.device.gatt.disconnect();
  assert.equal(closed, 2);
});

test('a Session over the BLE path drops to 20-byte chunks when the radio refuses long writes', async () => {
  const f = fakeBluetooth({ maxWrite: 20 });
  const tr = new BleTransport({ bluetooth: f.bt, store: new Store(null) });
  // a device that only answers HELLO, enough to see the client's chunking
  f.chars[UUID.ctlIn].writeValueWithoutResponse = async (c) => {
    if (c.length > 20) throw new DOMException('too long', 'NetworkError');
    f.chars[UUID.ctlIn].writes.push(c.slice());
  };
  const s = new Session(tr, { chunk: 180, timeoutMs: 100 });
  await assert.rejects(s.connect(), /timeout/);
  assert.ok(f.chars[UUID.ctlIn].writes.length >= 2);
  assert.ok(f.chars[UUID.ctlIn].writes.every((c) => c.length <= 20));
  assert.equal(s.conn, 'idle');
});

test('unsupported browsers are detected', () => {
  assert.equal(BleTransport.supported({}), false);
  assert.equal(BleTransport.supported({ bluetooth: {} }), true);
  assert.equal(BleTransport.supported(null), false);
});

// ---- the same fake, wired to the real gateway
const URL_ = process.env.DATAOPEN_SIM_URL;
test('end to end over the BLE code path: connect, enable help, stop', { skip: !URL_ }, async () => {
  const { WsTransport } = await import('../js/transport/ws.js');
  const wire = new WsTransport(URL_.replace(/^http/, 'ws') + 'ws');
  const f = fakeBluetooth({ gate: null });
  const { info, status } = await wire.open();
  f.chars[UUID.info].value = new DataView(info.buffer, info.byteOffset, info.byteLength);
  f.chars[UUID.status].value = new DataView(status.buffer, status.byteOffset, status.byteLength);
  wire.onChunk = (c) => f.chars[UUID.ctlOut].push(c);
  wire.onStatus = (b) => f.chars[UUID.status].push(b);
  f.chars[UUID.ctlIn].writeValueWithoutResponse = async (c) => wire.write(c);
  const tr = new BleTransport({ bluetooth: f.bt, store: new Store(null) });
  const s = new Session(tr, { chunk: 100, timeoutMs: 4000 });
  await s.connect();
  assert.equal(s.conn, 'connected');
  assert.ok(s.manifest.pages.length >= 3);
  await s.set('assist.strength', 6);
  assert.equal(s.state['assist.strength'], 6);
  await s.set('assist.strength', 5);
  const r = await s.stop();
  assert.equal(r.json.ok, true);
  assert.equal(T.STOP, 36);
  s.close();
  wire.close();
});
