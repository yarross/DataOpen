// CtlLink over Web Bluetooth. Four characteristics (info read, status read+notify, ctl_in write, ctl_out notify), all of which need an
// encrypted (bonded) link: the OS shows the pairing prompt the first time one is touched. Chrome on Android and desktop Chrome/Edge only;
// there is no Web Bluetooth in any browser on iPhone.
import { UUID } from './../constants.js';

const u8 = (dv) => new Uint8Array(dv.buffer, dv.byteOffset, dv.byteLength).slice();

export class BleTransport {
  static supported(nav = (typeof navigator !== 'undefined' ? navigator : null)) { return !!(nav && nav.bluetooth); }

  constructor({ bluetooth = (typeof navigator !== 'undefined' ? navigator.bluetooth : null), store = null } = {}) {
    this.bt = bluetooth;
    this.store = store;
    this.device = null;
    this.cin = null;
    this.onChunk = this.onStatus = this.onClose = () => {};
    this._dropped = () => { this.cin = null; this.onClose(); };
  }

  // Needs a user gesture the first time (the browser's own device chooser); afterwards the remembered device is used without one.
  async open({ interactive = true } = {}) {
    if (!this.bt) throw new Error('no Web Bluetooth');
    let device = this.device;
    if (!device) {
      const saved = this.store ? await this.store.get('ble:device') : null;
      if (typeof this.bt.getDevices === 'function') {
        try {
          const known = await this.bt.getDevices();
          device = known.find((d) => d.id === saved) || null;
        } catch { device = null; }
      }
      if (!device) {
        if (!interactive) throw new Error('needs-gesture');
        device = await this.bt.requestDevice({ filters: [{ services: [UUID.service] }] });
      }
      if (this.store) await this.store.set('ble:device', device.id);
    }
    this.device = device;
    device.removeEventListener?.('gattserverdisconnected', this._dropped);
    device.addEventListener('gattserverdisconnected', this._dropped);
    const server = await device.gatt.connect();
    const svc = await server.getPrimaryService(UUID.service);
    const info = await svc.getCharacteristic(UUID.info);
    const status = await svc.getCharacteristic(UUID.status);
    const cin = await svc.getCharacteristic(UUID.ctlIn);
    const cout = await svc.getCharacteristic(UUID.ctlOut);
    cout.addEventListener('characteristicvaluechanged', (e) => this.onChunk(u8(e.target.value)));
    status.addEventListener('characteristicvaluechanged', (e) => this.onStatus(u8(e.target.value)));
    await cout.startNotifications();
    await status.startNotifications();
    this.cin = cin;
    return { info: u8(await info.readValue()), status: u8(await status.readValue()) };
  }

  async write(chunk) {
    if (!this.cin) throw new Error('not connected');
    if (typeof this.cin.writeValueWithoutResponse === 'function') await this.cin.writeValueWithoutResponse(chunk);
    else await this.cin.writeValue(chunk);
  }

  close() {
    try { this.device?.gatt?.disconnect(); } catch { /* already gone */ }
    this.cin = null;
  }

  forget() { this.device = null; return this.store ? this.store.set('ble:device', null) : Promise.resolve(); }
}
