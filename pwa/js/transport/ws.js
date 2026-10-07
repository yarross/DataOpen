// CtlLink over a WebSocket: one chunk per binary frame, first byte = channel (0 ctl, 1 status, 2 info). The same page served by the
// device itself over its own Wi-Fi would use this; today it carries the simulator and the end-to-end tests.
export class WsTransport {
  constructor(url, WS = globalThis.WebSocket) {
    this.url = url;
    this.WS = WS;
    this.ws = null;
    this.onChunk = this.onStatus = this.onClose = () => {};
  }
  open() {
    return new Promise((resolve, reject) => {
      const ws = new this.WS(this.url);
      this.ws = ws;
      ws.binaryType = 'arraybuffer';
      let info = null, status = null, done = false;
      const finish = () => { if (!done && info && status) { done = true; resolve({ info, status }); } };
      ws.onerror = () => { if (!done) { done = true; reject(new Error('websocket error')); } };
      ws.onclose = () => { if (!done) { done = true; reject(new Error('websocket closed')); } else this.onClose(); };
      ws.onmessage = (ev) => {
        const b = new Uint8Array(ev.data);
        const body = b.slice(1);
        if (b[0] === 2) { info = body; finish(); }                                 // the device sends info, then status, at connect
        else if (b[0] === 1) { if (done) this.onStatus(body); else { status = body; finish(); } }
        else if (b[0] === 0) this.onChunk(body);
      };
    });
  }
  async write(chunk) {
    const out = new Uint8Array(chunk.length + 1);
    out.set(chunk, 1);
    this.ws.send(out);
  }
  close() { try { this.ws?.close(); } catch { /* already closed */ } }
}
