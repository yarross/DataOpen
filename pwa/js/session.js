// The session: one connection to one device over any transport. It owns the request/answer bookkeeping, the write queue (the safety
// action jumps it), the device's state tree and its manifest. It never draws anything.
//
// Transport contract (BLE and WebSocket both implement it):
//   open({interactive}) -> Promise<{ info: Uint8Array, status: Uint8Array|null }>   connects, subscribes, reads
//   write(chunk: Uint8Array) -> Promise                                              one chunk to the device
//   close()                                                                           ends the connection
//   onChunk(u8), onStatus(u8), onClose()                                              set by the session
import { T, VER, ERR, CHUNK_MIN, CHUNK_MAX } from './constants.js';
import { packMessage, packJson, unpackMessage, bodyJson, chunkMessage, Reassembler, unpackStatus, unpackInfo, crc32 } from './link.js';
import { errKeyFor, errParams } from './i18n.js';

export class SessionError extends Error {
  constructor(code, key, detail = '') {
    super(`${key}${detail ? `: ${detail}` : ''}`);
    this.code = code;
    this.key = key;
    this.detail = detail;
    this.params = errParams(detail);
  }
}

const hex32 = (n) => n.toString(16).padStart(8, '0');

export class Session {
  constructor(transport, { timeoutMs = 5000, chunk = 180, store = null, lang = 'en', now = () => Date.now(), timers = globalThis } = {}) {
    this.tr = transport;
    this.timeoutMs = timeoutMs;
    this.wantChunk = Math.min(Math.max(chunk, CHUNK_MIN), CHUNK_MAX);
    this.chunk = CHUNK_MIN;
    this.store = store;
    this.lang = lang;
    this.now = now;
    this.timers = timers;
    this.conn = 'idle';                  // idle | connecting | connected | lost
    this.linked = false;                 // the transport is up (the safety actions work from here on, before any HELLO)
    this.info = null;
    this.hello = null;
    this.status = null;
    this.state = {};
    this.rev = null;
    this.manifest = null;
    this.reasm = new Reassembler();
    this.pending = new Map();
    this.reqId = 0;
    this.outSeq = 0;
    this.qU = [];
    this.qN = [];
    this.pumping = false;
    this.handlers = {};
    this.refreshing = false;
    this.bad = 0;
  }

  on(name, fn) {
    (this.handlers[name] ||= new Set()).add(fn);
    return () => this.handlers[name].delete(fn);
  }
  _emit(name, arg) { for (const fn of this.handlers[name] || []) { try { fn(arg); } catch (e) { console.error(e); } } }
  _setConn(c) { if (this.conn !== c) { this.conn = c; this._emit('conn', c); } }

  // ---------------------------------------------------------------- connection
  async connect({ interactive = true } = {}) {
    this._setConn('connecting');
    this.reasm.reset();
    this.tr.onChunk = (c) => this._onChunk(c);
    this.tr.onStatus = (b) => this._onStatus(b);
    this.tr.onClose = () => this._lost();
    try {
      const { info, status } = await this.tr.open({ interactive });
      this.linked = true;
      this.info = unpackInfo(info);
      if (status) this._onStatus(status, true);
      this.chunk = CHUNK_MIN;                                           // HELLO goes at the smallest size: nothing is assumed yet
      const hint = this.info?.chunkHint || CHUNK_MIN;
      const r = await this.request(T.HELLO, { v: VER, chunk: Math.min(this.wantChunk, Math.max(hint, CHUNK_MIN)), lang: this.lang });
      this.hello = r.json;
      this.chunk = Math.min(Math.max(r.json.chunk | 0, CHUNK_MIN), CHUNK_MAX);
      this._setConn('connected');
      await this.refreshManifest();
      await this.refreshState();
    } catch (e) {
      this.linked = false;
      this.tr.close?.();
      this._failPending(e);
      this._setConn(this.conn === 'connected' ? 'lost' : 'idle');
      throw e;
    }
  }

  close() {
    this.linked = false;
    this.tr.close?.();
    this._failPending(new SessionError(0, 'err.disconnected'));
    this._setConn('idle');
  }

  _lost() {
    this.linked = false;
    if (this.conn === 'idle') return;
    this._failPending(new SessionError(0, 'err.disconnected'));
    this._setConn('lost');
  }

  _failPending(e) {
    for (const [, p] of this.pending) { this.timers.clearTimeout(p.timer); p.reject(e); }
    this.pending.clear();
    for (const q of [this.qU, this.qN]) { for (const it of q) it.reject(e); q.length = 0; }
  }

  // ---------------------------------------------------------------- incoming
  _onChunk(c) {
    const raw = this.reasm.feed(c, this.now());
    if (!raw) return;
    const m = unpackMessage(raw);
    if (!m) { this.bad++; return; }
    this._onMessage(m);
  }

  _onStatus(b, quiet = false) {
    const s = unpackStatus(b);
    if (!s) { this.bad++; return; }
    this.status = s;
    if (!quiet) this._emit('status', s);
    if (this.conn === 'connected' && this.rev !== null && s.stateRev !== this.rev && !this.refreshing) {
      this.refreshState().catch(() => {});                              // an event was missed: ask for the whole state again
    }
  }

  _onMessage(m) {
    if (m.type === T.EVENT) {
      let j;
      try { j = bodyJson(m); } catch { this.bad++; return; }
      if (typeof j.manifest === 'string' && this.manifest && this.hello && j.manifest !== this.hello.manifest_hash) {
        this.hello.manifest_hash = j.manifest;                          // the device has a new layout: fetch it (and cache it under its new hash)
        this.refreshManifest().catch(() => {});
      }
      if (j.state && typeof j.state === 'object') {
        Object.assign(this.state, j.state);
        if (Number.isInteger(j.rev)) this.rev = j.rev;
        this._emit('state', this.state);
      }
      this._emit('event', j);
      return;
    }
    const p = this.pending.get(m.req);
    if (!p) return;
    this.pending.delete(m.req);
    this.timers.clearTimeout(p.timer);
    if (m.type === T.ERR) {
      let j = {};
      try { j = bodyJson(m); } catch { /* keep the empty object */ }
      p.reject(new SessionError(j.code | 0, errKeyFor(j.code, j.detail), String(j.detail ?? '')));
    } else {
      let json = null;
      if (m.type !== T.DATA || m.body[0] === 0x7b) { try { json = bodyJson(m); } catch { json = null; } }
      p.resolve({ type: m.type, body: m.body, json: json ?? {} });
    }
  }

  // ---------------------------------------------------------------- outgoing
  request(type, obj = {}, { urgent = false, body = null, timeoutMs = this.timeoutMs } = {}) {
    this.reqId = (this.reqId % 0xffff) + 1;
    const req = this.reqId;
    const raw = body ? packMessage(type, req, body) : packJson(type, req, obj);
    return new Promise((resolve, reject) => {
      const timer = this.timers.setTimeout(() => {
        this.pending.delete(req);
        reject(new SessionError(0, 'err.timeout'));
      }, timeoutMs);
      this.pending.set(req, { resolve, reject, timer });
      this._send(raw, urgent).catch((e) => {
        if (this.pending.delete(req)) { this.timers.clearTimeout(timer); reject(e); }
      });
    });
  }

  _send(raw, urgent = false) {
    return new Promise((resolve, reject) => {
      (urgent ? this.qU : this.qN).push({ raw, urgent, idx: 0, chunks: null, size: this.chunk, started: false, resolve, reject });
      this._pump();
    });
  }

  async _pump() {
    if (this.pumping) return;
    this.pumping = true;
    try {
      for (;;) {
        const item = this.qU[0] ?? this.qN[0];
        if (!item) break;
        if (item.urgent) {                                               // the safety action cuts in front of a message that is half sent
          for (const n of this.qN) if (n.started) n.reject(new SessionError(0, 'err.interrupted'));
          this.qN = this.qN.filter((n) => !n.started);
        }
        if (!item.chunks) {
          const { chunks, next } = chunkMessage(item.raw, item.size, this.outSeq);
          item.chunks = chunks;
          this.outSeq = next;
        }
        try {
          await this.tr.write(item.chunks[item.idx]);
        } catch (e) {
          if (item.size > CHUNK_MIN) {                                   // a chunk too big for this link: the whole message again, small
            item.size = this.chunk = CHUNK_MIN;
            item.chunks = null;
            item.idx = 0;
            item.started = false;
            continue;
          }
          (item.urgent ? this.qU : this.qN).shift();
          item.reject(e);
          continue;
        }
        item.started = true;
        if (++item.idx >= item.chunks.length) {
          (item.urgent ? this.qU : this.qN).shift();
          item.resolve();
        }
      }
    } finally {
      this.pumping = false;
    }
  }

  // ---------------------------------------------------------------- what the app asks for
  async refreshState() {
    this.refreshing = true;
    try {
      const r = await this.request(T.GET, { what: 'state' });
      this.state = { ...r.json.state };
      this.rev = r.json.rev;
      this._emit('state', this.state);
    } finally {
      this.refreshing = false;
    }
  }

  async refreshManifest() {
    const key = `manifest:${this.info?.deviceId ?? ''}`;
    const cached = this.store ? await this.store.get(key) : null;
    if (cached && cached.hash === this.hello?.manifest_hash && cached.manifest) {
      this.manifest = cached.manifest;
      this._emit('manifest', this.manifest);
      return;
    }
    const r = await this.request(T.GET, { what: 'manifest' });
    let m;
    try { m = JSON.parse(new TextDecoder().decode(r.body)); } catch { throw new SessionError(0, 'manifest.failed'); }
    this.manifest = m;
    if (this.store) await this.store.set(key, { hash: hex32(crc32(r.body)), manifest: m });
    this._emit('manifest', m);
  }

  set(key, value) { return this.request(T.SET, { key, value }); }
  act(key, confirmed = false) { return this.request(T.ACT, { key, confirmed }); }
  confirm(keep) { return this.request(T.CONFIRM, { keep: !!keep }); }
  stop() { return this.request(T.STOP, {}, { urgent: true, body: new Uint8Array(0) }); }
  hardBypass() { return this.request(T.HARD_BYPASS, {}, { urgent: true, body: new Uint8Array(0) }); }
  putBundle(bytes) { return this.request(T.BUNDLE_PUT, {}, { body: bytes, timeoutMs: 20000 }); }
  // 'self' (a copy only this device can open) or another device's card (a parsed .docard); the file comes back sealed, the page never reads it
  async getBundle(target = 'self') { return (await this.request(T.GET, { what: 'bundle', for: target }, { timeoutMs: 20000 })).body; }
  async getIdentity() { return (await this.request(T.GET, { what: 'identity' })).json; }
  ping() { return this.request(T.PING, {}); }
}

export { ERR };
