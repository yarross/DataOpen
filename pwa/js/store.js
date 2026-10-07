// Small persistent preferences and a manifest cache. IndexedDB can be missing, blocked or throw (private windows, cleared site data):
// every call is wrapped and the app behaves identically from memory, just without remembering.
const DB = 'dataopen', STORE = 'kv';

export class Store {
  constructor(idb = (typeof indexedDB !== 'undefined' ? indexedDB : null)) {
    this.idb = idb;
    this.mem = new Map();
    this.db = null;
  }
  async _open() {
    if (this.db || !this.idb) return this.db;
    try {
      this.db = await new Promise((resolve, reject) => {
        const r = this.idb.open(DB, 1);
        r.onupgradeneeded = () => r.result.createObjectStore(STORE);
        r.onsuccess = () => resolve(r.result);
        r.onerror = () => reject(r.error);
        r.onblocked = () => reject(new Error('blocked'));
      });
    } catch { this.db = null; this.idb = null; }
    return this.db;
  }
  async get(key, dflt = null) {
    if (this.mem.has(key)) return this.mem.get(key);
    try {
      const db = await this._open();
      if (!db) return dflt;
      const v = await new Promise((resolve, reject) => {
        const r = db.transaction(STORE).objectStore(STORE).get(key);
        r.onsuccess = () => resolve(r.result);
        r.onerror = () => reject(r.error);
      });
      return v === undefined ? dflt : v;
    } catch { return dflt; }
  }
  async set(key, value) {
    this.mem.set(key, value);
    try {
      const db = await this._open();
      if (!db) return false;
      await new Promise((resolve, reject) => {
        const tx = db.transaction(STORE, 'readwrite');
        tx.objectStore(STORE).put(value, key);
        tx.oncomplete = resolve;
        tx.onerror = () => reject(tx.error);
      });
      return true;
    } catch { return false; }
  }
}
