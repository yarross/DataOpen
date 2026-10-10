// The profile and the model never leave the device (docs/RESIDENCY.md). What this proves about the CLIENT: it has no call that asks for them,
// a request for them does not even leave the page, a layout can not draw a button that would, and the page keeps nothing of the device.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, readdirSync } from 'node:fs';
import { Session, SessionError } from '../js/session.js';
import { buildView, walk } from '../js/view.js';
import { T, ERR, GET_KINDS, FILE_OPS } from '../js/constants.js';
import { STRINGS, errKeyFor } from '../js/i18n.js';

const manifest = JSON.parse(readFileSync(new URL('./manifest.json', import.meta.url), 'utf8'));
const jsDir = new URL('../js/', import.meta.url);
const source = (name) => readFileSync(new URL(name, jsDir), 'utf8');

test('the client can only ask for the closed list of things the device says', () => {
  assert.deepEqual(GET_KINDS, ['manifest', 'state', 'identity', 'slots', 'firmware', 'packages']);
  assert.deepEqual(FILE_OPS, ['bundle_put', 'fw_put', 'pkg_put', 'card_get']);
  const getters = Object.getOwnPropertyNames(Session.prototype).filter((n) => /^get|bundle|profile|model|weights|export|backup|download/i.test(n)).sort();
  assert.deepEqual(getters, ['getFirmware', 'getIdentity', 'getPackages', 'getSlots', 'putBundle']);       // putBundle carries a file IN
});

test('a request for the profile, the model or a copy fails inside the page and writes nothing to the device', async () => {
  const written = [];
  const transport = {
    open: async () => ({ info: new Uint8Array(16), status: null }),
    write: async (c) => { written.push(c); },
    close() {}, onChunk() {}, onStatus() {}, onClose() {},
  };
  const s = new Session(transport, { timeoutMs: 50 });
  for (const what of ['bundle', 'profile', 'profile_prev', 'model', 'weights', 'export', 'backup', 'slot_data', 'params', 'keys', 'download', 'copy']) {
    await assert.rejects(s.request(T.GET, { what }), (e) => e instanceof SessionError && e.code === ERR.RESIDENT && e.key === 'err.resident');
  }
  assert.equal(written.length, 0);
});

test('a layout from a sender or a package can not draw a button that takes data out', () => {
  const bad = structuredClone(manifest);
  bad.pages[0].controls.push(...['bundle_get', 'bundle_for_card', 'profile_get', 'model_get'].map((op, i) => (
    { id: `grab${i}`, type: 'file', op, accept: '.x', max_bytes: 100, label: { ru: 'Скачать', en: 'Download' } })));
  const all = [...walk(buildView(bad, {}, 'en').pages.flatMap((p) => p.nodes))];
  assert.deepEqual(all.filter((n) => /^grab/.test(n.id)).map((n) => n.kind), ['unknown', 'unknown', 'unknown', 'unknown']);
  assert.ok(!all.some((n) => n.kind === 'file' && !FILE_OPS.includes(n.op)));
  const files = all.filter((n) => n.kind === 'file').map((n) => n.op).sort();
  assert.deepEqual(files, ['bundle_put', 'card_get', 'fw_put', 'pkg_put']);
});

test('the page saves exactly one thing of the device, its public card, and keeps no secret anywhere', () => {
  const app = source('app.js');
  assert.equal(app.split('download(').length - 1, 1);
  assert.ok(app.includes('docard') && !app.includes('.dobundle`'));
  for (const f of readdirSync(jsDir).filter((n) => n.endsWith('.js'))) {
    const text = source(f);
    for (const needle of ['getBundle', 'bundle_get', 'bundle_for_card', "what: 'bundle'", "what: 'profile'", "what: 'model'"]) assert.ok(!text.includes(needle), `${f}: ${needle}`);
  }
  // the only things it stores: language, theme, the repeat-tap guard, and the layout by its hash
  const keys = new Set([...app.matchAll(/store\.(?:get|set)\('([a-zA-Z]+)'/g)].map((m) => m[1]));
  assert.deepEqual([...keys].sort(), ['lang', 'tapMs', 'theme']);
  assert.ok(/store\.set\(key, \{ hash: [^}]*manifest: m \}\)/.test(source('session.js')));            // and the manifest cache: a layout, not data
});

test('the refusal has words in both languages', () => {
  assert.equal(errKeyFor(ERR.RESIDENT, 'resident'), 'err.resident');
  assert.match(STRINGS.ru['err.resident'], /остаются на устройстве/);
  assert.match(STRINGS.en['err.resident'], /stay on the device/);
});
