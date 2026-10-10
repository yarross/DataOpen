// A person brings their own interface model (docs/MODELS.md). What this proves about the CLIENT: it stays dumb. It carries bytes, it has no
// parser for models, it shows the device's words for acceptance, refusal and application, and every reason the device can give is in both languages.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, readdirSync } from 'node:fs';
import { Session } from '../js/session.js';
import { buildView, walk } from '../js/view.js';
import { STRINGS, errKeyFor } from '../js/i18n.js';
import { ERR } from '../js/constants.js';

const jsDir = new URL('../js/', import.meta.url);
const manifest = JSON.parse(readFileSync(new URL('./manifest.json', import.meta.url), 'utf8'));

test('the client has no parser for models and does not look inside a package', () => {
  for (const f of readdirSync(jsDir).filter((n) => n.endsWith('.js'))) {
    const text = readFileSync(new URL(f, jsDir), 'utf8');
    assert.ok(!/onnx|protobuf|ModelProto|GraphProto|initializer|opset/i.test(text), `${f} mentions a model format`);
  }
  const packageMethods = Object.getOwnPropertyNames(Session.prototype).filter((n) => /package|model/i.test(n)).sort();
  assert.deepEqual(packageMethods, ['getPackages', 'putPackage']);          // one way in, one way to read the device's own words
});

test('the page offers the whole model flow with the device\'s words: load, apply, back, remove, forget senders', () => {
  const all = [...walk(buildView(manifest, {}, 'en').pages.flatMap((p) => p.nodes))];
  const keys = all.filter((n) => n.kind === 'action').map((n) => n.key);
  for (const k of ['pkg.apply', 'pkg.discard', 'pkg.revert', 'model.clear', 'trust.clear']) assert.ok(keys.includes(k), k);
  assert.ok(all.some((n) => n.kind === 'file' && n.op === 'pkg_put'));
  assert.ok(all.some((n) => n.key === 'model.from'));
  const danger = all.filter((n) => n.kind === 'action' && n.key === 'trust.clear').map((n) => !!n.danger);
  assert.deepEqual(danger, [true]);                                           // forgetting senders is marked, taking one model out is not
  const mine = all.find((n) => n.kind === 'action' && n.key === 'model.clear');
  assert.ok(!mine.danger && mine.confirm === 'two-step');
});

test('every refusal of a model has words in both languages, and the button for forgetting senders too', () => {
  for (const key of ['model_ops', 'model_files', 'model_io', 'model_cost', 'not_ui_model', 'bad_model', 'no_model', 'no_senders', 'wrong_device', 'bad_signature']) {
    for (const lang of ['ru', 'en']) {
      const text = STRINGS[lang][`err.pkg_rejected.${key}`];
      assert.ok(typeof text === 'string' && text.length > 10, `${lang} ${key}`);
    }
    assert.equal(errKeyFor(ERR.PKG_REJECTED, key), `err.pkg_rejected.${key}`);
  }
  assert.equal(errKeyFor(ERR.PHYSICAL, 'trust.clear'), 'err.physical.trust.clear');
  assert.equal(errKeyFor(ERR.PHYSICAL, 'model:7K2M-9QX4-ABCD-WXY3'), 'err.physical.model');
  assert.notEqual(STRINGS.ru['err.pkg_rejected.model_ops'], STRINGS.en['err.pkg_rejected.model_ops']);
});

test('the page keeps no weights: the only file of the device it saves is the public card', () => {
  const app = readFileSync(new URL('app.js', jsDir), 'utf8');
  assert.equal(app.split('download(').length - 1, 1);
  assert.ok(!/model\.bin|\.onnx|weights/i.test(app));
});

test('the page says in words when the scene help is off and when a model is too heavy, in both languages', () => {
  const status = (key) => manifest.pages.flatMap((p) => p.controls ?? []).find((c) => c.key === key);
  const scene = status('scene.state');
  assert.deepEqual(Object.keys(scene.map).sort(), ['none', 'ok', 'off', 'warming']);
  for (const v of Object.values(scene.map)) assert.ok(v.ru.length > 3 && v.en.length > 3 && v.ru !== v.en);
  const model = status('model.state');
  assert.ok(model.map.over_budget.ru && model.map.over_budget.en);
});
