import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { buildView, walk } from '../js/view.js';
import { pick } from '../js/i18n.js';

const manifest = JSON.parse(readFileSync(new URL('./manifest.json', import.meta.url)));
const state = { 'assist.on': true, 'assist.strength': 7, 'tremor.level': 5, 'calib.running': false, 'profile.fill': 64, 'profile.layers': 'asc' };

test('the default manifest becomes three pages with the controls the plan lists', () => {
  const v = buildView(manifest, state, 'ru');
  assert.deepEqual(v.pages.map((p) => p.id), ['main', 'profile', 'device']);
  const byKey = {};
  for (const n of walk(v.pages.flatMap((p) => p.nodes))) if (n.key) byKey[n.key] = n;
  assert.equal(byKey['assist.on'].kind, 'toggle');
  assert.equal(byKey['assist.on'].on, true);
  assert.equal(byKey['assist.strength'].kind, 'stepper');
  assert.equal(byKey['profile.fill'].kind, 'meter');
  assert.equal(byKey['profile.fill'].text, '64 %');
  assert.equal(byKey['profile.layers'].text, 'Готова помощь при движении');
});

test('a stepper knows its neighbours and speaks in words at the marked levels', () => {
  const find = (s, lang = 'ru') => [...walk(buildView(manifest, s, lang).pages[0].nodes)].find((n) => n.key === 'assist.strength');
  const mid = find({ 'assist.strength': 5 });
  assert.equal(mid.valueText, 'Как рекомендует профиль');
  assert.ok(mid.canDec && mid.canInc);
  const low = find({ 'assist.strength': 0 }), high = find({ 'assist.strength': 10 }, 'en');
  assert.ok(!low.canDec && low.canInc && high.canDec && !high.canInc);
  assert.equal(high.valueText, 'Maximum');
  const seven = find({ 'assist.strength': 7 });
  assert.equal(seven.valueText, 'Уровень 7 из 10');
  const unknown = find({});
  assert.equal(unknown.known, false);
  assert.ok(!unknown.canDec && !unknown.canInc);
  assert.equal(find({ 'assist.strength': 99 }).value, 10);                     // out-of-range values from the device are clamped for display
});

test('every control has a name and a help text in both languages where it is meant to', () => {
  for (const lang of ['ru', 'en']) {
    const v = buildView(manifest, state, lang);
    assert.ok(v.title);
    for (const p of v.pages) {
      assert.ok(p.title, `${p.id} title`);
      for (const n of walk(p.nodes)) assert.ok(n.label, `${p.id}/${n.id} label (${lang})`);
    }
  }
});

test('unknown control types and odd fields never break the page', () => {
  const m = structuredClone(manifest);
  m.pages[0].controls.push({ id: 'x1', type: 'hologram', label: { ru: 'Новое', en: 'New' } }, null, 5, { id: 'x2', type: 'stepper', min: 'a', max: {}, label: 'plain' },
    { id: 'x3', type: 'group', controls: 'nope', label: { en: 'G' } });
  m.pages.push(null, { id: 'bad', title: 5, controls: 7 });
  const v = buildView(m, state, 'en');
  const unknown = [...walk(v.pages[0].nodes)].find((n) => n.id === 'x1');
  assert.equal(unknown.kind, 'unknown');
  assert.match(unknown.note, /Update the app/);
  assert.ok(v.pages[0].nodes.length >= 5);
  assert.ok(buildView(null, null, 'ru').pages.length === 0);
  assert.ok(buildView({ pages: 'x' }, {}, 'ru').pages.length === 0);
});

test('a hostile manifest stays text: labels are returned as plain strings', () => {
  const m = structuredClone(manifest);
  m.pages[0].controls[0].label = { ru: '<img src=x onerror=alert(1)>', en: '<script>alert(1)</script>' };
  const n = buildView(m, state, 'en').pages[0].nodes[0];
  assert.equal(n.label, '<script>alert(1)</script>');                         // the DOM layer only ever uses textContent (see a11y.test.mjs)
});

test('pick falls back to English, then to anything, then to nothing', () => {
  assert.equal(pick({ ru: 'а', en: 'a' }, 'ru'), 'а');
  assert.equal(pick({ en: 'a' }, 'ru'), 'a');
  assert.equal(pick({ de: 'x' }, 'ru'), 'x');
  assert.equal(pick(undefined, 'ru'), '');
  assert.equal(pick({ ru: 5 }, 'ru'), '');
  assert.equal(pick('plain', 'en'), 'plain');
});
