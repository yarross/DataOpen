import test from 'node:test';
import assert from 'node:assert/strict';
import { STRINGS, t, detectLang, errKeyFor, LANGS } from '../js/i18n.js';
import { REASONS, ERR } from '../js/constants.js';

const ru = STRINGS.ru, en = STRINGS.en;

test('both languages have exactly the same keys', () => {
  assert.deepEqual(Object.keys(ru).sort(), Object.keys(en).sort());
  assert.deepEqual(LANGS, ['ru', 'en']);
});

test('no empty strings, no untranslated copies (apart from names and glyphs)', () => {
  const same = [];
  for (const k of Object.keys(ru)) {
    assert.ok(ru[k].trim() && en[k].trim(), k);
    if (ru[k] === en[k]) same.push(k);
  }
  assert.deepEqual(same, []);
});

test('placeholders agree between the languages', () => {
  const ph = (s) => (s.match(/\{\w+\}/g) || []).sort().join();
  for (const k of Object.keys(ru)) assert.equal(ph(ru[k]), ph(en[k]), k);
});

test('every bridge reason and every device error has words', () => {
  for (const r of REASONS) for (const l of [ru, en]) assert.ok(l[`reason.${r}`], r);
  for (const e of Object.keys(ERR)) for (const l of [ru, en]) assert.ok(l[`err.${e.toLowerCase()}`], e);
  for (const k of ['timeout', 'disconnected', 'interrupted', 'unknown']) assert.ok(ru[`err.${k}`] && en[`err.${k}`]);
});

test('hints exist only for reasons that exist', () => {
  for (const k of Object.keys(en).filter((x) => x.startsWith('hint.'))) assert.ok(REASONS.includes(k.slice(5)), k);
});

test('t() fills placeholders and falls back to English, then to the key', () => {
  assert.equal(t('ru', 'trial.compact', { n: 7 }), 'Стало лучше? Само вернётся через 7 с');
  assert.equal(t('de', 'common.ok'), 'Done');
  assert.equal(t('ru', 'no.such.key'), 'no.such.key');
  assert.equal(t('en', 'stepper.value', { v: 3 }), 'Level 3 of {max}');
});

test('the language is picked from the phone, defaulting to English', () => {
  assert.equal(detectLang({ languages: ['ru-RU', 'en'] }), 'ru');
  assert.equal(detectLang({ language: 'uk' }), 'ru');
  assert.equal(detectLang({ language: 'de-DE' }), 'en');
  assert.equal(detectLang(null), 'en');
});

test('device error codes map to texts, bundle errors to their own reasons', () => {
  assert.equal(errKeyFor(ERR.PHYSICAL), 'err.physical');
  assert.equal(errKeyFor(ERR.BAD_BUNDLE, 'damaged'), 'err.bad_bundle.damaged');
  assert.equal(errKeyFor(ERR.BAD_BUNDLE, 'whatever'), 'err.bad_bundle');
  assert.equal(errKeyFor(999), 'err.unknown');
});
