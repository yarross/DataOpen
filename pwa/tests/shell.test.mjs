import test from 'node:test';
import assert from 'node:assert/strict';
import { describe, trialModel, MODE } from '../js/shell.js';
import { SF, RB, REASONS, MODE_UNKNOWN } from '../js/constants.js';
import { STRINGS } from '../js/i18n.js';

const st = (o = {}) => ({ bridge: 1, mode: MODE.PASSTHRU, reason: 11, reasonName: 'CMD_PASSTHRU', flags: SF.BRIDGE_SEEN, fill: 0, ready: 0, trialLeftS: 0, ...o });
const withReason = (mode, name, o = {}) => st({ mode, reason: REASONS.indexOf(name), reasonName: name, ...o });

test('every mode x reason gives words in both languages and a known tone', () => {
  for (const lang of ['ru', 'en']) {
    for (const mode of [0, 1, 2, 3]) {
      for (const r of REASONS) {
        const d = describe(withReason(mode, r, { ready: RB.ASC }), 'connected', lang);
        assert.ok(d.title && d.title !== `reason.${r}` && !d.title.startsWith('reason.'), `${mode}/${r}/${lang}: ${d.title}`);
        assert.ok(['good', 'neutral', 'warn', 'error'].includes(d.tone));
        assert.ok(d.icon);
      }
    }
  }
});

test('help is "assisting" only when the bridge is in ASSIST, and "working" only when a layer is ready', () => {
  const on = describe(withReason(MODE.ASSIST, 'NONE', { ready: RB.ASC }), 'connected', 'en');
  assert.deepEqual([on.tone, on.assisting], ['good', true]);
  const idle = describe(withReason(MODE.ASSIST, 'NONE', { ready: 0 }), 'connected', 'en');
  assert.equal(idle.tone, 'neutral');
  assert.match(idle.title, /nothing to help with/);
  assert.equal(describe(withReason(MODE.PASSTHRU, 'CMD_PASSTHRU'), 'connected', 'en').assisting, false);
  assert.equal(describe(withReason(MODE.HW_BYPASS, 'CMD_BYPASS'), 'connected', 'en').assisting, false);
});

test('a latch the hand set says how the hand lifts it', () => {
  const panic = describe(withReason(MODE.PASSTHRU, 'PANIC', { flags: SF.LATCH_SOFT }), 'connected', 'en');
  assert.equal(panic.tone, 'warn');
  assert.match(panic.hint, /Panic/);
  assert.match(describe(withReason(MODE.HW_BYPASS, 'CMD_BYPASS'), 'connected', 'ru').hint, /Panic/);
  assert.match(describe(withReason(MODE.PASSTHRU, 'CHORD'), 'connected', 'en').hint, /Back and Forward/);
});

test('faults are errors, user choices are warnings, no mouse is neutral', () => {
  assert.equal(describe(withReason(MODE.HW_BYPASS, 'FATAL'), 'connected', 'en').tone, 'error');
  assert.equal(describe(withReason(MODE.HW_BYPASS, 'CRASHLOOP'), 'connected', 'en').tone, 'error');
  assert.equal(describe(withReason(MODE.HW_BYPASS, 'PANIC_LONG'), 'connected', 'en').tone, 'warn');
  assert.equal(describe(withReason(MODE.HW_BYPASS, 'NO_DEVICE'), 'connected', 'en').tone, 'neutral');
});

test('without a connection the phone says the device keeps working by itself', () => {
  for (const c of ['idle', 'connecting', 'lost']) {
    const d = describe(null, c, 'en');
    assert.equal(d.known, false);
    assert.match(d.hint, /works on its own/);
    assert.equal(d.assisting, false);
  }
  assert.equal(describe(null, 'lost', 'en').tone, 'warn');
});

test('a bridge nobody has heard from is reported as no data, not guessed', () => {
  const d = describe(st({ bridge: 0, mode: MODE_UNKNOWN }), 'connected', 'en');
  assert.equal(d.known, false);
  assert.equal(d.title, STRINGS.en['shell.nodata']);
});

test('calibration wins the headline: assistance is off while it runs', () => {
  const d = describe(withReason(MODE.PASSTHRU, 'CMD_PASSTHRU', { flags: SF.CALIBRATING }), 'connected', 'ru');
  assert.match(d.title, /калибровка/);
});

test('the trial banner counts down from the status', () => {
  assert.equal(trialModel(st(), 'en'), null);
  const t = trialModel(st({ flags: SF.TRIAL, trialLeftS: 14 }), 'en');
  assert.equal(t.left, 14);
  assert.match(t.text, /14 s/);
});
