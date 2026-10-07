// The safety shell's model: what to say about the device, from the fixed 20-byte status alone. No manifest is involved, so this works
// with any firmware that speaks the status layout. Pure: text and a tone in, nothing about the DOM.
import { SF, RB, MODE_UNKNOWN, SLOT_COUNT } from './constants.js';
import { t, STRINGS } from './i18n.js';

export const MODE = { HW_BYPASS: 0, PROBE: 1, PASSTHRU: 2, ASSIST: 3 };

// tone: good | neutral | warn | error  (always shown with an icon and words, never by colour alone)
export function describe(status, conn, lang) {
  if (conn !== 'connected') {
    const key = conn === 'connecting' ? 'conn.connecting' : conn === 'lost' ? 'conn.lost' : 'conn.none';
    return { tone: conn === 'lost' ? 'warn' : 'neutral', icon: '○', title: t(lang, key), hint: t(lang, 'conn.independent'), quiet: true, assisting: false, known: false };
  }
  if (!status || status.bridge === 0 || status.mode === MODE_UNKNOWN) {
    return { tone: 'warn', icon: '?', title: t(lang, 'shell.nodata'), hint: t(lang, 'reason.STALE_LINK'), assisting: false, known: false };
  }
  const r = status.reasonName;
  const hint = STRINGS.en[`hint.${r}`] ? t(lang, `hint.${r}`) : '';
  const out = (tone, icon, title, h = hint, assisting = false) => ({ tone, icon, title, hint: h, assisting, known: true });
  if (status.flags & SF.CALIBRATING) return out('neutral', '◔', t(lang, 'shell.calibrating'), '');
  switch (status.mode) {
    case MODE.ASSIST:
      return status.ready & (RB.ASC | RB.TREMOR) ? out('good', '✓', t(lang, 'reason.NONE'), '', true) : out('neutral', '✓', t(lang, 'shell.idle'), '', true);
    case MODE.PASSTHRU:
      if (r === 'CMD_PASSTHRU' || r === 'WAIT_PC' || r === 'NONE') return out('neutral', '○', t(lang, `reason.${r === 'NONE' ? 'CMD_PASSTHRU' : r}`), '');
      return out('warn', '‖', t(lang, `reason.${r}`));
    case MODE.PROBE:
      return out('neutral', '…', t(lang, `reason.${r === 'NONE' ? 'PROBING' : r}`), '');
    default: {                                                       // hardware bypass
      const user = r === 'CMD_BYPASS' || r === 'PANIC_LONG';
      return out(user ? 'warn' : r === 'NO_DEVICE' ? 'neutral' : 'error', user ? '⇄' : '!', t(lang, `reason.${r}`));
    }
  }
}

export function trialModel(status, lang) {
  if (!status || !(status.flags & SF.TRIAL)) return null;
  return { left: status.trialLeftS, text: t(lang, 'trial.compact', { n: status.trialLeftS }) };
}

// The hardware slots, from the fixed status (which slot is active, which hold a profile) and the names the device announces in its state.
// Part of the safety shell on purpose: whatever layout a slot brings, the way to the next slot is always here.
export function slotsModel(status, state, conn, lang) {
  if (conn !== 'connected' || !status) return null;
  const items = [];
  for (let k = 0; k < SLOT_COUNT; k++) {
    const name = typeof state?.[`slot.${k}.name`] === 'string' ? state[`slot.${k}.name`] : '';
    const has = !!(status.slotMask & (1 << k));
    const label = name || t(lang, 'slot.default', { n: k + 1 });
    items.push({ k, n: k + 1, name, label, title: name ? `${k + 1} · ${name}` : label, has, active: status.slot === k, hint: t(lang, has ? 'slot.has' : 'slot.empty') });
  }
  const cur = items[Math.min(status.slot, SLOT_COUNT - 1)];
  return { title: t(lang, 'slot.title'), change: t(lang, 'slot.change'), items, active: status.slot, current: t(lang, 'slot.current', { n: cur.n }) + (cur.name ? ` · ${cur.name}` : '') };
}
