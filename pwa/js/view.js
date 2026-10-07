// Manifest + state + language -> a plain tree the DOM layer draws. Pure, so it is tested without a browser. The client knows nine control
// kinds and no screen by name: anything else becomes a neutral 'not supported' note, never an error.
import { pick, t } from './i18n.js';

export const KINDS = ['status', 'toggle', 'stepper', 'action', 'meter', 'note', 'file', 'group', 'text'];

const txt = (o, lang) => pick(o, lang);

function node(c, state, lang) {
  const base = { id: String(c.id ?? ''), key: c.key ?? null, label: txt(c.label, lang), help: txt(c.help, lang) };
  const v = c.key != null ? state[c.key] : undefined;
  switch (c.type) {
    case 'toggle':
      return { ...base, kind: 'toggle', on: v === true, known: typeof v === 'boolean', confirm: c.confirm ?? 'none' };
    case 'stepper': {
      const min = Number.isInteger(c.min) ? c.min : 0, max = Number.isInteger(c.max) ? c.max : 10;
      const known = Number.isInteger(v);
      const marks = {};
      for (const [k, m] of Object.entries(c.marks && typeof c.marks === 'object' ? c.marks : {})) marks[k] = txt(m, lang);
      const value = known ? Math.min(Math.max(v, min), max) : min;
      return { ...base, kind: 'stepper', value, min, max, known, confirm: c.confirm ?? 'none', marks,
        valueText: known ? (marks[String(value)] || t(lang, 'stepper.value', { v: value, max })) : '—',
        levelText: known ? t(lang, 'stepper.value', { v: value, max }) : '', canDec: known && value > min, canInc: known && value < max };
    }
    case 'meter': {
      const min = Number.isInteger(c.min) ? c.min : 0, max = Number.isInteger(c.max) ? c.max : 100;
      const known = Number.isFinite(v);
      const value = known ? Math.min(Math.max(v, min), max) : min;
      const fmt = typeof c.fmt === 'string' && c.fmt.includes('{v}') ? c.fmt : '{v}';
      return { ...base, kind: 'meter', min, max, value, known, text: known ? fmt.replace('{v}', String(value)) : '—' };
    }
    case 'status': {
      const map = c.map && typeof c.map === 'object' ? c.map : {};
      return { ...base, kind: 'status', value: v ?? null, text: v != null && map[v] ? txt(map[v], lang) : v != null ? String(v) : '—' };
    }
    case 'action':
      return { ...base, kind: 'action', confirm: c.confirm ?? 'none', danger: c.danger === true };
    case 'note':
      return { ...base, kind: 'note' };
    case 'file':
      return { ...base, kind: 'file', op: c.op, accept: typeof c.accept === 'string' ? c.accept : '', maxBytes: Number.isInteger(c.max_bytes) ? c.max_bytes : 0,
        scope: c.scope === 'all' ? 'all' : 'active' };
    case 'text': {
      const maxLen = Number.isInteger(c.maxlen) && c.maxlen > 0 ? Math.min(c.maxlen, 40) : 24;
      return { ...base, kind: 'text', value: typeof v === 'string' ? v : '', known: typeof v === 'string', maxLen };
    }
    case 'group':
      return { ...base, kind: 'group', nodes: (Array.isArray(c.controls) ? c.controls : []).map((x) => node(x, state, lang)) };
    default:
      return { ...base, kind: 'unknown', note: t(lang, 'manifest.unknown') };
  }
}

export function buildView(manifest, state, lang) {
  const pages = (Array.isArray(manifest?.pages) ? manifest.pages : []).map((p, i) => ({
    id: String(p?.id ?? `page${i}`), title: txt(p?.title, lang),
    nodes: (Array.isArray(p?.controls) ? p.controls : []).map((c) => node(c ?? {}, state ?? {}, lang)),
  }));
  return { title: txt(manifest?.title, lang), pages };
}

// every text a node can show, for the "everything has a name in both languages" test and for accessibility checks
export function* walk(nodes) {
  for (const n of nodes) {
    yield n;
    if (n.nodes) yield* walk(n.nodes);
  }
}
