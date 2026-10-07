// What can be checked about accessibility, privacy and offline behaviour without a screen: contrast of every colour pair in every theme,
// the sizes in the stylesheet, the absence of gestures / markup injection / network calls in the sources, the offline file list.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, readdirSync, existsSync, statSync } from 'node:fs';
import { join, dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const read = (p) => readFileSync(join(ROOT, p), 'utf8');
const css = read('app.css');

function walkFiles(dir, out = []) {
  for (const e of readdirSync(join(ROOT, dir), { withFileTypes: true })) {
    const rel = dir ? `${dir}/${e.name}` : e.name;
    if (e.isDirectory()) walkFiles(rel, out); else out.push(rel);
  }
  return out;
}

// ---- contrast
const lum = (hex) => {
  const [r, g, b] = [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16) / 255).map((c) => (c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4));
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
};
const ratio = (a, b) => { const [hi, lo] = [lum(a), lum(b)].sort((x, y) => y - x); return (hi + 0.05) / (lo + 0.05); };

// every rule that defines the palette (the light default, the two media-query defaults and the three explicit themes)
const palettes = [...css.matchAll(/([^{}]+)\{([^{}]*--bg:[^{}]*)\}/g)].map((m) => {
  const vars = Object.fromEntries([...m[2].matchAll(/--([\w-]+):\s*(#[0-9a-fA-F]{6})\s*;/g)].map((v) => [v[1], v[2]]));
  return { sel: m[1].trim().replace(/\s+/g, ' '), vars };
});
const TEXT = [['text', 'bg'], ['text', 'surface'], ['muted', 'bg'], ['muted', 'surface'], ['on-accent', 'accent'], ['on-good', 'good'], ['on-danger', 'danger'],
  ['good-fg', 'good-bg'], ['warn-fg', 'warn-bg'], ['error-fg', 'error-bg'], ['neutral-fg', 'neutral-bg']];
const UI = [['border', 'bg'], ['border', 'surface'], ['focus', 'bg'], ['focus', 'surface'], ['accent', 'bg'], ['good', 'bg'], ['danger', 'bg']];

test('the stylesheet defines six palettes: light, dark, contrast as defaults and as explicit choices', () => {
  assert.equal(palettes.length, 6, palettes.map((p) => p.sel).join(' ; '));
  for (const p of palettes) for (const [a, b] of [...TEXT, ...UI]) assert.ok(p.vars[a] && p.vars[b], `${p.sel}: --${a} / --${b}`);
});

test('text contrast is at least 7:1 (WCAG AAA) in every palette', () => {
  for (const p of palettes) for (const [a, b] of TEXT) assert.ok(ratio(p.vars[a], p.vars[b]) >= 7, `${p.sel}: ${a} on ${b} = ${ratio(p.vars[a], p.vars[b]).toFixed(2)}`);
});

test('borders, focus rings and filled controls stand out at least 3:1 from their background', () => {
  for (const p of palettes) for (const [a, b] of UI) assert.ok(ratio(p.vars[a], p.vars[b]) >= 3, `${p.sel}: ${a} on ${b} = ${ratio(p.vars[a], p.vars[b]).toFixed(2)}`);
});

// ---- sizes
const rootPx = 16 * (parseFloat(css.match(/html\s*\{[^}]*font-size:\s*([\d.]+)%/)[1]) / 100);
const rem = (v) => parseFloat(v) * rootPx;

test('the base text is 20 px at the default browser size and everything is in rem', () => {
  assert.equal(rootPx, 20);
  assert.ok(!/font-size:\s*[\d.]+px/.test(css), 'font sizes must not be fixed in px');
  for (const m of css.matchAll(/font-size:\s*([\d.]+)rem/g)) assert.ok(rem(m[1]) >= 16, `font-size ${m[1]}rem is below 16 px`);
});

test('touch targets are at least 64 px, the stop button 80 px, the stepper buttons share the full width', () => {
  assert.ok(rem(css.match(/--target:\s*([\d.]+)rem/)[1]) >= 64);
  assert.match(css, /\.btn\s*\{[^}]*min-height:\s*var\(--target\)/);
  assert.match(css, /\.btn\s*\{[^}]*min-width:\s*var\(--target\)/);
  assert.ok(rem(css.match(/\.safety \.btn\s*\{[^}]*min-height:\s*([\d.]+)rem/)[1]) >= 80);
  assert.match(css, /\.stepper\s*\{[^}]*grid-template-columns:\s*1fr 1fr/);          // two wide buttons, the value above them
  assert.ok(rem(css.match(/--gap:\s*([\d.]+)rem/)[1]) >= 16);
});

test('focus is always visible, motion can be turned off, high contrast follows the phone', () => {
  assert.match(css, /:focus-visible\s*\{[^}]*outline:\s*4px/);
  assert.match(css, /prefers-reduced-motion:\s*reduce/);
  assert.match(css, /prefers-contrast:\s*more/);
  assert.match(css, /prefers-color-scheme:\s*dark/);
  assert.ok(!/outline:\s*none/.test(css));
});

// ---- sources
const js = walkFiles('js').filter((f) => f.endsWith('.js') && !f.endsWith('constants.js')).map((f) => [f, read(f)]);

test('nothing a device sends can become markup, code or a URL', () => {
  for (const [f, src] of js) {
    for (const bad of ['innerHTML', 'outerHTML', 'insertAdjacentHTML', 'document.write', 'eval(', 'new Function', 'setAttribute(\'style\'', 'srcdoc', 'importScripts']) {
      assert.ok(!src.includes(bad), `${f} uses ${bad}`);
    }
  }
});

test('the page stays dumb: no cryptography, no keys, no way to read what the device sealed', () => {
  for (const [f, src] of js) {
    const code = src.replace(/\/\/.*$/gm, '');
    assert.ok(!/\bcrypto\b|subtle|SubtleCrypto|importKey|exportKey|getRandomValues|decrypt\s*\(|encrypt\s*\(|ed25519|x25519|chacha|AES-|HKDF|Uint8Array\.from\(.*seed/i.test(code), `${f} touches cryptography`);
  }
});

test('the page makes no network requests of its own and has no analytics', () => {
  for (const [f, src] of js) {
    assert.ok(!/\bfetch\s*\(|XMLHttpRequest|sendBeacon|EventSource|importScripts/.test(src), `${f} talks to the network`);
    assert.ok(!/https?:\/\//.test(src.replace(/\/\/.*$/gm, '')), `${f} contains a URL`);
    assert.ok(!/analytics|gtag|telemetry|tracker|pixel/i.test(src.replace(/\/\/.*$/gm, '')), `${f} mentions analytics`);
  }
  const sw = read('sw.js');
  assert.ok(!/https?:\/\//.test(sw.replace(/\/\/.*$/gm, '')));
});

test('no gesture is ever required: no touch/pointer streams, double taps, long presses, swipes or hover', () => {
  for (const [f, src] of js) {
    assert.ok(!/touchstart|touchmove|touchend|pointerdown|pointermove|pointerup|mousedown|mouseup|mouseover|mouseenter|dblclick|contextmenu|wheel|swipe|longpress|keydown/.test(src), `${f} uses a gesture`);
  }
  assert.ok(!/:hover/.test(css), 'hover-only styling');
});

test('the page policy forbids inline code, inline styles and any host but itself', () => {
  const html = read('index.html');
  const csp = html.match(/Content-Security-Policy"\s+content="([^"]+)"/)[1];
  assert.ok(!/unsafe-inline|unsafe-eval|\*|https?:/.test(csp), csp);
  assert.match(csp, /default-src 'none'/);
  assert.match(csp, /script-src 'self'/);
  assert.match(csp, /connect-src 'self'/);
  assert.ok(!/<script(?![^>]*\bsrc=)/.test(html), 'inline script');
  assert.ok(!/\sstyle=/.test(html), 'inline style');
  assert.ok(!/on(click|load|error)=/.test(html));
  assert.ok(!/<link[^>]+href="https?:/.test(html) && !/<script[^>]+src="https?:/.test(html));
  assert.match(html, /<html lang="ru">/);
  assert.match(html, /name="viewport"/);
});

// ---- offline
test('everything the page imports is in the service worker list, and every listed file exists', () => {
  const sw = read('sw.js');
  const listed = [...sw.match(/const FILES = \[([^\]]*)\]/s)[1].matchAll(/'([^']+)'/g)].map((m) => m[1]);
  assert.ok(listed.includes('./') && listed.includes('index.html') && listed.includes('app.css') && listed.includes('manifest.webmanifest'));
  for (const f of listed) if (f !== './') assert.ok(existsSync(join(ROOT, f)), `${f} is listed but missing`);
  const seen = new Set(['js/app.js']);
  const queue = ['js/app.js'];
  while (queue.length) {
    const f = queue.pop();
    for (const m of read(f).matchAll(/from\s+'(\.[^']+)'/g)) {
      const p = join(dirname(f), m[1]).replace(/\\/g, '/');
      if (!seen.has(p)) { seen.add(p); queue.push(p); }
    }
  }
  for (const f of seen) assert.ok(listed.includes(f), `${f} is imported but not cached`);
  const html = read('index.html');
  for (const m of html.matchAll(/(?:href|src)="([^"#]+)"/g)) assert.ok(listed.includes(m[1]), `${m[1]} is used by index.html but not cached`);
  assert.ok(!listed.some((f) => f.startsWith('tests/')), 'test files must not be cached');
});

test('the web app manifest is valid and its icons are real PNGs of the stated size', () => {
  const m = JSON.parse(read('manifest.webmanifest'));
  assert.equal(m.display, 'standalone');
  assert.ok(m.start_url && m.scope && m.name && m.short_name);
  assert.ok(m.icons.some((i) => i.sizes === '192x192') && m.icons.some((i) => i.sizes === '512x512'));
  assert.ok(m.icons.some((i) => i.purpose === 'maskable'));
  for (const i of m.icons.filter((x) => x.type === 'image/png')) {
    const b = readFileSync(join(ROOT, i.src));
    assert.deepEqual([...b.subarray(0, 4)], [0x89, 0x50, 0x4e, 0x47]);
    assert.equal(`${b.readUInt32BE(16)}x${b.readUInt32BE(20)}`, i.sizes);
  }
  for (const i of m.icons) assert.ok(statSync(join(ROOT, i.src)).size > 100);
});
