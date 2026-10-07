// The glue: preferences, the connection with automatic reconnects, the service worker, and the model the UI draws. No screens are
// described here; the pages come from the device's manifest, plus one fixed page for the app's own settings and the hardware bypass.
import { Session, SessionError } from './session.js';
import { BleTransport } from './transport/ble.js';
import { WsTransport } from './transport/ws.js';
import { Store } from './store.js';
import { UI, download } from './dom.js';
import { buildView } from './view.js';
import { describe, trialModel } from './shell.js';
import { detectLang, t, LANGS } from './i18n.js';
import { VERSION } from './build.js';

const THEMES = ['auto', 'light', 'dark', 'contrast'];
const TAPS = [0, 400, 800];

export async function start(root = document.getElementById('app'), env = {}) {
  const store = env.store || new Store();
  const prefs = {
    lang: (await store.get('lang')) || detectLang(),
    theme: (await store.get('theme')) || 'auto',
    tapMs: Number.isFinite(await store.get('tapMs')) ? await store.get('tapMs') : 400,
  };
  if (!LANGS.includes(prefs.lang)) prefs.lang = 'en';
  const params = new URLSearchParams(location.search);
  const transport = env.transport || (params.get('transport') === 'ws'
    ? new WsTransport(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`)
    : BleTransport.supported() ? new BleTransport({ store }) : null);
  const session = transport ? new Session(transport, { store, lang: prefs.lang }) : null;
  const app = { page: 'main', message: null, updateAvailable: false, busy: false, failed: '', retry: 0, retryTimer: null, swReg: null, trialUntil: 0 };

  const say = (kind, text) => { app.message = { kind, text }; render(); };
  const errText = (e) => (e instanceof SessionError ? t(prefs.lang, e.key.startsWith('err.') ? e.key : 'err.unknown') : t(prefs.lang, 'err.unknown'));
  const guarded = async (fn) => { try { return await fn(); } catch (e) { say('error', errText(e)); return null; } };

  const ui = new UI(root, {
    tapMs: () => prefs.tapMs,
    set: (key, value) => guarded(() => session.set(key, value)),
    act: (key, confirmed, node) => guarded(async () => {
      if (key === '__bypass') { await session.hardBypass(); say('info', t(prefs.lang, 'bypass.done')); return; }
      await session.act(key, confirmed);
    }),
    choice: (key, value) => {
      if (key === 'lang') { prefs.lang = value; session && (session.lang = value); store.set('lang', value); }
      if (key === 'theme') { prefs.theme = value; applyTheme(); store.set('theme', value); }
      if (key === 'taps') { prefs.tapMs = Number(value); store.set('tapMs', prefs.tapMs); }
      render();
    },
    file: (node, file) => guarded(async () => {
      if (node.op === 'bundle_get') {
        const bytes = await session.getBundle();
        download(bytes, `dataopen-${new Date().toISOString().slice(0, 10)}.dobundle`);
        say('info', t(prefs.lang, 'file.saved'));
      } else if (file) {
        if (node.maxBytes && file.size > node.maxBytes) { say('error', t(prefs.lang, 'file.too_big')); return; }
        say('info', t(prefs.lang, 'file.working'));
        await session.putBundle(new Uint8Array(await file.arrayBuffer()));
        say('info', t(prefs.lang, 'file.loaded'));
      }
    }),
    confirm: (keep) => guarded(() => session.confirm(keep)),
    stop: () => session && session.stop().then(() => say('info', t(prefs.lang, 'shell.stopped')), (e) => say('error', errText(e))),
    connect: () => connect(true),
    page: (id) => { app.page = id; render(); },
    dismiss: () => { app.message = null; render(); },
    applyUpdate: () => { app.swReg?.waiting?.postMessage('skipWaiting'); },
    dismissUpdate: () => { app.updateAvailable = false; render(); },
  });

  function applyTheme() {
    if (prefs.theme === 'auto') delete document.documentElement.dataset.theme;
    else document.documentElement.dataset.theme = prefs.theme;
  }

  function settingsPage() {
    const lang = prefs.lang;
    return { id: '__settings', title: t(lang, 'nav.settings'), nodes: [
      { kind: 'action', id: 'bypass', key: '__bypass', label: t(lang, 'bypass.title'), help: t(lang, 'bypass.hint'), actionLabel: t(lang, 'bypass.do'),
        confirm: 'two-step', confirmLabel: t(lang, 'bypass.confirm'), danger: true },
      { kind: 'choice', id: 'lang', key: 'lang', label: t(lang, 'settings.lang'), help: '', value: lang, options: [{ value: 'ru', label: 'Русский' }, { value: 'en', label: 'English' }] },
      { kind: 'choice', id: 'theme', key: 'theme', label: t(lang, 'settings.theme'), help: '', value: prefs.theme, options: THEMES.map((v) => ({ value: v, label: t(lang, `settings.theme.${v}`) })) },
      { kind: 'choice', id: 'taps', key: 'taps', label: t(lang, 'settings.taps'), help: t(lang, 'settings.taps.hint'), value: prefs.tapMs,
        options: TAPS.map((v) => ({ value: v, label: t(lang, `settings.taps.${v}`) })) },
      { kind: 'note', id: 'about', label: `${t(lang, 'settings.offline')} ${t(lang, 'settings.about')}: ${env.version || VERSION}`, help: '' },
    ] };
  }

  function model() {
    const lang = prefs.lang;
    const conn = session ? session.conn : 'idle';
    const connected = conn === 'connected';
    const status = session?.status ?? null;
    let pages = [];
    if (connected && session.manifest) {
      const v = buildView(session.manifest, session.state, lang);
      pages = v.pages;
    }
    if (connected) pages = [...pages, settingsPage()];
    const supported = !!transport;
    const shell = supported ? describe(status, conn, lang) : { tone: 'warn', icon: '!', title: t(lang, 'conn.unsupported'), hint: t(lang, 'conn.unsupported.hint') };
    return {
      lang, shell, trial: connected ? trialModel(status, lang) : null, pages, currentPage: pages.some((p) => p.id === app.page) ? app.page : pages[0]?.id,
      online: connected, canStop: !!session?.linked, updateAvailable: app.updateAvailable, message: app.message,
      connect: { show: !connected, canConnect: supported, busy: conn === 'connecting' || app.busy,
        title: t(lang, 'app.title'), steps: [1, 2, 3].map((i) => t(lang, `conn.step${i}`)), note: app.failed || (supported ? t(lang, 'conn.pairing') : ''),
        button: t(lang, conn === 'lost' ? 'conn.retry' : 'conn.button') },
    };
  }

  function render() {
    document.title = t(prefs.lang, 'app.title');
    ui.update(model());
  }

  async function connect(interactive) {
    if (!session || session.conn === 'connecting') return;
    clearTimeout(app.retryTimer);
    app.failed = '';
    try {
      await session.connect({ interactive });
      app.retry = 0;
    } catch (e) {
      const cancelled = /cancel|chooser|NotFoundError/i.test(String(e?.message || e) + String(e?.name || ''));
      app.failed = t(prefs.lang, cancelled ? 'conn.cancelled' : e?.message === 'needs-gesture' ? 'conn.pairing' : 'conn.failed');
      if (!interactive && e?.message !== 'needs-gesture') scheduleRetry();
    }
    render();
  }

  function scheduleRetry() {
    const wait = Math.min(1000 * 2 ** app.retry++, 15000);
    app.retryTimer = setTimeout(() => connect(false), wait);
  }

  if (session) {
    for (const ev of ['conn', 'status', 'state', 'manifest']) session.on(ev, render);
    session.on('conn', (c) => { if (c === 'lost') { app.retry = 0; scheduleRetry(); } });
  }
  applyTheme();
  render();
  if (session && !params.has('noauto')) connect(false);                  // a remembered device reconnects without any tap; otherwise the button waits
  return { session, ui, prefs, app, render };
}

if (typeof document !== 'undefined' && document.getElementById('app') && !globalThis.__DATAOPEN_NO_AUTOSTART__) {
  start().then((ctx) => {
    if (!('serviceWorker' in navigator)) return;
    navigator.serviceWorker.register('sw.js').then((reg) => {
      ctx.app.swReg = reg;
      const watch = (w) => w && w.addEventListener('statechange', () => { if (w.state === 'installed' && navigator.serviceWorker.controller) { ctx.app.updateAvailable = true; ctx.render(); } });
      if (reg.waiting && navigator.serviceWorker.controller) { ctx.app.updateAvailable = true; ctx.render(); }
      reg.addEventListener('updatefound', () => watch(reg.installing));
    }).catch(() => {});
    let reloaded = false;
    navigator.serviceWorker.addEventListener('controllerchange', () => { if (!reloaded) { reloaded = true; location.reload(); } });
  });
}
