// Draws view models. The only place that touches the DOM. Text always goes in through textContent (never parsed as markup), so nothing the
// device sends can become markup. Every control is a native element, so screen readers, Switch Access and Voice Access see what is there.
import { t } from './i18n.js';

export function h(tag, attrs = {}, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === false || v == null) continue;
    if (k === 'class') el.className = v;
    else if (k === 'text') el.textContent = v;
    else if (k.startsWith('on')) el.addEventListener(k.slice(2), v);
    else el.setAttribute(k, v === true ? '' : String(v));
  }
  for (const c of kids.flat()) if (c != null) el.append(c);
  return el;
}

// a second tap on the same control within `ms` is ignored (a tremor often taps twice); the emergency button never goes through this
export function tapGuard(getMs, now = () => performance.now()) {
  const last = new Map();
  return (id, fn) => (ev) => {
    const ms = getMs(), n = now(), prev = last.get(id);
    if (prev !== undefined && n - prev < ms) { ev?.preventDefault?.(); return; }
    last.set(id, n);
    fn(ev);
  };
}

const field = (label, help) => [h('span', { class: 'label', text: label }), help ? h('p', { class: 'help', text: help }) : null];

// ---------------------------------------------------------------------------------------------------------------- node builders
// each returns { el, update(node) }
function buildToggle(n, ctx) {
  const id = `c-${n.id}`;
  const lab = h('span', { class: 'label', id: `${id}-l`, text: n.label });
  const mark = h('span', { class: 'mark' });
  const btn = h('button', { class: 'btn switch', type: 'button', role: 'switch', 'aria-labelledby': `${id}-l ${id}-m` });
  mark.id = `${id}-m`;
  const glyph = h('span', { 'aria-hidden': 'true' });
  btn.append(glyph, mark);
  btn.addEventListener('click', ctx.guard(n.id, () => ctx.onSet(n.key, !btn._on)));
  const help = n.help ? h('p', { class: 'help', text: n.help }) : null;
  const el = h('div', { class: 'control' }, lab, help, btn);
  return { el, update(m) {
    btn._on = m.on;
    btn.setAttribute('aria-checked', String(m.on));
    btn.disabled = !m.known || !ctx.online;
    mark.textContent = m.known ? t(ctx.lang, m.on ? 'common.on' : 'common.off') : '—';
    glyph.textContent = m.on ? '●' : '○';
    lab.textContent = m.label;
  } };
}

function buildStepper(n, ctx) {
  const id = `c-${n.id}`;
  const lab = h('span', { class: 'label', id: `${id}-l`, text: n.label });
  const val = h('div', { class: 'value', role: 'status', 'aria-live': 'polite', 'aria-atomic': 'true' });
  const dec = h('button', { class: 'btn', type: 'button', text: '−' });
  const inc = h('button', { class: 'btn', type: 'button', text: '+' });
  dec.addEventListener('click', ctx.guard(`${n.id}-`, () => ctx.onSet(n.key, dec._v - 1)));
  inc.addEventListener('click', ctx.guard(`${n.id}+`, () => ctx.onSet(n.key, inc._v + 1)));
  const help = n.help ? h('p', { class: 'help', text: n.help }) : null;
  const el = h('div', { class: 'control' }, lab, help, h('div', { class: 'stepper' }, dec, val, inc));
  return { el, update(m) {
    dec._v = inc._v = m.value;
    lab.textContent = m.label;
    val.replaceChildren(h('span', { text: m.valueText }), m.levelText && m.levelText !== m.valueText ? h('small', { text: m.levelText }) : null);
    dec.setAttribute('aria-label', `${t(ctx.lang, 'stepper.dec')}: ${m.label}`);
    inc.setAttribute('aria-label', `${t(ctx.lang, 'stepper.inc')}: ${m.label}`);
    dec.disabled = !m.canDec || !ctx.online;
    inc.disabled = !m.canInc || !ctx.online;
  } };
}

function buildMeter(n) {
  const id = `c-${n.id}`;
  const lab = h('span', { class: 'label', id: `${id}-l`, text: n.label });
  const bar = h('div');
  const meter = h('div', { class: 'meter', role: 'progressbar', 'aria-labelledby': `${id}-l` }, bar);
  const text = h('p', { class: 'meter-text' });
  const help = n.help ? h('p', { class: 'help', text: n.help }) : null;
  const el = h('div', { class: 'control' }, lab, help, meter, text);
  return { el, update(m) {
    lab.textContent = m.label;
    const pct = Math.round(((m.value - m.min) / Math.max(1, m.max - m.min)) * 100);
    bar.style.setProperty('--v', String(pct));
    meter.style.setProperty('--v', String(pct));
    meter.setAttribute('aria-valuemin', String(m.min));
    meter.setAttribute('aria-valuemax', String(m.max));
    meter.setAttribute('aria-valuenow', String(m.value));
    meter.setAttribute('aria-valuetext', m.text);
    text.textContent = m.text;
  } };
}

function buildStatus(n) {
  const lab = h('span', { class: 'label', text: n.label });
  const v = h('p', { class: 'meter-text', role: 'status', 'aria-live': 'polite' });
  const el = h('div', { class: 'control' }, lab, v);
  return { el, update(m) { lab.textContent = m.label; v.textContent = m.text; } };
}

function buildText(n, ctx) {
  const id = `c-${n.id}`;
  const lab = h('label', { class: 'label', for: `${id}-i` });
  const help = h('p', { class: 'help' });
  const input = h('input', { class: 'text', type: 'text', id: `${id}-i`, autocomplete: 'off', autocapitalize: 'sentences', spellcheck: 'false' });
  const save = h('button', { class: 'btn wide', type: 'submit' });
  const el = h('form', { class: 'control', autocomplete: 'off' }, lab, help, input, save);        // a form: the phone's own 'Done' key submits it too
  let model = n;
  const go = ctx.guard(n.id, () => ctx.onSet(model.key, input.value.trim()));
  el.addEventListener('submit', (e) => { e.preventDefault(); go(e); });
  return { el, update(m) {
    model = m;
    lab.textContent = m.label;
    help.textContent = m.help;
    help.hidden = !m.help;
    input.maxLength = m.maxLen;
    if (document.activeElement !== input) input.value = m.value;       // never overwrite what the person is typing
    save.textContent = t(ctx.lang, 'text.save');
    save.disabled = !m.known || !ctx.online;
    input.disabled = !m.known || !ctx.online;
  } };
}

function buildNote(n) {
  const p = h('p', { class: 'muted' });
  const el = h('div', { class: 'control' }, p);
  return { el, update(m) { p.textContent = m.label || m.help; } };
}

function buildUnknown(n) {
  const p = h('p', { class: 'muted' });
  const lab = h('span', { class: 'label' });
  const el = h('div', { class: 'control' }, lab, p);
  return { el, update(m) { lab.textContent = m.label; p.textContent = m.note; } };
}

function buildAction(n, ctx) {
  const lab = h('span', { class: 'label' });
  const help = h('p', { class: 'help' });
  const main = h('button', { class: 'btn wide', type: 'button' });
  const yes = h('button', { class: 'btn wide', type: 'button' });
  const no = h('button', { class: 'btn wide', type: 'button' });
  const row = h('div', { class: 'confirm-row' }, yes, no);
  row.hidden = true;
  const el = h('div', { class: 'control' }, lab, help, main, row);
  const showConfirm = (on) => {
    main.hidden = on;
    row.hidden = !on;
    lab.hidden = !on && main.textContent === model.label;          // while asking, the heading says WHAT is being confirmed
    (on ? no : main).focus();
  };
  let model = n;
  main.addEventListener('click', ctx.guard(n.id, () => { if (model.confirm === 'two-step') showConfirm(true); else ctx.onAct(model.key, false, model); }));
  yes.addEventListener('click', ctx.guard(`${n.id}!`, () => { showConfirm(false); ctx.onAct(model.key, true, model); }));
  no.addEventListener('click', () => showConfirm(false));
  return { el, update(m) {
    model = m;
    lab.textContent = m.label;
    help.textContent = m.help;
    help.hidden = !m.help;
    main.textContent = m.actionLabel || m.label;
    lab.hidden = main.textContent === m.label;                            // a button that says the same as its heading needs no heading
    main.className = `btn wide${m.danger ? ' danger' : ''}`;
    yes.textContent = m.confirmLabel || t(ctx.lang, 'common.confirm');
    yes.className = `btn wide${m.danger ? ' danger' : ' primary'}`;
    no.textContent = t(ctx.lang, 'common.cancel');
    main.disabled = !ctx.online && !m.offlineOk;
  } };
}

function buildFile(n, ctx) {
  const lab = h('span', { class: 'label' });
  const help = h('p', { class: 'help' });
  const takes = n.op === 'bundle_put' || n.op === 'fw_put' || n.op === 'pkg_put';   // these receive a file; the card is the only thing handed to the phone
  const input = takes ? h('input', { type: 'file', hidden: true, accept: n.accept }) : null;
  const btn = h('button', { class: 'btn wide', type: 'button' });
  const drop = takes ? h('div', { class: 'drop' }) : null;
  const el = h('div', { class: 'control' }, lab, help, btn, input, drop);
  let model = n;
  const take = (file) => { if (file) ctx.onFile(model, file); if (input) input.value = ''; };
  btn.addEventListener('click', ctx.guard(n.id, () => (input ? input.click() : ctx.onFile(model, null))));
  input?.addEventListener('change', () => take(input.files[0]));
  if (drop) {
    for (const ev of ['dragenter', 'dragover']) drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add('over'); });
    for (const ev of ['dragleave', 'drop']) drop.addEventListener(ev, () => drop.classList.remove('over'));
    drop.addEventListener('drop', (e) => { e.preventDefault(); take(e.dataTransfer?.files?.[0]); });
  }
  return { el, update(m) {
    model = m;
    lab.textContent = m.label;
    help.textContent = m.help;
    help.hidden = !m.help;
    btn.textContent = m.op === 'bundle_put' || m.op === 'fw_put' || m.op === 'pkg_put' ? t(ctx.lang, 'file.choose') : m.label;
    lab.hidden = btn.textContent === m.label;
    btn.disabled = !ctx.online;
    if (drop) drop.textContent = t(ctx.lang, 'file.drop');
  } };
}

function buildChoice(n, ctx) {
  const lab = h('span', { class: 'label', id: `c-${n.id}-l` });
  const help = h('p', { class: 'help' });
  const row = h('div', { class: 'choice', role: 'group', 'aria-labelledby': `c-${n.id}-l` });
  const el = h('div', { class: 'control' }, lab, help, row);
  let model = n;
  return { el, update(m) {
    model = m;
    lab.textContent = m.label;
    help.textContent = m.help;
    help.hidden = !m.help;
    row.replaceChildren(...m.options.map((o) => h('button', { class: 'btn', type: 'button', text: o.label, 'aria-pressed': String(o.value === m.value),
      onclick: ctx.guard(`${n.id}:${o.value}`, () => ctx.onChoice(model.key, o.value)) })));
  } };
}

function buildGroup(n, ctx) {
  const lab = h('h2', { class: 'label' });
  const inner = h('div');
  const mount = new Mount(inner, ctx);
  const el = h('div', {}, lab, inner);
  return { el, update(m) { lab.textContent = m.label; lab.hidden = !m.label; mount.update(m.nodes); } };
}

const BUILDERS = { toggle: buildToggle, stepper: buildStepper, meter: buildMeter, status: buildStatus, note: buildNote, unknown: buildUnknown,
  action: buildAction, file: buildFile, choice: buildChoice, group: buildGroup, text: buildText };

// Keyed reuse: while the list of (id, kind) stays the same only values change, so focus and scroll position survive every update.
export class Mount {
  constructor(container, ctx) { this.container = container; this.ctx = ctx; this.items = []; this.sig = ''; }
  update(nodes) {
    const sig = nodes.map((n) => `${n.id}:${n.kind}`).join('|');
    if (sig !== this.sig) {
      this.items = nodes.map((n) => (BUILDERS[n.kind] || BUILDERS.unknown)(n, this.ctx));
      this.container.replaceChildren(...this.items.map((i) => i.el));
      this.sig = sig;
    }
    nodes.forEach((n, i) => this.items[i].update(n));
  }
}

// ---------------------------------------------------------------------------------------------------------------- the whole page
export class UI {
  constructor(root, handlers) {
    this.root = root;
    this.hd = handlers;
    this.ctx = { lang: 'en', online: false, guard: tapGuard(() => this.hd.tapMs()), onSet: (k, v) => this.hd.set(k, v), onAct: (k, c, n) => this.hd.act(k, c, n),
      onFile: (n, f) => this.hd.file(n, f), onChoice: (k, v) => this.hd.choice(k, v) };
    this.stateIcon = h('div', { class: 'icon', 'aria-hidden': 'true' });
    this.stateTitle = h('p', { class: 'title' });
    this.stateHint = h('p', { class: 'hint' });
    this.stateSlot = h('p', { class: 'slot-line' });
    this.state = h('section', { class: 'state', role: 'status', 'aria-live': 'polite' }, this.stateIcon, h('div', {}, this.stateTitle, this.stateSlot, this.stateHint));
    this.trialText = h('p', { class: 'trial-text' });
    this.keep = h('button', { class: 'btn good', type: 'button', onclick: () => this.hd.confirm(true) });
    this.undo = h('button', { class: 'btn', type: 'button', onclick: () => this.hd.confirm(false) });
    this.trial = h('section', { class: 'banner trial', role: 'alert' }, this.trialText, h('div', { class: 'row' }, this.keep, this.undo));
    this.msgText = h('p');
    this.msgClose = h('button', { class: 'btn', type: 'button', onclick: () => this.hd.dismiss() });
    this.msgRetry = h('button', { class: 'btn primary', type: 'button', onclick: () => this.hd.retry() });
    this.msg = h('div', { class: 'message', role: 'status' }, this.msgText, h('div', { class: 'row' }, this.msgRetry, this.msgClose));
    this.updText = h('p');
    this.updGo = h('button', { class: 'btn primary', type: 'button', onclick: () => this.hd.applyUpdate() });
    this.updLater = h('button', { class: 'btn', type: 'button', onclick: () => this.hd.dismissUpdate() });
    this.upd = h('section', { class: 'banner' }, this.updText, h('div', { class: 'row' }, this.updGo, this.updLater));
    // closed by default: one row says 'change slot' and the controls stay in view; a slot picked closes it again
    this.slotTitle = h('h2', { class: 'label', id: 'slots-title' });
    this.slotRow = h('div', { class: 'slot-grid', role: 'group', 'aria-labelledby': 'slots-title' });
    this.slotSummary = h('summary', { class: 'btn slot-summary' });
    this.slots = h('details', { class: 'slots' }, this.slotSummary, this.slotTitle, this.slotRow);
    this.slotSig = '';
    this.nav = h('nav', { class: 'pages' });
    this.connSteps = h('ol');
    this.connNotes = h('div');
    this.connBtn = h('button', { class: 'btn primary', type: 'button', onclick: () => this.hd.connect() });
    this.connect = h('section', { class: 'connect' }, this.connBtn, this.connSteps, this.connNotes);
    this.pageBox = h('main');
    this.page = new Mount(this.pageBox, this.ctx);
    this.stop = h('button', { class: 'btn danger', type: 'button', onclick: () => this.hd.stop() });
    this.stopNote = h('p', { class: 'note' });
    this.safety = h('footer', { class: 'safety' }, h('div', { class: 'inner' }, this.stop, this.stopNote));
    // what must never scroll out of sight (the state, a running trial, a question or an error) lives in one sticky block at the top
    this.top = h('div', { class: 'top' }, this.state, this.trial, this.msg, this.upd);
    root.replaceChildren(this.top, this.slots, this.nav, this.connect, this.pageBox, this.safety);
    this.navSig = '';
  }

  update(m) {
    const { lang } = m;
    this.ctx.lang = lang;
    this.ctx.online = m.online;
    document.documentElement.lang = lang;
    this.state.dataset.tone = m.shell.tone;
    this.stateIcon.textContent = m.shell.icon;
    this.stateTitle.textContent = m.shell.title;
    this.stateHint.textContent = m.shell.hint;
    this.stateHint.hidden = !m.shell.hint || !!m.shell.quiet;
    this.stateSlot.hidden = !m.slots;
    if (m.slots) this.stateSlot.textContent = m.slots.current;
    this.slots.hidden = !m.slots;
    if (m.slots) this.updateSlots(m.slots, m.online);
    this.trial.hidden = !m.trial;
    if (m.trial) {
      this.trialText.textContent = m.trial.text;
      this.keep.textContent = t(lang, 'trial.keep');
      this.undo.textContent = t(lang, 'trial.undo');
    }
    this.upd.hidden = !m.updateAvailable;
    this.updText.textContent = t(lang, 'update.available');
    this.updGo.textContent = t(lang, 'update.apply');
    this.updLater.textContent = t(lang, 'update.later');
    this.msg.hidden = !m.message;
    if (m.message) {
      this.msg.dataset.kind = m.message.kind;
      this.msgText.textContent = m.message.text;
      this.msgClose.textContent = t(lang, 'common.close');
      this.msgRetry.textContent = t(lang, 'retry.again');
      this.msgRetry.hidden = !m.message.retry;                          // asked for the button on the device: one tap repeats the request
    }
    // connect screen or the pages
    const showConnect = m.connect.show;
    this.connect.hidden = !showConnect;
    this.nav.hidden = showConnect || !m.pages.length;
    this.pageBox.hidden = showConnect;
    if (showConnect) {
      this.connSteps.replaceChildren(...m.connect.steps.map((s) => h('li', { text: s })));
      this.connSteps.hidden = !m.connect.canConnect;
      this.connBtn.textContent = m.connect.button;
      this.connBtn.hidden = !m.connect.canConnect;
      this.connBtn.disabled = m.connect.busy;
      this.connNotes.replaceChildren(...m.connect.notes.filter(Boolean).map((s) => h('p', { class: 'muted', text: s })));
    } else {
      const sig = m.pages.map((p) => `${p.id}:${p.title}`).join('|') + m.currentPage;
      if (sig !== this.navSig) {
        this.nav.replaceChildren(...m.pages.map((p) => h('button', { class: 'btn', type: 'button', text: p.title, 'aria-current': p.id === m.currentPage ? 'page' : false,
          onclick: () => this.hd.page(p.id) })));
        this.navSig = sig;
      }
      const cur = m.pages.find((p) => p.id === m.currentPage) || m.pages[0];
      this.page.update(cur ? cur.nodes : []);
    }
    this.stop.textContent = `■ ${t(lang, 'shell.stop')}`;
    this.stop.disabled = !m.canStop;
    this.stopNote.textContent = m.canStop ? '' : t(lang, 'shell.stop.offline');
    this.stopNote.hidden = m.canStop;
  }
}

UI.prototype.updateSlots = function updateSlots(s, online) {
  this.slotTitle.textContent = s.title;
  this.slotSummary.textContent = `${s.change}: ${s.current}`;
  const sig = s.items.map((i) => i.k).join();
  if (sig !== this.slotSig) {                                          // built once; afterwards only the words and the pressed state change
    this.slotBtns = s.items.map((i) => {
      const name = h('span', { class: 'slot-name' });
      const hint = h('small', { class: 'slot-hint' });
      const btn = h('button', { class: 'btn slot', type: 'button' }, name, hint);
      btn.addEventListener('click', this.ctx.guard(`slot${i.k}`, () => { this.slots.open = false; this.hd.slot(i.k); }));
      return { btn, name, hint };
    });
    this.slotRow.replaceChildren(...this.slotBtns.map((x) => x.btn));
    this.slotSig = sig;
  }
  s.items.forEach((i, idx) => {
    const x = this.slotBtns[idx];
    x.name.textContent = i.title;
    x.hint.textContent = i.hint;
    x.btn.setAttribute('aria-pressed', String(i.active));
    x.btn.disabled = !online;
  });
};

export function download(bytes, name, type = 'application/octet-stream') {
  const url = URL.createObjectURL(new Blob([bytes], { type }));
  const a = h('a', { href: url, download: name });
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 10000);
}
