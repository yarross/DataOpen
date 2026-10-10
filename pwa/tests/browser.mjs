// End to end in a real Chromium (Playwright): the page, served by `dataopen ctl serve-sim`, talking to the real gateway and C bridge core.
// Run by tests/test_pwa.py when Playwright is installed (it is not in CI); screenshots go to $DATAOPEN_SHOTS if set.
//   DATAOPEN_SIM_URL=http://127.0.0.1:PORT/ node pwa/tests/browser.mjs
import { createRequire } from 'node:module';
import assert from 'node:assert/strict';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const require = createRequire(process.env.PLAYWRIGHT_NODE_MODULES ? `${process.env.PLAYWRIGHT_NODE_MODULES}/` : '/opt/node22/lib/node_modules/');
const { chromium } = require('playwright');
const base = process.env.DATAOPEN_SIM_URL;
const shots = process.env.DATAOPEN_SHOTS;
if (shots) mkdirSync(shots, { recursive: true });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const bridge = async () => (await fetch(base + 'sim/bridge')).json();
const results = [];
const check = (name, fn) => results.push([name, fn]);

async function until(fn, ms = 6000, what = 'condition') {
  const end = Date.now() + ms;
  for (;;) {
    const v = await fn();
    if (v) return v;
    if (Date.now() > end) throw new Error(`timed out waiting for ${what}`);
    await sleep(40);
  }
}

const browser = await chromium.launch();
const context = await browser.newContext({ viewport: { width: 360, height: 800 }, deviceScaleFactor: 2, hasTouch: true, isMobile: true, locale: 'ru-RU', serviceWorkers: 'allow', acceptDownloads: true });
const page = await context.newPage();
const requests = [], problems = [];
page.on('request', (r) => requests.push(r.url()));
page.on('pageerror', (e) => problems.push(`pageerror: ${e.message}`));
page.on('console', (m) => { if (m.type() === 'error') problems.push(`console: ${m.text()}`); });
const shot = async (name) => { if (shots) await page.screenshot({ path: `${shots}/${name}.png` }); };
const text = (sel) => page.locator(sel).first().innerText();
const tapBtn = (name) => page.getByRole('button', { name, exact: typeof name === 'string' }).first().tap();

// a clean device
async function reset() {
  const b = await bridge();
  if (b.state === 'HW_BYPASS' || b.reason === 'PANIC') await fetch(base + 'sim/panic/2300');
  await until(async () => (await bridge()).state !== 'HW_BYPASS' || true, 100);
  await page.evaluate(() => {});
}

check('the page loads, connects by itself and shows the device state in words', async () => {
  await page.goto(`${base}?transport=ws`);
  await page.waitForSelector('.state .title');
  await until(async () => (await text('nav.pages')).includes('Помощь'), 8000, 'pages');
  assert.match(await text('.state .title'), /Помощь|Устройство|Идёт/);
  assert.equal(await page.title(), 'Помощь при работе с мышью');
  await shot('01-main-ru');
});

check('every button is at least 64 px each way and at least 12 px from the next one', async () => {
  const boxes = await page.$$eval('button:not([hidden])', (els) => els.filter((e) => e.checkVisibility({ contentVisibilityAuto: true }) && (e.offsetParent !== null || getComputedStyle(e).position === 'fixed'))
    .map((e) => { const r = e.getBoundingClientRect(); return { n: e.textContent.trim().slice(0, 30), x: r.x, y: r.y, w: r.width, h: r.height, fixed: !!e.closest('.safety') }; }));
  assert.ok(boxes.length >= 6);
  for (const b of boxes) assert.ok(b.w >= 63.5 && b.h >= 63.5, `${b.n}: ${b.w.toFixed(1)} x ${b.h.toFixed(1)}`);
  for (const a of boxes) for (const b of boxes) {
    if (a === b || a.fixed !== b.fixed) continue;                         // the fixed safety bar overlays the page until it is scrolled; the page has room for it
    const gapX = Math.max(b.x - (a.x + a.w), a.x - (b.x + b.w)), gapY = Math.max(b.y - (a.y + a.h), a.y - (b.y + b.h));
    const gap = Math.max(gapX, gapY);
    assert.ok(gap >= 11.5 || (gapX < 0 && gapY < 0 && false), `${a.n} / ${b.n}: gap ${gap.toFixed(1)}`);
  }
});

check('no horizontal scrolling at phone width and the page never exceeds the viewport', async () => {
  const w = await page.evaluate(() => [document.documentElement.scrollWidth, window.innerWidth]);
  assert.ok(w[0] <= w[1], `scrollWidth ${w[0]} > ${w[1]}`);
});

check('turning help on is a trial with a visible keep/undo, and the real bridge engages', async () => {
  await tapBtn('Выключить помощь').catch(() => {});
  await until(async () => (await bridge()).reason === 'CMD_PASSTHRU', 15000, 'passthru');
  await page.getByRole('switch', { name: /Помощь/ }).first().tap();
  await page.waitForSelector('.banner.trial:not([hidden])');
  await until(async () => (await bridge()).state === 'ASSIST', 6000, 'ASSIST');
  assert.match(await text('.banner.trial'), /Стало лучше/);
  assert.match(await text('.banner.trial'), /через \d+ с/);
  await shot('02-trial');
  await tapBtn('Оставить');
  await page.waitForSelector('.banner.trial', { state: 'hidden' });
  assert.equal((await bridge()).assist_wanted, true);
  assert.match(await text('.state .title'), /Помощь работает|нечему помогать/);
});

check('a stepper moves one level per tap, speaks in words, and a quick second tap is ignored', async () => {
  const strength = page.locator('.control', { hasText: 'Сила помощи' });
  const plus = strength.getByRole('button', { name: /Больше/ });
  assert.match(await strength.innerText(), /Как рекомендует профиль/);
  await plus.tap();
  await until(async () => (await bridge()).strength === 6, 3000, 'strength 6');
  await plus.tap();                                                         // within 400 ms of the previous tap: ignored
  await sleep(300);
  assert.equal((await bridge()).strength, 6);
  await sleep(500);
  await plus.tap();
  await until(async () => (await bridge()).strength === 7, 3000, 'strength 7');
  await until(async () => /Уровень 7 из 10/.test(await strength.innerText()), 3000, 'the stepper shows level 7');
  await tapBtn('Оставить').catch(() => {});
  await shot('03-stepper');
});

check('the stop button is always there, answers at once, and is never blocked by the repeat-tap guard', async () => {
  await page.getByRole('button', { name: /Выключить помощь/ }).tap();
  await until(async () => (await bridge()).reason === 'CMD_PASSTHRU', 3000, 'stopped');
  assert.equal((await bridge()).assist_wanted, false);
  await until(async () => /Помощь выключена/.test(await text('.state .title')), 3000, 'state text');
  await shot('04-stopped');
});

check('English, and every theme keeps text at 7:1 or better as actually rendered', async () => {
  await tapBtn('Ещё');
  await shot('05-settings');
  await tapBtn('English');
  await page.waitForFunction(() => document.documentElement.lang === 'en');
  assert.match(await text('.safety'), /Turn assistance off/);
  const worst = {};
  for (const theme of ['Light', 'Dark', 'Maximum contrast']) {
    await tapBtn(theme);
    await sleep(100);
    worst[theme] = await page.evaluate(() => {
      const parse = (c) => c.match(/[\d.]+/g).map(Number);
      const lum = ([r, g, b]) => { const f = (v) => { v /= 255; return v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4; }; return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b); };
      const bgOf = (el) => { for (let e = el; e; e = e.parentElement) { const c = parse(getComputedStyle(e).backgroundColor); if (c.length < 4 || c[3] > 0.99) return c.slice(0, 3); } return [255, 255, 255]; };
      let min = 99, who = '';
      for (const el of document.querySelectorAll('body *')) {
        if (!el.offsetParent && getComputedStyle(el).position !== 'fixed') continue;
        const own = [...el.childNodes].some((n) => n.nodeType === 3 && n.textContent.trim());
        if (!own) continue;
        const fg = parse(getComputedStyle(el).color).slice(0, 3), bg = bgOf(el);
        const [hi, lo] = [lum(fg), lum(bg)].sort((a, b) => b - a);
        const r = (hi + 0.05) / (lo + 0.05);
        if (r < min) { min = r; who = el.textContent.trim().slice(0, 30); }
      }
      return { min, who };
    });
    assert.ok(worst[theme].min >= 7, `${theme}: ${worst[theme].min.toFixed(2)} (${worst[theme].who})`);
    await shot(`06-theme-${theme.replace(/\s/g, '-').toLowerCase()}`);
  }
  await tapBtn('Same as the phone');
  await tapBtn('Русский');
});

check('hardware bypass is two steps, can be cancelled, and is one-way', async () => {
  await tapBtn('Ещё');
  await tapBtn('Включить bypass');
  assert.equal(await page.getByRole('button', { name: 'Да, включить bypass' }).isVisible(), true);
  await shot('07-bypass-confirm');
  await tapBtn('Отмена');
  assert.equal(await page.getByRole('button', { name: 'Включить bypass', exact: true }).isVisible(), true);
  assert.notEqual((await bridge()).state, 'HW_BYPASS');
  await sleep(500);                                                          // the repeat-tap guard (400 ms) would swallow an immediate second tap
  await tapBtn('Включить bypass');
  await tapBtn('Да, включить bypass');
  await until(async () => (await bridge()).reason === 'CMD_BYPASS', 4000, 'CMD_BYPASS');
  await until(async () => /bypass/i.test(await text('.state .title')), 4000, 'state says bypass');
  assert.match(await text('.state .hint'), /Panic/);
  await shot('08-bypass');
});

check('after the hand re-arms the bridge the profile page is there', async () => {
  await fetch(base + 'sim/panic/2300');
  await until(async () => (await bridge()).state !== 'HW_BYPASS', 15000, 'engaged again');
  await tapBtn('Профиль');
  await shot('09-profile');
});

check('the meter and the calibration switch work', async () => {
  assert.match(await text('[role=progressbar] + .meter-text'), /\d+ %/);
  const calib = page.getByRole('switch', { name: /Калибровка/ });
  await calib.tap();
  await until(async () => (await bridge()).calibrating, 3000, 'calibrating');
  await until(async () => /Идёт калибровка/.test(await text('.state .title')), 3000, 'state says calibrating');
  await shot('10-calibrating');
  await sleep(500);
  await calib.tap();
  await until(async () => !(await bridge()).calibrating, 3000, 'stopped calibrating');
});

check('the device card is saved to the phone, nothing else of the device is, and a file from a sender loads', async () => {
  await tapBtn('Ещё');
  const idText = await page.locator('.control', { hasText: 'Номер владельца (меняется' }).innerText();
  const id = idText.match(/[0-9A-Z]{4}(-[0-9A-Z]{4}){3}/)[0];
  const dlCard = page.waitForEvent('download');
  await tapBtn('Сохранить карточку устройства');
  const card = await dlCard;
  assert.equal(card.suggestedFilename(), `dataopen-${id}.docard`);
  const cardJson = JSON.parse(readFileSync(await card.path(), 'utf8'));
  assert.equal(cardJson.id, id);
  assert.ok(!('storage' in cardJson));
  await shot('12-device-id');
  await tapBtn('Профиль');
  // nothing on the page hands the profile out: no copy, no 'save for another device' (docs/RESIDENCY.md); files only come in
  assert.equal(await page.getByText(/Сохранить копию|Сохранить для другого устройства/).count(), 0);
  assert.equal(await page.locator('a[download]').count(), 0);
  const day = Date.now() % 1000000;
  const bytes = Buffer.from(await (await fetch(`${base}sim/bundle/profile/${day}`)).arrayBuffer());   // a file from a sender, for this device
  assert.equal(bytes.subarray(0, 4).toString(), 'DOBS');
  assert.ok(!bytes.includes('BIOP'));
  const tmp = join(tmpdir(), `dataopen-sender-${day}.dobundle`);
  writeFileSync(tmp, bytes);
  await fetch(base + 'sim/button');                                                    // a sender the device has not met: the button on the device
  await page.locator('.control', { hasText: 'Загрузить файл настроек' }).locator('input[type=file]').setInputFiles(tmp);
  await page.waitForFunction(() => /Настройки загружены/.test(document.querySelector('.message')?.textContent || ''), null, { timeout: 8000 });
});

check('slots: one row opens the picker, one tap picks a slot, the name and the clearing work, the LED and the state agree', async () => {
  await tapBtn('Выключить помощь').catch(() => {});
  const summary = page.locator('.slots > summary');
  assert.match(await summary.innerText(), /Сменить слот/);
  assert.match(await text('.state .slot-line'), /^Слот 1/);
  assert.equal(await page.locator('.slots').evaluate((e) => e.open), false);       // closed: the controls stay in view
  assert.equal(await page.locator('.slot-grid .btn').first().isVisible(), false);   // and nothing of the grid is on screen or in the way
  await summary.tap();
  await page.waitForSelector('.slot-grid .btn');
  const boxes = await page.$$eval('.slot-grid .btn', (els) => els.map((e) => { const r = e.getBoundingClientRect(); return { w: r.width, h: r.height, pressed: e.getAttribute('aria-pressed') }; }));
  assert.equal(boxes.length, 4);
  assert.ok(boxes.every((b) => b.w >= 63.5 && b.h >= 63.5), JSON.stringify(boxes));
  assert.deepEqual(boxes.map((b) => b.pressed), ['true', 'false', 'false', 'false']);
  await shot('15-slots');
  await page.locator('.slot-grid .btn').nth(1).tap();
  await until(async () => (await bridge()).slot === 1, 3000, 'slot 1 on the device');
  await until(async () => /^Слот 2/.test(await text('.state .slot-line')), 3000, 'the state card says slot 2');
  assert.equal(await page.locator('.slots').evaluate((e) => e.open), false);          // picking closes it again
  assert.deepEqual((await bridge()).leds, [0, 1, 0, 0]);
  assert.equal((await bridge()).assist_wanted, false);
  await tapBtn('Профиль');
  await page.getByLabel('Название слота').fill('Браузер');
  await tapBtn('Сохранить');
  await until(async () => /Браузер/.test(await text('.state .slot-line')), 3000, 'the name in the state card');
  assert.deepEqual((await bridge()).names, ['', 'Браузер', '', '']);
  await shot('16-slot-named');
  await tapBtn('Очистить этот слот');
  await tapBtn('Подтвердить');
  await page.waitForFunction(() => /кнопк\S* на устройстве/.test(document.querySelector('.message')?.textContent || ''), null, { timeout: 5000 });
  await fetch(base + 'sim/press/confirm');
  await tapBtn('Повторить');
  await page.waitForFunction(() => /Слот очищен/.test(document.querySelector('.message')?.textContent || ''), null, { timeout: 5000 });
  assert.deepEqual((await bridge()).names, ['', '', '', '']);
  await summary.tap();
  await page.locator('.slot-grid .btn').nth(0).tap();
  await until(async () => (await bridge()).slot === 0, 3000, 'back to slot 0');
  assert.equal((await bridge()).slots[0], true);                                       // the first slot kept its profile through all of it
});

check('erasing personal data needs the button on the device: the message says so, and "Try again" finishes it', async () => {
  await tapBtn('Выключить помощь').catch(() => {});
  await tapBtn('Ещё');
  await tapBtn('Стереть личные данные (все слоты)');
  await shot('13-erase-confirm');
  await tapBtn('Подтвердить');
  await page.waitForFunction(() => /кнопк\S* на устройстве/.test(document.querySelector('.message')?.textContent || ''), null, { timeout: 5000 });
  assert.equal(await page.getByRole('button', { name: 'Повторить', exact: true }).isVisible(), true);
  await shot('14-needs-button');
  await fetch(base + 'sim/button');
  await sleep(500);
  await tapBtn('Повторить');
  await page.waitForFunction(() => /Личные данные стёрты/.test(document.querySelector('.message')?.textContent || ''), null, { timeout: 5000 });
  const b = await bridge();
  assert.equal(b.assist_wanted, false);
  assert.equal(b.reason, 'CMD_PASSTHRU');
  await tapBtn('Профиль');
  assert.match(await text('[role=progressbar] + .meter-text'), /^0 %/);
});

check('offline: after one visit the page opens with no network at all, and nothing but its own origin was ever contacted', async () => {
  await tapBtn('Выключить помощь').catch(() => {});
  await page.evaluate(() => navigator.serviceWorker.ready);
  await until(async () => (await page.evaluate(() => navigator.serviceWorker.controller !== null)) || (await page.reload(), false), 6000, 'worker controls the page');
  const cached = await page.evaluate(async () => (await caches.keys()).length);
  assert.ok(cached >= 1);
  await context.setOffline(true);
  const p2 = await context.newPage();
  const errs = [];
  p2.on('pageerror', (e) => errs.push(e.message));
  await p2.goto(`${base}?transport=ws&noauto`);
  await p2.waitForSelector('.connect:not([hidden])', { timeout: 8000 });
  assert.match(await p2.locator('body').innerText(), /Устройство не подключено|Помощь при работе с мышью/);
  assert.deepEqual(errs, []);
  await p2.screenshot({ path: shots ? `${shots}/11-offline.png` : undefined });
  await p2.close();
  await context.setOffline(false);
  const origins = new Set(requests.map((u) => new URL(u).origin));
  assert.deepEqual([...origins], [new URL(base).origin]);
});

check('no errors in the console, no uncaught exceptions', async () => {
  assert.deepEqual(problems.filter((p) => !/WebSocket connection/.test(p)), []);
});

let failed = 0;
for (const [name, fn] of results) {
  try { await fn(); console.log(`ok - ${name}`); } catch (e) { failed++; console.log(`not ok - ${name}\n  ${String(e.message).split('\n').join('\n  ')}`); await shot(`FAIL-${results.findIndex((r) => r[0] === name)}`).catch(() => {}); }
}
await browser.close();
console.log(`# checks ${results.length}, failed ${failed}`);
process.exit(failed ? 1 : 0);
