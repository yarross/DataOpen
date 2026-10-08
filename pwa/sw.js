// Offline-first: every file of the app is cached on the first visit and served from the cache afterwards. A new version is fetched in
// the background but only takes over when the person presses "Update" (the page then reloads once). Nothing is ever sent anywhere.
const VERSION = '95e90ac9a3da';
const FILES = [
  './',
  'app.css',
  'icons/icon-192.png',
  'icons/icon-512.png',
  'icons/icon-maskable-512.png',
  'icons/icon.svg',
  'index.html',
  'js/app.js',
  'js/build.js',
  'js/constants.js',
  'js/dom.js',
  'js/i18n.js',
  'js/link.js',
  'js/session.js',
  'js/shell.js',
  'js/store.js',
  'js/transport/ble.js',
  'js/transport/ws.js',
  'js/view.js',
  'manifest.webmanifest',
];
const CACHE = `dataopen-${VERSION}`;

self.addEventListener('install', (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(FILES)));
});

self.addEventListener('message', (e) => {
  if (e.data === 'skipWaiting') self.skipWaiting();
});

self.addEventListener('activate', (e) => {
  e.waitUntil(caches.keys().then((keys) => Promise.all(keys.filter((k) => k.startsWith('dataopen-') && k !== CACHE).map((k) => caches.delete(k)))).then(() => self.clients.claim()));
});

self.addEventListener('fetch', (e) => {
  const req = e.request;
  if (req.method !== 'GET' || new URL(req.url).origin !== self.location.origin) return;
  e.respondWith(caches.match(req, { ignoreSearch: true }).then((hit) => hit || fetch(req)));
});
