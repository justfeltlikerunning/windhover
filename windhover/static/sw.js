/* Windhover service worker — Web Push alerts. v3 */
self.addEventListener('install', (e) => self.skipWaiting());
self.addEventListener('activate', (e) => e.waitUntil(self.clients.claim()));

const PENDING = 'wh-pending-nav';

self.addEventListener('push', (event) => {
  let d = {};
  try { d = event.data ? event.data.json() : {}; } catch (_) { d = { body: event.data && event.data.text() }; }
  const title = d.title || 'Windhover';
  const options = {
    body: d.body || '',
    tag: d.tag || undefined,
    data: { url: d.url || '/' },
    icon: '/static/icon-192.png',
    badge: '/static/icon-192.png',
  };
  event.waitUntil(self.registration.showNotification(title, options));
});

self.addEventListener('notificationclick', (event) => {
  event.notification.close();
  const url = (event.notification.data && event.notification.data.url) || '/';
  event.waitUntil((async () => {
    // remember the target — iOS launches a closed PWA at start_url and drops
    // the fragment, so the page asks for this on boot (get-pending-nav)
    try {
      const c = await caches.open(PENDING);
      await c.put('/pending-nav', new Response(url, { headers: { 'x-ts': String(Date.now()) } }));
    } catch (_) {}
    const list = await self.clients.matchAll({ type: 'window', includeUncontrolled: true });
    if (list.length) {
      const c = list[0];
      try { await c.focus(); } catch (_) {}
      try { c.postMessage({ type: 'open-url', url }); } catch (_) {}
      return;
    }
    if (self.clients.openWindow) return self.clients.openWindow(url);
  })());
});

self.addEventListener('message', (event) => {
  const d = event.data || {};
  if (d.type !== 'get-pending-nav') return;
  event.waitUntil((async () => {
    try {
      const c = await caches.open(PENDING);
      const hit = await c.match('/pending-nav');
      if (!hit) return;
      const url = await hit.text();
      const ts = Number(hit.headers.get('x-ts') || 0);
      await c.delete('/pending-nav');
      // stale clicks (>2 min old) shouldn't yank the user around on a later launch
      if (url && Date.now() - ts < 120000 && event.source) {
        event.source.postMessage({ type: 'open-url', url });
      }
    } catch (_) {}
  })());
});
