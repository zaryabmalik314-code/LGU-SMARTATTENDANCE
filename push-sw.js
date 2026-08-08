/* SmartAttend service worker: app-shell caching + Web Push.

   Caching exists to kill the WebAPK splash -> first-paint white flash and to
   make cold launches instant: the shell is served from the Cache Storage the
   moment Chrome hands off, with no network round-trip in the gap.

   This is the SAFE version of the caching that was retired earlier. The old
   worker was cache-ONLY with no revalidation, so one bad cached copy stranded
   the app on a blank screen until storage was cleared by hand. This one is
   stale-while-revalidate and self-healing:
     - every GET is revalidated against the network in the background, so a bad
       entry is overwritten on the very next load and can never persist,
     - only 200 same-origin responses are stored (never API calls or errors),
     - activate() deletes every cache except the current version,
     - bump SHELL_CACHE to force a clean sweep on the next deploy.
   Because of skipWaiting + clients.claim, shipping a new push-sw.js (even one
   with the fetch handler removed) recovers every client within one launch —
   that is the kill switch if caching ever needs to come back out. */

var SHELL_CACHE = 'smartattend-shell-v1';

/* Only the handful of files needed to paint the first screen. Anything else is
   cached on demand by the fetch handler. Each is added independently so a
   single 404 can't abort the whole precache. */
var PRECACHE = ['./', './index.html', './signup.html', './manifest.json', './icon-192x192.png'];

self.addEventListener('install', function (event) {
  self.skipWaiting();
  event.waitUntil(
    caches.open(SHELL_CACHE).then(function (cache) {
      return Promise.all(PRECACHE.map(function (u) {
        return cache.add(u).catch(function () {});
      }));
    })
  );
});

self.addEventListener('activate', function (event) {
  event.waitUntil(
    caches.keys().then(function (keys) {
      return Promise.all(keys.map(function (k) {
        if (k !== SHELL_CACHE) { return caches.delete(k); }
      }));
    }).then(function () { return self.clients.claim(); })
  );
});

self.addEventListener('fetch', function (event) {
  var req = event.request;
  if (req.method !== 'GET') { return; }              // never touch POST etc.
  var url = new URL(req.url);
  if (url.origin !== self.location.origin) { return; } // API, fonts, CDNs -> straight to network

  event.respondWith(
    caches.open(SHELL_CACHE).then(function (cache) {
      return cache.match(req).then(function (cached) {
        // Kick off a background refresh regardless of a cache hit.
        var network = fetch(req).then(function (res) {
          if (res && res.status === 200 && res.type === 'basic') {
            cache.put(req, res.clone());
          }
          return res;
        }).catch(function () { return cached; }); // offline -> whatever we have
        // Cache-first for an instant paint; the network copy refreshes the
        // cache for next time (stale-while-revalidate).
        return cached || network;
      });
    })
  );
});

/* ===================== Web Push (unchanged behaviour) ===================== */

self.addEventListener('push', function (event) {
  var payload = { title: 'SmartAttend', body: '', url: './index.html', tag: undefined, icon: undefined };
  if (event.data) {
    try {
      var data = event.data.json();
      payload.title = data.title || payload.title;
      payload.body = data.body || '';
      payload.url = data.url || payload.url;
      payload.tag = data.tag;
      payload.icon = data.icon;
    } catch (e) {
      payload.body = event.data.text();
    }
  }
  event.waitUntil(
    self.registration.showNotification(payload.title, {
      body: payload.body,
      tag: payload.tag,
      icon: payload.icon || './icon-192x192.png',
      badge: './icon-192x192.png',
      data: { url: payload.url },
      renotify: !!payload.tag
    })
  );
});

self.addEventListener('notificationclick', function (event) {
  event.notification.close();
  var targetUrl = (event.notification.data && event.notification.data.url) || './index.html';
  event.waitUntil(
    self.clients.matchAll({ type: 'window', includeUncontrolled: true }).then(function (clientList) {
      for (var i = 0; i < clientList.length; i++) {
        var client = clientList[i];
        if ('focus' in client) {
          client.focus();
          if ('navigate' in client) {
            try { client.navigate(targetUrl); } catch (e) {}
          }
          return;
        }
      }
      if (self.clients.openWindow) {
        return self.clients.openWindow(targetUrl);
      }
    })
  );
});
