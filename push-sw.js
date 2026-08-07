/* SmartAttend push service worker.
   Deliberately minimal: it ONLY handles Web Push. No fetch handler, no caching,
   so the app keeps its "always live, no offline" behaviour — this worker exists
   purely so notifications can arrive while the app is closed. */

self.addEventListener('install', function (event) {
  self.skipWaiting();
});

self.addEventListener('activate', function (event) {
  event.waitUntil(self.clients.claim());
});

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
