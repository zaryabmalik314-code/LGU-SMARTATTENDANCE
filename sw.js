// Retired service worker - deliberately does nothing but remove itself.
//
// Earlier versions cached static assets. On devices that had one installed the
// app could come up blank and stay that way until the user cleared storage,
// because the resident worker kept serving its own stale cache. Deleting this
// file outright would not have helped: browsers keep running the last worker
// they installed, so the fix has to be a worker that tears itself down.
//
// There is intentionally NO fetch handler. Without one, requests bypass the
// worker entirely and go straight to the network, so nothing can be served
// stale even in the window before the unregister below completes.
//
// Do not add caching logic here. The app is a TWA shell over a live site and
// has no offline requirement.

self.addEventListener('install', function() {
  self.skipWaiting();
});

self.addEventListener('activate', function(e) {
  e.waitUntil(
    caches.keys()
      .then(function(keys) {
        return Promise.all(keys.map(function(k) { return caches.delete(k); }));
      })
      .then(function() {
        return self.registration.unregister();
      })
      .then(function() {
        // Reload open clients so they come back under no worker at all.
        return self.clients.matchAll({ type: 'window' });
      })
      .then(function(clients) {
        clients.forEach(function(c) {
          if ('navigate' in c) { c.navigate(c.url); }
        });
      })
      .catch(function() {})
  );
});
