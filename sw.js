var CACHE_NAME = 'smartattend-v9';
var PRECACHE = [
  './',
  './index.html',
  './manifest.json',
  './icon-192x192.png',
  './logo.png'
];

self.addEventListener('install', function(e) {
  e.waitUntil(
    caches.open(CACHE_NAME).then(function(cache) {
      return cache.addAll(PRECACHE);
    })
  );
  self.skipWaiting();
});

self.addEventListener('activate', function(e) {
  e.waitUntil(
    caches.keys().then(function(names) {
      return Promise.all(
        names.filter(function(n) { return n !== CACHE_NAME; })
             .map(function(n) { return caches.delete(n); })
      );
    })
  );
  self.clients.claim();
});

self.addEventListener('fetch', function(e) {
  var url = new URL(e.request.url);
  if (url.origin !== location.origin) return;
  if (e.request.method !== 'GET') return;

  var isHTML = e.request.mode === 'navigate' ||
    (e.request.headers.get('accept') || '').indexOf('text/html') !== -1;

  if (isHTML) {
    // Network-first for pages so code changes show up on the next launch.
    e.respondWith(
      fetch(e.request).then(function(res) {
        var clone = res.clone();
        caches.open(CACHE_NAME).then(function(cache) { cache.put(e.request, clone); });
        return res;
      }).catch(function() {
        return caches.match(e.request);
      })
    );
  } else {
    // Cache-first for static assets. These are content-stable, so going to the
    // network for them on every launch just stalls startup; revalidate in the
    // background instead and pick the update up next time.
    e.respondWith(
      caches.match(e.request).then(function(cached) {
        if (cached) {
          fetch(e.request).then(function(res) {
            if (res && res.ok) {
              caches.open(CACHE_NAME).then(function(cache) { cache.put(e.request, res); });
            }
          }).catch(function() {});
          return cached;
        }
        return fetch(e.request).then(function(res) {
          var clone = res.clone();
          caches.open(CACHE_NAME).then(function(cache) { cache.put(e.request, clone); });
          return res;
        });
      })
    );
  }
});
