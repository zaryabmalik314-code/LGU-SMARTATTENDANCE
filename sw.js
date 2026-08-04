var CACHE_NAME = 'smartattend-v6';
var PRECACHE = [
  './',
  './index.html',
  './signup.html',
  './icon-192x192.png',
  './icon-512x512.png',
  './logo.png'
];

self.addEventListener('install', function(e) {
  e.waitUntil(
    caches.open(CACHE_NAME).then(function(cache) {
      return cache.addAll(PRECACHE.map(function(url) {
        return new Request(url, {cache: 'no-store'});
      }));
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

  e.respondWith(
    fetch(e.request, {cache: 'no-store'}).then(function(resp) {
      var clone = resp.clone();
      caches.open(CACHE_NAME).then(function(cache) {
        cache.put(e.request, clone);
      });
      return resp;
    }).catch(function() {
      return caches.match(e.request);
    })
  );
});
