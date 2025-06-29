const CACHE_NAME = "analiz-cache-v1";
const urlsToCache = [
  "/",
  "/static/manifest.json",
  "/static/sw.js",
  "/static/icon.png"
];

self.addEventListener("install", event => {
  event.waitUntil(
    caches.open(CACHE_NAME).then(cache => {
      return cache.addAll(urlsToCache);
    })
  );
});

self.addEventListener("fetch", event => {
  event.respondWith(
    caches.match(event.request).then(response => {
      return response || fetch(event.request);
    })
  );
});