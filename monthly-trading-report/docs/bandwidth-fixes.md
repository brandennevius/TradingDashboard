# Private screenshot and trade-list transfer

Authenticated trade lists use body-derived weak ETags, `Cache-Control: private, no-cache`, and `Vary: Cookie`. Authentication always precedes conditional handling. Matching requests return 304; database reads and serialization still occur. This does not introduce shared CDN caching.

With existing `SCREENSHOT_STORAGE_WRITES=enabled`, inline image strings submitted by older clients are decoded without conversion. Immutable R2 uploads are read back and verified before screenshot records and persistent trade references commit together. Existing images are never resized or converted during extraction. Failed database transactions may leave unreferenced R2 objects; this workflow does not delete them.

The new-trade form keeps files in memory with object-URL previews, saves the trade, and uploads each image separately. Stable request/upload keys make retries idempotent. An ambiguous create retry returns the already-created trade rather than overwriting its attachments. New binary uploads retain the pre-existing upload encoder; existing inline originals bypass that encoder.

## Existing inline images

First verify a recoverable current database backup and current R2 access. Use a verified TLS connection, session pooler, and a private pgpass file. The tool does not initialize application tables.

```sh
node --import tsx scripts/extract-inline-trade-screenshots.ts --dry-run
```

Apply requires explicit `--apply`, `INLINE_MIGRATION_BACKUP_VERIFIED=yes`, exact `CLEANUP_EXPECTED_HOST` and `CLEANUP_EXPECTED_PROJECT_REF`, and `SCREENSHOT_STORAGE_WRITES=enabled`. Configure those through your existing private environment. Each trade uses a compare-and-swap on its original screenshot array, with lock and statement timeouts. A concurrent edit stops that update. Reruns skip already-extracted images.

Deploy the compatibility code before applying extraction so stale tabs cannot reintroduce persisted inline images. Re-run `scripts/audit-trade-payload.sql` afterward and verify each migrated object's length and hash.

## Direct image views

`SCREENSHOT_DIRECT_SERVING=enabled` enables authorized image/document views to redirect to a private R2 GET valid for 60 seconds. Metadata is loaded and authorized before signing; Vercel does not download the image bytes. Raster MIME types are allowlisted and signed response overrides preserve content type, disposition and private/no-store caching.

Browser `fetch()` requests and backups keep the verified same-origin proxy, avoiding a new R2 CORS dependency. `?proxy=1` explicitly requests that path. Legacy rows and unsupported formats also use the proxy. Watchlist aliases retain their entity-type restriction, trade routes retain their owner/trade checks, and shared screenshot routes retain their existing authenticated access.

Signing verifies metadata, but does not rehash every image download. Upload and migration verification remain in place. Anyone holding a signed URL can use it until expiry, including after session revocation. The final R2 response does not add `X-Content-Type-Options: nosniff`; direct serving is restricted to raster types and explicit signed MIME overrides. Set the flag blank and redeploy to return all views to proxy serving. Do not revert to code that persists inline strings after migration.

## Validation and measurement

```sh
npm test
npm run typecheck
npm run build
npm run test:e2e -- private-cache.spec.ts
BACKUP_TEST_DATABASE_URL=postgresql://localhost:55439/trading_dashboard_backup_test_bandwidth npm run test:e2e -- --screenshot-routes screenshot-serving.spec.ts
```

Screenshot route tests create/drop only a uniquely named schema in a guarded disposable local database. They use fake R2 signing credentials and intercept browser image requests; they do not access a real bucket. The root page's pre-existing add-trade modal currently has no opening handler; the upload implementation does not add a new entry point.

Measure client-facing encoded bytes and Vercel origin in/out usage separately. `scripts/bandwidth-audit.mjs --payload response.json` reports aggregate current/projected compression sizes without modifying the source. `--model scenario.json` leaves missing residual traffic explicitly incomplete. Collect representative usage from both journals, including backups and uploads, before making a hosting-plan decision.
