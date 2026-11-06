# Supabase image migration and bounded journal history

This release prepares TradingDashboard for the free-plan database cleanup. It does
not migrate production automatically. Preserve original screenshot bytes and all
screenshot rows. Do not delete by filename, entity ownership, or latest-state
membership: the shared screenshot table also backs setup examples and watchlists.

## Deployment prerequisites

1. Confirm the intended project is `oomzkupuddhseewfakyn`, including the production
   Postgres connection's project identity. Read current usage/restrictions and
   available Storage capacity/egress in Supabase; historical estimates are not
   sufficient. Provision operator access through secret settings and approved
   network configuration, never through committed files.
2. Take a custom-format full `pg_dump`, save a checksummed encrypted second copy,
   and restore into an isolated database. Compare current settings, trades,
   revision IDs, image row counts, lengths and SHA-256 hashes. Keep a final backup
   while writes/jobs are paused before clearing or pruning.
3. Apply `migrations/20261105-screenshot-storage.sql` before deploying this code.
   It adds metadata and allows null image bytes, with a backing-source constraint.
   Existing bytes are retained. The script is transactional, retryable, and fails
   on lock contention after five seconds.
4. Create the private `trade-screenshots` bucket only if current capacity and
   transfer allowance can accommodate measured original bytes plus a growth
   reserve. Set these **server-only production** variables:

   - `SUPABASE_STORAGE_URL=https://oomzkupuddhseewfakyn.supabase.co`
   - `SUPABASE_STORAGE_SERVICE_KEY`: server `service_role` key via secret settings
   - `SUPABASE_SCREENSHOT_BUCKET=trade-screenshots`

   Keep `SCREENSHOT_STORAGE_WRITES` unset for the
   compatibility deployment. Preview must have its own isolated DB, bucket, and
   credentials; do not point it at production. Existing sensitive Vercel secrets
   cannot be assumed retrievable through an environment pull.
5. Deploy the compatibility release. It can read bytea or Storage-backed images.
   Check existing screenshots and streamed export before changing write behavior.
6. Set `SCREENSHOT_STORAGE_WRITES=enabled`, redeploy, and check a new image,
   authorization and backup export. Every upload verifies the bucket is private,
   uploads to a content-addressed immutable path, then downloads and hashes it
   before committing a DB reference. Failed DB commits can leave unreferenced
   objects; do not automatically delete them during this cleanup.

## Operator commands

Use Node 24 with installed dependencies, running from `monthly-trading-report`.
Standalone scripts do not load `.env.local`. Supply variables through the
environment or Node's `--env-file` pointing at a private, untracked file. Never
place secrets in the commands or repository.

Required operator configuration:

- `DATABASE_URL`: direct/session connection, not transaction pooler, with a valid
  TLS trust chain; the operator tool verifies certificates.
- `CLEANUP_EXPECTED_HOST`: exact connection hostname.
- `CLEANUP_EXPECTED_PROJECT_REF=oomzkupuddhseewfakyn`: checks the direct hostname
  or pooler username, since multiple Supabase projects share pooler hosts and
  database names. The Storage URL must match this project too.
- Storage variables above.

Start read-only:

```sh
node --import tsx scripts/supabase-cleanup.ts audit
```

Mutation commands additionally require `CLEANUP_WRITES=confirmed` and
`CLEANUP_BACKUP_ATTESTATION` pointing to a private JSON file. Create it only after
a successful restore rehearsal. It records `host`, `database`, `projectRef`,
`dumpFile` (absolute local path), `sha256` (dump checksum), `restoreVerifiedAt`
(timestamp), and `secondCopyVerified: true`. The tool checks the dump checksum
and target. This operator attestation does not independently prove a restore;
retain the rehearsal results with it.

```sh
node --import tsx scripts/supabase-cleanup.ts copy /secure/cleanup-manifest.json
node --import tsx scripts/supabase-cleanup.ts verify /secure/cleanup-manifest.json
```

`copy` handles one image at a time, keeps the database bytes, verifies object
downloads, and records a resumable private manifest. It covers every row in both
tables, including apparently orphaned images. It also records explicit candidate
revision IDs and a fingerprint of the authoritative live Cam settings. Stop on
any unexpected error; rerunning `copy` refreshes the manifest and verifies objects
again. Immutable object paths include the table, row ID and original byte hash.

Verify authenticated HTTP image reads as well as object hashes. Cover both users,
read-only viewers, denied users, setup examples and both watchlist/Cam URL aliases.
All image responses use private/no-store caching so a shared CDN cannot skip app
authorization. Confirm legacy image URLs and original image quality are preserved.
Verify backup download and isolated restore while images are Storage-backed.

Before destructive cleanup, pause writes/uploads/imports/deletions and relevant
cron jobs, drain old deployments, take the final verified backup, and rerun copy
and verify. Review the manifest's exact `revisionIds` and image coverage. Set
`CLEANUP_MAINTENANCE=writes-and-jobs-paused` only after those controls are in place.

```sh
node --import tsx scripts/supabase-cleanup.ts prune /secure/cleanup-manifest.json
node --import tsx scripts/supabase-cleanup.ts release /secure/cleanup-manifest.json
```

`prune` deletes only the listed routine `save` snapshots, keeps the newest
pre-save snapshot, and aborts transactionally if candidates or live-state
fingerprint changed. Nightly, recovery, migration and unknown reasons stay intact.
`release` first verifies full table coverage and every object without bytea
fallback, then clears bytes conditionally against matching source/metadata.
Each cleared row retains its ID and a verified object. A partial release is
resumable; keep all writes paused if an error occurs. This does not remove rows,
inline images, columns, Storage objects, trades or settings.

## Compaction and acceptance

DELETE/UPDATE and plain VACUUM normally make internal space reusable; they do not
reliably return enough filesystem space to clear the quota. Confirm with Supabase
that table rewrites are supported, privileges exist, and temporary disk/WAL
headroom is sufficient. Rehearse lock duration and space on the restored database.
If these conditions fail, stop for a supported maintenance path.

With writes/jobs still paused, set
`CLEANUP_COMPACTION=headroom-and-locks-verified` and run:

```sh
node --import tsx scripts/supabase-cleanup.ts compact /secure/cleanup-manifest.json
node --import tsx scripts/supabase-cleanup.ts audit
```

Compaction serially runs `VACUUM (FULL, ANALYZE)` on the three affected tables
outside a transaction, with a five-second lock timeout. It requires exclusive
locks and temporary space. Measure final bytes; require the database to be under
the actual current plan limit, preferably below 400 MiB, and verify the dashboard
restriction clears. Database and egress restrictions are separate.

Ordinary saves write live `cam_journal_state` atomically and retain the newest
pre-save snapshot. The daily job preserves all existing nightly, recovery,
migration and unknown revisions. Nightly retries within a UTC day do not create
duplicate snapshots. The job returns database bytes and emits a warning above
400 MiB. No outbound notifications are configured.

Confirm save/reload, trade counts, current-state fingerprint, image counts and
hashes, monthly reports, setup/watchlist views and export/restore. Resume users
and cron jobs, then observe a normal save and the next scheduled nightly job.
`vercel.json` schedules that job at 05:15 UTC daily. Record its execution result,
database size, bucket capacity and transfer usage. Establish recurring independent
backups of database **and object bytes**, since pg_dump alone no longer includes
external images. Choose a notification owner/destination separately if desired.

## Backup format and rollback

Branden exports remain version 1 with an optional `sharedScreenshots` section for
images referenced by versioned setup settings and weekly watchlists. Export reads
one original image at a time and fails if an object/reference is missing; it does
not silently produce an incomplete file. Existing version-1 files still import.
Those older files cannot recover shared-image bytes they never contained.
Imports preserve other journal owners and reject shared-ID content/owner conflicts.

Before byte release, retain legacy bytes and the compatibility release. After
Storage-only rows exist, never roll back to old code that assumes non-null bytea.
Stop writes/jobs, rehydrate from verified immutable objects if capacity permits,
or coordinate full database/app/object recovery using the verified backup and
reconcile every post-backup write. Do not blindly restore over a live database;
restoring the original oversized dump also restores the quota problem.

No JPEG conversion, orphan sweep, plan change, authentication change, unrelated
RLS change, or LEDGR project work is included. Inline trade/settings images remain
intact; they can be externalized later if measured usage justifies it.
