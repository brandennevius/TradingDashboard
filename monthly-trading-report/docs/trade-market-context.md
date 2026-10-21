# Trade market context

Completed-session gauges are archived independently of broker imports. Each trade joins its original entry and current/exit cutoff to the session archive. Re-importing a statement or adding a partial exit does not replace historical gauge records. Correcting a trade's execution dates intentionally changes the relevant session links.

## Timing and evidence

- Daily bars are eligible at 16:15 America/New_York. Half-day sessions wait until this conservative cutoff too.
- The exchange-session calendar comes from SPY daily bars, so weekends and market holidays do not create synthetic sessions.
- Entry and exit context use the preceding session unless an explicitly zoned execution timestamp proves the same day's finalized close was already available. Date-only and unzoned broker times never use that day's closing information.
- Open positions use the latest available completed session, even if they have partial exit dates. Closed positions keep their exit cutoff.
- Original archived snapshots are immutable. Missing records can be filled later; a calculation change requires a new gauge version.
- Historical backfills are labeled Reconstructed and use the fixed versioned leader universe. They do not establish which stocks were leaders at the time. Missing constituents remain unavailable rather than changing a historical score's denominator.

## Persistence and refresh

`market_context_history` stores the session date, gauge version, complete gauge JSON and refresh timestamp in Postgres. Null snapshots preserve a known session whose evidence is incomplete. The development fallback is `data/market-context-history.json` with serialized, atomic file writes.

Refresh runs after successful Branden CF imports, at 22:30 UTC weekdays via the authenticated cron, and on demand when history is first read or stale. Archival reads bypass the live gauge's cache to prevent freezing an intraday bar. A refresh failure does not roll back or change broker trades. Both the live gauge and archived history use the same scoring functions.

## Display and analysis

Trade-log rows show Entry → Current/At exit with dated tooltips and separate filters for entry, endpoint and transition. Existing filtered statistics operate on that subset. Detail shows six comparison rows. Daily history is collapsed, renders only when expanded and has a bounded scroll area; the API sends compact history points instead of every day's full gauge.

`GET /api/journal/branden/market-context` returns summary contexts; `?tradeId=...` returns a detail context. Access is limited to Branden and his journal mentors. Owner-only POST refreshes history; cron requires CRON_SECRET. Responses are private/no-store.

Run `npm run test:market-context`, CF import regression tests, TypeScript checks and `npm run build`. Timing, missing evidence, immutable replays, compact rendering and authorization should be verified before release.
