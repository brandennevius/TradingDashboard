# Campus Fund dashboard ↔ DailyMarketChartPipeline contract

Schema version: `campus-fund-market-review-v1`

This integration is bound to the existing Campus Fund dashboard context. It intentionally has no `account_scope` or `consumer` field. Every request is correlated by the explicit review run ID, completed session, attempt, and exact source hashes. Neither side may select a generic newest source or result.

## Configuration

Dashboard environment:

- `DATABASE_URL`: existing dashboard PostgreSQL database.
- `APP_SECRET` or `MARKET_REVIEW_TOKEN_SECRET`: signs short-lived source and callback tokens.
- `DASHBOARD_WORKER_SECRET`: reusable shared secret; the identical value is stored only as a GitHub Actions secret in `DailyMarketChartPipeline`.
- `MARKET_REVIEW_GITHUB_TOKEN`: fine-grained GitHub token with Actions write access for the pipeline repository.
- `MARKET_REVIEW_GITHUB_REPOSITORY`: defaults to `brandennevius/DailyMarketChartPipeline`.
- `MARKET_REVIEW_GITHUB_WORKFLOW`: defaults to the existing Campus-specific `daily-review.yml`; an explicit environment value still overrides it.
- `MARKET_REVIEW_GITHUB_REF`: defaults to `main`.
- `MARKET_REVIEW_BASE_URL`: canonical deployed dashboard origin.
- `BLOB_READ_WRITE_TOKEN`: private Vercel Blob store used only for temporary MarketSurge source PDFs.

The reusable worker secret and short-lived tokens must never be printed, stored in result metadata, or placed in `workflow_dispatch` inputs. The public workflow obtains short-lived credentials as follows:

1. GitHub injects `DASHBOARD_WORKER_SECRET` from Actions secrets.
2. The worker calls `worker_input_url` with `Authorization: Bearer $DASHBOARD_WORKER_SECRET` over HTTPS.
3. The dashboard returns signed source URLs and a short-lived callback token in the response body.
4. The workflow must mask the response and must not echo it or run with shell tracing.

## Dashboard direct upload and start request

The PDF bytes never pass through a Next.js request body. The authenticated browser first calls the Vercel Blob client-upload handler:

`POST /api/journal/branden/market-review/upload`

The handler issues a short-lived client token only for `application/pdf`, at most 20 MB, and an exact private pathname bound to the requested session, one upload UUID, and the browser-calculated SHA-256. The browser uploads directly to the private Blob store. Blob URLs are never sent to GitHub or exposed to the worker.

The browser then sends a small authenticated JSON request:

`POST /api/journal/branden/market-review`

```json
{
  "session_date": "YYYY-MM-DD",
  "marketsurge_pdf": {
    "upload_id": "uuid",
    "session_date": "YYYY-MM-DD",
    "filename": "MarketSurge.pdf",
    "content_type": "application/pdf",
    "size_bytes": 12345678,
    "sha256": "64 lowercase hex",
    "blob_url": "https://{store}.private.blob.vercel-storage.com/market-review/source/...",
    "blob_pathname": "market-review/source/...",
    "blob_content_type": "application/pdf"
  }
}
```

Before creating a run, the dashboard fetches the private object with server credentials and verifies its store URL/path, metadata size and MIME, `.pdf` extension, `%PDF-` magic, 1–75 pages, and SHA-256. It internally generates the existing Daily Portfolio Snapshot with `writeExports:false` and no email. The PDF stays in private Blob; its exact URL/path/hash metadata are frozen in PostgreSQL. Snapshot sources remain private PostgreSQL `bytea` rows:

- `marketsurge_pdf`
- `snapshot_json`
- `snapshot_markdown`

Legacy multipart create requests are rejected with `MARKETSURGE_DIRECT_UPLOAD_REQUIRED`, preventing the Vercel request-body 413 path.

## GitHub workflow dispatch

The dashboard calls:

`POST /repos/{repository}/actions/workflows/{workflow}/dispatches`

Only these non-secret `workflow_dispatch.inputs` are sent:

```json
{
  "review_run_id": "uuid",
  "session_date": "YYYY-MM-DD",
  "attempt": "1",
  "marketsurge_pdf_sha256": "64 lowercase hex",
  "snapshot_json_sha256": "64 lowercase hex",
  "snapshot_markdown_sha256": "64 lowercase hex",
  "market_gauge_json_sha256": "64 lowercase hex",
  "worker_input_url": "https://dashboard.example/api/journal/branden/market-review/runs/{run_id}/worker-input",
  "worker_callback_url": "https://dashboard.example/api/journal/branden/market-review/runs/{run_id}/worker-callback"
}
```

There is no token input. The fourth hash binds the exact-session Dashboard Market Gauge JSON. None of these values is a credential, and no generic newest artifact is selected.

## Worker input

`GET /api/journal/branden/market-review/runs/{run_id}/worker-input`

Header:

`Authorization: Bearer {DASHBOARD_WORKER_SECRET}`

The worker must also repeat the non-secret dispatch correlation values as headers:

- `X-Review-Attempt: {attempt}`
- `X-MarketSurge-PDF-SHA256: {marketsurge_pdf_sha256}`
- `X-Snapshot-JSON-SHA256: {snapshot_json_sha256}`
- `X-Snapshot-Markdown-SHA256: {snapshot_markdown_sha256}`
- `X-Market-Gauge-JSON-SHA256: {market_gauge_json_sha256}`

The dashboard rejects the request if the attempt is no longer current or any hash differs. This prevents a delayed job from an earlier attempt from obtaining the current attempt's tokens.

Response:

```json
{
  "schema_version": "campus-fund-market-review-worker-input-v1",
  "review_run_id": "uuid",
  "session_date": "YYYY-MM-DD",
  "attempt": 1,
  "source_hashes": {
    "marketsurge_pdf_sha256": "...",
    "snapshot_json_sha256": "...",
    "snapshot_markdown_sha256": "...",
    "market_gauge_json_sha256": "..."
  },
  "portfolio_snapshot": {
    "download_url": "absolute signed URL",
    "sha256": "...",
    "markdown_download_url": "absolute signed URL",
    "markdown_sha256": "..."
  },
  "marketsurge_pdf": {
    "download_url": "absolute signed URL",
    "sha256": "...",
    "page_count": 12,
    "filename": "MarketSurge.pdf"
  },
  "market_gauge": {
    "download_url": "absolute signed URL",
    "sha256": "..."
  },
  "ocr_corrections": null,
  "callback": {
    "url": "absolute callback URL",
    "schema_version": "campus-fund-market-review-callback-v1",
    "token": "short-lived signed token"
  }
}
```

On a correction retry, `ocr_corrections` contains `download_url`, `sha256`, and `version`. All download URLs are short-lived, exact-kind URLs signed over run ID, session, attempt, all four source hashes, source kind, and the downloaded byte hash. The worker must independently hash every downloaded body and reject any mismatch.

## Callback authentication and common envelope

`POST /api/journal/branden/market-review/runs/{run_id}/worker-callback`

Header:

`Authorization: Bearer {callback.token from worker input}`

The token is short-lived and signed over run ID, session, attempt, and all four source hashes. The callback body repeats those values. Any mismatch is rejected.

Common metadata:

```json
{
  "schema_version": "campus-fund-market-review-callback-v1",
  "event_id": "globally unique and stable for retry",
  "event_type": "RUNNING | OCR_REVIEW_REQUIRED | FAILED | RESULTS_REGISTERED | DELIVERY_STATUS",
  "review_run_id": "uuid",
  "session_date": "YYYY-MM-DD",
  "attempt": 1,
  "source_hashes": {
    "marketsurge_pdf_sha256": "...",
    "snapshot_json_sha256": "...",
    "snapshot_markdown_sha256": "...",
    "market_gauge_json_sha256": "..."
  },
  "github": {
    "repository": "brandennevius/DailyMarketChartPipeline",
    "workflow_run_id": "123456",
    "workflow_run_attempt": 1,
    "workflow_url": "https://github.com/.../actions/runs/123456"
  }
}
```

`event_id` is the idempotency key. Repeating an already accepted event returns `duplicate:true` without changing state.

### State callbacks

`RUNNING`, `OCR_REVIEW_REQUIRED`, `FAILED`, and `DELIVERY_STATUS` use `Content-Type: application/json`.

- `OCR_REVIEW_REQUIRED` adds `ocr: {status, version, items, message?}`.
- `FAILED` adds `error: {code, message, details?}`.
- `DELIVERY_STATUS` adds `delivery: {status, error?}` where status is `NOT_REQUESTED`, `PENDING`, `SENDING`, `SENT`, or `FAILED`.

Delivery updates never change a completed build back to queued/running. A delivery failure leaves the review `COMPLETED` with `delivery_status:FAILED`; it cannot trigger report rebuilding.

### Completed result callback

`RESULTS_REGISTERED` uses `multipart/form-data` with exactly these fields:

- `metadata`: JSON string containing the common envelope plus:
  - `event_type: "RESULTS_REGISTERED"`
  - `audit: {status:"PASS", packet_sha256, evidence?}`
  - optional `delivery: {status, error?}`
  - `artifacts`: three metadata objects with `kind`, `filename`, `media_type`, `sha256`, and `size_bytes`; kinds are exactly `pdf`, `markdown`, and `json`.
- `pdf`: generated report PDF bytes.
- `markdown`: generated Markdown bytes.
- `packet`: generated canonical JSON packet bytes.

The dashboard recalculates every artifact size and SHA-256, validates PDF magic and JSON parsing, and requires `audit.packet_sha256` to equal the registered JSON artifact hash. It registers all three in one database transaction. `audit.status` must be `PASS`. Only after that transaction succeeds does it delete the temporary MarketSurge and snapshot sources and set `source_deleted_at`.

## Attempt and retry semantics

- A new review starts at `attempt:1` with a new immutable `review_run_id`.
- Retry is allowed only from `FAILED` or `NEEDS_REVIEW` while the exact source packet exists and has not expired.
- Retry retains the same run ID and frozen hashes, increments `attempt`, clears prior workflow/error fields, and dispatches again.
- Tokens from prior attempts fail correlation.
- A completed run cannot be retried, including when only email delivery failed.
- No endpoint resolves “latest” source material for a worker. Browser history listing is display-only.

## Source retention and cleanup

- Successful strict result registration deletes temporary inputs immediately.
- Failed and needs-review inputs expire no later than 24 hours after run creation.
- Unclaimed direct uploads are removed after 24 hours by the cleanup operation.
- Authenticated manual cleanup is available at `POST /api/journal/branden/market-review/cleanup`; no schedule is created by this change.
- Result artifacts remain private and download only through authenticated dashboard endpoints.

## OCR corrections

Authenticated browser request:

`POST /api/journal/branden/market-review/runs/{run_id}/ocr-corrections`

```json
{"expected_version":1,"corrections":[]}
```

Corrections use optimistic versioning, are hashed and frozen as `ocr_corrections_json`, and are supplied only on an explicit subsequent retry.
