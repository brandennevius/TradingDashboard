# Trading journal application

This Next.js application helps traders connect broker executions with their own risk decisions and reviews. Branden's React journal includes trade imports, portfolios, open-position risk, watchlists, setup checklists, market context, and exports. Cam's journal uses a separate browser JavaScript interface served by the same application, with broker CSV imports, manual annotations, setup analytics, and shared saved state. Monthly KPI reporting remains available.

Start with the **[visual project guide](docs/guide/README.md)** for a high-level explanation with diagrams. This page is the detailed setup and configuration reference. Deeper material: [verification/release](docs/verification.md), [interview guide](docs/interview-guide.md), [audit/evidence](docs/audit.md). The [repository README](../README.md) is the project entry point.

## Stack and data flow

| Component | Purpose |
| --- | --- |
| Node.js 24.19.0, Next.js 15.5.19 | Local/server runtime, App Router pages and API handlers; versions verified from lockfile/runtime |
| React 19, TypeScript 5 | Branden journal and monthly-report UI; shared types and server/domain logic |
| HTML/CSS/browser JavaScript | Cam journal in `public/cam-journal/`; shares authenticated Next.js APIs |
| PostgreSQL (`pg`) | Persistent deployed records, settings, screenshots, and Cam revisions |
| Local JSON | Non-production fallback under `data/` or `LOCAL_DATA_DIR`; no database required for demo |
| Recharts, lightweight-charts | Dashboard and market charts |
| CSV, XML, PDF/document libraries | Supported broker imports and review/export documents |
| `node:test`, `node:assert/strict`, `tsx` | Fast domain, local storage, mocked-service, and static/extracted-UI regression checks |
| Playwright (TypeScript) | Bounded real-browser import and persistence suite |
| GitHub Actions | Independent TypeScript, regression/build, and Chromium checks |

```text
Branden React UI ────────> Next.js API routes ──> lib/ parsers, calculations, store
Cam HTML/JS + CSV parser ─> /api/cam-journal ────> lib/store.ts
                                                   ├─ development: JSON files
                                                   └─ deployment: PostgreSQL
Optional market/AI/email features ──> server-side provider calls
```

Cam parses CSV and groups executions in the browser, merges imported records while retaining review fields, caches locally, then debounces a full-state PUT to the API. Reloading retrieves shared state. This is whole-state persistence, with no optimistic concurrency control; simultaneous writers can overwrite each other's changes. Local JSON is intended for a single local instance, not multi-user deployment.

Authentication uses configured trader passwords and an HMAC-signed, HTTP-only, same-site session cookie (`lib/auth.ts`). Production cookies are secure. Cam's write route requires the `cam` identity; other authenticated users can read it. This is a small shared-journal permission model. A built-in `tim` / `tim` read-only delegate to Branden exists even in production; an optional `codex` delegate is enabled by `CODEX_JOURNAL_PASSWORD`. The `/api/public/branden-open-positions` route intentionally exposes a subset of positions without requiring login. Review these policies, password handling, session lifetime, rate limiting, and the development secret fallback before wider use. This work does not certify or redesign authentication.

## Synthetic local setup

```bash
nvm install && nvm use
npm ci
cp .env.example .env.local
# Choose a fresh absolute path; do not reuse a directory with real journal records.
export LOCAL_DATA_DIR="$(mktemp -d /tmp/trading-demo-XXXXXX)"
npm run dev
```

Leave `DATABASE_URL` and provider/email credentials empty. Open [localhost:3000](http://localhost:3000), choose **Cam**, log in with `password`, and select Cam's journal. Import `e2e/fixtures/synthetic-schwab.csv` from the sidebar. Expected: one closed long `XYZ` trade, ten shares, entry $100, exit $103, and $28 P/L after $2 fees. Add a stop and notes in Trade Log, save, reload, and import again. The record should retain its annotations and report one ignored duplicate. `e2e/fixtures/invalid.csv` should explain supported inputs and leave saved records unchanged.

`branden` / `password` is also available with the example configuration. These credentials and the example secret are for an isolated local demo. Leave the data directory in place if you want to retain the demo; it is not automatically removed by `npm run dev`. The automated browser launcher creates and removes its own directory.

Local records use `monthly-reports.json`, `trade-logs.json`, and `app-settings.json`; Cam state is inside `app-settings.json`. `LOCAL_DATA_DIR` is ignored in production. Production writes require `DATABASE_URL`; `next start` without a database is not a supported JSON demo. Screenshot persistence has database-specific paths and is outside this demo.

## Commands

Run from this directory after `npm ci`:

| Command | Purpose |
| --- | --- |
| `npm run dev` | Development server on port 3000 |
| `npm run typecheck` | Strict TypeScript checking without incremental output |
| `npm test` | All 27 Node test files, explicitly under `tests/`, with inherited database/data-directory/provider settings cleared |
| `npm run test:cam` | Focused Cam support/analytics/CSV/paper regression group |
| `npm run test:auth` | Authentication configuration unit checks |
| `npm run test:cf-import` / `npm run test:ibkr-import` | Broker parser/reconciliation and local-store groups |
| `npm run test:daily-snapshot` / `npm run test:mtd-snapshot` | Snapshot calculations/export and injected-email checks |
| `npm run test:market-gauge` / `npm run test:market-context` | Market context/calculation/static-render group |
| `npx playwright install --with-deps chromium` | Install the browser matching the locked Playwright version |
| `npm run test:e2e` | Start disposable app on 127.0.0.1:3100 and run `e2e/` only |
| `npm run test:e2e -- --grep 'unrecognized CSV'` | Focus one browser case while investigating |
| `npm run test:e2e:report` | View latest HTML browser report |
| `npm run build` / `npm start` | Production build / production server (writes need PostgreSQL) |
| `npm run snapshot:daily -- --session YYYY-MM-DD` | Existing snapshot CLI (needs appropriate data/configuration; append `--send-email` only for intended delivery) |

Node and Playwright discovery are deliberately separate. All Node test commands use `scripts/run-node-tests.mjs`, which clears inherited database, local-data override, provider, and SMTP settings so storage tests can use their own temporary directories. The old `next lint` script was removed: it had no committed lint configuration and did not provide a dependable standalone check. No separate ESLint gate is claimed. Browser tests use one worker because Cam's journal state is shared, reset that disposable state before each case, and do not depend on test order. To use an already-installed Chromium in a restricted environment: `PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH=/usr/bin/chromium npm run test:e2e`; CI installs Playwright's pinned browser. Do not point the suite at a deployed server.

## Environment variables

All application configuration below is server-side; no secret belongs in a `NEXT_PUBLIC_` variable. Copy the committed `.env.example`, which contains only local placeholders.

| Name | Purpose / default |
| --- | --- |
| `APP_SECRET` | Signs sessions; set a strong unique secret in deployment; code otherwise uses a development fallback |
| `TRADER_USERS` | Comma-separated `id:password` pairs; development defaults exist; production trader list must be configured |
| `DATABASE_URL` | PostgreSQL connection; absent selects local JSON only outside production |
| `LOCAL_DATA_DIR` | Absolute development/test storage override; otherwise `<cwd>/data` |
| `CODEX_JOURNAL_PASSWORD` | Enables optional read-only Branden mentor login; empty disables dedicated login |
| `ALPACA_API_KEY`, `ALPACA_API_SECRET` | Alpaca historical market data; both required for that provider |
| `TRADE_EXCURSION_INTERVAL` | MAE/MFE bar resolution, default `5min` |
| `OPENAI_API_KEY` | Enables optional paid AI review features |
| `OPENAI_REVIEW_MODEL` | Watchlist-review model override; defaults to `gpt-4.1-mini`; separate from trade-review export |
| `CRON_SECRET` | Bearer authorization for scheduled market archiving and Cam backups |
| `SNAPSHOT_SMTP_HOST`, `SNAPSHOT_SMTP_PORT`, `SNAPSHOT_SMTP_SECURE` | Optional SMTP endpoint, numeric port, and literal `true` for TLS |
| `SNAPSHOT_SMTP_USERNAME`, `SNAPSHOT_SMTP_PASSWORD` | Optional SMTP authentication |
| `SNAPSHOT_EMAIL_FROM`, `SNAPSHOT_EMAIL_TO` | Optional snapshot sender/recipient; incomplete SMTP configuration disables sending |

`NODE_ENV` is managed by Next.js. Browser tooling uses generated `E2E_DATA_DIR` and `E2E_PASSWORD`; do not configure them manually. `PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH` is an optional local test override. `IBKR_HISTORY_FILE` and `IBKR_DAILY_FILE` enable an existing private-file regression that is otherwise skipped; never upload private statements as CI artifacts. `REVIEW_QA_PATH` optionally writes the existing generated document QA sample during Node tests.

## Integrations and limits

- **Broker imports:** Branden supports Capital.com/CF statement processing and IBKR Flex XML; Cam supports Thinkorswim/Schwab cash-ledger CSV and paperMoney stock Filled Orders. Formats, account boundaries, duplicate identity, and unsupported instruments matter. Partial valid imports can include warnings; an entirely unrecognized CSV is a no-op. The new browser suite covers only the synthetic Cam cash-ledger example. Other parsing/accounting edges stay in faster tests. No broker trading API is used by this workflow.
- **Market data:** Alpaca historical SIP bars, with Stooq/Yahoo fallbacks for daily consumers; other intervals retain Yahoo paths. MAE/MFE uses regular US sessions, raw bars, execution-aware quantities/FIFO trims/fees, and a delay margin for recent data. Results are bar approximations and exclude overnight risk. `.US500`, `.US100`, and `.US30` use labelled scaled ETF estimates; unsupported FX/intraday instruments remain unavailable. Provider availability, entitlements, price adjustment, and exact-session freshness affect results. See [market-context notes](docs/trade-market-context.md).
- **AI:** OpenAI watchlist review and trade-review Word exports are optional. Trade export currently selects `gpt-5.6-luna` in code, validates completeness, estimates a cost ceiling, and uses background Responses polling. Access to that configured model is unverified here. With AI enabled, trade reflections and screenshots can leave the application; review consent and provider retention before using real material. Mocks do not prove live model availability or quality.
- **Email:** Nodemailer sends optional daily/MTD snapshot attachments only with complete configuration. Tests inject a transport; no real mail delivery is verified.

## Delivery and deployment

Work on a feature branch, define observable acceptance criteria, implement with relevant tests, and open a pull request. Run the commands above locally; GitHub Actions repeats checks before review. Follow the [verification/release guide](docs/verification.md) for traceability, manual validation, and smoke checks. Performance plans in this repository remain plans unless supported by measurements.

The existing deployment setup targets **Vercel + PostgreSQL**. Set the Vercel root directory to `monthly-trading-report`, framework to Next.js, Node to 24.x, install command to `npm ci`, and build command to `npm run build`. Configure `APP_SECRET`, `TRADER_USERS`, and `DATABASE_URL` separately for the intended environment; optional integrations require their own configuration. Tables are created/extended by store helpers on first access, so the deployment database role needs appropriate schema permissions. Review changes against a disposable/staging database before production.

`vercel.json` retains two UTC schedules: market-context archive at 22:30 Monday–Friday and Cam backup at 05:15 daily. Set `CRON_SECRET` for scheduled access. Confirm scheduling support, backups, restoration, and database permissions in your hosting plan. JSON files on a deployment filesystem do not replace durable PostgreSQL storage.

A linked Vercel Next.js project named `monthly-trading-report` with Node 24.x was observed during this audit. The available metadata did not expose its root directory, Git production branch, or preview data isolation. Confirm those in Vercel before relying on preview validation; use a separate preview database and credentials. No deployment settings or branch-protection rules were changed, and no production deployment is added by this workflow. The PR branch triggered a successful existing Vercel branch build (recorded in the audit); deployed data paths were not smoke-tested. Existing Git-provider deployment behavior remains external to these checks.
