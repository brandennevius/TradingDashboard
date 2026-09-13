# Monthly Trading Report

A small shared dashboard for Branden and Cam to log monthly trading KPIs, review trends, and edit past monthly entries.

## What it includes

- Login for two traders, configured through `TRADER_USERS`
- Monthly KPI form that asks for raw inputs and calculates derived metrics
- Saved month-to-month history
- Dedicated edit tab for updating or deleting your own submitted months
- Filterable dashboard visuals for Branden, Cam, or both across monthly P&L, return, win rate, and R metrics
- Postgres persistence for deployment
- Local JSON persistence when `DATABASE_URL` is not set

## Local setup

```bash
cd monthly-trading-report
npm install
cp .env.example .env.local
npm run dev
```

Open `http://localhost:3000`.

Default local demo logins are:

- `branden` / `password`
- `cam` / `password`

Change these before sharing the app.

## Environment variables

```bash
APP_SECRET="a-long-random-secret"
TRADER_USERS="branden:password,cam:password"
DATABASE_URL="postgres://..."
FMP_API_KEY="..."
```

Trade-detail MAE/MFE uses execution-aware intraday bars from FMP. Optional configuration:

```bash
FMP_TRADE_EXCURSION_INTERVAL="5min"
FMP_US500_FUTURES_SYMBOL="ESUSD"
FMP_US100_FUTURES_SYMBOL="NQUSD"
FMP_US30_FUTURES_SYMBOL="YMUSD"
```

Listed securities and spot FX use their compatible provider symbols. The `.US500`, `.US100`, and `.US30` broker CFDs use scaled ES, NQ, and YM futures proxies and are explicitly labeled estimates because CFD spread and futures basis can differ. Missing or incompatible bars remain unavailable rather than being recorded as zero.

If `DATABASE_URL` is empty, local submissions are stored in `data/monthly-reports.json`.

### Read-only mentor login

Set the server-only `CODEX_JOURNAL_PASSWORD` environment variable to enable the
`codex` login for Branden's journal. Add it as a sensitive Production variable in
Vercel and redeploy. Keep the value out of source control and client-side variables.
The account uses the existing read-only permissions; trader credentials in
`TRADER_USERS` remain unchanged. If that list already contains `codex`, the mentor
configuration takes precedence and makes that identity read-only. Removing the
variable disables the dedicated login; also remove any separately configured
`codex` entry from `TRADER_USERS` when revoking all access for that identity.

## Deployment

This is ready for Vercel:

1. Create a Postgres database through Vercel Postgres, Neon, Supabase, or another hosted Postgres provider.
2. Add `APP_SECRET`, `TRADER_USERS`, and `DATABASE_URL` in Vercel project environment variables.
3. Deploy this `monthly-trading-report` folder.

The app creates the `monthly_reports` table automatically on first read/write.

## Entered fields

- Account size
- Net P&L
- Total trades
- Win rate
- Avg winning R
- Avg losing R
- Average risk
- Current risk percent
- Average trade length
- Notes

## Calculated fields

- Percent return
- Total return
- Total R
- Avg R
- Average win
- Average loss
- Expected value in R
- Return stability score

### AI trade review Word export

The Trade Log's **AI Review .docx** button reviews all currently filtered trades in display order. Every included trade must satisfy the shared Complete rules: nonzero risk, manual or calculated grade, assigned setup, all five required reflection fields, and at least one screenshot. The server rechecks these rules and returns each incomplete trade's missing fields before calling OpenAI.

The export uses the Responses API with `gpt-5.6-luna`, medium reasoning, structured output, background processing, and `store: false`. The page polls the background response until it is ready, so a long review is not tied to one server request. OpenAI temporarily retains a background response while it is available for polling. Set `OPENAI_API_KEY` on the server; the separate `OPENAI_REVIEW_MODEL` setting still belongs to the watchlist review and does not override this export. Model access must be enabled for the API project.

All six structured reflections (General Review is optional), legacy notes, mistake tags, executions, stored checklist assessments, assigned setup requirements, and full active strategy sources and examples are included. Readable trade charts are analyzed at high detail; reusable model-example charts use low detail to control vision tokens. Every chart is embedded in the Word file with its aspect ratio preserved. Unreadable images stop the export with a re-upload message. Chart links remain references and are not fetched. Disabled strategy sources/examples are excluded.

MAE/MFE are loaded or calculated using the existing excursion engine and its cache. They retain the full lifecycle/as-of scope, provider, proxy status and unavailable reasons. Period P&L/R uses in-range exits when available, as in the trade log. Unavailable excursion data is never presented as zero, and the review is instructed not to equate MFE with achievable profit.

**What to Work On** ranks one to three evidence-backed priorities: the period's primary process gap, supporting trade IDs and observations, likely effect on outcomes, an explicit rule going forward, a measurable adherence target/review horizon, and confidence. A repeated mistake needs at least two distinct trades; small samples and one-offs must be identified. The report does not promise hypothetical profits or claim improvement against an unseen prior period.

Evidence is not silently truncated. Oversized requests fail with instructions to narrow the filters or reduce active example charts/sources. Repeated comparison charts are loaded once and sent to the model once with every associated trade ID, oversized images are normalized to a readable review resolution, and image/excursion preparation runs with bounded concurrency. Before generation, the Responses input-token endpoint counts the exact request. Current Luna pricing and the full 16,000-token output allowance must remain at or below the $0.25 safety ceiling, leaving margin below the $0.30 report target; an over-budget report is blocked before the paid generation request. Other safeguards are 1,000,000 characters of text context, 450 unique image parts, and a request below 45 MiB; image decoding is limited to 20 MiB/40 megapixels per input. Each submit, status, and finalization request has its own bounded deadline, while the model continues independently between polls. A refusal, incomplete response, invalid trade references, or missing work priorities fails the export instead of producing a partial report.

Run export regression coverage with `node --import tsx --test tests/trade-review.test.ts tests/trade-review-export.test.ts tests/trade-log-csv.test.ts`. Set `REVIEW_QA_PATH` to write the synthetic Word fixture for layout checks. These tests mock the model and do not establish live API access or model quality.
