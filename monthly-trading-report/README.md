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
```

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
