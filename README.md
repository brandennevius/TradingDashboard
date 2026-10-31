# TradingDashboard

A shared trading journal for reviewing executions, risk, setups, and trading decisions. Traders can import supported broker statements, annotate trades, compare performance, and export review material rather than maintain disconnected spreadsheets.

The active application is **[monthly-trading-report/](monthly-trading-report/)**. The directory name reflects its original monthly-report feature; the app now includes separate Branden and Cam journals, trade logs, portfolios, watchlists, setup checklists, market context, and snapshots.

- **[Start here: visual project guide](monthly-trading-report/docs/guide/README.md):** a high-level explanation of the stack, data flow, tests, environments, and delivery process, with diagrams.
- **[Application setup and reference](monthly-trading-report/README.md):** verified stack, synthetic demo, configuration, integrations, and deployment.
- **[Verification and release](monthly-trading-report/docs/verification.md):** test layers, acceptance criteria, CI checks, evidence, and gaps.
- **[Interview walkthrough](monthly-trading-report/docs/interview-guide.md):** explanation, reading path, investigation example, and exercise.
- **[Audit record](monthly-trading-report/docs/audit.md):** baseline and validation results, with explicit limits.

```bash
cd monthly-trading-report
nvm install && nvm use          # Node 24.19.0; or install this version directly
npm ci
cp .env.example .env.local     # synthetic local settings only
npm run dev
```

Open [localhost:3000](http://localhost:3000), log in as `cam` / `password`, and open Cam's journal. Import `e2e/fixtures/synthetic-schwab.csv` using **Import Broker CSV**. It represents a fabricated ten-share round trip with $28 net profit. Use a fresh local data directory as described in the application guide.

From the application directory:

```bash
npm run typecheck
npm test
npm run build
npx playwright install --with-deps chromium
npm run test:e2e
```

The browser command starts its own isolated local app and cleans up its synthetic storage. It does not use a running development server or production credentials. The GitHub Actions workflow repeats these checks on pull requests and updates to `main`; it does not deploy.

AI assisted this project and these improvements. Tests demonstrate specific behavior under documented conditions; they do not establish live provider compatibility or production readiness. Historical screenshots, reference material, and standalone analysis scripts remain in the repository; the retired Python/Streamlit dashboard is not the application entry point.
