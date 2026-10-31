# Repository audit and execution evidence

Audit date: **2026-10-30, America/New_York**. Baseline: `0c15369` (`Support paperMoney stock statement imports in Cam Journal`). The workspace was clean before validation. No applicable `AGENTS.md` was found under `/workspace`. Findings below were established from this checkout and execution, not inherited from an earlier review.

## Baseline

- Active application: `monthly-trading-report/`; Next.js **15.5.19** from the lockfile, React **19**, TypeScript, Node **24.19.0** / npm **11.9.0** in the audit environment.
- Next.js App Router with API handlers, PostgreSQL via `pg`, development JSON fallback, and a separately served HTML/JavaScript Cam interface. Existing signed-session authentication and per-route permissions were inspected.
- Supported broker import code, market-data loaders, optional OpenAI review/export, snapshot email transport, and cron routes were present. No WebSocket feature was found in the selected import workflow.
- **27** test files were present under `tests/`. They mix domain tests, temporary local-store integrations, direct handler calls, mocked services, static React markup, source inspection, and VM-extracted JavaScript. File names do not establish browser coverage.
- No committed GitHub Actions workflow or Playwright browser suite was present. The lockfile's existing optional Next.js Playwright peer reference was not an implemented suite.
- Root README pointed at the correct app; application README still centered on monthly reports and referenced a missing `.env.example`. Existing targeted test commands did not provide one all-regression command or separate typecheck script. The `next lint` entry had no committed lint configuration.
- `vercel.json` defined market-context and Cam backup cron schedules. A linked Vercel Next.js project named `monthly-trading-report` using Node 24.x was observed. Its returned metadata did not expose Git production branch, root directory, or preview database isolation. Those settings were not changed.

Commands run **before application/test changes** from `monthly-trading-report/`:

| Command | Observed result |
| --- | --- |
| `node --import tsx --test tests/*.test.ts tests/*.test.mjs tests/*.test.js` | **229 total: 228 passed, 1 skipped, 0 failed**; about 17.8 seconds on this environment |
| `npx tsc --noEmit` | Exit 0, no diagnostics |
| `npm run build` | Exit 0; compiled, type checked, generated 60 static pages, and completed route/build output |

The skipped IBKR test requires private daily/history file paths (`IBKR_HISTORY_FILE`, `IBKR_DAILY_FILE`). The remaining IBKR tests use an embedded synthetic sample and ran. No pre-existing regression/type/build failure was found in these baseline commands. Performance duration is environment-specific test execution evidence, not an application performance claim.

## Changes introduced

- Three bounded Playwright cases in `e2e/broker-import.spec.ts`, using a fabricated two-fill Schwab-style fixture and unrecognized-input fixture. Two cases exercise browser workflows; the third checks real HTTP access controls within the Playwright suite.
- A launcher with temporary JSON storage, random configured credentials/session secret, cleared provider/database/email credentials, and cleanup; a dedicated non-reused server on port 3100. Authentication behavior was preserved.
- Minimal Cam semantics: an announced import status, a named trade dialog, and a busy state during initial journal load. Parsing, accounting, and upsert behavior were not rewritten.
- Explicit Node-versus-browser commands with inherited configuration cleared by the Node launcher, standalone typecheck, locked Playwright dependency, and Node runtime pin. Removed the unsupported standalone lint entry. Existing targeted regression commands remain.
- Three PR workflow jobs with lockfile installs, runtime pin, timeouts, read-only permissions, and retained browser failure artifacts. No deploy step or repository rule changes.
- Updated entry-point/setup/architecture/integration/deployment documentation, a criteria-to-test release guide, and an interview reading path/exercise that acknowledges AI assistance.

## Validation after changes

Run commands sequentially when Next.js is writing generated `.next/types` files. The CI jobs use independent clean runners.

| Command / inspection | Result |
| --- | --- |
| `npm ci --cache /workspace/.npm-cache` | Exit 0; 430 packages installed from the lockfile. Cache path is specific to this sandbox's writable directories. Existing Recharts 2.x deprecation warning remains. |
| `npm run typecheck` | Exit 0 after stopping the browser dev server |
| `npm test` | **229 total: 228 passed, 1 skipped, 0 failed**; see final launcher run below |
| `PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH=/usr/bin/chromium npm run test:e2e` | **3 passed**, about 12.4 seconds; real local HTTP and JSON persistence, fresh browser recovery, exact trade/execution equality, invalid-input state/byte equality, 401/403 checks |
| Local browser/runtime | Playwright **1.64.1**, installed system Chromium **154.0.8037.57** on Debian. CI installs the Chromium revision matching Playwright; that browser is a separate hosted validation target. |
| Temporary directory inspection after browser completion | No `trading-dashboard-e2e-*` directories remained under `/tmp` |
| `npm run build` | Exit 0; optimized production build completed with all 60 static pages |
| Workflow validation | Python YAML parse and `actionlint` 1.7.7 both passed; actionlint exit 0 with no diagnostics |
| `git diff --check` | No whitespace errors at inspection |

The final Node launcher was also run as `LOCAL_DATA_DIR=/tmp/trading-protected-journal DATABASE_URL=postgres://invalid.example/do-not-connect npm test`. All 228 runnable checks passed, one private-file case skipped, and sentinel settings/trade files in the inherited directory remained byte-for-byte unchanged. This verifies the new launcher prevents the inherited override from redirecting storage-test writes.

The selected browser path needed no external-service mocks: the real parser/merge, auth, journal API, store, and JSON disk writes ran locally. Optional providers and SMTP were disabled. Existing service tests used mocks/injected transports. No production data was mutated.

During test development, ambiguous accessible labels/text and a string-versus-number stop-display expectation failed and were corrected. A clean dependency installation interrupted an overlapping browser run; a typecheck overlapping Next dev regeneration reported missing generated files. Sequential final checks distinguish these test/command execution failures from application regressions. See the interview guide for the selector investigation. Failure screenshots/traces were generated during these failed runs; a later passing run replaces the local report.

## Hosted checks and deployment limits

PR and hosted Actions evidence will be recorded here after publication. Local passing commands do not establish hosted execution. The GitHub connector can access the repository; the shell `gh` token failed authentication, so publication may require the connected GitHub tools.

No staging or production smoke check, PostgreSQL transaction/concurrency/restore check, live provider/AI call, email delivery, or cron execution was performed. No production deployment was initiated. Existing Git-provider/Vercel branch deployment behavior and preview data isolation must be verified in hosting settings. See [known gaps](verification.md#known-gaps) before describing the scope of evidence.
