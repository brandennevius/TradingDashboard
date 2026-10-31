# Verification and release

Define expected trader behavior before changing code. Choose the lowest test layer that can establish it, then use a small browser suite for the boundaries that require a browser, server, and saved state. A passing job covers its assertions under its configuration; a closed ticket or green build does not independently close every acceptance criterion.

## Test layers

| Layer | Current examples | What the evidence establishes |
| --- | --- | --- |
| Unit/domain | `tests/cam-broker-csv-import.test.ts`, `cam-paper-import.test.mjs`, `cf-statement-import.test.ts`, `trade-merge.test.ts`, `trade-excursion.test.ts`, snapshot tests | Parsing, grouping, duplicate identities, quantity/fee calculations, and session rules for supplied examples |
| Local storage integration | `tests/trade-merge-store.test.ts`, `ibkr-flex.test.ts`, `setup-versioning.test.ts`, `trade-market-context.test.ts` | Domain changes survive real temporary JSON reads/writes; not PostgreSQL transaction behavior |
| Direct route invocation | `tests/daily-portfolio-snapshot.test.ts`, `month-to-date-snapshot.test.ts` include handler calls | Request/response validation for invoked functions; no browser or running HTTP server |
| Mocked external service | `tests/alpaca-market-data.test.ts`, `market-data.test.ts`, `trade-review-export.test.ts`; injected snapshot email transports | Parsing, fallbacks, request shapes, error handling, and mocked response lifecycle; not live entitlements, provider uptime, AI quality, or email delivery |
| Static/extracted UI | `tests/market-context-ui.test.ts` renders static React markup; `cam-analytics.test.mjs`, `cam-support.test.mjs`, `cam-paper-import.test.mjs` inspect source or execute extracted functions in a VM | Specific rendering/handler logic; not a complete browser interaction |
| Browser + HTTP + local persistence | `e2e/broker-import.spec.ts` | Actual CSV upload, render, edit, debounce-save, authenticated API, JSON records, reload, and recovery without browser cache |
| Manual exploratory / staging | Checklist below; record results separately | Only the environment, scenarios, and observations actually recorded |

The Node test launcher also clears inherited `DATABASE_URL`, `LOCAL_DATA_DIR`, provider, and SMTP settings. Existing storage tests create their own temporary directories; a development data override must not redirect those writes into a real journal.

Files can contain more than one layer. See [audit](audit.md) for executed results; the table is not a coverage percentage. The optional IBKR private-file case skips unless local file paths are supplied. All other Node checks are intended to run without production credentials.

## Broker-import acceptance example

All browser rows refer to `e2e/broker-import.spec.ts`. The synthetic fixture has **two executions**, not two trades: buy +10 at $100 with $1 fee, sell -10 at $103 with $1 fee. Independent expected net P/L is `10 × (103 − 100) − 1 − 1 = $28`.

| Acceptance criterion | Layer / test | Expected result | Evidence from execution |
| --- | --- | --- | --- |
| A supported statement creates the expected trade | Browser: `synthetic import persists, reimport preserves executions and manual review` | UI shows one closed XYZ long, $28; API object has size 10, entry 100, exit 103, two distinct execution identities and quantities +10/-10 | Playwright assertions on visible row, API response, and raw execution contents |
| Manual review survives repeat import | Same browser test | Stop 98 and note persist; second upload reports zero added/updated and one ignored | Exact deep comparison of all saved trade fields, including IDs and executions; disk JSON comparison |
| Saved work survives loss of browser cache | Same browser test | Reload and separately authenticated fresh context recover the trade, note, and stop | Real server GET and rendered dialog in a browser context with empty localStorage |
| Entirely unrecognized input is actionable and non-destructive | Browser: `unrecognized CSV explains supported input and leaves saved records unchanged` | Names supported statement format; API state and disk bytes remain unchanged; reload retains the trade | Status-text assertions, state equality, byte equality, reload |
| Anonymous users cannot access shared state; another trader cannot write Cam state | Browser-suite HTTP case: `journal API rejects anonymous access and another trader's writes` | GET/PUT return 401 without session; Branden PUT returns 403; saved state unchanged | Real HTTP statuses and before/after state comparison; this case is an API check in the Playwright suite |
| More complicated executions conserve quantities/cash and duplicate identity | Fast: `tests/cam-paper-import.test.mjs`, `cam-broker-csv-import.test.ts` | Reversals split correctly, identical fills remain distinct, repeats retain annotations, unsupported rows produce diagnostics | Node assertions; existing baseline coverage, not new browser claims |

The invalid fixture tests an unrecognized format, not every malformed or partially valid statement. Parser tests cover additional diagnostics; partial imports may legitimately import valid rows with warnings. The browser suite does not certify every broker format.

## Isolation and failure investigation

`npm run test:e2e` runs `scripts/run-browser-tests.mjs`. It creates a unique OS temporary directory, random local trader passwords and session secret, blanks database/provider/SMTP/mentor/cron credentials even if the caller had them set, and launches Playwright. The controlled Next.js development server inherits that environment. The suite rejects reuse of a server on port 3100. An occupied port fails safely.

Each test gets a fresh browser and resets the Cam journal through its real authenticated API on this disposable server. The application has a shared Cam identity, so one worker is intentional. Tests authenticate using the real login HTTP route; **the login form itself is not browser-covered**. No import parser, merge function, journal API, or persistence is mocked. The selected path needs no market/AI/email calls. Cleanup removes the temporary directory after success, failure, or handled termination; a hard kill may leave an OS temporary directory for later cleanup.

Playwright uses accessible labels/roles scoped to the relevant area, existing table/summary IDs where necessary, response waits for saves, and `aria-busy` for boot completion. There are no fixed sleeps. Node and browser tests live in separate directories and use separate commands. `test-results/` and `playwright-report/` are ignored.

For failures:

```bash
npm run test:e2e -- --grep 'synthetic import'
npm run test:e2e:report
npx playwright show-trace test-results/<failed-case>/trace.zip
```

Retained traces and screenshots contain synthetic test data and temporary session credentials; CI retains failure artifacts for seven days. Use the action log for TypeScript/Node/build failures. Do not attach real account statements or credentials to issues.

## Pull request checks

`.github/workflows/verification.yml` runs on all pull requests, pushes to `main`, and manual dispatch. It uses Node **24.19.0**, `npm ci`, commit-pinned Actions, Ubuntu 24.04, read-only repository permissions, bounded job timeouts, and cancellation of superseded runs. The independent jobs are:

- **TypeScript** — `npm run typecheck`.
- **Node regressions and build** — `npm test`, then `npm run build`.
- **Playwright Chromium** — install locked browser/dependencies, then `npm run test:e2e`; upload HTML report, screenshots, and traces on failure.

After the first successful hosted run, a repository admin can select those three exact check names in branch protection/rulesets and require review. Repository rules were not changed by this work. No selective execution or automatic production deployment is included. Pin updates should keep `.nvmrc`, the package engine, lockfile, and hosting runtime compatible. Reassess selective execution only after measuring pipeline duration.

## Feature to release

1. **Problem → criteria.** Example: repeated broker uploads must preserve manual review and avoid duplicate executions. State quantities, expected cash result, persistence, and error behavior explicitly.
2. **Implementation → tests.** Inspect parser/merge behavior, add fast cases for accounting edges, then a bounded browser assertion for any affected UI/storage boundary. Document unsupported formats.
3. **Pull request → evidence.** Include the problem, final behavior, criteria-to-test mapping, executed commands/results, skipped checks, and limitations. Do not treat mock output as live integration proof.
4. **Automated regression → review.** Require the three checks after configuring repository rules. Review financial calculations, account isolation, migration/deployment impacts, and the actual assertion contents.
5. **Staging validation.** On an isolated staging database, validate relevant PostgreSQL behavior, deployed secure-cookie login, import/reimport, annotations, and a new-browser reload. Use synthetic data; check console/server errors. Live providers/email need separately authorized controlled checks. Record environment, commit, steps, observations, evidence, and unresolved issues.
6. **Deployment.** An owner uses the existing Vercel integration after reviewing production branch/root settings and backup readiness. This workflow does not deploy. Never assume a preview has isolated data; verify its database and credentials first.
7. **Safe smoke check.** After deployment, confirm the deployed commit, page load, login with an approved account, and a read-only view of a known record. Confirm applicable logs/cron health. Run write/import smoke scenarios against staging or a designated synthetic account only. If checks fail, investigate and use the existing owner-approved rollback path; reconcile any database changes separately.

Manual results must identify the criterion actually checked. Neither staging nor production smoke was performed during the local implementation audit unless explicitly recorded in [audit.md](audit.md).

## Known gaps

PostgreSQL locking/rollback and migrations, concurrent saves, backup restore, production security, full login-form flow, screenshots/uploads, accessibility audit, mobile/cross-browser behavior, performance/load, live provider compatibility, paid AI output, real SMTP delivery, hosted cron execution, and most Branden workflows are outside the new browser suite. No WebSocket feature was found in this workflow, so none was added for demonstration. The repository's performance phase plan is not measured performance evidence.
