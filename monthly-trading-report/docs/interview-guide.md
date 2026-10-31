# Interview walkthrough

Use the actual code and [audit results](audit.md) as evidence. AI assisted implementation, tests, and documentation. Do not turn assistance or test results into claims of independent authorship, historical team practice, production-scale reliability, or familiarity with infrastructure this app does not use.

## A 60–90 second explanation

> I actively trade and wanted a journal that links broker executions to the decisions I make: entry, risk, setup, and review. TradingDashboard supports broker imports, trade annotations, portfolio risk, and performance review. It started with monthly reporting, which is why the Next.js application directory still has that name.
>
> The app uses TypeScript, React, and Next.js API routes on Node, with PostgreSQL for deployed storage and JSON for local development. Cam's journal is a separate browser JavaScript interface using the same server. Market data and AI review are optional integrations.
>
> I used AI assistance, and I want to be precise about what I verified. The project already had Node regression tests for parsing, calculations, imports, and data preservation. This improvement adds a small TypeScript Playwright suite that uploads a synthetic statement through the real browser, saves through the authenticated API, and checks the stored contents. It verifies that reimporting preserves notes and stops without duplicating executions. GitHub Actions repeats type checking, regressions, build, and browser checks.
>
> The work sample shows how I connect a trader's problem to acceptance criteria, investigate failures, and build repeatable evidence. Live providers, PostgreSQL concurrency, and production deployment still need their own validation.

## Follow one feature from requirement to evidence

**Problem:** Traders re-upload overlapping statements, but their manual trade review must survive. A row count alone can hide duplicated executions or replaced notes.

**Criteria:** A fabricated buy of ten shares at $100 followed by a sale at $103, with $1 fee on each side, must produce one closed long with $28 net P/L and two executions. After setting stop 98 and a note, repeat import must leave the complete saved trade unchanged. A fresh browser must recover the note from server storage. Unsupported input must explain the format and preserve saved state.

**Implementation:** `public/cam-journal/broker-csv-parser.js` exposes `statementExecutions`, `groupExecutionsIntoTrades`, and `upsertImportedTrades`. `importBrokerCsvFile` in `app.js` calls these in sequence, applies the merged result, renders it, and queues a save. `saveRemoteStateNow` sends a whole-state PUT. The Next.js route checks the session and Cam write permission; `saveCamJournalState` stores settings in JSON locally or PostgreSQL in production.

**Test choices:** Existing Node tests exercise detailed fills, reversals, fees, malformed rows, and duplicate identities cheaply. The browser case tests the wiring: file selection, rendered result, editing fields, save response, exact API/disk contents, reload, and a fresh browser. It uses the real login route, not a fabricated cookie or test-only authentication bypass. The sign-in form is still a coverage gap.

**Review/release:** Use the [traceability table](verification.md#broker-import-acceptance-example), PR diff, local results, and actual hosted run when available. Review and staging/database validation precede deployment; a read-only production smoke check follows it. A passing browser test on JSON does not prove PostgreSQL transactions or deployment correctness.

## Import integrity example to demonstrate

1. Start the synthetic local demo from the README and open Cam's journal.
2. Upload `e2e/fixtures/synthetic-schwab.csv`: two fills, one closed XYZ trade, $28 P/L.
3. Open Trade Log, edit Stop to `98` and add a note, then save.
4. Reimport: expect `New: 0. Updated: 0. Duplicates ignored: 1`.
5. Read the browser test's exact equality assertion. Explain that it checks trade ID, execution IDs, signed quantities, prices, P/L, stop, notes, and other persisted fields rather than only the number of rows.
6. Explain why the fresh-context check matters: a reload alone could display an intact localStorage cache even if server persistence failed.

Do not import real statements into a shared or production environment for the interview demo.

## A failure investigation supported by this work

During initial local browser execution, Playwright reported:

```text
getByLabel('Import Broker CSV', { exact: true }) resolved to 2 elements:
  #importInput
  #settingsImportInput
```

**Observation:** The test failed before selecting a file; the trace and error context showed both sidebar and Settings import controls. This was test-selector ambiguity, not evidence of a parser failure.

**Hypothesis and inspection:** Both labels legitimately use the same words. Read `public/cam-journal/index.html` and inspect the locator's scope.

**Correction:** Scope the accessible label to `page.locator('aside')`. A later fresh-browser step likewise needed the XYZ table cell rather than every XYZ text node. Wait for real save responses and completed boot (`aria-busy=false`) instead of inserting sleeps.

**Evidence:** The corrected suite completes the import and server/disk/fresh-browser assertions; final results are recorded in [audit.md](audit.md). Failure reports/traces are generated on a subsequent failing run, not permanent proof bundled into source. An interrupted dependency install/run was an execution mistake, not an application defect. No application accounting bug was established by these selector failures.

This example demonstrates classification of failures, inspection, a targeted correction, and rerun evidence. Do not present it as a discovered production incident.

## Local checks, CI, deployment, and smoke checks

| Activity | Meaning |
| --- | --- |
| Local typecheck | Static consistency; no runtime execution |
| Local Node suite | Fast assertions on domain logic plus selected integration/mocked/static checks |
| Local Playwright | Real Chromium, local HTTP server, synthetic data, JSON persistence |
| Hosted Actions | Repeats commands on a fresh pinned runner; only a recorded hosted result proves it ran there |
| Deployment | Builds and releases through existing Vercel settings; CI here does not deploy |
| Staging validation | Tests deployed auth, PostgreSQL, and relevant integration behavior with isolated data |
| Production smoke | Small approved read-only check that the released version loads and reads expected records |

## What existed and what this change adds

| Before this work, verified at commit `0c15369` | Added in this change |
| --- | --- |
| Next.js/React/TypeScript app, Cam browser JS, JSON/PostgreSQL store, signed-session auth | No stack migration or authentication bypass |
| Broker parsing/import/upsert and manual review preservation | Synthetic two-fill browser fixture and independent expected result |
| 27 Node test files; 228 passed and one optional private-file case skipped in baseline | Single all-regression command and separate typecheck command |
| Static markup/source/VM checks as well as domain and storage tests | Three Playwright cases with real local persistence and access checks |
| Existing Vercel configuration and cron routes | PR checks with reports/traces; no deployment action |
| Documentation centered on the original monthly report | Current app overview, verification/release mapping, audit, and this guide |

## Read these before presenting

Read in this order; be able to explain the return values and failure paths:

1. `e2e/fixtures/synthetic-schwab.csv` and `e2e/broker-import.spec.ts`: independent arithmetic, assertions, save waits, fresh browser, and invalid-input invariants.
2. `public/cam-journal/app.js`: `importBrokerCsvFile`, `queueRemoteSave`, `saveRemoteStateNow`, `loadRemoteJournalState`, and `bootJournal`.
3. `public/cam-journal/broker-csv-parser.js`: `statementExecutions`, `groupExecutionsIntoTrades`, `upsertImportedTrades`. Follow one fixture by hand.
4. `app/api/cam-journal/route.ts` and `lib/store.ts`: `getCamJournalState`, `saveCamJournalState`, JSON versus PostgreSQL, and full-state write limitations.
5. `scripts/run-browser-tests.mjs`, `playwright.config.ts`, and `../.github/workflows/verification.yml`: isolation, cleanup, test discovery, runtime, and artifact behavior.
6. `tests/cam-paper-import.test.mjs`: reversals and conservation checks; `tests/ibkr-flex.test.ts`: store preservation and the explicitly skipped optional input case.

## Short diagnostic exercise

Use a local feature branch and synthetic fixtures only:

1. In the browser test, temporarily change the expected `pl: 28` to `pl: 29` (do not change the fixture or application).
2. Run `npm run test:e2e -- --grep 'synthetic import'` (use the Chromium override from the README if needed).
3. Read the assertion diff and open `npm run test:e2e:report`. Follow the upload, PUT response, and saved record in the trace. Recalculate `10*(103-100)-2` by hand. Decide whether the fixture, expectation, or implementation is wrong.
4. Restore `pl: 28`, rerun, and inspect your diff to ensure no deliberate failure remains. Do not commit the failing expectation.
5. Optional extension: change only the synthetic sell price and amount to $104 and $1040, derive $38 independently, and update the relevant expectations. Explain why silently deriving expected P/L using the production parser would weaken the test.

## Likely follow-up questions

**Why Playwright when Node tests already exist?** Node tests cover more accounting edge cases quickly. They do not prove that the file input, UI state, save API, persisted JSON, and recovery work together.

**Why API and disk assertions in a browser test?** Visible rows can hide execution duplication; local cache can hide failed persistence. The assertions inspect complete saved trade contents, and a fresh browser verifies retrieval. Most parsing math stays below this layer.

**Does this test PostgreSQL?** No. It exercises the non-production JSON adapter. PostgreSQL transactions, migrations, backup restore, and concurrent saves need staging/database integration tests.

**Why one worker and a fixed local port?** Cam has one shared journal state. Serial cases reset a disposable server; reusing a user's server is forbidden. A busy port fails instead of risking unrelated data. Each case is independently seeded, so order is irrelevant.

**What happens when services are unavailable?** The chosen browser workflow needs no live service. Existing tests mock market/AI responses and email transport to inspect error handling and request/response contracts. They do not prove a live provider contract.

**What are the main data risks?** Duplicate identities, reversal quantity/cash allocation, overwriting review fields, stale market sessions, and whole-state concurrent saves. Current coverage addresses selected examples; whole-state writes still lack concurrency control.

**Is authentication production-ready?** It is a simple configured-password/HMAC-cookie model. Built-in delegate access, the public positions route, secret fallback, password storage/comparison, session lifetime, and lack of a demonstrated security review are explicit limits. Tests did not weaken auth.

**Why no NestJS, Go, Kubernetes, or WebSocket demo?** Those technologies do not solve the selected repository problem. The useful overlap is TypeScript, React, data correctness, investigation, and repeatable checks. Deepvue role descriptions and employee profiles do not prove its current architecture.

**What can I claim?** That you can explain and review this AI-assisted implementation, calculate the fixture result independently, distinguish test layers, investigate a test failure, and show the actual evidence recorded for local/hosted checks. Describe production experience and tool proficiency only from your own experience.
