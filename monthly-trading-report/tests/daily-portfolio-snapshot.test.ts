import assert from "node:assert/strict";
import { mkdtemp, readFile, readdir } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import {
  buildDailyPortfolioSnapshot,
  latestCompletedMarketSession,
  renderDailyPortfolioSnapshotMarkdown,
  resolveSnapshotSession,
  validateDailyPortfolioSnapshot
} from "../lib/daily-portfolio-snapshot";
import { generateDailyPortfolioSnapshot, SnapshotValidationError } from "../lib/daily-portfolio-snapshot-server";
import { sendDailyPortfolioSnapshotEmail, snapshotEmailConfiguration } from "../lib/snapshot-email";
import type { TradeLogEntry } from "../lib/types";

function trade(overrides: Partial<TradeLogEntry> = {}): TradeLogEntry {
  return {
    id: "trade-1", userId: "branden", importSource: "cf-statement-pdf", importRowKey: "row-1", symbol: "TEST", side: "LONG", status: "OPEN",
    entryDate: "2026-07-15", exitDate: "", openTime: "10:00:00", closeTime: "", avgEntry: 100, exitPrice: 0, stopPrice: 95, takeProfitPrice: 120,
    shares: 10, commission: 2, usedMargin: 1000, risk: 50, pnl: 0, rMultiple: 0, returnPercent: 0, daysInTrade: 1,
    setupTags: ["Breakout"], mistakeTags: [], customTags: [], manualGrade: "A", portfolioTag: "Main", emotion: "", tradeQuality: "",
    checklistItems: [{ id: "criterion-1", criteria: "Volume", met: true, points: 5, inputType: "boolean" }], notes: "Plan recorded.", screenshots: ["shot"], chartLinks: ["https://example.test/chart"],
    executions: [{ id: "entry", type: "ENTRY", date: "2026-07-15", time: "10:00:00", side: "LONG", shares: 10, price: 100, pnl: 0, commission: 1, source: "broker", sourceKey: "entry" }],
    hidden: false, groupId: "", groupRole: "none", createdAt: "2026-07-15T14:00:00Z", updatedAt: "2026-07-16T21:00:00Z", ...overrides
  };
}

function input(trades: TradeLogEntry[]) {
  return {
    requestedSession: "2026-07-16", latestCompletedMarketSession: "2026-07-16", generatedAt: "2026-07-17T00:00:00Z", accountName: "Main",
    portfolioMeta: { currentEquity: 100000, equityUpdatedAt: "2026-07-16T22:00:00Z", equityStatementDate: "2026-07-16" }, trades,
    setupTemplates: [], prices: new Map([["TEST", { symbol: "TEST", price: 110, timestamp: "2026-07-16", provider: "test" }]]),
    sourceEnvironment: "test", applicationVersion: "test"
  };
}

test("resolves weekends, holidays, and same-day cutoff to a completed session", () => {
  assert.equal(resolveSnapshotSession("2026-07-18", new Date("2026-07-20T13:00:00Z")).resolved, "2026-07-17");
  assert.equal(resolveSnapshotSession("2026-07-03", new Date("2026-07-06T22:00:00Z")).resolved, "2026-07-02");
  assert.equal(latestCompletedMarketSession(new Date("2026-07-16T19:00:00Z")), "2026-07-15");
  assert.equal(latestCompletedMarketSession(new Date("2026-07-16T21:00:00Z")), "2026-07-16");
});

test("builds current open positions without mutating stored inputs", () => {
  const trades = [trade()], before = structuredClone(trades);
  const snapshot = buildDailyPortfolioSnapshot(input(trades));
  assert.deepEqual(trades, before);
  assert.equal(snapshot.open_positions[0].current_pnl, 100);
  assert.equal(snapshot.portfolio_summary.gross_exposure_dollars, 1100);
  assert.equal(snapshot.open_positions[0].current_r_multiple, 2);
  assert.deepEqual(validateDailyPortfolioSnapshot(snapshot), []);
});

test("includes only trades whose final exit occurred during the selected session", () => {
  const closed = trade({ status: "WIN", pnl: 100, exitDate: "2026-07-16", exitPrice: 110, executions: [
    trade().executions[0],
    { id: "exit-1", type: "EXIT", date: "2026-07-16", time: "14:00:00", side: "LONG", shares: 4, price: 108, pnl: 32, commission: 1, source: "broker", sourceKey: "exit-1" },
    { id: "exit-2", type: "EXIT", date: "2026-07-16", time: "15:00:00", side: "LONG", shares: 6, price: 112, pnl: 68, commission: 1, source: "broker", sourceKey: "exit-2" }
  ] });
  const older = trade({ id: "old", status: "WIN", exitDate: "2026-07-15", executions: [...trade().executions, { id: "old-exit", type: "EXIT", date: "2026-07-15", time: "15:00:00", side: "LONG", shares: 10, price: 101, pnl: 10, commission: 1, source: "broker", sourceKey: "old" }] });
  const snapshot = buildDailyPortfolioSnapshot(input([closed, older]));
  assert.equal(snapshot.trades_closed_during_session.length, 1);
  assert.equal(snapshot.trades_closed_during_session[0].partial_exits.length, 2);
  assert.equal(snapshot.trades_closed_during_session[0].gross_pnl, 100);
  assert.equal(snapshot.trades_closed_during_session[0].net_pnl, 97);
  assert.equal(snapshot.trades_closed_during_session[0].realized_r_multiple, 1.94);
});

test("missing and stale inputs produce codes and null portfolio aggregates", () => {
  const missing = trade({ stopPrice: 0, risk: 0, setupTags: [], manualGrade: "", notes: "", screenshots: [], chartLinks: [], executions: [], importSource: "manual" });
  const value = input([missing]);
  value.portfolioMeta = { currentEquity: 0, equityUpdatedAt: "", equityStatementDate: "" };
  value.prices.set("TEST", { symbol: "TEST", price: 109, timestamp: "2026-07-15", provider: "test" });
  const snapshot = buildDailyPortfolioSnapshot(value);
  const codes = new Set(snapshot.warnings.map((item) => item.code));
  for (const code of ["BROKER_IMPORT_MISSING", "MISSING_STOP", "MISSING_INITIAL_RISK", "MISSING_SETUP", "MISSING_GRADE", "MISSING_NOTES", "MISSING_CHART", "MISSING_SCREENSHOT", "MISSING_EXECUTIONS", "CURRENT_PRICE_STALE"]) assert(codes.has(code as never), code);
  assert.equal(snapshot.portfolio_summary.gross_exposure_dollars, null);
  assert.equal(snapshot.snapshot_status, "INCOMPLETE");
});

test("grade consistency warnings do not rewrite the stored grade", () => {
  const inconsistent = trade({ manualGrade: "A", checklistItems: [{ id: "low", criteria: "Low", met: false, points: 10, inputType: "boolean" }] });
  const snapshot = buildDailyPortfolioSnapshot(input([inconsistent]));
  assert.equal(snapshot.open_positions[0].grade, "A");
  assert(snapshot.warnings.some((item) => item.code === "GRADE_CRITERIA_CONFLICT"));
});

test("No Trade Setup with an A grade produces a conflict warning", () => {
  const snapshot = buildDailyPortfolioSnapshot(input([trade({ setupTags: ["No Trade Setup"], manualGrade: "A" })]));
  assert(snapshot.warnings.some((item) => item.code === "GRADE_CRITERIA_CONFLICT" && item.message.includes("No Trade Setup")));
});

test("a prior broker statement is stale for the requested session", () => {
  const value = input([trade()]);
  value.portfolioMeta.equityStatementDate = "2026-07-15";
  const snapshot = buildDailyPortfolioSnapshot(value);
  assert(snapshot.warnings.some((item) => item.code === "BROKER_IMPORT_STALE"));
  assert.equal(snapshot.metadata.broker_import_complete, true);
});

test("a newer broker statement cannot be used as historical point-in-time portfolio state", () => {
  const value = input([trade()]);
  value.portfolioMeta.equityStatementDate = "2026-07-17";
  const snapshot = buildDailyPortfolioSnapshot(value);
  assert(snapshot.warnings.some((item) => item.code === "BROKER_IMPORT_INCOMPLETE" && item.message.includes("newer than requested session")));
  assert.equal(snapshot.metadata.broker_import_complete, true);
});

test("trade-log review tags and missing executions remain per-trade warnings instead of invalidating fresh broker metadata", () => {
  const reviewedInTradeLog = trade({ customTags: ["Needs review"], executions: [] });
  const snapshot = buildDailyPortfolioSnapshot(input([reviewedInTradeLog]));
  assert.equal(snapshot.metadata.broker_import_complete, true);
  assert(snapshot.open_positions[0].data_warnings.some((item) => item.code === "MISSING_EXECUTIONS"));
  assert(!snapshot.warnings.some((item) => item.code === "BROKER_IMPORT_INCOMPLETE"));
});

test("markdown is sourced from the JSON snapshot", () => {
  const snapshot = buildDailyPortfolioSnapshot(input([trade()]));
  const markdown = renderDailyPortfolioSnapshotMarkdown(snapshot);
  assert.match(markdown, new RegExp(snapshot.metadata.snapshot_id));
  assert.match(markdown, /TEST/);
});

test("server orchestration writes JSON and Markdown while data loaders remain read-only", async () => {
  const outputDirectory = await mkdtemp(path.join(os.tmpdir(), "snapshot-test-"));
  const trades = [trade()], before = structuredClone(trades);
  const result = await generateDailyPortfolioSnapshot({
    session: "2026-07-16", accountName: "Main", outputDirectory,
    dependencies: {
      now: () => new Date("2026-07-17T00:00:00Z"),
      loadTrades: async () => trades,
      loadPortfolioSettings: async () => ({ portfolios: ["Main"], defaultPortfolio: "Main", portfolioMeta: { Main: input([]).portfolioMeta } }),
      loadPrice: async (symbol, session) => ({ symbol, price: 110, timestamp: session, provider: "test" })
    }
  });
  assert.deepEqual(trades, before);
  assert.equal(JSON.parse(await readFile(result.jsonPath, "utf8")).metadata.snapshot_id, result.snapshot.metadata.snapshot_id);
  assert.equal(await readFile(result.markdownPath, "utf8"), result.markdown);
});

test("broker-import validation returns distinct safe codes and writes no exports", async () => {
  const cases = [
    { name: "not found", code: "BROKER_IMPORT_NOT_FOUND", trades: [trade({ importSource: "manual" })], meta: input([]).portfolioMeta },
    { name: "stale", code: "BROKER_IMPORT_STALE", trades: [trade()], meta: { ...input([]).portfolioMeta, equityStatementDate: "2026-07-15" } },
    { name: "needs review", code: "BROKER_IMPORT_NEEDS_REVIEW", trades: [trade({ customTags: ["Needs review"] })], meta: input([]).portfolioMeta },
    { name: "missing executions", code: "BROKER_IMPORT_MISSING_EXECUTIONS", trades: [trade({ executions: [] })], meta: input([]).portfolioMeta },
    { name: "portfolio mismatch", code: "BROKER_IMPORT_PORTFOLIO_MISMATCH", trades: [trade({ portfolioTag: "Other" })], meta: input([]).portfolioMeta },
    { name: "date coverage", code: "BROKER_IMPORT_DATE_COVERAGE_INSUFFICIENT", trades: [trade()], meta: { ...input([]).portfolioMeta, equityStatementDate: "2026-07-17" } }
  ] as const;
  for (const item of cases) {
    const outputDirectory = await mkdtemp(path.join(os.tmpdir(), `snapshot-broker-${item.name.replace(/ /g, "-")}-`));
    await assert.rejects(
      generateDailyPortfolioSnapshot({
        session: "2026-07-16", accountName: "Main", outputDirectory,
        dependencies: {
          now: () => new Date("2026-07-17T22:00:00Z"),
          loadTrades: async () => [...item.trades],
          loadPortfolioSettings: async () => ({ portfolios: ["Main"], defaultPortfolio: "Main", portfolioMeta: { Main: item.meta } }),
          loadPrice: async (symbol, session) => ({ symbol, price: 110, timestamp: session, provider: "test" })
        }
      }),
      (error) => error instanceof SnapshotValidationError
        && error.code === item.code
        && error.diagnostic?.validationCodes.includes(item.code)
        && error.diagnostic?.requestedSession === "2026-07-16"
        && error.diagnostic?.portfolio === "Main"
    );
    assert.deepEqual(await readdir(outputDirectory), []);
  }
});

test("post-session trade-log activity blocks a historical snapshot", async () => {
  const outputDirectory = await mkdtemp(path.join(os.tmpdir(), "snapshot-point-in-time-test-"));
  const laterExecution = { ...trade().executions[0], id: "later", date: "2026-07-17", sourceKey: "later" };
  await assert.rejects(
    generateDailyPortfolioSnapshot({
      session: "2026-07-16", accountName: "Main", outputDirectory,
      dependencies: {
        now: () => new Date("2026-07-17T22:00:00Z"),
        loadTrades: async () => [trade({ executions: [...trade().executions, laterExecution] })],
        loadPortfolioSettings: async () => ({ portfolios: ["Main"], defaultPortfolio: "Main", portfolioMeta: { Main: input([]).portfolioMeta } }),
        loadPrice: async (symbol, session) => ({ symbol, price: 110, timestamp: session, provider: "test" })
      }
    }),
    (error) => error instanceof SnapshotValidationError && error.code === "POINT_IN_TIME_UNAVAILABLE"
  );
  assert.deepEqual(await readdir(outputDirectory), []);
});


test("server does not write exports when a current price is stale", async () => {
  const outputDirectory = await mkdtemp(path.join(os.tmpdir(), "snapshot-stale-test-"));
  await assert.rejects(
    generateDailyPortfolioSnapshot({
      session: "2026-07-16", accountName: "Main", outputDirectory,
      dependencies: {
        now: () => new Date("2026-07-17T00:00:00Z"),
        loadTrades: async () => [trade()],
        loadPortfolioSettings: async () => ({ portfolios: ["Main"], defaultPortfolio: "Main", portfolioMeta: { Main: input([]).portfolioMeta } }),
        loadPrice: async (symbol) => ({ symbol, price: 109, timestamp: "2026-07-15", provider: "test" })
      }
    }),
    (error) => error instanceof SnapshotValidationError && error.code === "CURRENT_PRICES_INVALID"
  );
  assert.deepEqual(await readdir(outputDirectory), []);
});

test("server can return browser download payloads without writing deployment files", async () => {
  const outputDirectory = await mkdtemp(path.join(os.tmpdir(), "snapshot-browser-test-"));
  const result = await generateDailyPortfolioSnapshot({
    session: "2026-07-16", accountName: "Main", outputDirectory, writeExports: false,
    dependencies: {
      now: () => new Date("2026-07-17T00:00:00Z"),
      loadTrades: async () => [trade()],
      loadPortfolioSettings: async () => ({ portfolios: ["Main"], defaultPortfolio: "Main", portfolioMeta: { Main: input([]).portfolioMeta } }),
      loadPrice: async (symbol, session) => ({ symbol, price: 110, timestamp: session, provider: "test" })
    }
  });
  assert.equal(result.snapshot.open_positions.length, 1);
  assert.match(result.markdown, /TEST/);
  assert.deepEqual(await readdir(outputDirectory), []);
});

test("email is disabled by default and test transport receives no credentials", async () => {
  assert.equal(snapshotEmailConfiguration({}).configured, false);
  const snapshot = buildDailyPortfolioSnapshot(input([trade()]));
  let message: Record<string, unknown> | undefined;
  const result = await sendDailyPortfolioSnapshotEmail({
    snapshot, markdown: "report", baseName: "snapshot",
    environment: { SNAPSHOT_SMTP_HOST: "smtp.test", SNAPSHOT_SMTP_PORT: "465", SNAPSHOT_SMTP_USERNAME: "user", SNAPSHOT_SMTP_PASSWORD: "secret", SNAPSHOT_EMAIL_FROM: "from@test", SNAPSHOT_EMAIL_TO: "to@test" },
    transport: { sendMail: async (value) => { message = value; } }
  });
  assert.equal(result.status, "sent");
  assert.equal((message?.subject as string), "Trading Dashboard Snapshot — 2026-07-16");
  assert.match(message?.text as string, /Position count: 1/);
  assert(!JSON.stringify(message).includes("secret"));
});
