import assert from "node:assert/strict";
import { mkdtemp, readFile, readdir } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import {
  buildDailyPortfolioSnapshot,
  latestCompletedMarketSession,
  marketSessionCloseTimestamp,
  renderDailyPortfolioSnapshotMarkdown,
  resolveSnapshotSession,
  snapshotStatusFromWarnings,
  validateDailyPortfolioSnapshot
} from "../lib/daily-portfolio-snapshot";
import { buildDailySnapshotRequestBody, snapshotSessionFromRequestBody } from "../lib/daily-portfolio-snapshot-request";
import { generateDailyPortfolioSnapshot, SnapshotValidationError } from "../lib/daily-portfolio-snapshot-server";
import { sendDailyPortfolioSnapshotEmail, snapshotEmailConfiguration } from "../lib/snapshot-email";
import { reviewSectionsFromLegacyNotes } from "../lib/trade-review";
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

test("session resolution never silently replaces the selected date", () => {
  assert.deepEqual(
    { resolved: resolveSnapshotSession("2026-07-18", new Date("2026-07-20T13:00:00Z")).resolved, complete: resolveSnapshotSession("2026-07-18", new Date("2026-07-20T13:00:00Z")).complete },
    { resolved: "2026-07-18", complete: false }
  );
  assert.deepEqual(
    { resolved: resolveSnapshotSession("2026-07-03", new Date("2026-07-06T22:00:00Z")).resolved, complete: resolveSnapshotSession("2026-07-03", new Date("2026-07-06T22:00:00Z")).complete },
    { resolved: "2026-07-03", complete: false }
  );
  assert.equal(latestCompletedMarketSession(new Date("2026-07-16T19:00:00Z")), "2026-07-15");
  assert.equal(latestCompletedMarketSession(new Date("2026-07-16T19:59:00Z")), "2026-07-15");
  assert.equal(latestCompletedMarketSession(new Date("2026-07-16T20:00:00Z")), "2026-07-16");
});

test("request-body date takes precedence and preserves the selected value", () => {
  assert.equal(snapshotSessionFromRequestBody({ session: "2026-07-15" }, "2026-07-16"), "2026-07-15");
  assert.deepEqual(buildDailySnapshotRequestBody(" 2026-07-16 ", " CF_Statement ", false), {
    session: "2026-07-16", accountName: "CF_Statement", sendEmail: false
  });
  assert.deepEqual(buildDailySnapshotRequestBody(" 2026-07-17 ", " CF_Statement ", true), {
    session: "2026-07-17", accountName: "CF_Statement", sendEmail: true
  });
});

test("snapshot status reflects final warning severity", () => {
  assert.equal(snapshotStatusFromWarnings([]), "COMPLETE");
  assert.equal(snapshotStatusFromWarnings([{ code: "BROKER_IMPORT_UNRELATED_ROWS_NEED_REVIEW", message: "Historical row", severity: "warning" }]), "COMPLETE_WITH_WARNINGS");
  assert.equal(snapshotStatusFromWarnings([{ code: "CURRENT_PRICE_STALE", message: "Current price missing", severity: "critical" }]), "INCOMPLETE");
});

test("legacy labeled notes are split deterministically without changing the original notes", () => {
  const notes = `Setup:\nCup with handle\n\nWhat did I do right:\nWaited for volume\n\nWhat did I do wrong:\nSized too quickly\n\nExit strategy:\nSell into weakness`;
  assert.deepEqual(reviewSectionsFromLegacyNotes(notes), {
    setup: "Cup with handle",
    entry: "",
    exit: "Sell into weakness",
    didRight: "Waited for volume",
    didWrong: "Sized too quickly",
    general: ""
  });
  const snapshot = buildDailyPortfolioSnapshot(input([trade({ notes })]));
  assert.equal(snapshot.open_positions[0].review_sections.setup, "Cup with handle");
  assert.equal(snapshot.open_positions[0].review_sections.exit, "Sell into weakness");
  assert.equal(snapshot.open_positions[0].notes, notes);
  assert(!snapshot.open_positions[0].data_warnings.some((warning) => warning.code === "MISSING_NOTES"));
});

test("structured review fields are emitted independently of legacy notes", () => {
  const snapshot = buildDailyPortfolioSnapshot(input([trade({
    notes: "",
    reviewSections: { setup: "Pullback", entry: "50-day support", exit: "", didRight: "Followed risk", didWrong: "", general: "" }
  })]));
  assert.equal(snapshot.open_positions[0].review_sections.entry, "50-day support");
  assert.equal(snapshot.open_positions[0].notes, null);
  assert(!snapshot.open_positions[0].data_warnings.some((warning) => warning.code === "MISSING_NOTES"));
});

test("New York market timestamps carry the correct daylight-saving offset", () => {
  assert.equal(marketSessionCloseTimestamp("2026-07-17"), "2026-07-17T16:00:00-04:00");
  assert.equal(marketSessionCloseTimestamp("2026-01-16"), "2026-01-16T16:00:00-05:00");
});

test("builds current open positions without mutating stored inputs", () => {
  const trades = [trade()], before = structuredClone(trades);
  const snapshot = buildDailyPortfolioSnapshot(input(trades));
  assert.deepEqual(trades, before);
  assert.equal(snapshot.open_positions[0].unrealized_pnl, 100);
  assert.equal(snapshot.open_positions[0].realized_pnl_to_date, 0);
  assert.equal(snapshot.open_positions[0].total_trade_pnl, 100);
  assert.equal(snapshot.portfolio_summary.gross_exposure_dollars, 1100);
  assert.equal(snapshot.open_positions[0].open_r_multiple, 2);
  assert.equal(snapshot.open_positions[0].lifecycle_r_multiple, 2);
  assert.deepEqual(validateDailyPortfolioSnapshot(snapshot), []);
});

test("July 17 open and lifecycle R distinguish unrealized from realized P&L", () => {
  const partialExecutions = (symbol: string, realizedPnl: number) => [
    { ...trade().executions[0], id: `${symbol}-entry`, date: "2026-07-15", shares: 20, sourceKey: `${symbol}-entry` },
    { ...trade().executions[0], id: `${symbol}-exit`, type: "EXIT" as const, date: "2026-07-16", shares: 10, price: 100, pnl: realizedPnl, sourceKey: `${symbol}-exit` }
  ];
  const nvo = trade({ id: "nvo", symbol: "NVO", shares: 10, risk: 100, pnl: 274, executions: partialExecutions("nvo", 274) });
  const lly = trade({ id: "lly", symbol: "LLY", shares: 10, risk: 100, pnl: -13, executions: partialExecutions("lly", -13) });
  const value = input([nvo, lly]);
  value.requestedSession = "2026-07-17";
  value.latestCompletedMarketSession = "2026-07-17";
  value.portfolioMeta.equityStatementDate = "2026-07-17";
  value.prices = new Map([
    ["NVO", { symbol: "NVO", price: 126.6, sessionDate: "2026-07-17", timestamp: "2026-07-17T16:00:00-04:00", provider: "stooq", priceType: "delayed_close" as const }],
    ["LLY", { symbol: "LLY", price: 101.4, sessionDate: "2026-07-17", timestamp: "2026-07-17T16:00:00-04:00", provider: "stooq", priceType: "delayed_close" as const }]
  ]);
  const snapshot = buildDailyPortfolioSnapshot(value);
  const nvoPosition = snapshot.open_positions.find((position) => position.ticker === "NVO")!;
  const llyPosition = snapshot.open_positions.find((position) => position.ticker === "LLY")!;
  assert.equal(nvoPosition.open_r_multiple, 2.66);
  assert.equal(nvoPosition.lifecycle_r_multiple, 5.4);
  assert.equal(llyPosition.open_r_multiple, 0.14);
  assert.equal(llyPosition.lifecycle_r_multiple, 0.01);
  assert.equal(llyPosition.remaining_risk_to_stop_dollars, 64);
});

test("RELY planned risk remains mapped and LLY stop risk uses the remaining quantity", () => {
  const rely = trade({ id: "rely", symbol: "RELY", shares: 8, risk: 240, stopPrice: 92, executions: [{ ...trade().executions[0], id: "rely-entry", shares: 8 }] });
  const lly = trade({ id: "lly-risk", symbol: "LLY", shares: 6, risk: 180, stopPrice: 96, executions: [
    { ...trade().executions[0], id: "lly-entry", shares: 10 },
    { ...trade().executions[0], id: "lly-partial", type: "EXIT", shares: 4, price: 103, pnl: 12 }
  ] });
  const value = input([rely, lly]);
  value.prices = new Map([
    ["RELY", { symbol: "RELY", price: 110, timestamp: "2026-07-16", provider: "test" }],
    ["LLY", { symbol: "LLY", price: 105, timestamp: "2026-07-16", provider: "test" }]
  ]);
  const snapshot = buildDailyPortfolioSnapshot(value);
  const relyPosition = snapshot.open_positions.find((position) => position.ticker === "RELY")!;
  const llyPosition = snapshot.open_positions.find((position) => position.ticker === "LLY")!;
  assert.equal(relyPosition.planned_risk_dollars, 240);
  assert.equal(llyPosition.remaining_risk_to_stop_dollars, 54);
  assert.equal(snapshot.portfolio_summary.total_initial_risk, 420);
});

test("LLY linked staged exits use effective quantities and exclude the profit limit from downside risk", () => {
  const lly = trade({
    id: "lly-linked", symbol: "LLY", shares: 3, avgEntry: 1100, stopPrice: 1079.96, takeProfitPrice: 1287.94, risk: 300,
    executions: [{ ...trade().executions[0], id: "lly-linked-entry", shares: 3, price: 1100 }]
  });
  const rely = trade({
    id: "rely-risk", symbol: "RELY", shares: 10, avgEntry: 90, stopPrice: 67.26, risk: 250,
    executions: [{ ...trade().executions[0], id: "rely-risk-entry", shares: 10, price: 90 }]
  });
  const value = input([lly, rely]) as Parameters<typeof buildDailyPortfolioSnapshot>[0];
  value.requestedSession = "2026-07-17";
  value.latestCompletedMarketSession = "2026-07-17";
  value.portfolioMeta = { ...value.portfolioMeta, equityStatementDate: "2026-07-17", workingOrders: [
    { orderId: "lly-early-stop", orderDate: "2026-07-17", timeValue: "10:00:00", direction: "Sell", shares: 1, symbol: "LLY", orderType: "STOP", orderPrice: 1132.94 },
    { orderId: "lly-bracket-stop", orderDate: "2026-07-17", timeValue: "10:00:01", direction: "Sell", shares: 3, symbol: "LLY", orderType: "STOP", orderPrice: 1079.96 },
    { orderId: "lly-profit-limit", orderDate: "2026-07-17", timeValue: "10:00:02", direction: "Sell", shares: 1, symbol: "LLY", orderType: "LIMIT", orderPrice: 1287.94 }
  ] };
  value.prices = new Map([
    ["LLY", { symbol: "LLY", price: 1179.11, sessionDate: "2026-07-17", timestamp: "2026-07-17T16:00:00-04:00", provider: "stooq", priceType: "delayed_close" as const }],
    ["RELY", { symbol: "RELY", price: 100, sessionDate: "2026-07-17", timestamp: "2026-07-17T16:00:00-04:00", provider: "stooq", priceType: "delayed_close" as const }]
  ]);

  const snapshot = buildDailyPortfolioSnapshot(value);
  const position = snapshot.open_positions.find((item) => item.ticker === "LLY")!;
  assert.equal(position.stop_plan_type, "STAGED_LINKED_EXIT");
  assert.deepEqual(position.protective_levels.map((level) => ({ price: level.price, effective_quantity: level.effective_quantity, displayed_order_quantity: level.displayed_order_quantity })), [
    { price: 1132.94, effective_quantity: 1, displayed_order_quantity: 1 },
    { price: 1079.96, effective_quantity: 2, displayed_order_quantity: 3 }
  ]);
  assert.equal(position.stop_plan_provenance?.linkage, "BROKER_BRACKET_DYNAMIC_RESIZE");
  assert.equal(position.stop_plan_provenance?.dynamic_resize, true);
  assert.equal(position.stop_plan_provenance?.displayed_stop_quantity, 4);
  assert.equal(position.stop_plan_provenance?.effective_protective_quantity, 3);
  assert.deepEqual(position.profit_taking_orders.map((order) => ({ price: order.price, quantity: order.quantity })), [{ price: 1287.94, quantity: 1 }]);
  assert.equal(position.remaining_risk_to_stop_dollars, 244.47);
  assert.equal(snapshot.portfolio_summary.total_remaining_risk_to_stops, 571.87);
  assert(!position.data_warnings.some((warning) => warning.code === "POSITION_CALCULATION_MISMATCH"));
  assert.match(renderDailyPortfolioSnapshotMarkdown(snapshot), /STAGED_LINKED_EXIT/);
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

test("a missing stored stop on a closed trade is a noncritical documentation warning", () => {
  const closed = trade({ status: "LOSS", stopPrice: 0, pnl: -10, exitDate: "2026-07-16", executions: [
    trade().executions[0],
    { ...trade().executions[0], id: "closed-exit", type: "EXIT", date: "2026-07-16", shares: 10, price: 99, pnl: -10 }
  ] });
  const snapshot = buildDailyPortfolioSnapshot(input([closed]));
  const missingStop = snapshot.trades_closed_during_session[0].documentation_warnings.find((item) => item.code === "MISSING_STOP");
  assert.equal(missingStop?.severity, "warning");
  assert.equal(snapshot.critical_warning_count, 0);
  assert.equal(snapshot.snapshot_status, "COMPLETE_WITH_WARNINGS");
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

test("a statement covering a later date satisfies requested-session coverage", () => {
  const value = input([trade()]);
  value.portfolioMeta.equityStatementDate = "2026-07-17";
  const snapshot = buildDailyPortfolioSnapshot(value);
  assert(!snapshot.warnings.some((item) => item.code === "BROKER_IMPORT_INCOMPLETE"));
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

test("snapshot preserves exact broker time and explicit delayed-close price provenance", async () => {
  const exactBrokerTimestamp = "2026-07-17T16:00:35.000Z";
  const exactPriceTimestamp = "2026-07-17T16:00:00-04:00";
  const result = await generateDailyPortfolioSnapshot({
    session: "2026-07-17", accountName: "Main", writeExports: false,
    dependencies: {
      now: () => new Date("2026-07-17T20:08:00Z"),
      loadTrades: async () => [trade({ updatedAt: exactBrokerTimestamp })],
      loadPortfolioSettings: async () => ({ portfolios: ["Main"], defaultPortfolio: "Main", portfolioMeta: { Main: { ...input([]).portfolioMeta, equityUpdatedAt: exactBrokerTimestamp, equityStatementDate: "2026-07-17" } } }),
      loadPrice: async (symbol) => ({ symbol, price: 110, sessionDate: "2026-07-17", timestamp: exactPriceTimestamp, provider: "stooq", priceType: "delayed_close", retrievedAt: "2026-07-17T20:08:01.000Z" })
    }
  });
  assert.equal(result.snapshot.metadata.generated_at, "2026-07-17T20:08:00.000Z");
  assert.equal(result.snapshot.metadata.portfolio_data_as_of, exactBrokerTimestamp);
  assert.equal(result.snapshot.metadata.broker_import_timestamp, exactBrokerTimestamp);
  assert.equal(result.snapshot.metadata.broker_imported_at, exactBrokerTimestamp);
  assert.equal(result.snapshot.metadata.broker_position_state_as_of, "2026-07-17");
  assert.equal(result.snapshot.metadata.statement_coverage_date, "2026-07-17");
  assert.equal(result.snapshot.metadata.price_timestamp, exactPriceTimestamp);
  assert.equal(result.snapshot.metadata.price_as_of, exactPriceTimestamp);
  assert.equal(result.snapshot.metadata.price_retrieved_at, "2026-07-17T20:08:01.000Z");
  assert.equal(result.snapshot.metadata.price_source, "stooq");
  assert.equal(result.snapshot.metadata.price_type, "delayed_close");
  assert.equal(result.snapshot.metadata.valuation_context, "Broker position state reflects the imported statement. Current valuation uses post-close prices.");
  assert.equal(result.snapshot.open_positions[0].current_price_retrieved_at, "2026-07-17T20:08:01.000Z");
  assert(!result.snapshot.warnings.some((item) => item.code === "PORTFOLIO_DATA_STALE"));
  assert.match(result.markdown, /Broker position state reflects the imported statement\. Current valuation uses post-close prices\./);
});

test("a pre-close broker import timestamp is informational and never blocks generation", async () => {
  const result = await generateDailyPortfolioSnapshot({
    session: "2026-07-17", accountName: "Main", writeExports: false,
    dependencies: {
      now: () => new Date("2026-07-17T20:30:00Z"),
      loadTrades: async () => [trade({ updatedAt: "2026-07-17T16:00:35.000Z" })],
      loadPortfolioSettings: async () => ({ portfolios: ["Main"], defaultPortfolio: "Main", portfolioMeta: { Main: { ...input([]).portfolioMeta, equityUpdatedAt: "2026-07-17T16:00:35.000Z", equityStatementDate: "2026-07-17" } } }),
      loadPrice: async (symbol) => ({ symbol, price: 110, sessionDate: "2026-07-17", timestamp: "2026-07-17T16:00:00-04:00", provider: "stooq", priceType: "delayed_close", retrievedAt: "2026-07-17T20:08:01.000Z" })
    }
  });
  assert.equal(result.snapshot.metadata.broker_imported_at, "2026-07-17T16:00:35.000Z");
  assert(!result.snapshot.warnings.some((item) => item.code === "PORTFOLIO_DATA_STALE"));
  assert.notEqual(result.snapshot.snapshot_status, "INCOMPLETE");
});

test("July 15 and July 16 flow unchanged from selection to server evaluation", async () => {
  for (const selected of ["2026-07-15", "2026-07-16"]) {
    const result = await generateDailyPortfolioSnapshot({
      session: selected, accountName: "Main", writeExports: false,
      dependencies: {
        now: () => new Date("2026-07-17T22:00:00Z"),
        loadTrades: async () => [trade()],
        loadPortfolioSettings: async () => ({ portfolios: ["Main"], defaultPortfolio: "Main", portfolioMeta: { Main: input([]).portfolioMeta } }),
        loadPrice: async (symbol, session) => ({ symbol, price: 110, timestamp: session, provider: "test" })
      }
    });
    assert.deepEqual(result.datePath, {
      selectedDate: selected,
      submittedDate: selected,
      evaluatedDate: selected,
      latestCompletedSession: "2026-07-17"
    });
    assert.equal(result.snapshot.metadata.requested_session, selected);
  }
});

test("current New York session is rejected before close and evaluated after close without fallback", async () => {
  const dependencies = (now: Date) => ({
    now: () => now,
    loadTrades: async () => [trade()],
    loadPortfolioSettings: async () => ({ portfolios: ["Main"], defaultPortfolio: "Main", portfolioMeta: { Main: { ...input([]).portfolioMeta, equityStatementDate: "2026-07-17" } } }),
    loadPrice: async (symbol: string, session: string) => ({ symbol, price: 110, timestamp: session, provider: "test" })
  });
  await assert.rejects(
    generateDailyPortfolioSnapshot({ session: "2026-07-17", accountName: "Main", writeExports: false, dependencies: dependencies(new Date("2026-07-17T19:59:00Z")) }),
    (error) => error instanceof SnapshotValidationError
      && error.code === "SNAPSHOT_SESSION_NOT_COMPLETE"
      && error.diagnostic !== undefined
      && "selectedSession" in error.diagnostic
      && error.diagnostic.selectedSession === "2026-07-17"
      && error.diagnostic.latestCompletedSession === "2026-07-16"
  );
  const result = await generateDailyPortfolioSnapshot({
    session: "2026-07-17", accountName: "Main", writeExports: false,
    dependencies: dependencies(new Date("2026-07-17T20:00:00Z"))
  });
  assert.equal(result.datePath.evaluatedDate, "2026-07-17");
  assert.equal(result.snapshot.metadata.requested_session, "2026-07-17");
  assert.notEqual(result.datePath.evaluatedDate, "2026-07-16");
});

test("broker-import validation returns distinct safe codes and writes no exports", async () => {
  const cases = [
    { name: "not found", code: "BROKER_IMPORT_NOT_FOUND", trades: [trade({ importSource: "manual" })], meta: input([]).portfolioMeta },
    { name: "stale", code: "BROKER_IMPORT_STALE", trades: [trade()], meta: { ...input([]).portfolioMeta, equityStatementDate: "2026-07-15" } },
    { name: "needs review", code: "BROKER_IMPORT_NEEDS_REVIEW", trades: [trade({ customTags: ["Needs review"] })], meta: input([]).portfolioMeta },
    { name: "missing executions", code: "BROKER_IMPORT_MISSING_EXECUTIONS", trades: [trade({ executions: [] })], meta: input([]).portfolioMeta },
    { name: "portfolio mismatch", code: "BROKER_IMPORT_PORTFOLIO_MISMATCH", trades: [trade({ portfolioTag: "Other" })], meta: input([]).portfolioMeta }
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
        && error.diagnostic !== undefined
        && "portfolio" in error.diagnostic
        && error.diagnostic.validationCodes.includes(item.code)
        && error.diagnostic.requestedSession === "2026-07-16"
        && error.diagnostic.portfolio === "Main"
    );
    assert.deepEqual(await readdir(outputDirectory), []);
  }
});

test("statement coverage is inclusive and date-only values retain their U.S. market session", async () => {
  const run = async (coverageDate: string) => generateDailyPortfolioSnapshot({
    session: "2026-07-16", accountName: "Main", writeExports: false,
    dependencies: {
      now: () => new Date("2026-07-17T22:00:00Z"),
      loadTrades: async () => [trade()],
      loadPortfolioSettings: async () => ({ portfolios: ["Main"], defaultPortfolio: "Main", portfolioMeta: { Main: { ...input([]).portfolioMeta, equityStatementDate: coverageDate } } }),
      loadPrice: async (symbol, session) => ({ symbol, price: 110, timestamp: session, provider: "test" })
    }
  });
  await assert.rejects(run("2026-07-15"), (error) => error instanceof SnapshotValidationError && error.diagnostic !== undefined && "portfolio" in error.diagnostic && error.diagnostic.validationCodes.includes("BROKER_IMPORT_DATE_COVERAGE_INSUFFICIENT"));
  assert.equal((await run("2026-07-16")).snapshot.open_positions.length, 1);
  assert.equal((await run("2026-07-17")).snapshot.open_positions.length, 1);
  // 00:30 UTC is still July 16 in New York; a date-only July 16 must also stay July 16.
  assert.equal((await run("2026-07-17T00:30:00.000Z")).snapshot.open_positions.length, 1);
  await assert.rejects(run("2026-07-16T00:30:00.000Z"), (error) => error instanceof SnapshotValidationError && error.diagnostic !== undefined && "portfolio" in error.diagnostic && error.diagnostic.validationCodes.includes("BROKER_IMPORT_DATE_COVERAGE_INSUFFICIENT"));
});

test("unrelated Needs review rows warn without blocking a snapshot", async () => {
  const unrelated = trade({
    id: "old-review", status: "LOSS", entryDate: "2026-07-10", exitDate: "2026-07-15", customTags: ["Needs review"],
    executions: [{ ...trade().executions[0], date: "2026-07-10" }, { ...trade().executions[0], id: "old-exit", type: "EXIT", date: "2026-07-15", shares: 10, price: 99 }]
  });
  const result = await generateDailyPortfolioSnapshot({
    session: "2026-07-16", accountName: "Main", writeExports: false,
    dependencies: {
      now: () => new Date("2026-07-17T22:00:00Z"),
      loadTrades: async () => [trade(), unrelated],
      loadPortfolioSettings: async () => ({ portfolios: ["Main"], defaultPortfolio: "Main", portfolioMeta: { Main: input([]).portfolioMeta } }),
      loadPrice: async (symbol, session) => ({ symbol, price: 110, timestamp: session, provider: "test" })
    }
  });
  assert(result.snapshot.warnings.some((warning) => warning.code === "BROKER_IMPORT_UNRELATED_ROWS_NEED_REVIEW"));
  assert.equal(result.snapshot.snapshot_status, "COMPLETE_WITH_WARNINGS");
  assert.equal(result.snapshot.critical_warning_count, 0);
  assert.deepEqual(result.brokerDiagnostic?.needsReviewRows, [{
    ticker: "TEST", tradeId: "old-review", entryDate: "2026-07-10", exitDate: "2026-07-15", status: "CLOSED", affectsRequestedSnapshot: false, blockingReason: null
  }]);
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
  assert.deepEqual(snapshotEmailConfiguration({
    SNAPSHOT_SMTP_HOST: "smtp.gmail.com",
    SNAPSHOT_SMTP_PORT: "465",
    SNAPSHOT_SMTP_SECURE: "true",
    SNAPSHOT_SMTP_USERNAME: "sender@example.test",
    SNAPSHOT_SMTP_PASSWORD: "app-password",
    SNAPSHOT_EMAIL_FROM: "sender@example.test",
    SNAPSHOT_EMAIL_TO: "recipient@example.test"
  }), {
    configured: true,
    values: {
      host: "smtp.gmail.com",
      port: 465,
      secure: true,
      username: "sender@example.test",
      password: "app-password",
      from: "sender@example.test",
      to: "recipient@example.test"
    }
  });
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
