import assert from "node:assert/strict";
import { mkdtemp, readFile, readdir } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import {
  ACCOUNT_LOSS_THRESHOLD_DOLLARS,
  buildMonthToDateSnapshot,
  mtdStatusFromDiagnostics,
  renderMonthToDateSnapshotMarkdown,
  resolveMtdPeriod,
  validateMonthToDateSnapshot
} from "../lib/month-to-date-snapshot";
import { sendMonthToDateSnapshotEmail } from "../lib/month-to-date-snapshot-email";
import { generateMonthToDateSnapshot, MonthToDateSnapshotValidationError } from "../lib/month-to-date-snapshot-server";
import type { TradeExecution, TradeLogEntry } from "../lib/types";
import { createUserDefinedWeeklyFocus } from "../lib/weekly-focus";

function execution(overrides: Partial<TradeExecution> = {}): TradeExecution {
  return {
    id: "entry-1", type: "ENTRY", date: "2026-07-02", time: "10:00:00", side: "LONG", shares: 10,
    price: 100, pnl: 0, commission: 1, source: "cf-statement-pdf", sourceKey: "entry-1", ...overrides
  };
}

function trade(overrides: Partial<TradeLogEntry> = {}): TradeLogEntry {
  return {
    id: "trade-1", userId: "branden", importSource: "cf-statement-pdf", importRowKey: "row-1", symbol: "TEST", side: "LONG", status: "OPEN",
    entryDate: "2026-07-02", exitDate: "", openTime: "10:00:00", closeTime: "", avgEntry: 100, exitPrice: 0, stopPrice: 95, takeProfitPrice: 120,
    shares: 10, commission: 1, usedMargin: 1000, risk: 50, pnl: 0, rMultiple: 0, returnPercent: 0, daysInTrade: 1,
    setupTags: ["CANSLIM"], mistakeTags: ["System"], customTags: ["CF Statement"], manualGrade: "A+", portfolioTag: "Branden Log", emotion: "Calm", tradeQuality: "Planned",
    checklistItems: [{ id: "c1", criteria: "Volume", points: 5, met: true, groupName: "Technicals", inputType: "boolean" }],
    notes: "Legacy notes verbatim", reviewSections: { setup: "Setup text", entry: "Entry text", exit: "Exit text", didRight: "Right text", didWrong: "Wrong text", general: "Review text" },
    screenshots: ["/api/trades/trade-1/screenshots/shot-1"], chartLinks: ["https://example.test/chart"], executions: [execution()], hidden: false,
    groupId: "group-1", groupRole: "parent", createdAt: "2026-07-02T14:00:00Z", updatedAt: "2026-07-17T21:00:00Z", ...overrides
  };
}

function snapshotInput(trades: TradeLogEntry[], overrides: Record<string, unknown> = {}) {
  return {
    month: "2026-07", asOfDate: "2026-07-17", asOfTimestamp: "2026-07-17T23:59:59-04:00", generatedAt: "2026-07-18T01:00:00Z",
    portfolioName: "Branden Log", trades,
    portfolioMeta: { currentEquity: 700_000, equityStatementDate: "2026-07-17", equityUpdatedAt: "2026-07-17T20:05:00-04:00", equitySource: "CF_STATEMENT" },
    prices: new Map([["TEST", { symbol: "TEST", price: 110, sessionDate: "2026-07-17", timestamp: "2026-07-17T16:00:00-04:00", provider: "test", priceType: "official_close" as const }]]),
    weeklyFocus: createUserDefinedWeeklyFocus({ summary: "Structure first.", focusItems: ["Follow the plan"] }, new Date("2026-07-12T16:00:00Z")),
    sourceEnvironment: "test", applicationVersion: "test", ...overrides
  };
}

test("period defaults, prior months, as-of dates, weekends, and New York boundaries are deterministic", () => {
  assert.deepEqual(resolveMtdPeriod({ now: new Date("2026-07-20T14:00:00Z") }), {
    month: "2026-07", asOfDate: "2026-07-20", start: "2026-07-01T00:00:00-04:00", end: "2026-07-20T14:00:00.000Z", timezone: "America/New_York"
  });
  const prior = resolveMtdPeriod({ month: "2026-06", asOfDate: "2026-06-30", now: new Date("2026-07-20T14:00:00Z") });
  assert.equal(prior.start, "2026-06-01T00:00:00-04:00");
  assert.equal(prior.end, "2026-06-30T23:59:59-04:00");
  assert.equal(resolveMtdPeriod({ month: "2026-07", asOfDate: "2026-07-19", now: new Date("2026-07-20T14:00:00Z") }).asOfDate, "2026-07-19");
  assert.equal(resolveMtdPeriod({ month: "2026-01", asOfDate: "2026-01-31", now: new Date("2026-02-01T12:00:00Z") }).start, "2026-01-01T00:00:00-05:00");
  assert.throws(() => resolveMtdPeriod({ month: "2026-06", asOfDate: "2026-07-01" }), /within the selected month/);
});

test("status semantics distinguish complete, warnings, and blocked data", () => {
  assert.equal(mtdStatusFromDiagnostics([]), "COMPLETE");
  assert.equal(mtdStatusFromDiagnostics([{ code: "OPTIONAL", severity: "warning", message: "Optional data missing", blocking: false }]), "COMPLETE_WITH_WARNINGS");
  assert.equal(mtdStatusFromDiagnostics([{ code: "CORE", severity: "critical", message: "Core data missing", blocking: true }]), "BLOCKED");
});

test("as-of timestamp excludes later activity", () => {
  const value = trade({ executions: [
    execution(),
    execution({ id: "exit-before", type: "EXIT", date: "2026-07-17", time: "12:00:00", shares: 4, price: 105, pnl: 20, commission: 1, sourceKey: "exit-before" }),
    execution({ id: "exit-after", type: "EXIT", date: "2026-07-17", time: "16:30:00", shares: 6, price: 106, pnl: 36, commission: 1, sourceKey: "exit-after" })
  ] });
  const snapshot = buildMonthToDateSnapshot(snapshotInput([value], { asOfTimestamp: "2026-07-17T15:00:00-04:00" }));
  assert.equal(snapshot.trades[0].executions.length, 2);
  assert.equal(snapshot.trades[0].quantities.current_quantity, 6);
  assert.equal(snapshot.trades[0].financials.realized_mtd_pnl, 18);
});

test("cross-month partial exits allocate only in-period realized P&L without double counting", () => {
  const value = trade({ entryDate: "2026-06-20", executions: [
    execution({ id: "entry", date: "2026-06-20", shares: 12, commission: 1, sourceKey: "entry" }),
    execution({ id: "june-exit", type: "EXIT", date: "2026-06-30", shares: 2, price: 102, pnl: 4, commission: 1, sourceKey: "june-exit" }),
    execution({ id: "july-exit-1", type: "EXIT", date: "2026-07-10", shares: 4, price: 105, pnl: 20, commission: 1, sourceKey: "july-exit-1" }),
    execution({ id: "july-exit-2", type: "EXIT", date: "2026-07-17", shares: 2, price: 110, pnl: 20, commission: 1, sourceKey: "july-exit-2" })
  ], shares: 4 });
  const snapshot = buildMonthToDateSnapshot(snapshotInput([value]));
  const result = snapshot.trades[0];
  assert.equal(result.quantities.initial_quantity, 12);
  assert.equal(result.quantities.maximum_quantity, 12);
  assert.equal(result.quantities.current_quantity, 4);
  assert.equal(result.financials.realized_mtd_pnl, 38);
  assert.equal(result.financials.lifecycle_realized_pnl, 40);
  assert.equal(result.executions.filter((item) => item.in_period).length, 2);
  assert.equal(snapshot.account_summary.realized_mtd_pnl, 38);
});

test("multiple adds, short exits, fees, and breakeven values remain execution based", () => {
  const value = trade({ side: "SHORT", risk: 100, executions: [
    execution({ id: "entry", side: "SHORT", shares: 5, price: 100, sourceKey: "entry" }),
    execution({ id: "add", side: "SHORT", date: "2026-07-03", shares: 5, price: 102, commission: 2, sourceKey: "add" }),
    execution({ id: "partial", type: "EXIT", side: "SHORT", date: "2026-07-10", shares: 4, price: 98, pnl: 12, commission: 1, sourceKey: "partial" }),
    execution({ id: "final", type: "EXIT", side: "SHORT", date: "2026-07-17", shares: 6, price: 101, pnl: -2, commission: 1, sourceKey: "final" })
  ], status: "BREAKEVEN", exitDate: "2026-07-17", closeTime: "15:00:00", shares: 10 });
  const snapshot = buildMonthToDateSnapshot(snapshotInput([value], { prices: new Map() }));
  assert.equal(snapshot.trades[0].status, "CLOSED");
  assert.equal(snapshot.trades[0].financials.realized_mtd_pnl, 5);
  assert.equal(snapshot.trades[0].financials.realized_mtd_r, 0.05);
  assert.equal(snapshot.trades[0].executions[1].fill_type, "ADD");
  assert.equal(snapshot.trades[0].executions[2].fill_type, "PARTIAL_EXIT");
  assert.equal(snapshot.trades[0].executions[3].fill_type, "FINAL_EXIT");
});

test("realized, unrealized, open R, and lifecycle R remain separate", () => {
  const value = trade({ risk: 50, shares: 5, executions: [
    execution({ id: "entry", shares: 10, sourceKey: "entry" }),
    execution({ id: "partial", type: "EXIT", date: "2026-07-10", shares: 5, price: 104, pnl: 20, commission: 1, sourceKey: "partial" })
  ] });
  const snapshot = buildMonthToDateSnapshot(snapshotInput([value]));
  const result = snapshot.trades[0].financials;
  assert.equal(result.realized_mtd_pnl, 18);
  assert.equal(result.unrealized_pnl, 50);
  assert.equal(result.total_trade_pnl, 68);
  assert.equal(result.open_r, 1);
  assert.equal(result.lifecycle_r, 1.36);
});

test("drawdown cushion is current equity minus threshold exactly once", () => {
  const losing = trade({ status: "LOSS", exitDate: "2026-07-17", closeTime: "15:00:00", executions: [
    execution(), execution({ id: "exit", type: "EXIT", date: "2026-07-17", shares: 10, price: 90, pnl: -100, commission: 1, sourceKey: "exit" })
  ] });
  const snapshot = buildMonthToDateSnapshot(snapshotInput([losing], { prices: new Map() }));
  assert.equal(snapshot.account_summary.current_equity, 700_000);
  assert.equal(snapshot.account_summary.account_loss_threshold_dollars, ACCOUNT_LOSS_THRESHOLD_DOLLARS);
  assert.equal(snapshot.account_summary.remaining_drawdown_cushion, 12_000);
  assert.equal(snapshot.account_summary.realized_mtd_pnl, -102);
});

test("manual grade, saved criteria, notes, tags, custom fields, images, and weekly focus are preserved", () => {
  const value = trade();
  const before = structuredClone(value);
  const snapshot = buildMonthToDateSnapshot(snapshotInput([value]));
  const result = snapshot.trades[0];
  assert.equal(result.classification.user_assigned_grade, "A+");
  assert.equal(result.criteria_evaluation.score, 5);
  assert.equal(result.criteria_evaluation.max_score, 5);
  assert.equal(result.criteria_evaluation.derived_grade, null);
  assert.equal(result.review.setup_notes, "Setup text");
  assert.equal(result.review.legacy_notes, "Legacy notes verbatim");
  assert.deepEqual(result.classification.mistake_tags, ["System"]);
  assert.equal(result.custom_fields.emotion, "Calm");
  assert.equal(snapshot.image_manifest[0].durable, true);
  assert.equal(snapshot.weekly_focus.summary, "Structure first.");
  assert.deepEqual(value, before);
});

test("staged quantity-aware stops feed planned downside risk", () => {
  const value = trade({ symbol: "LLY", avgEntry: 1164.81, risk: 300, shares: 3, executions: [execution({ shares: 3, price: 1164.81 })] });
  const snapshot = buildMonthToDateSnapshot(snapshotInput([value], {
    prices: new Map([["LLY", { symbol: "LLY", price: 1179.11, sessionDate: "2026-07-17", timestamp: "2026-07-17T16:00:00-04:00", provider: "test", priceType: "official_close" as const }]]),
    portfolioMeta: {
      currentEquity: 700_000, equityStatementDate: "2026-07-17", equitySource: "CF_STATEMENT",
      workingOrders: [
        { orderId: "early", orderDate: "2026-07-17", timeValue: "15:00:00", direction: "Sell", shares: 1, symbol: "LLY", orderType: "STOP", orderPrice: 1132.94 },
        { orderId: "bracket", orderDate: "2026-07-17", timeValue: "15:00:00", direction: "Sell", shares: 3, symbol: "LLY", orderType: "STOP", orderPrice: 1079.96 },
        { orderId: "target", orderDate: "2026-07-17", timeValue: "15:00:00", direction: "Sell", shares: 1, symbol: "LLY", orderType: "LIMIT", orderPrice: 1287.94 }
      ]
    }
  }));
  assert.equal(snapshot.open_positions[0].stop_plan?.stop_plan_type, "STAGED_LINKED_EXIT");
  assert.equal(snapshot.open_positions[0].financials.remaining_downside_risk, 244.47);
  assert.equal(snapshot.risk_summary.current_planned_downside_risk, 244.47);
  assert.equal(snapshot.open_positions[0].stop_plan?.profit_taking_orders[0].price, 1287.94);
});

test("Markdown is materially sourced from JSON and validation catches missing equity", () => {
  const snapshot = buildMonthToDateSnapshot(snapshotInput([trade()]));
  const markdown = renderMonthToDateSnapshotMarkdown(snapshot);
  assert.match(markdown, /# Trading Dashboard Month-to-Date Snapshot/);
  assert.match(markdown, /Branden Log/);
  assert.match(markdown, /\$12,000\.00/);
  assert.match(markdown, /Structure first\./);
  assert.deepEqual(validateMonthToDateSnapshot(snapshot), []);
  const missing = buildMonthToDateSnapshot(snapshotInput([trade()], { portfolioMeta: { equityStatementDate: "2026-07-17" } }));
  assert.equal(missing.status, "BLOCKED");
  assert(validateMonthToDateSnapshot(missing).some((item) => item.includes("equity")));
});

test("server writes only immutable exports and refuses missing core inputs before writing", async () => {
  const outputDirectory = await mkdtemp(path.join(os.tmpdir(), "mtd-snapshot-"));
  const dependencies = {
    loadTrades: async () => [trade()],
    loadPortfolioSettings: async () => ({ portfolios: ["Branden Log"], defaultPortfolio: "Branden Log", portfolioMeta: { "Branden Log": { currentEquity: 700_000, equityStatementDate: "2026-07-17", equitySource: "CF_STATEMENT" } } }),
    loadWeeklyFocus: async () => createUserDefinedWeeklyFocus({ summary: "Focus", focusItems: [] }, new Date("2026-07-12T16:00:00Z")),
    loadPrice: async (symbol: string) => ({ symbol, price: 110, sessionDate: "2026-07-17", timestamp: "2026-07-17T16:00:00-04:00", provider: "test", priceType: "official_close" as const }),
    now: () => new Date("2026-07-18T01:00:00Z")
  };
  const original = trade();
  const result = await generateMonthToDateSnapshot({ month: "2026-07", asOfDate: "2026-07-17", outputDirectory, dependencies: { ...dependencies, loadTrades: async () => [original] } });
  assert.deepEqual((await readdir(outputDirectory)).sort(), [`${result.baseName}.json`, `${result.baseName}.md`]);
  const frozen = await readFile(result.jsonPath, "utf8");
  original.notes = "Edited later";
  assert.equal(await readFile(result.jsonPath, "utf8"), frozen);

  const blockedDirectory = await mkdtemp(path.join(os.tmpdir(), "mtd-blocked-"));
  await assert.rejects(
    generateMonthToDateSnapshot({ month: "2026-07", asOfDate: "2026-07-17", outputDirectory: blockedDirectory, dependencies: { ...dependencies, loadPortfolioSettings: async () => ({ portfolios: ["Branden Log"], defaultPortfolio: "Branden Log", portfolioMeta: {} }) } }),
    (error: unknown) => error instanceof MonthToDateSnapshotValidationError && error.code === "AUTHORITATIVE_EQUITY_UNAVAILABLE"
  );
  assert.deepEqual(await readdir(blockedDirectory), []);
});

test("server blocks relevant needs-review rows, missing executions, and insufficient coverage", async () => {
  const baseDependencies = {
    loadWeeklyFocus: async () => createUserDefinedWeeklyFocus({}, new Date("2026-07-12T16:00:00Z")),
    loadPrice: async (symbol: string) => ({ symbol, price: 110, sessionDate: "2026-07-17", timestamp: "2026-07-17T16:00:00-04:00", provider: "test" }),
    now: () => new Date("2026-07-18T01:00:00Z")
  };
  const settings = (coverage: string) => async () => ({ portfolios: ["Branden Log"], defaultPortfolio: "Branden Log", portfolioMeta: { "Branden Log": { currentEquity: 700_000, equityStatementDate: coverage } } });
  await assert.rejects(generateMonthToDateSnapshot({ month: "2026-07", asOfDate: "2026-07-17", writeExports: false, dependencies: { ...baseDependencies, loadTrades: async () => [trade({ customTags: ["Needs review"] })], loadPortfolioSettings: settings("2026-07-17") } }), (error: unknown) => error instanceof MonthToDateSnapshotValidationError && error.code === "BROKER_IMPORT_NEEDS_REVIEW");
  await assert.rejects(generateMonthToDateSnapshot({ month: "2026-07", asOfDate: "2026-07-17", writeExports: false, dependencies: { ...baseDependencies, loadTrades: async () => [trade({ executions: [] })], loadPortfolioSettings: settings("2026-07-17") } }), (error: unknown) => error instanceof MonthToDateSnapshotValidationError && error.code === "BROKER_IMPORT_MISSING_EXECUTIONS");
  await assert.rejects(generateMonthToDateSnapshot({ month: "2026-07", asOfDate: "2026-07-17", writeExports: false, dependencies: { ...baseDependencies, loadTrades: async () => [trade()], loadPortfolioSettings: settings("2026-07-16") } }), (error: unknown) => error instanceof MonthToDateSnapshotValidationError && error.code === "BROKER_IMPORT_DATE_COVERAGE_INSUFFICIENT");
});

test("email attaches JSON and Markdown with the selected period and does not send blocked output", async () => {
  const snapshot = buildMonthToDateSnapshot(snapshotInput([trade()]));
  const messages: Array<Record<string, unknown>> = [];
  const result = await sendMonthToDateSnapshotEmail({
    snapshot,
    markdown: renderMonthToDateSnapshotMarkdown(snapshot),
    baseName: "mtd-test",
    environment: { SNAPSHOT_SMTP_HOST: "smtp.test", SNAPSHOT_SMTP_PORT: "465", SNAPSHOT_SMTP_SECURE: "true", SNAPSHOT_SMTP_USERNAME: "user", SNAPSHOT_SMTP_PASSWORD: "secret", SNAPSHOT_EMAIL_FROM: "from@example.test", SNAPSHOT_EMAIL_TO: "to@example.test" },
    transport: { sendMail: async (message) => { messages.push(message); } }
  });
  assert.equal(result.status, "sent");
  assert.match(String(messages[0].subject), /2026-07 through 2026-07-17/);
  assert.deepEqual((messages[0].attachments as Array<{ filename: string }>).map((item) => item.filename), ["mtd-test.json", "mtd-test.md"]);
  const blocked = buildMonthToDateSnapshot(snapshotInput([trade()], { portfolioMeta: { equityStatementDate: "2026-07-17" } }));
  const blockedResult = await sendMonthToDateSnapshotEmail({
    snapshot: blocked,
    markdown: renderMonthToDateSnapshotMarkdown(blocked),
    baseName: "blocked",
    environment: { SNAPSHOT_SMTP_HOST: "smtp.test", SNAPSHOT_SMTP_PORT: "465", SNAPSHOT_SMTP_USERNAME: "user", SNAPSHOT_SMTP_PASSWORD: "secret", SNAPSHOT_EMAIL_FROM: "from@example.test", SNAPSHOT_EMAIL_TO: "to@example.test" },
    transport: { sendMail: async (message) => { messages.push(message); } }
  });
  assert.equal(blockedResult.status, "not_sent");
  assert.equal(messages.length, 1);
});
