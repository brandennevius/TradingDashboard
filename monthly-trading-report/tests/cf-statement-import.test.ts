import assert from "node:assert/strict";
import test from "node:test";
import { buildCfTradesFromExecutionHistory, parseCfStatementText, type ParsedOpenPositionRow } from "../lib/cf-statement";
import { applyManualFieldsToCfStatementTrade, reconcileCfStatementLifecycles } from "../lib/cf-import-reconciliation";
import { cfImportTradesEquivalent, mergeCfExecutionHistory, replaceActiveWorkingOrders, runAtomicCfImport, type CfWorkingOrderMetadata } from "../lib/cf-import-idempotency";
import type { TradeExecution, TradeLogEntry, TradeLogInput } from "../lib/types";

function importedTrade(): TradeLogEntry {
  return {
    id: "branden-lly", userId: "branden", importSource: "cf-statement-pdf", importRowKey: "lly-row", symbol: "LLY", side: "LONG", status: "OPEN",
    entryDate: "2026-07-15", exitDate: "", openTime: "10:00:00", closeTime: "", avgEntry: 1100, exitPrice: 0,
    stopPrice: 1079.96, takeProfitPrice: 1287.94, shares: 3, commission: 1.25, usedMargin: 3300, risk: 300,
    pnl: 25, rMultiple: 0.08, returnPercent: 0, daysInTrade: 2, setupTags: ["Breakout"], mistakeTags: [], customTags: ["CF Statement"],
    manualGrade: "A", portfolioTag: "CF_Statement", emotion: "", tradeQuality: "", checklistItems: [], notes: "Keep the linked bracket.",
    reviewSections: { setup: "Breakout", entry: "Waited for confirmation", exit: "", didRight: "Sized correctly", didWrong: "", general: "" },
    screenshots: ["chart.png"], chartLinks: ["https://example.test/lly"], executions: [
      { id: "entry", type: "ENTRY", date: "2026-07-15", time: "10:00:00", side: "LONG", shares: 3, price: 1100, pnl: 0, commission: 1.25, source: "broker", sourceKey: "entry" }
    ], hidden: false, groupId: "", groupRole: "none", createdAt: "2026-07-17T16:00:00Z", updatedAt: "2026-07-17T16:00:00Z"
  };
}

function llyOrders(): CfWorkingOrderMetadata[] {
  return [
    { orderId: "lly-early-stop", orderDate: "2026-07-17", timeValue: "10:00:00", direction: "Sell", shares: 1, symbol: "LLY", orderType: "STOP", orderPrice: 1132.94 },
    { orderId: "lly-bracket-stop", orderDate: "2026-07-17", timeValue: "10:00:01", direction: "Sell", shares: 3, symbol: "LLY", orderType: "STOP", orderPrice: 1079.96 },
    { orderId: "lly-profit-limit", orderDate: "2026-07-17", timeValue: "10:00:02", direction: "Sell", shares: 1, symbol: "LLY", orderType: "LIMIT", orderPrice: 1287.94 }
  ];
}

function froCfStatementTrade(): TradeLogInput {
  return {
    userId: "branden", importSource: "cf-statement-pdf", importRowKey: "cf-open-position:FRO|LONG|2026-09-01",
    symbol: "FRO", side: "LONG", status: "OPEN", entryDate: "2026-09-01", exitDate: "", openTime: "11:31",
    closeTime: "", avgEntry: 44.8885, exitPrice: 0, stopPrice: 43.73, takeProfitPrice: 50.28, shares: 100,
    commission: 0.71, usedMargin: 4542, risk: 0, pnl: 0, rMultiple: 0, returnPercent: 0, daysInTrade: 2,
    setupTags: [], mistakeTags: [], customTags: ["CF Statement", "Open Position"], manualGrade: "", portfolioTag: "CF_Statement",
    emotion: "", tradeQuality: "", checklistItems: [], notes: "", reviewSections: undefined, screenshots: [], chartLinks: [],
    executions: [
      { id: "fro-entry", type: "ENTRY", date: "2026-09-01", time: "11:31:37.015", side: "LONG", shares: 54, price: 44.37, pnl: 0, commission: 0.1, source: "FRO", sourceKey: "fro-entry" }
    ]
  };
}

function froManualReviewedTrade(): TradeLogEntry {
  return {
    id: "manual-fro", userId: "branden", importSource: "", importRowKey: "", symbol: "FRO", side: "LONG", status: "OPEN",
    entryDate: "2026-09-01", exitDate: "", openTime: "", closeTime: "", avgEntry: 44.89, exitPrice: 0, stopPrice: 43.73,
    takeProfitPrice: 50.28, shares: 100, commission: 0, usedMargin: 0, risk: 5, pnl: 0, rMultiple: 0, returnPercent: 0,
    daysInTrade: 2, setupTags: ["CANSLIM"], mistakeTags: [], customTags: ["Swing candidate"], manualGrade: "A",
    portfolioTag: "CF_Statement", emotion: "", tradeQuality: "", checklistItems: [], notes: "Reviewed FRO setup.",
    reviewSections: { setup: "Breakout setup", entry: "", exit: "", didRight: "", didWrong: "", general: "" },
    screenshots: [], chartLinks: [], executions: [], hidden: false, groupId: "", groupRole: "none",
    createdAt: "2026-09-01T16:00:00.000Z", updatedAt: "2026-09-01T16:00:00.000Z"
  };
}

test("CF open positions adopt the unique matching reviewed manual trade instead of creating a duplicate", () => {
  const rebuilt = froCfStatementTrade();
  const reviewedManual = froManualReviewedTrade();

  const reconciled = applyManualFieldsToCfStatementTrade(rebuilt, [reviewedManual]);

  assert.equal(reconciled.id, reviewedManual.id);
  assert.equal(reconciled.risk, 5);
  assert.deepEqual(reconciled.setupTags, ["CANSLIM"]);
  assert.equal(reconciled.manualGrade, "A");
  assert.equal(reconciled.notes, "Reviewed FRO setup.");
  assert.deepEqual(reconciled.reviewSections, reviewedManual.reviewSections);
  assert.deepEqual(reconciled.customTags, ["CF Statement", "Open Position", "Swing candidate"]);
  assert.equal(reconciled.importRowKey, rebuilt.importRowKey);
  assert.equal(reconciled.avgEntry, 44.8885);
  assert.equal(reconciled.shares, 100);
});

test("CF open-position adoption is skipped when multiple manual matches exist", () => {
  const rebuilt = froCfStatementTrade();
  const first = froManualReviewedTrade();
  const second = { ...froManualReviewedTrade(), id: "manual-fro-two", openTime: "" };

  const reconciled = applyManualFieldsToCfStatementTrade(rebuilt, [first, second]);

  assert.equal(reconciled.id, undefined);
  assert.equal(reconciled.risk, rebuilt.risk);
  assert.deepEqual(reconciled.setupTags, []);
  assert.equal(reconciled.manualGrade, "");
});

test("lifecycle identity does not depend on review completion", () => {
  const unreviewed = {
    ...froManualReviewedTrade(),
    id: "manual-fro-unreviewed",
    risk: 0,
    setupTags: [],
    manualGrade: "",
    notes: "",
    reviewSections: undefined
  };

  const result = reconcileCfStatementLifecycles([froCfStatementTrade()], [unreviewed]);

  assert.equal(result.ambiguities.length, 0);
  assert.equal(result.trades[0].id, unreviewed.id);
  assert.deepEqual(result.adoptedNonCfTradeIds, [unreviewed.id]);
  assert.equal(result.decisions[0].action, "OPEN_CONTINUATION");
});

test("a reviewed duplicate is consolidated even when an exact CF row already exists", () => {
  const rebuilt = froCfStatementTrade();
  const exactCf = {
    ...froManualReviewedTrade(),
    id: "existing-cf-fro",
    importSource: "cf-statement-pdf",
    importRowKey: rebuilt.importRowKey,
    setupTags: [],
    manualGrade: "",
    notes: "",
    executions: rebuilt.executions || []
  };
  const reviewed = froManualReviewedTrade();

  const result = reconcileCfStatementLifecycles([rebuilt], [exactCf, reviewed]);

  assert.equal(result.trades[0].id, reviewed.id);
  assert.equal(result.trades[0].manualGrade, "A");
  assert.equal(result.trades[0].notes, "Reviewed FRO setup.");
  assert.deepEqual(result.adoptedNonCfTradeIds, [reviewed.id]);
  assert.equal(result.decisions[0].action, "OPEN_CONTINUATION");
});

test("a carryover date change keeps the only open same-symbol lifecycle", () => {
  const manual = { ...froManualReviewedTrade(), entryDate: "2026-08-29" };
  const result = reconcileCfStatementLifecycles([froCfStatementTrade()], [manual]);

  assert.equal(result.trades[0].id, manual.id);
  assert.equal(result.decisions[0].action, "OPEN_CONTINUATION");
});

test("close and reopen cycles are not merged by ticker alone", () => {
  const closed = {
    ...froCfStatementTrade(),
    status: "WIN" as const,
    importRowKey: "cf-cycle:FRO:LONG:2026-08-01:first",
    entryDate: "2026-08-01",
    exitDate: "2026-08-10",
    executions: [
      { id: "old-entry", type: "ENTRY" as const, date: "2026-08-01", time: "10:00:00", side: "LONG" as const, shares: 10, price: 40, pnl: 0, commission: 0, source: "FRO", sourceKey: "cf-transaction:old-entry" },
      { id: "old-exit", type: "EXIT" as const, date: "2026-08-10", time: "10:00:00", side: "LONG" as const, shares: 10, price: 42, pnl: 20, commission: 0, source: "FRO", sourceKey: "cf-transaction:old-exit" }
    ]
  };
  const reopened = froCfStatementTrade();
  const staleManual = { ...froManualReviewedTrade(), entryDate: "2026-08-01" };

  const result = reconcileCfStatementLifecycles([closed, reopened], [staleManual]);

  assert.equal(result.trades[0].id, undefined);
  assert.equal(result.trades[1].id, undefined);
  assert.deepEqual(result.decisions.map((decision) => decision.action), ["NEW_LIFECYCLE", "NEW_LIFECYCLE"]);
});

test("ambiguous open rows block automatic lifecycle adoption", () => {
  const first = froManualReviewedTrade();
  const second = { ...froManualReviewedTrade(), id: "manual-fro-two" };
  const result = reconcileCfStatementLifecycles([froCfStatementTrade()], [first, second]);

  assert.equal(result.ambiguities.length, 1);
  assert.equal(result.ambiguities[0].action, "AMBIGUOUS");
  assert.deepEqual(result.ambiguities[0].candidateTradeIds, ["manual-fro", "manual-fro-two"]);
  assert.deepEqual(result.adoptedNonCfTradeIds, []);
});

test("replaying the same July 17 import leaves broker trades and manual fields unchanged", () => {
  const original = importedTrade();
  const incoming = structuredClone(original) as TradeLogInput;
  const state = { trades: [] as TradeLogEntry[], orders: [] as CfWorkingOrderMetadata[], equity: 100_000, tradeWrites: 0 };

  const importStatement = () => {
    if (!cfImportTradesEquivalent(state.trades, [incoming])) {
      state.trades = [structuredClone(original)];
      state.tradeWrites += 1;
    }
    state.orders = replaceActiveWorkingOrders(llyOrders());
  };
  importStatement();
  const afterFirst = structuredClone(state);
  importStatement();

  assert.deepEqual(state.trades, afterFirst.trades);
  assert.equal(state.tradeWrites, 1);
  assert.equal(state.trades.length, 1);
  assert.deepEqual(state.trades[0].executions, original.executions);
  assert.equal(state.trades[0].commission, original.commission);
  assert.equal(state.trades[0].pnl, original.pnl);
  assert.equal(state.trades[0].shares, original.shares);
  assert.equal(state.trades[0].notes, original.notes);
  assert.deepEqual(state.trades[0].reviewSections, original.reviewSections);
  assert.equal(state.trades[0].manualGrade, original.manualGrade);
  assert.deepEqual(state.trades[0].setupTags, original.setupTags);
  assert.equal(state.trades[0].stopPrice, original.stopPrice);
  assert.equal(state.trades[0].risk, original.risk);
  assert.equal(state.equity, afterFirst.equity);
  assert.deepEqual(state.orders, afterFirst.orders);
});

test("the CF execution-history rebuild is idempotent for executions, commissions, P&L, and open quantity", () => {
  const entry = { ...importedTrade().executions[0], source: "LLY" };
  const openPositions: ParsedOpenPositionRow[] = [{
    entryDate: "2026-07-15", timeValue: "10:00:00", side: "LONG", shares: 3, symbol: "LLY", entryPrice: 1100,
    currentPrice: 1179.11, usedMargin: 3300, stopPrice: 1079.96, takeProfitPrice: 1287.94, floatingPnl: 237.33, commission: 1.25
  }];
  const first = buildCfTradesFromExecutionHistory([entry], openPositions, llyOrders(), "branden", "CF_Statement");
  const replayHistory = [...first, ...first].flatMap((trade) => (trade.executions || []).map((execution) => ({ ...execution, source: trade.symbol })));
  const second = buildCfTradesFromExecutionHistory(replayHistory, openPositions, llyOrders(), "branden", "CF_Statement");
  assert.equal(first.length, 1);
  assert.equal(second.length, 1);
  assert.deepEqual(second[0].executions, first[0].executions);
  assert.equal(second[0].commission, first[0].commission);
  assert.equal(second[0].pnl, first[0].pnl);
  assert.equal(second[0].shares, first[0].shares);
  assert.equal(second[0].stopPrice, first[0].stopPrice);
  assert.equal(second[0].takeProfitPrice, first[0].takeProfitPrice);
});

test("the current statement replaces matching transaction IDs instead of accumulating P&L on every replay", () => {
  const currentDxcm = {
    ...importedTrade().executions[0], id: "dxcm-current", type: "EXIT" as const, side: "LONG" as const,
    sourceKey: "cf-transaction:dxcm-close", source: "DXCM", shares: 10, price: 74, pnl: -3, commission: 1
  };
  const previouslyAccumulated = { ...currentDxcm, id: "dxcm-stored", pnl: -6, commission: 2 };
  const olderTransaction = { ...currentDxcm, id: "older", sourceKey: "cf-transaction:older-close", pnl: -119.4 };
  const once = mergeCfExecutionHistory([olderTransaction, previouslyAccumulated], [currentDxcm]);
  const twice = mergeCfExecutionHistory(once, [currentDxcm]);
  const threeTimes = mergeCfExecutionHistory(twice, [currentDxcm]);
  assert.equal(once.length, 2);
  assert.equal(once.find((execution) => execution.sourceKey === currentDxcm.sourceKey)?.pnl, -3);
  assert.equal(once.find((execution) => execution.sourceKey === currentDxcm.sourceKey)?.commission, 1);
  assert.deepEqual(twice, once);
  assert.deepEqual(threeTimes, once);
  assert.equal(threeTimes.reduce((sum, execution) => sum + execution.pnl, 0), -122.4);
});

test("broad CF statement replay replaces stale prior broker history for symbols closed by statement end", () => {
  const staleJanuaryEntry: TradeExecution = {
    id: "jan-entry", type: "ENTRY", date: "2026-01-26", time: "13:01:00", side: "LONG",
    shares: 17, price: 28656.39, pnl: 0, commission: 0, source: ".US100", sourceKey: "cf-transaction:jan-entry"
  };
  const currentExitOnly: TradeExecution = {
    id: "sep-exit", type: "EXIT", date: "2026-09-01", time: "14:42:52", side: "LONG",
    shares: 0.2, price: 28983.05, pnl: 0, commission: 0, source: ".US100", sourceKey: "cf-transaction:sep-exit"
  };

  const merged = mergeCfExecutionHistory([staleJanuaryEntry], [currentExitOnly], {
    statementStartDate: "2026-07-01",
    statementEndDate: "2026-09-01",
    currentStatementSymbols: [".US100"],
    currentOpenSymbols: []
  });

  assert.deepEqual(merged, [currentExitOnly]);
});

test("broad CF statement replay replaces stale prior broker history even when symbol remains open", () => {
  const staleJanuaryEntry: TradeExecution = {
    id: "jan-entry", type: "ENTRY", date: "2026-01-26", time: "13:01:00", side: "LONG",
    shares: 17, price: 28656.39, pnl: 0, commission: 0, source: ".US100", sourceKey: "cf-transaction:jan-entry"
  };
  const currentShortEntry: TradeExecution = {
    id: "sep-short", type: "ENTRY", date: "2026-09-01", time: "14:42:52", side: "SHORT",
    shares: 0.2, price: 28983.05, pnl: 0, commission: 0, source: ".US100", sourceKey: "cf-transaction:sep-short"
  };

  const merged = mergeCfExecutionHistory([staleJanuaryEntry], [currentShortEntry], {
    statementStartDate: "2026-07-01",
    statementEndDate: "2026-09-01",
    currentStatementSymbols: [".US100"],
    currentOpenSymbols: [".US100"]
  });

  assert.deepEqual(merged, [currentShortEntry]);
});

test("narrow CF statement import keeps prior broker history for same-symbol carryover matching", () => {
  const priorEntry: TradeExecution = {
    id: "prior-entry", type: "ENTRY", date: "2026-08-29", time: "18:05:00", side: "LONG",
    shares: 0.8, price: 29000, pnl: 0, commission: 0, source: ".US100", sourceKey: "cf-transaction:prior-entry"
  };
  const currentExit: TradeExecution = {
    id: "current-exit", type: "EXIT", date: "2026-09-01", time: "14:42:52", side: "LONG",
    shares: 0.8, price: 28983.05, pnl: -13.56, commission: 0, source: ".US100", sourceKey: "cf-transaction:current-exit"
  };

  const merged = mergeCfExecutionHistory([priorEntry], [currentExit], {
    statementStartDate: "2026-09-01",
    statementEndDate: "2026-09-01",
    currentStatementSymbols: [".US100"],
    currentOpenSymbols: []
  });

  assert.deepEqual(merged, [priorEntry, currentExit]);
});

test("working orders replace by current statement state without duplicates or stale removed orders", () => {
  const duplicated = replaceActiveWorkingOrders([...llyOrders(), { ...llyOrders()[0] }]);
  assert.equal(duplicated.length, 3);
  const changed = replaceActiveWorkingOrders([
    { ...llyOrders()[1], orderPrice: 1085 },
    llyOrders()[2]
  ]);
  assert.equal(changed.length, 2);
  assert.equal(changed.find((order) => order.orderId === "lly-bracket-stop")?.orderPrice, 1085);
  assert(!changed.some((order) => order.orderId === "lly-early-stop"));
});

test("a working-order persistence failure rolls back the entire import transaction", async () => {
  const durable = { trades: [importedTrade()], orders: llyOrders() };
  let pending = structuredClone(durable);
  const transaction = {
    begin: async () => { pending = structuredClone(durable); },
    commit: async () => { durable.trades = pending.trades; durable.orders = pending.orders; },
    rollback: async () => { pending = structuredClone(durable); }
  };
  await assert.rejects(runAtomicCfImport(transaction, async () => {
    pending.trades = [{ ...importedTrade(), shares: 2 }];
    throw new Error("working-order metadata write failed");
  }), /metadata write failed/);
  assert.equal(durable.trades[0].shares, 3);
  assert.deepEqual(durable.orders, llyOrders());
});

test("statement coverage uses the period end instead of an older working-order date", () => {
  const parsed = parseCfStatementText([
    "21/07/2026 09:31 12659050 Sell 2.00 LLY STOP 1,130.22 1,164.09 GTC — —",
    "Created 22/07/2026 17:14 GMT-4",
    "21 Jul 2026 18:00 — 22 Jul 2026 17:14",
    "1659201:391119 22/07/2026 09:30:05.065 Sell 50.00 RELY 23.50 14070296 -67.00 0.35"
  ].join("\n"), "branden", "CF_Statement");

  assert.equal(parsed.equityStatementDate, "2026-07-22");
});

test("statement coverage falls back to the labeled Created date", () => {
  const parsed = parseCfStatementText([
    "21/07/2026 09:31 12659050 Sell 2.00 LLY STOP 1,130.22 1,164.09 GTC — —",
    "Created 22/07/2026 17:14 GMT-4"
  ].join("\n"), "branden", "CF_Statement");

  assert.equal(parsed.equityStatementDate, "2026-07-22");
});

test("statement coverage handles a period ending in a new month", () => {
  const parsed = parseCfStatementText(
    "31 Jul 2026 18:00 - 01 Aug 2026 17:14",
    "branden",
    "CF_Statement"
  );

  assert.equal(parsed.equityStatementDate, "2026-08-01");
});

test("statement coverage does not guess from transaction or working-order dates", () => {
  const parsed = parseCfStatementText([
    "21/07/2026 09:31 12659050 Sell 2.00 LLY STOP 1,130.22 1,164.09 GTC — —",
    "1659201:391119 22/07/2026 09:30:05.065 Sell 50.00 RELY 23.50 14070296 -67.00 0.35"
  ].join("\n"), "branden", "CF_Statement");

  assert.equal(parsed.equityStatementDate, "");
});

test("open-position parser handles glued CF statement numeric columns", () => {
  const parsed = parseCfStatementText(
    "01/09/2026 13:25 Sell 0.80 .US100 29,046.00 29,086.60 2,326.87 ‑23,269.2829,218.40 28,771.49 ‑32.48 0.00 —",
    "branden",
    "CF_Statement"
  );

  assert.equal(parsed.openPositions.length, 1);
  assert.deepEqual(parsed.openPositions[0], {
    entryDate: "2026-09-01",
    timeValue: "13:25",
    side: "SHORT",
    shares: 0.8,
    symbol: ".US100",
    entryPrice: 29046,
    currentPrice: 29086.6,
    usedMargin: 2326.87,
    stopPrice: 29218.4,
    takeProfitPrice: 28771.49,
    floatingPnl: -32.48,
    commission: 0
  });
});

test("current futures shorts from open positions do not merge into old long cycles", () => {
  const parsed = parseCfStatementText([
    "01 Jul 2026 00:00 - 01 Sep 2026 17:14",
    "01/09/2026 13:25 Sell 0.80 .US100 29,046.00 29,086.60 2,326.87 ‑23,269.2829,218.40 28,771.49 ‑32.48 0.00 —",
    "1659201:708018 21/08/2026 14:30:44.808 Buy 1.30 .US100 29278.35 1001 — 0.00",
    "1659151:485827 21/08/2026 14:49:02.370 Buy 0.10 .US100 29297.65 1002 — 0.00",
    "1659151:485885 21/08/2026 14:54:09.703 Buy 0.10 .US100 29312.15 1003 — 0.00",
    "1659201:708224 21/08/2026 14:57:31.698 Buy 0.10 .US100 29316.15 1004 — 0.00",
    "1659151:486004 21/08/2026 15:08:02.102 Buy 0.10 .US100 29322.45 1005 — 0.00",
    "1659201:708360 21/08/2026 15:11:44.499 Sell 1.70 .US100 29307.75 1006 36.48 0.00",
    "1834251:35568 26/08/2026 14:44:52.599 Buy 1.70 .US100 29227.75 2001 — 0.00",
    "1834251:35574 26/08/2026 14:45:05.498 Buy 1.30 .US100 29229.05 2002 — 0.00",
    "1834251:35587 26/08/2026 14:45:31.496 Buy 0.30 .US100 29241.75 2003 — 0.00",
    "1834251:35604 26/08/2026 14:47:03.908 Buy 0.10 .US100 29253.05 2004 — 0.00",
    "1834251:35801 26/08/2026 15:01:51.633 Buy 0.10 .US100 29269.05 2005 — 0.00",
    "1834251:37029 26/08/2026 15:49:08.528 Buy 0.10 .US100 29266.25 2006 — 0.00",
    "1834251:37050 26/08/2026 15:53:47.303 Sell 3.60 .US100 29230.55 2007 -6.32 0.00",
    "1834201:19764 26/08/2026 15:56:10.592 Buy 2.00 .US100 29237.15 2008 — 0.00",
    "1834251:37068 26/08/2026 15:57:27.209 Sell 2.00 .US100 29238.45 2009 2.60 0.00",
    "1834251:37099 26/08/2026 16:00:26.204 Buy 3.00 .US100 29222.80 2010 — 0.00",
    "1834251:37103 26/08/2026 16:00:32.065 Sell 3.00 .US100 29226.90 2011 12.30 0.00",
    "1834251:85285 01/09/2026 13:25:46.769 Sell 0.50 .US100 29074.85 3001 — 0.00",
    "1834201:49171 01/09/2026 14:33:59.034 Sell 0.10 .US100 29027.65 3002 — 0.00",
    "1834251:85932 01/09/2026 14:42:52.360 Sell 0.20 .US100 28983.05 3003 — 0.00"
  ].join("\n"), "branden", "CF_Statement");

  const us100Trades = parsed.trades.filter((trade) => trade.symbol === ".US100");
  const augustTwentyFirst = us100Trades.find((trade) => trade.entryDate === "2026-08-21");
  const septemberFirstShort = us100Trades.find((trade) => trade.entryDate === "2026-09-01" && trade.side === "SHORT");

  assert(augustTwentyFirst);
  assert.equal(augustTwentyFirst.exitDate, "2026-08-21");
  assert.equal(augustTwentyFirst.shares, 1.7);
  assert(!us100Trades.some((trade) => trade.entryDate < "2026-08-21"));
  assert(septemberFirstShort);
  assert.equal(septemberFirstShort.status, "OPEN");
  assert.equal(septemberFirstShort.shares, 0.8);
  assert(Math.abs(septemberFirstShort.avgEntry - 29046) < 0.000001);
  assert.equal(septemberFirstShort.stopPrice, 29218.4);
  assert.equal(septemberFirstShort.takeProfitPrice, 28771.49);
  assert.deepEqual(
    septemberFirstShort.executions?.map((execution) => execution.shares),
    [0.5, 0.1, 0.2]
  );
});

test("settled unmatched closing rows do not become fake open reverse positions", () => {
  const parsed = parseCfStatementText([
    "01 Jan 2026 00:00 - 01 Sep 2026 17:14",
    "1156051:223168 14/01/2026 16:05:28.734 Sell 11.00 UPS 107.19 7381829 81.62 0.08",
    "1156051:236428 15/01/2026 11:03:56.279 Buy 19.00 UPS 108.05 7420840 — 0.13"
  ].join("\n"), "branden", "CF_Statement");

  const upsTrades = parsed.trades.filter((trade) => trade.symbol === "UPS");
  assert.equal(upsTrades.filter((trade) => trade.status === "OPEN").length, 0);
  assert(upsTrades.some((trade) => trade.customTags.includes("Needs review") && trade.status !== "OPEN"));
});
