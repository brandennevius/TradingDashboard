import assert from "node:assert/strict";
import test from "node:test";
import { buildCfTradesFromExecutionHistory, type ParsedOpenPositionRow } from "../lib/cf-statement";
import { cfImportTradesEquivalent, replaceActiveWorkingOrders, runAtomicCfImport, type CfWorkingOrderMetadata } from "../lib/cf-import-idempotency";
import type { TradeLogEntry, TradeLogInput } from "../lib/types";

function importedTrade(): TradeLogEntry {
  return {
    id: "branden-lly", userId: "branden", importSource: "cf-statement-pdf", importRowKey: "lly-row", symbol: "LLY", side: "LONG", status: "OPEN",
    entryDate: "2026-07-15", exitDate: "", openTime: "10:00:00", closeTime: "", avgEntry: 1100, exitPrice: 0,
    stopPrice: 1079.96, takeProfitPrice: 1287.94, shares: 3, commission: 1.25, usedMargin: 3300, risk: 300,
    pnl: 25, rMultiple: 0.08, returnPercent: 0, daysInTrade: 2, setupTags: ["Breakout"], mistakeTags: [], customTags: ["CF Statement"],
    manualGrade: "A", portfolioTag: "CF_Statement", emotion: "", tradeQuality: "", checklistItems: [], notes: "Keep the linked bracket.",
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
