import assert from "node:assert/strict";
import test from "node:test";
import { buildOpenPositionRiskRow } from "../app/components/OpenHeatDashboard";
import type { BrokerPortfolioSnapshot } from "../lib/broker-portfolio-snapshot";
import type { TradeLogEntry } from "../lib/types";

function trade(overrides: Partial<TradeLogEntry> = {}) {
  return {
    id: "trade-1",
    userId: "branden",
    importSource: "cf-statement-pdf",
    importRowKey: "trade-1",
    symbol: "LLY",
    side: "LONG",
    status: "OPEN",
    entryDate: "2026-06-04",
    exitDate: "",
    openTime: "",
    closeTime: "",
    avgEntry: 1164.81,
    exitPrice: 0,
    stopPrice: 1079.96,
    takeProfitPrice: 1287.94,
    shares: 3,
    commission: 0,
    usedMargin: 0,
    risk: 300,
    pnl: 0,
    rMultiple: 0,
    returnPercent: 0,
    daysInTrade: 0,
    setupTags: [],
    mistakeTags: [],
    customTags: [],
    manualGrade: "",
    portfolioTag: "CF_Statement",
    emotion: "",
    tradeQuality: "",
    checklistItems: [],
    notes: "",
    screenshots: [],
    chartLinks: [],
    executions: [],
    hidden: false,
    groupId: "",
    groupRole: "none",
    createdAt: "2026-06-04T00:00:00Z",
    updatedAt: "2026-08-21T00:00:00Z",
    ...overrides
  } satisfies TradeLogEntry;
}

function snapshot(overrides: Partial<BrokerPortfolioSnapshot> = {}): BrokerPortfolioSnapshot {
  return {
    userId: "branden",
    portfolioTag: "CF_Statement",
    coverageDate: "2026-08-21",
    sourceHash: "a".repeat(64),
    sourceFilename: "statement.pdf",
    source: "CF_STATEMENT_PDF",
    importedAt: "2026-08-21T21:00:00Z",
    balance: 700000,
    currentEquity: 700500,
    statementEquity: 700500,
    floatingPnl: 500,
    openPositions: [{
      symbol: "LLY",
      side: "LONG",
      shares: 3,
      entryDate: "2026-06-04",
      entryPrice: 1164.81,
      currentPrice: 1178,
      usedMargin: null,
      stopPrice: 1079.96,
      takeProfitPrice: 1287.94,
      floatingPnl: null,
      commission: null
    }],
    workingOrders: [
      { orderId: "early", orderDate: "2026-08-21", timeValue: "", direction: "Sell", shares: 1, symbol: "LLY", orderType: "STOP", orderPrice: 1132.94 },
      { orderId: "bracket", orderDate: "2026-08-21", timeValue: "", direction: "Sell", shares: 3, symbol: "LLY", orderType: "STOP", orderPrice: 1079.96 },
      { orderId: "target", orderDate: "2026-08-21", timeValue: "", direction: "Sell", shares: 1, symbol: "LLY", orderType: "LIMIT", orderPrice: 1287.94 }
    ],
    ...overrides
  };
}

test("quantity-aware linked stops use one staged share and two bracket shares", () => {
  const tradeRow = trade();
  const broker = snapshot();
  const row = buildOpenPositionRiskRow(
    tradeRow,
    { price: 1179.11, date: "2026-08-21" },
    broker.currentEquity,
    broker.openPositions[0],
    broker
  );

  assert.equal(row.status, "ready");
  assert.equal(row.stopLabel, "1 @ 1,132.94; 2 @ 1,079.96");
  assert.equal(Number(row.dollarRisk?.toFixed(2)), 244.47);
  assert.doesNotMatch(row.stopLabel, /1,287\.94/);
});

test("statement price is used when the completed-session market close is unavailable", () => {
  const tradeRow = trade();
  const broker = snapshot();
  const row = buildOpenPositionRiskRow(
    tradeRow,
    { price: null, date: "", error: "market unavailable" },
    broker.currentEquity,
    broker.openPositions[0],
    broker
  );

  assert.equal(row.currentPrice, 1178);
  assert.equal(row.priceDate, "2026-08-21");
  assert.notEqual(row.floatingPnl, null);
});

test("Trade Log planned risk remains a fallback without inventing a current price", () => {
  const tradeRow = trade({ symbol: "HPE", avgEntry: 54.79, shares: 20, stopPrice: 0, risk: 146 });
  const row = buildOpenPositionRiskRow(
    tradeRow,
    { price: null, date: "", error: "market unavailable" },
    700000,
    undefined,
    undefined
  );

  assert.equal(row.currentPrice, null);
  assert.equal(row.positionValue, null);
  assert.equal(row.floatingPnl, null);
  assert.equal(row.status, "fallback");
  assert.equal(row.dollarRisk, 146);
});

test("floating P&L uses FIFO remaining lots after adds and trims", () => {
  const tradeRow = trade({
    symbol: "CPRT",
    avgEntry: 30.55,
    shares: 25,
    stopPrice: 29,
    risk: 200,
    executions: [
      { id: "entry-1", type: "ENTRY", date: "2026-08-14", time: "09:35:00", side: "LONG", shares: 55, price: 29.96, pnl: 0, commission: 0, source: "broker", sourceKey: "1" },
      { id: "entry-2", type: "ENTRY", date: "2026-08-20", time: "10:00:00", side: "LONG", shares: 25, price: 31.846, pnl: 0, commission: 0, source: "broker", sourceKey: "2" },
      { id: "exit-1", type: "EXIT", date: "2026-08-21", time: "10:30:00", side: "LONG", shares: 55, price: 34.02, pnl: 223.3, commission: 0, source: "broker", sourceKey: "3" }
    ]
  });

  const row = buildOpenPositionRiskRow(
    tradeRow,
    { price: 33.8, date: "2026-08-21" },
    704420,
    undefined,
    undefined
  );

  assert.equal(row.status, "ready");
  assert.equal(row.stopLabel, "25 @ 29");
  assert.equal(Number(row.entryPrice.toFixed(2)), 31.85);
  assert.equal(Number(row.floatingPnl?.toFixed(2)), 48.85);
});

test("saved Trade Log stops remain visible when authenticated review trades are used", () => {
  const tradeRow = trade({ symbol: "KARO", avgEntry: 63.14, shares: 20, stopPrice: 56.29, risk: 137 });
  const row = buildOpenPositionRiskRow(
    tradeRow,
    { price: 63.96, date: "2026-08-21" },
    704420,
    undefined,
    undefined
  );

  assert.equal(row.status, "ready");
  assert.equal(row.stopLabel, "20 @ 56.29");
  assert.equal(Number(row.dollarRisk?.toFixed(2)), 153.4);
});
