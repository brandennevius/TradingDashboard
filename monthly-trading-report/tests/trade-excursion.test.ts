import assert from "node:assert/strict";
import test from "node:test";
import { calculateTradeExcursion, excursionInstrument, tradeExcursionInputHash } from "../lib/trade-excursion";
import { fetchTradeIntradayBars } from "../lib/trade-excursion-market-data";
import type { TradeExecution, TradeLogEntry } from "../lib/types";

function execution(id: string, type: "ENTRY" | "EXIT", time: string, shares: number, price: number): TradeExecution {
  return { id, type, date: "2026-09-01", time, side: "LONG", shares, price, pnl: 0, commission: 0, source: "fixture", sourceKey: id };
}

function trade(overrides: Partial<TradeLogEntry> = {}): TradeLogEntry {
  return {
    id: "trade-1", userId: "branden", importSource: "cf-statement-pdf", importRowKey: "row-1", symbol: "FRO",
    side: "LONG", status: "WIN", entryDate: "2026-09-01", exitDate: "2026-09-01", openTime: "09:30:00", closeTime: "12:00:00",
    avgEntry: 102.5, exitPrice: 109, stopPrice: 95, takeProfitPrice: 0, shares: 20, commission: 0, usedMargin: 0,
    risk: 100, pnl: 180, rMultiple: 1.8, returnPercent: 0, daysInTrade: 0, setupTags: [], mistakeTags: [], customTags: [],
    manualGrade: "", portfolioTag: "CF Statement", emotion: "", tradeQuality: "", checklistItems: [], notes: "", screenshots: [], chartLinks: [],
    executions: [
      execution("e1", "ENTRY", "09:30:00", 10, 100), execution("e2", "ENTRY", "10:00:00", 10, 105),
      execution("x1", "EXIT", "11:00:00", 10, 110), execution("x2", "EXIT", "12:00:00", 10, 108)
    ], hidden: false, groupId: "", groupRole: "none", createdAt: "2026-09-01T00:00:00Z", updatedAt: "2026-09-01T00:00:00Z",
    ...overrides
  };
}

test("calculates execution-aware MFE and MAE across adds and FIFO trims", () => {
  const result = calculateTradeExcursion(trade(), [
    { time: "2026-09-01 09:30:00", open: 100, high: 103, low: 98, close: 101 },
    { time: "2026-09-01 10:00:00", open: 105, high: 112, low: 104, close: 110 },
    { time: "2026-09-01 11:00:00", open: 110, high: 115, low: 109, close: 111 },
    { time: "2026-09-01 12:00:00", open: 108, high: 109, low: 106, close: 108 }
  ], { asOf: "2026-09-01" });

  assert.equal(result.status, "AVAILABLE");
  assert.equal(result.mfeDollars, 250);
  assert.equal(result.maeDollars, -20);
  assert.equal(result.mfeR, 2.5);
  assert.equal(result.maeR, -0.2);
  assert.equal(result.mfeTimestamp, "2026-09-01 11:00:00");
});

test("inverts favorable and adverse prices for short trades", () => {
  const shortTrade = trade({
    side: "SHORT", avgEntry: 100, shares: 5, risk: 50,
    executions: [
      { ...execution("e1", "ENTRY", "09:30:00", 5, 100), side: "SHORT" },
      { ...execution("x1", "EXIT", "10:00:00", 5, 95), side: "SHORT" }
    ]
  });
  const result = calculateTradeExcursion(shortTrade, [
    { time: "2026-09-01 09:30:00", open: 100, high: 104, low: 90, close: 95 },
    { time: "2026-09-01 10:00:00", open: 95, high: 97, low: 93, close: 95 }
  ], { asOf: "2026-09-01" });
  assert.equal(result.mfeDollars, 50);
  assert.equal(result.maeDollars, -20);
  assert.equal(result.mfeR, 1);
});

test("maps index CFDs to ETF proxies and labels results as estimates", () => {
  const mappedTrade = trade({
    symbol: ".US500", avgEntry: 7500, shares: 1, risk: 100,
    executions: [execution("e1", "ENTRY", "09:30:00", 1, 7500), execution("x1", "EXIT", "10:00:00", 1, 7600)]
  });
  const instrument = excursionInstrument(mappedTrade.symbol);
  assert.equal(instrument.providerSymbol, "SPY");
  const result = calculateTradeExcursion(mappedTrade, [
    { time: "2026-09-01 09:30:00", open: 5000, high: 5100, low: 4950, close: 5000 },
    { time: "2026-09-01 10:00:00", open: 5060, high: 5080, low: 5050, close: 5066.67 }
  ], { instrument, asOf: "2026-09-01" });
  assert.equal(result.status, "ESTIMATED_PROXY");
  assert.equal(result.priceScale, 1.5);
  assert.equal(result.mfeDollars, 150);
  assert.equal(result.maeDollars, -75);
});

test("keeps dollar excursions but not R when planned risk is missing", () => {
  const result = calculateTradeExcursion(trade({ risk: 0 }), [
    { time: "2026-09-01 09:30:00", open: 100, high: 103, low: 98, close: 101 }
  ], { asOf: "2026-09-01" });
  assert.equal(result.mfeDollars, 30);
  assert.equal(result.maeDollars, -20);
  assert.equal(result.mfeR, null);
  assert.equal(result.maeR, null);
});

test("rejects malformed or absent market bars rather than returning zero", () => {
  const result = calculateTradeExcursion(trade(), [], { asOf: "2026-09-01" });
  assert.equal(result.status, "UNAVAILABLE");
  assert.equal(result.mfeDollars, null);
  assert.equal(result.maeDollars, null);
});

test("execution changes invalidate the excursion input hash", () => {
  const original = trade();
  const changed = trade({ executions: original.executions.map((item, index) => index === 0 ? { ...item, price: item.price + 1 } : item) });
  assert.notEqual(tradeExcursionInputHash(original, "2026-09-01"), tradeExcursionInputHash(changed, "2026-09-01"));
});

test("millisecond executions remain ordered and do not open at midnight", () => {
  const sample=trade({executions:[execution("x","EXIT","10:05:00.100",10,102),execution("e","ENTRY","10:00:00.100",10,100)]});
  const result=calculateTradeExcursion(sample,[
    {time:"2026-09-01 09:30:00",open:50,high:1000,low:1,close:100},
    {time:"2026-09-01 10:00:00",open:100,high:900,low:2,close:100},
    {time:"2026-09-01 10:05:00",open:101,high:103,low:99,close:102},
    {time:"2026-09-01 10:10:00",open:102,high:800,low:1,close:102}
  ]);
  assert.equal(result.status,"AVAILABLE");
  assert.equal(result.mfeDollars,30);
  assert.equal(result.maeDollars,-10);
  assert.equal(result.barsEvaluated,1);
});

test("fractional zero at a bar boundary equals whole seconds",()=>{
  const bars=[{time:"2026-09-01 09:30:00",open:100,high:103,low:98,close:101}];
  const base=trade({executions:[execution("e","ENTRY","09:30:00",10,100)]});
  const fractional=trade({executions:[execution("e","ENTRY","09:30:00.000",10,100)]});
  assert.equal(calculateTradeExcursion(base,bars).mfeDollars,calculateTradeExcursion(fractional,bars).mfeDollars);
});

test("risk, data-policy version and interval are part of cache identity",()=>{
  assert.notEqual(tradeExcursionInputHash(trade(),"2026-09-01"),tradeExcursionInputHash(trade({risk:200}),"2026-09-01"));
  assert.equal(excursionInstrument("USD/CAD").mode,"UNSUPPORTED");
});
