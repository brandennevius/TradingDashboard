import assert from "node:assert/strict";
import test from "node:test";
import { calculateTradeExcursion, excursionInstrument, tradeExcursionInputHash } from "../lib/trade-excursion";
import { fetchTradeIntradayBars, parseFmpIntradayBars } from "../lib/trade-excursion-market-data";
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

test("maps index CFDs to futures and labels results as estimates", () => {
  const mappedTrade = trade({
    symbol: ".US500", avgEntry: 7500, shares: 1, risk: 100,
    executions: [execution("e1", "ENTRY", "09:30:00", 1, 7500), execution("x1", "EXIT", "10:00:00", 1, 7600)]
  });
  const instrument = excursionInstrument(mappedTrade.symbol);
  assert.equal(instrument.providerSymbol, "ESUSD");
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

test("parses and sorts the strict FMP intraday response shape", () => {
  assert.deepEqual(parseFmpIntradayBars([
    { date: "2026-09-01 09:35:00", open: 101, high: 102, low: 100, close: 101.5 },
    { date: "2026-09-01 09:30:00", open: 100, high: 101, low: 99, close: 100.5 },
    { date: "bad", open: 1, high: 1, low: 1, close: 1 }
  ]).map((bar) => bar.time), ["2026-09-01 09:30:00", "2026-09-01 09:35:00"]);
});

test("execution changes invalidate the excursion input hash", () => {
  const original = trade();
  const changed = trade({ executions: original.executions.map((item, index) => index === 0 ? { ...item, price: item.price + 1 } : item) });
  assert.notEqual(tradeExcursionInputHash(original, "2026-09-01"), tradeExcursionInputHash(changed, "2026-09-01"));
});

test("FMP intraday request uses the stable endpoint and keeps its key out of the URL", async () => {
  const originalKey = process.env.FMP_API_KEY;
  const originalFetch = globalThis.fetch;
  process.env.FMP_API_KEY = "test-key";
  let requestedUrl = "";
  let requestHeaders: HeadersInit | undefined;
  globalThis.fetch = (async (input: string | URL | Request, init?: RequestInit) => {
    requestedUrl = String(input);
    requestHeaders = init?.headers;
    return new Response(JSON.stringify([{ date: "2026-09-01 09:30:00", open: 100, high: 101, low: 99, close: 100.5 }]));
  }) as typeof fetch;
  try {
    const result = await fetchTradeIntradayBars(excursionInstrument("FRO"), "2026-09-01", "2026-09-01", "5min");
    assert.equal(result.bars.length, 1);
    assert.match(requestedUrl, /\/stable\/historical-chart\/5min\?/);
    assert.match(requestedUrl, /symbol=FRO/);
    assert.doesNotMatch(requestedUrl, /test-key/);
    assert.equal((requestHeaders as Record<string, string>).apikey, "test-key");
  } finally {
    globalThis.fetch = originalFetch;
    if (originalKey === undefined) delete process.env.FMP_API_KEY;
    else process.env.FMP_API_KEY = originalKey;
  }
});
