import assert from "node:assert/strict";
import test from "node:test";
import { getMarketCandlesWithProvider, marketProviderSymbols } from "../lib/market-data";

test("slash-form FX pairs map to provider-specific symbols", () => {
  assert.deepEqual(marketProviderSymbols("#usd/cad"), {
    symbol: "USD/CAD",
    stooq: "usdcad",
    yahoo: "USDCAD=X"
  });
  assert.deepEqual(marketProviderSymbols("AAPL"), {
    symbol: "AAPL",
    stooq: "aapl.us",
    yahoo: "AAPL"
  });
});

test("FX daily prices use the Stooq currency pair instead of an equity suffix", async (context) => {
  const originalFetch = globalThis.fetch;
  const requestedUrls: string[] = [];
  context.after(() => {
    globalThis.fetch = originalFetch;
  });
  globalThis.fetch = async (input) => {
    requestedUrls.push(String(input));
    return new Response("Date,Open,High,Low,Close,Volume\n2026-08-07,1.371,1.375,1.369,1.372,0\n");
  };

  const result = await getMarketCandlesWithProvider("USD/CAD", "1d");

  assert.equal(result.symbol, "USD/CAD");
  assert.equal(result.provider, "stooq");
  assert.equal(result.candles.at(-1)?.time, "2026-08-07");
  assert.match(requestedUrls[0], /[?&]s=usdcad(?:&|$)/);
  assert.doesNotMatch(requestedUrls[0], /usdcad\.us/);
});

test("FX prices fall back to Yahoo's currency-pair symbol", async (context) => {
  const originalFetch = globalThis.fetch;
  const requestedUrls: string[] = [];
  context.after(() => {
    globalThis.fetch = originalFetch;
  });
  globalThis.fetch = async (input) => {
    const url = String(input);
    requestedUrls.push(url);
    if (url.includes("stooq.com")) {
      return new Response("Date,Open,High,Low,Close,Volume\n");
    }
    return Response.json({
      chart: {
        result: [{
          timestamp: [Date.parse("2026-08-07T00:00:00Z") / 1000],
          indicators: { quote: [{ open: [1.371], high: [1.375], low: [1.369], close: [1.372], volume: [0] }] }
        }]
      }
    });
  };

  const result = await getMarketCandlesWithProvider("USD/CAD", "1d");

  assert.equal(result.provider, "yahoo");
  assert.equal(result.candles.at(-1)?.time, "2026-08-07");
  assert(requestedUrls.some((url) => url.includes("/USDCAD%3DX?")));
});
