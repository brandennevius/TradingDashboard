import assert from "node:assert/strict";
import test from "node:test";
import { getExactMarketSessionPrice, getMarketCandlesWithProvider, marketProviderSymbols } from "../lib/market-data";

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
    return new Response("Date,Open,High,Low,Close,Volume\n2026-08-10,1.371,1.375,1.369,1.372,0\n");
  };

  const result = await getMarketCandlesWithProvider("USD/CAD", "1d");

  assert.equal(result.symbol, "USD/CAD");
  assert.equal(result.provider, "stooq");
  assert.equal(result.candles.at(-1)?.time, "2026-08-10");
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
          meta: { exchangeTimezoneName: "Europe/London" },
          timestamp: [
            Date.parse("2026-08-09T23:00:00Z") / 1000,
            Date.parse("2026-08-10T23:00:00Z") / 1000
          ],
          indicators: {
            quote: [{
              open: [1.371, 1.372],
              high: [1.375, 1.376],
              low: [1.369, 1.37],
              close: [1.372, 1.373],
              volume: [0, 0]
            }]
          }
        }]
      }
    });
  };

  const result = await getMarketCandlesWithProvider("USD/CAD", "1d");

  assert.equal(result.provider, "yahoo");
  assert.deepEqual(result.candles.map((candle) => candle.time), ["2026-08-10", "2026-08-11"]);
  assert(requestedUrls.some((url) => url.includes("/USDCAD%3DX?")));
});

test("exact-session prices use a date-bounded Stooq request", async (context) => {
  const originalFetch = globalThis.fetch;
  const requestedUrls: string[] = [];
  context.after(() => {
    globalThis.fetch = originalFetch;
  });
  globalThis.fetch = async (input) => {
    requestedUrls.push(String(input));
    return new Response("Date,Open,High,Low,Close,Volume\n2026-08-17,31.34,32.15,31.28,31.88,19657409\n");
  };

  const result = await getExactMarketSessionPrice("CPRT", "2026-08-17");

  assert.deepEqual(result, {
    symbol: "CPRT",
    requestedSession: "2026-08-17",
    sessionDate: "2026-08-17",
    price: 31.88,
    timestamp: null,
    provider: "stooq",
    priceType: "delayed_close"
  });
  assert.equal(requestedUrls.length, 1);
  assert.match(requestedUrls[0], /[?&]d1=20260817(?:&|$)/);
  assert.match(requestedUrls[0], /[?&]d2=20260817(?:&|$)/);
});

test("stale primary history falls through to an exact Yahoo daily close", async (context) => {
  const originalFetch = globalThis.fetch;
  const requestedUrls: string[] = [];
  context.after(() => {
    globalThis.fetch = originalFetch;
  });
  globalThis.fetch = async (input) => {
    const url = String(input);
    requestedUrls.push(url);
    if (url.includes("stooq.com")) {
      return new Response("Date,Open,High,Low,Close,Volume\n2026-08-14,31,32,30,31.25,1000\n");
    }
    return Response.json({
      chart: {
        result: [{
          meta: { exchangeTimezoneName: "America/New_York" },
          timestamp: [Date.parse("2026-08-17T13:30:00Z") / 1000],
          indicators: { quote: [{ open: [31.34], high: [32.15], low: [31.28], close: [31.88], volume: [19657409] }] }
        }]
      }
    });
  };

  const result = await getExactMarketSessionPrice("CPRT", "2026-08-17");

  assert.equal(result.provider, "yahoo");
  assert.equal(result.sessionDate, "2026-08-17");
  assert.equal(result.price, 31.88);
  assert.equal(result.priceType, "delayed_close");
  assert(requestedUrls.some((url) => url.includes("query1.finance.yahoo.com")));
  assert(requestedUrls.every((url) => !url.includes("period2=") || url.includes("period1=1786838400&period2=1787097600")));
});

test("an unfinished Yahoo daily row may use last trade only when its timestamp is the requested session", async (context) => {
  const originalFetch = globalThis.fetch;
  context.after(() => {
    globalThis.fetch = originalFetch;
  });
  globalThis.fetch = async (input) => {
    if (String(input).includes("stooq.com")) {
      return new Response("Date,Open,High,Low,Close,Volume\n2026-08-17,31.34,32.15,31.28,0,19657409\n");
    }
    return Response.json({
      chart: {
        result: [{
          meta: {
            exchangeTimezoneName: "America/New_York",
            regularMarketPrice: 31.88,
            regularMarketTime: Date.parse("2026-08-17T20:00:00Z") / 1000
          },
          timestamp: [Date.parse("2026-08-17T13:30:00Z") / 1000],
          indicators: { quote: [{ open: [31.34], high: [32.15], low: [31.28], close: [null], volume: [19657409] }] }
        }]
      }
    });
  };

  const result = await getExactMarketSessionPrice("CPRT", "2026-08-17");

  assert.equal(result.provider, "yahoo");
  assert.equal(result.sessionDate, "2026-08-17");
  assert.equal(result.price, 31.88);
  assert.equal(result.timestamp, "2026-08-17T20:00:00.000Z");
  assert.equal(result.priceType, "last_trade");
});

test("historical exact-session lookup never substitutes a stale close or a newer live quote", async (context) => {
  const originalFetch = globalThis.fetch;
  context.after(() => {
    globalThis.fetch = originalFetch;
  });
  globalThis.fetch = async (input) => {
    if (String(input).includes("stooq.com")) {
      return new Response("Date,Open,High,Low,Close,Volume\n2026-08-14,31,32,30,31.25,1000\n");
    }
    return Response.json({
      chart: {
        result: [{
          meta: {
            exchangeTimezoneName: "America/New_York",
            regularMarketPrice: 32.5,
            regularMarketTime: Date.parse("2026-08-18T15:00:00Z") / 1000
          },
          timestamp: [
            Date.parse("2026-08-14T13:30:00Z") / 1000,
            Date.parse("2026-08-17T13:30:00Z") / 1000
          ],
          indicators: {
            quote: [{
              open: [31, 31.34], high: [32, 32.15], low: [30, 31.28], close: [31.25, null], volume: [1000, 19657409]
            }]
          }
        }]
      }
    });
  };

  const result = await getExactMarketSessionPrice("CPRT", "2026-08-17");

  assert.equal(result.provider, "unavailable");
  assert.equal(result.sessionDate, null);
  assert.equal(result.price, null);
});
