import assert from "node:assert/strict";
import test from "node:test";
import { GET as marketDataRoute } from "../app/api/market-data/[symbol]/route";
import { getExactMarketSessionPrice, getMarketCandlesWithProvider, marketProviderSymbols } from "../lib/market-data";

test("slash-form FX pairs map to provider-specific symbols", () => {
  assert.deepEqual(marketProviderSymbols("#usd/cad"), {
    symbol: "USD/CAD",
    fmp: "USDCAD",
    stooq: "usdcad",
    yahoo: "USDCAD=X"
  });
  assert.deepEqual(marketProviderSymbols("AAPL"), {
    symbol: "AAPL",
    fmp: "AAPL",
    stooq: "aapl.us",
    yahoo: "AAPL"
  });
});

test("FMP is the primary bounded daily provider and keeps the key out of the URL", async (context) => {
  const originalFetch = globalThis.fetch;
  const originalKey = process.env.FMP_API_KEY;
  const requests: Array<{ url: string; headers: Headers }> = [];
  context.after(() => {
    globalThis.fetch = originalFetch;
    if (originalKey === undefined) delete process.env.FMP_API_KEY;
    else process.env.FMP_API_KEY = originalKey;
  });
  process.env.FMP_API_KEY = "test-key";
  globalThis.fetch = async (input, init) => {
    requests.push({ url: String(input), headers: new Headers(init?.headers) });
    return Response.json([
      { date: "2026-08-21", open: 100, high: 104, low: 99, close: 103, volume: 123456 },
      { date: "2026-08-20", open: 98, high: 101, low: 97, close: 100, volume: 100000 }
    ]);
  };

  const result = await getMarketCandlesWithProvider("AAPL", "1d");

  assert.equal(result.provider, "fmp");
  assert.deepEqual(result.candles.map((candle) => candle.time), ["2026-08-20", "2026-08-21"]);
  assert.equal(requests.length, 1);
  assert.match(requests[0].url, /historical-price-eod\/full/);
  assert.match(requests[0].url, /[?&]symbol=AAPL(?:&|$)/);
  assert.match(requests[0].url, /[?&]from=\d{4}-\d{2}-\d{2}(?:&|$)/);
  assert.match(requests[0].url, /[?&]to=\d{4}-\d{2}-\d{2}(?:&|$)/);
  assert.doesNotMatch(requests[0].url, /apikey/i);
  assert.equal(requests[0].headers.get("apikey"), "test-key");
});

test("market-data API route reuses the shared FMP-backed market data loader", async (context) => {
  const originalFetch = globalThis.fetch;
  const originalKey = process.env.FMP_API_KEY;
  const requestedUrls: string[] = [];
  context.after(() => {
    globalThis.fetch = originalFetch;
    if (originalKey === undefined) delete process.env.FMP_API_KEY;
    else process.env.FMP_API_KEY = originalKey;
  });
  process.env.FMP_API_KEY = "test-key";
  globalThis.fetch = async (input, init) => {
    requestedUrls.push(String(input));
    assert.equal(new Headers(init?.headers).get("apikey"), "test-key");
    return Response.json([
      { date: "2026-08-28", open: 86.2, high: 88.5, low: 85.9, close: 87.93, volume: 1234567 },
      { date: "2026-08-27", open: 88.1, high: 88.8, low: 86.5, close: 87.54, volume: 1111111 }
    ]);
  };

  const response = await marketDataRoute(
    new Request("https://example.test/api/market-data/GM?timeframe=1d"),
    { params: Promise.resolve({ symbol: "GM" }) }
  );
  const payload = await response.json();

  assert.equal(response.status, 200);
  assert.equal(payload.symbol, "GM");
  assert.equal(payload.provider, "fmp");
  assert.deepEqual(payload.candles.map((candle: { time: string }) => candle.time), ["2026-08-27", "2026-08-28"]);
  assert.match(requestedUrls[0], /financialmodelingprep\.com\/stable\/historical-price-eod\/full/);
});

test("FMP exact-session lookup uses the requested session and normalizes FX symbols", async (context) => {
  const originalFetch = globalThis.fetch;
  const originalKey = process.env.FMP_API_KEY;
  let requestedUrl = "";
  context.after(() => {
    globalThis.fetch = originalFetch;
    if (originalKey === undefined) delete process.env.FMP_API_KEY;
    else process.env.FMP_API_KEY = originalKey;
  });
  process.env.FMP_API_KEY = "test-key";
  globalThis.fetch = async (input, init) => {
    requestedUrl = String(input);
    assert.equal(new Headers(init?.headers).get("apikey"), "test-key");
    return Response.json([
      { date: "2026-08-21", open: 1.38, high: 1.39, low: 1.37, close: 1.385, volume: 0 },
      { date: "2026-08-20", open: 1.38, high: 1.37, low: 1.36, close: 1.365, volume: 0 }
    ]);
  };

  const result = await getExactMarketSessionPrice("USD/CAD", "2026-08-21");

  assert.equal(result.provider, "fmp");
  assert.equal(result.price, 1.385);
  assert.match(requestedUrl, /[?&]symbol=USDCAD(?:&|$)/);
  assert.match(requestedUrl, /[?&]from=2026-08-21(?:&|$)/);
  assert.match(requestedUrl, /[?&]to=2026-08-21(?:&|$)/);
  assert.doesNotMatch(requestedUrl, /apikey/i);
});

test("FMP rejects malformed OHLC rows", async (context) => {
  const originalFetch = globalThis.fetch;
  const originalKey = process.env.FMP_API_KEY;
  context.after(() => {
    globalThis.fetch = originalFetch;
    if (originalKey === undefined) delete process.env.FMP_API_KEY;
    else process.env.FMP_API_KEY = originalKey;
  });
  process.env.FMP_API_KEY = "test-key";
  globalThis.fetch = async () => Response.json([
    { date: "2026-08-21", open: 1.38, high: 1.37, low: 1.36, close: 1.365, volume: 0 },
    { date: "2026-08-20", open: 1.38, high: 1.39, low: 1.37, close: 1.385, volume: 0 }
  ]);

  const result = await getMarketCandlesWithProvider("USD/CAD", "1d");

  assert.equal(result.provider, "fmp");
  assert.deepEqual(result.candles.map((candle) => candle.time), ["2026-08-20"]);
});

test("FX daily prices use the Stooq currency pair instead of an equity suffix", async (context) => {
  const originalFetch = globalThis.fetch;
  const originalKey = process.env.FMP_API_KEY;
  const requestedUrls: string[] = [];
  context.after(() => {
    globalThis.fetch = originalFetch;
    if (originalKey === undefined) delete process.env.FMP_API_KEY;
    else process.env.FMP_API_KEY = originalKey;
  });
  delete process.env.FMP_API_KEY;
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
  const originalKey = process.env.FMP_API_KEY;
  const requestedUrls: string[] = [];
  context.after(() => {
    globalThis.fetch = originalFetch;
    if (originalKey === undefined) delete process.env.FMP_API_KEY;
    else process.env.FMP_API_KEY = originalKey;
  });
  delete process.env.FMP_API_KEY;
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
  const originalKey = process.env.FMP_API_KEY;
  const requestedUrls: string[] = [];
  context.after(() => {
    globalThis.fetch = originalFetch;
    if (originalKey === undefined) delete process.env.FMP_API_KEY;
    else process.env.FMP_API_KEY = originalKey;
  });
  delete process.env.FMP_API_KEY;
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
  const originalKey = process.env.FMP_API_KEY;
  const requestedUrls: string[] = [];
  context.after(() => {
    globalThis.fetch = originalFetch;
    if (originalKey === undefined) delete process.env.FMP_API_KEY;
    else process.env.FMP_API_KEY = originalKey;
  });
  delete process.env.FMP_API_KEY;
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
  const originalKey = process.env.FMP_API_KEY;
  context.after(() => {
    globalThis.fetch = originalFetch;
    if (originalKey === undefined) delete process.env.FMP_API_KEY;
    else process.env.FMP_API_KEY = originalKey;
  });
  delete process.env.FMP_API_KEY;
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
  const originalKey = process.env.FMP_API_KEY;
  context.after(() => {
    globalThis.fetch = originalFetch;
    if (originalKey === undefined) delete process.env.FMP_API_KEY;
    else process.env.FMP_API_KEY = originalKey;
  });
  delete process.env.FMP_API_KEY;
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
