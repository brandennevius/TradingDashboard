import assert from "node:assert/strict";
import test from "node:test";
import { GET as marketDataRoute } from "../app/api/market-data/[symbol]/route";
import { getExactMarketSessionPrice, getMarketCandlesWithProvider, marketProviderSymbols } from "../lib/market-data";

function credentials(context: { after: (fn: () => void) => void }) {
  const key = process.env.ALPACA_API_KEY, secret = process.env.ALPACA_API_SECRET;
  const fetch = globalThis.fetch;
  process.env.ALPACA_API_KEY = "test-key";
  process.env.ALPACA_API_SECRET = "test-secret";
  context.after(() => {
    globalThis.fetch = fetch;
    if (key === undefined) delete process.env.ALPACA_API_KEY; else process.env.ALPACA_API_KEY = key;
    if (secret === undefined) delete process.env.ALPACA_API_SECRET; else process.env.ALPACA_API_SECRET = secret;
  });
}

test("FX stays with daily fallback providers; listed symbols map to Alpaca", () => {
  assert.deepEqual(marketProviderSymbols("#usd/cad"), { symbol: "USD/CAD", alpaca: null, stooq: "usdcad", yahoo: "USDCAD=X" });
  assert.deepEqual(marketProviderSymbols("AAPL"), { symbol: "AAPL", alpaca: "AAPL", stooq: "aapl.us", yahoo: "AAPL" });
});

test("Alpaca is the primary daily provider and credentials stay in headers", async (context) => {
  credentials(context);
  let requestedUrl = "";
  globalThis.fetch = async (input, init) => {
    requestedUrl = String(input);
    assert.equal(new Headers(init?.headers).get("APCA-API-KEY-ID"), "test-key");
    assert.equal(new Headers(init?.headers).get("APCA-API-SECRET-KEY"), "test-secret");
    return Response.json({bars:{ AAPL:[
      {t:"2026-08-20T04:00:00Z",o:98,h:101,l:97,c:100,v:100000},
      {t:"2026-08-21T04:00:00Z",o:100,h:104,l:99,c:103,v:123456}
    ]},next_page_token:null});
  };
  const result=await getMarketCandlesWithProvider("AAPL","1d");
  assert.equal(result.provider,"alpaca");
  assert.deepEqual(result.candles.map(x=>x.time),["2026-08-20","2026-08-21"]);
  const url=new URL(requestedUrl);
  assert.equal(url.hostname,"data.alpaca.markets");
  assert.equal(url.searchParams.get("feed"),"sip");
  assert.equal(url.searchParams.get("timeframe"),"1Day");
  assert.doesNotMatch(requestedUrl,/test-key|test-secret/);
});

test("market-data route uses the shared Alpaca loader",async(context)=>{
  credentials(context);
  globalThis.fetch=async()=>Response.json({bars:{GM:[{t:"2026-08-28T04:00:00Z",o:86.2,h:88.5,l:85.9,c:87.93,v:1234567}]}});
  const response=await marketDataRoute(new Request("https://example.test/api/market-data/GM?timeframe=1d"),{params:Promise.resolve({symbol:"GM"})});
  const payload=await response.json();
  assert.equal(response.status,200);
  assert.equal(payload.provider,"alpaca");
  assert.equal(payload.candles[0].time,"2026-08-28");
});

test("Alpaca exact-session valuation excludes malformed and stale rows",async(context)=>{
  credentials(context);
  globalThis.fetch=async()=>Response.json({bars:{AAPL:[
    {t:"2026-08-20T04:00:00Z",o:100,h:104,l:99,c:103},
    {t:"2026-08-21T04:00:00Z",o:100,h:99,l:98,c:103},
    {t:"2026-08-21T04:00:00Z",o:100,h:104,l:99,c:102}
  ]}});
  const result=await getExactMarketSessionPrice("AAPL","2026-08-21");
  assert.equal(result.provider,"alpaca");
  assert.equal(result.sessionDate,"2026-08-21");
  assert.equal(result.price,102);
});

test("FX daily prices use the Stooq currency pair instead of an equity suffix", async (context) => {
  const originalFetch = globalThis.fetch;
  const originalKey = process.env.ALPACA_API_KEY;
  const requestedUrls: string[] = [];
  context.after(() => {
    globalThis.fetch = originalFetch;
    if (originalKey === undefined) delete process.env.ALPACA_API_KEY;
    else process.env.ALPACA_API_KEY = originalKey;
  });
  delete process.env.ALPACA_API_KEY;
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
  const originalKey = process.env.ALPACA_API_KEY;
  const requestedUrls: string[] = [];
  context.after(() => {
    globalThis.fetch = originalFetch;
    if (originalKey === undefined) delete process.env.ALPACA_API_KEY;
    else process.env.ALPACA_API_KEY = originalKey;
  });
  delete process.env.ALPACA_API_KEY;
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
  const originalKey = process.env.ALPACA_API_KEY;
  const requestedUrls: string[] = [];
  context.after(() => {
    globalThis.fetch = originalFetch;
    if (originalKey === undefined) delete process.env.ALPACA_API_KEY;
    else process.env.ALPACA_API_KEY = originalKey;
  });
  delete process.env.ALPACA_API_KEY;
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
  const originalKey = process.env.ALPACA_API_KEY;
  const requestedUrls: string[] = [];
  context.after(() => {
    globalThis.fetch = originalFetch;
    if (originalKey === undefined) delete process.env.ALPACA_API_KEY;
    else process.env.ALPACA_API_KEY = originalKey;
  });
  delete process.env.ALPACA_API_KEY;
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
  const originalKey = process.env.ALPACA_API_KEY;
  context.after(() => {
    globalThis.fetch = originalFetch;
    if (originalKey === undefined) delete process.env.ALPACA_API_KEY;
    else process.env.ALPACA_API_KEY = originalKey;
  });
  delete process.env.ALPACA_API_KEY;
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
  const originalKey = process.env.ALPACA_API_KEY;
  context.after(() => {
    globalThis.fetch = originalFetch;
    if (originalKey === undefined) delete process.env.ALPACA_API_KEY;
    else process.env.ALPACA_API_KEY = originalKey;
  });
  delete process.env.ALPACA_API_KEY;
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
