import assert from "node:assert/strict";
import test from "node:test";
import { alpacaStockSymbol, fetchAlpacaBars, parseAlpacaBars } from "../lib/alpaca-market-data";
import { fetchTradeIntradayBars } from "../lib/trade-excursion-market-data";
import { excursionInstrument } from "../lib/trade-excursion";

function setup(context: { after: (fn: () => void) => void }) {
  const original = globalThis.fetch;
  const key = process.env.ALPACA_API_KEY, secret = process.env.ALPACA_API_SECRET;
  process.env.ALPACA_API_KEY = "key-fixture";
  process.env.ALPACA_API_SECRET = "secret-fixture";
  context.after(() => {
    globalThis.fetch = original;
    if (key === undefined) delete process.env.ALPACA_API_KEY; else process.env.ALPACA_API_KEY = key;
    if (secret === undefined) delete process.env.ALPACA_API_SECRET; else process.env.ALPACA_API_SECRET = secret;
  });
}
const bar = (t: string) => ({ t, o: 100, h: 103, l: 99, c: 102, v: 1000 });

test("UTC bars map to New York dates across DST and midnight", () => {
  const bars = parseAlpacaBars([bar("2026-01-05T14:30:00Z"), bar("2026-07-06T13:30:00Z"), bar("2026-07-07T00:00:00Z")]);
  assert.deepEqual(bars.map((b) => b.time), ["2026-01-05 09:30:00", "2026-07-06 09:30:00", "2026-07-06 20:00:00"]);
  assert.equal(alpacaStockSymbol("BRK.B"), "BRK/B");
  assert.equal(alpacaStockSymbol("USD/CAD"), null);
});

test("intraday history follows pagination, deduplicates, and excludes extended hours", async (context) => {
  setup(context);
  const urls: string[] = [];
  globalThis.fetch = async (input, init) => {
    const url = String(input); urls.push(url);
    assert.equal(new Headers(init?.headers).get("APCA-API-KEY-ID"), "key-fixture");
    assert.equal(new Headers(init?.headers).get("APCA-API-SECRET-KEY"), "secret-fixture");
    assert.doesNotMatch(url, /key-fixture|secret-fixture/);
    const page = new URL(url).searchParams.get("page_token");
    return Response.json(page
      ? { bars: { AAPL: [bar("2026-09-01T13:30:00Z"), bar("2026-09-01T13:35:00Z"), bar("2026-09-01T20:00:00Z")] } }
      : { bars: { AAPL: [bar("2026-09-01T12:00:00Z"), bar("2026-09-01T13:30:00Z")] }, next_page_token: "next" });
  };
  const result = await fetchTradeIntradayBars(excursionInstrument("AAPL"), "2026-09-01", "2026-09-01");
  assert.equal(result.error, "");
  assert.equal(result.provider, "alpaca");
  assert.deepEqual(result.bars.map((b) => b.time), ["2026-09-01 09:30:00", "2026-09-01 09:35:00"]);
  assert.equal(urls.length, 2);
  assert.equal(new URL(urls[0]).searchParams.get("feed"), "sip");
  assert.equal(new URL(urls[0]).searchParams.get("adjustment"), "raw");
});

test("a failed later page discards earlier bars and never exposes the response body", async (context) => {
  setup(context);
  let count = 0;
  globalThis.fetch = async () => ++count === 1
    ? Response.json({ bars: { AAPL: [bar("2026-09-01T13:30:00Z")] }, next_page_token: "next" })
    : new Response("secret-fixture", { status: 403 });
  const result = await fetchTradeIntradayBars(excursionInstrument("AAPL"), "2026-09-01", "2026-09-01");
  assert.deepEqual(result.bars, []);
  assert.match(result.error, /HTTP 403/);
  assert.doesNotMatch(result.error, /secret-fixture/);
});

test("repeated pagination tokens are rejected", async (context) => {
  setup(context);
  globalThis.fetch = async () => Response.json({ bars: { AAPL: [bar("2026-09-01T13:30:00Z")] }, next_page_token: "same" });
  await assert.rejects(fetchAlpacaBars("AAPL", "5Min", "2026-09-01", "2026-09-01"), /pagination/);
});

test("free-tier requests end outside the latest fifteen minutes", async (context) => {
  setup(context);
  const now = Date.now();
  globalThis.fetch = async (input) => {
    const end = new URL(String(input)).searchParams.get("end")!;
    assert(Date.parse(end) <= Date.now() - 15 * 60_000);
    return Response.json({ bars: { AAPL: [] } });
  };
  await fetchAlpacaBars("AAPL", "5Min", "2026-01-01", new Date(now).toISOString().slice(0, 10));
});

test("missing credentials and unsupported FX produce unavailable data", async (context) => {
  setup(context);
  delete process.env.ALPACA_API_SECRET;
  globalThis.fetch = async () => { throw new Error("must not fetch"); };
  const missing = await fetchTradeIntradayBars(excursionInstrument("AAPL"), "2026-09-01", "2026-09-01");
  assert.match(missing.error, /credentials/);
  const unsupported = await fetchTradeIntradayBars(excursionInstrument("USD/CAD"), "2026-09-01", "2026-09-01");
  assert.equal(unsupported.bars.length, 0);
  assert.match(unsupported.error, /US stocks/);
});
