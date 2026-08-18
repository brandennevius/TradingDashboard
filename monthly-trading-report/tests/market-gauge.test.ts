import assert from "node:assert/strict";
import test from "node:test";
import { filterCandlesThroughSession, hasExactIndexSessionEvidence } from "../lib/market-gauge-session";

test("exact-session gauge never uses bars after the requested session", () => {
  const candles = [
    { time: "2026-08-13", open: 1, high: 2, low: 1, close: 2 },
    { time: "2026-08-14", open: 2, high: 3, low: 2, close: 3 },
    { time: "2026-08-17", open: 3, high: 4, low: 3, close: 4 }
  ];
  assert.deepEqual(filterCandlesThroughSession(candles, "2026-08-14").map((item) => item.time), ["2026-08-13", "2026-08-14"]);
});

test("exact-session gauge requires SPY, QQQ, and IWM evidence on the same session", () => {
  const regime = (symbol: string, date = "2026-08-14") => ({
    symbol, date, close: 1, ema21: 1, sma50: 1, sma200: 1,
    shortTerm: "Up", mediumTerm: "Up", longTerm: "Up",
    rawShortTerm: "Up", rawMediumTerm: "Up", rawLongTerm: "Up",
    above21Percent: 0, above50Percent: 0, extension: "Normal"
  } as const);
  const expected = ["SPY", "QQQ", "IWM"];
  assert.equal(hasExactIndexSessionEvidence([regime("SPY"), regime("QQQ"), regime("IWM")], "2026-08-14", expected), true);
  assert.equal(hasExactIndexSessionEvidence([regime("SPY"), regime("QQQ")], "2026-08-14", expected), false);
  assert.equal(hasExactIndexSessionEvidence([regime("SPY"), regime("QQQ"), regime("IWM", "2026-08-13")], "2026-08-14", expected), false);
});
