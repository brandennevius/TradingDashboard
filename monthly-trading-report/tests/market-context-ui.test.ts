import assert from "node:assert/strict";
import test from "node:test";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { MarketContextComparison } from "../app/components/TradeMarketContext";
import type { TradeMarketContext } from "../lib/trade-market-context";

function context(): TradeMarketContext {
  return {
    entryState: "Unavailable", endpointState: "Unavailable", entrySession: "2026-09-04", endpointSession: "2026-10-20", endpointLabel: "Current",
    transition: "Unavailable → Unavailable", changedDuringTrade: true, missingSessions: 1, historySessions: 221,
    entry: null, endpoint: null, refreshedAt: "2026-10-20T22:30:00Z",
    history: Array.from({ length: 220 }, (_, i) => ({ session: `history-${i}`, version: "v1", source: "Reconstructed", state: "Grow" }))
  };
}
test("long trade renders six comparisons and no expanded daily rows", () => {
  const html = renderToStaticMarkup(createElement(MarketContextComparison, { context: context() }));
  assert.equal((html.match(/scope="row"/g) || []).length, 6);
  assert.match(html, /At entry/); assert.match(html, /Current/);
  assert.match(html, /View market history \(220 saved sessions\)/);
  assert.doesNotMatch(html, /history-219/); assert.doesNotMatch(html, /<details open/);
  assert.match(html, /history is incomplete/); assert.match(html, /Unavailable/);
});
test("closed detail labels the endpoint At exit and shows source dates", () => {
  const c = { ...context(), endpointLabel: "At exit" as const };
  const html = renderToStaticMarkup(createElement(MarketContextComparison, { context: c }));
  assert.match(html, /At exit/); assert.match(html, /2026-09-04/); assert.match(html, /2026-10-20/);
  assert.doesNotMatch(html, />Current</);
});
