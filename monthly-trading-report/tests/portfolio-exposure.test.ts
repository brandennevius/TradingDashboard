import assert from "node:assert/strict";
import test from "node:test";
import { portfolioExposure } from "../lib/portfolio-exposure";

test("gross exposure includes both sides and uses distinct account denominators", () => {
  const result = portfolioExposure([50_000, -20_000], 700_000, 688_000);
  assert.equal(result.dollars, 70_000);
  assert.equal(result.equityPct, 10);
  assert.equal(result.remainingDrawdown, 12_000);
  assert.equal(result.drawdownPct, 70_000 / 12_000 * 100);
});

test("missing valuations do not silently understate exposure", () => {
  const result = portfolioExposure([500, null], 700_000, 688_000);
  assert.equal(result.dollars, null);
  assert.equal(result.equityPct, null);
  assert.equal(result.drawdownPct, null);
  assert.equal(result.missingCount, 1);
});

test("empty portfolios and unavailable or exhausted denominators", () => {
  assert.equal(portfolioExposure([], 700_000, 688_000).drawdownPct, 0);
  assert.equal(portfolioExposure([500], null, 688_000).equityPct, null);
  assert.equal(portfolioExposure([500], 700_000, null).drawdownPct, null);
  for (const equity of [688_000, 680_000]) {
    const result = portfolioExposure([500], equity, 688_000);
    assert.equal(result.remainingDrawdown, 0);
    assert.equal(result.drawdownPct, null);
  }
});
