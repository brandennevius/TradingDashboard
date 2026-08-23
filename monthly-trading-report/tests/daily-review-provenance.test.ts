import assert from "node:assert/strict";
import test from "node:test";
import {
  resolveDailyReviewProvenance,
  upsertBrokerPortfolioSnapshotCollection,
  type BrokerPortfolioSnapshotInput
} from "../lib/broker-portfolio-snapshot";

const hashA = "a".repeat(64);
const hashB = "b".repeat(64);

function statement(coverageDate: string, sourceHash = hashA, currentEquity = 100_000): BrokerPortfolioSnapshotInput {
  return {
    userId: "branden",
    portfolioTag: "CF_Statement",
    coverageDate,
    sourceHash,
    sourceFilename: "CF statement.pdf",
    balance: 99_500,
    currentEquity,
    statementEquity: 100_100,
    floatingPnl: 600,
    openPositions: [{
      entryDate: "2026-08-17", timeValue: "10:00:00", side: "LONG", shares: 10, symbol: "HPE",
      entryPrice: 50, currentPrice: 52, usedMargin: 500, stopPrice: 48, takeProfitPrice: 60, floatingPnl: 20, commission: 1
    }],
    workingOrders: []
  };
}

test("CF import stores a normalized dated broker snapshot without raw PDF material", () => {
  const stored = upsertBrokerPortfolioSnapshotCollection([], statement("2026-08-21"), "2026-08-21T21:00:00.000Z");
  assert.equal(stored.length, 1);
  assert.equal(stored[0].coverageDate, "2026-08-21");
  assert.equal(stored[0].sourceHash, hashA);
  assert.equal(stored[0].openPositions[0].symbol, "HPE");
  assert.equal(stored[0].openPositions[0].shares, 10);
  assert(!("pdf" in stored[0]));
});
test("reimporting the same statement and date is idempotent", () => {
  const first = upsertBrokerPortfolioSnapshotCollection([], statement("2026-08-21"), "2026-08-21T21:00:00.000Z");
  const second = upsertBrokerPortfolioSnapshotCollection(first, statement("2026-08-21"), "2026-08-22T01:00:00.000Z");
  assert.deepEqual(second, first);
});

test("a newer statement preserves the older dated snapshot", () => {
  const first = upsertBrokerPortfolioSnapshotCollection([], statement("2026-08-20"), "2026-08-20T21:00:00.000Z");
  const second = upsertBrokerPortfolioSnapshotCollection(first, statement("2026-08-21", hashB, 101_000), "2026-08-21T21:00:00.000Z");
  assert.deepEqual(second.map((snapshot) => snapshot.coverageDate), ["2026-08-20", "2026-08-21"]);
  assert.equal(second[0].currentEquity, 100_000);
  assert.equal(second[1].currentEquity, 101_000);
});

test("an exact selected date uses the broker snapshot and exact-date equity", () => {
  const snapshots = upsertBrokerPortfolioSnapshotCollection([], statement("2026-08-21"), "2026-08-21T21:00:00.000Z");
  const result = resolveDailyReviewProvenance(snapshots, "CF_Statement", "2026-08-21");
  assert.equal(result.label, "BROKER SNAPSHOT");
  assert.equal(result.anchorCoverageDate, "2026-08-21");
  assert.equal(result.accountEquity, 100_000);
});

test("a historical date never receives equity from a later snapshot", () => {
  const snapshots = upsertBrokerPortfolioSnapshotCollection([], statement("2026-08-21"), "2026-08-21T21:00:00.000Z");
  const result = resolveDailyReviewProvenance(snapshots, "CF_Statement", "2026-08-20");
  assert.equal(result.label, "ESTIMATED");
  assert.equal(result.anchorCoverageDate, null);
  assert.equal(result.accountEquity, null);
});

test("a prior statement is an explicit broker anchor and does not claim selected-date equity", () => {
  const snapshots = upsertBrokerPortfolioSnapshotCollection([], statement("2026-08-20"), "2026-08-20T21:00:00.000Z");
  const result = resolveDailyReviewProvenance(snapshots, "CF_Statement", "2026-08-21");
  assert.equal(result.label, "BROKER-ANCHORED RECONSTRUCTION");
  assert.equal(result.anchorCoverageDate, "2026-08-20");
  assert.equal(result.accountEquity, null);
});
