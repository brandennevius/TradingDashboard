import assert from "node:assert/strict";
import test from "node:test";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { calculateGauge, GAUGE_VERSION, type GaugeState } from "../lib/market-gauge";
import { buildTradeMarketContext, marketClock, type MarketArchive, type MarketSnapshot } from "../lib/trade-market-context";
import { readMarketArchive, saveMarketArchive } from "../lib/market-context-store";
import { buildArchiveUpdate } from "../lib/market-context-service";

function snapshot(session: string, state: GaugeState): MarketSnapshot {
  const symbols = ["SPY", "QQQ", "IWM", "NVDA", "MSFT", "META", "AMZN", "GOOGL", "AVGO", "TSLA", "AMD", "PLTR", "CRWD", "COIN", "APP"];
  const days = Array.from({ length: 220 }, (_, i) => ({ time: new Date(Date.UTC(2025, 0, i + 1)).toISOString().slice(0, 10), open: i + 100, high: i + 102, low: i + 99, close: i + 101 }));
  days[219] = { ...days[219], time: session };
  const gauge = calculateGauge(Object.fromEntries(symbols.map((s) => [s, days])), session)!;
  gauge.overall_state = state;
  return { session, version: GAUGE_VERSION, recordedAt: `${session}T22:30:00Z`, source: "Archived", gauge };
}
const dates = ["2026-09-04", "2026-09-08", "2026-09-09", "2026-09-10", "2026-09-11"];
const archive: MarketArchive = { sessions: dates, snapshots: dates.map((d, i) => snapshot(d, i === 2 ? "Protect" : "Grow")), refreshedAt: "2026-09-11T22:30:00Z" };
const trade = { entryDate: "2026-09-08", exitDate: "2026-09-11", openTime: "", closeTime: "", status: "WIN" as const };

test("entry uses the previous exchange session across weekends and holidays", () => {
  const c = buildTradeMarketContext(trade, archive, new Date("2026-09-14T15:00:00Z"));
  assert.equal(c.entrySession, "2026-09-04");
  assert.equal(c.endpointSession, "2026-09-10");
  assert.equal(c.endpointLabel, "At exit");
  assert.equal(c.changedDuringTrade, true); // Identical endpoints do not conceal Protect in between.
});
test("closed endpoints stay fixed when current history advances", () => {
  const future = snapshot("2026-09-14", "Protect");
  const later = { ...archive, sessions: [...dates, future.session], snapshots: [...archive.snapshots, future] };
  assert.deepEqual(buildTradeMarketContext(trade, archive, new Date("2026-09-14T15:00Z")), buildTradeMarketContext(trade, later, new Date("2026-09-15T15:00Z")));
});
test("open trades with partial exit dates still use current completed context", () => {
  const c = buildTradeMarketContext({ ...trade, status: "OPEN" }, archive, new Date("2026-09-11T19:30:00Z"));
  assert.equal(c.endpointSession, "2026-09-10");
  assert.equal(c.endpointLabel, "Current");
  assert.equal(buildTradeMarketContext({ ...trade, status: "OPEN" }, archive, new Date("2026-09-11T20:16:00Z")).endpointSession, "2026-09-11");
});
test("explicit after-hours execution may use the same completed close", () => {
  const c = buildTradeMarketContext({ ...trade, openTime: "2026-09-08T21:00:00Z", closeTime: "2026-09-11T21:00:00Z" }, archive, new Date("2026-09-14T15:00Z"));
  assert.equal(c.entrySession, "2026-09-08");
  assert.equal(c.endpointSession, "2026-09-11");
});
test("New York close finalization respects daylight saving time", () => {
  assert.equal(marketClock(new Date("2026-07-01T20:14:00Z")).finalized, false);
  assert.equal(marketClock(new Date("2026-07-01T20:15:00Z")).finalized, true);
  assert.equal(marketClock(new Date("2026-12-01T20:30:00Z")).finalized, false);
  assert.equal(marketClock(new Date("2026-12-01T21:15:00Z")).finalized, true);
});
test("missing snapshots stay unavailable instead of borrowing a stale score", () => {
  const incomplete = { ...archive, snapshots: archive.snapshots.filter((s) => s.session !== "2026-09-10") };
  const c = buildTradeMarketContext(trade, incomplete, new Date("2026-09-14T15:00Z"));
  assert.equal(c.endpointSession, "2026-09-10"); assert.equal(c.endpoint, null);
  assert.equal(c.endpointState, "Unavailable"); assert.equal(c.missingSessions, 1);
});
test("entries outside historical coverage remain unavailable", () => {
  const c = buildTradeMarketContext({ ...trade, entryDate: "2024-01-01" }, archive, new Date("2026-09-14T15:00Z"));
  assert.equal(c.entryState, "Unavailable"); assert.equal(c.entrySession, null);
});
test("reconstructed historical gauges cannot use future price bars", () => {
  const s = snapshot("2026-09-10", "Grow");
  assert.equal(s.gauge.session_date, "2026-09-10");
  assert.equal(s.gauge.index_regimes.every((r) => r.date === "2026-09-10"), true);
  const noData = calculateGauge({}, "2026-09-10"); assert.equal(noData, null);
  const symbols = s.gauge.universe.indexes.concat(s.gauge.universe.leaders);
  const days = Array.from({ length: 220 }, (_, i) => ({ time: new Date(Date.UTC(2026, 0, i + 1)).toISOString().slice(0, 10), open: 100, high: 102, low: 99, close: 101 + i / 10 }));
  const session = days.at(-1)!.time;
  const before = calculateGauge(Object.fromEntries(symbols.map((symbol) => [symbol, days])), session)!;
  const future = { time: "2026-12-31", open: 1, high: 1, low: 1, close: 1 };
  const after = calculateGauge(Object.fromEntries(symbols.map((symbol) => [symbol, [...days, future]])), session)!;
  assert.deepEqual(after.index_regimes, before.index_regimes);
  assert.deepEqual(after.components, before.components);
  assert.equal(after.overall_state, before.overall_state);
});
test("archive writes are idempotent, retain immutable snapshots, and can fill gaps", async () => {
  const cwd = process.cwd(); const db = process.env.DATABASE_URL;
  const tmp = await fs.mkdtemp(path.join(os.tmpdir(), "market-archive-test-"));
  delete process.env.DATABASE_URL; process.chdir(tmp);
  try {
    const first = snapshot("2026-09-04", "Grow");
    await saveMarketArchive(["2026-09-04", "2026-09-08"], [first]);
    await Promise.all([saveMarketArchive(["2026-09-04"], [snapshot("2026-09-04", "Protect")]), saveMarketArchive(["2026-09-08"], [snapshot("2026-09-08", "Neutral")])]);
    const saved = await readMarketArchive();
    assert.equal(saved.snapshots.length, 2);
    assert.deepEqual(saved.snapshots.find((s) => s.session === first.session), first);
    assert.deepEqual(saved.sessions, ["2026-09-04", "2026-09-08"]);
  } finally { process.chdir(cwd); if (db !== undefined) process.env.DATABASE_URL = db; await fs.rm(tmp, { recursive: true }); }
});

test("refresh archives only completed full-universe sessions, retaining known gaps", () => {
  const symbols = archive.snapshots[0].gauge.universe.indexes.concat(archive.snapshots[0].gauge.universe.leaders);
  const days = Array.from({ length: 205 }, (_, i) => ({ time: new Date(Date.UTC(2026, 0, i + 1)).toISOString().slice(0, 10), open: 100, high: 102, low: 99, close: 101 }));
  const today = days.at(-1)!.time;
  const data = Object.fromEntries(symbols.map((symbol) => [symbol, days]));
  const empty: MarketArchive = { sessions: [], snapshots: [], refreshedAt: null };
  const beforeClose = buildArchiveUpdate(data, empty, new Date(`${today}T15:00:00Z`));
  assert.equal(beforeClose.sessions.includes(today), false);
  const complete = buildArchiveUpdate(data, empty, new Date(`${today}T22:30:00Z`));
  assert.equal(complete.snapshots.at(-1)?.source, "Archived");
  assert.equal(complete.snapshots[0].source, "Reconstructed");
  const repeat = buildArchiveUpdate(data, { ...empty, snapshots: complete.snapshots }, new Date(`${today}T22:31:00Z`));
  assert.equal(repeat.snapshots.length, 0);
  const incomplete = buildArchiveUpdate({ ...data, APP: [] }, empty, new Date(`${today}T22:30:00Z`));
  assert.equal(incomplete.sessions.includes(today), true); assert.equal(incomplete.snapshots.length, 0);
  assert.throws(() => buildArchiveUpdate({}, empty), /No completed/);
});
