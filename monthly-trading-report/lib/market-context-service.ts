import { calculateGauge, GAUGE_VERSION, loadGaugeCandles, type Candle } from "./market-gauge";
import { readMarketArchive, saveMarketArchive } from "./market-context-store";
import { marketClock, type MarketArchive, type MarketSnapshot } from "./trade-market-context";

export function buildArchiveUpdate(data: Record<string, Candle[]>, existing: MarketArchive, now = new Date()) {
  const clock = marketClock(now);
  const sessions = [...new Set((data.SPY || []).filter((b) => b.time < clock.date || (clock.finalized && b.time === clock.date)).map((b) => b.time))].sort();
  if (!sessions.length) throw new Error("No completed market sessions available; existing history retained.");
  const known = new Set(existing.snapshots.map((s) => s.session));
  const snapshots: MarketSnapshot[] = [];
  for (const session of sessions) {
    if (known.has(session)) continue;
    const gauge = calculateGauge(data, session);
    if (!gauge || gauge.leadership.total !== gauge.universe.leader_count) continue;
    snapshots.push({ session, version: GAUGE_VERSION, recordedAt: now.toISOString(),
      source: session === clock.date ? "Archived" : "Reconstructed", gauge });
  }
  return { sessions, snapshots };
}

let inFlight: Promise<{ saved: number; sessions: number }> | null = null;
export function refreshMarketArchive(now = new Date()) {
  if (inFlight) return inFlight;
  inFlight = (async () => {
    const existing = await readMarketArchive();
    const data = await loadGaugeCandles(true);
    const { sessions, snapshots } = buildArchiveUpdate(data, existing, now);
    await saveMarketArchive(sessions, snapshots);
    return { saved: snapshots.length, sessions: sessions.length };
  })().finally(() => { inFlight = null; });
  return inFlight;
}
