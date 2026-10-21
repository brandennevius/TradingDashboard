import type { GaugePayload, GaugeState } from "./market-gauge";
import type { TradeLogEntry } from "./types";

export type MarketSnapshot = { session: string; version: string; recordedAt: string; source: "Archived" | "Reconstructed"; gauge: GaugePayload };
export type MarketArchive = { sessions: string[]; snapshots: MarketSnapshot[]; refreshedAt: string | null };
export type MarketSummary = {
  entryState: GaugeState | "Unavailable"; endpointState: GaugeState | "Unavailable";
  entrySession: string | null; endpointSession: string | null; endpointLabel: "Current" | "At exit";
  transition: string; changedDuringTrade: boolean; missingSessions: number; historySessions: number;
};
export type MarketHistoryPoint = Pick<MarketSnapshot, "session" | "version" | "source"> & { state: GaugeState };
export type TradeMarketContext = MarketSummary & { entry: MarketSnapshot | null; endpoint: MarketSnapshot | null; history: MarketHistoryPoint[]; refreshedAt: string | null };

// Daily bars are eligible only after the close plus a 15-minute finalization window.
// Waiting until 16:15 ET on half-days is conservative and avoids treating partial bars as final.
export function marketClock(now = new Date()) {
  const parts = Object.fromEntries(new Intl.DateTimeFormat("en-CA", {
    timeZone: "America/New_York", year: "numeric", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", hourCycle: "h23"
  }).formatToParts(now).map((p) => [p.type, p.value]));
  return { date: `${parts.year}-${parts.month}-${parts.day}`, finalized: Number(parts.hour) * 60 + Number(parts.minute) >= 16 * 60 + 15 };
}

function executionCutoff(date: string, time: string) {
  // Only an explicitly zoned timestamp can establish that the same-day close was known.
  // Date-only and broker-local times conservatively use the preceding session.
  if (time && /(?:Z|[+-]\d{2}:?\d{2})$/.test(time) && Number.isFinite(Date.parse(time))) {
    const clock = marketClock(new Date(time));
    return { date: clock.date, inclusive: clock.finalized };
  }
  return { date, inclusive: false };
}

export function buildTradeMarketContext(trade: Pick<TradeLogEntry, "entryDate" | "exitDate" | "openTime" | "closeTime" | "status">, archive: MarketArchive, now = new Date()): TradeMarketContext {
  const clock = marketClock(now);
  const sessions = [...new Set(archive.sessions)].sort().filter((d) => d < clock.date || (clock.finalized && d === clock.date));
  const entryCutoff = executionCutoff(trade.entryDate, trade.openTime);
  const open = trade.status === "OPEN";
  const endCutoff = open ? { date: clock.date, inclusive: clock.finalized } : executionCutoff(trade.exitDate, trade.closeTime);
  const pick = (cutoff: { date: string; inclusive: boolean }) => cutoff.date
    ? sessions.filter((d) => d < cutoff.date || (cutoff.inclusive && d === cutoff.date)).at(-1) || null : null;
  const entrySession = pick(entryCutoff);
  const endpointSession = pick(endCutoff);
  const bySession = new Map(archive.snapshots.map((s) => [s.session, s]));
  const entry = entrySession ? bySession.get(entrySession) || null : null;
  const endpoint = endpointSession ? bySession.get(endpointSession) || null : null;
  const expected = entrySession && endpointSession ? sessions.filter((d) => d >= entrySession && d <= endpointSession) : [];
  const history = expected.flatMap((d) => {
    const snapshot = bySession.get(d);
    return snapshot ? [{ session: d, version: snapshot.version, source: snapshot.source, state: snapshot.gauge.overall_state }] : [];
  });
  const entryState = entry?.gauge.overall_state || "Unavailable";
  const endpointState = endpoint?.gauge.overall_state || "Unavailable";
  return {
    entryState, endpointState, entrySession, endpointSession, endpointLabel: open ? "Current" : "At exit",
    transition: `${entryState} → ${endpointState}`,
    changedDuringTrade: new Set(history.map((s) => s.state)).size > 1,
    missingSessions: expected.length - history.length, historySessions: expected.length,
    entry, endpoint, history, refreshedAt: archive.refreshedAt
  };
}

export function summarizeMarketContext(context: TradeMarketContext): MarketSummary {
  const { entry: _entry, endpoint: _endpoint, history: _history, refreshedAt: _refreshedAt, ...summary } = context;
  return summary;
}
