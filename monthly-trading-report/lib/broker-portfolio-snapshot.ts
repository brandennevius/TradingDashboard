import type { CfWorkingOrderMetadata } from "./cf-import-idempotency";
import type { ParsedOpenPositionRow } from "./cf-statement";

export type BrokerPortfolioSnapshotSource = "CF_STATEMENT_PDF";

export type BrokerPortfolioPosition = {
  symbol: string;
  side: "LONG" | "SHORT";
  shares: number;
  entryDate: string;
  entryPrice: number | null;
  currentPrice: number | null;
  usedMargin: number | null;
  stopPrice: number | null;
  takeProfitPrice: number | null;
  floatingPnl: number | null;
  commission: number | null;
};

export type BrokerPortfolioSnapshotInput = {
  userId: string;
  portfolioTag: string;
  coverageDate: string;
  sourceHash: string;
  sourceFilename?: string;
  source?: BrokerPortfolioSnapshotSource;
  balance?: number;
  currentEquity?: number;
  statementEquity?: number;
  floatingPnl?: number;
  openPositions?: ParsedOpenPositionRow[];
  workingOrders?: CfWorkingOrderMetadata[];
};

export type BrokerPortfolioSnapshot = {
  userId: string;
  portfolioTag: string;
  coverageDate: string;
  sourceHash: string;
  sourceFilename: string;
  source: BrokerPortfolioSnapshotSource;
  importedAt: string;
  balance: number | null;
  currentEquity: number | null;
  statementEquity: number | null;
  floatingPnl: number | null;
  openPositions: BrokerPortfolioPosition[];
  workingOrders: CfWorkingOrderMetadata[];
};

export type DailyReviewProvenance = {
  kind: "BROKER_SNAPSHOT" | "BROKER_ANCHORED_RECONSTRUCTION" | "EXECUTION_CLOSE_ESTIMATE";
  label: "BROKER SNAPSHOT" | "BROKER-ANCHORED RECONSTRUCTION" | "ESTIMATED";
  selectedDate: string;
  anchorCoverageDate: string | null;
  accountEquity: number | null;
  snapshot: BrokerPortfolioSnapshot | null;
};

function nullablePositive(value: unknown) {
  const parsed = Number(value);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : null;
}
function nullableNumber(value: unknown) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function normalizedDate(value: unknown) {
  const text = String(value || "").trim();
  return /^\d{4}-\d{2}-\d{2}$/.test(text) ? text : "";
}

function normalizePositions(value: unknown): BrokerPortfolioPosition[] {
  if (!Array.isArray(value)) return [];
  return value.flatMap((item) => {
    if (!item || typeof item !== "object" || Array.isArray(item)) return [];
    const row = item as Record<string, unknown>;
    const symbol = String(row.symbol || "").trim().toUpperCase();
    const side = row.side === "SHORT" ? "SHORT" : "LONG";
    const shares = Number(row.shares);
    if (!symbol || !Number.isFinite(shares) || shares <= 0) return [];
    return [{
      symbol,
      side,
      shares,
      entryDate: normalizedDate(row.entryDate),
      entryPrice: nullablePositive(row.entryPrice),
      currentPrice: nullablePositive(row.currentPrice),
      usedMargin: nullableNumber(row.usedMargin),
      stopPrice: nullablePositive(row.stopPrice),
      takeProfitPrice: nullablePositive(row.takeProfitPrice),
      floatingPnl: nullableNumber(row.floatingPnl),
      commission: nullableNumber(row.commission)
    } satisfies BrokerPortfolioPosition];
  }).sort((a, b) => a.symbol.localeCompare(b.symbol) || a.side.localeCompare(b.side));
}

function normalizeOrders(value: unknown): CfWorkingOrderMetadata[] {
  if (!Array.isArray(value)) return [];
  return value.flatMap((item) => {
    if (!item || typeof item !== "object" || Array.isArray(item)) return [];
    const row = item as Record<string, unknown>;
    const direction = row.direction === "Buy" ? "Buy" : row.direction === "Sell" ? "Sell" : "";
    const shares = Number(row.shares);
    const orderPrice = Number(row.orderPrice);
    if (!direction || !Number.isFinite(shares) || shares <= 0 || !Number.isFinite(orderPrice) || orderPrice <= 0) return [];
    return [{
      orderId: String(row.orderId || ""),
      orderDate: normalizedDate(row.orderDate),
      timeValue: String(row.timeValue || ""),
      direction,
      shares,
      symbol: String(row.symbol || "").trim().toUpperCase(),
      orderType: String(row.orderType || "").trim().toUpperCase(),
      orderPrice
    } satisfies CfWorkingOrderMetadata];
  }).sort((a, b) => a.orderId.localeCompare(b.orderId) || a.symbol.localeCompare(b.symbol));
}

export function normalizeBrokerPortfolioSnapshot(
  input: BrokerPortfolioSnapshotInput | BrokerPortfolioSnapshot,
  importedAt = "importedAt" in input ? input.importedAt : new Date().toISOString()
): BrokerPortfolioSnapshot {
  const coverageDate = normalizedDate(input.coverageDate);
  const sourceHash = String(input.sourceHash || "").trim().toLowerCase();
  if (!input.userId.trim() || !input.portfolioTag.trim() || !coverageDate) {
    throw new Error("Broker portfolio snapshot requires user, portfolio, and a valid coverage date.");
  }
  if (!/^[a-f0-9]{64}$/.test(sourceHash)) {
    throw new Error("Broker portfolio snapshot requires a SHA-256 source hash.");
  }

  return {
    userId: input.userId.trim(),
    portfolioTag: input.portfolioTag.trim(),
    coverageDate,
    sourceHash,
    sourceFilename: String(input.sourceFilename || "").trim().slice(0, 255),
    source: "CF_STATEMENT_PDF",
    importedAt,
    balance: nullablePositive(input.balance),
    currentEquity: nullablePositive(input.currentEquity),
    statementEquity: nullablePositive(input.statementEquity),
    floatingPnl: nullableNumber(input.floatingPnl),
    openPositions: normalizePositions(input.openPositions),
    workingOrders: normalizeOrders(input.workingOrders)
  };
}

export function upsertBrokerPortfolioSnapshotCollection(
  current: BrokerPortfolioSnapshot[],
  input: BrokerPortfolioSnapshotInput,
  importedAt = new Date().toISOString()
) {
  const incoming = normalizeBrokerPortfolioSnapshot(input, importedAt);
  const key = `${incoming.userId}\u0000${incoming.portfolioTag}\u0000${incoming.coverageDate}`;
  const existing = current.find((snapshot) => `${snapshot.userId}\u0000${snapshot.portfolioTag}\u0000${snapshot.coverageDate}` === key);
  const normalizedIncoming = existing?.sourceHash === incoming.sourceHash
    ? { ...incoming, importedAt: existing.importedAt }
    : incoming;
  return [
    ...current.filter((snapshot) => `${snapshot.userId}\u0000${snapshot.portfolioTag}\u0000${snapshot.coverageDate}` !== key),
    normalizedIncoming
  ].sort((a, b) => a.coverageDate.localeCompare(b.coverageDate));
}

export function resolveDailyReviewProvenance(
  snapshots: BrokerPortfolioSnapshot[],
  portfolioTag: string,
  selectedDate: string
): DailyReviewProvenance {
  const eligible = snapshots
    .filter((snapshot) => snapshot.portfolioTag === portfolioTag && snapshot.coverageDate <= selectedDate)
    .sort((a, b) => b.coverageDate.localeCompare(a.coverageDate));
  const snapshot = eligible[0] || null;
  if (snapshot?.coverageDate === selectedDate) {
    return {
      kind: "BROKER_SNAPSHOT",
      label: "BROKER SNAPSHOT",
      selectedDate,
      anchorCoverageDate: snapshot.coverageDate,
      accountEquity: snapshot.currentEquity ?? snapshot.statementEquity,
      snapshot
    };
  }
  if (snapshot) {
    return {
      kind: "BROKER_ANCHORED_RECONSTRUCTION",
      label: "BROKER-ANCHORED RECONSTRUCTION",
      selectedDate,
      anchorCoverageDate: snapshot.coverageDate,
      accountEquity: null,
      snapshot
    };
  }
  return {
    kind: "EXECUTION_CLOSE_ESTIMATE",
    label: "ESTIMATED",
    selectedDate,
    anchorCoverageDate: null,
    accountEquity: null,
    snapshot: null
  };
}
