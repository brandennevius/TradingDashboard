import type { TradeLogEntry, TradeLogInput } from "./types";

export type CfStatementReplacementTrade = TradeLogInput & { id?: string };

const CF_SYSTEM_TAGS = new Set([
  "CF Statement",
  "Open Position",
  "Closed Transaction",
  "Partial exits",
  "Needs review",
  "Combined trade",
  "Auto recalculated"
]);

function tradeManualKey(trade: Pick<TradeLogEntry | TradeLogInput, "symbol" | "side" | "entryDate" | "openTime">) {
  return [trade.symbol, trade.side, trade.entryDate, trade.openTime || ""].join("|");
}

function hasManualReviewData(trade: TradeLogEntry) {
  return Boolean(
    trade.risk ||
      trade.setupTags.length ||
      trade.mistakeTags.length ||
      trade.manualGrade ||
      trade.emotion ||
      trade.tradeQuality ||
      trade.checklistItems.length ||
      trade.notes ||
      trade.reviewSections ||
      trade.screenshots.length ||
      trade.chartLinks.length
  );
}

function findManualOpenAdoptionCandidate(rebuiltTrade: TradeLogInput, existingTrades: TradeLogEntry[]) {
  if (rebuiltTrade.status !== "OPEN") return null;

  const candidates = existingTrades.filter(
    (trade) =>
      !trade.hidden &&
      trade.importSource !== "cf-statement-pdf" &&
      trade.userId === rebuiltTrade.userId &&
      trade.portfolioTag === rebuiltTrade.portfolioTag &&
      trade.status === "OPEN" &&
      trade.symbol === rebuiltTrade.symbol &&
      trade.side === rebuiltTrade.side &&
      trade.entryDate === rebuiltTrade.entryDate &&
      hasManualReviewData(trade)
  );

  if (candidates.length === 1) return candidates[0];

  const exactTimeCandidates = candidates.filter((trade) => tradeManualKey(trade) === tradeManualKey(rebuiltTrade));
  return exactTimeCandidates.length === 1 ? exactTimeCandidates[0] : null;
}

export function applyManualFieldsToCfStatementTrade(
  rebuiltTrade: TradeLogInput,
  existingTrades: TradeLogEntry[]
): CfStatementReplacementTrade {
  const exact = existingTrades.find((trade) => trade.importRowKey === rebuiltTrade.importRowKey);
  const fallback = existingTrades.find((trade) => tradeManualKey(trade) === tradeManualKey(rebuiltTrade));
  const manualOpenAdoption = findManualOpenAdoptionCandidate(rebuiltTrade, existingTrades);
  const existing = exact || fallback || manualOpenAdoption;

  if (!existing) {
    return rebuiltTrade;
  }

  const manualTags = existing.customTags.filter((tag) => !CF_SYSTEM_TAGS.has(tag));

  return {
    ...rebuiltTrade,
    id: manualOpenAdoption?.id,
    risk: existing.risk || rebuiltTrade.risk,
    setupTags: existing.setupTags.length ? existing.setupTags : rebuiltTrade.setupTags,
    mistakeTags: existing.mistakeTags.length ? existing.mistakeTags : rebuiltTrade.mistakeTags,
    customTags: Array.from(new Set([...rebuiltTrade.customTags, ...manualTags])),
    manualGrade: existing.manualGrade || rebuiltTrade.manualGrade,
    emotion: existing.emotion || rebuiltTrade.emotion,
    tradeQuality: existing.tradeQuality || rebuiltTrade.tradeQuality,
    checklistItems: existing.checklistItems.length ? existing.checklistItems : rebuiltTrade.checklistItems,
    notes: existing.notes || rebuiltTrade.notes,
    reviewSections: existing.reviewSections || rebuiltTrade.reviewSections,
    screenshots: existing.screenshots.length ? existing.screenshots : rebuiltTrade.screenshots,
    chartLinks: existing.chartLinks.length ? existing.chartLinks : rebuiltTrade.chartLinks,
    hidden: exact ? existing.hidden : rebuiltTrade.hidden
  };
}

