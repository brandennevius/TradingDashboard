import type { TradeExecution, TradeLogEntry, TradeLogInput } from "./types";

export type CfStatementReplacementTrade = TradeLogInput & { id?: string };

export type CfLifecycleReconciliationDecision = {
  rebuiltImportRowKey: string;
  symbol: string;
  side: TradeLogInput["side"];
  status: TradeLogInput["status"];
  action: "EXACT_IMPORT_KEY" | "EXECUTION_OVERLAP" | "OPEN_CONTINUATION" | "NEW_LIFECYCLE" | "AMBIGUOUS";
  matchedTradeId: string | null;
  candidateTradeIds: string[];
  reason: string;
};

export type CfLifecycleReconciliationResult = {
  trades: CfStatementReplacementTrade[];
  decisions: CfLifecycleReconciliationDecision[];
  adoptedNonCfTradeIds: string[];
  ambiguities: CfLifecycleReconciliationDecision[];
};

const CF_SYSTEM_TAGS = new Set([
  "CF Statement",
  "Open Position",
  "Closed Transaction",
  "Partial exits",
  "Needs review",
  "Combined trade",
  "Auto recalculated"
]);

function executionKeys(executions: TradeExecution[] | undefined) {
  return new Set((executions || []).map((execution) => String(execution.sourceKey || execution.id || "").trim()).filter(Boolean));
}

function executionOverlap(left: TradeExecution[] | undefined, right: TradeExecution[] | undefined) {
  const rightKeys = executionKeys(right);
  return Array.from(executionKeys(left)).filter((key) => rightKeys.has(key)).length;
}

function matchingIdentity(trade: TradeLogEntry, rebuilt: TradeLogInput) {
  return (
    trade.userId === rebuilt.userId &&
    trade.portfolioTag === rebuilt.portfolioTag &&
    trade.symbol === rebuilt.symbol &&
    trade.side === rebuilt.side
  );
}

function preserveJournalFields(rebuiltTrade: TradeLogInput, sources: TradeLogEntry[]): CfStatementReplacementTrade {
  return sources.reduce<CfStatementReplacementTrade>((next, existing) => {
    const userTags = existing.customTags.filter((tag) => !CF_SYSTEM_TAGS.has(tag));
    return {
      ...next,
      risk: existing.risk || next.risk,
      setupTags: existing.setupTags.length ? existing.setupTags : next.setupTags,
      mistakeTags: existing.mistakeTags.length ? existing.mistakeTags : next.mistakeTags,
      customTags: Array.from(new Set([...next.customTags, ...userTags])),
      manualGrade: existing.manualGrade || next.manualGrade,
      emotion: existing.emotion || next.emotion,
      tradeQuality: existing.tradeQuality || next.tradeQuality,
      checklistItems: existing.checklistItems.length ? existing.checklistItems : next.checklistItems,
      notes: existing.notes || next.notes,
      reviewSections: existing.reviewSections || next.reviewSections,
      screenshots: existing.screenshots.length ? existing.screenshots : next.screenshots,
      chartLinks: existing.chartLinks.length ? existing.chartLinks : next.chartLinks,
      hidden: existing.importSource === "cf-statement-pdf" ? existing.hidden : next.hidden
    };
  }, rebuiltTrade);
}

function strongestExecutionMatch(rebuilt: TradeLogInput, candidates: TradeLogEntry[]) {
  const ranked = candidates
    .map((trade) => ({ trade, overlap: executionOverlap(trade.executions, rebuilt.executions) }))
    .filter((candidate) => candidate.overlap > 0)
    .sort((a, b) => b.overlap - a.overlap || a.trade.id.localeCompare(b.trade.id));
  if (!ranked.length) return { match: null, ambiguous: [] as TradeLogEntry[] };
  const strongest = ranked.filter((candidate) => candidate.overlap === ranked[0].overlap).map((candidate) => candidate.trade);
  return strongest.length === 1
    ? { match: strongest[0], ambiguous: [] as TradeLogEntry[] }
    : { match: null, ambiguous: strongest };
}

/** Broker executions and ending positions define identity. Review state never does. */
export function reconcileCfStatementLifecycles(
  rebuiltTrades: TradeLogInput[],
  existingTrades: TradeLogEntry[]
): CfLifecycleReconciliationResult {
  const usedTradeIds = new Set<string>();
  const decisions: CfLifecycleReconciliationDecision[] = [];
  const adoptedNonCfTradeIds = new Set<string>();

  const trades = rebuiltTrades.map((rebuilt): CfStatementReplacementTrade => {
    const eligible = existingTrades.filter((trade) => !trade.hidden && !usedTradeIds.has(trade.id) && matchingIdentity(trade, rebuilt));
    const rebuiltIdentityCount = rebuiltTrades.filter(
      (candidate) => candidate.symbol === rebuilt.symbol && candidate.side === rebuilt.side
    ).length;
    const exact = eligible.filter(
      (trade) => Boolean(rebuilt.importRowKey) && trade.importSource === "cf-statement-pdf" && trade.importRowKey === rebuilt.importRowKey
    );
    const manualOpen = rebuilt.status === "OPEN"
      ? eligible.filter(
          (trade) =>
            trade.importSource !== "cf-statement-pdf" &&
            trade.status === "OPEN" &&
            (trade.entryDate === rebuilt.entryDate || rebuiltIdentityCount === 1)
        )
      : [];

    if (exact.length > 1) {
      const decision: CfLifecycleReconciliationDecision = {
        rebuiltImportRowKey: rebuilt.importRowKey,
        symbol: rebuilt.symbol,
        side: rebuilt.side,
        status: rebuilt.status,
        action: "AMBIGUOUS",
        matchedTradeId: null,
        candidateTradeIds: exact.map((trade) => trade.id).sort(),
        reason: "Multiple persisted broker rows have the same lifecycle key."
      };
      decisions.push(decision);
      return rebuilt;
    }

    if (manualOpen.length > 1) {
      const decision: CfLifecycleReconciliationDecision = {
        rebuiltImportRowKey: rebuilt.importRowKey,
        symbol: rebuilt.symbol,
        side: rebuilt.side,
        status: rebuilt.status,
        action: "AMBIGUOUS",
        matchedTradeId: null,
        candidateTradeIds: manualOpen.map((trade) => trade.id).sort(),
        reason: "Multiple open journal rows match the same broker position; no row was selected automatically."
      };
      decisions.push(decision);
      return rebuilt;
    }

    const executionMatch = strongestExecutionMatch(rebuilt, eligible.filter((trade) => trade.importSource === "cf-statement-pdf"));
    if (executionMatch.ambiguous.length) {
      const decision: CfLifecycleReconciliationDecision = {
        rebuiltImportRowKey: rebuilt.importRowKey,
        symbol: rebuilt.symbol,
        side: rebuilt.side,
        status: rebuilt.status,
        action: "AMBIGUOUS",
        matchedTradeId: null,
        candidateTradeIds: executionMatch.ambiguous.map((trade) => trade.id).sort(),
        reason: "Multiple broker lifecycles contain the same strongest execution overlap."
      };
      decisions.push(decision);
      return rebuilt;
    }

    const manual = manualOpen[0] || null;
    const brokerMatch = exact.length === 1 ? exact[0] : executionMatch.match;
    const matched = manual || brokerMatch;
    const journalSources = [brokerMatch, manual].filter((trade): trade is TradeLogEntry => Boolean(trade));

    if (!matched) {
      decisions.push({
        rebuiltImportRowKey: rebuilt.importRowKey,
        symbol: rebuilt.symbol,
        side: rebuilt.side,
        status: rebuilt.status,
        action: "NEW_LIFECYCLE",
        matchedTradeId: null,
        candidateTradeIds: [],
        reason: "No persisted lifecycle has an exact broker key, execution overlap, or unique open-position continuation."
      });
      return rebuilt;
    }

    usedTradeIds.add(matched.id);
    if (brokerMatch) usedTradeIds.add(brokerMatch.id);
    if (manual) adoptedNonCfTradeIds.add(manual.id);
    const action = manual ? "OPEN_CONTINUATION" : exact.length === 1 ? "EXACT_IMPORT_KEY" : "EXECUTION_OVERLAP";
    decisions.push({
      rebuiltImportRowKey: rebuilt.importRowKey,
      symbol: rebuilt.symbol,
      side: rebuilt.side,
      status: rebuilt.status,
      action,
      matchedTradeId: matched.id,
      candidateTradeIds: [matched.id],
      reason: manual
        ? "A single open journal lifecycle matches the broker position by portfolio, symbol, and side."
        : action === "EXACT_IMPORT_KEY"
          ? "The broker lifecycle key is unchanged."
          : "Broker transaction identifiers overlap a persisted lifecycle."
    });

    return { ...preserveJournalFields(rebuilt, journalSources), id: matched.id };
  });

  return {
    trades,
    decisions,
    adoptedNonCfTradeIds: Array.from(adoptedNonCfTradeIds).sort(),
    ambiguities: decisions.filter((decision) => decision.action === "AMBIGUOUS")
  };
}

export function applyManualFieldsToCfStatementTrade(
  rebuiltTrade: TradeLogInput,
  existingTrades: TradeLogEntry[]
): CfStatementReplacementTrade {
  return reconcileCfStatementLifecycles([rebuiltTrade], existingTrades).trades[0];
}
