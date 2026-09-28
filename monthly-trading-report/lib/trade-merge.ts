import type { TradeLogEntry, TradeExecution } from "./types";

export const MERGED_TRADE_SOURCE = "manual-merge";
export function executionMergeKey(execution: TradeExecution) {
  return `${execution.sourceKey || execution.id}|${execution.type}|${execution.side}`;
}

export function validateMergeSelection(trades: TradeLogEntry[]) {
  if (trades.length < 2) throw new Error("Select at least two trades to merge.");
  const first = trades[0];
  if (trades.some(t => t.hidden || t.groupRole !== "none" || t.importSource === MERGED_TRADE_SOURCE)) {
    throw new Error("Undo existing merges before merging these trades.");
  }
  if (trades.some(t => t.userId !== first.userId || t.symbol !== first.symbol || t.side !== first.side || t.portfolioTag !== first.portfolioTag)) {
    throw new Error("Selected trades must have the same ticker, direction, and portfolio.");
  }
}

export function buildMergedTrade(members: TradeLogEntry[], id: string, previous?: TradeLogEntry): TradeLogEntry {
  const sorted = [...members].sort((a, b) => `${a.entryDate} ${a.openTime}`.localeCompare(`${b.entryDate} ${b.openTime}`) || a.id.localeCompare(b.id));
  if (!sorted.length) throw new Error("Merged trade has no source records.");
  const first = sorted[0];
  const executions = sorted.flatMap(t => t.executions).sort((a, b) => `${a.date} ${a.time}`.localeCompare(`${b.date} ${b.time}`));
  const entries = executions.filter(e => e.type === "ENTRY");
  const exits = executions.filter(e => e.type === "EXIT");
  const unknownQuantity = executions.some(e => e.shares <= 0) || sorted.some(t => t.customTags.includes("Quantity unavailable"));
  const entryShares = entries.reduce((sum, e) => sum + e.shares, 0);
  const entryValue = entries.reduce((sum, e) => sum + e.shares * e.price, 0);
  const exitShares = exits.reduce((sum, e) => sum + e.shares, 0);
  const hasCompleteExecutions = sorted.every(t => t.executions.length > 0) && !unknownQuantity && entryShares > 0 && exitShares <= entryShares + 0.000001;
  const open = sorted.some(t => t.status === "OPEN");
  const pnl = Math.round(sorted.reduce((sum, t) => sum + t.pnl, 0) * 100) / 100;
  const unique = (key: "setupTags" | "mistakeTags" | "customTags" | "screenshots" | "chartLinks") => Array.from(new Set(sorted.flatMap(t => t[key])));
  const last = [...sorted].sort((a, b) => `${b.exitDate} ${b.closeTime}`.localeCompare(`${a.exitDate} ${a.closeTime}`))[0];
  const entryDate = entries[0]?.date || first.entryDate;
  const exitDate = open ? "" : exits.at(-1)?.date || last.exitDate;
  const reviewSource = previous || sorted.find(t => t.risk || t.reviewSections || t.notes || t.setupTags.length) || first;
  const risk = reviewSource.risk;
  return {
    ...first,
    id, importSource: MERGED_TRADE_SOURCE, importRowKey: id, groupId: id, groupRole: "parent",
    hidden: previous?.hidden || false,
    status: open ? "OPEN" : pnl > 0 ? "WIN" : pnl < 0 ? "LOSS" : "BREAKEVEN",
    entryDate, openTime: entries[0]?.time || first.openTime,
    exitDate, closeTime: open ? "" : exits.at(-1)?.time || last.closeTime,
    avgEntry: hasCompleteExecutions ? entryValue / entryShares : 0,
    exitPrice: !unknownQuantity && exitShares ? exits.reduce((sum, e) => sum + e.price * e.shares, 0) / exitShares : 0,
    shares: open ? sorted.filter(t => t.status === "OPEN").reduce((sum, t) => sum + t.shares, 0) : hasCompleteExecutions ? entryShares : 0,
    pnl, commission: sorted.reduce((sum, t) => sum + t.commission, 0),
    usedMargin: sorted.filter(t => t.status === "OPEN").reduce((sum, t) => sum + t.usedMargin, 0),
    risk, rMultiple: risk > 0 ? pnl / risk : 0,
    returnPercent: hasCompleteExecutions && entryValue ? pnl / entryValue * 100 : 0,
    daysInTrade: exitDate ? Math.max(0, Math.round((Date.parse(exitDate) - Date.parse(entryDate)) / 86400000)) : first.daysInTrade,
    setupTags: previous?.setupTags || unique("setupTags"), mistakeTags: previous?.mistakeTags || unique("mistakeTags"),
    customTags: Array.from(new Set([...(previous?.customTags || []), ...unique("customTags"), "Merged trade", ...(!hasCompleteExecutions ? ["Needs review", "Quantity unavailable"] : [])])),
    screenshots: Array.from(new Set([...(previous?.screenshots || []), ...unique("screenshots")])),
    chartLinks: Array.from(new Set([...(previous?.chartLinks || []), ...unique("chartLinks")])),
    notes: previous?.notes || sorted.filter(t => t.notes).map(t => `${t.entryDate}: ${t.notes}`).join("\n\n"),
    reviewSections: previous?.reviewSections || Object.fromEntries(["setup", "entry", "exit", "didRight", "didWrong", "general"].map(key => [key, sorted.map(t => t.reviewSections?.[key as keyof NonNullable<TradeLogEntry["reviewSections"]>]).filter(Boolean).join("\n\n")])) as TradeLogEntry["reviewSections"],
    manualGrade: reviewSource.manualGrade, checklistItems: reviewSource.checklistItems,
    emotion: reviewSource.emotion, tradeQuality: reviewSource.tradeQuality,
    executions, createdAt: previous?.createdAt || new Date().toISOString(), updatedAt: new Date().toISOString()
  };
}

/** Restore durable membership after the CF importer replaces its raw trade rows. */
export function reconcileTradeMerges(trades: TradeLogEntry[]): TradeLogEntry[] {
  const parents = trades.filter(t => t.importSource === MERGED_TRADE_SOURCE);
  const claims = new Map<string, string>();
  let result = [...trades];
  for (const parent of parents) {
    const keys = new Set(parent.executions.map(executionMergeKey));
    const members = trades.filter(t => t.id !== parent.id && t.importSource !== MERGED_TRADE_SOURCE && t.userId === parent.userId && t.portfolioTag === parent.portfolioTag && t.symbol === parent.symbol && t.side === parent.side &&
      (t.groupId === parent.id || t.executions.some(e => keys.has(executionMergeKey(e)))));
    const foundKeys = new Set(members.flatMap(t => t.executions.map(executionMergeKey)));
    if ([...keys].some(key => !foundKeys.has(key))) throw new Error(`Statement omits executions from merged ${parent.symbol}. Undo the merge before importing this statement.`);
    if (!members.length) throw new Error(`Cannot reconcile merged ${parent.symbol} trade. Undo the merge before importing this statement.`);
    for (const member of members) {
      if (claims.has(member.id)) throw new Error(`Statement combines multiple merged ${parent.symbol} trades. Undo those merges before importing.`);
      claims.set(member.id, parent.id);
    }
    const ids = new Set(members.map(t => t.id));
    const merged = buildMergedTrade(members, parent.id, parent);
    result = result.map(t => t.id === parent.id ? merged : ids.has(t.id) ? { ...t, hidden: true, groupId: parent.id, groupRole: "child" as const } : t);
  }
  return result;
}

export function mergeTradeRecords(trades: TradeLogEntry[], ids: string[], userId: string, id: string) {
  if (new Set(ids).size !== ids.length) throw new Error("Duplicate trade selection.");
  const members = ids.map(key => trades.find(t => t.id === key && t.userId === userId));
  if (members.some(t => !t)) throw new Error("One or more selected trades are unavailable.");
  const selected = members as TradeLogEntry[];
  validateMergeSelection(selected);
  const merged = buildMergedTrade(selected, id);
  return [...trades.map(t => ids.includes(t.id) ? { ...t, hidden: true, groupId: id, groupRole: "child" as const } : t), merged];
}

export function undoTradeMerge(trades: TradeLogEntry[], id: string, userId: string) {
  if (!trades.some(t => t.id === id && t.userId === userId && t.importSource === MERGED_TRADE_SOURCE)) throw new Error("Merged trade not found.");
  return trades.filter(t => t.id !== id).map(t => t.userId === userId && t.groupId === id ? { ...t, hidden: false, groupId: "", groupRole: "none" as const } : t);
}
