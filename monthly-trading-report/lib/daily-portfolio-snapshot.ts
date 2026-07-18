import crypto from "node:crypto";
import type { SetupChecklistTemplate, TradeChecklistItem, TradeExecution, TradeLogEntry } from "./types";
import { hasTradeReviewContent, resolvedTradeReviewSections } from "./trade-review";

export const DAILY_PORTFOLIO_SNAPSHOT_SCHEMA_VERSION = "daily-portfolio-snapshot-v1";

export type SnapshotWarningCode =
  | "BROKER_IMPORT_MISSING"
  | "BROKER_IMPORT_STALE"
  | "BROKER_IMPORT_INCOMPLETE"
  | "PORTFOLIO_DATA_STALE"
  | "CURRENT_PRICE_STALE"
  | "MISSING_STOP"
  | "MISSING_INITIAL_RISK"
  | "MISSING_SETUP"
  | "MISSING_GRADE"
  | "MISSING_NOTES"
  | "MISSING_CHART"
  | "MISSING_SCREENSHOT"
  | "MISSING_EXECUTIONS"
  | "MISSING_EARNINGS_DATE"
  | "GRADE_CRITERIA_CONFLICT"
  | "R_MULTIPLE_UNAVAILABLE"
  | "MFE_UNAVAILABLE"
  | "MAE_UNAVAILABLE"
  | "POSITION_CALCULATION_MISMATCH"
  | "PARTIAL_EXIT_CALCULATION_MISMATCH"
  | "REQUESTED_SESSION_ADJUSTED"
  | "PRICE_DATA_UNAVAILABLE"
  | "INITIAL_STOP_UNAVAILABLE"
  | "STOP_HISTORY_UNAVAILABLE"
  | "BROKER_IMPORT_UNRELATED_ROWS_NEED_REVIEW";

export type SnapshotWarning = {
  code: SnapshotWarningCode;
  message: string;
  severity: "critical" | "warning" | "info";
  trade_id?: string;
  ticker?: string;
};

export type DailyPortfolioSnapshotStatus = "COMPLETE" | "COMPLETE_WITH_WARNINGS" | "INCOMPLETE";

export function snapshotStatusFromWarnings(warnings: SnapshotWarning[]): DailyPortfolioSnapshotStatus {
  if (warnings.some((item) => item.severity === "critical")) return "INCOMPLETE";
  if (warnings.length) return "COMPLETE_WITH_WARNINGS";
  return "COMPLETE";
}

export type SnapshotPrice = {
  symbol: string;
  price: number | null;
  timestamp: string | null;
  provider: string;
  sessionDate?: string | null;
  priceType?: "official_close" | "delayed_close" | "last_trade";
  retrievedAt?: string | null;
};

type PortfolioMeta = {
  currentEquity?: number;
  statementEquity?: number;
  floatingPnl?: number;
  equitySource?: string;
  equityUpdatedAt?: string;
  equityStatementDate?: string;
  workingOrders?: Array<{
    orderId: string;
    orderDate: string;
    timeValue: string;
    direction: "Buy" | "Sell";
    shares: number;
    symbol: string;
    orderType: string;
    orderPrice: number;
  }>;
};

type SnapshotInput = {
  requestedSession: string;
  latestCompletedMarketSession: string;
  generatedAt?: string;
  accountName: string;
  portfolioMeta?: PortfolioMeta;
  trades: TradeLogEntry[];
  setupTemplates: SetupChecklistTemplate[];
  prices: Map<string, SnapshotPrice>;
  sourceEnvironment: string;
  applicationVersion: string;
};

function round(value: number | null, digits = 2) {
  if (value === null || !Number.isFinite(value)) return null;
  const factor = 10 ** digits;
  return Math.round(value * factor) / factor;
}

function dateOnly(value: string | undefined) {
  return String(value || "").slice(0, 10);
}

export function newYorkTimestamp(date: string, time: string) {
  if (!/^\d{4}-\d{2}-\d{2}$/.test(date) || !/^\d{2}:\d{2}:\d{2}$/.test(time)) return null;
  const offsetPart = new Intl.DateTimeFormat("en-US", {
    timeZone: "America/New_York", timeZoneName: "longOffset"
  }).formatToParts(new Date(`${date}T12:00:00Z`)).find((part) => part.type === "timeZoneName")?.value;
  return offsetPart ? `${date}T${time}${offsetPart.replace("GMT", "")}` : null;
}

export function marketSessionCloseTimestamp(date: string) {
  return newYorkTimestamp(date, "16:00:00");
}

function nthWeekday(year: number, month: number, weekday: number, nth: number) {
  const first = new Date(Date.UTC(year, month, 1));
  return new Date(Date.UTC(year, month, 1 + ((7 + weekday - first.getUTCDay()) % 7) + (nth - 1) * 7));
}

function lastWeekday(year: number, month: number, weekday: number) {
  const last = new Date(Date.UTC(year, month + 1, 0));
  return new Date(Date.UTC(year, month, last.getUTCDate() - ((7 + last.getUTCDay() - weekday) % 7)));
}

function easterSunday(year: number) {
  const a = year % 19, b = Math.floor(year / 100), c = year % 100, d = Math.floor(b / 4), e = b % 4;
  const f = Math.floor((b + 8) / 25), g = Math.floor((b - f + 1) / 3), h = (19 * a + b - d - g + 15) % 30;
  const i = Math.floor(c / 4), k = c % 4, l = (32 + 2 * e + 2 * i - h - k) % 7, m = Math.floor((a + 11 * h + 22 * l) / 451);
  const month = Math.floor((h + l - 7 * m + 114) / 31) - 1, day = ((h + l - 7 * m + 114) % 31) + 1;
  return new Date(Date.UTC(year, month, day));
}

function observed(date: Date) {
  if (date.getUTCDay() === 6) date.setUTCDate(date.getUTCDate() - 1);
  else if (date.getUTCDay() === 0) date.setUTCDate(date.getUTCDate() + 1);
  return date;
}

export function isCompletedUsTradingSession(value: string) {
  if (!/^\d{4}-\d{2}-\d{2}$/.test(value)) return false;
  const date = new Date(`${value}T12:00:00Z`), day = date.getUTCDay();
  if (day === 0 || day === 6) return false;
  const year = date.getUTCFullYear();
  const holidayDates = [
    observed(new Date(Date.UTC(year, 0, 1))), nthWeekday(year, 0, 1, 3), nthWeekday(year, 1, 1, 3),
    new Date(easterSunday(year).getTime() - 2 * 86400000), lastWeekday(year, 4, 1),
    observed(new Date(Date.UTC(year, 5, 19))), observed(new Date(Date.UTC(year, 6, 4))),
    nthWeekday(year, 8, 1, 1), nthWeekday(year, 10, 4, 4), observed(new Date(Date.UTC(year, 11, 25)))
  ].map((holiday) => holiday.toISOString().slice(0, 10));
  return !holidayDates.includes(value);
}

function previousTradingSession(value: string) {
  const cursor = new Date(`${value}T12:00:00Z`);
  do cursor.setUTCDate(cursor.getUTCDate() - 1); while (!isCompletedUsTradingSession(cursor.toISOString().slice(0, 10)));
  return cursor.toISOString().slice(0, 10);
}

function newYorkParts(now: Date) {
  const parts = new Intl.DateTimeFormat("en-CA", { timeZone: "America/New_York", year: "numeric", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", hour12: false }).formatToParts(now);
  return Object.fromEntries(parts.map((part) => [part.type, part.value]));
}

export const SNAPSHOT_SESSION_COMPLETION_TIME = "16:00 America/New_York";

export function latestCompletedMarketSession(now = new Date()) {
  const parts = newYorkParts(now), today = `${parts.year}-${parts.month}-${parts.day}`;
  const minutes = Number(parts.hour) * 60 + Number(parts.minute);
  if (isCompletedUsTradingSession(today) && minutes >= 16 * 60) return today;
  return previousTradingSession(today);
}

export function resolveSnapshotSession(requested: string, now = new Date()) {
  if (!/^\d{4}-\d{2}-\d{2}$/.test(requested)) throw new Error("Session must use YYYY-MM-DD.");
  const parts = newYorkParts(now);
  const latest = latestCompletedMarketSession(now);
  const complete = isCompletedUsTradingSession(requested) && requested <= latest;
  return {
    requested,
    resolved: requested,
    latestCompleted: latest,
    adjusted: false,
    complete,
    currentNewYorkDateTime: `${parts.year}-${parts.month}-${parts.day} ${parts.hour}:${parts.minute}`,
    regularSessionCompletionTime: SNAPSHOT_SESSION_COMPLETION_TIME
  };
}

function warning(code: SnapshotWarningCode, message: string, severity: SnapshotWarning["severity"], trade?: TradeLogEntry): SnapshotWarning {
  return { code, message, severity, ...(trade ? { trade_id: trade.id, ticker: trade.symbol } : {}) };
}

function executionTimestamp(execution: TradeExecution) {
  return `${execution.date || ""}T${execution.time || "00:00:00"}`;
}

function sortedExecutions(trade: TradeLogEntry) {
  return [...(trade.executions || [])].sort((a, b) => executionTimestamp(a).localeCompare(executionTimestamp(b)));
}

function executionShares(trade: TradeLogEntry) {
  return sortedExecutions(trade).reduce((shares, execution) => shares + (execution.type === "ENTRY" ? execution.shares : -execution.shares), 0);
}

function checklistScore(items: TradeChecklistItem[]) {
  const maximum = items.reduce((sum, item) => sum + Math.max(0, Number(item.points || 0)), 0);
  const score = items.reduce((sum, item) => {
    const maximumForItem = Math.max(0, Number(item.points || 0));
    return sum + ((item.inputType || "boolean") === "points" ? Math.max(0, Math.min(maximumForItem, Number(item.score || 0))) : item.met ? maximumForItem : 0);
  }, 0);
  return { score: round(score), maximum: round(maximum), percent: maximum ? round(score / maximum * 100) : null };
}

function gradeFor(trade: TradeLogEntry, templates: SetupChecklistTemplate[]) {
  if (trade.manualGrade.trim()) return trade.manualGrade.trim();
  const setup = trade.setupTags[0] || "", template = templates.find((item) => item.setupName.trim().toLowerCase() === setup.trim().toLowerCase());
  const result = checklistScore(trade.checklistItems);
  if (!template || result.score === null || !result.maximum) return null;
  return [...template.gradeBands].sort((a, b) => b.minScore - a.minScore).find((band) => result.score! >= band.minScore && (band.maxScore === null || result.score! <= band.maxScore))?.label || null;
}

function gradeWarnings(trade: TradeLogEntry, grade: string | null, setup: string | null, scorePercent: number | null, accountValue: number | null) {
  const warnings: SnapshotWarning[] = [], upperGrade = String(grade || "").toUpperCase(), setupLower = String(setup || "").toLowerCase();
  if ((upperGrade === "A" || upperGrade === "A+") && scorePercent !== null && scorePercent < 50) warnings.push(warning("GRADE_CRITERIA_CONFLICT", `${grade} grade conflicts with a ${scorePercent}% setup-criteria score.`, "warning", trade));
  if ((upperGrade === "A" || upperGrade === "A+") && setupLower === "no trade setup") warnings.push(warning("GRADE_CRITERIA_CONFLICT", `${grade} grade conflicts with No Trade Setup.`, "warning", trade));
  if ((upperGrade.startsWith("D") || setupLower.includes("research")) && accountValue && trade.risk / accountValue >= 0.01) warnings.push(warning("GRADE_CRITERIA_CONFLICT", "D-grade or research trade used at least 1% of account value as planned risk.", "warning", trade));
  const claimsCompliance = /system.?compliant/i.test(trade.tradeQuality) || trade.customTags.some((tag) => /system.?compliant/i.test(tag));
  if (claimsCompliance && (!setup || !trade.checklistItems.length)) warnings.push(warning("GRADE_CRITERIA_CONFLICT", "System-compliant label has no setup-criteria evidence.", "warning", trade));
  return warnings;
}

function documentationWarnings(trade: TradeLogEntry, grade: string | null, setup: string | null, scorePercent: number | null, accountValue: number | null, lifecycleState: "open" | "closed") {
  const warnings: SnapshotWarning[] = [];
  if (!trade.stopPrice) warnings.push(warning("MISSING_STOP", lifecycleState === "open" ? "No current stop is stored." : "No stored stop is available for this closed trade.", lifecycleState === "open" ? "critical" : "warning", trade));
  if (!trade.risk) warnings.push(warning("MISSING_INITIAL_RISK", "No planned risk is stored.", lifecycleState === "open" ? "critical" : "warning", trade));
  if (!setup) warnings.push(warning("MISSING_SETUP", "No setup is stored.", "warning", trade));
  if (!grade) warnings.push(warning("MISSING_GRADE", "No grade is stored or derivable.", "warning", trade));
  if (!trade.notes.trim() && !hasTradeReviewContent(resolvedTradeReviewSections(trade.reviewSections, trade.notes))) warnings.push(warning("MISSING_NOTES", "No structured review or legacy notes are stored.", "warning", trade));
  if (!trade.chartLinks.length) warnings.push(warning("MISSING_CHART", "No chart link is stored.", "info", trade));
  if (!trade.screenshots.length) warnings.push(warning("MISSING_SCREENSHOT", "No screenshot is stored.", "info", trade));
  if (!trade.executions.length) warnings.push(warning("MISSING_EXECUTIONS", "No executions are stored.", "critical", trade));
  warnings.push(...gradeWarnings(trade, grade, setup, scorePercent, accountValue));
  return warnings;
}

function protectivePlan(trade: TradeLogEntry, meta: PortfolioMeta | undefined, session: string) {
  const exitDirection = trade.side === "LONG" ? "Sell" : "Buy";
  const sessionOrders = meta?.equityStatementDate === session ? meta.workingOrders || [] : [];
  const orders = sessionOrders.filter((order) =>
    order.symbol.toUpperCase() === trade.symbol.toUpperCase()
    && order.direction === exitDirection
    && order.shares > 0
    && order.orderPrice > 0
  );
  const stopOrders = orders.filter((order) => order.orderType === "STOP");
  const profitTakingOrders = orders.filter((order) => order.orderType === "LIMIT").map((order) => ({
    price: round(order.orderPrice), quantity: round(order.shares, 4), order_type: "LIMIT",
    source: "CF_STATEMENT_WORKING_ORDER", order_id: order.orderId || null, order_date: order.orderDate || null, order_time: order.timeValue || null
  }));

  if (!stopOrders.length) {
    return {
      stopPlanType: trade.stopPrice ? "SINGLE_STORED_STOP" : "UNAVAILABLE",
      protectiveLevels: trade.stopPrice ? [{
        price: round(trade.stopPrice), effective_quantity: round(trade.shares, 4), displayed_order_quantity: round(trade.shares, 4),
        role: "POSITION_STOP", order_type: "STOP"
      }] : [],
      provenance: trade.stopPrice ? {
        source: "TRADE_LOG_STORED_STOP", linkage: null, dynamic_resize: false, statement_coverage_date: meta?.equityStatementDate || null
      } : null,
      profitTakingOrders
    };
  }

  const sortedStops = [...stopOrders].sort((a, b) => trade.side === "LONG" ? b.orderPrice - a.orderPrice : a.orderPrice - b.orderPrice);
  const bracketStop = sortedStops.at(-1)!;
  const stagedStops = sortedStops.slice(0, -1);
  const displayedStopQuantity = stopOrders.reduce((sum, order) => sum + order.shares, 0);
  const stagedLinked = stagedStops.length > 0 && bracketStop.shares >= trade.shares && displayedStopQuantity > trade.shares;
  let remainingQuantity = trade.shares;
  const protectiveLevels: Array<Record<string, string | number | null>> = [];

  for (const order of stagedStops) {
    const effectiveQuantity = Math.min(order.shares, remainingQuantity);
    if (effectiveQuantity <= 0) continue;
    protectiveLevels.push({
      price: round(order.orderPrice), effective_quantity: round(effectiveQuantity, 4), displayed_order_quantity: round(order.shares, 4),
      role: "STAGED_RISK_REDUCTION", order_type: "STOP", order_id: order.orderId || null, order_date: order.orderDate || null, order_time: order.timeValue || null
    });
    remainingQuantity -= effectiveQuantity;
  }
  if (remainingQuantity > 0) {
    protectiveLevels.push({
      price: round(bracketStop.orderPrice), effective_quantity: round(remainingQuantity, 4), displayed_order_quantity: round(bracketStop.shares, 4),
      role: stagedLinked ? "DYNAMIC_REMAINDER_BRACKET_STOP" : "REMAINING_POSITION_STOP", order_type: "STOP",
      order_id: bracketStop.orderId || null, order_date: bracketStop.orderDate || null, order_time: bracketStop.timeValue || null
    });
  }

  return {
    stopPlanType: stagedLinked ? "STAGED_LINKED_EXIT" : protectiveLevels.length > 1 ? "MULTI_LEVEL_PROTECTIVE" : "SINGLE_WORKING_STOP",
    protectiveLevels,
    provenance: {
      source: "CF_STATEMENT_WORKING_ORDERS",
      linkage: stagedLinked ? "BROKER_BRACKET_DYNAMIC_RESIZE" : null,
      dynamic_resize: stagedLinked,
      displayed_stop_quantity: round(displayedStopQuantity, 4),
      effective_protective_quantity: round(trade.shares, 4),
      statement_coverage_date: meta?.equityStatementDate || null
    },
    profitTakingOrders
  };
}

function timeHeld(trade: TradeLogEntry, finalExit: TradeExecution | undefined) {
  const firstEntry = sortedExecutions(trade).find((execution) => execution.type === "ENTRY");
  const start = new Date(`${firstEntry?.date || trade.entryDate}T${firstEntry?.time || trade.openTime || "00:00:00"}Z`).getTime();
  const end = new Date(`${finalExit?.date || trade.exitDate}T${finalExit?.time || trade.closeTime || "00:00:00"}Z`).getTime();
  if (!Number.isFinite(start) || !Number.isFinite(end) || end < start) return null;
  const minutes = Math.round((end - start) / 60000), days = Math.floor(minutes / 1440), hours = Math.floor((minutes % 1440) / 60);
  return `${days}d ${hours}h ${minutes % 60}m`;
}

function brokerMetadata(trades: TradeLogEntry[], meta: PortfolioMeta | undefined, session: string) {
  const imported = trades.filter((trade) => trade.importSource === "cf-statement-pdf");
  const importedAt = [meta?.equityUpdatedAt || "", ...imported.map((trade) => trade.updatedAt)].filter(Boolean).sort().at(-1) || null;
  const statementDate = meta?.equityStatementDate || null;
  const hasBrokerImport = imported.length > 0 || meta?.equitySource === "CF Import";
  const warnings: SnapshotWarning[] = [];
  if (!hasBrokerImport || !importedAt) warnings.push(warning("BROKER_IMPORT_MISSING", "No broker import metadata was found for this account.", "critical"));
  if (statementDate && statementDate < session) warnings.push(warning("BROKER_IMPORT_STALE", `Latest broker statement is ${statementDate}; requested session is ${session}. Exposure percentages exclude stale account value.`, "warning"));
  if (!statementDate && hasBrokerImport) warnings.push(warning("BROKER_IMPORT_INCOMPLETE", "Broker-import metadata has no stored statement date.", "critical"));
  const complete = Boolean(hasBrokerImport && importedAt && statementDate);
  return { importedAt, statementDate, complete, warnings };
}

export function buildDailyPortfolioSnapshot(input: SnapshotInput) {
  const generatedAt = input.generatedAt || new Date().toISOString(), accountValue = input.portfolioMeta?.currentEquity || null;
  const trades = input.trades.filter((trade) => !trade.hidden && (!input.accountName || trade.portfolioTag === input.accountName));
  const broker = brokerMetadata(trades, input.portfolioMeta, input.requestedSession);
  const topWarnings: SnapshotWarning[] = [...broker.warnings];
  const openTrades = trades.filter((trade) => trade.status === "OPEN");

  const openPositions = openTrades.map((trade) => {
    const setup = trade.setupTags[0] || null, score = checklistScore(trade.checklistItems), grade = gradeFor(trade, input.setupTemplates);
    const price = input.prices.get(trade.symbol), priceSession = price?.sessionDate || dateOnly(price?.timestamp || ""), currentPrice = priceSession === input.requestedSession ? price?.price ?? null : null;
    const warnings = documentationWarnings(trade, grade, setup, score.percent, accountValue, "open");
    warnings.push(warning("INITIAL_STOP_UNAVAILABLE", "The repository stores the current stop but not a distinct initial-stop history.", "info", trade));
    warnings.push(warning("STOP_HISTORY_UNAVAILABLE", "Stop-change history is not stored.", "info", trade));
    warnings.push(warning("MISSING_EARNINGS_DATE", "No reliably stored earnings date is available.", "info", trade));
    if (!price?.price) warnings.push(warning("PRICE_DATA_UNAVAILABLE", "No price was available for the requested session.", "critical", trade));
    else if (priceSession !== input.requestedSession) warnings.push(warning("CURRENT_PRICE_STALE", `Latest price session is ${priceSession || "unknown"}; requested session is ${input.requestedSession}.`, "critical", trade));
    if (trade.executions.length && Math.abs(executionShares(trade) - trade.shares) > 0.0001) warnings.push(warning("POSITION_CALCULATION_MISMATCH", `Execution-derived shares ${round(executionShares(trade), 4)} do not match stored shares ${round(trade.shares, 4)}.`, "critical", trade));
    const direction = trade.side === "SHORT" ? -1 : 1, marketValue = currentPrice === null ? null : currentPrice * trade.shares * direction;
    const unrealizedPnl = currentPrice === null ? null : (currentPrice - trade.avgEntry) * trade.shares * direction;
    const currentReturn = currentPrice === null || !trade.avgEntry ? null : (currentPrice - trade.avgEntry) / trade.avgEntry * 100 * direction;
    const realizedPnlToDate = Number(trade.pnl || 0), totalTradePnl = unrealizedPnl === null ? null : realizedPnlToDate + unrealizedPnl;
    const openR = unrealizedPnl === null || !trade.risk ? null : unrealizedPnl / trade.risk;
    const lifecycleR = totalTradePnl === null || !trade.risk ? null : totalTradePnl / trade.risk;
    if (openR === null || lifecycleR === null) warnings.push(warning("R_MULTIPLE_UNAVAILABLE", "Open and lifecycle R multiples require current price and planned risk.", "warning", trade));
    const stopPlan = protectivePlan(trade, input.portfolioMeta, input.requestedSession);
    let remainingRisk: number | null = null;
    if (currentPrice !== null && stopPlan.protectiveLevels.length) {
      const levelRisks = stopPlan.protectiveLevels.map((level) => {
        const price = Number(level.price), quantity = Number(level.effective_quantity);
        const perShare = direction === 1 ? currentPrice - price : price - currentPrice;
        return perShare >= 0 ? perShare * quantity : null;
      });
      remainingRisk = levelRisks.every((value) => value !== null) ? levelRisks.reduce<number>((sum, value) => sum + Number(value), 0) : null;
      if (remainingRisk === null) warnings.push(warning("POSITION_CALCULATION_MISMATCH", "Stored stop is not on the protective side of the current price.", "critical", trade));
    }
    const exits = sortedExecutions(trade).filter((execution) => execution.type === "EXIT");
    return {
      trade_id: trade.id, position_id: trade.id, ticker: trade.symbol, asset_name: null, side: trade.side,
      entry_date: trade.entryDate, average_entry: round(trade.avgEntry), current_price: round(currentPrice), current_price_session: priceSession || null, current_price_timestamp: price?.timestamp || null,
      current_price_source: price?.provider || null, current_price_type: price?.priceType || null, current_price_retrieved_at: price?.retrievedAt || null,
      shares: round(trade.shares, 4), cost_basis: round(trade.avgEntry * trade.shares), market_value: round(marketValue), position_weight_pct: marketValue !== null && accountValue ? round(Math.abs(marketValue) / accountValue * 100) : null,
      unrealized_pnl: round(unrealizedPnl), realized_pnl_to_date: round(realizedPnlToDate), total_trade_pnl: round(totalTradePnl), current_return_pct: round(currentReturn),
      open_r_multiple: round(openR), lifecycle_r_multiple: round(lifecycleR), planned_risk_dollars: trade.risk || null,
      initial_stop: null, current_stop: trade.stopPrice || null, stop_last_updated_at: null, stop_plan_type: stopPlan.stopPlanType,
      protective_levels: stopPlan.protectiveLevels, stop_plan_provenance: stopPlan.provenance,
      remaining_risk_to_stop_dollars: round(remainingRisk), remaining_risk_to_stop_pct: remainingRisk !== null && accountValue ? round(remainingRisk / accountValue * 100) : null,
      take_profit: trade.takeProfitPrice || null, profit_taking_orders: stopPlan.profitTakingOrders, setup, grade, setup_criteria_score: score.score, setup_criteria_max: score.maximum,
      setup_criteria_results: trade.checklistItems, mistake_tags: trade.mistakeTags, system: trade.tradeQuality || null,
      review_sections: resolvedTradeReviewSections(trade.reviewSections, trade.notes), notes: trade.notes || null,
      chart_links: trade.chartLinks, screenshot_references: trade.screenshots, partial_exits: exits, stop_change_history: null, earnings_date: null,
      execution_count: trade.executions.length, executions: sortedExecutions(trade), data_warnings: warnings
    };
  });

  const closedTrades = trades.filter((trade) => {
    if (trade.status === "OPEN") return false;
    const finalExit = sortedExecutions(trade).filter((execution) => execution.type === "EXIT").at(-1);
    return (finalExit?.date || trade.exitDate) === input.requestedSession;
  }).map((trade) => {
    const executions = sortedExecutions(trade), exits = executions.filter((execution) => execution.type === "EXIT"), finalExit = exits.at(-1);
    const setup = trade.setupTags[0] || null, score = checklistScore(trade.checklistItems), grade = gradeFor(trade, input.setupTemplates);
    const warnings = documentationWarnings(trade, grade, setup, score.percent, accountValue, "closed");
    warnings.push(warning("MFE_UNAVAILABLE", "Reliable intraday excursion data is unavailable; MFE was not calculated.", "info", trade));
    warnings.push(warning("MAE_UNAVAILABLE", "Reliable intraday excursion data is unavailable; MAE was not calculated.", "info", trade));
    const executionPnl = exits.length ? exits.reduce((sum, execution) => sum + execution.pnl, 0) : null;
    if (executionPnl !== null && Math.abs(executionPnl - trade.pnl) > 0.01) warnings.push(warning("PARTIAL_EXIT_CALCULATION_MISMATCH", `Execution P&L ${round(executionPnl)} does not match stored P&L ${round(trade.pnl)}.`, "critical", trade));
    const commission = executions.length ? executions.reduce((sum, execution) => sum + execution.commission, 0) : trade.commission;
    const grossPnl = executionPnl ?? (Number.isFinite(trade.pnl) ? trade.pnl : null), netPnl = grossPnl === null ? null : grossPnl - commission;
    const realizedR = trade.risk ? netPnl! / trade.risk : null;
    if (realizedR === null) warnings.push(warning("R_MULTIPLE_UNAVAILABLE", "Realized R multiple requires planned risk.", "warning", trade));
    const entryShares = executions.filter((execution) => execution.type === "ENTRY").reduce((sum, execution) => sum + execution.shares, 0) || trade.shares;
    const averageExit = exits.length ? exits.reduce((sum, execution) => sum + execution.price * execution.shares, 0) / exits.reduce((sum, execution) => sum + execution.shares, 0) : trade.exitPrice || null;
    return {
      trade_id: trade.id, ticker: trade.symbol, asset_name: null, side: trade.side,
      entry_timestamp: newYorkTimestamp(trade.entryDate, trade.openTime || "00:00:00"), exit_timestamp: newYorkTimestamp(finalExit?.date || trade.exitDate, finalExit?.time || trade.closeTime || "00:00:00"),
      time_held: timeHeld(trade, finalExit), average_entry: round(trade.avgEntry), average_exit: round(averageExit), shares: round(entryShares, 4), executions,
      partial_exits: exits.length > 1 ? exits : [], gross_pnl: round(grossPnl), commission: round(commission), net_pnl: round(netPnl),
      position_return_pct: trade.returnPercent || (trade.avgEntry && averageExit ? round((averageExit - trade.avgEntry) / trade.avgEntry * 100 * (trade.side === "SHORT" ? -1 : 1)) : null),
      planned_risk_dollars: trade.risk || null, realized_r_multiple: round(realizedR), initial_stop: null, final_stop: trade.stopPrice || null, take_profit: trade.takeProfitPrice || null,
      mfe_dollars: null, mfe_r: null, mae_dollars: null, mae_r: null, setup, grade, setup_criteria_score: score.score, setup_criteria_max: score.maximum,
      setup_criteria_results: trade.checklistItems, mistake_tags: trade.mistakeTags, system: trade.tradeQuality || null,
      review_sections: resolvedTradeReviewSections(trade.reviewSections, trade.notes), notes: trade.notes || null,
      chart_links: trade.chartLinks, screenshot_references: trade.screenshots, documentation_warnings: warnings
    };
  });

  const allPriceCurrent = openPositions.every((position) => position.current_price !== null && position.current_price_session === input.requestedSession);
  const marketValues = openPositions.map((position) => position.market_value).filter((value): value is number => typeof value === "number");
  const longValue = marketValues.filter((value) => value > 0).reduce((sum, value) => sum + value, 0), shortValue = Math.abs(marketValues.filter((value) => value < 0).reduce((sum, value) => sum + value, 0));
  const openPnlValues = openPositions.map((position) => position.unrealized_pnl).filter((value): value is number => typeof value === "number");
  const riskValues = openPositions.map((position) => position.remaining_risk_to_stop_dollars).filter((value): value is number => typeof value === "number");
  if (!allPriceCurrent && openPositions.length) topWarnings.push(warning("CURRENT_PRICE_STALE", "Portfolio aggregates requiring current prices are null because one or more positions lack a current-session price.", "critical"));
  const portfolioDataAsOf = broker.importedAt || trades.map((trade) => trade.updatedAt).filter(Boolean).sort().at(-1) || null;
  const priceDates = openPositions.map((position) => position.current_price_timestamp).filter((value): value is string => Boolean(value)).sort();
  const priceRetrievedDates = openPositions.map((position) => position.current_price_retrieved_at).filter((value): value is string => Boolean(value)).sort();
  const allWarnings = [...topWarnings, ...openPositions.flatMap((position) => position.data_warnings), ...closedTrades.flatMap((trade) => trade.documentation_warnings)];
  const aggregatesComplete = broker.complete && allPriceCurrent;
  const summary = {
    gross_exposure_dollars: aggregatesComplete ? round(longValue + shortValue) : null, gross_exposure_pct: aggregatesComplete && accountValue ? round((longValue + shortValue) / accountValue * 100) : null,
    net_exposure_dollars: aggregatesComplete ? round(longValue - shortValue) : null, net_exposure_pct: aggregatesComplete && accountValue ? round((longValue - shortValue) / accountValue * 100) : null,
    long_exposure_dollars: aggregatesComplete ? round(longValue) : null, long_exposure_pct: aggregatesComplete && accountValue ? round(longValue / accountValue * 100) : null,
    short_exposure_dollars: aggregatesComplete ? round(shortValue) : null, short_exposure_pct: aggregatesComplete && accountValue ? round(shortValue / accountValue * 100) : null,
    net_market_value_dollars: aggregatesComplete ? round(longValue - shortValue) : null, total_open_pnl: aggregatesComplete ? round(openPnlValues.reduce((sum, value) => sum + value, 0)) : null,
    total_initial_risk: broker.complete && openPositions.every((position) => position.planned_risk_dollars !== null) ? round(openPositions.reduce((sum, position) => sum + Number(position.planned_risk_dollars), 0)) : null,
    total_remaining_risk_to_stops: broker.complete && riskValues.length === openPositions.length ? round(riskValues.reduce((sum, value) => sum + value, 0)) : null,
    total_remaining_risk_pct: broker.complete && riskValues.length === openPositions.length && accountValue ? round(riskValues.reduce((sum, value) => sum + value, 0) / accountValue * 100) : null,
    open_position_count: openPositions.length, long_position_count: openPositions.filter((position) => position.side === "LONG").length, short_position_count: openPositions.filter((position) => position.side === "SHORT").length,
    positions_missing_stops: openPositions.filter((position) => position.current_stop === null).length, positions_with_upcoming_earnings: null
  };

  return {
    metadata: {
      schema_version: DAILY_PORTFOLIO_SNAPSHOT_SCHEMA_VERSION, snapshot_id: crypto.randomUUID(), generated_at: generatedAt,
      requested_session: input.requestedSession, latest_completed_market_session: input.latestCompletedMarketSession,
      portfolio_data_as_of: portfolioDataAsOf, price_data_as_of: priceDates.at(0) || null, broker_import_as_of: broker.statementDate || broker.importedAt,
      broker_import_timestamp: broker.importedAt, broker_imported_at: broker.importedAt,
      broker_position_state_as_of: broker.statementDate, statement_coverage_date: broker.statementDate,
      price_timestamp: priceDates.at(0) || null, price_as_of: priceDates.at(0) || null, price_retrieved_at: priceRetrievedDates.at(-1) || null,
      price_source: Array.from(new Set(openPositions.map((position) => position.current_price_source).filter(Boolean))).join(", ") || null,
      price_type: Array.from(new Set(openPositions.map((position) => position.current_price_type).filter(Boolean))).join(", ") || null,
      valuation_context: "Broker position state reflects the imported statement. Current valuation uses post-close prices.",
      broker_import_complete: broker.complete, account_name: input.accountName || null, account_value: accountValue,
      source_environment: input.sourceEnvironment, application_version: input.applicationVersion
    },
    snapshot_status: snapshotStatusFromWarnings(allWarnings),
    portfolio_summary: summary, open_positions: openPositions, trades_closed_during_session: closedTrades,
    warnings: allWarnings, critical_warning_count: allWarnings.filter((item) => item.severity === "critical").length
  };
}

function formatMoney(value: number | null) { return value === null ? "—" : new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 2 }).format(value); }
function formatPercent(value: number | null) { return value === null ? "—" : `${value.toFixed(2)}%`; }

export function renderDailyPortfolioSnapshotMarkdown(snapshot: ReturnType<typeof buildDailyPortfolioSnapshot>) {
  const lines = [
    `# Daily Portfolio Snapshot — ${snapshot.metadata.requested_session}`, "",
    `Status: **${snapshot.snapshot_status}**`,
    `Snapshot ID: ${snapshot.metadata.snapshot_id}`,
    `Account: ${snapshot.metadata.account_name || "—"}`,
    `Broker import complete: ${snapshot.metadata.broker_import_complete}`,
    `Broker imported at: ${snapshot.metadata.broker_imported_at || "—"}`,
    `Broker position state as of: ${snapshot.metadata.broker_position_state_as_of || "—"}`,
    `Statement coverage date: ${snapshot.metadata.statement_coverage_date || "—"}`,
    `Price as of / retrieved at: ${snapshot.metadata.price_as_of || "—"} / ${snapshot.metadata.price_retrieved_at || "—"}`,
    `Price source / type: ${snapshot.metadata.price_source || "—"} / ${snapshot.metadata.price_type || "—"}`,
    snapshot.metadata.valuation_context,
    `Critical warnings: ${snapshot.critical_warning_count}`, "", "## Portfolio summary", "",
    `- Account value: ${formatMoney(snapshot.metadata.account_value)}`,
    `- Gross exposure: ${formatMoney(snapshot.portfolio_summary.gross_exposure_dollars)} (${formatPercent(snapshot.portfolio_summary.gross_exposure_pct)})`,
    `- Net exposure: ${formatMoney(snapshot.portfolio_summary.net_exposure_dollars)} (${formatPercent(snapshot.portfolio_summary.net_exposure_pct)})`,
    `- Open P&L: ${formatMoney(snapshot.portfolio_summary.total_open_pnl)}`,
    `- Remaining risk to stops: ${formatMoney(snapshot.portfolio_summary.total_remaining_risk_to_stops)} (${formatPercent(snapshot.portfolio_summary.total_remaining_risk_pct)})`,
    `- Open positions: ${snapshot.portfolio_summary.open_position_count}`, "", "## Open positions", "",
    "| Ticker | Side | Shares | Entry | Current | Market value | P&L | R | Stop | Weight |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"
  ];
  for (const position of snapshot.open_positions) lines.push(`| ${position.ticker} | ${position.side} | ${position.shares ?? "—"} | ${formatMoney(position.average_entry)} | ${formatMoney(position.current_price)} | ${formatMoney(position.market_value)} | ${formatMoney(position.unrealized_pnl)} | ${position.open_r_multiple ?? "—"} | ${formatMoney(position.current_stop)} | ${formatPercent(position.position_weight_pct)} |`);
  if (!snapshot.open_positions.length) lines.push("| — | — | — | — | — | — | — | — | — | — |");
  lines.push("", "## Position details and warnings", "");
  for (const position of snapshot.open_positions) {
    const protectiveLevels = position.protective_levels.map((level) => `${level.effective_quantity} @ ${formatMoney(Number(level.price))} (${level.role})`).join("; ") || "—";
    const profitOrders = position.profit_taking_orders.map((order) => `${order.quantity} @ ${formatMoney(order.price)}`).join("; ") || "—";
    const review = (key: keyof typeof position.review_sections) => position.review_sections[key]?.replace(/\s*\n\s*/g, " ") || "—";
    lines.push(`### ${position.ticker}`, "", `- Setup / grade: ${position.setup || "—"} / ${position.grade || "—"}`, `- P&L: unrealized ${formatMoney(position.unrealized_pnl)}; realized to date ${formatMoney(position.realized_pnl_to_date)}; lifecycle ${formatMoney(position.total_trade_pnl)}`, `- R: open ${position.open_r_multiple === null ? "—" : `${position.open_r_multiple.toFixed(2)}R`}; lifecycle ${position.lifecycle_r_multiple === null ? "—" : `${position.lifecycle_r_multiple.toFixed(2)}R`}`, `- Price: ${formatMoney(position.current_price)} at ${position.current_price_timestamp || "—"} (${position.current_price_source || "—"}, ${position.current_price_type || "—"})`, `- Stop plan: ${position.stop_plan_type}; ${protectiveLevels}`, `- Profit-taking orders: ${profitOrders}`, `- Remaining risk: ${formatMoney(position.remaining_risk_to_stop_dollars)}`, `- Criteria: ${position.setup_criteria_score ?? "—"} / ${position.setup_criteria_max ?? "—"}`, `- Review — setup: ${review("setup")}`, `- Review — entry: ${review("entry")}`, `- Review — exit: ${review("exit")}`, `- Review — did right: ${review("didRight")}`, `- Review — did wrong: ${review("didWrong")}`, `- Review — general: ${review("general")}`, `- Legacy notes: ${position.notes || "—"}`, `- Charts / screenshots: ${position.chart_links.length} / ${position.screenshot_references.length}`, ...position.data_warnings.map((item) => `- [${item.severity}] ${item.code}: ${item.message}`), "");
  }
  lines.push("## Trades closed during the session", "", "| Ticker | Side | Entry | Exit | Net P&L | R | Setup | Grade |", "|---|---|---:|---:|---:|---:|---|---|");
  for (const trade of snapshot.trades_closed_during_session) lines.push(`| ${trade.ticker} | ${trade.side} | ${formatMoney(trade.average_entry)} | ${formatMoney(trade.average_exit)} | ${formatMoney(trade.net_pnl)} | ${trade.realized_r_multiple ?? "—"} | ${trade.setup || "—"} | ${trade.grade || "—"} |`);
  if (!snapshot.trades_closed_during_session.length) lines.push("| — | — | — | — | — | — | — | — |");
  lines.push("", "## Documentation and data-quality warnings", "");
  if (!snapshot.warnings.length) lines.push("- None.");
  else snapshot.warnings.forEach((item) => lines.push(`- [${item.severity}] ${item.code}${item.ticker ? ` (${item.ticker})` : ""}: ${item.message}`));
  return `${lines.join("\n")}\n`;
}

export function validateDailyPortfolioSnapshot(snapshot: ReturnType<typeof buildDailyPortfolioSnapshot>) {
  const errors: string[] = [];
  if (snapshot.metadata.schema_version !== DAILY_PORTFOLIO_SNAPSHOT_SCHEMA_VERSION) errors.push("schema_version");
  if (!snapshot.metadata.snapshot_id) errors.push("snapshot_id");
  if (!/^\d{4}-\d{2}-\d{2}$/.test(snapshot.metadata.requested_session)) errors.push("requested_session");
  if (!Array.isArray(snapshot.open_positions)) errors.push("open_positions");
  if (!Array.isArray(snapshot.trades_closed_during_session)) errors.push("trades_closed_during_session");
  if (!Array.isArray(snapshot.warnings) || snapshot.warnings.some((item) => !item.code || !item.message)) errors.push("warnings");
  return errors;
}
