import crypto from "node:crypto";
import { buildDailyPortfolioSnapshot, newYorkTimestamp, type SnapshotPrice } from "./daily-portfolio-snapshot";
import { resolvedTradeReviewSections } from "./trade-review";
import type { TradeChecklistItem, TradeExecution, TradeLogEntry } from "./types";
import { normalizeWeeklyFocus, type WeeklyFocus } from "./weekly-focus";

export const MONTH_TO_DATE_SNAPSHOT_SCHEMA_VERSION = "month-to-date-trading-snapshot-v1";
export const ACCOUNT_LOSS_THRESHOLD_DOLLARS = 688_000;
export type MtdSnapshotStatus = "COMPLETE" | "COMPLETE_WITH_WARNINGS" | "BLOCKED";
export type MtdDiagnostic = {
  code: string;
  severity: "info" | "warning" | "critical";
  message: string;
  blocking: boolean;
  trade_id?: string;
  symbol?: string;
  field?: string;
  suggested_remediation?: string;
};

export function mtdStatusFromDiagnostics(diagnostics: MtdDiagnostic[]): MtdSnapshotStatus {
  if (diagnostics.some((item) => item.blocking)) return "BLOCKED";
  return diagnostics.length ? "COMPLETE_WITH_WARNINGS" : "COMPLETE";
}

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

export type MonthToDateSnapshotInput = {
  month: string;
  asOfDate: string;
  asOfTimestamp?: string;
  generatedAt: string;
  portfolioId?: string;
  portfolioName: string;
  trades: TradeLogEntry[];
  portfolioMeta?: PortfolioMeta;
  prices: Map<string, SnapshotPrice>;
  weeklyFocus?: WeeklyFocus;
  sourceEnvironment: string;
  applicationVersion: string;
};

function round(value: number | null, digits = 2) {
  if (value === null || !Number.isFinite(value)) return null;
  const factor = 10 ** digits;
  return Math.round(value * factor) / factor;
}

function nyDate(value: Date) {
  return new Intl.DateTimeFormat("en-CA", {
    timeZone: "America/New_York", year: "numeric", month: "2-digit", day: "2-digit"
  }).format(value);
}

export function resolveMtdPeriod(input: { month?: string; asOfDate?: string; asOfTimestamp?: string; now?: Date }) {
  const now = input.now || new Date();
  const asOfDate = String(input.asOfDate || nyDate(now)).trim();
  const month = String(input.month || asOfDate.slice(0, 7)).trim();
  if (!/^\d{4}-\d{2}$/.test(month)) throw new Error("Month must use YYYY-MM.");
  if (!/^\d{4}-\d{2}-\d{2}$/.test(asOfDate)) throw new Error("As-of date must use YYYY-MM-DD.");
  if (!asOfDate.startsWith(`${month}-`)) throw new Error("As-of date must fall within the selected month.");
  const start = newYorkTimestamp(`${month}-01`, "00:00:00");
  const requestedEnd = input.asOfTimestamp || newYorkTimestamp(asOfDate, "23:59:59");
  if (!start || !requestedEnd) throw new Error("Could not resolve New York reporting boundaries.");
  const currentNyDate = nyDate(now);
  const end = asOfDate === currentNyDate && !input.asOfTimestamp && now.getTime() < new Date(requestedEnd).getTime()
    ? now.toISOString()
    : requestedEnd;
  if (new Date(end).getTime() < new Date(start).getTime()) throw new Error("As-of timestamp precedes the reporting month.");
  return { month, asOfDate, start, end, timezone: "America/New_York" as const };
}

function executionTimestamp(execution: TradeExecution) {
  return newYorkTimestamp(execution.date, /^\d{2}:\d{2}:\d{2}$/.test(execution.time) ? execution.time : "00:00:00");
}

function executionValue(execution: TradeExecution) {
  return Number(execution.price || 0) * Number(execution.shares || 0);
}

function isBetween(timestamp: string | null, start: string, end: string) {
  if (!timestamp) return false;
  const value = new Date(timestamp).getTime();
  return value >= new Date(start).getTime() && value <= new Date(end).getTime();
}

function beforeOrAt(timestamp: string | null, end: string) {
  return Boolean(timestamp && new Date(timestamp).getTime() <= new Date(end).getTime());
}

function scoreChecklist(items: TradeChecklistItem[]) {
  const maximum = items.reduce((sum, item) => sum + Math.max(0, Number(item.points || 0)), 0);
  const score = items.reduce((sum, item) => {
    const max = Math.max(0, Number(item.points || 0));
    return sum + ((item.inputType || "boolean") === "points" ? Math.max(0, Math.min(max, Number(item.score || 0))) : item.met ? max : 0);
  }, 0);
  return { score: round(score), maximum: round(maximum), percent: maximum ? round(score / maximum * 100) : null };
}

function tradeTimestamps(trade: TradeLogEntry) {
  const opened = newYorkTimestamp(trade.entryDate, /^\d{2}:\d{2}:\d{2}$/.test(trade.openTime) ? trade.openTime : "00:00:00");
  const closed = trade.exitDate ? newYorkTimestamp(trade.exitDate, /^\d{2}:\d{2}:\d{2}$/.test(trade.closeTime) ? trade.closeTime : "23:59:59") : null;
  return { opened, closed };
}

function quantities(executions: TradeExecution[]) {
  let current = 0;
  let maximum = 0;
  let initial = 0;
  for (const execution of executions) {
    if (execution.type === "ENTRY") {
      current += Math.max(0, Number(execution.shares || 0));
      if (!initial) initial = Math.max(0, Number(execution.shares || 0));
      maximum = Math.max(maximum, current);
    } else {
      current -= Math.max(0, Number(execution.shares || 0));
    }
  }
  return { initial: round(initial, 4) || 0, maximum: round(maximum, 4) || 0, current: round(Math.max(0, current), 4) || 0, rawCurrent: current };
}

function weightedPrice(executions: TradeExecution[]) {
  const shares = executions.reduce((sum, item) => sum + Number(item.shares || 0), 0);
  return shares ? executions.reduce((sum, item) => sum + executionValue(item), 0) / shares : null;
}

function holdingDuration(opened: string | null, closed: string | null, end: string) {
  if (!opened) return { seconds: null, display: null };
  const seconds = Math.max(0, Math.round((new Date(closed || end).getTime() - new Date(opened).getTime()) / 1000));
  const days = Math.floor(seconds / 86400);
  return { seconds, display: days ? `${days} day${days === 1 ? "" : "s"}` : `${Math.floor(seconds / 3600)} hours` };
}

function diagnostic(code: string, message: string, severity: MtdDiagnostic["severity"], trade?: TradeLogEntry, field?: string): MtdDiagnostic {
  return {
    code, severity, message, blocking: severity === "critical",
    ...(trade ? { trade_id: trade.id, symbol: trade.symbol } : {}),
    ...(field ? { field } : {})
  };
}

function screenshotRecord(reference: string, trade: TradeLogEntry, index: number) {
  const durable = reference.startsWith("/api/trades/") || /^https?:\/\//i.test(reference);
  return {
    image_id: reference.match(/screenshots\/([^/?#]+)/)?.[1] || `${trade.id}-${index + 1}`,
    trade_id: trade.id,
    symbol: trade.symbol,
    image_type: "review",
    timestamp: null,
    caption: null,
    original_filename: null,
    mime_type: null,
    size_bytes: null,
    storage_provider: reference.startsWith("/api/") ? "TRADING_DASHBOARD_DATABASE" : /^https?:\/\//i.test(reference) ? "EXTERNAL_URL" : "UNKNOWN",
    storage_reference: durable ? reference : null,
    durable,
    accessibility_status: reference.startsWith("/api/") ? "AUTHENTICATED_DASHBOARD_REQUIRED" : durable ? "REFERENCE_AVAILABLE" : "LOCAL_ONLY_OR_INVALID"
  };
}

function groupSummary<T>(trades: T[], key: (trade: T) => string | null, pnl: (trade: T) => number) {
  const groups = new Map<string, { count: number; realized_mtd_pnl: number }>();
  for (const trade of trades) {
    const label = key(trade) || "UNAVAILABLE";
    const current = groups.get(label) || { count: 0, realized_mtd_pnl: 0 };
    current.count += 1;
    current.realized_mtd_pnl += pnl(trade);
    groups.set(label, current);
  }
  return Array.from(groups.entries()).map(([label, value]) => ({ label, count: value.count, realized_mtd_pnl: round(value.realized_mtd_pnl) }));
}

function multiGroupSummary<T>(trades: T[], keys: (trade: T) => string[], pnl: (trade: T) => number) {
  const expanded = trades.flatMap((trade) => (keys(trade).length ? keys(trade) : ["UNAVAILABLE"]).map((label) => ({ trade, label })));
  return groupSummary(expanded, (item) => item.label, (item) => pnl(item.trade));
}

function median(values: number[]) {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

function streaks(results: number[]) {
  let currentType = 0;
  let currentLength = 0;
  let maximumWins = 0;
  let maximumLosses = 0;
  for (const result of results) {
    const type = result > 0 ? 1 : result < 0 ? -1 : 0;
    if (!type) {
      currentType = 0;
      currentLength = 0;
      continue;
    }
    if (type === currentType) currentLength += 1;
    else {
      currentType = type;
      currentLength = 1;
    }
    if (type === 1) maximumWins = Math.max(maximumWins, currentLength);
    else maximumLosses = Math.max(maximumLosses, currentLength);
  }
  return { maximumWins, maximumLosses, currentLosingStreak: currentType === -1 ? currentLength : 0 };
}

export function buildMonthToDateSnapshot(input: MonthToDateSnapshotInput) {
  const period = resolveMtdPeriod({ month: input.month, asOfDate: input.asOfDate, asOfTimestamp: input.asOfTimestamp, now: new Date(input.generatedAt) });
  const diagnostics: MtdDiagnostic[] = [];
  const relevantTrades = input.trades.filter((trade) => {
    const eligibleExecutions = trade.executions.filter((item) => beforeOrAt(executionTimestamp(item), period.end));
    const { opened, closed } = tradeTimestamps(trade);
    const activeDuringPeriod = Boolean(opened && new Date(opened).getTime() <= new Date(period.end).getTime() && (!closed || new Date(closed).getTime() >= new Date(period.start).getTime()));
    const updatedDuringPeriod = isBetween(trade.updatedAt || null, period.start, period.end);
    return activeDuringPeriod || updatedDuringPeriod || eligibleExecutions.some((item) => isBetween(executionTimestamp(item), period.start, period.end));
  });

  const asOfTradeStates = relevantTrades.map((trade) => {
    const executions = trade.executions.filter((item) => beforeOrAt(executionTimestamp(item), period.end)).sort((a, b) => String(executionTimestamp(a)).localeCompare(String(executionTimestamp(b))));
    const q = quantities(executions);
    return { ...trade, executions, shares: q.current, status: q.current > 0.000001 ? "OPEN" as const : trade.status };
  });
  const dailyProjection = buildDailyPortfolioSnapshot({
    requestedSession: period.asOfDate,
    latestCompletedMarketSession: period.asOfDate,
    generatedAt: input.generatedAt,
    accountName: input.portfolioName,
    portfolioMeta: input.portfolioMeta,
    trades: asOfTradeStates,
    setupTemplates: [],
    prices: input.prices,
    sourceEnvironment: input.sourceEnvironment,
    applicationVersion: input.applicationVersion,
    weeklyFocus: input.weeklyFocus
  });
  const projectedPositions = new Map(dailyProjection.open_positions.map((position) => [position.trade_id, position]));

  const imageManifest: ReturnType<typeof screenshotRecord>[] = [];
  const trades = relevantTrades.map((trade) => {
    const allExecutions = trade.executions
      .filter((item) => beforeOrAt(executionTimestamp(item), period.end))
      .sort((a, b) => String(executionTimestamp(a)).localeCompare(String(executionTimestamp(b))));
    const inPeriod = allExecutions.filter((item) => isBetween(executionTimestamp(item), period.start, period.end));
    const q = quantities(allExecutions);
    const entries = allExecutions.filter((item) => item.type === "ENTRY");
    const exits = allExecutions.filter((item) => item.type === "EXIT");
    const periodExits = inPeriod.filter((item) => item.type === "EXIT");
    const lifecycleCommission = allExecutions.reduce((sum, item) => sum + Number(item.commission || 0), 0);
    const periodCommission = inPeriod.reduce((sum, item) => sum + Number(item.commission || 0), 0);
    const realizedMtd = periodExits.reduce((sum, item) => sum + Number(item.pnl || 0), 0) - periodCommission;
    const lifecycleRealized = exits.reduce((sum, item) => sum + Number(item.pnl || 0), 0) - lifecycleCommission;
    const position = projectedPositions.get(trade.id);
    const unrealized = q.current > 0 ? position?.unrealized_pnl ?? null : 0;
    const totalPnl = unrealized === null ? null : lifecycleRealized + unrealized;
    const score = scoreChecklist(trade.checklistItems);
    const { opened, closed } = tradeTimestamps(trade);
    const duration = holdingDuration(opened, q.current > 0 ? null : closed, period.end);
    const tradeDiagnostics: MtdDiagnostic[] = [];
    const activeDuringPeriod = Boolean(opened && new Date(opened).getTime() <= new Date(period.end).getTime() && (!closed || new Date(closed).getTime() >= new Date(period.start).getTime()));
    if (!allExecutions.length) tradeDiagnostics.push(diagnostic("MISSING_EXECUTIONS", "No broker executions are stored for this included trade.", activeDuringPeriod ? "critical" : "warning", trade, "executions"));
    if (q.rawCurrent < -0.0001) tradeDiagnostics.push(diagnostic("EXECUTION_QUANTITY_MISMATCH", "Exit quantity exceeds entry quantity as of the selected timestamp.", "critical", trade, "quantities.current_quantity"));
    if (!trade.risk) tradeDiagnostics.push(diagnostic("PLANNED_RISK_UNAVAILABLE", "Planned risk is not stored; R metrics are unavailable.", q.current > 0 ? "critical" : "warning", trade, "financials.planned_risk"));
    if (q.current > 0 && !trade.stopPrice && !position?.protective_levels.length) tradeDiagnostics.push(diagnostic("CURRENT_STOP_UNAVAILABLE", "No current protective stop is stored for this open position.", "critical", trade, "prices.current_stop"));
    if (!trade.screenshots.length) tradeDiagnostics.push(diagnostic("SCREENSHOTS_UNAVAILABLE", "No screenshot references are stored.", "info", trade, "screenshots"));
    if (!trade.chartLinks.length) tradeDiagnostics.push(diagnostic("CHART_LINKS_UNAVAILABLE", "No chart links are stored.", "info", trade, "chart_links"));
    if (trade.screenshots.some((item) => !item.startsWith("/api/trades/") && !/^https?:\/\//i.test(item))) tradeDiagnostics.push(diagnostic("SCREENSHOT_REFERENCE_NOT_DURABLE", "One or more screenshot references are not durable outside the current browser context.", "warning", trade, "screenshots"));
    tradeDiagnostics.push(diagnostic("STOP_HISTORY_UNAVAILABLE", "Historical stop modifications are not persisted.", "info", trade, "stop_history"));
    tradeDiagnostics.push(diagnostic("CRITERIA_DERIVED_GRADE_UNAVAILABLE", "Saved criteria are exported, but a historical criteria-derived grade is not persisted independently.", "info", trade, "criteria_evaluation.derived_grade"));
    diagnostics.push(...tradeDiagnostics);
    const screenshots = trade.screenshots.map((reference, index) => screenshotRecord(reference, trade, index));
    imageManifest.push(...screenshots);
    const executionRows = allExecutions.map((execution) => ({
      execution_id: execution.id,
      trade_id: trade.id,
      fill_type: execution.type === "ENTRY" ? (entries[0]?.id === execution.id ? "ENTRY" : "ADD") : (exits.at(-1)?.id === execution.id && q.current === 0 ? "FINAL_EXIT" : "PARTIAL_EXIT"),
      side: execution.side,
      timestamp: executionTimestamp(execution),
      trade_date: execution.date,
      price: round(execution.price),
      quantity: round(execution.shares, 4),
      gross_value: round(executionValue(execution)),
      commission: round(execution.commission),
      fees: 0,
      realized_pnl: execution.type === "EXIT" ? round(execution.pnl) : 0,
      source: execution.source || null,
      source_key: execution.sourceKey || null,
      broker_import_id: null,
      broker_statement_reference: null,
      in_period: inPeriod.some((item) => item.id === execution.id),
      reconciliation_status: "STORED_EXECUTION"
    }));
    const review = resolvedTradeReviewSections(trade.reviewSections, trade.notes);
    return {
      trade_id: trade.id,
      portfolio_id: input.portfolioId || input.portfolioName,
      portfolio_name: input.portfolioName,
      symbol: trade.symbol,
      company_name: null,
      asset_type: "UNKNOWN",
      direction: trade.side,
      status: q.current > 0.000001 ? "OPEN" : "CLOSED",
      opened_at: opened,
      closed_at: q.current > 0.000001 ? null : closed,
      holding_duration_seconds: duration.seconds,
      holding_duration_display: duration.display,
      quantities: { initial_quantity: q.initial || trade.shares, maximum_quantity: q.maximum || trade.shares, current_quantity: q.current },
      prices: {
        average_entry: round(weightedPrice(entries) ?? trade.avgEntry),
        average_exit: round(weightedPrice(exits) ?? (trade.exitPrice || null)),
        current_price: q.current > 0 ? position?.current_price ?? null : null,
        current_price_timestamp: q.current > 0 ? position?.current_price_timestamp ?? null : null,
        price_source: q.current > 0 ? position?.current_price_source ?? null : null,
        price_type: q.current > 0 ? position?.current_price_type ?? null : null,
        initial_stop: null,
        current_stop: trade.stopPrice || null,
        take_profit: trade.takeProfitPrice || null
      },
      financials: {
        planned_risk: trade.risk || null,
        realized_mtd_pnl: round(realizedMtd),
        lifecycle_realized_pnl: round(lifecycleRealized),
        unrealized_pnl: round(unrealized),
        total_trade_pnl: round(totalPnl),
        commissions_mtd: round(periodCommission),
        lifecycle_commissions: round(lifecycleCommission),
        fees_mtd: 0,
        used_margin: trade.usedMargin || null,
        position_return_pct: trade.returnPercent || null,
        realized_mtd_r: trade.risk ? round(realizedMtd / trade.risk) : null,
        lifecycle_r: trade.risk && totalPnl !== null ? round(totalPnl / trade.risk) : null,
        open_r: trade.risk && unrealized !== null ? round(unrealized / trade.risk) : null,
        remaining_downside_risk: position?.remaining_risk_to_stop_dollars ?? null
      },
      period_activity: {
        execution_count: inPeriod.length,
        exit_execution_count: periodExits.length,
        closed_during_period: q.current <= 0.000001 && Boolean(exits.at(-1) && isBetween(executionTimestamp(exits.at(-1)!), period.start, period.end))
      },
      classification: {
        setup: trade.setupTags[0] || null,
        setup_tags: trade.setupTags,
        setup_subtype: null,
        user_assigned_grade: trade.manualGrade || null,
        mistake_tags: trade.mistakeTags,
        tags: trade.customTags,
        sector: null,
        industry_group: null,
        theme: null,
        catalyst: null,
        earnings_date: null
      },
      review: {
        setup_notes: review.setup || null,
        entry_notes: review.entry || null,
        exit_notes: review.exit || null,
        exit_plan: review.exit || null,
        what_went_right: review.didRight || null,
        what_went_wrong: review.didWrong || null,
        general_review: review.general || null,
        pre_trade_notes: null,
        during_trade_notes: null,
        post_trade_notes: null,
        lessons: null,
        review_comments: null,
        legacy_notes: trade.notes || null
      },
      criteria_evaluation: {
        score: score.score,
        max_score: score.maximum,
        score_percent: score.percent,
        derived_grade: null,
        criteria: trade.checklistItems.map((item) => ({
          criterion_id: item.id,
          category: item.groupName || null,
          label: item.criteria,
          criterion_type: item.inputType || "boolean",
          met: item.met,
          awarded_score: (item.inputType || "boolean") === "points" ? Number(item.score || 0) : item.met ? Number(item.points || 0) : 0,
          maximum_score: Number(item.points || 0),
          entered_value: item.inputType === "points" ? Number(item.score || 0) : null,
          persisted_at: null,
          source_template_or_version: null
        }))
      },
      executions: executionRows,
      stop_history: [],
      stop_plan: position ? {
        stop_plan_type: position.stop_plan_type,
        protective_levels: position.protective_levels,
        provenance: position.stop_plan_provenance,
        profit_taking_orders: position.profit_taking_orders
      } : null,
      chart_links: trade.chartLinks,
      screenshots,
      custom_fields: {
        import_source: trade.importSource || null,
        import_row_key: trade.importRowKey || null,
        emotion: trade.emotion || null,
        trade_quality: trade.tradeQuality || null,
        group_id: trade.groupId || null,
        group_role: trade.groupRole,
        created_at: trade.createdAt,
        updated_at: trade.updatedAt
      },
      data_quality: { status: tradeDiagnostics.some((item) => item.blocking) ? "INCOMPLETE" : tradeDiagnostics.length ? "COMPLETE_WITH_WARNINGS" : "COMPLETE", diagnostics: tradeDiagnostics }
    };
  });

  const currentEquity = Number(input.portfolioMeta?.currentEquity || 0) || null;
  if (currentEquity === null) diagnostics.unshift(diagnostic("AUTHORITATIVE_EQUITY_UNAVAILABLE", "Authoritative current account equity is unavailable.", "critical", undefined, "account_summary.current_equity"));
  diagnostics.push(diagnostic("STARTING_EQUITY_UNAVAILABLE", "Beginning-of-month authoritative equity is not persisted.", "warning", undefined, "account_summary.starting_equity"));
  diagnostics.push(diagnostic("DAILY_EQUITY_HISTORY_UNAVAILABLE", "Daily equity history is not persisted; daily equity and maximum drawdown are unavailable.", "warning", undefined, "daily_equity"));
  diagnostics.push(diagnostic("CASH_MOVEMENT_LEDGER_UNAVAILABLE", "Deposits, withdrawals, and account adjustments are not persisted in a dedicated ledger.", "warning", undefined, "account_summary"));
  diagnostics.push(diagnostic("WEEKLY_FOCUS_HISTORY_UNAVAILABLE", "Only the current weekly focus is stored; historical focus changes are unavailable.", "info", undefined, "weekly_focus_history"));
  diagnostics.push(diagnostic("FUNDAMENTAL_CLASSIFICATION_UNAVAILABLE", "Sector, industry group, catalyst, and earnings-date fields are not stored on Trade Detail records.", "info"));

  const closed = trades.filter((trade) => trade.period_activity.closed_during_period);
  const wins = closed.filter((trade) => Number(trade.financials.realized_mtd_pnl) > 0);
  const losses = closed.filter((trade) => Number(trade.financials.realized_mtd_pnl) < 0);
  const breakeven = closed.filter((trade) => Number(trade.financials.realized_mtd_pnl) === 0);
  const grossProfit = wins.reduce((sum, trade) => sum + Number(trade.financials.realized_mtd_pnl), 0);
  const grossLoss = losses.reduce((sum, trade) => sum + Number(trade.financials.realized_mtd_pnl), 0);
  const realizedMtd = trades.reduce((sum, trade) => sum + Number(trade.financials.realized_mtd_pnl), 0);
  const realizedMtdR = trades.reduce((sum, trade) => sum + Number(trade.financials.realized_mtd_r || 0), 0);
  const unrealized = trades.reduce((sum, trade) => sum + Number(trade.financials.unrealized_pnl || 0), 0);
  const openTrades = trades.filter((trade) => trade.status === "OPEN");
  const openRiskValues = openTrades.map((trade) => trade.financials.remaining_downside_risk).filter((value): value is number => typeof value === "number");
  const currentRisk = openRiskValues.length === openTrades.length ? openRiskValues.reduce((sum, value) => sum + value, 0) : null;
  const currentGross = openTrades.reduce((sum, trade) => sum + Math.abs(Number(trade.prices.current_price || 0) * Number(trade.quantities.current_quantity)), 0);
  const currentNet = openTrades.reduce((sum, trade) => sum + Number(trade.prices.current_price || 0) * Number(trade.quantities.current_quantity) * (trade.direction === "SHORT" ? -1 : 1), 0);
  const cushion = currentEquity === null ? null : currentEquity - ACCOUNT_LOSS_THRESHOLD_DOLLARS;
  const completedNotes = trades.filter((trade) => Object.values(trade.review).some(Boolean)).length;
  const withScreenshots = trades.filter((trade) => trade.screenshots.length > 0).length;
  const closedChronologically = [...closed].sort((a, b) => String(a.closed_at).localeCompare(String(b.closed_at)));
  const streak = streaks(closedChronologically.map((trade) => Number(trade.financials.realized_mtd_pnl)));
  const holdingDurations = closed.map((trade) => Number(trade.holding_duration_seconds || 0)).filter((value) => value > 0);
  const riskOverruns = closed.map((trade) => {
    const loss = Math.abs(Math.min(0, Number(trade.financials.realized_mtd_pnl)));
    return trade.financials.planned_risk === null ? null : Math.max(0, loss - Number(trade.financials.planned_risk));
  }).filter((value): value is number => value !== null);
  const lossOverrunCount = closed.filter((trade) => trade.financials.planned_risk !== null && Math.abs(Math.min(0, Number(trade.financials.realized_mtd_pnl))) > Number(trade.financials.planned_risk)).length;
  const dailyMap = new Map<string, { realized: number; r: number; opened: Set<string>; closed: Set<string>; fees: number }>();
  for (const trade of trades) {
    for (const execution of trade.executions.filter((item) => item.in_period)) {
      const row = dailyMap.get(execution.trade_date) || { realized: 0, r: 0, opened: new Set<string>(), closed: new Set<string>(), fees: 0 };
      const net = Number(execution.realized_pnl || 0) - Number(execution.commission || 0) - Number(execution.fees || 0);
      row.realized += net;
      row.fees += Number(execution.commission || 0) + Number(execution.fees || 0);
      if (trade.financials.planned_risk) row.r += net / Number(trade.financials.planned_risk);
      if (execution.fill_type === "ENTRY") row.opened.add(trade.trade_id);
      if (execution.fill_type === "FINAL_EXIT") row.closed.add(trade.trade_id);
      dailyMap.set(execution.trade_date, row);
    }
  }
  let cumulativePnl = 0;
  let cumulativeR = 0;
  const dailyPerformance = Array.from(dailyMap.entries()).sort(([a], [b]) => a.localeCompare(b)).map(([date, row]) => {
    cumulativePnl += row.realized;
    cumulativeR += row.r;
    return {
      date,
      beginning_equity: null,
      ending_equity: null,
      daily_equity_change: null,
      realized_pnl: round(row.realized),
      unrealized_pnl_change: null,
      commissions_and_fees: round(row.fees),
      deposits: null,
      withdrawals: null,
      trades_opened: row.opened.size,
      trades_closed: row.closed.size,
      gross_exposure: null,
      net_exposure: null,
      planned_open_risk: null,
      drawdown_cushion: null,
      daily_r: round(row.r),
      cumulative_mtd_realized_pnl: round(cumulativePnl),
      cumulative_mtd_r: round(cumulativeR),
      data_quality_status: "PARTIAL_EXECUTION_DATA_ONLY"
    };
  });
  const status = mtdStatusFromDiagnostics(diagnostics);

  return {
    snapshot_type: "MONTH_TO_DATE" as const,
    schema_version: MONTH_TO_DATE_SNAPSHOT_SCHEMA_VERSION,
    snapshot_id: crypto.randomUUID(),
    status,
    generated_at: input.generatedAt,
    timezone: period.timezone,
    period,
    portfolio: { portfolio_id: input.portfolioId || input.portfolioName, portfolio_name: input.portfolioName },
    account_summary: {
      starting_equity: null,
      current_equity: round(currentEquity),
      equity_source: input.portfolioMeta?.equitySource || null,
      equity_as_of: input.portfolioMeta?.equityUpdatedAt || input.portfolioMeta?.equityStatementDate || null,
      statement_coverage_date: input.portfolioMeta?.equityStatementDate || null,
      account_loss_threshold_dollars: ACCOUNT_LOSS_THRESHOLD_DOLLARS,
      remaining_drawdown_cushion: round(cushion),
      realized_mtd_pnl: round(realizedMtd),
      current_unrealized_pnl: round(unrealized),
      commissions_and_fees_mtd: round(trades.reduce((sum, trade) => sum + Number(trade.financials.commissions_mtd), 0)),
      deposits_mtd: null,
      withdrawals_mtd: null,
      reconciliation: { status: "UNAVAILABLE", unexplained_change: null }
    },
    data_timestamps: {
      broker_imported_at: input.portfolioMeta?.equityUpdatedAt || null,
      broker_statement_created_at: null,
      broker_position_state_as_of: input.portfolioMeta?.equityStatementDate || null,
      statement_coverage_date: input.portfolioMeta?.equityStatementDate || null,
      price_as_of: Array.from(input.prices.values()).map((price) => price.timestamp).filter(Boolean).sort().at(0) || null,
      price_retrieved_at: Array.from(input.prices.values()).map((price) => price.retrievedAt).filter(Boolean).sort().at(-1) || null,
      price_sources: Array.from(new Set(Array.from(input.prices.values()).map((price) => price.provider).filter(Boolean))),
      price_types: Array.from(new Set(Array.from(input.prices.values()).map((price) => price.priceType).filter(Boolean))),
      valuation_context: "Broker position state reflects the imported statement. Current valuation uses completed-session prices."
    },
    risk_summary: {
      current_gross_exposure: round(currentGross),
      current_net_exposure: round(currentNet),
      maximum_simultaneous_gross_exposure: null,
      current_planned_downside_risk: round(currentRisk),
      largest_individual_position_risk: round(openRiskValues.length ? Math.max(...openRiskValues) : null),
      planned_downside_risk_pct_equity: currentRisk !== null && currentEquity ? round(currentRisk / currentEquity * 100) : null,
      planned_downside_risk_pct_cushion: currentRisk !== null && cushion && cushion > 0 ? round(currentRisk / cushion * 100) : null,
      realized_loss_pct_cushion: cushion && cushion > 0 ? round(Math.abs(Math.min(0, realizedMtd)) / cushion * 100) : null,
      maximum_mtd_drawdown: null
    },
    performance_summary: {
      total_included_trades: trades.length,
      trades_opened_during_period: trades.filter((trade) => isBetween(trade.opened_at, period.start, period.end)).length,
      trades_closed_during_period: closed.length,
      trades_still_open: openTrades.length,
      winning_closed_trades: wins.length,
      losing_closed_trades: losses.length,
      breakeven_trades: breakeven.length,
      win_rate_pct: closed.length ? round(wins.length / closed.length * 100) : null,
      gross_profit: round(grossProfit),
      gross_loss: round(grossLoss),
      profit_factor: grossLoss < 0 ? round(grossProfit / Math.abs(grossLoss)) : null,
      realized_mtd_pnl: round(realizedMtd),
      realized_mtd_r: round(realizedMtdR),
      unrealized_pnl: round(unrealized),
      average_winner: wins.length ? round(grossProfit / wins.length) : null,
      average_loser: losses.length ? round(grossLoss / losses.length) : null,
      average_winner_r: wins.length ? round(wins.reduce((sum, trade) => sum + Number(trade.financials.realized_mtd_r || 0), 0) / wins.length) : null,
      average_loser_r: losses.length ? round(losses.reduce((sum, trade) => sum + Number(trade.financials.realized_mtd_r || 0), 0) / losses.length) : null,
      payoff_ratio: wins.length && losses.length ? round((grossProfit / wins.length) / Math.abs(grossLoss / losses.length)) : null,
      expectancy_per_trade: closed.length ? round(realizedMtd / closed.length) : null,
      expectancy_r: closed.length ? round(realizedMtdR / closed.length) : null,
      largest_winner: wins.length ? round(Math.max(...wins.map((trade) => Number(trade.financials.realized_mtd_pnl)))) : null,
      largest_loser: losses.length ? round(Math.min(...losses.map((trade) => Number(trade.financials.realized_mtd_pnl)))) : null,
      largest_risk_overrun: riskOverruns.length ? round(Math.max(...riskOverruns)) : null,
      average_holding_duration_seconds: closed.length ? round(closed.reduce((sum, trade) => sum + Number(trade.holding_duration_seconds || 0), 0) / closed.length, 0) : null,
      median_holding_duration_seconds: round(median(holdingDurations), 0),
      maximum_consecutive_wins: streak.maximumWins,
      maximum_consecutive_losses: streak.maximumLosses,
      current_losing_streak: streak.currentLosingStreak,
      maximum_mtd_drawdown: null,
      results_by_setup: groupSummary(trades, (trade) => trade.classification.setup, (trade) => Number(trade.financials.realized_mtd_pnl)),
      results_by_user_assigned_grade: groupSummary(trades, (trade) => trade.classification.user_assigned_grade, (trade) => Number(trade.financials.realized_mtd_pnl)),
      results_by_criteria_derived_grade: groupSummary(trades, (trade) => trade.criteria_evaluation.derived_grade, (trade) => Number(trade.financials.realized_mtd_pnl)),
      results_by_sector: groupSummary(trades, (trade) => trade.classification.sector, (trade) => Number(trade.financials.realized_mtd_pnl)),
      results_by_industry_group: groupSummary(trades, (trade) => trade.classification.industry_group, (trade) => Number(trade.financials.realized_mtd_pnl)),
      results_by_theme: groupSummary(trades, (trade) => trade.classification.theme, (trade) => Number(trade.financials.realized_mtd_pnl)),
      results_by_mistake_tag: multiGroupSummary(trades, (trade) => trade.classification.mistake_tags, (trade) => Number(trade.financials.realized_mtd_pnl)),
      results_by_side: groupSummary(trades, (trade) => trade.direction, (trade) => Number(trade.financials.realized_mtd_pnl)),
      notes_completion_pct: trades.length ? round(completedNotes / trades.length * 100) : null,
      screenshot_coverage_pct: trades.length ? round(withScreenshots / trades.length * 100) : null,
      actual_loss_exceeded_planned_risk_pct: closed.length ? round(lossOverrunCount / closed.length * 100) : null,
      stop_missing_pct: trades.length ? round(trades.filter((trade) => !trade.prices.current_stop).length / trades.length * 100) : null,
      trades_with_adds_pct: trades.length ? round(trades.filter((trade) => trade.executions.filter((item) => item.fill_type === "ADD").length).length / trades.length * 100) : null,
      trades_with_partial_exits_pct: trades.length ? round(trades.filter((trade) => trade.executions.some((item) => item.fill_type === "PARTIAL_EXIT")).length / trades.length * 100) : null
    },
    weekly_focus: normalizeWeeklyFocus(input.weeklyFocus),
    weekly_focus_history: [],
    daily_equity: [],
    daily_performance: dailyPerformance,
    open_positions: openTrades,
    trades,
    image_manifest: imageManifest,
    diagnostics
  };
}

export function validateMonthToDateSnapshot(snapshot: ReturnType<typeof buildMonthToDateSnapshot>) {
  const errors: string[] = [];
  if (snapshot.snapshot_type !== "MONTH_TO_DATE") errors.push("snapshot_type must be MONTH_TO_DATE");
  if (!snapshot.portfolio.portfolio_name) errors.push("portfolio name is required");
  if (!/^\d{4}-\d{2}$/.test(snapshot.period.month)) errors.push("period month is invalid");
  if (snapshot.account_summary.current_equity === null) errors.push("authoritative current equity is required");
  for (const trade of snapshot.trades) {
    if (!trade.trade_id || !trade.symbol) errors.push("each trade requires an ID and symbol");
    if (trade.quantities.current_quantity < 0) errors.push(`${trade.symbol} has a negative current quantity`);
  }
  return errors;
}

function money(value: number | null) {
  return value === null ? "—" : new Intl.NumberFormat("en-US", { style: "currency", currency: "USD" }).format(value);
}

function metric(value: number | null, suffix = "") {
  return value === null ? "—" : `${value}${suffix}`;
}

export function renderMonthToDateSnapshotMarkdown(snapshot: ReturnType<typeof buildMonthToDateSnapshot>) {
  const p = snapshot.performance_summary;
  const lines = [
    "# Trading Dashboard Month-to-Date Snapshot", "",
    "## Reporting Period", "",
    `- Portfolio: ${snapshot.portfolio.portfolio_name}`,
    `- Month: ${snapshot.period.month}`,
    `- Period start: ${snapshot.period.start}`,
    `- As-of: ${snapshot.period.end}`,
    `- Timezone: ${snapshot.timezone}`,
    `- Snapshot status: ${snapshot.status}`, "",
    "## Account Summary", "",
    `- Starting equity: ${money(snapshot.account_summary.starting_equity)}`,
    `- Current equity: ${money(snapshot.account_summary.current_equity)}`,
    `- Account-loss threshold: ${money(snapshot.account_summary.account_loss_threshold_dollars)}`,
    `- Remaining drawdown cushion: ${money(snapshot.account_summary.remaining_drawdown_cushion)}`,
    `- Realized MTD P&L: ${money(snapshot.account_summary.realized_mtd_pnl)}`,
    `- Current unrealized P&L: ${money(snapshot.account_summary.current_unrealized_pnl)}`,
    `- Commissions and fees: ${money(snapshot.account_summary.commissions_and_fees_mtd)}`,
    `- Deposits / withdrawals: ${money(snapshot.account_summary.deposits_mtd)} / ${money(snapshot.account_summary.withdrawals_mtd)}`, "",
    snapshot.data_timestamps.valuation_context,
    `- Broker imported at: ${snapshot.data_timestamps.broker_imported_at || "—"}`,
    `- Statement coverage: ${snapshot.data_timestamps.statement_coverage_date || "—"}`,
    `- Price as of / retrieved at: ${snapshot.data_timestamps.price_as_of || "—"} / ${snapshot.data_timestamps.price_retrieved_at || "—"}`,
    `- Price sources / types: ${snapshot.data_timestamps.price_sources.join(", ") || "—"} / ${snapshot.data_timestamps.price_types.join(", ") || "—"}`, "",
    "## Risk Summary", "",
    `- Current gross exposure: ${money(snapshot.risk_summary.current_gross_exposure)}`,
    `- Current net exposure: ${money(snapshot.risk_summary.current_net_exposure)}`,
    `- Current planned downside risk: ${money(snapshot.risk_summary.current_planned_downside_risk)}`,
    `- Planned risk / equity: ${metric(snapshot.risk_summary.planned_downside_risk_pct_equity, "%")}`,
    `- Planned risk / drawdown cushion: ${metric(snapshot.risk_summary.planned_downside_risk_pct_cushion, "%")}`,
    `- Largest position risk: ${money(snapshot.risk_summary.largest_individual_position_risk)}`, "",
    "## Performance Summary", "",
    `- Closed trades with MTD realization: ${p.trades_closed_during_period}`,
    `- Open trades: ${p.trades_still_open}`,
    `- Win rate: ${metric(p.win_rate_pct, "%")}`,
    `- Gross profit / loss: ${money(p.gross_profit)} / ${money(p.gross_loss)}`,
    `- Profit factor: ${metric(p.profit_factor)}`,
    `- Expectancy: ${money(p.expectancy_per_trade)} (${metric(p.expectancy_r, "R")})`,
    `- Realized R: ${metric(p.realized_mtd_r, "R")}`,
    `- Average winner / loser: ${money(p.average_winner)} / ${money(p.average_loser)}`,
    `- Largest winner / loss: ${money(p.largest_winner)} / ${money(p.largest_loser)}`, ""
  ];
  const table = (title: string, rows: Array<{ label: string; count: number; realized_mtd_pnl: number | null }>) => {
    lines.push(`## ${title}`, "", "| Group | Trades | Realized MTD P&L |", "|---|---:|---:|");
    rows.forEach((row) => lines.push(`| ${row.label} | ${row.count} | ${money(row.realized_mtd_pnl)} |`));
    if (!rows.length) lines.push("| — | 0 | — |");
    lines.push("");
  };
  table("Results by Setup", p.results_by_setup);
  table("Results by User-Assigned Grade", p.results_by_user_assigned_grade);
  table("Results by Criteria-Derived Grade", p.results_by_criteria_derived_grade);
  table("Results by Sector and Theme", [...p.results_by_sector, ...p.results_by_theme]);
  table("Mistake Tag Summary", p.results_by_mistake_tag);
  lines.push("## Daily Results", "", "| Date | Realized P&L | Fees | Opened | Closed | Daily R | Cumulative MTD P&L |", "|---|---:|---:|---:|---:|---:|---:|");
  snapshot.daily_performance.forEach((day) => lines.push(`| ${day.date} | ${money(day.realized_pnl)} | ${money(day.commissions_and_fees)} | ${day.trades_opened} | ${day.trades_closed} | ${metric(day.daily_r, "R")} | ${money(day.cumulative_mtd_realized_pnl)} |`));
  if (!snapshot.daily_performance.length) lines.push("| — | — | — | — | — | — | — |");
  lines.push("");
  lines.push("## Open Positions", "", "| Symbol | Side | Quantity | Price | Unrealized P&L | Downside risk |", "|---|---|---:|---:|---:|---:|");
  snapshot.open_positions.forEach((trade) => lines.push(`| ${trade.symbol} | ${trade.direction} | ${trade.quantities.current_quantity} | ${money(trade.prices.current_price)} | ${money(trade.financials.unrealized_pnl)} | ${money(trade.financials.remaining_downside_risk)} |`));
  if (!snapshot.open_positions.length) lines.push("| — | — | — | — | — | — |");
  lines.push("", "## Closed Trade Index", "", "| Symbol | Status | Entry | Exit | MTD P&L | Lifecycle P&L | Manual grade | Criteria grade | Setup | Risk | MTD R |", "|---|---|---|---|---:|---:|---|---|---|---:|---:|");
  snapshot.trades.filter((trade) => trade.financials.realized_mtd_pnl !== 0).forEach((trade) => lines.push(`| ${trade.symbol} | ${trade.status} | ${trade.opened_at || "—"} | ${trade.closed_at || "—"} | ${money(trade.financials.realized_mtd_pnl)} | ${money(trade.financials.total_trade_pnl)} | ${trade.classification.user_assigned_grade || "—"} | ${trade.criteria_evaluation.derived_grade || "—"} | ${trade.classification.setup || "—"} | ${money(trade.financials.planned_risk)} | ${metric(trade.financials.realized_mtd_r, "R")} |`));
  lines.push("", "## Weekly Process Focus", "");
  if (snapshot.weekly_focus.status === "AVAILABLE") {
    lines.push(`- Week start: ${snapshot.weekly_focus.week_start || "—"}`, `- Summary: ${snapshot.weekly_focus.summary || "—"}`, ...snapshot.weekly_focus.focus_items.map((item) => `- ${item}`));
  } else lines.push(`- Status: ${snapshot.weekly_focus.status}`);
  lines.push("", "## Data-Quality Warnings", "");
  snapshot.diagnostics.forEach((item) => lines.push(`- [${item.severity}] ${item.code}${item.symbol ? ` (${item.symbol})` : ""}: ${item.message}`));
  if (!snapshot.diagnostics.length) lines.push("- None.");
  return `${lines.join("\n")}\n`;
}
