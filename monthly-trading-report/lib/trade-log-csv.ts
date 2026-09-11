import { resolvedTradeReviewSections } from "./trade-review";
import type { TradeExcursionResult } from "./trade-excursion";
import type { TradeLogEntry } from "./types";

export type TradeLogCsvItem = {
  trade: TradeLogEntry;
  periodTrade: TradeLogEntry;
  grade: string;
  reviewStatus: "Complete" | "Needs Review";
  excursion?: TradeExcursionResult | null;
};

export type TradeLogCsvContext = {
  startDate: string;
  endDate: string;
  portfolio: string;
  baseUrl: string;
};

export const tradeLogCsvHeaders = [
  "row_type", "trade_id", "execution_id", "portfolio", "import_source", "import_row_key",
  "lifecycle_status", "period_status", "side", "symbol", "setup_tags", "entry_date", "entry_time",
  "avg_entry", "trade_shares", "exit_date", "exit_time", "avg_exit", "stop_price", "take_profit_price",
  "initial_risk", "lifecycle_pnl", "stored_lifecycle_r", "calculated_lifecycle_r", "period_start", "period_end",
  "period_last_exit_date", "period_pnl", "period_r",
  "commission", "used_margin", "return_percent", "days_in_trade", "grade", "review_status", "mistake_tags",
  "custom_tags", "review_setup", "review_entry", "review_exit", "review_did_right", "review_did_wrong",
  "review_general", "legacy_notes", "screenshot_links", "chart_links", "excursion_status", "excursion_reason", "mfe_dollars",
  "mfe_r", "mae_dollars", "mae_r", "mfe_timestamp", "mae_timestamp", "maximum_profit_captured_percent",
  "execution_type", "execution_date", "execution_time", "execution_side", "execution_shares", "execution_price",
  "execution_pnl", "execution_commission", "execution_source", "execution_source_key"
] as const;

function referenceUrl(value: string, baseUrl: string) {
  const reference = String(value || "").trim();
  if (!reference) return "";
  try {
    return new URL(reference, baseUrl).toString();
  } catch {
    return reference;
  }
}

function csvCell(value: unknown) {
  const raw = Array.isArray(value) ? value.join("; ") : String(value ?? "");
  // Prevent spreadsheet programs from interpreting user-authored review text as a formula.
  const safe = typeof value === "string" && /^[\t\r\n ]*[=+\-@]/.test(raw) ? `'${raw}` : raw;
  return `"${safe.replace(/"/g, '""')}"`;
}

function blankRow() {
  return Object.fromEntries(tradeLogCsvHeaders.map((header) => [header, ""])) as Record<(typeof tradeLogCsvHeaders)[number], unknown>;
}

function capturedPercent(trade: TradeLogEntry, excursion?: TradeExcursionResult | null) {
  if (trade.status === "OPEN" || trade.pnl <= 0 || !excursion?.mfeDollars || excursion.mfeDollars <= 0) return "";
  return Math.round((trade.pnl / excursion.mfeDollars) * 10_000) / 100;
}

function calculatedLifecycleR(trade: TradeLogEntry) {
  if (!Number.isFinite(trade.pnl) || !Number.isFinite(trade.risk) || trade.risk <= 0) return "";
  return trade.pnl / trade.risk;
}

export function buildTradeLogCsv(items: TradeLogCsvItem[], context: TradeLogCsvContext) {
  const rows: Array<Record<(typeof tradeLogCsvHeaders)[number], unknown>> = [];

  for (const item of items) {
    const { trade, periodTrade, excursion } = item;
    const review = resolvedTradeReviewSections(trade.reviewSections, trade.notes);
    const tradeRow = blankRow();
    Object.assign(tradeRow, {
      row_type: "TRADE",
      trade_id: trade.id,
      portfolio: trade.portfolioTag,
      import_source: trade.importSource,
      import_row_key: trade.importRowKey,
      lifecycle_status: trade.status,
      period_status: periodTrade.status,
      side: trade.side,
      symbol: trade.symbol,
      setup_tags: trade.setupTags,
      entry_date: trade.entryDate,
      entry_time: trade.openTime,
      avg_entry: trade.avgEntry,
      trade_shares: trade.shares,
      exit_date: trade.exitDate,
      exit_time: trade.closeTime,
      avg_exit: trade.exitPrice,
      stop_price: trade.stopPrice,
      take_profit_price: trade.takeProfitPrice,
      initial_risk: trade.risk,
      lifecycle_pnl: trade.pnl,
      stored_lifecycle_r: trade.rMultiple,
      calculated_lifecycle_r: calculatedLifecycleR(trade),
      period_start: context.startDate,
      period_end: context.endDate,
      period_last_exit_date: periodTrade.exitDate,
      period_pnl: periodTrade.pnl,
      period_r: periodTrade.rMultiple,
      commission: trade.commission,
      used_margin: trade.usedMargin,
      return_percent: trade.returnPercent,
      days_in_trade: trade.daysInTrade,
      grade: item.grade,
      review_status: item.reviewStatus,
      mistake_tags: trade.mistakeTags,
      custom_tags: trade.customTags,
      review_setup: review.setup,
      review_entry: review.entry,
      review_exit: review.exit,
      review_did_right: review.didRight,
      review_did_wrong: review.didWrong,
      review_general: review.general,
      legacy_notes: trade.notes,
      screenshot_links: trade.screenshots.map((value) => referenceUrl(value, context.baseUrl)),
      chart_links: trade.chartLinks.map((value) => referenceUrl(value, context.baseUrl)),
      excursion_status: excursion?.status || "NOT_REQUESTED",
      excursion_reason: excursion?.reason || "",
      mfe_dollars: excursion?.mfeDollars ?? "",
      mfe_r: excursion?.mfeR ?? "",
      mae_dollars: excursion?.maeDollars ?? "",
      mae_r: excursion?.maeR ?? "",
      mfe_timestamp: excursion?.mfeTimestamp ?? "",
      mae_timestamp: excursion?.maeTimestamp ?? "",
      maximum_profit_captured_percent: capturedPercent(trade, excursion)
    });
    rows.push(tradeRow);

    for (const execution of trade.executions || []) {
      const executionRow = blankRow();
      Object.assign(executionRow, {
        row_type: "EXECUTION",
        trade_id: trade.id,
        execution_id: execution.id,
        portfolio: trade.portfolioTag,
        import_source: trade.importSource,
        lifecycle_status: trade.status,
        side: trade.side,
        symbol: trade.symbol,
        execution_type: execution.type,
        execution_date: execution.date,
        execution_time: execution.time,
        execution_side: execution.side,
        execution_shares: execution.shares,
        execution_price: execution.price,
        execution_pnl: execution.pnl,
        execution_commission: execution.commission,
        execution_source: execution.source,
        execution_source_key: execution.sourceKey
      });
      rows.push(executionRow);
    }
  }

  const lines = [tradeLogCsvHeaders, ...rows.map((row) => tradeLogCsvHeaders.map((header) => row[header]))];
  return `\uFEFF${lines.map((row) => row.map(csvCell).join(",")).join("\r\n")}`;
}

export function tradeLogCsvFilename(context: Pick<TradeLogCsvContext, "portfolio" | "startDate" | "endDate">) {
  const portfolio = context.portfolio.trim().replace(/[^a-z0-9_-]+/gi, "-") || "all-portfolios";
  return `branden-trade-log-${portfolio}-${context.startDate}-to-${context.endDate}.csv`;
}
