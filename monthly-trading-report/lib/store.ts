import fs from "fs/promises";
import crypto from "crypto";
import path from "path";
import { Pool, type PoolClient } from "pg";
import {
  normalizeBrokerPortfolioSnapshot,
  upsertBrokerPortfolioSnapshotCollection,
  type BrokerPortfolioSnapshot,
  type BrokerPortfolioSnapshotInput
} from "./broker-portfolio-snapshot";
import type { CfStatementReplacementTrade } from "./cf-import-reconciliation";
import { runAtomicCfImport, type CfWorkingOrderMetadata } from "./cf-import-idempotency";
import { normalizeTradeReviewSections } from "./trade-review";
import { createUserDefinedWeeklyFocus, normalizeWeeklyFocus, type WeeklyFocus } from "./weekly-focus";
import type {
  ChecklistGradeBand,
  ChecklistInputType,
  FeedbackMessageAuthor,
  FeedbackTicket,
  FeedbackTicketMessage,
  MarketCycleEntry,
  MarketCycleEntryInput,
  MonthlyReport,
  MonthlyReportInput,
  SetupChecklistGroup,
  SetupChecklistTemplate,
  SetupTemplateCriterion,
  TradeChecklistItem,
  TradeExecution,
  TradeLogEntry,
  TradeLogInput,
  TradeSide,
  TradeStatus,
  WatchlistItem,
  WeeklyWatchlist
} from "./types";

let pool: Pool | null = null;

export function getPool() {
  if (!process.env.DATABASE_URL) {
    return null;
  }

  if (!pool) {
    pool = new Pool({
      connectionString: process.env.DATABASE_URL,
      ssl: process.env.DATABASE_URL.includes("localhost") ? false : { rejectUnauthorized: false }
    });
  }

  return pool;
}

const localDataFile = path.join(process.cwd(), "data", "monthly-reports.json");
const localTradesFile = path.join(process.cwd(), "data", "trade-logs.json");
const localSettingsFile = path.join(process.cwd(), "data", "app-settings.json");
const defaultChecklistGradeBands: ChecklistGradeBand[] = [
  { id: "grade-a-plus", label: "A+", minScore: 10, maxScore: null },
  { id: "grade-a", label: "A", minScore: 8, maxScore: 9 },
  { id: "grade-b-plus", label: "B+", minScore: 7, maxScore: 7 },
  { id: "grade-b", label: "B", minScore: 6, maxScore: 6 },
  { id: "grade-c", label: "C", minScore: 0, maxScore: 5 }
];

const columns = [
  "id",
  "user_id",
  "month",
  "account_size",
  "total_return",
  "percent_return",
  "net_pnl",
  "total_payouts",
  "total_trades",
  "win_rate",
  "avg_r",
  "total_r",
  "avg_win_r",
  "avg_loss_r",
  "avg_win",
  "avg_loss",
  "avg_risk",
  "current_risk_percent",
  "expected_value_r",
  "sharpe_ratio",
  "avg_trade_length",
  "avg_swing_length",
  "longest_win_streak",
  "longest_loss_streak",
  "notes",
  "created_at",
  "updated_at"
].join(", ");

const tradeColumns = [
  "id",
  "user_id",
  "import_source",
  "import_row_key",
  "symbol",
  "side",
  "status",
  "entry_date",
  "exit_date",
  "open_time",
  "close_time",
  "avg_entry",
  "exit_price",
  "stop_price",
  "take_profit_price",
  "shares",
  "commission",
  "used_margin",
  "risk",
  "pnl",
  "r_multiple",
  "return_percent",
  "days_in_trade",
  "setup_tags",
  "mistake_tags",
  "custom_tags",
  "manual_grade",
  "portfolio_tag",
  "emotion",
  "trade_quality",
  "checklist_items",
  "notes",
  "review_sections",
  "screenshots",
  "chart_links",
  "executions",
  "hidden",
  "group_id",
  "group_role",
  "created_at",
  "updated_at"
].join(", ");

function rowToReport(row: Record<string, string | number | null>): MonthlyReport {
  return {
    id: String(row.id),
    userId: String(row.user_id),
    month: String(row.month),
    accountSize: Number(row.account_size),
    totalReturn: Number(row.total_return),
    percentReturn: Number(row.percent_return),
    netPnl: Number(row.net_pnl),
    totalPayouts: Number(row.total_payouts),
    totalTrades: Number(row.total_trades),
    winRate: Number(row.win_rate),
    avgR: Number(row.avg_r),
    totalR: Number(row.total_r),
    avgWinR: Number(row.avg_win_r),
    avgLossR: Number(row.avg_loss_r),
    avgWin: Number(row.avg_win),
    avgLoss: Number(row.avg_loss),
    avgRisk: Number(row.avg_risk),
    currentRiskPercent: Number(row.current_risk_percent),
    expectedValueR: Number(row.expected_value_r),
    sharpeRatio: Number(row.sharpe_ratio),
    avgTradeLength: Number(row.avg_trade_length),
    avgSwingLength: Number(row.avg_swing_length),
    longestWinStreak: Number(row.longest_win_streak),
    longestLossStreak: Number(row.longest_loss_streak),
    notes: String(row.notes || ""),
    createdAt: new Date(String(row.created_at)).toISOString(),
    updatedAt: new Date(String(row.updated_at)).toISOString()
  };
}

function stringArray(value: unknown): string[] {
  if (Array.isArray(value)) {
    return value.map(String).filter(Boolean);
  }

  if (typeof value === "string") {
    try {
      const parsed = JSON.parse(value);
      return Array.isArray(parsed) ? parsed.map(String).filter(Boolean) : [];
    } catch {
      return [];
    }
  }

  return [];
}

export type BrandenPortfolioSettings = {
  portfolios: string[];
  defaultPortfolio: string;
  portfolioMeta?: Record<
    string,
    {
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
    }
  >;
};

function normalizePortfolioNames(names: unknown) {
  return Array.from(new Set(stringArray(names).map((name) => name.trim()).filter(Boolean))).sort((a, b) =>
    a.localeCompare(b)
  );
}

function normalizeBrandenPortfolioSettings(value: unknown): BrandenPortfolioSettings {
  const source =
    value && typeof value === "object" && !Array.isArray(value)
      ? (value as { portfolios?: unknown; defaultPortfolio?: unknown; portfolioMeta?: unknown })
      : { portfolios: value };
  const portfolios = normalizePortfolioNames(source.portfolios);
  const defaultPortfolio = String(source.defaultPortfolio || "").trim();
  const rawMeta =
    source.portfolioMeta && typeof source.portfolioMeta === "object" && !Array.isArray(source.portfolioMeta)
      ? (source.portfolioMeta as Record<string, unknown>)
      : {};
  const portfolioMeta: NonNullable<BrandenPortfolioSettings["portfolioMeta"]> = {};

  Object.entries(rawMeta).forEach(([portfolio, meta]) => {
    if (!meta || typeof meta !== "object" || Array.isArray(meta)) {
      return;
    }

    const raw = meta as Record<string, unknown>;
    const currentEquity = Number(raw.currentEquity);
    const statementEquity = Number(raw.statementEquity);
    const floatingPnl = Number(raw.floatingPnl);
    const workingOrders = Array.isArray(raw.workingOrders)
      ? raw.workingOrders.flatMap((value) => {
          if (!value || typeof value !== "object" || Array.isArray(value)) return [];
          const order = value as Record<string, unknown>;
          const direction = String(order.direction || "");
          const shares = Number(order.shares);
          const orderPrice = Number(order.orderPrice);
          if ((direction !== "Buy" && direction !== "Sell") || !Number.isFinite(shares) || shares <= 0 || !Number.isFinite(orderPrice) || orderPrice <= 0) return [];
          return [{
            orderId: String(order.orderId || ""),
            orderDate: String(order.orderDate || ""),
            timeValue: String(order.timeValue || ""),
            direction: direction as "Buy" | "Sell",
            shares,
            symbol: String(order.symbol || "").trim().toUpperCase(),
            orderType: String(order.orderType || "").trim().toUpperCase(),
            orderPrice
          }];
        })
      : [];
    portfolioMeta[portfolio] = {
      currentEquity: Number.isFinite(currentEquity) && currentEquity > 0 ? currentEquity : undefined,
      statementEquity: Number.isFinite(statementEquity) && statementEquity > 0 ? statementEquity : undefined,
      floatingPnl: Number.isFinite(floatingPnl) ? floatingPnl : undefined,
      equitySource: String(raw.equitySource || ""),
      equityUpdatedAt: String(raw.equityUpdatedAt || ""),
      equityStatementDate: String(raw.equityStatementDate || ""),
      workingOrders
    };
  });

  return {
    portfolios,
    defaultPortfolio: defaultPortfolio && portfolios.includes(defaultPortfolio) ? defaultPortfolio : "",
    portfolioMeta
  };
}

function checklistItems(value: unknown): TradeChecklistItem[] {
  const rawItems = Array.isArray(value)
    ? value
    : typeof value === "string"
      ? (() => {
          try {
            const parsed = JSON.parse(value);
            return Array.isArray(parsed) ? parsed : [];
          } catch {
            return [];
          }
        })()
      : [];

  return rawItems
    .map((item, index) => {
      if (!item || typeof item !== "object") {
        return null;
      }

      const rawItem = item as Record<string, unknown>;
      const criteria = String(rawItem.criteria || "").trim();
      const points = Number(rawItem.points);

      if (!criteria || !Number.isFinite(points)) {
        return null;
      }

      return {
        id: String(rawItem.id || `criteria-${index}`),
        criteria,
        points,
        met: Boolean(rawItem.met)
      };
    })
    .filter(Boolean) as TradeChecklistItem[];
}

function tradeExecutions(value: unknown): TradeExecution[] {
  const rawItems = Array.isArray(value)
    ? value
    : typeof value === "string"
      ? (() => {
          try {
            const parsed = JSON.parse(value);
            return Array.isArray(parsed) ? parsed : [];
          } catch {
            return [];
          }
        })()
      : [];

  return rawItems
    .map((item, index) => {
      if (!item || typeof item !== "object") {
        return null;
      }

      const rawItem = item as Record<string, unknown>;
      const id = String(rawItem.id || rawItem.sourceKey || `execution-${index}`).trim();
      const type = rawItem.type === "EXIT" ? "EXIT" : "ENTRY";
      const side = rawItem.side === "SHORT" ? "SHORT" : "LONG";

      return {
        id,
        type,
        date: String(rawItem.date || ""),
        time: String(rawItem.time || ""),
        side,
        shares: numberValue(rawItem.shares),
        price: numberValue(rawItem.price),
        pnl: numberValue(rawItem.pnl),
        commission: numberValue(rawItem.commission),
        source: String(rawItem.source || ""),
        sourceKey: String(rawItem.sourceKey || "")
      };
    })
    .filter(Boolean) as TradeExecution[];
}

function rowToTrade(row: Record<string, unknown>): TradeLogEntry {
  return {
    id: String(row.id),
    userId: String(row.user_id),
    importSource: String(row.import_source || ""),
    importRowKey: String(row.import_row_key || ""),
    symbol: String(row.symbol || ""),
    side: String(row.side || "LONG") as TradeSide,
    status: String(row.status || "OPEN") as TradeStatus,
    entryDate: String(row.entry_date || ""),
    exitDate: String(row.exit_date || ""),
    openTime: String(row.open_time || ""),
    closeTime: String(row.close_time || ""),
    avgEntry: Number(row.avg_entry),
    exitPrice: Number(row.exit_price),
    stopPrice: Number(row.stop_price),
    takeProfitPrice: Number(row.take_profit_price),
    shares: Number(row.shares),
    commission: Number(row.commission),
    usedMargin: Number(row.used_margin),
    risk: Number(row.risk),
    pnl: Number(row.pnl),
    rMultiple: Number(row.r_multiple),
    returnPercent: Number(row.return_percent),
    daysInTrade: Number(row.days_in_trade),
    setupTags: stringArray(row.setup_tags),
    mistakeTags: stringArray(row.mistake_tags),
    customTags: stringArray(row.custom_tags),
    manualGrade: String(row.manual_grade || ""),
    portfolioTag: String(row.portfolio_tag || ""),
    emotion: String(row.emotion || ""),
    tradeQuality: String(row.trade_quality || ""),
    checklistItems: checklistItems(row.checklist_items),
    notes: String(row.notes || ""),
    reviewSections: normalizeTradeReviewSections(row.review_sections),
    screenshots: stringArray(row.screenshots),
    chartLinks: stringArray(row.chart_links),
    executions: tradeExecutions(row.executions),
    hidden: Boolean(row.hidden),
    groupId: "",
    groupRole: "none",
    createdAt: new Date(String(row.created_at)).toISOString(),
    updatedAt: new Date(String(row.updated_at)).toISOString()
  };
}

function numberValue(value: unknown) {
  const number = Number(value);
  return Number.isFinite(number) ? number : 0;
}

function weightedAverage<T>(items: T[], value: (item: T) => number, weight: (item: T) => number) {
  const weighted = items.reduce((sum, item) => sum + value(item) * Math.abs(weight(item)), 0);
  const weights = items.reduce((sum, item) => sum + Math.abs(weight(item)), 0);
  return weights ? weighted / weights : 0;
}

const CF_SYSTEM_TAGS = new Set([
  "CF Statement",
  "Open Position",
  "Closed Transaction",
  "Partial exits",
  "Needs review",
  "Combined trade",
  "Auto recalculated"
]);

function isCfBaselineRow(importRowKey: string) {
  return importRowKey.startsWith("cf-baseline:");
}

function executionSourceKeys(executions: TradeExecution[] | undefined) {
  return new Set(
    (executions || [])
      .map((execution) => String(execution.sourceKey || "").trim())
      .filter(Boolean)
  );
}

function stableTradeId(input: TradeLogInput) {
  if (input.importSource.trim() && input.importRowKey.trim()) {
    const hash = crypto
      .createHash("sha1")
      .update([input.userId, input.portfolioTag || "", input.importSource, input.importRowKey].join("|"))
      .digest("hex")
      .slice(0, 16);
    return `${input.userId}-${hash}`;
  }

  return `${input.userId}-${crypto.randomUUID()}`;
}

function executionTimestamp(execution: TradeExecution) {
  return `${execution.date || ""}T${String(execution.time || "00:00:00").replace(/\.\d+$/, "")}`;
}

function uniqueExecutions(executions: TradeExecution[]) {
  const seen = new Set<string>();
  return executions
    .filter((execution) => {
      const key = execution.sourceKey || execution.id;
      if (seen.has(key)) {
        return false;
      }
      seen.add(key);
      return true;
    })
    .sort((a, b) => executionTimestamp(a).localeCompare(executionTimestamp(b)));
}

function mergedCfCustomTags(existing: string[], incoming: string[]) {
  const preservedUserTags = existing.filter((tag) => !CF_SYSTEM_TAGS.has(tag));
  return Array.from(new Set([...incoming, ...preservedUserTags]));
}

function mergeCfImportIntoExisting(existing: TradeLogEntry, input: TradeLogInput, preserveBrokerFields = false): TradeLogInput {
  const nextStop = input.stopPrice || existing.stopPrice || 0;
  const nextTakeProfit = input.takeProfitPrice || existing.takeProfitPrice || 0;

  return {
    ...input,
    status: preserveBrokerFields ? existing.status : input.status,
    entryDate: preserveBrokerFields ? existing.entryDate : input.entryDate,
    exitDate: preserveBrokerFields ? existing.exitDate : input.exitDate,
    openTime: preserveBrokerFields ? existing.openTime : input.openTime || existing.openTime || "",
    closeTime: preserveBrokerFields ? existing.closeTime : input.closeTime || existing.closeTime || "",
    avgEntry: preserveBrokerFields ? existing.avgEntry : input.avgEntry,
    exitPrice: preserveBrokerFields ? existing.exitPrice : input.exitPrice,
    shares: preserveBrokerFields ? existing.shares : input.shares,
    commission: preserveBrokerFields ? existing.commission : input.commission,
    usedMargin: preserveBrokerFields ? existing.usedMargin : input.usedMargin,
    pnl: preserveBrokerFields ? existing.pnl : input.pnl,
    rMultiple: preserveBrokerFields ? existing.rMultiple : input.rMultiple,
    returnPercent: preserveBrokerFields ? existing.returnPercent : input.returnPercent,
    daysInTrade: preserveBrokerFields ? existing.daysInTrade : input.daysInTrade,
    risk: input.risk || existing.risk || 0,
    stopPrice: nextStop,
    takeProfitPrice: nextTakeProfit,
    setupTags: input.setupTags.length ? input.setupTags : existing.setupTags,
    mistakeTags: input.mistakeTags.length ? input.mistakeTags : existing.mistakeTags,
    customTags: mergedCfCustomTags(existing.customTags, input.customTags),
    manualGrade: input.manualGrade || existing.manualGrade || "",
    emotion: input.emotion || existing.emotion || "",
    tradeQuality: input.tradeQuality || existing.tradeQuality || "",
    checklistItems: input.checklistItems.length ? input.checklistItems : existing.checklistItems,
    notes: input.notes || existing.notes || "",
    reviewSections: normalizeTradeReviewSections(input.reviewSections || existing.reviewSections),
    screenshots: input.screenshots.length ? input.screenshots : existing.screenshots,
    chartLinks: input.chartLinks.length ? input.chartLinks : existing.chartLinks,
    executions: input.executions?.length ? input.executions : existing.executions
  };
}

function mergeCfContinuationIntoOpenTrade(existing: TradeLogEntry, input: TradeLogInput): TradeLogInput {
  const executions = uniqueExecutions([...(existing.executions || []), ...(input.executions || [])]);
  const entryExecutions = executions.filter((execution) => execution.type === "ENTRY");
  const exitExecutions = executions.filter((execution) => execution.type === "EXIT");
  const entryShares = entryExecutions.reduce((sum, execution) => sum + Math.max(0, numberValue(execution.shares)), 0);
  const exitShares = exitExecutions.reduce((sum, execution) => sum + Math.max(0, numberValue(execution.shares)), 0);
  const remainingShares = Math.max(0, entryShares - exitShares);
  const entryValue = entryExecutions.reduce((sum, execution) => sum + numberValue(execution.price) * Math.max(0, numberValue(execution.shares)), 0);
  const exitValue = exitExecutions.reduce((sum, execution) => sum + numberValue(execution.price) * Math.max(0, numberValue(execution.shares)), 0);
  const pnl = exitExecutions.reduce((sum, execution) => sum + numberValue(execution.pnl), 0);
  const commission = executions.reduce((sum, execution) => sum + numberValue(execution.commission), 0);
  const latestExit = [...exitExecutions].sort((a, b) => executionTimestamp(a).localeCompare(executionTimestamp(b))).at(-1);
  const isOpen = remainingShares > 0.000001;
  const risk = existing.risk || input.risk || 0;
  const avgEntry = entryShares ? entryValue / entryShares : existing.avgEntry || input.avgEntry || 0;
  const exitPrice = exitShares ? exitValue / exitShares : existing.exitPrice || input.exitPrice || 0;
  const customTags = mergedCfCustomTags(existing.customTags, input.customTags);

  if (exitExecutions.length > 1 || (isOpen && exitExecutions.length > 0)) {
    customTags.push("Partial exits");
  }

  return {
    ...input,
    importRowKey: existing.importRowKey || input.importRowKey,
    status: tradeStatusFromPnl(pnl, !isOpen, risk ? pnl / risk : 0),
    entryDate: existing.entryDate || input.entryDate,
    exitDate: isOpen ? "" : latestExit?.date || input.exitDate || "",
    openTime: existing.openTime || input.openTime || "",
    closeTime: isOpen ? "" : latestExit?.time || input.closeTime || "",
    avgEntry,
    exitPrice,
    stopPrice: input.stopPrice || existing.stopPrice || 0,
    takeProfitPrice: input.takeProfitPrice || existing.takeProfitPrice || 0,
    shares: isOpen ? remainingShares : entryShares || existing.shares || input.shares,
    commission,
    usedMargin: input.usedMargin || existing.usedMargin || 0,
    risk,
    pnl,
    rMultiple: risk ? pnl / risk : 0,
    returnPercent: entryValue ? (pnl / entryValue) * 100 : 0,
    daysInTrade: daysBetween(existing.entryDate || input.entryDate, isOpen ? new Date().toISOString().slice(0, 10) : latestExit?.date || input.exitDate || ""),
    setupTags: input.setupTags.length ? input.setupTags : existing.setupTags,
    mistakeTags: input.mistakeTags.length ? input.mistakeTags : existing.mistakeTags,
    customTags: Array.from(new Set(customTags)),
    manualGrade: input.manualGrade || existing.manualGrade || "",
    emotion: input.emotion || existing.emotion || "",
    tradeQuality: input.tradeQuality || existing.tradeQuality || "",
    checklistItems: input.checklistItems.length ? input.checklistItems : existing.checklistItems,
    notes: existing.notes || input.notes || "",
    reviewSections: normalizeTradeReviewSections(existing.reviewSections || input.reviewSections),
    screenshots: existing.screenshots.length ? existing.screenshots : input.screenshots,
    chartLinks: existing.chartLinks.length ? existing.chartLinks : input.chartLinks,
    executions
  };
}

const BREAKEVEN_R_THRESHOLD = 0.1;

function tradeStatusFromPnl(pnl: number, closed: boolean, rMultiple = 0): TradeStatus {
  if (!closed) {
    return "OPEN";
  }
  if (Math.abs(rMultiple) < BREAKEVEN_R_THRESHOLD) {
    return "BREAKEVEN";
  }
  if (pnl > 0) {
    return "WIN";
  }
  if (pnl < 0) {
    return "LOSS";
  }
  return "BREAKEVEN";
}

async function findExistingCfTradeByExecutionOverlap(
  userId: string,
  portfolioTag: string,
  symbol: string,
  side: TradeSide,
  executions: TradeExecution[]
) {
  const sourceKeys = Array.from(executionSourceKeys(executions));

  if (!sourceKeys.length) {
    return null;
  }

  const db = getPool();

  if (!db) {
    const trades = await readLocalTrades();
    const candidates = trades.filter(
      (trade) =>
        trade.userId === userId &&
        trade.portfolioTag === portfolioTag &&
        trade.importSource === "cf-statement-pdf" &&
        trade.symbol === symbol &&
        trade.side === side &&
        !trade.hidden
    );

    return (
      candidates
        .map((trade) => ({
          trade,
          overlap: Array.from(executionSourceKeys(trade.executions)).filter((key) => sourceKeys.includes(key)).length
        }))
        .filter((candidate) => candidate.overlap > 0)
        .sort((a, b) => {
          const baselineDelta = Number(isCfBaselineRow(a.trade.importRowKey)) - Number(isCfBaselineRow(b.trade.importRowKey));
          if (baselineDelta !== 0) {
            return baselineDelta;
          }
          return b.overlap - a.overlap;
        })[0]?.trade || null
    );
  }

  await ensureTradeTable();
  const result = await db.query(
    `
      select ${tradeColumns}
      from trade_logs
      where user_id = $1
        and portfolio_tag = $2
        and import_source = 'cf-statement-pdf'
        and symbol = $3
        and side = $4
        and hidden = false
      order by created_at asc
    `,
    [userId, portfolioTag, symbol, side]
  );

  const candidates = result.rows.map(rowToTrade);
  return (
    candidates
      .map((trade) => ({
        trade,
        overlap: Array.from(executionSourceKeys(trade.executions)).filter((key) => sourceKeys.includes(key)).length
      }))
      .filter((candidate) => candidate.overlap > 0)
      .sort((a, b) => {
        const baselineDelta = Number(isCfBaselineRow(a.trade.importRowKey)) - Number(isCfBaselineRow(b.trade.importRowKey));
        if (baselineDelta !== 0) {
          return baselineDelta;
        }
        return b.overlap - a.overlap;
      })[0]?.trade || null
  );
}

async function findExistingOpenCfTradeForContinuation(
  userId: string,
  portfolioTag: string,
  symbol: string,
  side: TradeSide,
  entryDate: string
) {
  const db = getPool();

  if (!db) {
    const trades = await readLocalTrades();
    return (
      trades
        .filter(
          (trade) =>
            trade.userId === userId &&
            trade.portfolioTag === portfolioTag &&
            trade.importSource === "cf-statement-pdf" &&
            trade.symbol === symbol &&
            trade.side === side &&
            trade.status === "OPEN" &&
            !trade.hidden &&
            (!entryDate || trade.entryDate <= entryDate)
        )
        .sort((a, b) => b.entryDate.localeCompare(a.entryDate) || b.openTime.localeCompare(a.openTime))[0] || null
    );
  }

  await ensureTradeTable();
  const result = await db.query(
    `
      select ${tradeColumns}
      from trade_logs
      where user_id = $1
        and portfolio_tag = $2
        and import_source = 'cf-statement-pdf'
        and symbol = $3
        and side = $4
        and status = 'OPEN'
        and hidden = false
        and ($5 = '' or entry_date <= $5)
      order by entry_date desc, open_time desc, created_at desc
      limit 1
    `,
    [userId, portfolioTag, symbol, side, entryDate || ""]
  );

  return result.rows[0] ? rowToTrade(result.rows[0]) : null;
}

function daysBetween(start: string, end: string) {
  if (!start || !end) {
    return 0;
  }

  const startDate = new Date(`${start}T00:00:00`);
  const endDate = new Date(`${end}T00:00:00`);
  const diff = endDate.getTime() - startDate.getTime();
  return Number.isFinite(diff) ? Math.max(0, Math.round(diff / 86400000)) : 0;
}

async function ensureTable() {
  const db = getPool();

  if (!db) {
    return;
  }

  await db.query(`
    create table if not exists monthly_reports (
      id text primary key,
      user_id text not null,
      month text not null,
      account_size numeric not null default 0,
      total_return numeric not null default 0,
      percent_return numeric not null default 0,
      net_pnl numeric not null default 0,
      total_payouts numeric not null default 0,
      total_trades integer not null default 0,
      win_rate numeric not null default 0,
      avg_r numeric not null default 0,
      total_r numeric not null default 0,
      avg_win_r numeric not null default 0,
      avg_loss_r numeric not null default 0,
      avg_win numeric not null default 0,
      avg_loss numeric not null default 0,
      avg_risk numeric not null default 0,
      current_risk_percent numeric not null default 0,
      expected_value_r numeric not null default 0,
      sharpe_ratio numeric not null default 0,
      avg_trade_length numeric not null default 0,
      avg_swing_length numeric not null default 0,
      longest_win_streak integer not null default 0,
      longest_loss_streak integer not null default 0,
      notes text not null default '',
      created_at timestamptz not null default now(),
      updated_at timestamptz not null default now(),
      unique(user_id, month)
    );
  `);
}

async function ensureTradeTable() {
  const db = getPool();

  if (!db) {
    return;
  }

  await db.query(`
    create table if not exists trade_logs (
      id text primary key,
      user_id text not null,
      import_source text not null default '',
      import_row_key text not null default '',
      symbol text not null,
      side text not null,
      status text not null,
      entry_date text not null,
      exit_date text not null default '',
      open_time text not null default '',
      close_time text not null default '',
      avg_entry numeric not null default 0,
      exit_price numeric not null default 0,
      stop_price numeric not null default 0,
      take_profit_price numeric not null default 0,
      shares numeric not null default 0,
      commission numeric not null default 0,
      used_margin numeric not null default 0,
      risk numeric not null default 0,
      pnl numeric not null default 0,
      r_multiple numeric not null default 0,
      return_percent numeric not null default 0,
      days_in_trade numeric not null default 0,
      setup_tags jsonb not null default '[]'::jsonb,
      mistake_tags jsonb not null default '[]'::jsonb,
      custom_tags jsonb not null default '[]'::jsonb,
      manual_grade text not null default '',
      portfolio_tag text not null default '',
      emotion text not null default '',
      trade_quality text not null default '',
      checklist_items jsonb not null default '[]'::jsonb,
      notes text not null default '',
      review_sections jsonb not null default '{}'::jsonb,
      screenshots jsonb not null default '[]'::jsonb,
      chart_links jsonb not null default '[]'::jsonb,
      executions jsonb not null default '[]'::jsonb,
      hidden boolean not null default false,
      group_id text not null default '',
      group_role text not null default 'none',
      created_at timestamptz not null default now(),
      updated_at timestamptz not null default now()
    );
  `);
  await db.query("alter table trade_logs add column if not exists checklist_items jsonb not null default '[]'::jsonb");
  await db.query("alter table trade_logs add column if not exists review_sections jsonb not null default '{}'::jsonb");
  await db.query("alter table trade_logs add column if not exists chart_links jsonb not null default '[]'::jsonb");
  await db.query("alter table trade_logs add column if not exists executions jsonb not null default '[]'::jsonb");
  await db.query("alter table trade_logs add column if not exists manual_grade text not null default ''");
  await db.query("alter table trade_logs add column if not exists open_time text not null default ''");
  await db.query("alter table trade_logs add column if not exists close_time text not null default ''");
  await db.query("alter table trade_logs add column if not exists import_source text not null default ''");
  await db.query("alter table trade_logs add column if not exists import_row_key text not null default ''");
  await db.query("alter table trade_logs add column if not exists stop_price numeric not null default 0");
  await db.query("alter table trade_logs add column if not exists take_profit_price numeric not null default 0");
  await db.query("alter table trade_logs add column if not exists commission numeric not null default 0");
  await db.query("alter table trade_logs add column if not exists used_margin numeric not null default 0");
  await db.query("alter table trade_logs add column if not exists portfolio_tag text not null default ''");
  await db.query("alter table trade_logs add column if not exists emotion text not null default ''");
  await db.query("alter table trade_logs add column if not exists trade_quality text not null default ''");
  await db.query("alter table trade_logs add column if not exists hidden boolean not null default false");
  await db.query("alter table trade_logs add column if not exists group_id text not null default ''");
  await db.query("alter table trade_logs add column if not exists group_role text not null default 'none'");
  await db.query("create index if not exists trade_logs_user_hidden_idx on trade_logs (user_id, hidden)");
  await db.query("create index if not exists trade_logs_user_status_hidden_idx on trade_logs (user_id, status, hidden)");
  await db.query("create index if not exists trade_logs_user_portfolio_idx on trade_logs (user_id, portfolio_tag)");
  await db.query("create index if not exists trade_logs_user_entry_date_idx on trade_logs (user_id, entry_date)");
  await db.query("create index if not exists trade_logs_user_exit_date_idx on trade_logs (user_id, exit_date)");
}

async function ensureSettingsTable() {
  const db = getPool();

  if (!db) {
    return;
  }

  await db.query(`
    create table if not exists app_settings (
      key text primary key,
      value jsonb not null,
      updated_at timestamptz not null default now()
    );
  `);
}

async function ensureBrokerPortfolioSnapshotTable() {
  const db = getPool();
  if (!db) return;
  await db.query(`
    create table if not exists broker_portfolio_snapshots (
      user_id text not null,
      portfolio_tag text not null,
      coverage_date text not null,
      source_hash text not null,
      source_filename text not null default '',
      source text not null default 'CF_STATEMENT_PDF',
      balance numeric,
      current_equity numeric,
      statement_equity numeric,
      floating_pnl numeric,
      open_positions jsonb not null default '[]'::jsonb,
      working_orders jsonb not null default '[]'::jsonb,
      imported_at timestamptz not null default now(),
      primary key (user_id, portfolio_tag, coverage_date)
    );
  `);
  await db.query("create index if not exists broker_portfolio_snapshots_lookup_idx on broker_portfolio_snapshots (user_id, portfolio_tag, coverage_date desc)");
}

async function readLocalReports(): Promise<MonthlyReport[]> {
  if (process.env.NODE_ENV === "production") {
    return [];
  }

  try {
    const raw = await fs.readFile(localDataFile, "utf8");
    return JSON.parse(raw) as MonthlyReport[];
  } catch {
    return [];
  }
}

async function writeLocalReports(reports: MonthlyReport[]) {
  await fs.mkdir(path.dirname(localDataFile), { recursive: true });
  await fs.writeFile(localDataFile, JSON.stringify(reports, null, 2));
}

async function readLocalTrades(): Promise<TradeLogEntry[]> {
  if (process.env.NODE_ENV === "production") {
    return [];
  }

  try {
    const raw = await fs.readFile(localTradesFile, "utf8");
    return (JSON.parse(raw) as TradeLogEntry[]).map((trade) => ({
      ...trade,
      reviewSections: normalizeTradeReviewSections(trade.reviewSections)
    }));
  } catch {
    return [];
  }
}

async function writeLocalTrades(trades: TradeLogEntry[]) {
  await fs.mkdir(path.dirname(localTradesFile), { recursive: true });
  await fs.writeFile(localTradesFile, JSON.stringify(trades, null, 2));
}

function tradeLogOrder(a: TradeLogEntry, b: TradeLogEntry) {
  return (
    b.entryDate.localeCompare(a.entryDate) ||
    (b.openTime || "").localeCompare(a.openTime || "") ||
    b.createdAt.localeCompare(a.createdAt) ||
    b.id.localeCompare(a.id)
  );
}

async function readLocalSettings(): Promise<Record<string, unknown>> {
  if (process.env.NODE_ENV === "production") {
    return {};
  }

  try {
    const raw = await fs.readFile(localSettingsFile, "utf8");
    return JSON.parse(raw) as Record<string, unknown>;
  } catch {
    return {};
  }
}

async function writeLocalSettings(settings: Record<string, unknown>) {
  await fs.mkdir(path.dirname(localSettingsFile), { recursive: true });
  await fs.writeFile(localSettingsFile, JSON.stringify(settings, null, 2));
}

function normalizeGradeBands(value: unknown): ChecklistGradeBand[] {
  if (!Array.isArray(value)) {
    return defaultChecklistGradeBands;
  }

  const bands = value
    .map((band, index) => {
      if (!band || typeof band !== "object") {
        return null;
      }

      const rawBand = band as Record<string, unknown>;
      const label = String(rawBand.label || "").trim();
      const minScore = Number(rawBand.minScore);
      const maxScore = rawBand.maxScore === null || rawBand.maxScore === "" ? null : Number(rawBand.maxScore);

      if (!label || !Number.isFinite(minScore) || (maxScore !== null && !Number.isFinite(maxScore))) {
        return null;
      }

      return {
        id: String(rawBand.id || `grade-${index}`),
        label,
        minScore,
        maxScore
      };
    })
    .filter(Boolean) as ChecklistGradeBand[];

  return bands.length ? bands.sort((a, b) => b.minScore - a.minScore) : defaultChecklistGradeBands;
}

function normalizeSetupTemplates(value: unknown): SetupChecklistTemplate[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value
    .map((template, index) => {
      if (!template || typeof template !== "object") {
        return null;
      }

      const rawTemplate = template as Record<string, unknown>;
      const setupName = String(rawTemplate.setupName || "").trim();

      if (!setupName) {
        return null;
      }

      const groups = normalizeTemplateGroups(rawTemplate.groups, rawTemplate.criteria);

      return {
        id: String(rawTemplate.id || `setup-template-${index}`),
        setupName,
        description: String(rawTemplate.description || ""),
        knowledgeSources: normalizeSetupKnowledgeSources(rawTemplate.knowledgeSources),
        strategyExamples: normalizeSetupStrategyExamples(rawTemplate.strategyExamples),
        gradeBands: normalizeGradeBands(rawTemplate.gradeBands),
        criteria: groups.flatMap((group) => group.criteria),
        groups
      };
    })
    .filter(Boolean) as SetupChecklistTemplate[];
}

function normalizeSetupKnowledgeSources(value: unknown) {
  if (!Array.isArray(value)) {
    return [];
  }

  return value
    .map((source, index) => {
      if (!source || typeof source !== "object") {
        return null;
      }

      const rawSource = source as Record<string, unknown>;
      const title = String(rawSource.title || "").trim();
      const content = String(rawSource.content || "").trim();
      const url = String(rawSource.url || "").trim();
      const sourceType = String(rawSource.sourceType || "notes").toLowerCase();

      if (!title && !content && !url) {
        return null;
      }

      return {
        id: String(rawSource.id || `setup-knowledge-${index}`),
        title: title || "Strategy knowledge",
        sourceType: sourceType === "resource" || sourceType === "document" ? sourceType : "notes",
        url,
        content,
        chunks: normalizeSetupKnowledgeChunks(rawSource.chunks),
        active: rawSource.active === false ? false : true,
        createdAt: String(rawSource.createdAt || new Date().toISOString()),
        updatedAt: String(rawSource.updatedAt || new Date().toISOString())
      };
    })
    .filter(Boolean);
}

function normalizeSetupKnowledgeChunks(value: unknown) {
  if (!Array.isArray(value)) {
    return [];
  }

  return value
    .map((chunk, index) => {
      if (!chunk || typeof chunk !== "object") {
        return null;
      }

      const rawChunk = chunk as Record<string, unknown>;
      const content = String(rawChunk.content || "").trim();

      if (!content) {
        return null;
      }

      return {
        id: String(rawChunk.id || `setup-knowledge-chunk-${index}`),
        title: String(rawChunk.title || `Section ${index + 1}`).trim() || `Section ${index + 1}`,
        content,
        order: Number.isFinite(Number(rawChunk.order)) ? Number(rawChunk.order) : index
      };
    })
    .filter((chunk): chunk is { id: string; title: string; content: string; order: number } => Boolean(chunk))
    .sort((a, b) => a.order - b.order);
}

function normalizeSetupStrategyExamples(value: unknown) {
  if (!Array.isArray(value)) {
    return [];
  }

  return value
    .map((example, index) => {
      if (!example || typeof example !== "object") {
        return null;
      }

      const rawExample = example as Record<string, unknown>;
      const symbol = String(rawExample.symbol || "").trim().toUpperCase();
      const setupType = String(rawExample.setupType || "").trim();
      const notes = String(rawExample.notes || "").trim();
      const sourceUrl = String(rawExample.sourceUrl || "").trim();
      const source = String(rawExample.source || "").trim();
      const screenshots = Array.isArray(rawExample.screenshots)
        ? rawExample.screenshots.map((screenshot) => String(screenshot || "").trim()).filter(Boolean)
        : [];
      const quality = String(rawExample.quality || "good").toLowerCase();

      if (!symbol && !setupType && !notes && !sourceUrl && !screenshots.length) {
        return null;
      }

      return {
        id: String(rawExample.id || `setup-example-${index}`),
        symbol,
        setupType,
        quality: quality === "ideal" || quality === "failed" || quality === "bad" || quality === "cautionary" ? quality : "good",
        outcome: String(rawExample.outcome || "").trim(),
        source,
        sourceUrl,
        notes,
        screenshots,
        active: rawExample.active === false ? false : true,
        createdAt: String(rawExample.createdAt || new Date().toISOString()),
        updatedAt: String(rawExample.updatedAt || new Date().toISOString())
      };
    })
    .filter(Boolean);
}

function normalizeChecklistInputType(value: unknown): ChecklistInputType {
  return String(value || "").toLowerCase() === "points" ? "points" : "boolean";
}

function normalizeTemplateCriteria(value: unknown): SetupTemplateCriterion[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value
    .map((item, index) => {
      if (!item || typeof item !== "object") {
        return null;
      }

      const rawItem = item as Record<string, unknown>;
      const criteria = String(rawItem.criteria || "").trim();
      const points = Number(rawItem.points);

      if (!criteria || !Number.isFinite(points)) {
        return null;
      }

      return {
        id: String(rawItem.id || `criteria-${index}`),
        criteria,
        points,
        inputType: normalizeChecklistInputType(rawItem.inputType),
        importTagKey: String(rawItem.importTagKey || "").trim(),
        importTagValue: String(rawItem.importTagValue || "").trim()
      } satisfies SetupTemplateCriterion;
    })
    .filter(Boolean) as SetupTemplateCriterion[];
}

function normalizeTemplateGroups(groupsValue: unknown, criteriaValue?: unknown): SetupChecklistGroup[] {
  if (Array.isArray(groupsValue) && groupsValue.length) {
    const groups = groupsValue
      .map((group, index) => {
        if (!group || typeof group !== "object") {
          return null;
        }

        const rawGroup = group as Record<string, unknown>;
        const name = String(rawGroup.name || "").trim();
        const criteria = normalizeTemplateCriteria(rawGroup.criteria);

        if (!name && !criteria.length) {
          return null;
        }

        return {
          id: String(rawGroup.id || `group-${index}`),
          name: name || `Group ${index + 1}`,
          criteria
        } satisfies SetupChecklistGroup;
      })
      .filter(Boolean) as SetupChecklistGroup[];

    return groups;
  }

  const legacyCriteria = normalizeTemplateCriteria(criteriaValue);
  return legacyCriteria.length
    ? [
        {
          id: "group-default",
          name: "Checklist",
          criteria: legacyCriteria
        }
      ]
    : [];
}

export function getDefaultChecklistGradeBands() {
  return defaultChecklistGradeBands;
}

export async function getChecklistGradeBands() {
  const db = getPool();

  if (!db) {
    const settings = await readLocalSettings();
    return normalizeGradeBands(settings.checklistGradeBands);
  }

  await ensureSettingsTable();
  const result = await db.query("select value from app_settings where key = $1", ["checklist_grade_bands"]);
  return normalizeGradeBands(result.rows[0]?.value);
}

export async function saveChecklistGradeBands(bands: ChecklistGradeBand[]) {
  const normalized = normalizeGradeBands(bands);
  const db = getPool();

  if (!db) {
    if (process.env.NODE_ENV === "production") {
      throw new Error("DATABASE_URL is required in production before settings can be saved.");
    }

    const settings = await readLocalSettings();
    settings.checklistGradeBands = normalized;
    await writeLocalSettings(settings);
    return normalized;
  }

  await ensureSettingsTable();
  const result = await db.query(
    `
      insert into app_settings (key, value)
      values ($1, $2::jsonb)
      on conflict (key) do update set value = excluded.value, updated_at = now()
      returning value;
    `,
    ["checklist_grade_bands", JSON.stringify(normalized)]
  );
  return normalizeGradeBands(result.rows[0]?.value);
}

export async function getBrandenPortfolioNames() {
  const settings = await getBrandenPortfolioSettings();
  return settings.portfolios;
}

function normalizeBrokerPortfolioSnapshotList(value: unknown): BrokerPortfolioSnapshot[] {
  if (!Array.isArray(value)) return [];
  return value.flatMap((item) => {
    try {
      if (!item || typeof item !== "object" || Array.isArray(item)) return [];
      return [normalizeBrokerPortfolioSnapshot(item as BrokerPortfolioSnapshot)];
    } catch {
      return [];
    }
  });
}

function rowToBrokerPortfolioSnapshot(row: Record<string, unknown>): BrokerPortfolioSnapshot {
  return normalizeBrokerPortfolioSnapshot({
    userId: String(row.user_id || ""),
    portfolioTag: String(row.portfolio_tag || ""),
    coverageDate: String(row.coverage_date || ""),
    sourceHash: String(row.source_hash || ""),
    sourceFilename: String(row.source_filename || ""),
    source: "CF_STATEMENT_PDF",
    importedAt: new Date(String(row.imported_at)).toISOString(),
    balance: row.balance === null ? undefined : Number(row.balance),
    currentEquity: row.current_equity === null ? undefined : Number(row.current_equity),
    statementEquity: row.statement_equity === null ? undefined : Number(row.statement_equity),
    floatingPnl: row.floating_pnl === null ? undefined : Number(row.floating_pnl),
    openPositions: Array.isArray(row.open_positions) ? row.open_positions as never[] : [],
    workingOrders: Array.isArray(row.working_orders) ? row.working_orders as CfWorkingOrderMetadata[] : []
  });
}

export async function listBrokerPortfolioSnapshots(userId: string, portfolioTag?: string) {
  const db = getPool();
  if (!db) {
    const settings = await readLocalSettings();
    return normalizeBrokerPortfolioSnapshotList(settings.brandenBrokerPortfolioSnapshots)
      .filter((snapshot) => snapshot.userId === userId && (!portfolioTag || snapshot.portfolioTag === portfolioTag));
  }
  try {
    const result = portfolioTag
      ? await db.query(
          "select * from broker_portfolio_snapshots where user_id = $1 and portfolio_tag = $2 order by coverage_date",
          [userId, portfolioTag]
        )
      : await db.query(
          "select * from broker_portfolio_snapshots where user_id = $1 order by portfolio_tag, coverage_date",
          [userId]
        );
    return result.rows.map(rowToBrokerPortfolioSnapshot);
  } catch (error) {
    if ((error as { code?: string }).code === "42P01") return [];
    throw error;
  }
}

async function upsertBrokerPortfolioSnapshotWithClient(
  client: PoolClient,
  input: BrokerPortfolioSnapshotInput,
  importedAt: string
) {
  const snapshot = normalizeBrokerPortfolioSnapshot(input, importedAt);
  await client.query(
    `
      insert into broker_portfolio_snapshots (
        user_id, portfolio_tag, coverage_date, source_hash, source_filename, source,
        balance, current_equity, statement_equity, floating_pnl, open_positions, working_orders, imported_at
      ) values ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11::jsonb, $12::jsonb, $13)
      on conflict (user_id, portfolio_tag, coverage_date) do update set
        source_hash = excluded.source_hash,
        source_filename = excluded.source_filename,
        source = excluded.source,
        balance = excluded.balance,
        current_equity = excluded.current_equity,
        statement_equity = excluded.statement_equity,
        floating_pnl = excluded.floating_pnl,
        open_positions = excluded.open_positions,
        working_orders = excluded.working_orders,
        imported_at = excluded.imported_at
      where broker_portfolio_snapshots.source_hash is distinct from excluded.source_hash
         or broker_portfolio_snapshots.balance is distinct from excluded.balance
         or broker_portfolio_snapshots.current_equity is distinct from excluded.current_equity
         or broker_portfolio_snapshots.statement_equity is distinct from excluded.statement_equity
         or broker_portfolio_snapshots.floating_pnl is distinct from excluded.floating_pnl
         or broker_portfolio_snapshots.open_positions is distinct from excluded.open_positions
         or broker_portfolio_snapshots.working_orders is distinct from excluded.working_orders
    `,
    [
      snapshot.userId, snapshot.portfolioTag, snapshot.coverageDate, snapshot.sourceHash,
      snapshot.sourceFilename, snapshot.source, snapshot.balance, snapshot.currentEquity,
      snapshot.statementEquity, snapshot.floatingPnl, JSON.stringify(snapshot.openPositions),
      JSON.stringify(snapshot.workingOrders), snapshot.importedAt
    ]
  );
}

export async function getBrandenPortfolioSettings(): Promise<BrandenPortfolioSettings> {
  if (!getPool()) {
    const settings = await readLocalSettings();
    return normalizeBrandenPortfolioSettings(settings.brandenPortfolioNames);
  }

  await ensureSettingsTable();
  const db = getPool();

  if (!db) {
    return { portfolios: [], defaultPortfolio: "", portfolioMeta: {} };
  }

  const result = await db.query("select value from app_settings where key = $1", ["branden_portfolio_names"]);
  return normalizeBrandenPortfolioSettings(result.rows[0]?.value);
}

export async function saveBrandenPortfolioNames(names: string[]) {
  const current = await getBrandenPortfolioSettings();
  const settings = await saveBrandenPortfolioSettings({
    portfolios: names,
    defaultPortfolio: current.defaultPortfolio,
    portfolioMeta: current.portfolioMeta
  });
  return settings.portfolios;
}

export async function saveBrandenPortfolioSettings(settings: BrandenPortfolioSettings) {
  const normalized = normalizeBrandenPortfolioSettings(settings);

  if (!getPool()) {
    const localSettings = await readLocalSettings();
    localSettings.brandenPortfolioNames = normalized;
    await writeLocalSettings(localSettings);
    return normalized;
  }

  await ensureSettingsTable();
  const db = getPool();

  if (!db) {
    return normalized;
  }

  const result = await db.query(
    `
      insert into app_settings (key, value)
      values ($1, $2::jsonb)
      on conflict (key) do update set value = excluded.value, updated_at = now()
      returning value;
    `,
    ["branden_portfolio_names", JSON.stringify(normalized)]
  );

  return normalizeBrandenPortfolioSettings(result.rows[0]?.value);
}

export async function saveBrandenPortfolioMeta(
  portfolio: string,
  meta: {
    currentEquity?: number;
    statementEquity?: number;
    floatingPnl?: number;
    equitySource?: string;
    equityStatementDate?: string;
    workingOrders?: NonNullable<BrandenPortfolioSettings["portfolioMeta"]>[string]["workingOrders"];
  }
) {
  const normalizedPortfolio = portfolio.trim();
  const current = await getBrandenPortfolioSettings();

  if (!normalizedPortfolio) {
    return current;
  }

  return saveBrandenPortfolioSettings(portfolioSettingsWithImportMeta(current, normalizedPortfolio, meta));
}

function portfolioSettingsWithImportMeta(
  current: BrandenPortfolioSettings,
  normalizedPortfolio: string,
  meta: {
    currentEquity?: number;
    statementEquity?: number;
    floatingPnl?: number;
    equitySource?: string;
    equityStatementDate?: string;
    workingOrders?: CfWorkingOrderMetadata[];
  },
  importedAt = new Date().toISOString()
) {
  const nextPortfolios = Array.from(new Set([...current.portfolios, normalizedPortfolio])).sort((a, b) => a.localeCompare(b));
  return normalizeBrandenPortfolioSettings({
    ...current,
    portfolios: nextPortfolios,
    portfolioMeta: {
      ...(current.portfolioMeta || {}),
      [normalizedPortfolio]: {
        ...((current.portfolioMeta || {})[normalizedPortfolio] || {}),
        currentEquity: meta.currentEquity,
        statementEquity: meta.statementEquity,
        floatingPnl: meta.floatingPnl,
        equitySource: meta.equitySource || "CF Import",
        equityStatementDate: meta.equityStatementDate || "",
        workingOrders: meta.workingOrders || [],
        equityUpdatedAt: importedAt
      }
    }
  });
}

export async function getBrandenColumnPreferences() {
  const db = getPool();

  if (!db) {
    const settings = await readLocalSettings();
    return settings.brandenColumnPreferences && typeof settings.brandenColumnPreferences === "object"
      ? (settings.brandenColumnPreferences as Record<string, unknown>)
      : {};
  }

  await ensureSettingsTable();
  const result = await db.query("select value from app_settings where key = $1", ["branden_column_preferences"]);
  return result.rows[0]?.value && typeof result.rows[0].value === "object"
    ? (result.rows[0].value as Record<string, unknown>)
    : {};
}

export async function saveBrandenColumnPreferences(preferences: Record<string, unknown>) {
  const normalized = preferences && typeof preferences === "object" ? preferences : {};
  const db = getPool();

  if (!db) {
    const settings = await readLocalSettings();
    settings.brandenColumnPreferences = normalized;
    await writeLocalSettings(settings);
    return normalized;
  }

  await ensureSettingsTable();
  const result = await db.query(
    `
      insert into app_settings (key, value)
      values ($1, $2::jsonb)
      on conflict (key) do update set value = excluded.value, updated_at = now()
      returning value;
    `,
    ["branden_column_preferences", JSON.stringify(normalized)]
  );

  return result.rows[0]?.value && typeof result.rows[0].value === "object"
    ? (result.rows[0].value as Record<string, unknown>)
    : normalized;
}

export async function getSetupChecklistTemplates() {
  const db = getPool();

  if (!db) {
    const settings = await readLocalSettings();
    return normalizeSetupTemplates(settings.setupChecklistTemplates);
  }

  await ensureSettingsTable();
  const result = await db.query("select value from app_settings where key = $1", ["setup_checklist_templates"]);
  const migration = await migrateSetupTemplateEmbeddedScreenshots(normalizeSetupTemplates(result.rows[0]?.value));
  if (migration.changed) {
    await db.query(
      `
        insert into app_settings (key, value)
        values ($1, $2::jsonb)
        on conflict (key) do update set value = excluded.value, updated_at = now()
      `,
      ["setup_checklist_templates", JSON.stringify(migration.templates)]
    );
  }
  return migration.templates;
}

function normalizeFeedbackTickets(value: unknown): FeedbackTicket[] {
  const tickets = Array.isArray(value) ? value : [];

  return tickets
    .map((ticket, index) => {
      if (!ticket || typeof ticket !== "object") {
        return null;
      }

      const rawTicket = ticket as Record<string, unknown>;
      const kind = String(rawTicket.kind || "BUG").toUpperCase() === "FEATURE" ? "FEATURE" : "BUG";
      const statusValue = String(rawTicket.status || "OPEN").toUpperCase();
      const status = statusValue === "COMPLETED" || statusValue === "IN_PROGRESS" ? statusValue : "OPEN";
      const title = String(rawTicket.title || "").trim();

      if (!title) {
        return null;
      }

      const normalizedMessages = normalizeFeedbackMessages(rawTicket.messages);
      const legacyBody = [
        String(rawTicket.summary || "").trim(),
        String(rawTicket.details || "").trim(),
        String(rawTicket.expectedBehavior || "").trim()
          ? `Expected: ${String(rawTicket.expectedBehavior || "").trim()}`
          : "",
        String(rawTicket.reproductionSteps || "").trim()
          ? `Reproduction: ${String(rawTicket.reproductionSteps || "").trim()}`
          : "",
        String(rawTicket.businessValue || "").trim()
          ? `Why it matters: ${String(rawTicket.businessValue || "").trim()}`
          : ""
      ]
        .filter(Boolean)
        .join("\n\n");
      const messages =
        normalizedMessages.length > 0
          ? normalizedMessages
          : legacyBody
            ? [
                {
                  id: `message-${index}`,
                  author: "CAM" as FeedbackMessageAuthor,
                  body: legacyBody,
                  screenshots: stringArray(rawTicket.screenshots),
                  createdAt: String(rawTicket.createdAt || new Date().toISOString())
                }
              ]
            : [];

      return {
        id: String(rawTicket.id || `ticket-${index}`),
        kind,
        status,
        title,
        summary: String(rawTicket.summary || "").trim(),
        details: String(rawTicket.details || "").trim(),
        expectedBehavior: String(rawTicket.expectedBehavior || "").trim(),
        reproductionSteps: String(rawTicket.reproductionSteps || "").trim(),
        businessValue: String(rawTicket.businessValue || "").trim(),
        screenshots: stringArray(rawTicket.screenshots),
        submittedBy: String(rawTicket.submittedBy || "").trim(),
        source: String(rawTicket.source || "").trim(),
        messages,
        resolutionNotes: String(rawTicket.resolutionNotes || "").trim(),
        createdAt: String(rawTicket.createdAt || new Date().toISOString()),
        updatedAt: String(rawTicket.updatedAt || rawTicket.createdAt || new Date().toISOString()),
        completedAt: String(rawTicket.completedAt || "")
      } satisfies FeedbackTicket;
    })
    .filter((ticket): ticket is FeedbackTicket => Boolean(ticket))
    .sort((a, b) => b.createdAt.localeCompare(a.createdAt));
}

function normalizeMarketCycleEntries(value: unknown, userId: string): MarketCycleEntry[] {
  const entries = Array.isArray(value) ? value : [];

  return entries
    .map((entry, index) => {
      if (!entry || typeof entry !== "object") {
        return null;
      }

      const rawEntry = entry as Record<string, unknown>;
      const date = String(rawEntry.date || "");

      if (!/^\d{4}-\d{2}-\d{2}$/.test(date)) {
        return null;
      }

      return {
        id: String(rawEntry.id || `market-cycle-${date}-${index}`),
        userId: String(rawEntry.userId || userId),
        date,
        trendDay: Number(rawEntry.trendDay || 0),
        phase: String(rawEntry.phase || ""),
        notes: String(rawEntry.notes || ""),
        createdAt: String(rawEntry.createdAt || new Date().toISOString()),
        updatedAt: String(rawEntry.updatedAt || new Date().toISOString())
      };
    })
    .filter(Boolean)
    .sort((a, b) => String(a!.date).localeCompare(String(b!.date))) as MarketCycleEntry[];
}

export async function getMarketCycleEntries(userId: string) {
  const key = `market_cycle_entries_${userId}`;
  const db = getPool();

  if (!db) {
    const settings = await readLocalSettings();
    return normalizeMarketCycleEntries(settings[key], userId);
  }

  await ensureSettingsTable();
  const result = await db.query("select value from app_settings where key = $1", [key]);
  return normalizeMarketCycleEntries(result.rows[0]?.value, userId);
}

export async function saveMarketCycleEntry(input: MarketCycleEntryInput) {
  const key = `market_cycle_entries_${input.userId}`;
  const entries = await getMarketCycleEntries(input.userId);
  const now = new Date().toISOString();
  const existing = entries.find((entry) => entry.date === input.date);
  const nextEntry: MarketCycleEntry = {
    id: existing?.id || `market-cycle-${input.userId}-${input.date}`,
    userId: input.userId,
    date: input.date,
    trendDay: Number(input.trendDay || 0),
    phase: input.phase || "",
    notes: input.notes || "",
    createdAt: existing?.createdAt || now,
    updatedAt: now
  };
  const normalized = [...entries.filter((entry) => entry.date !== input.date), nextEntry].sort((a, b) => a.date.localeCompare(b.date));
  const db = getPool();

  if (!db) {
    if (process.env.NODE_ENV === "production") {
      throw new Error("DATABASE_URL is required in production before market cycle entries can be saved.");
    }

    const settings = await readLocalSettings();
    settings[key] = normalized;
    await writeLocalSettings(settings);
    return normalized;
  }

  await ensureSettingsTable();
  const result = await db.query(
    `
      insert into app_settings (key, value)
      values ($1, $2::jsonb)
      on conflict (key) do update set value = excluded.value, updated_at = now()
      returning value;
    `,
    [key, JSON.stringify(normalized)]
  );

  return normalizeMarketCycleEntries(result.rows[0]?.value, input.userId);
}

export async function getWeeklyProcessFocus(userId: string): Promise<WeeklyFocus> {
  const key = `weekly_process_focus_${userId}`;
  const db = getPool();
  if (!db) {
    const settings = await readLocalSettings();
    return normalizeWeeklyFocus(settings[key]);
  }
  await ensureSettingsTable();
  const result = await db.query("select value from app_settings where key = $1", [key]);
  return normalizeWeeklyFocus(result.rows[0]?.value);
}

export async function saveWeeklyProcessFocus(
  userId: string,
  input: { summary?: unknown; focusItems?: unknown },
  now = new Date()
): Promise<WeeklyFocus> {
  const key = `weekly_process_focus_${userId}`;
  const focus = createUserDefinedWeeklyFocus(input, now);
  const db = getPool();
  if (!db) {
    if (process.env.NODE_ENV === "production") throw new Error("DATABASE_URL is required in production before weekly focus can be saved.");
    const settings = await readLocalSettings();
    settings[key] = focus;
    await writeLocalSettings(settings);
    return focus;
  }
  await ensureSettingsTable();
  const result = await db.query(
    `insert into app_settings (key, value) values ($1, $2::jsonb)
     on conflict (key) do update set value = excluded.value, updated_at = now()
     returning value`,
    [key, JSON.stringify(focus)]
  );
  return normalizeWeeklyFocus(result.rows[0]?.value);
}

function normalizeFeedbackMessages(value: unknown): FeedbackTicketMessage[] {
  const messages = Array.isArray(value) ? value : [];

  return messages
    .map((message, index) => {
      if (!message || typeof message !== "object") {
        return null;
      }

      const rawMessage = message as Record<string, unknown>;
      const body = String(rawMessage.body || "").trim();

      if (!body) {
        return null;
      }

      return {
        id: String(rawMessage.id || `message-${index}`),
        author: String(rawMessage.author || "").toUpperCase() === "ADMIN" ? "ADMIN" : "CAM",
        body,
        screenshots: stringArray(rawMessage.screenshots),
        createdAt: String(rawMessage.createdAt || new Date().toISOString())
      } satisfies FeedbackTicketMessage;
    })
    .filter((message): message is FeedbackTicketMessage => Boolean(message))
    .sort((a, b) => a.createdAt.localeCompare(b.createdAt));
}

export async function saveSetupChecklistTemplates(templates: SetupChecklistTemplate[]) {
  const normalized = normalizeSetupTemplates(templates);
  const db = getPool();

  if (!db) {
    if (process.env.NODE_ENV === "production") {
      throw new Error("DATABASE_URL is required in production before settings can be saved.");
    }

    const settings = await readLocalSettings();
    settings.setupChecklistTemplates = normalized;
    await writeLocalSettings(settings);
    return normalized;
  }

  await ensureSettingsTable();
  const migration = await migrateSetupTemplateEmbeddedScreenshots(normalized);
  const result = await db.query(
    `
      insert into app_settings (key, value)
      values ($1, $2::jsonb)
      on conflict (key) do update set value = excluded.value, updated_at = now()
      returning value;
    `,
    ["setup_checklist_templates", JSON.stringify(migration.templates)]
  );
  return normalizeSetupTemplates(result.rows[0]?.value);
}

async function migrateSetupTemplateEmbeddedScreenshots(templates: SetupChecklistTemplate[]) {
  const db = getPool();
  if (!db) return { templates, changed: false };

  let changed = false;
  const migrated: SetupChecklistTemplate[] = [];
  for (const template of templates) {
    const strategyExamples = [];
    for (const example of template.strategyExamples || []) {
      const screenshots: string[] = [];
      for (let index = 0; index < (example.screenshots || []).length; index += 1) {
        const screenshot = example.screenshots[index];
        const image = imageBufferFromDataUrl(screenshot);
        if (!image) {
          screenshots.push(screenshot);
          continue;
        }
        await ensureCamJournalStorageTables();
        const id = crypto.randomUUID();
        await db.query(
          `insert into cam_journal_screenshots
            (id, entity_type, entity_id, file_name, mime_type, image_data)
           values ($1, $2, $3, $4, $5, $6)`,
          [
            id,
            "setup-strategy-example",
            example.id,
            `${example.symbol || "strategy-example"}-${index + 1}`,
            image.mimeType,
            image.data
          ]
        );
        screenshots.push(`/api/cam-journal/screenshots/${encodeURIComponent(id)}`);
        changed = true;
      }
      strategyExamples.push({ ...example, screenshots });
    }
    migrated.push({ ...template, strategyExamples });
  }

  return {
    templates: changed ? normalizeSetupTemplates(migrated) : templates,
    changed
  };
}

function normalizeWatchlistChecklistItems(value: unknown): TradeChecklistItem[] {
  const items = Array.isArray(value) ? value : [];
  return items
    .map((item, index) => {
      if (!item || typeof item !== "object") return null;
      const raw = item as Record<string, unknown>;
      const criteria = String(raw.criteria || "").trim();
      const points = Number(raw.points || 0);
      if (!criteria || !Number.isFinite(points) || points <= 0) return null;
      const inputType: ChecklistInputType = raw.inputType === "points" ? "points" : "boolean";
      return {
        id: String(raw.id || `watchlist-criterion-${index}`),
        criteria,
        points,
        met: Boolean(raw.met),
        score: Math.max(0, Math.min(points, Number(raw.score || 0))),
        inputType,
        groupName: String(raw.groupName || "Checklist"),
        importTagKey: String(raw.importTagKey || ""),
        importTagValue: String(raw.importTagValue || "")
      } satisfies TradeChecklistItem;
    })
    .filter(Boolean) as TradeChecklistItem[];
}

function normalizeWatchlistItems(value: unknown, now = new Date().toISOString()): WatchlistItem[] {
  const items = Array.isArray(value) ? value : [];
  return items
    .map((item, index) => {
      if (!item || typeof item !== "object") return null;
      const raw = item as Record<string, unknown>;
      const symbol = String(raw.symbol || "").trim().replace(/^#/, "").toUpperCase();
      if (!symbol) return null;
      return {
        id: String(raw.id || `watchlist-item-${index}-${crypto.randomUUID()}`),
        symbol,
        side: raw.side === "SHORT" ? "SHORT" : "LONG",
        setupTag: String(raw.setupTag || "").trim(),
        setupGrade: String(raw.setupGrade || ""),
        checklistItems: normalizeWatchlistChecklistItems(raw.checklistItems),
        plannedEntry: numberValue(raw.plannedEntry),
        stopPrice: numberValue(raw.stopPrice),
        takeProfitPrice: numberValue(raw.takeProfitPrice),
        entryCriteria: String(raw.entryCriteria || ""),
        entryNotes: String(raw.entryNotes || ""),
        invalidation: String(raw.invalidation || ""),
        notes: String(raw.notes || ""),
        screenshots: stringArray(raw.screenshots),
        chartLinks: stringArray(raw.chartLinks),
        aiReview: raw.aiReview && typeof raw.aiReview === "object" ? (raw.aiReview as Record<string, unknown>) : undefined,
        createdAt: String(raw.createdAt || now),
        updatedAt: now
      } satisfies WatchlistItem;
    })
    .filter(Boolean) as WatchlistItem[];
}

function imageBufferFromDataUrl(value: string) {
  const match = String(value || "").match(/^data:([^;]+);base64,(.+)$/);
  if (!match?.[1] || !match[2]) return null;
  try {
    return {
      mimeType: match[1],
      data: Buffer.from(match[2], "base64")
    };
  } catch {
    return null;
  }
}

function normalizeWeeklyWatchlists(value: unknown, userId: string): WeeklyWatchlist[] {
  const watchlists = Array.isArray(value) ? value : [];
  const normalized = watchlists
    .map((watchlist) => {
      if (!watchlist || typeof watchlist !== "object") return null;
      const raw = watchlist as Record<string, unknown>;
      const weekNumber = Number(raw.weekNumber || 0);
      const year = Number(raw.year || 0);
      const weekKey = String(raw.weekKey || "");
      if (!weekKey || !Number.isInteger(weekNumber) || !Number.isInteger(year)) return null;
      const now = new Date().toISOString();
      return {
        id: String(raw.id || `${userId}-${weekKey}`),
        userId,
        weekKey,
        year,
        weekNumber,
        startDate: String(raw.startDate || ""),
        endDate: String(raw.endDate || ""),
        title: String(raw.title || `W${weekNumber} Watchlist`),
        items: normalizeWatchlistItems(raw.items, now),
        createdAt: String(raw.createdAt || now),
        updatedAt: String(raw.updatedAt || now)
      } satisfies WeeklyWatchlist;
    })
    .filter(Boolean) as WeeklyWatchlist[];
  return normalized.sort((a, b) => b.weekKey.localeCompare(a.weekKey));
}

async function migrateWeeklyWatchlistEmbeddedScreenshots(userId: string, watchlists: WeeklyWatchlist[]) {
  const db = getPool();
  if (!db) return watchlists;

  let changed = false;
  const migrated: WeeklyWatchlist[] = [];
  for (const watchlist of watchlists) {
    const items: WatchlistItem[] = [];
    for (const item of watchlist.items) {
      const screenshots: string[] = [];
      for (let index = 0; index < item.screenshots.length; index += 1) {
        const screenshot = item.screenshots[index];
        const image = imageBufferFromDataUrl(screenshot);
        if (!image) {
          screenshots.push(screenshot);
          continue;
        }
        await ensureCamJournalStorageTables();
        const id = crypto.randomUUID();
        await db.query(
          `insert into cam_journal_screenshots
            (id, entity_type, entity_id, file_name, mime_type, image_data)
           values ($1, $2, $3, $4, $5, $6)`,
          [
            id,
            "watchlist-item",
            item.id,
            `${item.symbol || "watchlist"}-${index + 1}`,
            image.mimeType,
            image.data
          ]
        );
        screenshots.push(`/api/cam-journal/screenshots/${encodeURIComponent(id)}`);
        changed = true;
      }
      items.push({ ...item, screenshots });
    }
    migrated.push({ ...watchlist, items });
  }

  if (!changed) return watchlists;
  const key = `weekly_watchlists_${userId}`;
  const normalized = normalizeWeeklyWatchlists(migrated, userId);
  await ensureSettingsTable();
  await db.query(
    `
      insert into app_settings (key, value)
      values ($1, $2::jsonb)
      on conflict (key) do update set value = excluded.value, updated_at = now()
    `,
    [key, JSON.stringify(normalized)]
  );
  return normalized;
}

export async function getWeeklyWatchlists(userId: string) {
  const key = `weekly_watchlists_${userId}`;
  const db = getPool();
  if (!db) {
    const settings = await readLocalSettings();
    return normalizeWeeklyWatchlists(settings[key], userId);
  }
  await ensureSettingsTable();
  const result = await db.query("select value from app_settings where key = $1", [key]);
  return migrateWeeklyWatchlistEmbeddedScreenshots(userId, normalizeWeeklyWatchlists(result.rows[0]?.value, userId));
}

export async function saveWeeklyWatchlists(userId: string, watchlists: WeeklyWatchlist[]) {
  const key = `weekly_watchlists_${userId}`;
  const normalized = normalizeWeeklyWatchlists(watchlists, userId);
  const db = getPool();
  if (!db) {
    if (process.env.NODE_ENV === "production") {
      throw new Error("DATABASE_URL is required in production before watchlists can be saved.");
    }
    const settings = await readLocalSettings();
    settings[key] = normalized;
    await writeLocalSettings(settings);
    return normalized;
  }
  await ensureSettingsTable();
  const result = await db.query(
    `
      insert into app_settings (key, value)
      values ($1, $2::jsonb)
      on conflict (key) do update set value = excluded.value, updated_at = now()
      returning value;
    `,
    [key, JSON.stringify(normalized)]
  );
  return normalizeWeeklyWatchlists(result.rows[0]?.value, userId);
}

type CamScreenshotReference = {
  id: string;
  name: string;
  dataUrl: string;
  url: string;
  addedAt: string;
};

function embeddedScreenshotData(shot: unknown) {
  const source =
    typeof shot === "string"
      ? shot
      : shot && typeof shot === "object"
        ? String((shot as Record<string, unknown>).dataUrl || (shot as Record<string, unknown>).url || "")
        : "";
  const match = source.match(/^data:(image\/[a-zA-Z0-9.+-]+);base64,([A-Za-z0-9+/=\s]+)$/);
  if (!match) return null;
  return {
    mimeType: match[1],
    imageData: Buffer.from(match[2].replace(/\s/g, ""), "base64")
  };
}

async function migrateCamJournalEmbeddedScreenshots(state: Record<string, unknown>) {
  const db = getPool();
  if (!db) return state;

  const next = structuredClone(state);
  const pending: Array<{
    owner: unknown[];
    index: number;
    entityType: string;
    entityId: string;
    shot: unknown;
    embedded: { mimeType: string; imageData: Buffer };
  }> = [];

  const collect = (shots: unknown, entityType: string, entityId: string) => {
    if (!Array.isArray(shots)) return;
    shots.forEach((shot, index) => {
      const embedded = embeddedScreenshotData(shot);
      if (embedded) pending.push({ owner: shots, index, entityType, entityId, shot, embedded });
    });
  };

  const trades = Array.isArray(next.trades) ? next.trades : [];
  for (const value of trades) {
    if (!value || typeof value !== "object") continue;
    const trade = value as Record<string, unknown>;
    collect(trade.screenshots, "trade", String(trade.id || crypto.randomUUID()));
  }

  const setups = Array.isArray(next.setups) ? next.setups : [];
  for (const value of setups) {
    if (!value || typeof value !== "object") continue;
    const setup = value as Record<string, unknown>;
    const setupId = String(setup.id || crypto.randomUUID());
    collect(setup.screenshots, "setup", setupId);
    if (!Array.isArray(setup.versions)) continue;
    for (const versionValue of setup.versions) {
      if (!versionValue || typeof versionValue !== "object") continue;
      const version = versionValue as Record<string, unknown>;
      collect(version.screenshots, "setup-version", `${setupId}:${String(version.version || "")}`);
    }
  }

  if (!pending.length) return state;

  await ensureCamJournalStorageTables();
  const client = await db.connect();
  try {
    await client.query("begin");
    await client.query("insert into cam_journal_revisions (state, reason) values ($1::jsonb, 'embedded-screenshot-migration')", [
      JSON.stringify(state)
    ]);

    for (const item of pending) {
      const id = crypto.randomUUID();
      const original = item.shot && typeof item.shot === "object" ? item.shot as Record<string, unknown> : {};
      const extension = item.embedded.mimeType.split("/")[1]?.replace("jpeg", "jpg") || "jpg";
      const fileName = String(original.name || `screenshot.${extension}`);
      await client.query(
        `insert into cam_journal_screenshots
          (id, entity_type, entity_id, file_name, mime_type, image_data)
         values ($1, $2, $3, $4, $5, $6)`,
        [id, item.entityType, item.entityId, fileName, item.embedded.mimeType, item.embedded.imageData]
      );
      const url = `/api/cam-journal/screenshots/${encodeURIComponent(id)}`;
      item.owner[item.index] = {
        ...original,
        id,
        name: fileName,
        dataUrl: url,
        url,
        addedAt: String(original.addedAt || new Date().toISOString())
      } satisfies CamScreenshotReference;
    }

    await client.query(
      "update app_settings set value = $1::jsonb, updated_at = now() where key = 'cam_journal_state'",
      [JSON.stringify(next)]
    );
    await client.query("commit");
    return next;
  } catch (error) {
    await client.query("rollback");
    throw error;
  } finally {
    client.release();
  }
}

export async function getCamJournalState() {
  const db = getPool();

  if (!db) {
    const settings = await readLocalSettings();
    return (settings.camJournalState && typeof settings.camJournalState === "object" ? settings.camJournalState : {}) as Record<string, unknown>;
  }

  await ensureSettingsTable();
  const result = await db.query("select value from app_settings where key = $1", ["cam_journal_state"]);
  const state = (result.rows[0]?.value && typeof result.rows[0].value === "object" ? result.rows[0].value : {}) as Record<string, unknown>;
  return migrateCamJournalEmbeddedScreenshots(state);
}

async function ensureCamJournalStorageTables() {
  const db = getPool();
  if (!db) return;

  await db.query(`
    create table if not exists cam_journal_screenshots (
      id text primary key,
      entity_type text not null,
      entity_id text not null,
      file_name text not null default '',
      mime_type text not null,
      image_data bytea not null,
      created_at timestamptz not null default now()
    );
    create index if not exists cam_journal_screenshots_entity_idx
      on cam_journal_screenshots (entity_type, entity_id);

    create table if not exists cam_journal_revisions (
      id bigserial primary key,
      state jsonb not null,
      reason text not null default 'save',
      created_at timestamptz not null default now()
    );
    create index if not exists cam_journal_revisions_created_at_idx
      on cam_journal_revisions (created_at desc);
  `);
}

export async function saveCamJournalScreenshot(
  entityType: string,
  entityId: string,
  fileName: string,
  mimeType: string,
  imageData: Buffer
) {
  const db = getPool();
  if (!db) throw new Error("DATABASE_URL is required before screenshots can be saved.");

  await ensureCamJournalStorageTables();
  const id = crypto.randomUUID();
  await db.query(
    `insert into cam_journal_screenshots
      (id, entity_type, entity_id, file_name, mime_type, image_data)
     values ($1, $2, $3, $4, $5, $6)`,
    [id, entityType, entityId, fileName, mimeType, imageData]
  );
  return { id, url: `/api/cam-journal/screenshots/${encodeURIComponent(id)}` };
}

export async function getCamJournalScreenshot(id: string) {
  const db = getPool();
  if (!db) return null;

  await ensureCamJournalStorageTables();
  const result = await db.query(
    `select id, entity_type, entity_id, file_name, mime_type, image_data
     from cam_journal_screenshots where id = $1`,
    [id]
  );
  const row = result.rows[0];
  if (!row) return null;
  return {
    id: String(row.id),
    entityType: String(row.entity_type),
    entityId: String(row.entity_id),
    fileName: String(row.file_name || ""),
    mimeType: String(row.mime_type),
    imageData: row.image_data as Buffer
  };
}

export async function deleteCamJournalScreenshot(id: string) {
  const db = getPool();
  if (!db) throw new Error("DATABASE_URL is required before screenshots can be deleted.");
  await ensureCamJournalStorageTables();
  await db.query("delete from cam_journal_screenshots where id = $1", [id]);
}

export async function snapshotCamJournalState(reason = "nightly") {
  const db = getPool();
  if (!db) throw new Error("DATABASE_URL is required before backups can be created.");

  await ensureSettingsTable();
  await ensureCamJournalStorageTables();
  const current = await db.query("select value from app_settings where key = $1", ["cam_journal_state"]);
  const state = current.rows[0]?.value;
  if (!state) return { created: false };

  await db.query("insert into cam_journal_revisions (state, reason) values ($1::jsonb, $2)", [
    JSON.stringify(state),
    reason
  ]);
  await db.query("delete from cam_journal_revisions where created_at < now() - interval '90 days'");
  return { created: true };
}

export async function saveCamJournalState(state: Record<string, unknown>) {
  const normalized = {
    trades: Array.isArray(state.trades) ? state.trades : [],
    setups: Array.isArray(state.setups) ? state.setups : [],
    portfolios: Array.isArray(state.portfolios) ? state.portfolios : [],
    tags: state.tags && typeof state.tags === "object" ? state.tags : { secondary: [], mistakes: [] },
    monthlyReviews: state.monthlyReviews && typeof state.monthlyReviews === "object" ? state.monthlyReviews : {},
    watchlistItems: Array.isArray(state.watchlistItems) ? state.watchlistItems : [],
    updatedAt: new Date().toISOString()
  };
  const db = getPool();

  if (!db) {
    if (process.env.NODE_ENV === "production") {
      throw new Error("DATABASE_URL is required in production before Cam journal state can be saved.");
    }

    const settings = await readLocalSettings();
    settings.camJournalState = normalized;
    await writeLocalSettings(settings);
    return normalized;
  }

  await ensureSettingsTable();
  await ensureCamJournalStorageTables();
  const client = await db.connect();
  try {
    await client.query("begin");
    const current = await client.query("select value from app_settings where key = $1 for update", ["cam_journal_state"]);
    if (current.rows[0]?.value) {
      await client.query("insert into cam_journal_revisions (state, reason) values ($1::jsonb, 'save')", [
        JSON.stringify(current.rows[0].value)
      ]);
    }
    const result = await client.query(
      `
      insert into app_settings (key, value)
      values ($1, $2::jsonb)
      on conflict (key) do update set value = excluded.value, updated_at = now()
      returning value;
      `,
      ["cam_journal_state", JSON.stringify(normalized)]
    );
    await client.query("commit");
    return result.rows[0]?.value || normalized;
  } catch (error) {
    await client.query("rollback");
    throw error;
  } finally {
    client.release();
  }
}

export async function listFeedbackTickets() {
  const db = getPool();

  if (!db) {
    const settings = await readLocalSettings();
    return normalizeFeedbackTickets(settings.feedbackTickets);
  }

  await ensureSettingsTable();
  const result = await db.query("select value from app_settings where key = $1", ["feedback_tickets"]);
  return normalizeFeedbackTickets(result.rows[0]?.value);
}

export async function createFeedbackTicket(ticket: Omit<FeedbackTicket, "id" | "createdAt" | "updatedAt" | "completedAt">) {
  const tickets = await listFeedbackTickets();
  const now = new Date().toISOString();
  const next: FeedbackTicket = {
    ...ticket,
    id: crypto.randomUUID(),
    createdAt: now,
    updatedAt: now,
    completedAt: ""
  };

  const saved = await saveFeedbackTickets([next, ...tickets]);
  return saved.find((item) => item.id === next.id) || next;
}

export async function updateFeedbackTicket(
  id: string,
  updates: Partial<Pick<FeedbackTicket, "status" | "resolutionNotes">>
) {
  const tickets = await listFeedbackTickets();
  const nextTickets = tickets.map((ticket) =>
    ticket.id === id
      ? {
          ...ticket,
          status: updates.status || ticket.status,
          resolutionNotes: updates.resolutionNotes ?? ticket.resolutionNotes,
          updatedAt: new Date().toISOString(),
          completedAt: updates.status === "COMPLETED" ? new Date().toISOString() : updates.status ? "" : ticket.completedAt
        }
      : ticket
  );

  return saveFeedbackTickets(nextTickets);
}

export async function appendFeedbackTicketMessage(
  id: string,
  input: { author: FeedbackMessageAuthor; body: string; screenshots?: string[] }
) {
  const tickets = await listFeedbackTickets();
  const now = new Date().toISOString();
  const nextTickets = tickets.map((ticket) => {
    if (ticket.id !== id) {
      return ticket;
    }

    const message: FeedbackTicketMessage = {
      id: crypto.randomUUID(),
      author: input.author,
      body: String(input.body || "").trim(),
      screenshots: stringArray(input.screenshots),
      createdAt: now
    };

    return {
      ...ticket,
      messages: [...ticket.messages, message],
      updatedAt: now,
      status: ticket.status === "COMPLETED" ? "IN_PROGRESS" : ticket.status,
      completedAt: ticket.status === "COMPLETED" ? "" : ticket.completedAt
    };
  });

  return saveFeedbackTickets(nextTickets);
}

async function saveFeedbackTickets(tickets: FeedbackTicket[]) {
  const normalized = normalizeFeedbackTickets(tickets);
  const db = getPool();

  if (!db) {
    if (process.env.NODE_ENV === "production") {
      throw new Error("DATABASE_URL is required in production before feedback tickets can be saved.");
    }

    const settings = await readLocalSettings();
    settings.feedbackTickets = normalized;
    await writeLocalSettings(settings);
    return normalized;
  }

  await ensureSettingsTable();
  const result = await db.query(
    `
      insert into app_settings (key, value)
      values ($1, $2::jsonb)
      on conflict (key) do update set value = excluded.value, updated_at = now()
      returning value;
    `,
    ["feedback_tickets", JSON.stringify(normalized)]
  );

  return normalizeFeedbackTickets(result.rows[0]?.value);
}

export async function listReports() {
  const db = getPool();

  if (!db) {
    return readLocalReports();
  }

  await ensureTable();
  const result = await db.query(`select ${columns} from monthly_reports order by month asc, user_id asc`);
  return result.rows.map(rowToReport);
}

export async function upsertReport(input: MonthlyReportInput) {
  const now = new Date().toISOString();
  const id = `${input.userId}-${input.month}`;
  const db = getPool();

  if (!db) {
    if (process.env.NODE_ENV === "production") {
      throw new Error("DATABASE_URL is required in production before reports can be saved.");
    }

    const reports = await readLocalReports();
    const existing = reports.find((report) => report.id === id);
    const next: MonthlyReport = {
      ...input,
      id,
      createdAt: existing?.createdAt || now,
      updatedAt: now
    };
    const filtered = reports.filter((report) => report.id !== id);
    await writeLocalReports([...filtered, next].sort((a, b) => a.month.localeCompare(b.month)));
    return next;
  }

  await ensureTable();
  const values = [
    id,
    input.userId,
    input.month,
    input.accountSize,
    input.totalReturn,
    input.percentReturn,
    input.netPnl,
    input.totalPayouts,
    input.totalTrades,
    input.winRate,
    input.avgR,
    input.totalR,
    input.avgWinR,
    input.avgLossR,
    input.avgWin,
    input.avgLoss,
    input.avgRisk,
    input.currentRiskPercent,
    input.expectedValueR,
    input.sharpeRatio,
    input.avgTradeLength,
    input.avgSwingLength,
    input.longestWinStreak,
    input.longestLossStreak,
    input.notes
  ];

  const result = await db.query(
    `
      insert into monthly_reports (
        id, user_id, month, account_size, total_return, percent_return, net_pnl,
        total_payouts, total_trades, win_rate, avg_r, total_r, avg_win_r, avg_loss_r,
        avg_win, avg_loss, avg_risk, current_risk_percent, expected_value_r, sharpe_ratio,
        avg_trade_length, avg_swing_length, longest_win_streak, longest_loss_streak, notes
      )
      values (
        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13,
        $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25
      )
      on conflict (user_id, month) do update set
        account_size = excluded.account_size,
        total_return = excluded.total_return,
        percent_return = excluded.percent_return,
        net_pnl = excluded.net_pnl,
        total_payouts = excluded.total_payouts,
        total_trades = excluded.total_trades,
        win_rate = excluded.win_rate,
        avg_r = excluded.avg_r,
        total_r = excluded.total_r,
        avg_win_r = excluded.avg_win_r,
        avg_loss_r = excluded.avg_loss_r,
        avg_win = excluded.avg_win,
        avg_loss = excluded.avg_loss,
        avg_risk = excluded.avg_risk,
        current_risk_percent = excluded.current_risk_percent,
        expected_value_r = excluded.expected_value_r,
        sharpe_ratio = excluded.sharpe_ratio,
        avg_trade_length = excluded.avg_trade_length,
        avg_swing_length = excluded.avg_swing_length,
        longest_win_streak = excluded.longest_win_streak,
        longest_loss_streak = excluded.longest_loss_streak,
        notes = excluded.notes,
        updated_at = now()
      returning ${columns};
    `,
    values
  );

  return rowToReport(result.rows[0]);
}

export async function deleteReport(id: string, userId: string) {
  const db = getPool();

  if (!db) {
    if (process.env.NODE_ENV === "production") {
      throw new Error("DATABASE_URL is required in production before reports can be deleted.");
    }

    const reports = await readLocalReports();
    await writeLocalReports(reports.filter((report) => !(report.id === id && report.userId === userId)));
    return;
  }

  await ensureTable();
  await db.query("delete from monthly_reports where id = $1 and user_id = $2", [id, userId]);
}

export async function listTrades() {
  const db = getPool();

  if (!db) {
    return (await readLocalTrades()).sort(tradeLogOrder);
  }

  await ensureTradeTable();
  const result = await db.query(`select ${tradeColumns} from trade_logs order by entry_date desc, open_time desc, created_at desc, id desc`);
  return result.rows.map(rowToTrade);
}

export async function listBrandenVisibleTrades() {
  const db = getPool();

  if (!db) {
    return (await readLocalTrades())
      .filter((trade) => trade.userId === "branden" && !trade.hidden)
      .sort(tradeLogOrder);
  }

  await ensureTradeTable();
  const result = await db.query(
    `select ${tradeColumns} from trade_logs where user_id = $1 and hidden = false order by entry_date desc, open_time desc, created_at desc, id desc`,
    ["branden"]
  );
  return result.rows.map(rowToTrade);
}

const openPositionTradeColumns = [
  "id",
  "user_id",
  "symbol",
  "side",
  "status",
  "entry_date",
  "exit_date",
  "open_time",
  "close_time",
  "avg_entry",
  "exit_price",
  "stop_price",
  "take_profit_price",
  "shares",
  "commission",
  "used_margin",
  "risk",
  "pnl",
  "r_multiple",
  "return_percent",
  "days_in_trade",
  "portfolio_tag",
  "hidden",
  "created_at",
  "updated_at"
].join(", ");

function rowToOpenPositionTrade(row: Record<string, unknown>): TradeLogEntry {
  return {
    ...rowToTrade({
      ...row,
      import_source: "",
      import_row_key: "",
      setup_tags: [],
      mistake_tags: [],
      custom_tags: [],
      manual_grade: "",
      emotion: "",
      trade_quality: "",
      checklist_items: [],
      notes: "",
      screenshots: [],
      chart_links: [],
      executions: [],
      group_id: "",
      group_role: "none"
    })
  };
}

export async function listBrandenOpenPositionTrades() {
  const db = getPool();

  if (!db) {
    return (await readLocalTrades())
      .filter((trade) => trade.userId === "branden" && !trade.hidden && trade.status === "OPEN")
      .sort((a, b) => a.symbol.localeCompare(b.symbol));
  }

  await ensureTradeTable();
  const result = await db.query(
    `select ${openPositionTradeColumns} from trade_logs where user_id = $1 and hidden = false and status = 'OPEN' order by symbol asc, entry_date desc`,
    ["branden"]
  );
  return result.rows.map(rowToOpenPositionTrade);
}

const calendarTradeColumns = [
  "id",
  "user_id",
  "symbol",
  "side",
  "status",
  "entry_date",
  "exit_date",
  "avg_entry",
  "risk",
  "pnl",
  "r_multiple",
  "days_in_trade",
  "portfolio_tag",
  "custom_tags",
  "executions",
  "notes",
  "hidden",
  "created_at",
  "updated_at"
].join(", ");

function rowToCalendarTrade(row: Record<string, unknown>): TradeLogEntry {
  return rowToTrade({
    ...row,
    import_source: "",
    import_row_key: "",
    open_time: "",
    close_time: "",
    exit_price: 0,
    stop_price: 0,
    take_profit_price: 0,
    shares: 0,
    commission: 0,
    used_margin: 0,
    return_percent: 0,
    setup_tags: [],
    mistake_tags: [],
    manual_grade: "",
    emotion: "",
    trade_quality: "",
    checklist_items: [],
    screenshots: [],
    chart_links: [],
    group_id: "",
    group_role: "none"
  });
}

export async function listBrandenCalendarTrades() {
  const db = getPool();

  if (!db) {
    return (await readLocalTrades())
      .filter((trade) => trade.userId === "branden" && !trade.hidden)
      .sort((a, b) => (b.exitDate || b.entryDate).localeCompare(a.exitDate || a.entryDate));
  }

  await ensureTradeTable();
  const result = await db.query(
    `select ${calendarTradeColumns} from trade_logs where user_id = $1 and hidden = false order by entry_date desc, created_at desc`,
    ["branden"]
  );
  return result.rows.map(rowToCalendarTrade);
}

const dailyReviewTradeColumns = [
  "id",
  "user_id",
  "symbol",
  "side",
  "status",
  "entry_date",
  "exit_date",
  "open_time",
  "close_time",
  "avg_entry",
  "exit_price",
  "stop_price",
  "take_profit_price",
  "shares",
  "commission",
  "risk",
  "pnl",
  "r_multiple",
  "days_in_trade",
  "portfolio_tag",
  "executions",
  "hidden",
  "created_at",
  "updated_at"
].join(", ");

function rowToDailyReviewTrade(row: Record<string, unknown>): TradeLogEntry {
  return rowToTrade({
    ...row,
    import_source: "",
    import_row_key: "",
    used_margin: 0,
    return_percent: 0,
    setup_tags: [],
    mistake_tags: [],
    custom_tags: [],
    manual_grade: "",
    emotion: "",
    trade_quality: "",
    checklist_items: [],
    notes: "",
    screenshots: [],
    chart_links: [],
    group_id: "",
    group_role: "none"
  });
}

export async function listBrandenDailyReviewTrades() {
  const db = getPool();

  if (!db) {
    return (await readLocalTrades())
      .filter((trade) => trade.userId === "branden" && !trade.hidden)
      .sort((a, b) => (b.entryDate || "").localeCompare(a.entryDate || ""));
  }

  await ensureTradeTable();
  const result = await db.query(
    `select ${dailyReviewTradeColumns} from trade_logs where user_id = $1 and hidden = false order by entry_date desc, created_at desc`,
    ["branden"]
  );
  return result.rows.map(rowToDailyReviewTrade);
}

const exitAnalysisTradeColumns = [
  "id",
  "user_id",
  "symbol",
  "side",
  "status",
  "entry_date",
  "exit_date",
  "avg_entry",
  "exit_price",
  "stop_price",
  "shares",
  "commission",
  "risk",
  "pnl",
  "r_multiple",
  "setup_tags",
  "portfolio_tag",
  "executions",
  "hidden",
  "created_at",
  "updated_at"
].join(", ");

function rowToExitAnalysisTrade(row: Record<string, unknown>): TradeLogEntry {
  return rowToTrade({
    ...row,
    import_source: "",
    import_row_key: "",
    open_time: "",
    close_time: "",
    take_profit_price: 0,
    used_margin: 0,
    return_percent: 0,
    days_in_trade: 0,
    mistake_tags: [],
    custom_tags: [],
    manual_grade: "",
    emotion: "",
    trade_quality: "",
    checklist_items: [],
    notes: "",
    screenshots: [],
    chart_links: [],
    group_id: "",
    group_role: "none"
  });
}

export async function listBrandenExitAnalysisTrades() {
  const db = getPool();

  if (!db) {
    return (await readLocalTrades())
      .filter((trade) => trade.userId === "branden" && !trade.hidden && trade.status !== "OPEN")
      .sort((a, b) => (b.exitDate || "").localeCompare(a.exitDate || ""));
  }

  await ensureTradeTable();
  const result = await db.query(
    `select ${exitAnalysisTradeColumns} from trade_logs where user_id = $1 and hidden = false and status <> 'OPEN' order by exit_date desc, entry_date desc`,
    ["branden"]
  );
  return result.rows.map(rowToExitAnalysisTrade);
}

const benchmarkTradeColumns = [
  "id",
  "user_id",
  "symbol",
  "side",
  "status",
  "entry_date",
  "exit_date",
  "avg_entry",
  "exit_price",
  "return_percent",
  "portfolio_tag",
  "hidden",
  "created_at",
  "updated_at"
].join(", ");

function rowToBenchmarkTrade(row: Record<string, unknown>): TradeLogEntry {
  return rowToTrade({
    ...row,
    import_source: "",
    import_row_key: "",
    open_time: "",
    close_time: "",
    stop_price: 0,
    take_profit_price: 0,
    shares: 0,
    commission: 0,
    used_margin: 0,
    risk: 0,
    pnl: 0,
    r_multiple: 0,
    days_in_trade: 0,
    setup_tags: [],
    mistake_tags: [],
    custom_tags: [],
    manual_grade: "",
    emotion: "",
    trade_quality: "",
    checklist_items: [],
    notes: "",
    screenshots: [],
    chart_links: [],
    executions: [],
    group_id: "",
    group_role: "none"
  });
}

export async function listBrandenBenchmarkTrades() {
  const db = getPool();

  if (!db) {
    return (await readLocalTrades())
      .filter((trade) => trade.userId === "branden" && !trade.hidden && trade.status !== "OPEN")
      .sort((a, b) => (b.exitDate || "").localeCompare(a.exitDate || ""));
  }

  await ensureTradeTable();
  const result = await db.query(
    `select ${benchmarkTradeColumns} from trade_logs where user_id = $1 and hidden = false and status <> 'OPEN' order by exit_date desc, entry_date desc`,
    ["branden"]
  );
  return result.rows.map(rowToBenchmarkTrade);
}

export async function listCfStatementTrades(userId: string, portfolioTag: string) {
  const db = getPool();

  if (!db) {
    const trades = await readLocalTrades();
    return trades.filter(
      (trade) => trade.userId === userId && trade.portfolioTag === portfolioTag && trade.importSource === "cf-statement-pdf"
    );
  }

  await ensureTradeTable();
  const result = await db.query(
    `select ${tradeColumns} from trade_logs where user_id = $1 and portfolio_tag = $2 and import_source = 'cf-statement-pdf' order by entry_date asc, open_time asc, created_at asc`,
    [userId, portfolioTag]
  );
  return result.rows.map(rowToTrade);
}

function materializeCfStatementTrades(trades: CfStatementReplacementTrade[], portfolioTag: string, now: string) {
  return trades.map((trade): TradeLogEntry => ({
      ...trade,
      id: trade.id || stableTradeId(trade),
      importSource: "cf-statement-pdf",
      importRowKey: trade.importRowKey || "",
      chartLinks: trade.chartLinks || [],
      manualGrade: trade.manualGrade || "",
      openTime: trade.openTime || "",
      closeTime: trade.closeTime || "",
      portfolioTag: trade.portfolioTag || portfolioTag,
      takeProfitPrice: trade.takeProfitPrice || 0,
      commission: trade.commission || 0,
      usedMargin: trade.usedMargin || 0,
      emotion: trade.emotion || "",
      tradeQuality: trade.tradeQuality || "",
      executions: trade.executions || [],
      hidden: Boolean(trade.hidden),
      groupId: "",
      groupRole: "none",
      createdAt: now,
      updatedAt: now
    }));
}

async function replaceCfStatementTradesWithClient(
  client: PoolClient,
  userId: string,
  portfolioTag: string,
  trades: CfStatementReplacementTrade[],
  adoptedNonCfTradeIds: string[] = []
) {
    if (adoptedNonCfTradeIds.length) {
      await client.query(
        "delete from trade_logs where user_id = $1 and portfolio_tag = $2 and import_source <> 'cf-statement-pdf' and id = any($3::text[])",
        [userId, portfolioTag, adoptedNonCfTradeIds]
      );
    }
    await client.query(
      "delete from trade_logs where user_id = $1 and portfolio_tag = $2 and import_source = 'cf-statement-pdf'",
      [userId, portfolioTag]
    );

    for (const trade of trades) {
      const values = [
        trade.id || stableTradeId(trade),
        trade.userId,
        "cf-statement-pdf",
        trade.importRowKey || "",
        trade.symbol,
        trade.side,
        trade.status,
        trade.entryDate,
        trade.exitDate,
        trade.openTime || "",
        trade.closeTime || "",
        trade.avgEntry,
        trade.exitPrice,
        trade.stopPrice,
        trade.takeProfitPrice,
        trade.shares,
        trade.commission,
        trade.usedMargin,
        trade.risk,
        trade.pnl,
        trade.rMultiple,
        trade.returnPercent,
        trade.daysInTrade,
        JSON.stringify(trade.setupTags),
        JSON.stringify(trade.mistakeTags),
        JSON.stringify(trade.customTags),
        trade.manualGrade || "",
        trade.portfolioTag || portfolioTag,
        trade.emotion || "",
        trade.tradeQuality || "",
        JSON.stringify(trade.checklistItems),
        trade.notes,
        JSON.stringify(normalizeTradeReviewSections(trade.reviewSections)),
        JSON.stringify(trade.screenshots),
        JSON.stringify(trade.chartLinks || []),
        JSON.stringify(trade.executions || []),
        Boolean(trade.hidden),
        "",
        "none"
      ];

      await client.query(
        `
          insert into trade_logs (
            id, user_id, import_source, import_row_key, symbol, side, status, entry_date, exit_date, open_time, close_time, avg_entry, exit_price,
            stop_price, take_profit_price, shares, commission, used_margin, risk, pnl, r_multiple, return_percent, days_in_trade, setup_tags,
            mistake_tags, custom_tags, manual_grade, portfolio_tag, emotion, trade_quality, checklist_items, notes, review_sections, screenshots, chart_links, executions, hidden, group_id, group_role
          )
          values (
            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
            $11, $12, $13, $14, $15, $16, $17, $18, $19, $20,
            $21, $22, $23, $24::jsonb, $25::jsonb, $26::jsonb, $27, $28, $29, $30, $31::jsonb, $32, $33::jsonb, $34::jsonb, $35::jsonb, $36::jsonb, $37, $38, $39
          )
        `,
        values
      );
    }
}

export async function replaceCfStatementTrades(userId: string, portfolioTag: string, trades: CfStatementReplacementTrade[]) {
  const now = new Date().toISOString();
  const db = getPool();

  if (!db) {
    const current = await readLocalTrades();
    const retained = current.filter(
      (trade) => !(trade.userId === userId && trade.portfolioTag === portfolioTag && trade.importSource === "cf-statement-pdf")
    );
    const nextTrades = materializeCfStatementTrades(trades, portfolioTag, now);
    await writeLocalTrades([...retained, ...nextTrades].sort(tradeLogOrder));
    return { count: nextTrades.length };
  }

  await ensureTradeTable();
  const client = await db.connect();

  try {
    return await runAtomicCfImport(
      { begin: () => client.query("begin").then(() => undefined), commit: () => client.query("commit").then(() => undefined), rollback: () => client.query("rollback").then(() => undefined) },
      async () => {
        await replaceCfStatementTradesWithClient(client, userId, portfolioTag, trades);
        return { count: trades.length };
      }
    );
  } finally {
    client.release();
  }
}

export async function replaceCfStatementImport(
  userId: string,
  portfolioTag: string,
  trades: CfStatementReplacementTrade[],
  meta: {
    currentEquity?: number;
    statementEquity?: number;
    floatingPnl?: number;
    equitySource?: string;
    equityStatementDate?: string;
    workingOrders?: CfWorkingOrderMetadata[];
  },
  brokerSnapshot: BrokerPortfolioSnapshotInput,
  replaceTrades: boolean,
  adoptedNonCfTradeIds: string[] = []
) {
  const now = new Date().toISOString();
  const db = getPool();

  if (!db) {
    const previousTrades = await readLocalTrades();
    const previousSettings = await readLocalSettings();
    const currentPortfolioSettings = normalizeBrandenPortfolioSettings(previousSettings.brandenPortfolioNames);
    const nextPortfolioSettings = portfolioSettingsWithImportMeta(currentPortfolioSettings, portfolioTag.trim(), meta, now);
    const nextBrokerSnapshots = upsertBrokerPortfolioSnapshotCollection(
      normalizeBrokerPortfolioSnapshotList(previousSettings.brandenBrokerPortfolioSnapshots),
      brokerSnapshot,
      now
    );
    try {
      if (replaceTrades) {
        const retained = previousTrades.filter(
          (trade) =>
            !(trade.userId === userId && trade.portfolioTag === portfolioTag && trade.importSource === "cf-statement-pdf") &&
            !adoptedNonCfTradeIds.includes(trade.id)
        );
        await writeLocalTrades([...retained, ...materializeCfStatementTrades(trades, portfolioTag, now)].sort(tradeLogOrder));
      }
      await writeLocalSettings({
        ...previousSettings,
        brandenPortfolioNames: nextPortfolioSettings,
        brandenBrokerPortfolioSnapshots: nextBrokerSnapshots
      });
      return { count: trades.length, tradesReplaced: replaceTrades };
    } catch (error) {
      await writeLocalTrades(previousTrades);
      await writeLocalSettings(previousSettings);
      throw error;
    }
  }

  await ensureTradeTable();
  await ensureSettingsTable();
  await ensureBrokerPortfolioSnapshotTable();
  const client = await db.connect();
  try {
    return await runAtomicCfImport(
      { begin: () => client.query("begin").then(() => undefined), commit: () => client.query("commit").then(() => undefined), rollback: () => client.query("rollback").then(() => undefined) },
      async () => {
        const settingsResult = await client.query("select value from app_settings where key = $1 for update", ["branden_portfolio_names"]);
        const currentSettings = normalizeBrandenPortfolioSettings(settingsResult.rows[0]?.value);
        const nextSettings = portfolioSettingsWithImportMeta(currentSettings, portfolioTag.trim(), meta, now);
        if (replaceTrades) {
          await replaceCfStatementTradesWithClient(client, userId, portfolioTag, trades, adoptedNonCfTradeIds);
        }
        await upsertBrokerPortfolioSnapshotWithClient(client, brokerSnapshot, now);
        await client.query(
          `insert into app_settings (key, value) values ($1, $2::jsonb)
           on conflict (key) do update set value = excluded.value, updated_at = now()`,
          ["branden_portfolio_names", JSON.stringify(nextSettings)]
        );
        return { count: trades.length, tradesReplaced: replaceTrades };
      }
    );
  } finally {
    client.release();
  }
}

export async function upsertTrade(input: TradeLogInput) {
  const now = new Date().toISOString();
  const db = getPool();
  const hasImportKey = Boolean(input.importSource.trim() && input.importRowKey.trim());
  const isCfImport = input.importSource === "cf-statement-pdf";

  if (!db) {
    if (process.env.NODE_ENV === "production") {
      throw new Error("DATABASE_URL is required in production before trades can be saved.");
    }

    const trades = await readLocalTrades();
    let existing = hasImportKey
      ? trades.find(
          (trade) =>
            trade.userId === input.userId &&
            trade.importSource === input.importSource &&
            trade.importRowKey === input.importRowKey
        )
      : null;
    const isContinuation = Boolean(isCfImport && isCfBaselineRow(input.importRowKey) && !existing);
    if (isContinuation) {
      existing = await findExistingOpenCfTradeForContinuation(
        input.userId,
        input.portfolioTag || "",
        input.symbol,
        input.side,
        input.entryDate
      );
    }
    if (!existing && isCfImport) {
      existing = await findExistingCfTradeByExecutionOverlap(
        input.userId,
        input.portfolioTag || "",
        input.symbol,
        input.side,
        input.executions || []
      );
    }
    const preserveBrokerFields = Boolean(
      isCfImport && existing && isCfBaselineRow(input.importRowKey) && !isCfBaselineRow(existing.importRowKey) && !isContinuation
    );
    const mergedInput =
      isCfImport && existing
        ? isContinuation
          ? mergeCfContinuationIntoOpenTrade(existing, input)
          : mergeCfImportIntoExisting(existing, input, preserveBrokerFields)
        : input;
    const id = existing?.id || stableTradeId(input);
    const next: TradeLogEntry = {
      ...mergedInput,
      importSource: mergedInput.importSource || "",
      importRowKey:
        preserveBrokerFields && existing
          ? existing.importRowKey
          : mergedInput.importRowKey || existing?.importRowKey || "",
      chartLinks: mergedInput.chartLinks || [],
      manualGrade: mergedInput.manualGrade || "",
      openTime: mergedInput.openTime || "",
      closeTime: mergedInput.closeTime || "",
      portfolioTag: mergedInput.portfolioTag || "",
      takeProfitPrice: mergedInput.takeProfitPrice || 0,
      commission: mergedInput.commission || 0,
      usedMargin: mergedInput.usedMargin || 0,
      emotion: mergedInput.emotion || "",
      tradeQuality: mergedInput.tradeQuality || "",
      executions: mergedInput.executions || existing?.executions || [],
      hidden: existing?.hidden || Boolean(input.hidden),
      groupId: "",
      groupRole: "none",
      id,
      createdAt: existing?.createdAt || now,
      updatedAt: now
    };
    const filtered = trades.filter((trade) => trade.id !== id);
    await writeLocalTrades([next, ...filtered].sort(tradeLogOrder));
    return { trade: next, mode: existing ? "updated" : "created" } as const;
  }

  await ensureTradeTable();
  let id = stableTradeId(input);
  let existingImportedTradeId = "";
  let existingImportedTrade: TradeLogEntry | null = null;
  let isContinuation = false;

  if (hasImportKey) {
    const existing = await db.query(
      `select ${tradeColumns} from trade_logs where user_id = $1 and import_source = $2 and import_row_key = $3 and portfolio_tag = $4 order by created_at asc limit 1`,
      [input.userId, input.importSource, input.importRowKey, input.portfolioTag || ""]
    );

    if (existing.rows[0]) {
      existingImportedTrade = rowToTrade(existing.rows[0]);
      existingImportedTradeId = String(existingImportedTrade.id);
      id = existingImportedTradeId;
    }
  }

  if (!existingImportedTrade && isCfImport && isCfBaselineRow(input.importRowKey)) {
    existingImportedTrade = await findExistingOpenCfTradeForContinuation(
      input.userId,
      input.portfolioTag || "",
      input.symbol,
      input.side,
      input.entryDate
    );

    if (existingImportedTrade) {
      isContinuation = true;
      existingImportedTradeId = existingImportedTrade.id;
      id = existingImportedTrade.id;
    }
  }

  if (!existingImportedTrade && isCfImport) {
    existingImportedTrade = await findExistingCfTradeByExecutionOverlap(
      input.userId,
      input.portfolioTag || "",
      input.symbol,
      input.side,
      input.executions || []
    );

    if (existingImportedTrade) {
      existingImportedTradeId = existingImportedTrade.id;
      id = existingImportedTrade.id;
    }
  }

  if (!existingImportedTradeId) {
    id = stableTradeId(input);
  }

  const preserveBrokerFields = Boolean(
    isCfImport &&
      existingImportedTrade &&
      isCfBaselineRow(input.importRowKey) &&
      !isCfBaselineRow(existingImportedTrade.importRowKey) &&
      !isContinuation
  );
  const mergedInput =
    isCfImport && existingImportedTrade
      ? isContinuation
        ? mergeCfContinuationIntoOpenTrade(existingImportedTrade, input)
        : mergeCfImportIntoExisting(existingImportedTrade, input, preserveBrokerFields)
      : input;

  const values = [
    id,
    mergedInput.userId,
    mergedInput.importSource || "",
    preserveBrokerFields && existingImportedTrade
      ? existingImportedTrade.importRowKey
      : mergedInput.importRowKey || existingImportedTrade?.importRowKey || "",
    mergedInput.symbol,
    mergedInput.side,
    mergedInput.status,
    mergedInput.entryDate,
    mergedInput.exitDate,
    mergedInput.openTime || "",
    mergedInput.closeTime || "",
    mergedInput.avgEntry,
    mergedInput.exitPrice,
    mergedInput.stopPrice,
    mergedInput.takeProfitPrice,
    mergedInput.shares,
    mergedInput.commission,
    mergedInput.usedMargin,
    mergedInput.risk,
    mergedInput.pnl,
    mergedInput.rMultiple,
    mergedInput.returnPercent,
    mergedInput.daysInTrade,
    JSON.stringify(mergedInput.setupTags),
    JSON.stringify(mergedInput.mistakeTags),
    JSON.stringify(mergedInput.customTags),
    mergedInput.manualGrade || "",
    mergedInput.portfolioTag || "",
    mergedInput.emotion || "",
    mergedInput.tradeQuality || "",
    JSON.stringify(mergedInput.checklistItems),
    mergedInput.notes,
    JSON.stringify(normalizeTradeReviewSections(mergedInput.reviewSections)),
    JSON.stringify(mergedInput.screenshots),
    JSON.stringify(mergedInput.chartLinks || []),
    JSON.stringify(mergedInput.executions || []),
    Boolean(mergedInput.hidden),
    "",
    "none"
  ];

  const result = await db.query(
    `
      insert into trade_logs (
        id, user_id, import_source, import_row_key, symbol, side, status, entry_date, exit_date, open_time, close_time, avg_entry, exit_price,
        stop_price, take_profit_price, shares, commission, used_margin, risk, pnl, r_multiple, return_percent, days_in_trade, setup_tags,
        mistake_tags, custom_tags, manual_grade, portfolio_tag, emotion, trade_quality, checklist_items, notes, review_sections, screenshots, chart_links, executions, hidden, group_id, group_role
      )
      values (
        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
        $11, $12, $13, $14, $15, $16, $17, $18, $19, $20,
        $21, $22, $23, $24::jsonb, $25::jsonb, $26::jsonb, $27, $28, $29, $30, $31::jsonb, $32, $33::jsonb, $34::jsonb, $35::jsonb, $36::jsonb, $37, $38, $39
      )
      on conflict (id) do update set
        import_source = excluded.import_source,
        import_row_key = excluded.import_row_key,
        symbol = excluded.symbol,
        side = excluded.side,
        status = excluded.status,
        entry_date = excluded.entry_date,
        exit_date = excluded.exit_date,
        open_time = excluded.open_time,
        close_time = excluded.close_time,
        avg_entry = excluded.avg_entry,
        exit_price = excluded.exit_price,
        stop_price = excluded.stop_price,
        take_profit_price = excluded.take_profit_price,
        shares = excluded.shares,
        commission = excluded.commission,
        used_margin = excluded.used_margin,
        risk = excluded.risk,
        pnl = excluded.pnl,
        r_multiple = excluded.r_multiple,
        return_percent = excluded.return_percent,
        days_in_trade = excluded.days_in_trade,
        setup_tags = excluded.setup_tags,
        mistake_tags = excluded.mistake_tags,
        custom_tags = excluded.custom_tags,
        manual_grade = excluded.manual_grade,
        portfolio_tag = excluded.portfolio_tag,
        emotion = excluded.emotion,
        trade_quality = excluded.trade_quality,
        checklist_items = excluded.checklist_items,
        notes = excluded.notes,
        review_sections = excluded.review_sections,
        screenshots = excluded.screenshots,
        chart_links = excluded.chart_links,
        executions = excluded.executions,
        hidden = case when excluded.import_source = 'manual-combine' then excluded.hidden else trade_logs.hidden end,
        group_id = '',
        group_role = 'none',
        updated_at = now()
      returning ${tradeColumns};
    `,
    values
  );

  return { trade: rowToTrade(result.rows[0]), mode: existingImportedTradeId ? "updated" : "created" } as const;
}

export async function updateTrade(id: string, userId: string, input: TradeLogInput) {
  const now = new Date().toISOString();
  const db = getPool();

  if (!db) {
    if (process.env.NODE_ENV === "production") {
      throw new Error("DATABASE_URL is required in production before trades can be updated.");
    }

    const trades = await readLocalTrades();
    const existing = trades.find((trade) => trade.id === id && trade.userId === userId);

    if (!existing) {
      throw new Error("Trade not found.");
    }

    const next: TradeLogEntry = {
      ...input,
      symbol: existing.symbol,
      importSource: input.importSource || existing.importSource || "",
      importRowKey: input.importRowKey || existing.importRowKey || "",
      manualGrade: input.manualGrade || "",
      openTime: input.openTime || "",
      closeTime: input.closeTime || "",
      portfolioTag: input.portfolioTag || "",
      takeProfitPrice: input.takeProfitPrice || 0,
      commission: input.commission || 0,
      usedMargin: input.usedMargin || 0,
      emotion: input.emotion || "",
      tradeQuality: input.tradeQuality || "",
      reviewSections: normalizeTradeReviewSections(input.reviewSections || existing.reviewSections),
      executions: input.executions || existing.executions || [],
      groupId: "",
      groupRole: "none",
      hidden: existing.hidden,
      id,
      userId,
      createdAt: existing.createdAt,
      updatedAt: now
    };
    await writeLocalTrades(
      trades.map((trade) => (trade.id === id && trade.userId === userId ? next : trade)).sort(tradeLogOrder)
    );
    return next;
  }

  await ensureTradeTable();
  const values = [
    id,
    userId,
    input.importSource || "",
    input.importRowKey || "",
    input.side,
    input.status,
    input.entryDate,
    input.exitDate,
    input.openTime || "",
    input.closeTime || "",
    input.avgEntry,
    input.exitPrice,
    input.stopPrice,
    input.takeProfitPrice,
    input.shares,
    input.commission,
    input.usedMargin,
    input.risk,
    input.pnl,
    input.rMultiple,
    input.returnPercent,
    input.daysInTrade,
    JSON.stringify(input.setupTags),
    JSON.stringify(input.mistakeTags),
    JSON.stringify(input.customTags),
    input.manualGrade || "",
    input.portfolioTag || "",
    input.emotion || "",
    input.tradeQuality || "",
    JSON.stringify(input.checklistItems),
    input.notes,
    JSON.stringify(normalizeTradeReviewSections(input.reviewSections)),
    JSON.stringify(input.screenshots),
    JSON.stringify(input.chartLinks || []),
    JSON.stringify(input.executions || [])
  ];

  const result = await db.query(
    `
      update trade_logs set
        import_source = $3,
        import_row_key = $4,
        side = $5,
        status = $6,
        entry_date = $7,
        exit_date = $8,
        open_time = $9,
        close_time = $10,
        avg_entry = $11,
        exit_price = $12,
        stop_price = $13,
        take_profit_price = $14,
        shares = $15,
        commission = $16,
        used_margin = $17,
        risk = $18,
        pnl = $19,
        r_multiple = $20,
        return_percent = $21,
        days_in_trade = $22,
        setup_tags = $23::jsonb,
        mistake_tags = $24::jsonb,
        custom_tags = $25::jsonb,
        manual_grade = $26,
        portfolio_tag = $27,
        emotion = $28,
        trade_quality = $29,
        checklist_items = $30::jsonb,
        notes = $31,
        review_sections = $32::jsonb,
        screenshots = $33::jsonb,
        chart_links = $34::jsonb,
        executions = $35::jsonb,
        updated_at = now()
      where id = $1 and user_id = $2
      returning ${tradeColumns};
    `,
    values
  );

  if (!result.rows[0]) {
    throw new Error("Trade not found.");
  }

  return rowToTrade(result.rows[0]);
}

export async function hideLegacyCfStatementRows(userId: string, portfolioTag: string) {
  const db = getPool();

  if (!db) {
    const trades = await readLocalTrades();
    let hidden = 0;
    const nextTrades = trades.map((trade) => {
      if (
        trade.userId === userId &&
        trade.portfolioTag === portfolioTag &&
        trade.importSource === "cf-statement-pdf" &&
        (
          trade.importRowKey.startsWith("cf-open:") ||
          trade.importRowKey.startsWith("cf-transaction:") ||
          trade.importRowKey.startsWith("cf-position:") ||
          trade.importRowKey.startsWith("cf-position-open:") ||
          trade.importRowKey.startsWith("cf-position-unmatched:")
        )
      ) {
        hidden += trade.hidden ? 0 : 1;
        return { ...trade, hidden: true, groupId: "", groupRole: "none" as const, updatedAt: new Date().toISOString() };
      }

      return trade;
    });
    await writeLocalTrades(nextTrades);
    return hidden;
  }

  await ensureTradeTable();
  const result = await db.query(
    `
      update trade_logs
      set hidden = true,
          group_id = '',
          group_role = 'none',
          updated_at = now()
      where user_id = $1
        and portfolio_tag = $2
        and import_source = 'cf-statement-pdf'
        and (
          import_row_key like 'cf-open:%'
          or import_row_key like 'cf-transaction:%'
          or import_row_key like 'cf-position:%'
          or import_row_key like 'cf-position-open:%'
          or import_row_key like 'cf-position-unmatched:%'
        )
        and hidden = false
    `,
    [userId, portfolioTag]
  );

  return result.rowCount || 0;
}

function oppositeCfPositionKey(importRowKey: string) {
  const longPrefix = "cf-position:";
  if (!importRowKey.startsWith(longPrefix)) {
    return "";
  }

  const parts = importRowKey.split(":");
  if (parts.length < 4 || (parts[2] !== "LONG" && parts[2] !== "SHORT")) {
    return "";
  }

  parts[2] = parts[2] === "LONG" ? "SHORT" : "LONG";
  return parts.join(":");
}

export async function hideSupersededCfPositionRows(userId: string, portfolioTag: string, importRowKeys: string[]) {
  const staleKeys = Array.from(new Set(importRowKeys.map(oppositeCfPositionKey).filter(Boolean)));

  if (!staleKeys.length) {
    return 0;
  }

  const db = getPool();

  if (!db) {
    const trades = await readLocalTrades();
    let hidden = 0;
    const nextTrades = trades.map((trade) => {
      if (
        trade.userId === userId &&
        trade.portfolioTag === portfolioTag &&
        trade.importSource === "cf-statement-pdf" &&
        staleKeys.includes(trade.importRowKey)
      ) {
        hidden += trade.hidden ? 0 : 1;
        return { ...trade, hidden: true, groupId: "", groupRole: "none" as const, updatedAt: new Date().toISOString() };
      }

      return trade;
    });
    await writeLocalTrades(nextTrades);
    return hidden;
  }

  await ensureTradeTable();
  const result = await db.query(
    `
      update trade_logs
      set hidden = true,
          group_id = '',
          group_role = 'none',
          updated_at = now()
      where user_id = $1
        and portfolio_tag = $2
        and import_source = 'cf-statement-pdf'
        and import_row_key = any($3::text[])
        and hidden = false
    `,
    [userId, portfolioTag, staleKeys]
  );

  return result.rowCount || 0;
}

function cfAggregateOpenComponentPrefixes(importRowKeys: string[]) {
  return Array.from(
    new Set(
      importRowKeys
        .filter((key) => key.startsWith("cf-position-open:"))
        .map((key) => {
          const parts = key.split(":");

          if (parts.length < 4 || (parts[3] !== "LONG" && parts[3] !== "SHORT")) {
            return "";
          }

          return `cf-position:${parts[2]}:${parts[3]}:`;
        })
        .filter(Boolean)
    )
  );
}

export async function hideSupersededCfOpenComponentRows(userId: string, portfolioTag: string, importRowKeys: string[]) {
  const prefixes = cfAggregateOpenComponentPrefixes(importRowKeys);

  if (!prefixes.length) {
    return 0;
  }

  const db = getPool();

  if (!db) {
    const trades = await readLocalTrades();
    let hidden = 0;
    const nextTrades = trades.map((trade) => {
      if (
        trade.userId === userId &&
        trade.portfolioTag === portfolioTag &&
        trade.importSource === "cf-statement-pdf" &&
        trade.status === "OPEN" &&
        !trade.importRowKey.startsWith("cf-position-open:") &&
        prefixes.some((prefix) => trade.importRowKey.startsWith(prefix))
      ) {
        hidden += trade.hidden ? 0 : 1;
        return { ...trade, hidden: true, groupId: "", groupRole: "none" as const, updatedAt: new Date().toISOString() };
      }

      return trade;
    });
    await writeLocalTrades(nextTrades);
    return hidden;
  }

  await ensureTradeTable();
  const patterns = prefixes.map((prefix) => `${prefix}%`);
  const result = await db.query(
    `
      update trade_logs
      set hidden = true,
          group_id = '',
          group_role = 'none',
          updated_at = now()
      where user_id = $1
        and portfolio_tag = $2
        and import_source = 'cf-statement-pdf'
        and status = 'OPEN'
        and import_row_key not like 'cf-position-open:%'
        and import_row_key like any($3::text[])
        and hidden = false
    `,
    [userId, portfolioTag, patterns]
  );

  return result.rowCount || 0;
}

export async function hideStaleCfOpenRows(userId: string, portfolioTag: string, currentOpenImportRowKeys: string[]) {
  const currentOpenKeys = Array.from(new Set(currentOpenImportRowKeys.filter(Boolean)));

  if (!currentOpenKeys.length) {
    return 0;
  }

  const db = getPool();

  if (!db) {
    const trades = await readLocalTrades();
    let hidden = 0;
    const nextTrades = trades.map((trade) => {
      if (
        trade.userId === userId &&
        trade.portfolioTag === portfolioTag &&
        trade.importSource === "cf-statement-pdf" &&
        trade.status === "OPEN" &&
        !currentOpenKeys.includes(trade.importRowKey)
      ) {
        hidden += trade.hidden ? 0 : 1;
        return { ...trade, hidden: true, updatedAt: new Date().toISOString() };
      }

      return trade;
    });
    await writeLocalTrades(nextTrades);
    return hidden;
  }

  await ensureTradeTable();
  const result = await db.query(
    `
      update trade_logs
      set hidden = true,
          updated_at = now()
      where user_id = $1
        and portfolio_tag = $2
        and import_source = 'cf-statement-pdf'
        and status = 'OPEN'
        and import_row_key <> all($3::text[])
        and hidden = false
    `,
    [userId, portfolioTag, currentOpenKeys]
  );

  return result.rowCount || 0;
}

function removeLegacyCombineTags(tags: string[]) {
  return tags.filter((tag) => tag !== "Combined trade" && tag !== "Auto recalculated");
}

export async function cleanupLegacyCombinedTrades(userId: string) {
  const db = getPool();

  if (!db) {
    if (process.env.NODE_ENV === "production") {
      throw new Error("DATABASE_URL is required in production before legacy combined trades can be cleaned up.");
    }

    const trades = await readLocalTrades();
    let deletedParents = 0;
    let resetChildren = 0;

    const nextTrades = trades
      .filter((trade) => {
        if (trade.userId === userId && trade.importSource === "manual-combine") {
          deletedParents += 1;
          return false;
        }
        return true;
      })
      .map((trade) => {
        if (trade.userId !== userId) {
          return trade;
        }

        const nextCustomTags = removeLegacyCombineTags(trade.customTags);
        const shouldResetGrouping = Boolean(trade.groupId) || trade.groupRole !== "none";
        const shouldUpdateTags = nextCustomTags.length !== trade.customTags.length;

        if (!shouldResetGrouping && !shouldUpdateTags) {
          return trade;
        }

        if (shouldResetGrouping) {
          resetChildren += 1;
        }

        return {
          ...trade,
          groupId: "",
          groupRole: "none" as const,
          customTags: nextCustomTags,
          updatedAt: new Date().toISOString()
        };
      });

    await writeLocalTrades(nextTrades);
    return { deletedParents, resetTrades: resetChildren };
  }

  await ensureTradeTable();

  const deleteParents = await db.query(
    `
      delete from trade_logs
      where user_id = $1
        and import_source = 'manual-combine'
      returning id
    `,
    [userId]
  );

  const resetTrades = await db.query(
    `
      update trade_logs
      set group_id = '',
          group_role = 'none',
          custom_tags = (
            select coalesce(jsonb_agg(value), '[]'::jsonb)
            from jsonb_array_elements_text(coalesce(trade_logs.custom_tags, '[]'::jsonb)) as value
            where value not in ('Combined trade', 'Auto recalculated')
          ),
          updated_at = now()
      where user_id = $1
        and (
          group_id <> ''
          or group_role <> 'none'
          or coalesce(trade_logs.custom_tags, '[]'::jsonb) ? 'Combined trade'
          or coalesce(trade_logs.custom_tags, '[]'::jsonb) ? 'Auto recalculated'
        )
    `,
    [userId]
  );

  return {
    deletedParents: deleteParents.rowCount || 0,
    resetTrades: resetTrades.rowCount || 0
  };
}

export async function setTradeHidden(id: string, userId: string, hidden: boolean, customTags?: string[]) {
  const db = getPool();

  if (!db) {
    if (process.env.NODE_ENV === "production") {
      throw new Error("DATABASE_URL is required in production before trades can be updated.");
    }

    const trades = await readLocalTrades();
    let updatedTrade: TradeLogEntry | null = null;
    const nextTrades = trades.map((trade) => {
      if (trade.id !== id || trade.userId !== userId) {
        return trade;
      }

      updatedTrade = { ...trade, hidden, customTags: customTags || trade.customTags, updatedAt: new Date().toISOString() };
      return updatedTrade;
    });

    if (!updatedTrade) {
      throw new Error("Trade not found.");
    }

    await writeLocalTrades(nextTrades);
    return updatedTrade;
  }

  await ensureTradeTable();
  const result = await db.query(
    `update trade_logs set hidden = $3, custom_tags = coalesce($4::jsonb, custom_tags), updated_at = now() where id = $1 and user_id = $2 returning ${tradeColumns}`,
    [id, userId, hidden, customTags ? JSON.stringify(customTags) : null]
  );

  if (!result.rows[0]) {
    throw new Error("Trade not found.");
  }

  return rowToTrade(result.rows[0]);
}

export async function deleteTrade(id: string, userId: string) {
  const db = getPool();

  if (!db) {
    if (process.env.NODE_ENV === "production") {
      throw new Error("DATABASE_URL is required in production before trades can be deleted.");
    }

    const trades = await readLocalTrades();
    await writeLocalTrades(trades.filter((trade) => !(trade.id === id && trade.userId === userId)));
    return;
  }

  await ensureTradeTable();
  await db.query("delete from trade_logs where id = $1 and user_id = $2", [id, userId]);
}

async function ensureTradeScreenshotTable() {
  const db = getPool();

  if (!db) {
    return;
  }

  await db.query(`
    create table if not exists trade_screenshots (
      id text primary key,
      trade_id text not null,
      user_id text not null,
      file_name text not null default '',
      mime_type text not null,
      image_data bytea not null,
      created_at timestamptz not null default now()
    )
  `);
  await db.query("create index if not exists trade_screenshots_trade_id_idx on trade_screenshots (trade_id)");
}

export async function saveTradeScreenshot(
  tradeId: string,
  userId: string,
  fileName: string,
  mimeType: string,
  imageData: Buffer
) {
  const db = getPool();

  if (!db) {
    throw new Error("DATABASE_URL is required before screenshots can be saved.");
  }

  await ensureTradeTable();
  await ensureTradeScreenshotTable();
  const client = await db.connect();
  const id = crypto.randomUUID();
  const url = `/api/trades/${encodeURIComponent(tradeId)}/screenshots/${encodeURIComponent(id)}`;

  try {
    await client.query("begin");
    const tradeResult = await client.query("select id from trade_logs where id = $1 and user_id = $2 for update", [
      tradeId,
      userId
    ]);

    if (!tradeResult.rows[0]) {
      throw new Error("Trade not found.");
    }

    await client.query(
      "insert into trade_screenshots (id, trade_id, user_id, file_name, mime_type, image_data) values ($1, $2, $3, $4, $5, $6)",
      [id, tradeId, userId, fileName, mimeType, imageData]
    );
    await client.query(
      "update trade_logs set screenshots = coalesce(screenshots, '[]'::jsonb) || jsonb_build_array($3::text), updated_at = now() where id = $1 and user_id = $2",
      [tradeId, userId, url]
    );
    await client.query("commit");
    return { id, url };
  } catch (error) {
    await client.query("rollback");
    throw error;
  } finally {
    client.release();
  }
}

export async function getTradeScreenshot(id: string) {
  const db = getPool();

  if (!db) {
    return null;
  }

  await ensureTradeScreenshotTable();
  const result = await db.query(
    "select id, trade_id, user_id, file_name, mime_type, image_data from trade_screenshots where id = $1",
    [id]
  );
  const row = result.rows[0];

  if (!row) {
    return null;
  }

  return {
    id: String(row.id),
    tradeId: String(row.trade_id),
    userId: String(row.user_id),
    fileName: String(row.file_name || ""),
    mimeType: String(row.mime_type),
    imageData: row.image_data as Buffer
  };
}

const BRANDEN_BACKUP_SETTING_KEYS = [
  "checklist_grade_bands",
  "branden_portfolio_names",
  "branden_column_preferences",
  "setup_checklist_templates",
  "market_cycle_entries_branden",
  "weekly_watchlists_branden",
  "weekly_process_focus_branden"
] as const;

export type BrandenJournalBackup = {
  format: "branden-journal-backup";
  version: 1;
  exportedAt: string;
  userId: "branden";
  reports: MonthlyReport[];
  trades: TradeLogEntry[];
  settings: Record<string, unknown>;
  tradeScreenshots: Array<{
    id: string;
    tradeId: string;
    userId: string;
    fileName: string;
    mimeType: string;
    imageDataBase64: string;
    createdAt: string;
  }>;
};

export async function exportBrandenJournalBackup(): Promise<BrandenJournalBackup> {
  const exportedAt = new Date().toISOString();
  const db = getPool();

  if (!db) {
    const reports = (await readLocalReports()).filter((report) => report.userId === "branden");
    const trades = (await readLocalTrades()).filter((trade) => trade.userId === "branden");
    const localSettings = await readLocalSettings();
    const settings: Record<string, unknown> = {
      checklist_grade_bands: localSettings.checklistGradeBands,
      branden_portfolio_names: localSettings.brandenPortfolioNames,
      branden_column_preferences: localSettings.brandenColumnPreferences,
      setup_checklist_templates: localSettings.setupChecklistTemplates,
      market_cycle_entries_branden: localSettings.market_cycle_entries_branden,
      weekly_watchlists_branden: localSettings.weekly_watchlists_branden,
      weekly_process_focus_branden: localSettings.weekly_process_focus_branden
    };
    return {
      format: "branden-journal-backup",
      version: 1,
      exportedAt,
      userId: "branden",
      reports,
      trades,
      settings,
      tradeScreenshots: []
    };
  }

  await ensureTable();
  await ensureTradeTable();
  await ensureSettingsTable();
  await ensureTradeScreenshotTable();
  const [reportResult, tradeResult, settingResult, screenshotResult] = await Promise.all([
    db.query(`select ${columns} from monthly_reports where user_id = $1 order by month asc`, ["branden"]),
    db.query(`select ${tradeColumns} from trade_logs where user_id = $1 order by entry_date desc, created_at desc`, ["branden"]),
    db.query("select key, value from app_settings where key = any($1::text[])", [BRANDEN_BACKUP_SETTING_KEYS]),
    db.query(
      `select id, trade_id, user_id, file_name, mime_type, image_data, created_at
       from trade_screenshots where user_id = $1 order by created_at asc`,
      ["branden"]
    )
  ]);

  return {
    format: "branden-journal-backup",
    version: 1,
    exportedAt,
    userId: "branden",
    reports: reportResult.rows.map(rowToReport),
    trades: tradeResult.rows.map(rowToTrade),
    settings: Object.fromEntries(settingResult.rows.map((row) => [String(row.key), row.value])),
    tradeScreenshots: screenshotResult.rows.map((row) => ({
      id: String(row.id),
      tradeId: String(row.trade_id),
      userId: String(row.user_id),
      fileName: String(row.file_name || ""),
      mimeType: String(row.mime_type || "image/png"),
      imageDataBase64: (row.image_data as Buffer).toString("base64"),
      createdAt: new Date(String(row.created_at)).toISOString()
    }))
  };
}

function validBrandenBackup(value: unknown): value is BrandenJournalBackup {
  if (!value || typeof value !== "object") return false;
  const backup = value as Partial<BrandenJournalBackup>;
  return (
    backup.format === "branden-journal-backup" &&
    backup.version === 1 &&
    backup.userId === "branden" &&
    Array.isArray(backup.reports) &&
    Array.isArray(backup.trades) &&
    Boolean(backup.settings && typeof backup.settings === "object" && !Array.isArray(backup.settings)) &&
    Array.isArray(backup.tradeScreenshots)
  );
}

export async function importBrandenJournalBackup(value: unknown) {
  if (!validBrandenBackup(value)) {
    throw new Error("This is not a valid Branden journal backup file.");
  }

  if (value.reports.some((report) => report.userId !== "branden") || value.trades.some((trade) => trade.userId !== "branden")) {
    throw new Error("The backup contains records for another journal owner.");
  }

  const db = getPool();
  if (!db) {
    if (process.env.NODE_ENV === "production") {
      throw new Error("DATABASE_URL is required in production before a backup can be imported.");
    }
    const currentReports = await readLocalReports();
    const currentTrades = await readLocalTrades();
    await writeLocalReports([...currentReports.filter((report) => report.userId !== "branden"), ...value.reports]);
    await writeLocalTrades([...currentTrades.filter((trade) => trade.userId !== "branden"), ...value.trades].sort(tradeLogOrder));
    const settings = await readLocalSettings();
    const backupSettings = value.settings;
    settings.checklistGradeBands = backupSettings.checklist_grade_bands;
    settings.brandenPortfolioNames = backupSettings.branden_portfolio_names;
    settings.brandenColumnPreferences = backupSettings.branden_column_preferences;
    settings.setupChecklistTemplates = backupSettings.setup_checklist_templates;
    settings.market_cycle_entries_branden = backupSettings.market_cycle_entries_branden;
    settings.weekly_watchlists_branden = backupSettings.weekly_watchlists_branden;
    settings.weekly_process_focus_branden = backupSettings.weekly_process_focus_branden;
    await writeLocalSettings(settings);
    return { reports: value.reports.length, trades: value.trades.length, screenshots: 0 };
  }

  await ensureTable();
  await ensureTradeTable();
  await ensureSettingsTable();
  await ensureTradeScreenshotTable();
  const client = await db.connect();
  try {
    await client.query("begin");
    await client.query("delete from trade_screenshots where user_id = $1", ["branden"]);
    await client.query("delete from trade_logs where user_id = $1", ["branden"]);
    await client.query("delete from monthly_reports where user_id = $1", ["branden"]);
    await client.query("delete from app_settings where key = any($1::text[])", [BRANDEN_BACKUP_SETTING_KEYS]);

    for (const report of value.reports) {
      await client.query(
        `
          insert into monthly_reports (
            id, user_id, month, account_size, total_return, percent_return, net_pnl, total_payouts,
            total_trades, win_rate, avg_r, total_r, avg_win_r, avg_loss_r, avg_win, avg_loss,
            avg_risk, current_risk_percent, expected_value_r, sharpe_ratio, avg_trade_length,
            avg_swing_length, longest_win_streak, longest_loss_streak, notes, created_at, updated_at
          ) values (
            $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19,$20,$21,$22,$23,$24,$25,$26,$27
          )
        `,
        [
          report.id, report.userId, report.month, report.accountSize, report.totalReturn, report.percentReturn,
          report.netPnl, report.totalPayouts, report.totalTrades, report.winRate, report.avgR, report.totalR,
          report.avgWinR, report.avgLossR, report.avgWin, report.avgLoss, report.avgRisk,
          report.currentRiskPercent, report.expectedValueR, report.sharpeRatio, report.avgTradeLength,
          report.avgSwingLength, report.longestWinStreak, report.longestLossStreak, report.notes,
          report.createdAt, report.updatedAt
        ]
      );
    }

    for (const trade of value.trades) {
      await client.query(
        `
          insert into trade_logs (
            id,user_id,import_source,import_row_key,symbol,side,status,entry_date,exit_date,open_time,close_time,
            avg_entry,exit_price,stop_price,take_profit_price,shares,commission,used_margin,risk,pnl,r_multiple,
            return_percent,days_in_trade,setup_tags,mistake_tags,custom_tags,manual_grade,portfolio_tag,emotion,
            trade_quality,checklist_items,notes,review_sections,screenshots,chart_links,executions,hidden,group_id,group_role,
            created_at,updated_at
          ) values (
            $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19,$20,$21,$22,$23,
            $24::jsonb,$25::jsonb,$26::jsonb,$27,$28,$29,$30,$31::jsonb,$32,$33::jsonb,$34::jsonb,$35::jsonb,
            $36::jsonb,$37,$38,$39,$40,$41
          )
        `,
        [
          trade.id, trade.userId, trade.importSource, trade.importRowKey, trade.symbol, trade.side, trade.status,
          trade.entryDate, trade.exitDate, trade.openTime, trade.closeTime, trade.avgEntry, trade.exitPrice,
          trade.stopPrice, trade.takeProfitPrice, trade.shares, trade.commission, trade.usedMargin, trade.risk,
          trade.pnl, trade.rMultiple, trade.returnPercent, trade.daysInTrade, JSON.stringify(trade.setupTags),
          JSON.stringify(trade.mistakeTags), JSON.stringify(trade.customTags), trade.manualGrade,
          trade.portfolioTag, trade.emotion, trade.tradeQuality, JSON.stringify(trade.checklistItems), trade.notes,
          JSON.stringify(normalizeTradeReviewSections(trade.reviewSections)), JSON.stringify(trade.screenshots), JSON.stringify(trade.chartLinks), JSON.stringify(trade.executions),
          trade.hidden, trade.groupId || "", trade.groupRole || "none", trade.createdAt, trade.updatedAt
        ]
      );
    }

    for (const [key, settingValue] of Object.entries(value.settings)) {
      if (!BRANDEN_BACKUP_SETTING_KEYS.includes(key as (typeof BRANDEN_BACKUP_SETTING_KEYS)[number])) continue;
      if (settingValue === undefined) continue;
      await client.query("insert into app_settings (key, value) values ($1, $2::jsonb)", [
        key,
        JSON.stringify(settingValue)
      ]);
    }

    for (const screenshot of value.tradeScreenshots) {
      if (screenshot.userId !== "branden" || !value.trades.some((trade) => trade.id === screenshot.tradeId)) {
        throw new Error("The backup contains an invalid screenshot reference.");
      }
      await client.query(
        `insert into trade_screenshots
          (id, trade_id, user_id, file_name, mime_type, image_data, created_at)
         values ($1,$2,$3,$4,$5,$6,$7)`,
        [
          screenshot.id, screenshot.tradeId, screenshot.userId, screenshot.fileName, screenshot.mimeType,
          Buffer.from(screenshot.imageDataBase64, "base64"), screenshot.createdAt
        ]
      );
    }

    await client.query("commit");
    return {
      reports: value.reports.length,
      trades: value.trades.length,
      screenshots: value.tradeScreenshots.length
    };
  } catch (error) {
    await client.query("rollback");
    throw error;
  } finally {
    client.release();
  }
}
