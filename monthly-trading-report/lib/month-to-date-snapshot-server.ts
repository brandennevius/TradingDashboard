import { mkdir, rename, writeFile } from "node:fs/promises";
import path from "node:path";
import packageJson from "../package.json";
import { latestCompletedMarketSession, marketSessionCloseTimestamp, type SnapshotPrice } from "./daily-portfolio-snapshot";
import { getMarketCandlesWithProvider } from "./market-data";
import { buildMonthToDateSnapshot, renderMonthToDateSnapshotMarkdown, resolveMtdPeriod, validateMonthToDateSnapshot } from "./month-to-date-snapshot";
import { getBrandenPortfolioSettings, getWeeklyProcessFocus, listBrandenVisibleTrades } from "./store";
import type { TradeLogEntry } from "./types";
import type { WeeklyFocus } from "./weekly-focus";

type Dependencies = {
  loadTrades: () => Promise<TradeLogEntry[]>;
  loadPortfolioSettings: typeof getBrandenPortfolioSettings;
  loadWeeklyFocus: () => Promise<WeeklyFocus>;
  loadPrice: (symbol: string, asOfDate: string) => Promise<SnapshotPrice>;
  now: () => Date;
};

export type GenerateMonthToDateSnapshotOptions = {
  month?: string;
  asOfDate?: string;
  asOfTimestamp?: string;
  portfolioName?: string;
  outputDirectory?: string;
  writeExports?: boolean;
  dependencies?: Partial<Dependencies>;
};

export class MonthToDateSnapshotValidationError extends Error {
  constructor(public readonly code: string, message: string, public readonly diagnostic?: Record<string, unknown>) {
    super(message);
    this.name = "MonthToDateSnapshotValidationError";
  }
}

function nyDate(now: Date) {
  return new Intl.DateTimeFormat("en-CA", { timeZone: "America/New_York", year: "numeric", month: "2-digit", day: "2-digit" }).format(now);
}

function executionAsOfQuantity(trade: TradeLogEntry, asOfDate: string) {
  return trade.executions
    .filter((item) => item.date <= asOfDate)
    .reduce((quantity, item) => quantity + (item.type === "ENTRY" ? item.shares : -item.shares), 0);
}

async function loadPrice(symbol: string, asOfDate: string): Promise<SnapshotPrice> {
  const result = await getMarketCandlesWithProvider(symbol, "1d");
  const candle = result.candles.filter((item) => item.time <= asOfDate).at(-1);
  return {
    symbol,
    price: candle?.close ?? null,
    timestamp: candle?.time ? marketSessionCloseTimestamp(candle.time) : null,
    sessionDate: candle?.time || null,
    provider: result.provider,
    priceType: "delayed_close",
    retrievedAt: new Date().toISOString()
  };
}

async function mapWithConcurrency<T, R>(values: T[], limit: number, task: (value: T) => Promise<R>) {
  const output = new Array<R>(values.length);
  let next = 0;
  async function worker() {
    while (next < values.length) {
      const index = next++;
      output[index] = await task(values[index]);
    }
  }
  await Promise.all(Array.from({ length: Math.min(limit, values.length) }, worker));
  return output;
}

async function atomicWrite(filePath: string, contents: string) {
  await mkdir(path.dirname(filePath), { recursive: true });
  const temporaryPath = `${filePath}.${process.pid}.tmp`;
  await writeFile(temporaryPath, contents, "utf8");
  await rename(temporaryPath, filePath);
}

export async function generateMonthToDateSnapshot(options: GenerateMonthToDateSnapshotOptions = {}) {
  const dependencies: Dependencies = {
    loadTrades: listBrandenVisibleTrades,
    loadPortfolioSettings: getBrandenPortfolioSettings,
    loadWeeklyFocus: () => getWeeklyProcessFocus("branden"),
    loadPrice,
    now: () => new Date(),
    ...options.dependencies
  };
  const now = dependencies.now();
  const period = resolveMtdPeriod({ month: options.month, asOfDate: options.asOfDate, asOfTimestamp: options.asOfTimestamp, now });
  if (period.asOfDate > nyDate(now)) {
    throw new MonthToDateSnapshotValidationError("AS_OF_DATE_IN_FUTURE", "The selected as-of date is in the future.", { selectedAsOfDate: period.asOfDate, currentNewYorkDate: nyDate(now) });
  }
  const [allTrades, settings, weeklyFocus] = await Promise.all([
    dependencies.loadTrades(), dependencies.loadPortfolioSettings(), dependencies.loadWeeklyFocus()
  ]);
  const portfolioName = String(options.portfolioName || settings.defaultPortfolio || "").trim();
  if (!portfolioName || (settings.portfolios.length && !settings.portfolios.includes(portfolioName))) {
    throw new MonthToDateSnapshotValidationError("PORTFOLIO_UNRESOLVED", "The selected portfolio could not be resolved.", { selectedPortfolio: portfolioName || null });
  }
  const portfolioMeta = settings.portfolioMeta?.[portfolioName];
  if (!portfolioMeta?.currentEquity) {
    throw new MonthToDateSnapshotValidationError("AUTHORITATIVE_EQUITY_UNAVAILABLE", "Authoritative account equity is required before an MTD snapshot can be generated.", { portfolio: portfolioName });
  }
  const coverage = String(portfolioMeta.equityStatementDate || "").slice(0, 10);
  const requiredMarketStateDate = latestCompletedMarketSession(new Date(period.end));
  if (!coverage || coverage < requiredMarketStateDate) {
    throw new MonthToDateSnapshotValidationError("BROKER_IMPORT_DATE_COVERAGE_INSUFFICIENT", "The latest broker statement does not cover the selected as-of date.", {
      portfolio: portfolioName, requestedAsOfDate: period.asOfDate, requiredMarketStateDate, statementCoverageDate: coverage || null
    });
  }
  const trades = allTrades.filter((trade) => !trade.hidden && trade.portfolioTag === portfolioName);
  const included = trades.filter((trade) => {
    const hasExecution = trade.executions.some((item) => item.date <= period.asOfDate && item.date >= `${period.month}-01`);
    const active = trade.entryDate <= period.asOfDate && (!trade.exitDate || trade.exitDate >= `${period.month}-01`);
    return hasExecution || active;
  });
  const relevantNeedsReview = included.filter((trade) => trade.customTags.some((tag) => tag.trim().toLowerCase() === "needs review"));
  if (relevantNeedsReview.length) {
    throw new MonthToDateSnapshotValidationError("BROKER_IMPORT_NEEDS_REVIEW", "One or more broker-import rows affecting the selected period still require review.", {
      portfolio: portfolioName,
      count: relevantNeedsReview.length,
      sample: relevantNeedsReview.slice(0, 5).map((trade) => ({ tradeId: trade.id, symbol: trade.symbol }))
    });
  }
  const missingExecutions = included.filter((trade) => !trade.executions.length);
  if (missingExecutions.length) {
    throw new MonthToDateSnapshotValidationError("BROKER_IMPORT_MISSING_EXECUTIONS", "Execution data are incomplete for one or more trades affecting the selected period.", {
      portfolio: portfolioName,
      count: missingExecutions.length,
      sample: missingExecutions.slice(0, 5).map((trade) => ({ tradeId: trade.id, symbol: trade.symbol }))
    });
  }
  const openSymbols = Array.from(new Set(included.filter((trade) => executionAsOfQuantity(trade, period.asOfDate) > 0.000001).map((trade) => trade.symbol))).sort();
  // An MTD snapshot may be generated during the current open session. Valuation
  // must use the latest completed close, never an incomplete same-day candle.
  const loadedPrices = await mapWithConcurrency(openSymbols, 4, (symbol) => dependencies.loadPrice(symbol, requiredMarketStateDate));
  const prices = new Map(loadedPrices.map((price) => [price.symbol, price]));
  const expectedPriceSession = requiredMarketStateDate;
  const missingPrices = loadedPrices.filter((price) => price.price === null || price.sessionDate !== expectedPriceSession);
  if (missingPrices.length) {
    throw new MonthToDateSnapshotValidationError("CURRENT_PRICES_INVALID", "Current valuation prices are missing or invalid for one or more open positions.", {
      portfolio: portfolioName, expectedPriceSession, symbols: missingPrices.map((price) => price.symbol)
    });
  }
  const snapshot = buildMonthToDateSnapshot({
    month: period.month,
    asOfDate: period.asOfDate,
    asOfTimestamp: period.end,
    generatedAt: now.toISOString(),
    portfolioName,
    trades,
    portfolioMeta,
    prices,
    weeklyFocus,
    sourceEnvironment: process.env.NODE_ENV || "development",
    applicationVersion: packageJson.version
  });
  const includedIds = new Set(snapshot.trades.map((trade) => trade.trade_id));
  const unrelatedNeedsReview = trades.filter((trade) =>
    !includedIds.has(trade.id) && trade.customTags.some((tag) => tag.trim().toLowerCase() === "needs review")
  );
  if (unrelatedNeedsReview.length) {
    snapshot.diagnostics.push({
      code: "BROKER_IMPORT_UNRELATED_ROWS_NEED_REVIEW",
      severity: "warning",
      message: `${unrelatedNeedsReview.length} historical broker-import row${unrelatedNeedsReview.length === 1 ? "" : "s"} require review but do not affect this MTD snapshot: ${unrelatedNeedsReview.slice(0, 5).map((trade) => `${trade.symbol} (${trade.id})`).join(", ")}.`,
      blocking: false
    });
  }
  const validationErrors = validateMonthToDateSnapshot(snapshot);
  if (validationErrors.length || snapshot.status === "BLOCKED") {
    throw new MonthToDateSnapshotValidationError("SNAPSHOT_VALIDATION_FAILED", validationErrors.join(" ") || "Core MTD snapshot data are incomplete.", {
      portfolio: portfolioName,
      validationErrors,
      blockingDiagnostics: snapshot.diagnostics.filter((item) => item.blocking).slice(0, 10)
    });
  }
  const markdown = renderMonthToDateSnapshotMarkdown(snapshot);
  const baseName = `month-to-date-trading-snapshot-${period.month}-through-${period.asOfDate}`;
  const outputDirectory = path.resolve(options.outputDirectory || path.join(process.cwd(), "data", "exports", "month-to-date-snapshots"));
  const jsonPath = path.join(outputDirectory, `${baseName}.json`);
  const markdownPath = path.join(outputDirectory, `${baseName}.md`);
  if (options.writeExports !== false) {
    await Promise.all([
      atomicWrite(jsonPath, `${JSON.stringify(snapshot, null, 2)}\n`),
      atomicWrite(markdownPath, markdown)
    ]);
  }
  return { snapshot, markdown, baseName, jsonPath, markdownPath };
}
