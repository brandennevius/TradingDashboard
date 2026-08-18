import { mkdir, rename, writeFile } from "node:fs/promises";
import path from "node:path";
import packageJson from "../package.json";
import {
  buildDailyPortfolioSnapshot,
  marketSessionCloseTimestamp,
  renderDailyPortfolioSnapshotMarkdown,
  resolveSnapshotSession,
  snapshotStatusFromWarnings,
  type SnapshotPrice,
  type SnapshotWarning
} from "./daily-portfolio-snapshot";
import { getExactMarketSessionPrice } from "./market-data";
import { getBrandenPortfolioSettings, getWeeklyProcessFocus, listBrandenVisibleTrades } from "./store";
import type { TradeLogEntry } from "./types";
import type { WeeklyFocus } from "./weekly-focus";

type SnapshotDependencies = {
  loadTrades: () => Promise<TradeLogEntry[]>;
  loadPortfolioSettings: typeof getBrandenPortfolioSettings;
  loadWeeklyFocus: () => Promise<WeeklyFocus>;
  loadPrice: (symbol: string, session: string) => Promise<SnapshotPrice>;
  now: () => Date;
};

export type GenerateDailyPortfolioSnapshotOptions = {
  session: string;
  accountName?: string;
  outputDirectory?: string;
  writeExports?: boolean;
  dependencies?: Partial<SnapshotDependencies>;
};

export class SnapshotValidationError extends Error {
  constructor(
    public readonly code: SnapshotValidationCode,
    message: string,
    public readonly diagnostic?: SnapshotValidationDiagnostic | SnapshotSessionDiagnostic
  ) {
    super(message);
    this.name = "SnapshotValidationError";
  }
}

export type BrokerImportValidationCode =
  | "BROKER_IMPORT_NOT_FOUND"
  | "BROKER_IMPORT_STALE"
  | "BROKER_IMPORT_NEEDS_REVIEW"
  | "BROKER_IMPORT_MISSING_EXECUTIONS"
  | "BROKER_IMPORT_PORTFOLIO_MISMATCH"
  | "BROKER_IMPORT_DATE_COVERAGE_INSUFFICIENT";

export type SnapshotValidationCode =
  | "PORTFOLIO_UNRESOLVED"
  | "POINT_IN_TIME_UNAVAILABLE"
  | "CURRENT_PRICES_INVALID"
  | "SNAPSHOT_SESSION_NOT_COMPLETE"
  | BrokerImportValidationCode;

export type SnapshotSessionDiagnostic = {
  selectedSession: string;
  submittedSession: string;
  currentNewYorkDateTime: string;
  latestCompletedSession: string;
  regularSessionCompletionTime: string;
  validationCodes: ["SNAPSHOT_SESSION_NOT_COMPLETE"];
};

export type SnapshotValidationDiagnostic = {
  requestedSession: string;
  portfolio: string;
  latestBrokerImportTimestamp: string | null;
  latestStatementCoverageDate: string | null;
  totalImportedTradeCount: number;
  needsReviewCount: number;
  missingExecutionsCount: number;
  validationCodes: BrokerImportValidationCode[];
  samples: Partial<Record<BrokerImportValidationCode, Array<{ ticker: string; tradeId: string }>>>;
  needsReviewRows: NeedsReviewRowDiagnostic[];
};

export type NeedsReviewRowDiagnostic = {
  ticker: string;
  tradeId: string;
  entryDate: string;
  exitDate: string | null;
  status: "OPEN" | "CLOSED";
  affectsRequestedSnapshot: boolean;
  blockingReason: string | null;
};

type BrokerPortfolioMeta = {
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

const BROKER_IMPORT_SOURCE = "cf-statement-pdf";
const DIAGNOSTIC_SAMPLE_LIMIT = 5;

function diagnosticSample(trades: TradeLogEntry[]) {
  return trades.slice(0, DIAGNOSTIC_SAMPLE_LIMIT).map((trade) => ({ ticker: trade.symbol, tradeId: trade.id }));
}

function usMarketDate(value: string | undefined) {
  const raw = String(value || "").trim();
  if (!raw) return null;
  // A statement date without a time is already a U.S. market calendar date.
  // Parsing it as UTC would incorrectly shift it to the previous NY session.
  if (/^\d{4}-\d{2}-\d{2}$/.test(raw)) return raw;
  const timestamp = new Date(raw);
  if (!Number.isFinite(timestamp.getTime())) return null;
  const parts = new Intl.DateTimeFormat("en-US", {
    timeZone: "America/New_York", year: "numeric", month: "2-digit", day: "2-digit"
  }).formatToParts(timestamp);
  const values = Object.fromEntries(parts.map((part) => [part.type, part.value]));
  return `${values.year}-${values.month}-${values.day}`;
}

function needsReviewRelevance(trade: TradeLogEntry, session: string): NeedsReviewRowDiagnostic {
  const executionsDuringSession = trade.executions.some((execution) => execution.date === session);
  const closedDuringSession = trade.exitDate === session;
  const openAtSnapshot = trade.status === "OPEN" && trade.entryDate <= session;
  const blockingReason = openAtSnapshot
    ? "Open position quantity or cost basis may affect the requested snapshot."
    : closedDuringSession
      ? "Trade closed during the requested session."
      : executionsDuringSession
        ? "Execution is included in the requested session."
        : null;
  return {
    ticker: trade.symbol,
    tradeId: trade.id,
    entryDate: trade.entryDate,
    exitDate: trade.exitDate || null,
    status: trade.status === "OPEN" ? "OPEN" : "CLOSED",
    affectsRequestedSnapshot: Boolean(blockingReason),
    blockingReason
  };
}

function brokerImportDiagnostic(trades: TradeLogEntry[], portfolio: string, session: string, portfolioMeta: BrokerPortfolioMeta | undefined): SnapshotValidationDiagnostic | null {
  const importedTrades = trades.filter((trade) => !trade.hidden && trade.importSource === BROKER_IMPORT_SOURCE);
  const accountImportedTrades = importedTrades.filter((trade) => trade.portfolioTag === portfolio);
  const needsReview = accountImportedTrades.filter((trade) => trade.customTags.some((tag) => tag.trim().toLowerCase() === "needs review"));
  const needsReviewRows = needsReview.map((trade) => needsReviewRelevance(trade, session));
  const relevantNeedsReview = needsReview.filter((trade) => needsReviewRelevance(trade, session).affectsRequestedSnapshot);
  const missingExecutions = accountImportedTrades.filter((trade) => !trade.executions.length);
  const coverageDate = usMarketDate(portfolioMeta?.equityStatementDate);
  const validationCodes: BrokerImportValidationCode[] = [];
  const samples: SnapshotValidationDiagnostic["samples"] = {};

  if (!accountImportedTrades.length) {
    if (importedTrades.length) {
      validationCodes.push("BROKER_IMPORT_PORTFOLIO_MISMATCH");
      samples.BROKER_IMPORT_PORTFOLIO_MISMATCH = diagnosticSample(importedTrades);
    } else {
      validationCodes.push("BROKER_IMPORT_NOT_FOUND");
    }
  }
  if (coverageDate && coverageDate < session) {
    validationCodes.push("BROKER_IMPORT_STALE", "BROKER_IMPORT_DATE_COVERAGE_INSUFFICIENT");
  }
  if (relevantNeedsReview.length) {
    validationCodes.push("BROKER_IMPORT_NEEDS_REVIEW");
    samples.BROKER_IMPORT_NEEDS_REVIEW = diagnosticSample(relevantNeedsReview);
  }
  if (missingExecutions.length) {
    validationCodes.push("BROKER_IMPORT_MISSING_EXECUTIONS");
    samples.BROKER_IMPORT_MISSING_EXECUTIONS = diagnosticSample(missingExecutions);
  }

  if (!validationCodes.length && !needsReviewRows.length) return null;
  return {
    requestedSession: session,
    portfolio,
    latestBrokerImportTimestamp: accountImportedTrades.map((trade) => trade.updatedAt).filter(Boolean).sort().at(-1) || null,
    latestStatementCoverageDate: coverageDate,
    totalImportedTradeCount: accountImportedTrades.length,
    needsReviewCount: needsReview.length,
    missingExecutionsCount: missingExecutions.length,
    validationCodes,
    samples,
    needsReviewRows
  };
}

function brokerValidationMessage(diagnostic: SnapshotValidationDiagnostic) {
  const details = diagnostic.validationCodes.map((code) => {
    if (code === "BROKER_IMPORT_NOT_FOUND") return "No CF broker-import trades were found for this portfolio.";
    if (code === "BROKER_IMPORT_STALE") return `Latest statement coverage is ${diagnostic.latestStatementCoverageDate}; it does not cover ${diagnostic.requestedSession}.`;
    if (code === "BROKER_IMPORT_NEEDS_REVIEW") return `${diagnostic.needsReviewCount} imported row${diagnostic.needsReviewCount === 1 ? "" : "s"} still require review.`;
    if (code === "BROKER_IMPORT_MISSING_EXECUTIONS") return `${diagnostic.missingExecutionsCount} imported trade${diagnostic.missingExecutionsCount === 1 ? "" : "s"} have no execution records.`;
    if (code === "BROKER_IMPORT_PORTFOLIO_MISMATCH") return "CF broker-import trades exist, but not for the selected portfolio.";
    return `Statement coverage is unavailable or does not match ${diagnostic.requestedSession}.`;
  });
  return `Snapshot not generated. ${details.join(" ")}`;
}

async function loadSessionPrice(symbol: string, session: string): Promise<SnapshotPrice> {
  const result = await getExactMarketSessionPrice(symbol, session);
  return {
    symbol,
    price: result.price,
    timestamp: result.timestamp || (result.sessionDate ? marketSessionCloseTimestamp(result.sessionDate) : null),
    sessionDate: result.sessionDate,
    provider: result.provider,
    priceType: result.priceType,
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

export async function generateDailyPortfolioSnapshot(options: GenerateDailyPortfolioSnapshotOptions) {
  const dependencies: SnapshotDependencies = {
    loadTrades: listBrandenVisibleTrades,
    loadPortfolioSettings: getBrandenPortfolioSettings,
    loadWeeklyFocus: () => getWeeklyProcessFocus("branden"),
    loadPrice: loadSessionPrice,
    now: () => new Date(),
    ...options.dependencies
  };
  const now = dependencies.now();
  const session = resolveSnapshotSession(options.session, now);
  if (!session.complete) {
    const diagnostic: SnapshotSessionDiagnostic = {
      selectedSession: options.session,
      submittedSession: session.requested,
      currentNewYorkDateTime: session.currentNewYorkDateTime,
      latestCompletedSession: session.latestCompleted,
      regularSessionCompletionTime: session.regularSessionCompletionTime,
      validationCodes: ["SNAPSHOT_SESSION_NOT_COMPLETE"]
    };
    throw new SnapshotValidationError(
      "SNAPSHOT_SESSION_NOT_COMPLETE",
      "Snapshot not generated. The selected market session has not completed.",
      diagnostic
    );
  }
  const [trades, portfolioSettings, weeklyFocus] = await Promise.all([
    dependencies.loadTrades(),
    dependencies.loadPortfolioSettings(),
    dependencies.loadWeeklyFocus()
  ]);
  const accountName = String(options.accountName || portfolioSettings.defaultPortfolio || "").trim();
  if (!accountName) throw new SnapshotValidationError("PORTFOLIO_UNRESOLVED", "No portfolio is selected and no default portfolio is configured.");
  if (portfolioSettings.portfolios.length && !portfolioSettings.portfolios.includes(accountName)) {
    throw new SnapshotValidationError("PORTFOLIO_UNRESOLVED", `Portfolio ${accountName} could not be resolved.`);
  }
  const accountTrades = trades.filter((trade) => !trade.hidden && trade.portfolioTag === accountName);
  const portfolioMeta = portfolioSettings.portfolioMeta?.[accountName];
  const brokerDiagnostic = brokerImportDiagnostic(trades, accountName, session.resolved, portfolioMeta);
  if (brokerDiagnostic?.validationCodes.length) {
    throw new SnapshotValidationError(
      brokerDiagnostic.validationCodes[0],
      brokerValidationMessage(brokerDiagnostic),
      brokerDiagnostic
    );
  }
  const laterActivity = accountTrades.filter((trade) =>
    trade.entryDate > session.resolved || trade.executions.some((execution) => execution.date > session.resolved)
  );
  if (laterActivity.length) {
    const symbols = Array.from(new Set(laterActivity.map((trade) => trade.symbol))).sort();
    throw new SnapshotValidationError(
      "POINT_IN_TIME_UNAVAILABLE",
      `Snapshot not generated: trade-log activity after ${session.resolved} exists for ${symbols.join(", ")}. Current trade-log position state cannot be used for that earlier session.`
    );
  }
  const openSymbols = Array.from(new Set(accountTrades.filter((trade) => trade.status === "OPEN").map((trade) => trade.symbol))).sort();
  const loadedPrices = await mapWithConcurrency(openSymbols, 2, (symbol) => dependencies.loadPrice(symbol, session.resolved));
  const prices = new Map(loadedPrices.map((price) => [price.symbol, price]));
  const sessionPortfolioMeta = portfolioMeta?.equityStatementDate === session.resolved
    ? portfolioMeta
    : portfolioMeta
      ? { ...portfolioMeta, currentEquity: undefined, statementEquity: undefined, floatingPnl: undefined }
      : undefined;
  const snapshot = buildDailyPortfolioSnapshot({
    requestedSession: session.resolved,
    latestCompletedMarketSession: session.latestCompleted,
    generatedAt: now.toISOString(),
    accountName,
    portfolioMeta: sessionPortfolioMeta,
    trades: accountTrades,
    // The existing template getter can perform a screenshot migration. A snapshot
    // must not trigger that write, so only trade-stored manual grades/checklists
    // are used until the repository exposes a strictly read-only template query.
    setupTemplates: [],
    prices,
    sourceEnvironment: process.env.NODE_ENV || "development",
    applicationVersion: packageJson.version,
    weeklyFocus
  });
  const unrelatedNeedsReview = brokerDiagnostic?.needsReviewRows.filter((row) => !row.affectsRequestedSnapshot) || [];
  if (unrelatedNeedsReview.length) {
    const sample = unrelatedNeedsReview.slice(0, DIAGNOSTIC_SAMPLE_LIMIT).map((row) => `${row.ticker} (${row.tradeId})`).join(", ");
    const warning: SnapshotWarning = {
      code: "BROKER_IMPORT_UNRELATED_ROWS_NEED_REVIEW",
      severity: "warning",
      message: `${unrelatedNeedsReview.length} imported row${unrelatedNeedsReview.length === 1 ? "" : "s"} still require review but do not affect ${session.resolved}: ${sample}.`
    };
    snapshot.warnings.unshift(warning);
  }
  if (session.adjusted) {
    snapshot.warnings.unshift({
      code: "REQUESTED_SESSION_ADJUSTED",
      severity: "warning",
      message: `Requested session ${session.requested} was adjusted to completed U.S. session ${session.resolved}.`
    });
  }
  snapshot.snapshot_status = snapshotStatusFromWarnings(snapshot.warnings);
  snapshot.critical_warning_count = snapshot.warnings.filter((item) => item.severity === "critical").length;
  const invalidPrices = snapshot.open_positions
    .filter((position) => position.current_price === null || position.current_price_session !== session.resolved)
    .map((position) => position.ticker);
  if (invalidPrices.length) {
    throw new SnapshotValidationError(
      "CURRENT_PRICES_INVALID",
      `Snapshot not generated: exact-session prices for ${session.resolved} are missing or stale for ${invalidPrices.join(", ")}.`
    );
  }
  const markdown = renderDailyPortfolioSnapshotMarkdown(snapshot);
  const baseName = `daily-portfolio-snapshot-${session.resolved}`;
  const outputDirectory = path.resolve(options.outputDirectory || path.join(process.cwd(), "data", "exports", "daily-portfolio-snapshots"));
  const jsonPath = path.join(outputDirectory, `${baseName}.json`);
  const markdownPath = path.join(outputDirectory, `${baseName}.md`);
  if (options.writeExports !== false) {
    await Promise.all([
      atomicWrite(jsonPath, `${JSON.stringify(snapshot, null, 2)}\n`),
      atomicWrite(markdownPath, markdown)
    ]);
  }
  return {
    snapshot,
    markdown,
    jsonPath,
    markdownPath,
    baseName,
    brokerDiagnostic: brokerDiagnostic || undefined,
    datePath: {
      selectedDate: options.session,
      submittedDate: session.requested,
      evaluatedDate: session.resolved,
      latestCompletedSession: session.latestCompleted
    }
  };
}
