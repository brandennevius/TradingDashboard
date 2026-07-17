import { mkdir, rename, writeFile } from "node:fs/promises";
import path from "node:path";
import packageJson from "../package.json";
import {
  buildDailyPortfolioSnapshot,
  renderDailyPortfolioSnapshotMarkdown,
  resolveSnapshotSession,
  type SnapshotPrice
} from "./daily-portfolio-snapshot";
import { getMarketCandlesWithProvider } from "./market-data";
import { getBrandenPortfolioSettings, listBrandenVisibleTrades } from "./store";
import type { TradeLogEntry } from "./types";

type SnapshotDependencies = {
  loadTrades: () => Promise<TradeLogEntry[]>;
  loadPortfolioSettings: typeof getBrandenPortfolioSettings;
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
  constructor(public readonly code: "PORTFOLIO_UNRESOLVED" | "BROKER_IMPORT_INCOMPLETE" | "CURRENT_PRICES_INVALID", message: string) {
    super(message);
    this.name = "SnapshotValidationError";
  }
}

async function loadSessionPrice(symbol: string, session: string): Promise<SnapshotPrice> {
  const result = await getMarketCandlesWithProvider(symbol, "1d");
  const eligible = result.candles.filter((candle) => candle.time <= session);
  const candle = eligible.at(-1);
  return {
    symbol,
    price: candle?.close ?? null,
    timestamp: candle?.time ?? null,
    provider: result.provider
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
    loadPrice: loadSessionPrice,
    now: () => new Date(),
    ...options.dependencies
  };
  const now = dependencies.now();
  const session = resolveSnapshotSession(options.session, now);
  const [trades, portfolioSettings] = await Promise.all([
    dependencies.loadTrades(),
    dependencies.loadPortfolioSettings()
  ]);
  const accountName = String(options.accountName || portfolioSettings.defaultPortfolio || "").trim();
  if (!accountName) throw new SnapshotValidationError("PORTFOLIO_UNRESOLVED", "No portfolio is selected and no default portfolio is configured.");
  if (portfolioSettings.portfolios.length && !portfolioSettings.portfolios.includes(accountName)) {
    throw new SnapshotValidationError("PORTFOLIO_UNRESOLVED", `Portfolio ${accountName} could not be resolved.`);
  }
  const accountTrades = trades.filter((trade) => !trade.hidden && trade.portfolioTag === accountName);
  const openSymbols = Array.from(new Set(accountTrades.filter((trade) => trade.status === "OPEN").map((trade) => trade.symbol))).sort();
  const loadedPrices = await mapWithConcurrency(openSymbols, 4, (symbol) => dependencies.loadPrice(symbol, session.resolved));
  const prices = new Map(loadedPrices.map((price) => [price.symbol, price]));
  const snapshot = buildDailyPortfolioSnapshot({
    requestedSession: session.resolved,
    latestCompletedMarketSession: session.latestCompleted,
    generatedAt: now.toISOString(),
    accountName,
    portfolioMeta: portfolioSettings.portfolioMeta?.[accountName],
    trades: accountTrades,
    // The existing template getter can perform a screenshot migration. A snapshot
    // must not trigger that write, so only trade-stored manual grades/checklists
    // are used until the repository exposes a strictly read-only template query.
    setupTemplates: [],
    prices,
    sourceEnvironment: process.env.NODE_ENV || "development",
    applicationVersion: packageJson.version
  });
  if (session.adjusted) {
    snapshot.warnings.unshift({
      code: "REQUESTED_SESSION_ADJUSTED",
      severity: "warning",
      message: `Requested session ${session.requested} was adjusted to completed U.S. session ${session.resolved}.`
    });
    snapshot.snapshot_status = snapshot.snapshot_status === "COMPLETE" ? "COMPLETE_WITH_WARNINGS" : snapshot.snapshot_status;
  }
  if (!snapshot.metadata.broker_import_complete) {
    const brokerReasons = snapshot.warnings
      .filter((item) => item.code === "BROKER_IMPORT_MISSING" || item.code === "BROKER_IMPORT_STALE" || item.code === "BROKER_IMPORT_INCOMPLETE")
      .map((item) => item.message)
      .join(" ");
    throw new SnapshotValidationError(
      "BROKER_IMPORT_INCOMPLETE",
      `Snapshot not generated: ${brokerReasons || `the broker import for ${accountName} is missing, stale, or incomplete for ${session.resolved}.`}`
    );
  }
  const invalidPrices = snapshot.open_positions
    .filter((position) => position.current_price === null || position.current_price_timestamp !== session.resolved)
    .map((position) => position.ticker);
  if (invalidPrices.length) {
    throw new SnapshotValidationError(
      "CURRENT_PRICES_INVALID",
      `Snapshot not generated: current-session prices are missing or stale for ${invalidPrices.join(", ")}.`
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
  return { snapshot, markdown, jsonPath, markdownPath, baseName };
}
