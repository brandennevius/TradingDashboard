import { NextResponse } from "next/server";
import { getSessionUser } from "@/lib/auth";
import { analyzeTradeExit, tradeExitAnalysisEligibility, type TradeExitAnalysis } from "@/lib/exit-analysis";
import { getMarketCandles } from "@/lib/market-data";
import { listBrandenExitAnalysisTrades } from "@/lib/store";
import type { TradeLogEntry } from "@/lib/types";

export const maxDuration = 60;

type StrategySummary = {
  strategy: string;
  category: string;
  trades: number;
  totalR: number;
  averageR: number;
  medianR: number;
  winRate: number;
  maxDrawdownR: number;
  markedOpen: number;
  deltaVsActualR: number;
  earlierDeltaVsActualR: number;
  recentDeltaVsActualR: number;
  earlierAverageR: number;
  recentAverageR: number;
  performanceToDrawdown: number;
};

type StrategyRecommendation = {
  suggested: StrategySummary | null;
  bestRaw: StrategySummary | null;
  bestRiskAdjusted: StrategySummary | null;
  rationale: string;
};

function cleanValue(value: string | null) {
  return String(value || "").trim();
}

function cleanSymbol(value: string) {
  return value.trim().replace(/^#/, "").toUpperCase();
}

function supportedSymbol(value: string) {
  return /^[A-Z][A-Z0-9.-]*$/.test(cleanSymbol(value));
}

function primarySetup(trade: TradeLogEntry) {
  return trade.setupTags[0] || "No setup";
}

function uniqueSorted(values: string[]) {
  return Array.from(new Set(values.map((value) => value.trim()).filter(Boolean))).sort((a, b) => a.localeCompare(b));
}

function median(values: number[]) {
  const sorted = [...values].sort((a, b) => a - b);
  if (!sorted.length) return 0;
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

function strategyCategory(strategy: string) {
  if (strategy === "Actual") return "Actual";
  if (strategy.includes(" + ")) return "Hybrid";
  if (strategy.includes("EMA")) return "Moving average";
  if (strategy.includes("session")) return "Time stop";
  if (strategy.includes("trailing")) return "Trailing stop";
  return "Target";
}

function sum(values: number[]) {
  return values.reduce((total, value) => total + value, 0);
}

function average(values: number[]) {
  return values.length ? sum(values) / values.length : 0;
}

function summarizeStrategies(items: { trade: TradeLogEntry; analysis: TradeExitAnalysis }[]) {
  const chronological = [...items].sort((a, b) => a.analysis.finalExitDate.localeCompare(b.analysis.finalExitDate));
  const splitIndex = Math.max(1, Math.floor(chronological.length / 2));
  const earlierItems = chronological.slice(0, splitIndex);
  const recentItems = chronological.slice(splitIndex);
  const strategyNames = items[0]?.analysis.strategies.map((strategy) => strategy.name) || [];
  const summaries: StrategySummary[] = strategyNames.map((strategyName) => {
    const results = chronological.map(({ analysis }) => analysis.strategies.find((strategy) => strategy.name === strategyName)!);
    const values = results.map((result) => result.resultR);
    let equity = 0;
    let peak = 0;
    let maxDrawdownR = 0;

    results.forEach((result) => {
      equity += result.resultR;
      peak = Math.max(peak, equity);
      maxDrawdownR = Math.min(maxDrawdownR, equity - peak);
    });

    const earlierValues = earlierItems.map(({ analysis }) =>
      analysis.strategies.find((strategy) => strategy.name === strategyName)!.resultR
    );
    const recentValues = recentItems.map(({ analysis }) =>
      analysis.strategies.find((strategy) => strategy.name === strategyName)!.resultR
    );
    const totalR = sum(values);

    return {
      strategy: strategyName,
      category: strategyCategory(strategyName),
      trades: values.length,
      totalR,
      averageR: average(values),
      medianR: median(values),
      winRate: values.filter((value) => value > 0.05).length / values.length,
      maxDrawdownR,
      markedOpen: results.filter((result) => result.outcome === "marked").length,
      deltaVsActualR: 0,
      earlierDeltaVsActualR: 0,
      recentDeltaVsActualR: 0,
      earlierAverageR: average(earlierValues),
      recentAverageR: average(recentValues),
      performanceToDrawdown: totalR > 0 ? totalR / Math.max(0.25, Math.abs(maxDrawdownR)) : 0
    };
  });

  const actual = summaries.find((summary) => summary.strategy === "Actual");
  const actualTotalR = actual?.totalR || 0;
  summaries.forEach((summary) => {
    summary.deltaVsActualR = summary.totalR - actualTotalR;
    summary.earlierDeltaVsActualR = (summary.earlierAverageR - (actual?.earlierAverageR || 0)) * earlierItems.length;
    summary.recentDeltaVsActualR = (summary.recentAverageR - (actual?.recentAverageR || 0)) * recentItems.length;
  });

  const alternatives = summaries.filter((summary) => summary.strategy !== "Actual");
  const resolvedEnough = alternatives.filter(
    (summary) => summary.trades > 0 && summary.markedOpen / summary.trades <= 0.2
  );
  const bestRaw = [...alternatives].sort((a, b) => b.totalR - a.totalR)[0] || null;
  const bestRiskAdjusted = [...resolvedEnough]
    .filter((summary) => summary.totalR > 0)
    .sort((a, b) => b.performanceToDrawdown - a.performanceToDrawdown || b.totalR - a.totalR)[0] || null;
  const robustCandidates = resolvedEnough
    .filter((summary) =>
      summary.trades >= 10 &&
      summary.totalR > actualTotalR &&
      summary.earlierDeltaVsActualR >= 0 &&
      summary.recentDeltaVsActualR >= 0
    )
    .sort((a, b) =>
      Math.min(b.earlierAverageR, b.recentAverageR) - Math.min(a.earlierAverageR, a.recentAverageR) ||
      b.performanceToDrawdown - a.performanceToDrawdown ||
      b.totalR - a.totalR
    );
  const suggested = robustCandidates[0] || null;
  const recommendation: StrategyRecommendation = {
    suggested,
    bestRaw,
    bestRiskAdjusted,
    rationale: suggested
      ? `${suggested.strategy} beat the actual exits in both chronological sample halves, improved total R by ${suggested.deltaVsActualR.toFixed(2)}R, and left ${suggested.markedOpen} simulations unresolved.`
      : "No alternative beat the actual exits in both chronological sample halves with at least 10 trades and no more than 20% unresolved simulations. Keep the current exits as the baseline and forward-test alternatives before changing rules."
  };

  return { summaries, recommendation };
}

async function loadCandles(symbols: string[]) {
  const candles = new Map<string, Awaited<ReturnType<typeof getMarketCandles>>>();
  const failed = new Set<string>();

  for (let index = 0; index < symbols.length; index += 6) {
    const batch = symbols.slice(index, index + 6);
    const results = await Promise.all(
      batch.map(async (symbol) => ({ symbol, values: await getMarketCandles(symbol, "1d").catch(() => []) }))
    );

    results.forEach(({ symbol, values }) => {
      if (values.length) candles.set(symbol, values);
      else failed.add(symbol);
    });
  }

  return { candles, failed };
}

export async function GET(request: Request) {
  const user = await getSessionUser();
  if (!user) {
    return NextResponse.json({ error: "Unauthorized." }, { status: 401 });
  }

  try {
    const url = new URL(request.url);
    const startDate = cleanValue(url.searchParams.get("start"));
    const endDate = cleanValue(url.searchParams.get("end"));
    const setup = cleanValue(url.searchParams.get("setup"));
    const portfolio = cleanValue(url.searchParams.get("portfolio"));
    const allClosedTrades = await listBrandenExitAnalysisTrades();
    const setupOptions = uniqueSorted(allClosedTrades.map(primarySetup));
    const portfolioOptions = uniqueSorted(allClosedTrades.map((trade) => trade.portfolioTag));
    const filteredTrades = allClosedTrades.filter((trade) => {
      const exitDate = trade.exitDate;
      if (startDate && exitDate < startDate) return false;
      if (endDate && exitDate > endDate) return false;
      if (setup && primarySetup(trade) !== setup) return false;
      if (portfolio && trade.portfolioTag !== portfolio) return false;
      return true;
    });

    const missingRisk = filteredTrades.filter((trade) => tradeExitAnalysisEligibility(trade).reason === "missing-risk").length;
    const incompleteExecutions = filteredTrades.filter(
      (trade) => tradeExitAnalysisEligibility(trade).reason === "incomplete-executions"
    ).length;
    const eligibleTrades = filteredTrades.filter(
      (trade) => tradeExitAnalysisEligibility(trade).eligible && supportedSymbol(trade.symbol)
    );
    const unsupportedSymbols = filteredTrades.filter(
      (trade) => tradeExitAnalysisEligibility(trade).eligible && !supportedSymbol(trade.symbol)
    ).length;
    const symbols = uniqueSorted(eligibleTrades.map((trade) => cleanSymbol(trade.symbol)));
    const marketData = await loadCandles(symbols);
    const analyzed = eligibleTrades
      .map((trade) => {
        const candles = marketData.candles.get(cleanSymbol(trade.symbol));
        const analysis = candles ? analyzeTradeExit(trade, candles) : null;
        return analysis ? { trade, analysis } : null;
      })
      .filter((item): item is { trade: TradeLogEntry; analysis: TradeExitAnalysis } => Boolean(item));
    const noMarketData = eligibleTrades.filter((trade) => marketData.failed.has(cleanSymbol(trade.symbol))).length;
    const strategyAnalysis = summarizeStrategies(analyzed);
    const summaries = strategyAnalysis.summaries;
    const measurableExcursions = analyzed.filter(({ analysis }) => analysis.mfeR !== null && analysis.maeR !== null);
    const winnerCaptureRates = measurableExcursions
      .filter(({ analysis }) => analysis.actualR > 0 && analysis.captureRate !== null)
      .map(({ analysis }) => Math.max(0, Math.min(1, analysis.captureRate || 0)));
    const actualTotalR = summaries.find((summary) => summary.strategy === "Actual")?.totalR || 0;
    const reached3R = analyzed.filter(({ analysis }) => (analysis.mfeR || 0) >= 3);
    const halfAt3RTotal = analyzed.reduce((sum, { analysis }) => {
      return sum + ((analysis.mfeR || 0) >= 3 ? 0.5 * 3 + 0.5 * analysis.actualR : analysis.actualR);
    }, 0);

    return NextResponse.json({
      user,
      filters: { startDate, endDate, setup, portfolio },
      options: { setups: setupOptions, portfolios: portfolioOptions },
      coverage: {
        closedTrades: filteredTrades.length,
        analyzedTrades: analyzed.length,
        missingRisk,
        incompleteExecutions,
        unsupportedSymbols,
        noMarketData
      },
      metrics: {
        actualTotalR,
        averageMfeR: average(measurableExcursions.map(({ analysis }) => analysis.mfeR || 0)),
        averageMaeR: average(measurableExcursions.map(({ analysis }) => analysis.maeR || 0)),
        averageGivebackR: average(measurableExcursions.map(({ analysis }) => analysis.givebackR || 0)),
        averageCaptureRate: average(winnerCaptureRates),
        reached2R: measurableExcursions.filter(({ analysis }) => (analysis.mfeR || 0) >= 2).length,
        reached3R: reached3R.length,
        highGiveback: measurableExcursions.filter(
          ({ analysis }) => (analysis.mfeR || 0) >= 2 && analysis.actualR < 1
        ).length,
        halfAt3RTotal,
        halfAt3RDelta: halfAt3RTotal - actualTotalR
      },
      summaries,
      recommendation: strategyAnalysis.recommendation
    });
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Could not calculate aggregate exit analysis." },
      { status: 500 }
    );
  }
}
