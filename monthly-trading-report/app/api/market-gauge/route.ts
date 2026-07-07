import { NextResponse } from "next/server";
import { createApiTimer } from "@/lib/apiTiming";

type Candle = {
  time: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume?: number;
};
type TrendState = "Down" | "Neutral" | "Up";
type GaugeState = "Protect" | "Neutral" | "Grow";
type ExtensionState = "Normal" | "Caution" | "Extended";
type SymbolRegime = {
  symbol: string;
  close: number;
  date: string;
  ema21: number;
  sma50: number;
  sma200: number;
  shortTerm: TrendState;
  mediumTerm: TrendState;
  longTerm: TrendState;
  rawShortTerm: TrendState;
  rawMediumTerm: TrendState;
  rawLongTerm: TrendState;
  above21Percent: number;
  above50Percent: number;
  extension: ExtensionState;
};
type GaugeComponent = {
  label: string;
  detail: string;
  state: GaugeState;
  previousState?: GaugeState;
  pendingState?: GaugeState;
};
type RegimePair = {
  current: SymbolRegime | null;
  previous: SymbolRegime | null;
};

const indexSymbols = ["SPY", "QQQ", "IWM"];
const defaultLeaderWatchlist = ["NVDA", "MSFT", "META", "AMZN", "GOOGL", "AVGO", "TSLA", "AMD", "PLTR", "CRWD", "COIN", "APP"];

function numberValue(value: string) {
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function average(values: number[]) {
  return values.length ? values.reduce((total, value) => total + value, 0) / values.length : 0;
}

function sma(values: number[], period: number) {
  return values.length < period ? 0 : average(values.slice(-period));
}

function ema(values: number[], period: number) {
  if (values.length < period) return 0;
  const multiplier = 2 / (period + 1);
  let current = average(values.slice(0, period));
  for (let index = period; index < values.length; index += 1) {
    current = (values[index] - current) * multiplier + current;
  }
  return current;
}

function percentAbove(value: number, basis: number) {
  return basis ? ((value - basis) / basis) * 100 : 0;
}

function trendState(close: number, averageValue: number): TrendState {
  if (!averageValue) return "Neutral";
  const distance = percentAbove(close, averageValue);
  if (distance > 0.5) return "Up";
  if (distance < -0.5) return "Down";
  return "Neutral";
}

function movingAverageSideState(close: number, averageValue: number): TrendState {
  if (!averageValue) return "Neutral";
  if (close > averageValue) return "Up";
  if (close < averageValue) return "Down";
  return "Neutral";
}

function confirmedTrendState(
  close: number,
  averageValue: number,
  previousClose: number | undefined,
  previousAverageValue: number
): TrendState {
  if (!averageValue || !previousAverageValue || !Number.isFinite(previousClose)) return movingAverageSideState(close, averageValue);
  const priorClose = previousClose as number;
  if (close > averageValue && priorClose > previousAverageValue) return "Up";
  if (close < averageValue && priorClose < previousAverageValue) return "Down";
  return "Neutral";
}

function extensionState(above21Percent: number, above50Percent: number): ExtensionState {
  if (above21Percent >= 8 || above50Percent >= 15) return "Extended";
  if (above21Percent >= 4 || above50Percent >= 8) return "Caution";
  return "Normal";
}

function gaugeStateFromScore(score: number): GaugeState {
  if (score >= 67) return "Grow";
  if (score >= 40) return "Neutral";
  return "Protect";
}

function scoreTrend(state: TrendState) {
  if (state === "Up") return 100;
  if (state === "Neutral") return 50;
  return 0;
}

function scoreExtension(state: ExtensionState) {
  if (state === "Extended") return 20;
  if (state === "Caution") return 55;
  return 85;
}

function parseStooqCsv(csv: string): Candle[] {
  const candles: Candle[] = [];
  csv
    .trim()
    .split(/\r?\n/)
    .slice(1)
    .forEach((line) => {
      const [date, openRaw, highRaw, lowRaw, closeRaw, volumeRaw] = line.split(",");
      const open = numberValue(openRaw);
      const high = numberValue(highRaw);
      const low = numberValue(lowRaw);
      const close = numberValue(closeRaw);
      const volume = numberValue(volumeRaw);

      if (!/^\d{4}-\d{2}-\d{2}$/.test(date) || open === null || high === null || low === null || close === null) return;
      candles.push({ time: date, open, high, low, close, volume: volume || 0 });
    });
  return candles;
}

function parseYahooResponse(payload: unknown): Candle[] {
  const result =
    payload &&
    typeof payload === "object" &&
    "chart" in payload &&
    (payload as { chart?: { result?: unknown[] } }).chart?.result?.[0];

  if (!result || typeof result !== "object") return [];

  const quote = (result as { indicators?: { quote?: unknown[] } }).indicators?.quote?.[0] as
    | { open?: unknown[]; high?: unknown[]; low?: unknown[]; close?: unknown[]; volume?: unknown[] }
    | undefined;
  const timestamps = (result as { timestamp?: unknown[] }).timestamp || [];

  if (!quote) return [];

  return timestamps
    .map((timestamp, index) => {
      const unixTimestamp = Number(timestamp);
      const open = Number(quote.open?.[index]);
      const high = Number(quote.high?.[index]);
      const low = Number(quote.low?.[index]);
      const close = Number(quote.close?.[index]);
      const volume = Number(quote.volume?.[index] || 0);

      if (![unixTimestamp, open, high, low, close].every(Number.isFinite)) return null;
      return {
        time: new Date(unixTimestamp * 1000).toISOString().slice(0, 10),
        open,
        high,
        low,
        close,
        volume: Number.isFinite(volume) ? volume : 0
      };
    })
    .filter(Boolean) as Candle[];
}

async function fetchCandles(symbol: string) {
  const stooqSymbol = `${symbol.toLowerCase()}.us`;
  const stooqResponse = await fetch(`https://stooq.com/q/d/l/?s=${encodeURIComponent(stooqSymbol)}&i=d`, {
    next: { revalidate: 60 * 60 * 6 }
  });

  if (stooqResponse.ok) {
    const candles = parseStooqCsv(await stooqResponse.text());
    if (candles.length) return candles.slice(-520);
  }

  const now = Math.floor(Date.now() / 1000);
  const start = now - 60 * 60 * 24 * 730;
  const yahooResponse = await fetch(
    `https://query1.finance.yahoo.com/v8/finance/chart/${encodeURIComponent(symbol)}?period1=${start}&period2=${now}&interval=1d&events=history`,
    { next: { revalidate: 60 * 60 * 6 } }
  );

  if (!yahooResponse.ok) return [];
  return parseYahooResponse(await yahooResponse.json()).slice(-520);
}

function buildRegimeAtIndex(symbol: string, candles: Candle[], candleIndex: number): SymbolRegime | null {
  if (candleIndex < 0) return null;

  const candleWindow = candles.slice(0, candleIndex + 1);
  const closes = candleWindow.map((candle) => candle.close).filter(Number.isFinite);
  const latest = candleWindow[candleWindow.length - 1];

  if (!latest || closes.length < 200) return null;

  const ema21 = ema(closes, 21);
  const sma50 = sma(closes, 50);
  const sma200 = sma(closes, 200);
  const previousCloses = closes.slice(0, -1);
  const previousClose = previousCloses[previousCloses.length - 1];
  const previousEma21 = ema(previousCloses, 21);
  const previousSma50 = sma(previousCloses, 50);
  const previousSma200 = sma(previousCloses, 200);
  const above21Percent = percentAbove(latest.close, ema21);
  const above50Percent = percentAbove(latest.close, sma50);

  return {
    symbol,
    close: latest.close,
    date: latest.time,
    ema21,
    sma50,
    sma200,
    shortTerm: confirmedTrendState(latest.close, ema21, previousClose, previousEma21),
    mediumTerm: confirmedTrendState(latest.close, sma50, previousClose, previousSma50),
    longTerm: confirmedTrendState(latest.close, sma200, previousClose, previousSma200),
    rawShortTerm: movingAverageSideState(latest.close, ema21),
    rawMediumTerm: movingAverageSideState(latest.close, sma50),
    rawLongTerm: movingAverageSideState(latest.close, sma200),
    above21Percent,
    above50Percent,
    extension: extensionState(above21Percent, above50Percent)
  };
}

function buildRegime(symbol: string, candles: Candle[]): SymbolRegime | null {
  return buildRegimeAtIndex(symbol, candles, candles.length - 1);
}

function buildRegimePair(symbol: string, candles: Candle[]): RegimePair {
  return {
    current: buildRegimeAtIndex(symbol, candles, candles.length - 1),
    previous: buildRegimeAtIndex(symbol, candles, candles.length - 2)
  };
}

function componentWithPrevious(
  label: string,
  detail: string,
  state: GaugeState,
  previousState?: GaugeState,
  pendingState?: GaugeState
): GaugeComponent {
  return {
    label,
    detail,
    state,
    ...(previousState && previousState !== state ? { previousState } : {}),
    ...(pendingState && pendingState !== state ? { pendingState } : {})
  };
}

function buildComponents(
  indexRegimes: SymbolRegime[],
  leaderRegimes: SymbolRegime[],
  previousIndexRegimes: SymbolRegime[],
  previousLeaderRegimes: SymbolRegime[]
) {
  const above21 = leaderRegimes.filter((item) => item.shortTerm === "Up").length;
  const percentAbove21 = leaderRegimes.length ? (above21 / leaderRegimes.length) * 100 : 0;
  const leadershipState = gaugeStateFromScore(percentAbove21);
  const previousAbove21 = previousLeaderRegimes.filter((item) => item.shortTerm === "Up").length;
  const previousPercentAbove21 = previousLeaderRegimes.length ? (previousAbove21 / previousLeaderRegimes.length) * 100 : 0;
  const previousLeadershipState = previousLeaderRegimes.length ? gaugeStateFromScore(previousPercentAbove21) : undefined;
  const shortScore = average(indexRegimes.map((item) => scoreTrend(item.shortTerm)));
  const mediumScore = average(indexRegimes.map((item) => scoreTrend(item.mediumTerm)));
  const longScore = average(indexRegimes.map((item) => scoreTrend(item.longTerm)));
  const rawShortScore = average(indexRegimes.map((item) => scoreTrend(item.rawShortTerm)));
  const rawMediumScore = average(indexRegimes.map((item) => scoreTrend(item.rawMediumTerm)));
  const rawLongScore = average(indexRegimes.map((item) => scoreTrend(item.rawLongTerm)));
  const extensionScore = average(indexRegimes.map((item) => scoreExtension(item.extension)));
  const previousShortScore = average(previousIndexRegimes.map((item) => scoreTrend(item.shortTerm)));
  const previousMediumScore = average(previousIndexRegimes.map((item) => scoreTrend(item.mediumTerm)));
  const previousLongScore = average(previousIndexRegimes.map((item) => scoreTrend(item.longTerm)));
  const previousExtensionScore = average(previousIndexRegimes.map((item) => scoreExtension(item.extension)));
  const hasPreviousIndexes = previousIndexRegimes.length > 0;
  const components = [
    componentWithPrevious(
      "Market leaders",
      `${above21}/${leaderRegimes.length} leaders above 21EMA`,
      leadershipState,
      previousLeadershipState
    ),
    componentWithPrevious(
      "Short term",
      "Indexes vs 21EMA",
      gaugeStateFromScore(shortScore),
      hasPreviousIndexes ? gaugeStateFromScore(previousShortScore) : undefined,
      gaugeStateFromScore(rawShortScore)
    ),
    componentWithPrevious(
      "Medium term",
      "Indexes vs 50SMA",
      gaugeStateFromScore(mediumScore),
      hasPreviousIndexes ? gaugeStateFromScore(previousMediumScore) : undefined,
      gaugeStateFromScore(rawMediumScore)
    ),
    componentWithPrevious(
      "Long term",
      "Indexes vs 200SMA",
      gaugeStateFromScore(longScore),
      hasPreviousIndexes ? gaugeStateFromScore(previousLongScore) : undefined,
      gaugeStateFromScore(rawLongScore)
    ),
    componentWithPrevious(
      "Extension warning",
      "% above 21EMA / 50SMA",
      gaugeStateFromScore(extensionScore),
      hasPreviousIndexes ? gaugeStateFromScore(previousExtensionScore) : undefined
    )
  ];

  return { above21, percentAbove21, leadershipState, components };
}

export async function GET() {
  const logTiming = createApiTimer("/api/market-gauge");
  try {
    const indexPairs = await Promise.all(indexSymbols.map(async (symbol) => buildRegimePair(symbol, await fetchCandles(symbol))));
    const leaderResults = await Promise.allSettled(
      defaultLeaderWatchlist.map(async (symbol) => buildRegimePair(symbol, await fetchCandles(symbol)))
    );
    const indexRegimes = indexPairs.map((item) => item.current).filter((item): item is SymbolRegime => Boolean(item));
    const previousIndexRegimes = indexPairs.map((item) => item.previous).filter((item): item is SymbolRegime => Boolean(item));
    const leaderPairs = leaderResults
      .filter((result): result is PromiseFulfilledResult<RegimePair> => result.status === "fulfilled")
      .map((result) => result.value)
      .filter((item) => item.current);
    const leaderRegimes = leaderPairs.map((item) => item.current).filter((item): item is SymbolRegime => Boolean(item));
    const previousLeaderRegimes = leaderPairs.map((item) => item.previous).filter((item): item is SymbolRegime => Boolean(item));
    const { above21, percentAbove21, leadershipState, components } = buildComponents(
      indexRegimes,
      leaderRegimes,
      previousIndexRegimes,
      previousLeaderRegimes
    );
    const overallState = gaugeStateFromScore(
      average(
        components.map((component) => {
          if (component.state === "Grow") return 100;
          if (component.state === "Neutral") return 50;
          return 0;
        })
      )
    );
    logTiming(200, { indexRegimes: indexRegimes.length, leaderRegimes: leaderRegimes.length, overallState });

    return NextResponse.json(
      {
        overallState,
        leadership: { above21, total: leaderRegimes.length, percentAbove21, state: leadershipState },
        components,
        indexRegimes,
        leaderRegimes: leaderRegimes.map((item) => ({ symbol: item.symbol, shortTerm: item.shortTerm }))
      },
      {
        headers: {
          "Cache-Control": "public, s-maxage=900, stale-while-revalidate=3600"
        }
      }
    );
  } catch (error) {
    logTiming(500, { error: error instanceof Error ? error.message : "unknown" });
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Could not load market gauge." },
      { status: 500 }
    );
  }
}
