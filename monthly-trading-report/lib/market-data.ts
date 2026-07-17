export type MarketCandle = {
  time: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume?: number;
};

export type MarketTimeframe = "1h" | "4h" | "1d" | "1wk" | "1mo";
export type MarketDataProvider = "stooq" | "yahoo" | "unavailable";

export function cleanMarketSymbol(value: string) {
  return value
    .trim()
    .replace(/^#/, "")
    .replace(/[^a-zA-Z0-9.^=-]/g, "")
    .toUpperCase();
}

function numberValue(value: string) {
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function parseStooqCsv(csv: string): MarketCandle[] {
  const candles: MarketCandle[] = [];

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

    if (!/^\d{4}-\d{2}-\d{2}$/.test(date) || open === null || high === null || low === null || close === null || open <= 0 || high <= 0 || low <= 0 || close <= 0) {
        return;
      }

      candles.push({ time: date, open, high, low, close, volume: volume || 0 });
    });

  return candles;
}

export function marketTimeframe(value: string | null): MarketTimeframe {
  return value === "1h" || value === "4h" || value === "1d" || value === "1wk" || value === "1mo" ? value : "1d";
}

function yahooInterval(timeframe: MarketTimeframe) {
  return timeframe === "4h" ? "1h" : timeframe;
}

function lookbackSeconds(timeframe: MarketTimeframe) {
  if (timeframe === "1h" || timeframe === "4h") {
    return 60 * 60 * 24 * 120;
  }

  if (timeframe === "1mo") {
    return 60 * 60 * 24 * 365 * 8;
  }

  return 60 * 60 * 24 * 730;
}

function parseYahooResponse(payload: unknown, timeframe: MarketTimeframe): MarketCandle[] {
  const result =
    payload &&
    typeof payload === "object" &&
    "chart" in payload &&
    (payload as { chart?: { result?: unknown[] } }).chart?.result?.[0];

  if (!result || typeof result !== "object") {
    return [];
  }

  const quote = (result as { indicators?: { quote?: unknown[] } }).indicators?.quote?.[0] as
    | { open?: unknown[]; high?: unknown[]; low?: unknown[]; close?: unknown[]; volume?: unknown[] }
    | undefined;
  const timestamps = (result as { timestamp?: unknown[] }).timestamp || [];

  if (!quote) {
    return [];
  }

  const candles: MarketCandle[] = [];

  timestamps.forEach((timestamp, index) => {
    const unixTimestamp = Number(timestamp);
    const open = Number(quote.open?.[index]);
    const high = Number(quote.high?.[index]);
    const low = Number(quote.low?.[index]);
    const close = Number(quote.close?.[index]);
    const volume = Number(quote.volume?.[index] || 0);

    if (![unixTimestamp, open, high, low, close].every(Number.isFinite) || open <= 0 || high <= 0 || low <= 0 || close <= 0) {
      return;
    }

    candles.push({
      time:
        timeframe === "1h" || timeframe === "4h"
          ? new Date(unixTimestamp * 1000).toISOString().replace(".000Z", "Z")
          : new Date(unixTimestamp * 1000).toISOString().slice(0, 10),
      open,
      high,
      low,
      close,
      volume: Number.isFinite(volume) ? volume : 0
    });
  });

  return candles;
}

function aggregateFourHourCandles(candles: MarketCandle[]) {
  const groups = new Map<string, MarketCandle[]>();

  candles.forEach((candle) => {
    const date = new Date(candle.time);
    const hour = date.getUTCHours();
    date.setUTCMinutes(0, 0, 0);
    date.setUTCHours(Math.floor(hour / 4) * 4);
    const key = date.toISOString().replace(".000Z", "Z");
    groups.set(key, [...(groups.get(key) || []), candle]);
  });

  return Array.from(groups.entries())
    .map(([time, group]) => ({
      time,
      open: group[0].open,
      high: Math.max(...group.map((candle) => candle.high)),
      low: Math.min(...group.map((candle) => candle.low)),
      close: group[group.length - 1].close,
      volume: group.reduce((total, candle) => total + (candle.volume || 0), 0)
    }))
    .sort((a, b) => a.time.localeCompare(b.time));
}

async function fetchStooqCandles(symbol: string, timeframe: MarketTimeframe) {
  if (timeframe !== "1d") {
    return [];
  }
  if (symbol.startsWith("^")) return [];

  const stooqSymbol = symbol.includes(".") || symbol.includes("=") ? symbol.toLowerCase() : `${symbol.toLowerCase()}.us`;
  const response = await fetch(`https://stooq.com/q/d/l/?s=${encodeURIComponent(stooqSymbol)}&i=d`, {
    next: { revalidate: 60 * 60 * 6 }
  });

  if (!response.ok) {
    return [];
  }

  return parseStooqCsv(await response.text());
}

export async function getYahooMarketCandles(symbolValue: string, timeframe: MarketTimeframe = "1d") {
  const symbol = cleanMarketSymbol(symbolValue);
  if (!symbol) return [];
  const now = Math.floor(Date.now() / 1000);
  const start = now - lookbackSeconds(timeframe);
  const response = await fetch(
    `https://query1.finance.yahoo.com/v8/finance/chart/${encodeURIComponent(symbol)}?period1=${start}&period2=${now}&interval=${yahooInterval(timeframe)}&events=history`,
    { next: { revalidate: 60 * 60 * 6 } }
  );

  if (!response.ok) {
    return [];
  }

  const candles = parseYahooResponse(await response.json(), timeframe);
  return timeframe === "4h" ? aggregateFourHourCandles(candles) : candles;
}

export async function getMarketCandlesWithProvider(symbolValue: string, timeframe: MarketTimeframe) {
  const symbol = cleanMarketSymbol(symbolValue);
  if (!symbol) {
    return { symbol, timeframe, provider: "unavailable" as MarketDataProvider, candles: [] as MarketCandle[] };
  }

  const stooqCandles = await fetchStooqCandles(symbol, timeframe).catch(() => []);
  const yahooCandles = stooqCandles.length ? [] : await getYahooMarketCandles(symbol, timeframe).catch(() => []);
  const candles = (stooqCandles.length ? stooqCandles : yahooCandles).slice(-520);
  const provider: MarketDataProvider = stooqCandles.length ? "stooq" : yahooCandles.length ? "yahoo" : "unavailable";
  return { symbol, timeframe, provider, candles };
}

export async function getMarketCandles(symbolValue: string, timeframe: MarketTimeframe) {
  return (await getMarketCandlesWithProvider(symbolValue, timeframe)).candles;
}
