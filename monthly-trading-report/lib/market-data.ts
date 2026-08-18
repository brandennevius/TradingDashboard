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

export type ExactMarketSessionPrice = {
  symbol: string;
  requestedSession: string;
  sessionDate: string | null;
  price: number | null;
  timestamp: string | null;
  provider: MarketDataProvider;
  priceType: "delayed_close" | "last_trade";
};

export function cleanMarketSymbol(value: string) {
  return value
    .trim()
    .replace(/^#/, "")
    .replace(/[^a-zA-Z0-9.^=-]/g, "")
    .toUpperCase();
}

export function marketProviderSymbols(value: string) {
  const raw = value.trim().replace(/^#/, "").toUpperCase();
  const forexPair = raw.match(/^([A-Z]{3})\s*\/\s*([A-Z]{3})$/);

  if (forexPair) {
    const base = forexPair[1];
    const quote = forexPair[2];
    return {
      symbol: `${base}/${quote}`,
      stooq: `${base}${quote}`.toLowerCase(),
      yahoo: `${base}${quote}=X`
    };
  }

  const symbol = cleanMarketSymbol(value);
  return {
    symbol,
    stooq: !symbol || symbol.startsWith("^")
      ? null
      : symbol.includes(".") || symbol.includes("=")
        ? symbol.toLowerCase()
        : `${symbol.toLowerCase()}.us`,
    yahoo: symbol
  };
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

function dateInTimeZone(unixTimestamp: number, timeZone: string) {
  try {
    const parts = new Intl.DateTimeFormat("en-US", {
      timeZone,
      year: "numeric",
      month: "2-digit",
      day: "2-digit"
    }).formatToParts(new Date(unixTimestamp * 1000));
    const values = Object.fromEntries(parts.map((part) => [part.type, part.value]));
    return `${values.year}-${values.month}-${values.day}`;
  } catch {
    return new Date(unixTimestamp * 1000).toISOString().slice(0, 10);
  }
}

function validPositiveNumber(value: unknown) {
  const number = Number(value);
  return Number.isFinite(number) && number > 0 ? number : null;
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
  const exchangeTimeZone = (result as { meta?: { exchangeTimezoneName?: unknown } }).meta?.exchangeTimezoneName;
  const dailyTimeZone = typeof exchangeTimeZone === "string" && exchangeTimeZone ? exchangeTimeZone : "UTC";

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
          : dateInTimeZone(unixTimestamp, dailyTimeZone),
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

async function fetchStooqCandles(symbolValue: string, timeframe: MarketTimeframe) {
  if (timeframe !== "1d") {
    return [];
  }
  const symbol = marketProviderSymbols(symbolValue).stooq;
  if (!symbol) return [];

  const response = await fetch(`https://stooq.com/q/d/l/?s=${encodeURIComponent(symbol)}&i=d`, {
    next: { revalidate: 60 * 60 * 6 }
  });

  if (!response.ok) {
    return [];
  }

  return parseStooqCsv(await response.text());
}

async function fetchExactStooqSession(symbolValue: string, session: string) {
  const symbol = marketProviderSymbols(symbolValue).stooq;
  if (!symbol) return null;
  const compactSession = session.replaceAll("-", "");
  const response = await fetch(
    `https://stooq.com/q/d/l/?s=${encodeURIComponent(symbol)}&d1=${compactSession}&d2=${compactSession}&i=d`,
    { cache: "no-store" }
  );
  if (!response.ok) return null;
  return parseStooqCsv(await response.text()).find((candle) => candle.time === session) || null;
}

function yahooSessionWindow(session: string) {
  const sessionStart = Date.parse(`${session}T00:00:00Z`);
  if (!Number.isFinite(sessionStart)) return null;
  return {
    period1: Math.floor((sessionStart - 86_400_000) / 1000),
    period2: Math.floor((sessionStart + 2 * 86_400_000) / 1000)
  };
}

function exactYahooSessionPrice(payload: unknown, symbol: string, session: string): ExactMarketSessionPrice | null {
  const result =
    payload &&
    typeof payload === "object" &&
    "chart" in payload &&
    (payload as { chart?: { result?: unknown[] } }).chart?.result?.[0];
  if (!result || typeof result !== "object") return null;

  const exactCandle = parseYahooResponse(payload, "1d").find((candle) => candle.time === session);
  if (exactCandle) {
    return {
      symbol,
      requestedSession: session,
      sessionDate: session,
      price: exactCandle.close,
      timestamp: null,
      provider: "yahoo",
      priceType: "delayed_close"
    };
  }

  const meta = (result as {
    meta?: {
      exchangeTimezoneName?: unknown;
      regularMarketPrice?: unknown;
      regularMarketTime?: unknown;
    };
  }).meta;
  const price = validPositiveNumber(meta?.regularMarketPrice);
  const unixTimestamp = Number(meta?.regularMarketTime);
  const timeZone = typeof meta?.exchangeTimezoneName === "string" && meta.exchangeTimezoneName
    ? meta.exchangeTimezoneName
    : "America/New_York";
  if (price === null || !Number.isFinite(unixTimestamp) || dateInTimeZone(unixTimestamp, timeZone) !== session) {
    return null;
  }
  return {
    symbol,
    requestedSession: session,
    sessionDate: session,
    price,
    timestamp: new Date(unixTimestamp * 1000).toISOString(),
    provider: "yahoo",
    priceType: "last_trade"
  };
}

async function fetchExactYahooSession(symbolValue: string, session: string) {
  const symbol = marketProviderSymbols(symbolValue).yahoo;
  const window = yahooSessionWindow(session);
  if (!symbol || !window) return null;
  for (const host of ["query1.finance.yahoo.com", "query2.finance.yahoo.com"]) {
    const response = await fetch(
      `https://${host}/v8/finance/chart/${encodeURIComponent(symbol)}?period1=${window.period1}&period2=${window.period2}&interval=1d&events=history`,
      { cache: "no-store" }
    ).catch(() => null);
    if (!response?.ok) continue;
    const exact = exactYahooSessionPrice(await response.json().catch(() => null), marketProviderSymbols(symbolValue).symbol, session);
    if (exact) return exact;
  }
  return null;
}

export async function getExactMarketSessionPrice(symbolValue: string, session: string): Promise<ExactMarketSessionPrice> {
  const symbol = marketProviderSymbols(symbolValue).symbol;
  const unavailable: ExactMarketSessionPrice = {
    symbol,
    requestedSession: session,
    sessionDate: null,
    price: null,
    timestamp: null,
    provider: "unavailable",
    priceType: "delayed_close"
  };
  if (!symbol || !/^\d{4}-\d{2}-\d{2}$/.test(session)) return unavailable;

  const stooqCandle = await fetchExactStooqSession(symbol, session).catch(() => null);
  if (stooqCandle) {
    return {
      symbol,
      requestedSession: session,
      sessionDate: session,
      price: stooqCandle.close,
      timestamp: null,
      provider: "stooq",
      priceType: "delayed_close"
    };
  }
  return await fetchExactYahooSession(symbol, session) || unavailable;
}

export async function getYahooMarketCandles(symbolValue: string, timeframe: MarketTimeframe = "1d") {
  const symbol = marketProviderSymbols(symbolValue).yahoo;
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
  const symbol = marketProviderSymbols(symbolValue).symbol;
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
