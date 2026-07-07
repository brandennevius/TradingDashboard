import { NextResponse } from "next/server";

type Candle = {
  time: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume?: number;
};
type Timeframe = "1h" | "4h" | "1d" | "1wk" | "1mo";

function cleanSymbol(value: string) {
  return value
    .trim()
    .replace(/^#/, "")
    .replace(/[^a-zA-Z0-9.=-]/g, "")
    .toUpperCase();
}

function numberValue(value: string) {
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
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

      if (!/^\d{4}-\d{2}-\d{2}$/.test(date) || open === null || high === null || low === null || close === null) {
        return;
      }

      candles.push({ time: date, open, high, low, close, volume: volume || 0 });
    });

  return candles;
}

function timeframeParam(value: string | null): Timeframe {
  return value === "1h" || value === "4h" || value === "1d" || value === "1wk" || value === "1mo" ? value : "1d";
}

function yahooInterval(timeframe: Timeframe) {
  if (timeframe === "4h") {
    return "1h";
  }

  return timeframe;
}

function lookbackSeconds(timeframe: Timeframe) {
  if (timeframe === "1h" || timeframe === "4h") {
    return 60 * 60 * 24 * 120;
  }

  if (timeframe === "1mo") {
    return 60 * 60 * 24 * 365 * 8;
  }

  return 60 * 60 * 24 * 730;
}

function parseYahooResponse(payload: unknown, timeframe: Timeframe): Candle[] {
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

  const candles: Candle[] = [];

  timestamps.forEach((timestamp, index) => {
    const unixTimestamp = Number(timestamp);
    const open = Number(quote.open?.[index]);
    const high = Number(quote.high?.[index]);
    const low = Number(quote.low?.[index]);
    const close = Number(quote.close?.[index]);
    const volume = Number(quote.volume?.[index] || 0);

    if (![unixTimestamp, open, high, low, close].every(Number.isFinite)) {
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

function aggregateFourHourCandles(candles: Candle[]) {
  const groups = new Map<string, Candle[]>();

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

async function fetchStooqCandles(symbol: string, timeframe: Timeframe) {
  if (timeframe !== "1d") {
    return [];
  }

  const stooqSymbol = symbol.includes(".") || symbol.includes("=") ? symbol.toLowerCase() : `${symbol.toLowerCase()}.us`;
  const response = await fetch(`https://stooq.com/q/d/l/?s=${encodeURIComponent(stooqSymbol)}&i=d`, {
    next: { revalidate: 60 * 60 * 6 }
  });

  if (!response.ok) {
    return [];
  }

  return parseStooqCsv(await response.text());
}

async function fetchYahooCandles(symbol: string, timeframe: Timeframe) {
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

export async function GET(request: Request, { params }: { params: Promise<{ symbol: string }> }) {
  const { symbol: rawSymbol } = await params;
  const symbol = cleanSymbol(rawSymbol || "");
  const timeframe = timeframeParam(new URL(request.url).searchParams.get("timeframe"));

  if (!symbol) {
    return NextResponse.json({ error: "A symbol is required." }, { status: 400 });
  }

  try {
    const stooqCandles = await fetchStooqCandles(symbol, timeframe).catch(() => []);
    const yahooCandles = stooqCandles.length ? [] : await fetchYahooCandles(symbol, timeframe).catch(() => []);
    const candles = (stooqCandles.length ? stooqCandles : yahooCandles).slice(-520);

    if (!candles.length) {
      return NextResponse.json({ error: `Could not load ${symbol} candles.` }, { status: 502 });
    }

    return NextResponse.json({ symbol, timeframe, candles });
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : `Could not load ${symbol} candles.` },
      { status: 500 }
    );
  }
}
