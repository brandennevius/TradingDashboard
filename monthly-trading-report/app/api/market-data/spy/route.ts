import { NextResponse } from "next/server";

type SpyCandle = {
  time: string;
  open: number;
  high: number;
  low: number;
  close: number;
};

function numberValue(value: string) {
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function parseStooqCsv(csv: string): SpyCandle[] {
  return csv
    .trim()
    .split(/\r?\n/)
    .slice(1)
    .map((line) => {
      const [date, openRaw, highRaw, lowRaw, closeRaw] = line.split(",");
      const open = numberValue(openRaw);
      const high = numberValue(highRaw);
      const low = numberValue(lowRaw);
      const close = numberValue(closeRaw);

      if (!/^\d{4}-\d{2}-\d{2}$/.test(date) || open === null || high === null || low === null || close === null) {
        return null;
      }

      return {
        time: date,
        open,
        high,
        low,
        close
      };
    })
    .filter((candle): candle is SpyCandle => Boolean(candle));
}

function parseYahooResponse(payload: unknown): SpyCandle[] {
  const result =
    payload &&
    typeof payload === "object" &&
    "chart" in payload &&
    (payload as { chart?: { result?: unknown[] } }).chart?.result?.[0];

  if (!result || typeof result !== "object") {
    return [];
  }

  const quote = (result as { indicators?: { quote?: unknown[] } }).indicators?.quote?.[0] as
    | { open?: unknown[]; high?: unknown[]; low?: unknown[]; close?: unknown[] }
    | undefined;
  const timestamps = (result as { timestamp?: unknown[] }).timestamp || [];

  if (!quote) {
    return [];
  }

  return timestamps
    .map((timestamp, index) => {
      const unixTimestamp = Number(timestamp);
      const open = Number(quote.open?.[index]);
      const high = Number(quote.high?.[index]);
      const low = Number(quote.low?.[index]);
      const close = Number(quote.close?.[index]);

      if (![unixTimestamp, open, high, low, close].every(Number.isFinite)) {
        return null;
      }

      return {
        time: new Date(unixTimestamp * 1000).toISOString().slice(0, 10),
        open,
        high,
        low,
        close
      };
    })
    .filter((candle): candle is SpyCandle => Boolean(candle));
}

async function fetchStooqCandles() {
  const response = await fetch("https://stooq.com/q/d/l/?s=spy.us&i=d", {
    next: { revalidate: 60 * 60 * 6 }
  });

  if (!response.ok) {
    return [];
  }

  const csv = await response.text();
  return parseStooqCsv(csv);
}

async function fetchYahooCandles() {
  const now = Math.floor(Date.now() / 1000);
  const start = now - 60 * 60 * 24 * 730;
  const response = await fetch(
    `https://query1.finance.yahoo.com/v8/finance/chart/SPY?period1=${start}&period2=${now}&interval=1d&events=history`,
    { next: { revalidate: 60 * 60 * 6 } }
  );

  if (!response.ok) {
    return [];
  }

  return parseYahooResponse(await response.json());
}

export async function GET() {
  try {
    const stooqCandles = await fetchStooqCandles().catch(() => []);
    const yahooCandles = stooqCandles.length ? [] : await fetchYahooCandles().catch(() => []);
    const candles = (stooqCandles.length ? stooqCandles : yahooCandles).slice(-420);

    if (!candles.length) {
      return NextResponse.json({ error: "Could not load SPY candles." }, { status: 502 });
    }

    return NextResponse.json({ candles });
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Could not load SPY candles." },
      { status: 500 }
    );
  }
}
