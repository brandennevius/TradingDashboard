import type { ExcursionBar, ExcursionInstrument } from "./trade-excursion";

type IntradayFetchResult = {
  bars: ExcursionBar[];
  provider: "fmp";
  interval: string;
  error: string;
};

function isoDate(value: Date) {
  return value.toISOString().slice(0, 10);
}

function dateChunks(from: string, to: string, maximumDays = 30) {
  const chunks: { from: string; to: string }[] = [];
  let cursor = new Date(`${from}T00:00:00Z`);
  const end = new Date(`${to}T00:00:00Z`);
  if (!Number.isFinite(cursor.getTime()) || !Number.isFinite(end.getTime()) || cursor > end) return chunks;
  while (cursor <= end) {
    const chunkEnd = new Date(cursor);
    chunkEnd.setUTCDate(chunkEnd.getUTCDate() + maximumDays - 1);
    if (chunkEnd > end) chunkEnd.setTime(end.getTime());
    chunks.push({ from: isoDate(cursor), to: isoDate(chunkEnd) });
    cursor = new Date(chunkEnd);
    cursor.setUTCDate(cursor.getUTCDate() + 1);
  }
  return chunks;
}

export function parseFmpIntradayBars(payload: unknown): ExcursionBar[] {
  if (!Array.isArray(payload)) return [];
  return payload.flatMap((item) => {
    if (!item || typeof item !== "object" || Array.isArray(item)) return [];
    const row = item as Record<string, unknown>;
    const time = String(row.date || row.datetime || "").replace("T", " ").replace(/Z$/, "");
    const open = Number(row.open);
    const high = Number(row.high);
    const low = Number(row.low);
    const close = Number(row.close);
    if (!/^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$/.test(time) || ![open, high, low, close].every((value) => Number.isFinite(value) && value > 0)) return [];
    return [{ time, open, high, low, close }];
  }).sort((a, b) => a.time.localeCompare(b.time));
}

export async function fetchTradeIntradayBars(
  instrument: ExcursionInstrument,
  from: string,
  to: string,
  interval = process.env.FMP_TRADE_EXCURSION_INTERVAL?.trim() || "5min",
  signal?: AbortSignal
): Promise<IntradayFetchResult> {
  const apiKey = process.env.FMP_API_KEY?.trim();
  if (!apiKey) return { bars: [], provider: "fmp", interval, error: "FMP_API_KEY is not configured in this environment." };
  if (!instrument.providerSymbol) return { bars: [], provider: "fmp", interval, error: "No compatible FMP symbol is configured." };
  if (!/^(1min|5min|15min|30min|1hour|4hour)$/.test(interval)) {
    return { bars: [], provider: "fmp", interval, error: "The configured intraday interval is unsupported." };
  }

  const chunks = dateChunks(from, to);
  if (!chunks.length) return { bars: [], provider: "fmp", interval, error: "The trade date range is invalid." };
  const bars: ExcursionBar[] = [];
  for (let index = 0; index < chunks.length; index += 4) {
    const batch = chunks.slice(index, index + 4);
    const results = await Promise.all(batch.map(async (chunk) => {
      const url = new URL(`https://financialmodelingprep.com/stable/historical-chart/${interval}`);
      url.searchParams.set("symbol", instrument.providerSymbol!);
      url.searchParams.set("from", chunk.from);
      url.searchParams.set("to", chunk.to);
      const response = await fetch(url, {
        headers: { apikey: apiKey },
        signal: signal ? AbortSignal.any([signal, AbortSignal.timeout(30_000)]) : AbortSignal.timeout(30_000),
        next: { revalidate: 60 * 60 * 6 }
      });
      if (!response.ok) return { bars: [] as ExcursionBar[], error: `FMP intraday request returned HTTP ${response.status}.` };
      return { bars: parseFmpIntradayBars(await response.json()), error: "" };
    }));
    const failed = results.find((result) => result.error);
    if (failed) return { bars: [], provider: "fmp", interval, error: failed.error };
    results.forEach((result) => bars.push(...result.bars));
  }

  const unique = Array.from(new Map(bars.map((bar) => [bar.time, bar])).values()).sort((a, b) => a.time.localeCompare(b.time));
  return {
    bars: unique,
    provider: "fmp",
    interval,
    error: unique.length ? "" : `FMP returned no ${interval} bars for ${instrument.providerSymbol}.`
  };
}
