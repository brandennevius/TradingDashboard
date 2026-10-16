export type AlpacaBar = {
  time: string; session: string; open: number; high: number; low: number; close: number; volume: number;
};

export function alpacaStockSymbol(symbol: string) {
  const cleaned = symbol.trim().replace(/^#/, "").toUpperCase();
  return /^[A-Z][A-Z0-9.-]*$/.test(cleaned) ? cleaned.replaceAll(".", "/") : null;
}

const nyClock = new Intl.DateTimeFormat("en-CA", {
  timeZone: "America/New_York", year: "numeric", month: "2-digit", day: "2-digit",
  hour: "2-digit", minute: "2-digit", second: "2-digit", hourCycle: "h23"
});

export function parseAlpacaBars(payload: unknown): AlpacaBar[] {
  if (!Array.isArray(payload)) throw new Error("Invalid market-data bars response.");
  return payload.flatMap((value) => {
    if (!value || typeof value !== "object") return [];
    const row = value as Record<string, unknown>;
    const stamp = Date.parse(String(row.t || ""));
    const [open, high, low, close] = [row.o, row.h, row.l, row.c].map(Number);
    if (!Number.isFinite(stamp) || ![open, high, low, close].every((n) => Number.isFinite(n) && n > 0)
      || high < Math.max(open, low, close) || low > Math.min(open, high, close)) return [];
    const parts = Object.fromEntries(nyClock.formatToParts(new Date(stamp)).map((p) => [p.type, p.value]));
    const session = `${parts.year}-${parts.month}-${parts.day}`;
    const volume = Number(row.v ?? 0);
    return [{ session, time: `${session} ${parts.hour}:${parts.minute}:${parts.second}`,
      open, high, low, close, volume: Number.isFinite(volume) && volume >= 0 ? volume : 0 }];
  });
}

// Only this fixed market-data host receives credentials; no trading endpoints are used.
export async function fetchAlpacaBars(
  symbolValue: string, timeframe: string, from: string, to: string, signal?: AbortSignal
): Promise<AlpacaBar[]> {
  const key = process.env.ALPACA_API_KEY?.trim();
  const secret = process.env.ALPACA_API_SECRET?.trim();
  if (!key || !secret) throw new Error("Alpaca market-data credentials are not configured.");
  const symbol = alpacaStockSymbol(symbolValue);
  if (!symbol) throw new Error("Alpaca intraday data supports listed US stocks and ETFs only.");
  if (!/^(?:[1-9]|[1-5][0-9])Min$|^(?:[1-9]|1[0-9]|2[0-3])Hour$|^1(?:Day|Week|Month)$/.test(timeframe)) {
    throw new Error("Unsupported market-data interval.");
  }
  const start = Date.parse(`${from}T00:00:00Z`);
  // Include the entire New York end date across DST, then filter extra UTC hours.
  const end = Math.min(Date.parse(`${to}T00:00:00Z`) + 30 * 3_600_000 - 1, Date.now() - 16 * 60_000);
  if (!/^\d{4}-\d{2}-\d{2}$/.test(from) || !/^\d{4}-\d{2}-\d{2}$/.test(to)
    || !Number.isFinite(start) || !Number.isFinite(end) || from > to) throw new Error("Invalid market-data date range.");
  if (start > end) return [];
  const url = new URL("https://data.alpaca.markets/v2/stocks/bars");
  url.search = new URLSearchParams({ symbols: symbol, timeframe, start: new Date(start).toISOString(),
    end: new Date(end).toISOString(), feed: "sip", adjustment: "raw", limit: "10000", sort: "asc" }).toString();
  const abort = AbortSignal.any([...(signal ? [signal] : []), AbortSignal.timeout(45_000)]);
  const bars = new Map<string, AlpacaBar>();
  const seenTokens = new Set<string>();
  for (let page = 0; page < 100; page += 1) {
    const response = await fetch(url, { headers: { "APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret },
      signal: abort, cache: "no-store" });
    if (!response.ok) throw new Error(`Alpaca market-data request returned HTTP ${response.status}.`);
    const payload = await response.json() as { bars?: Record<string, unknown>; next_page_token?: unknown };
    if (!payload || typeof payload !== "object" || !payload.bars || typeof payload.bars !== "object") {
      throw new Error("Invalid Alpaca market-data response.");
    }
    for (const bar of parseAlpacaBars(payload.bars[symbol] ?? [])) {
      if (bar.session >= from && bar.session <= to) bars.set(bar.time, bar);
    }
    if (payload.next_page_token == null || payload.next_page_token === "") {
      return [...bars.values()].sort((a, b) => a.time.localeCompare(b.time));
    }
    if (typeof payload.next_page_token !== "string" || seenTokens.has(payload.next_page_token)) {
      throw new Error("Alpaca pagination did not complete; partial data was discarded.");
    }
    seenTokens.add(payload.next_page_token);
    url.searchParams.set("page_token", payload.next_page_token);
  }
  throw new Error("Market-data history exceeded the page limit; partial data was discarded.");
}
