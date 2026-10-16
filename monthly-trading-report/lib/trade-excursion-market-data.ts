import type { ExcursionBar, ExcursionInstrument } from "./trade-excursion";
import { fetchAlpacaBars } from "./alpaca-market-data";

export async function fetchTradeIntradayBars(
  instrument: ExcursionInstrument, from: string, to: string,
  interval = process.env.TRADE_EXCURSION_INTERVAL?.trim() || "5min", signal?: AbortSignal
): Promise<{ bars: ExcursionBar[]; provider: "alpaca"; interval: string; error: string }> {
  const empty = (error: string) => ({ bars: [], provider: "alpaca" as const, interval, error });
  const intervals: Record<string, string> = { "1min": "1Min", "5min": "5Min", "15min": "15Min", "30min": "30Min", "1hour": "1Hour", "4hour": "4Hour" };
  if (!intervals[interval]) return empty("The configured intraday interval is unsupported.");
  if (!instrument.providerSymbol) return empty("Intraday MAE/MFE is available for US stocks, ETFs, and labelled index ETF proxies only.");
  try {
    const all = await fetchAlpacaBars(instrument.providerSymbol, intervals[interval], from, to, signal);
    // Keep regular-session coverage explicit and stable for cache identity and exports.
    const bars = all.filter((bar) => bar.time.slice(11) >= "09:30:00" && bar.time.slice(11) < "16:00:00");
    return { bars, provider: "alpaca", interval, error: bars.length ? "" : "No regular-session intraday bars were returned for this instrument and date range." };
  } catch (error) {
    signal?.throwIfAborted();
    return empty(error instanceof Error ? error.message : "Intraday market data could not be loaded.");
  }
}
