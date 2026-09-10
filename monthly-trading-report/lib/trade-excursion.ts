import crypto from "crypto";
import type { TradeExecution, TradeLogEntry } from "./types";

export const TRADE_EXCURSION_VERSION = "trade-excursion-v1";

export type ExcursionInstrumentMode = "EXACT" | "FUTURES_PROXY" | "UNSUPPORTED";

export type ExcursionInstrument = {
  sourceSymbol: string;
  providerSymbol: string | null;
  mode: ExcursionInstrumentMode;
  label: string;
};

export type ExcursionBar = {
  time: string;
  open: number;
  high: number;
  low: number;
  close: number;
};

export type TradeExcursionResult = {
  status: "AVAILABLE" | "ESTIMATED_PROXY" | "UNAVAILABLE";
  reason: string;
  algorithmVersion: string;
  inputHash: string;
  provider: string;
  providerSymbol: string | null;
  instrumentMode: ExcursionInstrumentMode;
  instrumentLabel: string;
  interval: string;
  asOf: string;
  isOpen: boolean;
  mfeDollars: number | null;
  maeDollars: number | null;
  mfeR: number | null;
  maeR: number | null;
  mfeTimestamp: string | null;
  maeTimestamp: string | null;
  priceScale: number | null;
  barsEvaluated: number;
};

const CFD_FUTURES_MAP: Record<string, Omit<ExcursionInstrument, "sourceSymbol">> = {
  ".US500": { providerSymbol: "ESUSD", mode: "FUTURES_PROXY", label: "S&P 500 CFD mapped to ES futures" },
  ".US100": { providerSymbol: "NQUSD", mode: "FUTURES_PROXY", label: "Nasdaq-100 CFD mapped to NQ futures" },
  ".US30": { providerSymbol: "YMUSD", mode: "FUTURES_PROXY", label: "Dow CFD mapped to YM futures" }
};

const CFD_SYMBOL_ENV: Record<string, string> = {
  ".US500": "FMP_US500_FUTURES_SYMBOL",
  ".US100": "FMP_US100_FUTURES_SYMBOL",
  ".US30": "FMP_US30_FUTURES_SYMBOL"
};

function cleanSymbol(value: string) {
  return value.trim().replace(/^#/, "").toUpperCase();
}

export function excursionInstrument(value: string): ExcursionInstrument {
  const symbol = cleanSymbol(value);
  const mapped = CFD_FUTURES_MAP[symbol];
  if (mapped) {
    const configuredSymbol = process.env[CFD_SYMBOL_ENV[symbol]]?.trim().toUpperCase();
    return { sourceSymbol: symbol, ...mapped, providerSymbol: configuredSymbol || mapped.providerSymbol };
  }

  const forex = symbol.match(/^([A-Z]{3})\/([A-Z]{3})$/);
  if (forex) {
    return { sourceSymbol: symbol, providerSymbol: `${forex[1]}${forex[2]}`, mode: "EXACT", label: "Spot FX" };
  }

  if (/^[A-Z][A-Z0-9.-]*$/.test(symbol)) {
    return { sourceSymbol: symbol, providerSymbol: symbol, mode: "EXACT", label: "Listed security" };
  }

  return { sourceSymbol: symbol, providerSymbol: null, mode: "UNSUPPORTED", label: "Unsupported instrument" };
}

function timestamp(date: string, time: string) {
  const normalizedTime = /^\d{2}:\d{2}(:\d{2})?$/.test(time) ? (time.length === 5 ? `${time}:00` : time) : "00:00:00";
  return `${date} ${normalizedTime}`;
}

function finite(value: unknown) {
  const number = Number(value);
  return Number.isFinite(number) ? number : 0;
}

function round(value: number | null, digits = 4) {
  if (value === null) return null;
  const factor = 10 ** digits;
  return Math.round(value * factor) / factor;
}

function normalizedExecutions(trade: TradeLogEntry) {
  const executions = (trade.executions || [])
    .filter((execution) => execution.date && finite(execution.shares) > 0 && finite(execution.price) > 0)
    .map((execution) => ({ ...execution, time: execution.time || "00:00:00" }));

  if (executions.length) {
    return executions.sort((a, b) => timestamp(a.date, a.time).localeCompare(timestamp(b.date, b.time)) || a.type.localeCompare(b.type));
  }

  if (!trade.entryDate || !finite(trade.avgEntry) || !finite(trade.shares)) return [];
  const synthetic: TradeExecution[] = [{
    id: `${trade.id}-synthetic-entry`, type: "ENTRY", date: trade.entryDate, time: trade.openTime || "00:00:00",
    side: trade.side, shares: trade.shares, price: trade.avgEntry, pnl: 0, commission: 0, source: "trade-row", sourceKey: ""
  }];
  if (trade.status !== "OPEN" && trade.exitDate && finite(trade.exitPrice)) {
    synthetic.push({
      id: `${trade.id}-synthetic-exit`, type: "EXIT", date: trade.exitDate, time: trade.closeTime || "23:59:59",
      side: trade.side, shares: trade.shares, price: trade.exitPrice, pnl: trade.pnl, commission: 0, source: "trade-row", sourceKey: ""
    });
  }
  return synthetic;
}

export function tradeExcursionInputHash(trade: TradeLogEntry, asOf: string) {
  const payload = JSON.stringify({
    version: TRADE_EXCURSION_VERSION,
    id: trade.id,
    symbol: cleanSymbol(trade.symbol),
    side: trade.side,
    status: trade.status,
    risk: finite(trade.risk),
    entryDate: trade.entryDate,
    exitDate: trade.exitDate,
    openTime: trade.openTime,
    closeTime: trade.closeTime,
    avgEntry: finite(trade.avgEntry),
    exitPrice: finite(trade.exitPrice),
    shares: finite(trade.shares),
    executions: normalizedExecutions(trade).map(({ type, date, time, shares, price, commission, sourceKey }) => ({
      type, date, time, shares: finite(shares), price: finite(price), commission: finite(commission), sourceKey
    })),
    asOf: trade.status === "OPEN" ? asOf : trade.exitDate
  });
  return crypto.createHash("sha256").update(payload).digest("hex");
}

function unavailable(trade: TradeLogEntry, instrument: ExcursionInstrument, reason: string, provider: string, interval: string, asOf: string): TradeExcursionResult {
  return {
    status: "UNAVAILABLE", reason, algorithmVersion: TRADE_EXCURSION_VERSION,
    inputHash: tradeExcursionInputHash(trade, asOf), provider, providerSymbol: instrument.providerSymbol,
    instrumentMode: instrument.mode, instrumentLabel: instrument.label, interval, asOf, isOpen: trade.status === "OPEN",
    mfeDollars: null, maeDollars: null, mfeR: null, maeR: null, mfeTimestamp: null, maeTimestamp: null,
    priceScale: null, barsEvaluated: 0
  };
}

export function calculateTradeExcursion(
  trade: TradeLogEntry,
  rawBars: ExcursionBar[],
  options: { provider?: string; interval?: string; asOf?: string; instrument?: ExcursionInstrument } = {}
): TradeExcursionResult {
  const instrument = options.instrument || excursionInstrument(trade.symbol);
  const provider = options.provider || "fmp";
  const interval = options.interval || "5min";
  const asOf = options.asOf || trade.exitDate || new Date().toISOString().slice(0, 10);
  const executions = normalizedExecutions(trade);
  if (!instrument.providerSymbol || instrument.mode === "UNSUPPORTED") return unavailable(trade, instrument, "No compatible market-data symbol is configured.", provider, interval, asOf);
  if (!executions.length) return unavailable(trade, instrument, "Entry executions or a complete trade row are required.", provider, interval, asOf);

  const bars = rawBars
    .filter((bar) => bar.time && finite(bar.high) > 0 && finite(bar.low) > 0 && finite(bar.close) > 0)
    .sort((a, b) => a.time.localeCompare(b.time));
  if (!bars.length) return unavailable(trade, instrument, "No compatible intraday bars were returned.", provider, interval, asOf);

  const firstEntry = executions.find((execution) => execution.type === "ENTRY")!;
  const firstEntryTimestamp = timestamp(firstEntry.date, firstEntry.time);
  const anchorBar = bars.find((bar) => bar.time >= firstEntryTimestamp) || bars.at(-1)!;
  const priceScale = instrument.mode === "FUTURES_PROXY" ? finite(firstEntry.price) / finite(anchorBar.close) : 1;
  if (!Number.isFinite(priceScale) || priceScale <= 0) return unavailable(trade, instrument, "The futures proxy could not be anchored to the broker entry price.", provider, interval, asOf);

  const direction = trade.side === "SHORT" ? -1 : 1;
  const lots: { shares: number; price: number }[] = [];
  let realized = 0;
  let commission = 0;
  let eventIndex = 0;
  let evaluated = 0;
  let invalid = "";
  let mfe = 0;
  let mae = 0;
  let mfeTimestamp: string | null = firstEntryTimestamp;
  let maeTimestamp: string | null = firstEntryTimestamp;

  function apply(execution: TradeExecution) {
    let shares = finite(execution.shares);
    commission += finite(execution.commission);
    if (execution.type === "ENTRY") {
      lots.push({ shares, price: finite(execution.price) });
      return;
    }
    while (shares > 1e-8 && lots.length) {
      const lot = lots[0];
      const matched = Math.min(shares, lot.shares);
      realized += direction * (finite(execution.price) - lot.price) * matched;
      lot.shares -= matched;
      shares -= matched;
      if (lot.shares <= 1e-8) lots.shift();
    }
    if (shares > 1e-6) invalid = "Exit quantity exceeds the open quantity in the stored execution lifecycle.";
  }

  function marked(price: number) {
    return realized - commission + lots.reduce((sum, lot) => sum + direction * (price - lot.price) * lot.shares, 0);
  }

  for (const bar of bars) {
    while (eventIndex < executions.length && timestamp(executions[eventIndex].date, executions[eventIndex].time) < bar.time) {
      apply(executions[eventIndex]);
      eventIndex += 1;
    }
    while (eventIndex < executions.length && timestamp(executions[eventIndex].date, executions[eventIndex].time) === bar.time && executions[eventIndex].type === "ENTRY") {
      apply(executions[eventIndex]);
      eventIndex += 1;
    }
    if (invalid) break;
    if (lots.length) {
      const high = finite(bar.high) * priceScale;
      const low = finite(bar.low) * priceScale;
      const favorable = marked(direction === 1 ? high : low);
      const adverse = marked(direction === 1 ? low : high);
      if (favorable > mfe) { mfe = favorable; mfeTimestamp = bar.time; }
      if (adverse < mae) { mae = adverse; maeTimestamp = bar.time; }
      evaluated += 1;
    }
    while (eventIndex < executions.length && timestamp(executions[eventIndex].date, executions[eventIndex].time) === bar.time) {
      apply(executions[eventIndex]);
      eventIndex += 1;
    }
  }

  while (!invalid && eventIndex < executions.length) {
    apply(executions[eventIndex]);
    eventIndex += 1;
  }
  if (invalid) return unavailable(trade, instrument, invalid, provider, interval, asOf);
  if (!evaluated) return unavailable(trade, instrument, "No bars overlapped an open position in the lifecycle.", provider, interval, asOf);

  const risk = finite(trade.risk);
  return {
    status: instrument.mode === "FUTURES_PROXY" ? "ESTIMATED_PROXY" : "AVAILABLE",
    reason: instrument.mode === "FUTURES_PROXY"
      ? "Calculated from a futures proxy scaled to the broker entry; basis and spread differences can affect the result."
      : "Calculated from the stored execution lifecycle and intraday bars.",
    algorithmVersion: TRADE_EXCURSION_VERSION, inputHash: tradeExcursionInputHash(trade, asOf), provider,
    providerSymbol: instrument.providerSymbol, instrumentMode: instrument.mode, instrumentLabel: instrument.label,
    interval, asOf, isOpen: trade.status === "OPEN", mfeDollars: round(mfe, 2), maeDollars: round(mae, 2),
    mfeR: risk > 0 ? round(mfe / risk) : null, maeR: risk > 0 ? round(mae / risk) : null,
    mfeTimestamp, maeTimestamp, priceScale: round(priceScale, 8), barsEvaluated: evaluated
  };
}

export function unavailableTradeExcursion(trade: TradeLogEntry, reason: string, options: { provider?: string; interval?: string; asOf?: string } = {}) {
  const asOf = options.asOf || trade.exitDate || new Date().toISOString().slice(0, 10);
  return unavailable(trade, excursionInstrument(trade.symbol), reason, options.provider || "fmp", options.interval || "5min", asOf);
}
