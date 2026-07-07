import type { TradeSide } from "./types";

export type TradeAssetClass = "equity" | "forex" | "metal" | "index-cfd" | "leveraged";

const fiatCurrencies = new Set([
  "AUD",
  "CAD",
  "CHF",
  "EUR",
  "GBP",
  "JPY",
  "NZD",
  "USD"
]);

export function classifyTradeAsset(symbol: string): TradeAssetClass {
  const normalized = symbol.trim().replace(/^#/, "").toUpperCase();

  if (normalized.startsWith(".")) {
    return "index-cfd";
  }

  const [base, quote, ...rest] = normalized.split("/");

  if (base && quote && !rest.length) {
    if (base === "XAU" || base === "XAG") {
      return "metal";
    }

    if (fiatCurrencies.has(base) && fiatCurrencies.has(quote)) {
      return "forex";
    }

    return "leveraged";
  }

  return "equity";
}

export function tradeReturnLabel(symbol: string) {
  return classifyTradeAsset(symbol) === "equity" ? "Position return" : "Price return";
}

export function displayTradeReturnPercent(input: {
  symbol: string;
  side: TradeSide;
  avgEntry: number;
  exitPrice: number;
  shares: number;
  pnl: number;
  fallbackReturnPercent?: number;
}) {
  const assetClass = classifyTradeAsset(input.symbol);

  if (assetClass !== "equity") {
    if (!input.avgEntry || !input.exitPrice) {
      return null;
    }

    const direction = input.side === "SHORT" ? -1 : 1;
    return ((input.exitPrice - input.avgEntry) / input.avgEntry) * 100 * direction;
  }

  const costBasis = input.avgEntry * input.shares;

  if (costBasis) {
    return (input.pnl / costBasis) * 100;
  }

  return Number.isFinite(input.fallbackReturnPercent) ? input.fallbackReturnPercent ?? null : null;
}
