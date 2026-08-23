"use client";

import { useEffect, useMemo, useState } from "react";
import type { TradeLogEntry } from "@/lib/types";

type PortfolioMeta = {
  currentEquity?: number;
  statementEquity?: number;
  floatingPnl?: number;
  equityUpdatedAt?: string;
  equityStatementDate?: string;
};

type Props = {
  trades: TradeLogEntry[];
  activePortfolio: string;
  portfolioMeta?: Record<string, PortfolioMeta>;
};

type Candle = {
  time: string;
  close: number;
};

type PriceProfile = {
  symbol: string;
  price: number;
  date: string;
  rvf: number;
  error?: string;
};

type HoldingActual = {
  symbol: string;
  shares: number;
  value: number;
  weightPct: number;
  rvf: number;
  price: number;
};

const rprpColumns = [
  { min: 10, max: 15, basePct: 10 },
  { min: 8, max: 12, basePct: 12.5 },
  { min: 7, max: 10, basePct: 15 },
  { min: 6, max: 9, basePct: 17.5 },
  { min: 5, max: 8, basePct: 20 },
  { min: 4, max: 7, basePct: 22.5 },
  { min: 4, max: 6, basePct: 25 }
];

function cleanSymbol(value: string) {
  return value.trim().replace(/^#/, "").replace(/[^a-zA-Z0-9.=-]/g, "").toUpperCase();
}

function numberValue(value: unknown) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : 0;
}

function money(value: number) {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0
  }).format(Number.isFinite(value) ? value : 0);
}

function pct(value: number, digits = 2) {
  return `${(Number.isFinite(value) ? value : 0).toFixed(digits)}%`;
}

function numberInputValue(value: unknown) {
  const numeric = Number(String(value ?? "").replace(/[$,%\s,]/g, ""));
  return Number.isFinite(numeric) ? numeric : 0;
}

function price(value: number) {
  if (!Number.isFinite(value) || !value) return "-";
  return new Intl.NumberFormat("en-US", { maximumFractionDigits: value < 10 ? 4 : 2 }).format(value);
}

function dailyReturns(candles: Candle[]) {
  return candles
    .slice(-90)
    .map((candle, index, recent) => {
      if (!index) return 0;
      const previous = numberValue(recent[index - 1]?.close);
      const close = numberValue(candle.close);
      return previous && close ? (close - previous) / previous : 0;
    })
    .filter((value) => Number.isFinite(value) && value !== 0);
}

function standardDeviation(values: number[]) {
  if (values.length < 2) return 0;
  const average = values.reduce((sum, value) => sum + value, 0) / values.length;
  const variance = values.reduce((sum, value) => sum + Math.pow(value - average, 2), 0) / (values.length - 1);
  return Math.sqrt(variance);
}

function estimateRvf(candles: Candle[], spyCandles: Candle[]) {
  const stockVolatility = standardDeviation(dailyReturns(candles));
  const spyVolatility = standardDeviation(dailyReturns(spyCandles));
  if (!stockVolatility || !spyVolatility) return 2.5;
  return Math.max(1, Math.min(6, stockVolatility / spyVolatility));
}

function pickPortfolioMeta(portfolioMeta: Record<string, PortfolioMeta> | undefined, activePortfolio: string) {
  if (!portfolioMeta) return undefined;
  if (activePortfolio && portfolioMeta[activePortfolio]) return portfolioMeta[activePortfolio];
  return Object.values(portfolioMeta)
    .filter((meta) => Number(meta.currentEquity) > 0)
    .sort((a, b) => {
      const bTime = Date.parse(b.equityUpdatedAt || b.equityStatementDate || "");
      const aTime = Date.parse(a.equityUpdatedAt || a.equityStatementDate || "");
      return (Number.isFinite(bTime) ? bTime : 0) - (Number.isFinite(aTime) ? aTime : 0);
    })[0];
}

async function fetchCandles(symbol: string) {
  const response = await fetch(`/api/market-data/${encodeURIComponent(symbol)}?timeframe=1d`, { cache: "no-store" });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.error || `Could not load ${symbol}.`);
  return (Array.isArray(data.candles) ? data.candles : [])
    .map((candle: { time?: unknown; close?: unknown }) => ({ time: String(candle.time || ""), close: numberValue(candle.close) }))
    .filter((candle: Candle) => candle.time && candle.close > 0);
}

function targetPctFor(rvf: number, columnIndex: number) {
  return rprpColumns[columnIndex].basePct / Math.max(rvf, 0.1);
}

function capStatus(weightPct: number, maxPct: number, hasData: boolean) {
  if (!hasData || !maxPct) return "Needs price data";
  if (weightPct > maxPct * 1.05) return "Over cap";
  if (weightPct >= maxPct * 0.85) return "Near cap";
  return "Under cap";
}

export default function RprpSizer({ trades, activePortfolio, portfolioMeta }: Props) {
  const [profiles, setProfiles] = useState<Record<string, PriceProfile>>({});
  const [newSymbol, setNewSymbol] = useState("");
  const [analyzedSymbol, setAnalyzedSymbol] = useState("");
  const [selectedColumnIndex, setSelectedColumnIndex] = useState(0);
  const [manualEquity, setManualEquity] = useState("");
  const [riskPct, setRiskPct] = useState("0.25");
  const [entryPrice, setEntryPrice] = useState("");
  const [stopPrice, setStopPrice] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [status, setStatus] = useState("");

  const openPositions = useMemo(
    () =>
      trades
        .filter((trade) => trade.status === "OPEN")
        .filter((trade) => !activePortfolio || trade.portfolioTag === activePortfolio),
    [activePortfolio, trades]
  );

  const holdingsBySymbol = useMemo(() => {
    const groups = new Map<string, { symbol: string; shares: number; avgEntryValue: number }>();
    openPositions.forEach((trade) => {
      const symbol = cleanSymbol(trade.symbol);
      if (!symbol) return;
      const shares = Math.abs(numberValue(trade.shares));
      const current = groups.get(symbol) || { symbol, shares: 0, avgEntryValue: 0 };
      current.shares += shares;
      current.avgEntryValue += shares * numberValue(trade.avgEntry);
      groups.set(symbol, current);
    });
    return Array.from(groups.values()).sort((a, b) => a.symbol.localeCompare(b.symbol));
  }, [openPositions]);

  const meta = useMemo(() => pickPortfolioMeta(portfolioMeta, activePortfolio), [activePortfolio, portfolioMeta]);
  const estimatedPositionValue = useMemo(
    () =>
      holdingsBySymbol.reduce((sum, holding) => {
        const profile = profiles[holding.symbol];
        const fallbackPrice = holding.shares ? holding.avgEntryValue / holding.shares : 0;
        return sum + holding.shares * (profile?.price || fallbackPrice);
      }, 0),
    [holdingsBySymbol, profiles]
  );
  const accountEquity = numberValue(manualEquity) || numberValue(meta?.currentEquity) || estimatedPositionValue || 0;
  const symbolsToLoad = useMemo(
    () => Array.from(new Set([...holdingsBySymbol.map((holding) => holding.symbol), analyzedSymbol].filter(Boolean))),
    [analyzedSymbol, holdingsBySymbol]
  );

  useEffect(() => {
    let cancelled = false;

    async function loadProfiles() {
      if (!symbolsToLoad.length) {
        setProfiles({});
        return;
      }

      setIsLoading(true);
      setStatus("Estimating RVF from recent daily volatility...");

      try {
        const spyCandles = await fetchCandles("SPY");
        const nextProfiles: Record<string, PriceProfile> = {};
        await Promise.all(
          symbolsToLoad.map(async (symbol) => {
            try {
              const candles = await fetchCandles(symbol);
              const latest = candles[candles.length - 1];
              nextProfiles[symbol] = {
                symbol,
                price: latest?.close || 0,
                date: latest?.time || "",
                rvf: estimateRvf(candles, spyCandles)
              };
            } catch (error) {
              nextProfiles[symbol] = {
                symbol,
                price: 0,
                date: "",
                rvf: 2.5,
                error: error instanceof Error ? error.message : `Could not load ${symbol}.`
              };
            }
          })
        );

        if (!cancelled) {
          setProfiles(nextProfiles);
          setStatus("");
        }
      } catch (error) {
        if (!cancelled) {
          setStatus(error instanceof Error ? error.message : "Could not load market data.");
        }
      } finally {
        if (!cancelled) setIsLoading(false);
      }
    }

    loadProfiles();

    return () => {
      cancelled = true;
    };
  }, [symbolsToLoad.join("|")]);

  const actuals = useMemo<HoldingActual[]>(() => {
    return holdingsBySymbol.map((holding) => {
      const profile = profiles[holding.symbol];
      const fallbackPrice = holding.shares ? holding.avgEntryValue / holding.shares : 0;
      const latestPrice = profile?.price || fallbackPrice;
      const value = holding.shares * latestPrice;
      return {
        symbol: holding.symbol,
        shares: holding.shares,
        value,
        weightPct: accountEquity ? (value / accountEquity) * 100 : 0,
        rvf: profile?.rvf || 2.5,
        price: latestPrice
      };
    });
  }, [accountEquity, holdingsBySymbol, profiles]);

  const selectedColumn = rprpColumns[selectedColumnIndex];
  const analyzedProfile = analyzedSymbol ? profiles[analyzedSymbol] : undefined;
  const activeRvf = analyzedProfile?.rvf || 2.5;
  const analyzedTargetPct = targetPctFor(activeRvf, selectedColumnIndex);
  const analyzedDollarTarget = accountEquity ? (accountEquity * analyzedTargetPct) / 100 : 0;
  const analyzedShares = analyzedProfile?.price ? Math.floor(analyzedDollarTarget / analyzedProfile.price) : 0;
  const parsedRiskPct = numberInputValue(riskPct);
  const parsedEntry = numberInputValue(entryPrice);
  const parsedStop = numberInputValue(stopPrice);
  const riskDollars = accountEquity && parsedRiskPct ? (accountEquity * parsedRiskPct) / 100 : 0;
  const riskPerShare = parsedEntry && parsedStop ? Math.abs(parsedEntry - parsedStop) : 0;
  const riskBasedShares = riskPerShare ? Math.floor(riskDollars / riskPerShare) : 0;
  const riskBasedValue = riskBasedShares * parsedEntry;
  const riskBasedWeight = accountEquity ? (riskBasedValue / accountEquity) * 100 : 0;
  const volatilityMaxShares = parsedEntry ? Math.floor(analyzedDollarTarget / parsedEntry) : analyzedShares;
  const finalShares = riskBasedShares && volatilityMaxShares ? Math.min(riskBasedShares, volatilityMaxShares) : riskBasedShares || volatilityMaxShares;
  const finalValue = finalShares * (parsedEntry || analyzedProfile?.price || 0);
  const finalReason =
    riskBasedShares && volatilityMaxShares
      ? riskBasedShares <= volatilityMaxShares
        ? "Stop-based risk is stricter than the volatility cap."
        : "Volatility cap is stricter than the stop-based risk size."
      : riskBasedShares
        ? "Enter a ticker to compare against its volatility cap."
        : "Enter entry and stop prices to calculate stop-based size.";

  function analyzeTicker() {
    const symbol = cleanSymbol(newSymbol);
    if (!symbol) {
      setStatus("Enter a ticker to analyze.");
      return;
    }
    setAnalyzedSymbol(symbol);
  }

  return (
    <section className="rprp-tool">
      <div className="rprp-hero">
        <div>
          <p className="eyebrow">Volatility + risk sizing</p>
          <h2>Position Size Check</h2>
          <p>
            Use your stop first, then cap the position if the stock is too volatile for the portfolio.
          </p>
        </div>
        <div className="rprp-hero-stats">
          <article><span>Portfolio equity</span><strong>{money(accountEquity)}</strong></article>
          <article><span>Open holdings</span><strong>{actuals.length}</strong></article>
          <article><span>Target model</span><strong>{selectedColumn.min}-{selectedColumn.max} names</strong></article>
        </div>
      </div>

      <div className="rprp-controls">
        <label>
          Account equity
          <input value={manualEquity} onChange={(event) => setManualEquity(event.target.value)} placeholder={accountEquity ? money(accountEquity) : "$"} inputMode="decimal" />
        </label>
        <label>
          Risk per trade %
          <input value={riskPct} onChange={(event) => setRiskPct(event.target.value)} placeholder="0.25" inputMode="decimal" />
        </label>
        <label>
          Target positions
          <select value={selectedColumnIndex} onChange={(event) => setSelectedColumnIndex(Number(event.target.value))}>
            {rprpColumns.map((column, index) => (
              <option key={`${column.min}-${column.max}`} value={index}>
                {column.min}-{column.max} positions / base {pct(column.basePct, 1)}
              </option>
            ))}
          </select>
        </label>
        <label>
          New ticker
          <input value={newSymbol} onChange={(event) => setNewSymbol(event.target.value.toUpperCase())} onKeyDown={(event) => { if (event.key === "Enter") analyzeTicker(); }} placeholder="AAPL" />
        </label>
        <button type="button" onClick={analyzeTicker}>Analyze ticker</button>
      </div>

      <div className="rprp-controls rprp-trade-inputs">
        <label>
          Entry price
          <input value={entryPrice} onChange={(event) => setEntryPrice(event.target.value)} placeholder="100.00" inputMode="decimal" />
        </label>
        <label>
          Stop price
          <input value={stopPrice} onChange={(event) => setStopPrice(event.target.value)} placeholder="90.00" inputMode="decimal" />
        </label>
        <article>
          <span>Risk dollars</span>
          <strong>{money(riskDollars)}</strong>
        </article>
        <article>
          <span>Risk / share</span>
          <strong>{riskPerShare ? money(riskPerShare) : "-"}</strong>
        </article>
      </div>

      {status ? <p className="status trade-log-status">{status}</p> : null}
      {isLoading ? <p className="status">Loading prices and RVF estimates...</p> : null}

      <div className="rprp-new-stock-card">
        <div>
          <span className="eyebrow">Sizing decision</span>
          <h3>{analyzedSymbol || "Enter a ticker"}</h3>
          <p>{finalReason}</p>
        </div>
        <article>
          <span>Risk-based size</span>
          <strong>{riskBasedShares ? riskBasedShares.toLocaleString() : "-"}</strong>
          <em>{riskBasedValue ? `${money(riskBasedValue)} · ${pct(riskBasedWeight, 1)}` : "Needs entry + stop"}</em>
        </article>
        <article>
          <span>Volatility max</span>
          <strong>{volatilityMaxShares ? volatilityMaxShares.toLocaleString() : "-"}</strong>
          <em>RVF {activeRvf.toFixed(2)} · max {pct(analyzedTargetPct, 1)}</em>
        </article>
        <article>
          <span>Final suggested size</span>
          <strong>{finalShares ? finalShares.toLocaleString() : "-"}</strong>
          <em>{finalValue ? money(finalValue) : "Waiting for inputs"}</em>
        </article>
      </div>

      <div className="rprp-table-wrap">
        <div className="rprp-section-heading">
          <div>
            <p className="eyebrow">Current holdings</p>
            <h3>Volatility cap check</h3>
          </div>
          <span>Final size uses the smaller of stop-based risk and volatility cap.</span>
        </div>
        <table className="rprp-table rprp-position-table">
          <thead>
            <tr>
              <th>Symbol</th>
              <th>Shares</th>
              <th>Current value</th>
              <th>Current weight</th>
              <th>RVF</th>
              <th>Suggested max</th>
              <th>Max dollars</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            {actuals.map((actual) => {
              const maxPct = targetPctFor(actual.rvf, selectedColumnIndex);
              const maxDollars = accountEquity ? (accountEquity * maxPct) / 100 : 0;
              const hasData = !profiles[actual.symbol]?.error;
              const statusText = capStatus(actual.weightPct, maxPct, hasData);
              return (
                <tr key={actual.symbol}>
                  <td>{actual.symbol}</td>
                  <td>{actual.shares.toLocaleString()}</td>
                  <td>{money(actual.value)}</td>
                  <td>{pct(actual.weightPct, 2)}</td>
                  <td>{hasData ? `${actual.rvf.toFixed(2)}x` : "-"}</td>
                  <td>{hasData ? pct(maxPct, 2) : "-"}</td>
                  <td>{hasData ? money(maxDollars) : "-"}</td>
                  <td><span className={`rprp-status ${statusText.toLowerCase().replace(/\s+/g, "-")}`}>{statusText}</span></td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}
