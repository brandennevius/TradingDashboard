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

const rprpRows = [
  { label: "", rvf: 1 },
  { label: "", rvf: 1.5 },
  { label: "SSO", rvf: 2 },
  { label: "", rvf: 2.5 },
  { label: "UPRO", rvf: 3 },
  { label: "", rvf: 3.5 },
  { label: "TQQQ", rvf: 4 },
  { label: "", rvf: 4.5 },
  { label: "", rvf: 5 },
  { label: "", rvf: 5.5 },
  { label: "", rvf: 6 }
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

function nearestRprpRow(rvf: number) {
  return rprpRows.reduce((closest, row) => (Math.abs(row.rvf - rvf) < Math.abs(closest.rvf - rvf) ? row : closest), rprpRows[0]);
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

export default function RprpSizer({ trades, activePortfolio, portfolioMeta }: Props) {
  const [profiles, setProfiles] = useState<Record<string, PriceProfile>>({});
  const [newSymbol, setNewSymbol] = useState("");
  const [analyzedSymbol, setAnalyzedSymbol] = useState("");
  const [selectedColumnIndex, setSelectedColumnIndex] = useState(0);
  const [manualEquity, setManualEquity] = useState("");
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

  const actualsByRow = useMemo(() => {
    const groups = new Map<number, HoldingActual[]>();
    actuals.forEach((actual) => {
      const row = nearestRprpRow(actual.rvf);
      groups.set(row.rvf, [...(groups.get(row.rvf) || []), actual]);
    });
    return groups;
  }, [actuals]);

  const selectedColumn = rprpColumns[selectedColumnIndex];
  const analyzedProfile = analyzedSymbol ? profiles[analyzedSymbol] : undefined;
  const analyzedTargetPct = analyzedProfile ? targetPctFor(analyzedProfile.rvf, selectedColumnIndex) : 0;
  const analyzedDollarTarget = accountEquity ? (accountEquity * analyzedTargetPct) / 100 : 0;
  const analyzedShares = analyzedProfile?.price ? Math.floor(analyzedDollarTarget / analyzedProfile.price) : 0;

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
          <p className="eyebrow">Reverse positional risk parity</p>
          <h2>RPRP Position Sizer</h2>
          <p>
            Equalizes position exposure by shrinking high-volatility names and expanding lower-volatility names. Actuals are
            pulled from current open positions.
          </p>
        </div>
        <div className="rprp-hero-stats">
          <article><span>Portfolio equity</span><strong>{money(accountEquity)}</strong></article>
          <article><span>Open holdings</span><strong>{actuals.length}</strong></article>
          <article><span>Selected column</span><strong>{selectedColumn.min}-{selectedColumn.max}</strong></article>
        </div>
      </div>

      <div className="rprp-controls">
        <label>
          Portfolio equity override
          <input value={manualEquity} onChange={(event) => setManualEquity(event.target.value)} placeholder={accountEquity ? money(accountEquity) : "$"} inputMode="decimal" />
        </label>
        <label>
          Sizing column
          <select value={selectedColumnIndex} onChange={(event) => setSelectedColumnIndex(Number(event.target.value))}>
            {rprpColumns.map((column, index) => (
              <option key={`${column.min}-${column.max}`} value={index}>
                {column.min}-{column.max} stocks / base {pct(column.basePct, 1)}
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

      {status ? <p className="status trade-log-status">{status}</p> : null}
      {isLoading ? <p className="status">Loading prices and RVF estimates...</p> : null}

      {analyzedSymbol ? (
        <div className="rprp-new-stock-card">
          <div>
            <span className="eyebrow">New stock line</span>
            <h3>{analyzedSymbol}</h3>
            <p>
              {analyzedProfile
                ? `Estimated RVF ${analyzedProfile.rvf.toFixed(2)}. It fits closest to the ${nearestRprpRow(analyzedProfile.rvf).rvf.toFixed(1)} row.`
                : "Loading RVF..."}
            </p>
          </div>
          <article><span>Target %</span><strong>{pct(analyzedTargetPct)}</strong></article>
          <article><span>Dollar size</span><strong>{money(analyzedDollarTarget)}</strong></article>
          <article><span>Shares</span><strong>{analyzedShares ? analyzedShares.toLocaleString() : "-"}</strong><em>@ {price(analyzedProfile?.price || 0)}</em></article>
        </div>
      ) : null}

      <div className="rprp-table-wrap">
        <table className="rprp-table">
          <thead>
            <tr>
              <th colSpan={2} className="rprp-title-cell">RPRP: Reverse Positional Risk Parity</th>
              {rprpColumns.map((column) => <th key={`min-${column.min}-${column.max}`}>{column.min}</th>)}
              <th rowSpan={4}>Actuals</th>
            </tr>
            <tr>
              <th colSpan={2}># of stocks min =&gt;</th>
              {rprpColumns.map((column) => <th key={`max-${column.min}-${column.max}`}>{column.max}</th>)}
            </tr>
            <tr>
              <th>Ticker</th>
              <th>RVF</th>
              {rprpColumns.map((column) => <th key={`base-${column.basePct}`}>{column.basePct.toFixed(column.basePct % 1 ? 1 : 0)}%</th>)}
            </tr>
          </thead>
          <tbody>
            {rprpRows.map((row) => {
              const rowActuals = actualsByRow.get(row.rvf) || [];
              const isAnalyzedRow = analyzedProfile && nearestRprpRow(analyzedProfile.rvf).rvf === row.rvf;
              return (
                <tr key={row.rvf} className={[row.label === "(AVERAGE)" ? "average" : "", isAnalyzedRow ? "selected" : ""].filter(Boolean).join(" ")}>
                  <td>{row.label}</td>
                  <td>{row.rvf.toFixed(1)}</td>
                  {rprpColumns.map((column) => (
                    <td key={`${row.rvf}-${column.basePct}`}>{pct(column.basePct / row.rvf)}</td>
                  ))}
                  <td className="rprp-actuals-cell">
                    {rowActuals.length ? (
                      rowActuals.map((actual) => (
                        <span key={actual.symbol}>
                          {actual.symbol} {pct(actual.weightPct, 1)}
                        </span>
                      ))
                    ) : isAnalyzedRow && analyzedProfile ? (
                      <span className="rprp-new-symbol">{analyzedSymbol} target {pct(analyzedTargetPct, 1)}</span>
                    ) : (
                      <em>-</em>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div className="rprp-actuals-grid">
        {actuals.map((actual) => (
          <article key={actual.symbol}>
            <strong>{actual.symbol}</strong>
            <span>Actual {pct(actual.weightPct)} / RVF {actual.rvf.toFixed(2)}</span>
            <em>{money(actual.value)} · {actual.shares.toLocaleString()} sh @ {price(actual.price)}</em>
          </article>
        ))}
      </div>
    </section>
  );
}
