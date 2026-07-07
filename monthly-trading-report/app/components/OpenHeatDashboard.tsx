"use client";

import { useEffect, useMemo, useState } from "react";
import type { TradeLogEntry } from "@/lib/types";

type Props = {
  trades: TradeLogEntry[];
  activePortfolio: string;
  onSelectTrade?: (tradeId: string) => void;
  portfolioMeta?: Record<string, PortfolioMeta>;
};

type LatestPrice = {
  price: number;
  date: string;
  error?: string;
};

type RiskRow = {
  trade: TradeLogEntry;
  currentPrice: number;
  priceDate: string;
  positionValue: number;
  floatingPnl: number;
  floatingPct: number;
  dollarRisk: number;
  stopOutcome: number;
  riskPct: number;
  weightPct: number;
  status: "ready" | "fallback" | "missing";
  note: string;
};
type PortfolioMeta = {
  currentEquity?: number;
  statementEquity?: number;
  floatingPnl?: number;
  equitySource?: string;
  equityUpdatedAt?: string;
  equityStatementDate?: string;
};
type PortfolioSettingsResponse = {
  portfolioMeta?: Record<string, PortfolioMeta>;
};

const DEFAULT_EQUITY = 700000;
const DEFAULT_DRAWDOWN_FLOOR = 688000;

function formatCurrency(value: number) {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0
  }).format(Number.isFinite(value) ? value : 0);
}

function formatPrice(value: number) {
  if (!Number.isFinite(value) || value === 0) {
    return "-";
  }

  return new Intl.NumberFormat("en-US", {
    maximumFractionDigits: value < 10 ? 4 : 2
  }).format(value);
}

function formatPercent(value: number) {
  return `${(Number.isFinite(value) ? value : 0).toFixed(2)}%`;
}

function pickLatestImportedMeta(portfolioMeta: Record<string, PortfolioMeta> | undefined) {
  if (!portfolioMeta) {
    return undefined;
  }

  return Object.values(portfolioMeta)
    .filter((meta) => Number.isFinite(Number(meta.currentEquity)) && Number(meta.currentEquity) > 0)
    .sort((a, b) => {
      const bTime = Date.parse(b.equityUpdatedAt || b.equityStatementDate || "");
      const aTime = Date.parse(a.equityUpdatedAt || a.equityStatementDate || "");
      return (Number.isFinite(bTime) ? bTime : 0) - (Number.isFinite(aTime) ? aTime : 0);
    })[0];
}

async function fetchLatestPrice(symbol: string): Promise<LatestPrice> {
  const response = await fetch(`/api/market-data/${encodeURIComponent(symbol)}?timeframe=1d`, { cache: "no-store" });
  const data = await response.json().catch(() => ({}));

  if (!response.ok) {
    return { price: 0, date: "", error: data.error || `Could not load ${symbol}.` };
  }

  const candles = Array.isArray(data.candles) ? data.candles : [];
  const latest = [...candles].reverse().find((candle) => {
    const close = Number(candle?.close);
    return Boolean(candle?.time) && Number.isFinite(close) && close > 0;
  });

  return {
    price: Number(latest?.close) || 0,
    date: latest?.time ? String(latest.time) : ""
  };
}

function buildRiskRow(trade: TradeLogEntry, price: LatestPrice | undefined, currentEquity: number): RiskRow {
  const shares = Math.abs(Number(trade.shares) || 0);
  const latestPrice = Number(price?.price) || 0;
  const entryPrice = Number(trade.avgEntry) || 0;
  const currentPrice = latestPrice || entryPrice;
  const stopPrice = Number(trade.stopPrice) || 0;
  const storedRisk = Math.abs(Number(trade.risk) || 0);
  const positionValue = currentPrice && shares ? currentPrice * shares : Math.abs(entryPrice * shares);
  const floatingPnl =
    entryPrice && currentPrice && shares
      ? trade.side === "SHORT"
        ? (entryPrice - currentPrice) * shares
        : (currentPrice - entryPrice) * shares
      : 0;
  const floatingPct =
    entryPrice && currentPrice
      ? trade.side === "SHORT"
        ? ((entryPrice - currentPrice) / entryPrice) * 100
        : ((currentPrice - entryPrice) / entryPrice) * 100
      : 0;
  let stopOutcome = 0;
  let barRisk = 0;
  let status: RiskRow["status"] = "missing";
  let note = "Needs average entry, shares, and stop.";

  if (entryPrice && stopPrice && shares) {
    stopOutcome =
      trade.side === "SHORT"
        ? (entryPrice - stopPrice) * shares
        : (stopPrice - entryPrice) * shares;
    barRisk =
      currentPrice && stopPrice
        ? trade.side === "SHORT"
          ? Math.max(0, (stopPrice - currentPrice) * shares)
          : Math.max(0, (currentPrice - stopPrice) * shares)
        : Math.max(0, -stopOutcome);
    status = "ready";
    note = stopOutcome >= 0 ? "Stop locks profit versus average cost; BAR is current giveback to stop." : "Stop loss versus average cost.";
  } else if (storedRisk) {
    stopOutcome = -storedRisk;
    barRisk = storedRisk;
    status = "fallback";
    note = currentPrice ? "Using saved original risk fallback." : "Using stored risk fallback.";
  } else if (price?.error) {
    note = price.error;
  }

  const dollarRisk = barRisk;

  return {
    trade,
    currentPrice,
    priceDate: price?.date || "",
    positionValue,
    floatingPnl,
    floatingPct,
    dollarRisk,
    stopOutcome,
    riskPct: currentEquity ? (dollarRisk / currentEquity) * 100 : 0,
    weightPct: currentEquity && positionValue ? (positionValue / currentEquity) * 100 : 0,
    status,
    note
  };
}

export default function OpenHeatDashboard({ trades, activePortfolio, onSelectTrade, portfolioMeta }: Props) {
  const [accountBalance, setAccountBalance] = useState(DEFAULT_EQUITY);
  const [accountEquity, setAccountEquity] = useState(DEFAULT_EQUITY);
  const [statementPnl, setStatementPnl] = useState(0);
  const [prices, setPrices] = useState<Record<string, LatestPrice>>({});
  const [isLoadingPrices, setIsLoadingPrices] = useState(false);

  useEffect(() => {
    let cancelled = false;

    async function loadPortfolioEquity() {
      if (!activePortfolio) {
        return;
      }

      let meta = activePortfolio ? portfolioMeta?.[activePortfolio] : pickLatestImportedMeta(portfolioMeta);

      if (!meta) {
        const response = await fetch("/api/settings/branden-portfolios", { cache: "no-store" });
        const data = (await response.json().catch(() => ({}))) as PortfolioSettingsResponse;

        if (!response.ok || cancelled) {
          return;
        }

        meta = activePortfolio ? data?.portfolioMeta?.[activePortfolio] : pickLatestImportedMeta(data?.portfolioMeta);
      }

      if (cancelled) {
        return;
      }

      const statementBalance = Number(meta?.currentEquity);
      const statementEquity = Number(meta?.statementEquity);
      const importedFloatingPnl = Number(meta?.floatingPnl);
      const hasBalance = Number.isFinite(statementBalance) && statementBalance > 0;
      const hasEquity = Number.isFinite(statementEquity) && statementEquity > 0;

      if (hasBalance || hasEquity) {
        const nextBalance = hasBalance ? statementBalance : statementEquity;
        const nextEquity = hasEquity ? statementEquity : statementBalance;
        setAccountBalance(nextBalance);
        setAccountEquity(nextEquity);

        const derivedPnl = hasBalance && hasEquity ? nextEquity - nextBalance : importedFloatingPnl;
        setStatementPnl(Number.isFinite(derivedPnl) ? derivedPnl : 0);
      }
    }

    loadPortfolioEquity();

    return () => {
      cancelled = true;
    };
  }, [activePortfolio, portfolioMeta]);

  const openPositions = useMemo(
    () =>
      trades
        .filter((trade) => trade.status === "OPEN")
        .filter((trade) => !activePortfolio || trade.portfolioTag === activePortfolio)
        .sort((a, b) => a.symbol.localeCompare(b.symbol)),
    [trades, activePortfolio]
  );

  const symbols = useMemo(
    () => Array.from(new Set(openPositions.map((trade) => trade.symbol).filter(Boolean))),
    [openPositions]
  );

  useEffect(() => {
    let isMounted = true;

    async function loadPrices() {
      if (!symbols.length) {
        setPrices({});
        return;
      }

      setIsLoadingPrices(true);
      const results = await Promise.all(symbols.map(async (symbol) => [symbol, await fetchLatestPrice(symbol)] as const));

      if (isMounted) {
        setPrices(Object.fromEntries(results));
        setIsLoadingPrices(false);
      }
    }

    loadPrices().catch(() => {
      if (isMounted) {
        setIsLoadingPrices(false);
      }
    });

    return () => {
      isMounted = false;
    };
  }, [symbols.join("|")]);

  const riskRows = useMemo(
    () => openPositions.map((trade) => buildRiskRow(trade, prices[trade.symbol], accountEquity)),
    [openPositions, prices, accountEquity]
  );

  const totalOpenHeat = riskRows.reduce((sum, row) => sum + row.dollarRisk, 0);
  const netStopPnl = riskRows.reduce((sum, row) => sum + row.stopOutcome, 0);
  const profitableStopProfit = riskRows.reduce((sum, row) => sum + Math.max(0, row.stopOutcome), 0);
  const balanceAtRiskPct = accountEquity ? (totalOpenHeat / accountEquity) * 100 : 0;
  const statementPnlPct = accountEquity ? (statementPnl / accountEquity) * 100 : 0;
  const netStopPnlPct = accountEquity ? (netStopPnl / accountEquity) * 100 : 0;
  const distanceToDrawdownFloor = accountEquity - DEFAULT_DRAWDOWN_FLOOR;
  const survivalBuffer = distanceToDrawdownFloor - totalOpenHeat;
  const worstCaseEquity = accountEquity - totalOpenHeat;
  const missingRiskCount = riskRows.filter((row) => row.status === "missing").length;
  const newTradesAllowed = missingRiskCount === 0 && survivalBuffer > 0;
  const warningText = missingRiskCount
    ? `${missingRiskCount} open ${missingRiskCount === 1 ? "position is" : "positions are"} missing stop/risk data.`
    : survivalBuffer <= 0
      ? "Open heat would break the drawdown floor if every stop is hit."
      : "Open heat is inside the current survival buffer.";

  return (
    <section className="open-heat-panel">
      <div className="column-settings-head">
        <div>
          <p className="eyebrow">REBAR / Balance at Risk</p>
          <h3>Open Positions</h3>
        </div>
        <span className={`open-heat-status ${newTradesAllowed ? "allowed" : "blocked"}`}>
          {newTradesAllowed ? "New trades allowed" : "Block new trades"}
        </span>
      </div>

      <div className="open-heat-kpi-grid">
        <article>
          <span>Statement balance</span>
          <strong>{formatCurrency(accountBalance)}</strong>
        </article>
        <article>
          <span>Statement equity</span>
          <strong>{formatCurrency(accountEquity)}</strong>
        </article>
        <article>
          <span>B-A-R / giveback $</span>
          <strong className={totalOpenHeat > 0 ? "open-heat-negative" : ""}>{formatCurrency(totalOpenHeat)}</strong>
        </article>
        <article>
          <span>B-A-R / giveback %</span>
          <strong className={balanceAtRiskPct > 0 ? "open-heat-negative" : ""}>-{formatPercent(balanceAtRiskPct)}</strong>
        </article>
        <article>
          <span>Statement P&L $</span>
          <strong className={statementPnl >= 0 ? "open-heat-positive" : "open-heat-negative"}>{formatCurrency(statementPnl)}</strong>
        </article>
        <article>
          <span>Statement P&L %</span>
          <strong className={statementPnlPct >= 0 ? "open-heat-positive" : "open-heat-negative"}>{formatPercent(statementPnlPct)}</strong>
        </article>
        <article>
          <span>Net stop P&L</span>
          <strong className={netStopPnl >= 0 ? "open-heat-positive" : "open-heat-negative"}>{formatCurrency(netStopPnl)}</strong>
        </article>
        <article>
          <span>Net stop P&L %</span>
          <strong className={netStopPnlPct >= 0 ? "open-heat-positive" : "open-heat-negative"}>{formatPercent(netStopPnlPct)}</strong>
        </article>
        <article>
          <span>Profitable stops locked</span>
          <strong className={profitableStopProfit > 0 ? "open-heat-positive" : ""}>{formatCurrency(profitableStopProfit)}</strong>
        </article>
        <article>
          <span>Worst-case equity</span>
          <strong>{formatCurrency(worstCaseEquity)}</strong>
        </article>
        <article>
          <span>Distance to drawdown floor</span>
          <strong>{formatCurrency(distanceToDrawdownFloor)}</strong>
        </article>
        <article>
          <span>Survival buffer</span>
          <strong className={survivalBuffer >= 0 ? "open-heat-positive" : "open-heat-negative"}>{formatCurrency(survivalBuffer)}</strong>
        </article>
      </div>

      <div className="open-heat-warning">
        <strong>{warningText}</strong>
      </div>

      <div className="open-heat-table-wrap">
        <table className="open-heat-table">
          <thead>
            <tr>
              <th>Symbol</th>
              <th>Side</th>
              <th>Shares</th>
              <th>Entry</th>
              <th>Current</th>
              <th>Floating $</th>
              <th>Floating %</th>
              <th>Stop</th>
              <th>Position value</th>
              <th>Portfolio weight</th>
              <th>Stop P&L</th>
              <th>B-A-R $</th>
              <th>B-A-R %</th>
            </tr>
          </thead>
          <tbody>
            {riskRows.map((row) => (
              <tr
                key={row.trade.id}
                className={onSelectTrade ? "clickable-row" : undefined}
                onClick={() => onSelectTrade?.(row.trade.id)}
              >
                <td>{row.trade.symbol}</td>
                <td>{row.trade.side}</td>
                <td>{formatPrice(row.trade.shares)}</td>
                <td>{formatPrice(row.trade.avgEntry)}</td>
                <td>
                  {formatPrice(row.currentPrice)}
                  {row.priceDate ? <span>{row.priceDate}</span> : null}
                </td>
                <td className={row.floatingPnl >= 0 ? "open-heat-positive" : "open-heat-negative"}>
                  {row.floatingPnl >= 0 ? "+" : ""}
                  {formatCurrency(row.floatingPnl)}
                </td>
                <td className={row.floatingPct >= 0 ? "open-heat-positive" : "open-heat-negative"}>
                  {row.floatingPct >= 0 ? "+" : ""}
                  {formatPercent(row.floatingPct)}
                </td>
                <td>{formatPrice(row.trade.stopPrice)}</td>
                <td>{formatCurrency(row.positionValue)}</td>
                <td>{formatPercent(row.weightPct)}</td>
                <td className={row.stopOutcome >= 0 ? "open-heat-positive" : "open-heat-negative"}>
                  {row.stopOutcome >= 0 ? "+" : ""}
                  {formatCurrency(row.stopOutcome)}
                </td>
                <td className={row.dollarRisk ? "open-heat-negative" : ""}>{formatCurrency(row.dollarRisk)}</td>
                <td className={row.riskPct ? "open-heat-negative" : ""}>-{formatPercent(row.riskPct)}</td>
              </tr>
            ))}
            {!riskRows.length ? (
              <tr>
                <td className="trade-empty" colSpan={13}>
                  No open positions in this portfolio.
                </td>
              </tr>
            ) : null}
          </tbody>
        </table>
      </div>
    </section>
  );
}
