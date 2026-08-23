"use client";

import { useEffect, useMemo, useState } from "react";
import type { BrokerPortfolioPosition, BrokerPortfolioSnapshot } from "@/lib/broker-portfolio-snapshot";
import type { TradeLogEntry } from "@/lib/types";

type Props = {
  trades: TradeLogEntry[];
  activePortfolio: string;
  onSelectTrade?: (tradeId: string) => void;
  portfolioMeta?: Record<string, PortfolioMeta>;
  brokerPortfolioSnapshots?: BrokerPortfolioSnapshot[];
};

type LatestPrice = {
  price: number | null;
  date: string;
  error?: string;
};

type RiskRow = {
  trade: TradeLogEntry;
  shares: number;
  entryPrice: number;
  stopLabel: string;
  currentPrice: number | null;
  priceDate: string;
  positionValue: number | null;
  floatingPnl: number | null;
  floatingPct: number | null;
  dollarRisk: number | null;
  stopOutcome: number | null;
  riskPct: number | null;
  weightPct: number | null;
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
function formatCurrency(value: number | null) {
  if (value === null || !Number.isFinite(value)) return "—";
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0
  }).format(Number.isFinite(value) ? value : 0);
}

function formatPrice(value: number | null) {
  if (value === null || !Number.isFinite(value) || value === 0) {
    return "-";
  }

  return new Intl.NumberFormat("en-US", {
    maximumFractionDigits: value < 10 ? 4 : 2
  }).format(value);
}

function formatPercent(value: number | null) {
  return value === null || !Number.isFinite(value) ? "—" : `${value.toFixed(2)}%`;
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
    return { price: null, date: "", error: data.error || `Could not load ${symbol}.` };
  }

  const candles = Array.isArray(data.candles) ? data.candles : [];
  const latest = [...candles].reverse().find((candle) => {
    const close = Number(candle?.close);
    return Boolean(candle?.time) && Number.isFinite(close) && close > 0;
  });

  return {
    price: Number(latest?.close) || null,
    date: latest?.time ? String(latest.time) : ""
  };
}

function positive(value: unknown) {
  const number = Number(value);
  return Number.isFinite(number) && number > 0 ? number : null;
}

function brokerPositionForTrade(snapshot: BrokerPortfolioSnapshot | undefined, trade: TradeLogEntry) {
  if (
    snapshot
    && (trade.entryDate > snapshot.coverageDate
      || (trade.executions || []).some((execution) => execution.date > snapshot.coverageDate))
  ) {
    return undefined;
  }
  return snapshot?.openPositions.find(
    (position) => position.symbol.toUpperCase() === trade.symbol.toUpperCase() && position.side === trade.side
  );
}

function protectiveLevels(
  trade: TradeLogEntry,
  shares: number,
  storedStop: number | null,
  snapshot: BrokerPortfolioSnapshot | undefined
) {
  const exitDirection = trade.side === "LONG" ? "Sell" : "Buy";
  const stopOrders = (snapshot?.workingOrders || [])
    .filter((order) =>
      order.symbol.toUpperCase() === trade.symbol.toUpperCase()
      && order.direction === exitDirection
      && order.orderType === "STOP"
      && order.shares > 0
      && order.orderPrice > 0
    )
    .sort((a, b) => trade.side === "LONG" ? b.orderPrice - a.orderPrice : a.orderPrice - b.orderPrice);

  if (!stopOrders.length) {
    return storedStop ? [{ price: storedStop, quantity: shares }] : [];
  }

  const bracketStop = stopOrders.at(-1)!;
  const stagedStops = stopOrders.slice(0, -1);
  let remaining = shares;
  const levels: Array<{ price: number; quantity: number }> = [];
  stagedStops.forEach((order) => {
    const quantity = Math.min(order.shares, remaining);
    if (quantity > 0) {
      levels.push({ price: order.orderPrice, quantity });
      remaining -= quantity;
    }
  });
  if (remaining > 0) levels.push({ price: bracketStop.orderPrice, quantity: remaining });
  return levels;
}

export function buildOpenPositionRiskRow(
  trade: TradeLogEntry,
  price: LatestPrice | undefined,
  currentEquity: number | null,
  brokerPosition: BrokerPortfolioPosition | undefined,
  brokerSnapshot: BrokerPortfolioSnapshot | undefined
): RiskRow {
  const shares = positive(brokerPosition?.shares) ?? positive(trade.shares) ?? 0;
  const entryPrice = positive(brokerPosition?.entryPrice) ?? positive(trade.avgEntry) ?? 0;
  const marketPrice = positive(price?.price);
  const statementPrice = positive(brokerPosition?.currentPrice);
  const currentPrice = marketPrice ?? statementPrice;
  const stopPrice = positive(brokerPosition?.stopPrice) ?? positive(trade.stopPrice);
  const storedRisk = Math.abs(Number(trade.risk) || 0);
  const positionValue = currentPrice && shares ? currentPrice * shares : null;
  const floatingPnl =
    entryPrice && currentPrice && shares
      ? trade.side === "SHORT"
        ? (entryPrice - currentPrice) * shares
        : (currentPrice - entryPrice) * shares
      : null;
  const floatingPct =
    entryPrice && currentPrice
      ? trade.side === "SHORT"
        ? ((entryPrice - currentPrice) / entryPrice) * 100
        : ((currentPrice - entryPrice) / entryPrice) * 100
      : null;
  const levels = protectiveLevels(trade, shares, stopPrice, brokerSnapshot);
  let stopOutcome: number | null = null;
  let barRisk: number | null = null;
  let status: RiskRow["status"] = "missing";
  let note = "Needs quantity-aware stop or saved risk data.";

  if (entryPrice && currentPrice && shares && levels.length) {
    stopOutcome = levels.reduce(
      (total, level) => total + (trade.side === "SHORT"
        ? (entryPrice - level.price) * level.quantity
        : (level.price - entryPrice) * level.quantity),
      0
    );
    barRisk = levels.reduce(
      (total, level) => total + Math.max(0, trade.side === "SHORT"
        ? (level.price - currentPrice) * level.quantity
        : (currentPrice - level.price) * level.quantity),
      0
    );
    status = "ready";
    note = stopOutcome >= 0 ? "Stop locks profit versus average cost; BAR is current giveback to stop." : "Stop loss versus average cost.";
  } else if (storedRisk) {
    stopOutcome = -storedRisk;
    barRisk = storedRisk;
    status = "fallback";
    note = "Using the Trade Log's saved planned-risk fallback.";
  } else if (price?.error) {
    note = price.error;
  }

  return {
    trade,
    shares,
    entryPrice,
    stopLabel: levels.map((level) => `${formatPrice(level.quantity)} @ ${formatPrice(level.price)}`).join("; ") || "—",
    currentPrice,
    priceDate: marketPrice ? price?.date || "" : statementPrice ? brokerSnapshot?.coverageDate || "" : "",
    positionValue,
    floatingPnl,
    floatingPct,
    dollarRisk: barRisk,
    stopOutcome,
    riskPct: currentEquity && barRisk !== null ? (barRisk / currentEquity) * 100 : null,
    weightPct: currentEquity && positionValue !== null ? (positionValue / currentEquity) * 100 : null,
    status,
    note
  };
}

export default function OpenHeatDashboard({ trades, activePortfolio, onSelectTrade, portfolioMeta, brokerPortfolioSnapshots = [] }: Props) {
  const [prices, setPrices] = useState<Record<string, LatestPrice>>({});
  const [isLoadingPrices, setIsLoadingPrices] = useState(false);

  const brokerSnapshot = useMemo(
    () => brokerPortfolioSnapshots
      .filter((snapshot) => !activePortfolio || snapshot.portfolioTag === activePortfolio)
      .sort((a, b) => b.coverageDate.localeCompare(a.coverageDate))[0],
    [activePortfolio, brokerPortfolioSnapshots]
  );
  const importedMeta = activePortfolio ? portfolioMeta?.[activePortfolio] : pickLatestImportedMeta(portfolioMeta);
  const accountBalance = positive(brokerSnapshot?.balance) ?? positive(importedMeta?.currentEquity);
  const accountEquity = positive(brokerSnapshot?.currentEquity)
    ?? positive(brokerSnapshot?.statementEquity)
    ?? positive(importedMeta?.statementEquity);
  const importedPnl = Number(brokerSnapshot?.floatingPnl ?? importedMeta?.floatingPnl);
  const statementPnl = Number.isFinite(importedPnl)
    ? importedPnl
    : accountBalance !== null && accountEquity !== null
      ? accountEquity - accountBalance
      : null;

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
    () => openPositions.map((trade) => {
      const brokerPosition = brokerPositionForTrade(brokerSnapshot, trade);
      return buildOpenPositionRiskRow(
        trade,
        prices[trade.symbol],
        accountEquity,
        brokerPosition,
        brokerPosition ? brokerSnapshot : undefined
      );
    }),
    [openPositions, prices, accountEquity, brokerSnapshot]
  );

  const missingRiskCount = riskRows.filter((row) => row.status === "missing").length;
  const fallbackRiskCount = riskRows.filter((row) => row.status === "fallback").length;
  const totalOpenHeat = missingRiskCount ? null : riskRows.reduce((sum, row) => sum + (row.dollarRisk || 0), 0);
  const netStopPnl = missingRiskCount ? null : riskRows.reduce((sum, row) => sum + (row.stopOutcome || 0), 0);
  const profitableStopProfit = missingRiskCount
    ? null
    : riskRows.reduce((sum, row) => sum + Math.max(0, row.stopOutcome || 0), 0);
  const balanceAtRiskPct = accountEquity && totalOpenHeat !== null ? (totalOpenHeat / accountEquity) * 100 : null;
  const statementPnlPct = accountEquity && statementPnl !== null ? (statementPnl / accountEquity) * 100 : null;
  const netStopPnlPct = accountEquity && netStopPnl !== null ? (netStopPnl / accountEquity) * 100 : null;
  const worstCaseEquity = accountEquity && totalOpenHeat !== null ? accountEquity - totalOpenHeat : null;
  const latestPriceDate = riskRows.map((row) => row.priceDate).filter(Boolean).sort().at(-1) || "—";
  const riskState = missingRiskCount ? "incomplete" : fallbackRiskCount ? "fallback" : "complete";
  const warningText = missingRiskCount
    ? `${missingRiskCount} open ${missingRiskCount === 1 ? "position is" : "positions are"} missing stop/risk data.`
    : fallbackRiskCount
      ? `${fallbackRiskCount} ${fallbackRiskCount === 1 ? "position uses" : "positions use"} saved Trade Log risk because a current quantity-aware stop plan is unavailable.`
      : "Every open position has a current quantity-aware downside-risk calculation.";

  return (
    <section className="open-heat-panel">
      <div className="column-settings-head">
        <div>
          <p className="eyebrow">REBAR / Balance at Risk</p>
          <h3>Open Positions</h3>
        </div>
        <span className={`open-heat-status ${riskState === "complete" ? "allowed" : "blocked"}`}>
          {riskState === "complete" ? "Risk data complete" : riskState === "fallback" ? "Trade Log fallback" : "Risk data incomplete"}
        </span>
      </div>

      <div className="open-heat-warning">
        <strong>Position state through {brokerSnapshot?.coverageDate || importedMeta?.equityStatementDate || "Trade Log"}. Prices through {latestPriceDate}.</strong>
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
          <strong className={totalOpenHeat !== null && totalOpenHeat > 0 ? "open-heat-negative" : ""}>{formatCurrency(totalOpenHeat)}</strong>
        </article>
        <article>
          <span>B-A-R / giveback %</span>
          <strong className={balanceAtRiskPct !== null && balanceAtRiskPct > 0 ? "open-heat-negative" : ""}>{balanceAtRiskPct === null ? "—" : `-${formatPercent(balanceAtRiskPct)}`}</strong>
        </article>
        <article>
          <span>Statement P&L $</span>
          <strong className={statementPnl !== null && statementPnl >= 0 ? "open-heat-positive" : "open-heat-negative"}>{formatCurrency(statementPnl)}</strong>
        </article>
        <article>
          <span>Statement P&L %</span>
          <strong className={statementPnlPct !== null && statementPnlPct >= 0 ? "open-heat-positive" : "open-heat-negative"}>{formatPercent(statementPnlPct)}</strong>
        </article>
        <article>
          <span>Net stop P&L</span>
          <strong className={netStopPnl !== null && netStopPnl >= 0 ? "open-heat-positive" : "open-heat-negative"}>{formatCurrency(netStopPnl)}</strong>
        </article>
        <article>
          <span>Net stop P&L %</span>
          <strong className={netStopPnlPct !== null && netStopPnlPct >= 0 ? "open-heat-positive" : "open-heat-negative"}>{formatPercent(netStopPnlPct)}</strong>
        </article>
        <article>
          <span>Profitable stops locked</span>
          <strong className={profitableStopProfit !== null && profitableStopProfit > 0 ? "open-heat-positive" : ""}>{formatCurrency(profitableStopProfit)}</strong>
        </article>
        <article>
          <span>Worst-case equity</span>
          <strong>{formatCurrency(worstCaseEquity)}</strong>
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
                <td>{formatPrice(row.shares)}</td>
                <td>{formatPrice(row.entryPrice)}</td>
                <td>
                  {formatPrice(row.currentPrice)}
                  {row.priceDate ? <span>{row.priceDate}</span> : null}
                </td>
                <td className={row.floatingPnl !== null && row.floatingPnl >= 0 ? "open-heat-positive" : "open-heat-negative"}>
                  {row.floatingPnl !== null && row.floatingPnl >= 0 ? "+" : ""}
                  {formatCurrency(row.floatingPnl)}
                </td>
                <td className={row.floatingPct !== null && row.floatingPct >= 0 ? "open-heat-positive" : "open-heat-negative"}>
                  {row.floatingPct !== null && row.floatingPct >= 0 ? "+" : ""}
                  {formatPercent(row.floatingPct)}
                </td>
                <td>{row.stopLabel}</td>
                <td>{formatCurrency(row.positionValue)}</td>
                <td>{formatPercent(row.weightPct)}</td>
                <td className={row.stopOutcome !== null && row.stopOutcome >= 0 ? "open-heat-positive" : "open-heat-negative"}>
                  {row.stopOutcome !== null && row.stopOutcome >= 0 ? "+" : ""}
                  {formatCurrency(row.stopOutcome)}
                </td>
                <td className={row.dollarRisk ? "open-heat-negative" : ""}>{formatCurrency(row.dollarRisk)}</td>
                <td className={row.riskPct ? "open-heat-negative" : ""}>{row.riskPct === null ? "—" : `-${formatPercent(row.riskPct)}`}</td>
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
