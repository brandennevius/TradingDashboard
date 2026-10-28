"use client";

import { usePortfolioSelection } from "@/lib/portfolio-selection";

import { useEffect, useMemo, useState } from "react";
import {
  resolveDailyReviewProvenance,
  type BrokerPortfolioSnapshot
} from "@/lib/broker-portfolio-snapshot";
import type { TradeExecution, TradeLogEntry, TraderUser } from "@/lib/types";

type Candle = {
  time: string;
  close: number;
};

type DailyActivity = {
  id: string;
  kind: "opened" | "added" | "reduced" | "closed";
  symbol: string;
  side: TradeLogEntry["side"];
  shares: number;
  price: number;
  realizedPnl: number;
};

type PerformanceRow = {
  symbol: string;
  side: TradeLogEntry["side"];
  startShares: number;
  endShares: number;
  previousClose: number;
  close: number;
  returnPercent: number;
  dailyPnl: number;
};

function sortedUnique(values: string[]) {
  return Array.from(new Set(values.map((value) => value.trim()).filter(Boolean))).sort((a, b) => a.localeCompare(b));
}

function formatCurrency(value: number) {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: 2,
    maximumFractionDigits: 2
  }).format(value);
}

function formatNumber(value: number) {
  return new Intl.NumberFormat("en-US", { maximumFractionDigits: 2 }).format(value);
}

function formatPercent(value: number) {
  return `${value >= 0 ? "+" : ""}${value.toFixed(2)}%`;
}

function formatDate(value: string) {
  if (!value) {
    return "";
  }

  return new Intl.DateTimeFormat("en-US", {
    weekday: "long",
    month: "long",
    day: "numeric",
    year: "numeric",
    timeZone: "UTC"
  }).format(new Date(`${value}T12:00:00Z`));
}

function weightedPrice(executions: TradeExecution[]) {
  const shares = executions.reduce((sum, execution) => sum + execution.shares, 0);
  return shares ? executions.reduce((sum, execution) => sum + execution.price * execution.shares, 0) / shares : 0;
}

function normalizedExecutions(trade: TradeLogEntry): TradeExecution[] {
  if (trade.executions?.length) {
    return trade.executions;
  }

  const executions: TradeExecution[] = [];

  if (trade.entryDate && trade.shares > 0) {
    executions.push({
      id: `${trade.id}-entry`,
      type: "ENTRY",
      date: trade.entryDate,
      time: trade.openTime,
      side: trade.side,
      shares: trade.shares,
      price: trade.avgEntry,
      pnl: 0,
      commission: 0,
      source: "trade",
      sourceKey: ""
    });
  }

  if (trade.exitDate && trade.status !== "OPEN" && trade.shares > 0) {
    executions.push({
      id: `${trade.id}-exit`,
      type: "EXIT",
      date: trade.exitDate,
      time: trade.closeTime,
      side: trade.side,
      shares: trade.shares,
      price: trade.exitPrice,
      pnl: trade.pnl,
      commission: trade.commission,
      source: "trade",
      sourceKey: ""
    });
  }

  return executions;
}

function sharesBefore(executions: TradeExecution[], date: string) {
  return executions
    .filter((execution) => execution.date < date)
    .reduce((sum, execution) => sum + (execution.type === "ENTRY" ? execution.shares : -execution.shares), 0);
}

function sideDirection(side: TradeLogEntry["side"]) {
  return side === "SHORT" ? -1 : 1;
}

function executionTimestamp(execution: TradeExecution) {
  return `${execution.date}T${execution.time || "00:00:00"}`;
}

function calculateDailyPnl(
  startingShares: number,
  executions: TradeExecution[],
  previousClose: number,
  close: number,
  direction: number
) {
  const lots: Array<{ shares: number; basis: number }> = startingShares > 0
    ? [{ shares: startingShares, basis: previousClose }]
    : [];
  let dailyPnl = 0;

  [...executions].sort((a, b) => executionTimestamp(a).localeCompare(executionTimestamp(b))).forEach((execution) => {
    if (execution.type === "ENTRY") {
      lots.push({ shares: execution.shares, basis: execution.price });
      dailyPnl -= execution.commission;
      return;
    }

    let sharesToExit = execution.shares;

    while (sharesToExit > 0.000001 && lots.length) {
      const lot = lots[0];
      const matchedShares = Math.min(sharesToExit, lot.shares);
      dailyPnl += matchedShares * (execution.price - lot.basis) * direction;
      lot.shares -= matchedShares;
      sharesToExit -= matchedShares;

      if (lot.shares <= 0.000001) {
        lots.shift();
      }
    }

    dailyPnl -= execution.commission;
  });

  lots.forEach((lot) => {
    dailyPnl += lot.shares * (close - lot.basis) * direction;
  });

  return dailyPnl;
}

function activityLabel(kind: DailyActivity["kind"]) {
  if (kind === "opened") return "Opened";
  if (kind === "added") return "Added";
  if (kind === "reduced") return "Trimmed";
  return "Closed";
}

export default function DailyReviewPage() {
  const [user, setUser] = useState<TraderUser | null>(null);
  const [trades, setTrades] = useState<TradeLogEntry[]>([]);
  const [brokerPortfolioSnapshots, setBrokerPortfolioSnapshots] = useState<BrokerPortfolioSnapshot[]>([]);
  const [activePortfolio, setActivePortfolio, initializePortfolio] = usePortfolioSelection("");
  const [selectedDate, setSelectedDate] = useState("");
  const [marketSeries, setMarketSeries] = useState<Record<string, Candle[]>>({});
  const [isLoading, setIsLoading] = useState(true);
  const [isLoadingMarket, setIsLoadingMarket] = useState(false);
  const [error, setError] = useState("");
  const [contributionDescending, setContributionDescending] = useState(true);

  useEffect(() => {
    let cancelled = false;

    async function loadPage() {
      const reviewResponse = await fetch("/api/journal/branden/daily-review", { cache: "no-store" });
      const reviewData = await reviewResponse.json().catch(() => ({}));

      if (cancelled) {
        return;
      }

      if (!reviewResponse.ok || !reviewData.user) {
        setError(reviewData.error || "Sign in to view Daily Review.");
        setIsLoading(false);
        return;
      }

      const nextTrades: TradeLogEntry[] = Array.isArray(reviewData.trades) ? reviewData.trades : [];
      const params = new URLSearchParams(window.location.search);
      const latestExecutionDate = nextTrades
        .flatMap((trade) => normalizedExecutions(trade).map((execution) => execution.date))
        .filter(Boolean)
        .sort()
        .at(-1);

      setUser(reviewData.user || null);
      setTrades(nextTrades);
      setBrokerPortfolioSnapshots(Array.isArray(reviewData.brokerPortfolioSnapshots) ? reviewData.brokerPortfolioSnapshots : []);
      initializePortfolio(String(reviewData.defaultPortfolio || ""));
      setSelectedDate(params.get("date") || latestExecutionDate || new Date().toISOString().slice(0, 10));
      setIsLoading(false);
    }

    loadPage().catch((loadError) => {
      if (!cancelled) {
        setError(loadError instanceof Error ? loadError.message : "Could not load the daily review.");
        setIsLoading(false);
      }
    });

    return () => {
      cancelled = true;
    };
  }, []);

  const brandenTrades = useMemo(
    () =>
      trades.filter(
        (trade) => trade.userId === "branden" && !trade.hidden && (!activePortfolio || trade.portfolioTag === activePortfolio)
      ),
    [activePortfolio, trades]
  );

  const provenance = useMemo(
    () => resolveDailyReviewProvenance(brokerPortfolioSnapshots, activePortfolio, selectedDate),
    [activePortfolio, brokerPortfolioSnapshots, selectedDate]
  );

  const activities = useMemo<DailyActivity[]>(() => {
    const rows: DailyActivity[] = [];

    brandenTrades.forEach((trade) => {
      const executions = normalizedExecutions(trade);
      const startingShares = sharesBefore(executions, selectedDate);
      const entries = executions.filter((execution) => execution.date === selectedDate && execution.type === "ENTRY");
      const exits = executions.filter((execution) => execution.date === selectedDate && execution.type === "EXIT");
      const entryShares = entries.reduce((sum, execution) => sum + execution.shares, 0);
      const exitShares = exits.reduce((sum, execution) => sum + execution.shares, 0);
      const endingShares = Math.max(0, startingShares + entryShares - exitShares);

      if (entryShares > 0) {
        rows.push({
          id: `${trade.id}-entry-${selectedDate}`,
          kind: startingShares > 0 ? "added" : "opened",
          symbol: trade.symbol,
          side: trade.side,
          shares: entryShares,
          price: weightedPrice(entries),
          realizedPnl: 0
        });
      }

      if (exitShares > 0) {
        rows.push({
          id: `${trade.id}-exit-${selectedDate}`,
          kind: endingShares > 0 ? "reduced" : "closed",
          symbol: trade.symbol,
          side: trade.side,
          shares: exitShares,
          price: weightedPrice(exits),
          realizedPnl: exits.reduce((sum, execution) => sum + execution.pnl - execution.commission, 0)
        });
      }
    });

    return rows.sort((a, b) => a.symbol.localeCompare(b.symbol));
  }, [brandenTrades, selectedDate]);

  const relevantSymbols = useMemo(() => {
    const dailyExecutionSymbols = brandenTrades
      .filter((trade) => normalizedExecutions(trade).some((execution) => execution.date === selectedDate))
      .map((trade) => trade.symbol);

    if (provenance.kind === "BROKER_SNAPSHOT") {
      const snapshotSymbols = provenance.snapshot?.openPositions.map((position) => position.symbol) || [];
      return sortedUnique([...snapshotSymbols, ...dailyExecutionSymbols]);
    }

    return sortedUnique(brandenTrades
      .filter((trade) => {
        const executions = normalizedExecutions(trade);
        return sharesBefore(executions, selectedDate) > 0 || executions.some((execution) => execution.date === selectedDate);
      })
      .map((trade) => trade.symbol));
  }, [brandenTrades, provenance, selectedDate]);

  useEffect(() => {
    let cancelled = false;

    async function loadMarketData() {
      if (!relevantSymbols.length || !selectedDate) {
        setMarketSeries({});
        return;
      }

      setIsLoadingMarket(true);
      const results = await Promise.all(
        relevantSymbols.map(async (symbol) => {
          const response = await fetch(`/api/market-data/${encodeURIComponent(symbol)}?timeframe=1d`);
          const data = await response.json().catch(() => ({}));
          return [symbol, response.ok && Array.isArray(data.candles) ? data.candles : []] as const;
        })
      );

      if (!cancelled) {
        setMarketSeries(Object.fromEntries(results));
        setIsLoadingMarket(false);
      }
    }

    loadMarketData().catch(() => {
      if (!cancelled) {
        setMarketSeries({});
        setIsLoadingMarket(false);
      }
    });

    return () => {
      cancelled = true;
    };
  }, [relevantSymbols, selectedDate]);

  useEffect(() => {
    if (!selectedDate || isLoading) {
      return;
    }

    const url = new URL(window.location.href);
    url.searchParams.set("date", selectedDate);
    window.history.replaceState(null, "", `${url.pathname}${url.search}`);
  }, [isLoading, selectedDate]);

  const performance = useMemo<PerformanceRow[]>(() => {
    return relevantSymbols
      .map((symbol) => {
        const symbolTrades = brandenTrades.filter((trade) => trade.symbol === symbol);
        const snapshotPositions = provenance.kind === "BROKER_SNAPSHOT"
          ? provenance.snapshot?.openPositions.filter((position) => position.symbol === symbol) || []
          : [];
        const side = snapshotPositions[0]?.side || symbolTrades[0]?.side || "LONG";
        const allExecutions = symbolTrades.flatMap(normalizedExecutions);
        const entries = allExecutions.filter((execution) => execution.date === selectedDate && execution.type === "ENTRY");
        const exits = allExecutions.filter((execution) => execution.date === selectedDate && execution.type === "EXIT");
        const entryShares = entries.reduce((sum, execution) => sum + execution.shares, 0);
        const exitShares = exits.reduce((sum, execution) => sum + execution.shares, 0);
        const reconstructedStartingShares = sharesBefore(allExecutions, selectedDate);
        const reconstructedEndingShares = Math.max(0, reconstructedStartingShares + entryShares - exitShares);
        const snapshotShares = snapshotPositions.length
          ? snapshotPositions.reduce((sum, position) => sum + position.shares, 0)
          : null;
        const endingShares = snapshotShares ?? reconstructedEndingShares;
        const startingShares = snapshotShares !== null
          ? Math.max(0, endingShares - entryShares + exitShares)
          : reconstructedStartingShares;
        const candles = (marketSeries[symbol] || []).filter((candle) => candle.time <= selectedDate);
        const selectedIndex = candles.findIndex((candle) => candle.time === selectedDate);
        const close = selectedIndex >= 0 ? candles[selectedIndex]?.close || 0 : 0;
        const previousClose = selectedIndex > 0 ? candles[selectedIndex - 1]?.close || 0 : 0;

        if (!close || !previousClose) {
          return null;
        }

        const direction = sideDirection(side);
        const dailyExecutions = allExecutions.filter((execution) => execution.date === selectedDate);

        return {
          symbol,
          side,
          startShares: startingShares,
          endShares: endingShares,
          previousClose,
          close,
          returnPercent: ((close - previousClose) / previousClose) * 100 * direction,
          dailyPnl: calculateDailyPnl(startingShares, dailyExecutions, previousClose, close, direction)
        };
      })
      .filter((row): row is PerformanceRow => row !== null)
      .sort((a, b) => b.returnPercent - a.returnPercent);
  }, [brandenTrades, marketSeries, provenance, relevantSymbols, selectedDate]);

  const missingMarketSymbols = useMemo(() => relevantSymbols.filter((symbol) => {
    const candles = (marketSeries[symbol] || []).filter((candle) => candle.time <= selectedDate);
    const selectedIndex = candles.findIndex((candle) => candle.time === selectedDate);
    return selectedIndex <= 0 || !candles[selectedIndex]?.close || !candles[selectedIndex - 1]?.close;
  }), [marketSeries, relevantSymbols, selectedDate]);
  const dailyPnl = !isLoadingMarket && performance.length ? performance.reduce((sum, row) => sum + row.dailyPnl, 0) : null;
  const startingMarketValue = performance.reduce((sum, row) => sum + Math.abs(row.startShares * row.previousClose), 0);
  const dailyReturn = dailyPnl !== null && startingMarketValue ? (dailyPnl / startingMarketValue) * 100 : null;
  const activitySummary = (["closed", "opened", "added", "reduced"] as const)
    .map((kind) => ({ kind, count: activities.filter((activity) => activity.kind === kind).length }))
    .filter(({ count }) => count > 0)
    .map(({ kind, count }) => `${count} ${kind === "reduced" ? "trimmed" : kind}`)
    .join(" · ");
  const sortedPerformance = [...performance].sort((a, b) => contributionDescending ? b.dailyPnl - a.dailyPnl : a.dailyPnl - b.dailyPnl);
  const maxContribution = Math.max(...performance.map((row) => Math.abs(row.dailyPnl)), 1);

  function moveDate(offset: number) {
    if (!selectedDate) return;
    const date = new Date(`${selectedDate}T12:00:00Z`);
    date.setUTCDate(date.getUTCDate() + offset);
    setSelectedDate(date.toISOString().slice(0, 10));
  }

  const selectedPortfolioEquity = provenance.accountEquity;

  return (
    <div className="branden-journal-content daily-review-page">
        <header className="branden-route-header daily-review-header">
          <div>
            <p className="eyebrow">Journal / Daily Review</p>
            <h1>Daily Review</h1>
            <p className="daily-review-subtitle">Your portfolio, at a glance.</p>
          </div>
          <div className="daily-review-controls">
            <div className="daily-date-control">
              <button type="button" aria-label="Previous day" disabled={!selectedDate} onClick={() => moveDate(-1)}>‹</button>
              <input aria-label="Review date" title={formatDate(selectedDate)} type="date" value={selectedDate} onChange={(event) => { if (event.target.value) setSelectedDate(event.target.value); }} />
              <button type="button" aria-label="Next day" disabled={!selectedDate} onClick={() => moveDate(1)}>›</button>
            </div>
            <span className="daily-review-account"><span aria-hidden="true">{user?.name.charAt(0) || "G"}</span>{user?.name || "Read-only view"}</span>
          </div>
        </header>

        {error ? <p className="status error">{error}</p> : null}
        {isLoading ? <p className="status">Loading daily review...</p> : null}

        {!isLoading && !error ? (
          <>
            <section className="daily-review-source" aria-label="Daily Review data provenance">
              <div className="daily-source-summary">
                <span className={`daily-source-label ${provenance.kind === "BROKER_SNAPSHOT" ? "exact" : ""}`}>
                  <span aria-hidden="true">●</span>{provenance.kind === "BROKER_SNAPSHOT" ? "Dated broker snapshot" : provenance.label}
                </span>
                <span>As of {formatDate(selectedDate)}</span>
              </div>
              <details className="daily-source-details" key={selectedDate}>
                <summary>View source details</summary>
                <dl>
                  <div><dt>Selected session</dt><dd>{selectedDate}</dd></div>
                  <div><dt>Broker anchor</dt><dd>{provenance.anchorCoverageDate || "No dated broker anchor"}</dd></div>
                  <div><dt>Valuation</dt><dd>Exact-session historical closes</dd></div>
                  {provenance.kind === "BROKER_SNAPSHOT" && provenance.snapshot ? <div><dt>Source SHA-256</dt><dd>{provenance.snapshot.sourceHash}</dd></div> : null}
                </dl>
              </details>
            </section>

            <section className="daily-review-hero">
              <div>
                <span>Tracked daily return</span>
                <strong className={dailyReturn === null ? "" : dailyReturn >= 0 ? "daily-positive" : "daily-negative"}>
                  {isLoadingMarket ? "Loading..." : dailyReturn === null ? "—" : formatPercent(dailyReturn)}
                </strong>
              </div>
              <div>
                <span>Tracked daily P&amp;L</span>
                <strong className={dailyPnl === null ? "" : dailyPnl >= 0 ? "daily-positive" : "daily-negative"}>
                  {isLoadingMarket ? "Loading..." : dailyPnl === null ? "—" : formatCurrency(dailyPnl)}
                </strong>
              </div>
              <div>
                <span>Account equity</span>
                <strong>{selectedPortfolioEquity !== null ? formatCurrency(selectedPortfolioEquity) : "—"}</strong>
                <small>{provenance.kind === "BROKER_SNAPSHOT" ? `As of ${selectedDate}` : "Requires an exact-date broker snapshot"}</small>
              </div>
              <div>
                <span>Position changes</span>
                <strong>{activities.length}</strong>
                <small>{activitySummary || "No position changes"}</small>
              </div>
            </section>

            {!isLoadingMarket && missingMarketSymbols.length ? (
              <p className="status error">
                Exact-session historical closes are unavailable for {missingMarketSymbols.join(", ")}. These instruments are excluded from tracked daily P&amp;L and return rather than using stale prices.
              </p>
            ) : null}

            <section className="daily-activity-overview">
              <div className="daily-section-heading">
                <h2>Portfolio activity</h2>
                <span>{activities.length} changes</span>
              </div>
              <div className="table-wrap daily-feed-table">
                <table>
                  <thead><tr><th>Action</th><th>Asset</th><th>Details</th><th>Realized P&amp;L</th></tr></thead>
                  <tbody>
                    {activities.map((activity) => (
                      <tr key={activity.id}>
                        <td><span className={`daily-action ${activity.kind}`}><span aria-hidden="true">{activity.kind === "opened" || activity.kind === "added" ? "+" : "−"}</span>{activityLabel(activity.kind)}</span></td>
                        <td><strong>{activity.symbol}</strong></td>
                        <td>{formatNumber(activity.shares)} shares at {formatCurrency(activity.price)} <span className="daily-muted">· {activity.side}</span></td>
                        <td className={activity.kind === "closed" || activity.kind === "reduced" ? activity.realizedPnl >= 0 ? "daily-positive" : "daily-negative" : "daily-muted"}>
                          {activity.kind === "closed" || activity.kind === "reduced" ? formatCurrency(activity.realizedPnl) : "—"}
                        </td>
                      </tr>
                    ))}
                    {!activities.length ? <tr><td colSpan={4} className="empty-cell">No position changes for this date.</td></tr> : null}
                  </tbody>
                </table>
              </div>
            </section>

            <section className="daily-review-card">
              <div className="daily-card-heading">
                <div>
                  <h2>Position performance</h2>
                  <p className="daily-review-subtitle">Daily contribution across {performance.length} tracked assets.</p>
                </div>
                <button type="button" className="daily-sort-button" onClick={() => setContributionDescending((value) => !value)} aria-label={`Sort daily P&L ${contributionDescending ? "ascending" : "descending"}`}>Contribution {contributionDescending ? "↓" : "↑"}</button>
              </div>
              <div className="table-wrap daily-performance-table">
                <table>
                  <thead>
                    <tr>
                      <th>Asset</th>
                      <th>Shares (start → end)</th>
                      <th>Previous close</th>
                      <th>Daily close</th>
                      <th>Asset return</th>
                      <th aria-sort={contributionDescending ? "descending" : "ascending"}>Daily P&amp;L</th>
                    </tr>
                  </thead>
                  <tbody>
                    {sortedPerformance.map((row) => (
                      <tr key={row.symbol}>
                        <td><strong>{row.symbol}</strong><small className="daily-asset-side">{row.side}</small></td>
                        <td>{formatNumber(row.startShares)} <span className="daily-muted">→</span> {formatNumber(row.endShares)}</td>
                        <td>{formatCurrency(row.previousClose)}</td>
                        <td>{formatCurrency(row.close)}</td>
                        <td className={row.returnPercent >= 0 ? "daily-positive" : "daily-negative"}>
                          {formatPercent(row.returnPercent)}
                        </td>
                        <td className={row.dailyPnl >= 0 ? "daily-positive" : "daily-negative"}>
                          <span className="daily-contribution"><span className="daily-contribution-track" aria-hidden="true"><span style={{ width: `${Math.abs(row.dailyPnl) / maxContribution * 100}%` }} /></span><span>{formatCurrency(row.dailyPnl)}</span></span>
                        </td>
                      </tr>
                    ))}
                    {!isLoadingMarket && !performance.length ? (
                      <tr>
                        <td className="empty-cell" colSpan={6}>No market performance data was available for this date.</td>
                      </tr>
                    ) : null}
                  </tbody>
                </table>
              </div>
            </section>

            <p className="daily-review-disclaimer">
              {provenance.kind === "BROKER_SNAPSHOT"
                ? "Position state and equity use the dated broker snapshot. Daily performance uses exact-session historical closes and saved executions."
                : provenance.kind === "BROKER_ANCHORED_RECONSTRUCTION"
                  ? `Position activity is reconstructed from the ${provenance.anchorCoverageDate} broker anchor and saved executions. Daily performance uses exact-session historical closes; equity is not claimed for ${selectedDate}.`
                  : "Position activity is reconstructed from saved executions and exact-session historical closes. No broker-exact equity is claimed for this date."}
            </p>
          </>
        ) : null}
      </div>
  );
}
