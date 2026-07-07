"use client";

import { useEffect, useMemo, useState } from "react";
import type { TradeExecution, TradeLogEntry, TraderUser } from "@/lib/types";

type PortfolioMeta = {
  currentEquity?: number;
};

type PortfolioSettingsResponse = {
  portfolios?: string[];
  defaultPortfolio?: string;
  portfolioMeta?: Record<string, PortfolioMeta>;
};

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

function ActivitySection({
  title,
  eyebrow,
  activities,
  emptyMessage
}: {
  title: string;
  eyebrow: string;
  activities: DailyActivity[];
  emptyMessage: string;
}) {
  return (
    <article className="daily-review-card daily-activity-section">
      <div className="daily-card-heading">
        <div>
          <p className="eyebrow">{eyebrow}</p>
          <h2>{title}</h2>
        </div>
        <span>{activities.length}</span>
      </div>
      {activities.length ? (
        <div className="daily-activity-list">
          {activities.map((activity) => (
            <div className="daily-activity-row" key={activity.id}>
              <span className={`daily-activity-icon ${activity.kind}`}>
                {activity.kind === "opened" || activity.kind === "added" ? "+" : "−"}
              </span>
              <div>
                <strong>{activityLabel(activity.kind)} {activity.symbol}</strong>
                <span>{formatNumber(activity.shares)} shares at {formatCurrency(activity.price)} · {activity.side}</span>
              </div>
              {activity.kind === "closed" || activity.kind === "reduced" ? (
                <strong className={activity.realizedPnl >= 0 ? "daily-positive" : "daily-negative"}>
                  {formatCurrency(activity.realizedPnl)}
                </strong>
              ) : null}
            </div>
          ))}
        </div>
      ) : (
        <div className="daily-empty-state daily-empty-state-compact">
          <span>{emptyMessage}</span>
        </div>
      )}
    </article>
  );
}

export default function DailyReviewPage() {
  const [user, setUser] = useState<TraderUser | null>(null);
  const [trades, setTrades] = useState<TradeLogEntry[]>([]);
  const [portfolioMeta, setPortfolioMeta] = useState<Record<string, PortfolioMeta>>({});
  const [activePortfolio, setActivePortfolio] = useState("");
  const [selectedDate, setSelectedDate] = useState("");
  const [marketSeries, setMarketSeries] = useState<Record<string, Candle[]>>({});
  const [isLoading, setIsLoading] = useState(true);
  const [isLoadingMarket, setIsLoadingMarket] = useState(false);
  const [error, setError] = useState("");

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
      setPortfolioMeta(reviewData.portfolioMeta || {});
      setActivePortfolio(String(reviewData.defaultPortfolio || ""));
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

  const relevantSymbols = useMemo(
    () =>
      sortedUnique(
        brandenTrades
          .filter((trade) => {
            const executions = normalizedExecutions(trade);
            return sharesBefore(executions, selectedDate) > 0 || executions.some((execution) => execution.date === selectedDate);
          })
          .map((trade) => trade.symbol)
      ),
    [brandenTrades, selectedDate]
  );

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
        const side = symbolTrades[0]?.side || "LONG";
        const allExecutions = symbolTrades.flatMap(normalizedExecutions);
        const startingShares = sharesBefore(allExecutions, selectedDate);
        const entries = allExecutions.filter((execution) => execution.date === selectedDate && execution.type === "ENTRY");
        const exits = allExecutions.filter((execution) => execution.date === selectedDate && execution.type === "EXIT");
        const entryShares = entries.reduce((sum, execution) => sum + execution.shares, 0);
        const exitShares = exits.reduce((sum, execution) => sum + execution.shares, 0);
        const endingShares = Math.max(0, startingShares + entryShares - exitShares);
        const candles = (marketSeries[symbol] || []).filter((candle) => candle.time <= selectedDate);
        const selectedIndex = candles.findIndex((candle) => candle.time === selectedDate);
        const closeIndex = selectedIndex >= 0 ? selectedIndex : candles.length - 1;
        const close = candles[closeIndex]?.close || 0;
        const previousClose = candles[closeIndex - 1]?.close || 0;

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
  }, [brandenTrades, marketSeries, relevantSymbols, selectedDate]);

  const dailyPnl = performance.reduce((sum, row) => sum + row.dailyPnl, 0);
  const startingMarketValue = performance.reduce((sum, row) => sum + Math.abs(row.startShares * row.previousClose), 0);
  const dailyReturn = startingMarketValue ? (dailyPnl / startingMarketValue) * 100 : 0;
  const closedActivities = activities.filter((activity) => activity.kind === "closed");
  const openedActivities = activities.filter((activity) => activity.kind === "opened");
  const updatedActivities = activities.filter((activity) => activity.kind === "added" || activity.kind === "reduced");
  const selectedPortfolioEquity = activePortfolio ? portfolioMeta[activePortfolio]?.currentEquity : undefined;

  return (
    <div className="branden-journal-content daily-review-page">
        <header className="branden-route-header">
          <div>
            <p className="eyebrow">Portfolio change report</p>
            <h1>Daily Review</h1>
            <span>{formatDate(selectedDate)}</span>
          </div>
        </header>

        <section className="branden-route-toolbar">
          <label>
            Review date
            <input type="date" value={selectedDate} onChange={(event) => setSelectedDate(event.target.value)} />
          </label>
          {user ? <span>Signed in as {user.name}</span> : <span>Public read-only view</span>}
        </section>

        {error ? <p className="status error">{error}</p> : null}
        {isLoading ? <p className="status">Loading daily review...</p> : null}

        {!isLoading && !error ? (
          <>
            <section className="daily-review-hero">
              <div>
                <span>Tracked daily return</span>
                <strong className={dailyReturn >= 0 ? "daily-positive" : "daily-negative"}>
                  {isLoadingMarket ? "Loading..." : formatPercent(dailyReturn)}
                </strong>
              </div>
              <div>
                <span>Tracked daily P&amp;L</span>
                <strong className={dailyPnl >= 0 ? "daily-positive" : "daily-negative"}>
                  {isLoadingMarket ? "Loading..." : formatCurrency(dailyPnl)}
                </strong>
              </div>
              <div>
                <span>Portfolio equity</span>
                <strong>{selectedPortfolioEquity ? formatCurrency(selectedPortfolioEquity) : "—"}</strong>
              </div>
              <div>
                <span>Position changes</span>
                <strong>{activities.length}</strong>
              </div>
            </section>

            <section className="daily-activity-overview">
              <div className="daily-section-heading">
                <div>
                  <p className="eyebrow">What changed</p>
                  <h2>Portfolio activity</h2>
                </div>
                <span>{activities.length} total changes</span>
              </div>
              <div className="daily-activity-grid">
                <ActivitySection
                  eyebrow="Exits"
                  title="Closed positions"
                  activities={closedActivities}
                  emptyMessage="No positions were closed."
                />
                <ActivitySection
                  eyebrow="New exposure"
                  title="Newly opened positions"
                  activities={openedActivities}
                  emptyMessage="No new positions were opened."
                />
                <ActivitySection
                  eyebrow="Adds and trims"
                  title="Updated positions"
                  activities={updatedActivities}
                  emptyMessage="No existing positions were added to or trimmed."
                />
              </div>
            </section>

            <section className="daily-review-card">
              <div className="daily-card-heading">
                <div>
                  <p className="eyebrow">Position performance</p>
                  <h2>How each asset performed</h2>
                </div>
                <span>{performance.length} tracked assets</span>
              </div>
              <div className="table-wrap daily-performance-table">
                <table>
                  <thead>
                    <tr>
                      <th>Asset</th>
                      <th>Side</th>
                      <th>Start shares</th>
                      <th>End shares</th>
                      <th>Previous close</th>
                      <th>Daily close</th>
                      <th>Asset return</th>
                      <th>P&amp;L contribution</th>
                    </tr>
                  </thead>
                  <tbody>
                    {performance.map((row) => (
                      <tr key={row.symbol}>
                        <td><strong>{row.symbol}</strong></td>
                        <td>{row.side}</td>
                        <td>{formatNumber(row.startShares)}</td>
                        <td>{formatNumber(row.endShares)}</td>
                        <td>{formatCurrency(row.previousClose)}</td>
                        <td>{formatCurrency(row.close)}</td>
                        <td className={row.returnPercent >= 0 ? "daily-positive" : "daily-negative"}>
                          {formatPercent(row.returnPercent)}
                        </td>
                        <td className={row.dailyPnl >= 0 ? "daily-positive" : "daily-negative"}>
                          {formatCurrency(row.dailyPnl)}
                        </td>
                      </tr>
                    ))}
                    {!isLoadingMarket && !performance.length ? (
                      <tr>
                        <td className="empty-cell" colSpan={8}>No market performance data was available for this date.</td>
                      </tr>
                    ) : null}
                  </tbody>
                </table>
              </div>
            </section>

            <p className="daily-review-disclaimer">
              This version reconstructs activity from saved executions and estimates daily performance from market closes.
              Broker-exact total return will require storing a portfolio equity snapshot at each market close.
            </p>
          </>
        ) : null}
      </div>
  );
}
