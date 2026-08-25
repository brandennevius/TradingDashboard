"use client";

import { useEffect, useMemo, useState } from "react";
import {
  resolveDailyReviewProvenance,
  type BrokerPortfolioSnapshot
} from "@/lib/broker-portfolio-snapshot";
import type { TradeExecution, TradeLogEntry, TraderUser } from "@/lib/types";
import type { WeeklyFocus } from "@/lib/weekly-focus";

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
  const [brokerPortfolioSnapshots, setBrokerPortfolioSnapshots] = useState<BrokerPortfolioSnapshot[]>([]);
  const [activePortfolio, setActivePortfolio] = useState("");
  const [selectedDate, setSelectedDate] = useState("");
  const [marketSeries, setMarketSeries] = useState<Record<string, Candle[]>>({});
  const [isLoading, setIsLoading] = useState(true);
  const [isLoadingMarket, setIsLoadingMarket] = useState(false);
  const [error, setError] = useState("");
  const [weeklyFocus, setWeeklyFocus] = useState<WeeklyFocus | null>(null);
  const [weeklyFocusSummary, setWeeklyFocusSummary] = useState("");
  const [weeklyFocusItems, setWeeklyFocusItems] = useState("");
  const [weeklyFocusMessage, setWeeklyFocusMessage] = useState("");
  const [isSavingWeeklyFocus, setIsSavingWeeklyFocus] = useState(false);
  const [isWeeklyFocusVisible, setIsWeeklyFocusVisible] = useState(false);
  const [isWeeklyFocusExpanded, setIsWeeklyFocusExpanded] = useState(false);

  useEffect(() => {
    let cancelled = false;

    async function loadPage() {
      const [reviewResponse, focusResponse] = await Promise.all([
        fetch("/api/journal/branden/daily-review", { cache: "no-store" }),
        fetch("/api/settings/weekly-focus", { cache: "no-store" })
      ]);
      const [reviewData, focusData] = await Promise.all([
        reviewResponse.json().catch(() => ({})),
        focusResponse.json().catch(() => ({}))
      ]);

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
      setActivePortfolio(String(reviewData.defaultPortfolio || ""));
      setSelectedDate(params.get("date") || latestExecutionDate || new Date().toISOString().slice(0, 10));
      if (focusResponse.ok && focusData.focus) {
        const focus = focusData.focus as WeeklyFocus;
        setWeeklyFocus(focus);
        setWeeklyFocusSummary(focus.summary || "");
        setWeeklyFocusItems(focus.focus_items.join("\n"));
      } else if (!focusResponse.ok) {
        setWeeklyFocusMessage(focusData.error || "Could not load the saved weekly focus.");
      }
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
  const closedActivities = activities.filter((activity) => activity.kind === "closed");
  const openedActivities = activities.filter((activity) => activity.kind === "opened");
  const updatedActivities = activities.filter((activity) => activity.kind === "added" || activity.kind === "reduced");
  const selectedPortfolioEquity = provenance.accountEquity;

  async function saveWeeklyFocus(clear = false) {
    setIsSavingWeeklyFocus(true);
    setWeeklyFocusMessage("");
    try {
      const response = await fetch("/api/settings/weekly-focus", {
        method: "PUT",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          summary: clear ? "" : weeklyFocusSummary,
          focusItems: clear ? [] : weeklyFocusItems.split("\n").map((item) => item.trim()).filter(Boolean)
        })
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok || !data.focus) throw new Error(data.error || "Could not save the weekly focus.");
      const focus = data.focus as WeeklyFocus;
      setWeeklyFocus(focus);
      setWeeklyFocusSummary(focus.summary || "");
      setWeeklyFocusItems(focus.focus_items.join("\n"));
      setWeeklyFocusMessage(clear ? "Weekly focus cleared." : "Weekly focus saved.");
    } catch (saveError) {
      setWeeklyFocusMessage(saveError instanceof Error ? saveError.message : "Could not save the weekly focus.");
    } finally {
      setIsSavingWeeklyFocus(false);
    }
  }

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
            <section className="daily-review-card daily-weekly-focus">
              <div className="daily-card-heading daily-weekly-focus-heading">
                <div>
                  <p className="eyebrow">Weekend review</p>
                  <h2>Weekly Process Focus</h2>
                  {isWeeklyFocusVisible ? <p className="daily-weekly-focus-summary">
                    {weeklyFocus?.summary || "No weekly process focus is currently set."}
                  </p> : null}
                </div>
                <div className="daily-weekly-focus-actions">
                  <span>{weeklyFocus?.status || "NOT_SET"}</span>
                  <button
                    type="button"
                    className="secondary"
                    aria-expanded={isWeeklyFocusVisible}
                    onClick={() => setIsWeeklyFocusVisible((visible) => !visible)}
                  >
                    {isWeeklyFocusVisible ? "Hide" : "Show"}
                  </button>
                  <button
                    type="button"
                    className="secondary"
                    aria-expanded={isWeeklyFocusExpanded}
                    aria-controls="weekly-focus-editor"
                    onClick={() => setIsWeeklyFocusExpanded((expanded) => !expanded)}
                  >
                    {isWeeklyFocusExpanded ? "Close" : user?.readOnly ? "View" : "Edit"}
                  </button>
                </div>
              </div>
              <div className="daily-weekly-focus-meta">
                <span>{weeklyFocus?.week_start ? `Week of ${formatDate(weeklyFocus.week_start)}` : "No active week"}</span>
                <span>Used unchanged by Daily Snapshot and Market Review</span>
              </div>
              {isWeeklyFocusExpanded ? <div id="weekly-focus-editor" className="daily-weekly-focus-editor">
                <p>This is copied exactly into each daily snapshot and market review until you replace or clear it.</p>
                <label>
                  Summary
                  <textarea
                    rows={2}
                    value={weeklyFocusSummary}
                    onChange={(event) => setWeeklyFocusSummary(event.target.value)}
                    disabled={Boolean(user?.readOnly)}
                    placeholder="Enter the weekly process focus in your own words."
                  />
                </label>
                <label>
                  Ordered focus items (one per line)
                  <textarea
                    rows={5}
                    value={weeklyFocusItems}
                    onChange={(event) => setWeeklyFocusItems(event.target.value)}
                    disabled={Boolean(user?.readOnly)}
                    placeholder="Structure first and size second"
                  />
                </label>
                {!user?.readOnly ? (
                  <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>
                    <button type="button" onClick={() => saveWeeklyFocus(false)} disabled={isSavingWeeklyFocus}>
                      {isSavingWeeklyFocus ? "Saving..." : "Save weekly focus"}
                    </button>
                    <button type="button" className="secondary" onClick={() => saveWeeklyFocus(true)} disabled={isSavingWeeklyFocus}>
                      Clear focus
                    </button>
                  </div>
                ) : null}
                {weeklyFocusMessage ? <p className="status">{weeklyFocusMessage}</p> : null}
              </div> : null}
            </section>

            <section className="daily-review-provenance" aria-label="Daily Review data provenance">
              <strong>{provenance.label}</strong>
              <span>Selected session: {selectedDate}</span>
              {provenance.anchorCoverageDate ? <span>Broker anchor: {provenance.anchorCoverageDate}</span> : <span>No dated broker anchor</span>}
              <span>Valuation: exact-session historical closes</span>
              {provenance.kind === "BROKER_SNAPSHOT" && provenance.snapshot ? (
                <span title={provenance.snapshot.sourceHash}>Source SHA-256: {provenance.snapshot.sourceHash.slice(0, 12)}…</span>
              ) : null}
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
                <span>Broker-exact equity</span>
                <strong>{selectedPortfolioEquity !== null ? formatCurrency(selectedPortfolioEquity) : "—"}</strong>
                <small>{provenance.kind === "BROKER_SNAPSHOT" ? `As of ${selectedDate}` : "Requires an exact-date broker snapshot"}</small>
              </div>
              <div>
                <span>Position changes</span>
                <strong>{activities.length}</strong>
              </div>
            </section>

            {!isLoadingMarket && missingMarketSymbols.length ? (
              <p className="status error">
                Exact-session historical closes are unavailable for {missingMarketSymbols.join(", ")}. These instruments are excluded from tracked daily P&amp;L and return rather than using stale prices.
              </p>
            ) : null}

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
