"use client";

import { useEffect, useMemo, useState } from "react";
import type { TradeLogEntry, TraderUser } from "@/lib/types";

const BREAKEVEN_R_THRESHOLD = 0.1;

type CalendarDayTradeRow = {
  trade: TradeLogEntry;
  date: string;
  pnl: number;
  rMultiple: number;
  activityKind: "closed" | "partial_exit";
};

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

function pluralize(count: number, singular: string) {
  return `${count} ${singular}${count === 1 ? "" : "s"}`;
}

function formatCalendarMonth(value: string) {
  const [year, month] = value.split("-").map(Number);
  return new Date(year, month - 1, 1).toLocaleDateString("en-US", {
    month: "long",
    year: "numeric"
  });
}

function shiftMonth(value: string, amount: number) {
  const [year, month] = value.split("-").map(Number);
  const next = new Date(year, month - 1 + amount, 1);
  return `${next.getFullYear()}-${String(next.getMonth() + 1).padStart(2, "0")}`;
}

function formatWeekRange(start: string, end: string) {
  const startDate = new Date(`${start}T12:00:00Z`);
  const endDate = new Date(`${end}T12:00:00Z`);
  const sameMonth = startDate.getUTCMonth() === endDate.getUTCMonth();
  const startLabel = startDate.toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    timeZone: "UTC"
  });
  const endLabel = endDate.toLocaleDateString("en-US", {
    month: sameMonth ? undefined : "short",
    day: "numeric",
    timeZone: "UTC"
  });
  return `${startLabel}–${endLabel}`;
}

function hasPartialExitTag(tags: string[]) {
  return tags.some((tag) => tag.trim().toLowerCase() === "partial exits");
}

function tradeStatusForPnl(pnl: number, closed: boolean, rMultiple = 0): TradeLogEntry["status"] {
  if (!closed) return "OPEN";
  if (Math.abs(rMultiple) < BREAKEVEN_R_THRESHOLD) return "BREAKEVEN";
  if (pnl > 0) return "WIN";
  if (pnl < 0) return "LOSS";
  return "BREAKEVEN";
}

function tradePnlDate(trade: TradeLogEntry) {
  if (trade.status !== "OPEN" && trade.exitDate) {
    return trade.exitDate;
  }

  if (hasPartialExitTag(trade.customTags) && numberValue(trade.pnl) !== 0) {
    const latestExecutionExitDate =
      trade.executions
        ?.filter((execution) => execution.type === "EXIT" && execution.date)
        .map((execution) => execution.date)
        .sort()
        .at(-1) || "";

    if (latestExecutionExitDate) {
      return latestExecutionExitDate;
    }
  }

  return trade.entryDate;
}

function calendarRowsForDate(trades: TradeLogEntry[], date: string): CalendarDayTradeRow[] {
  return trades
    .map((trade) => {
      const executionsForDate = (trade.executions || []).filter(
        (execution) => execution.type === "EXIT" && execution.date === date
      );

      if (executionsForDate.length) {
        const pnl = executionsForDate.reduce((total, execution) => total + numberValue(execution.pnl), 0);
        return {
          trade,
          date,
          pnl,
          rMultiple: numberValue(trade.risk) ? pnl / numberValue(trade.risk) : 0,
          activityKind: trade.status === "OPEN" ? "partial_exit" : "closed"
        };
      }

      if ((trade.executions || []).length || trade.status === "OPEN" || tradePnlDate(trade) !== date) {
        return null;
      }

      const pnl = numberValue(trade.pnl);
      return {
        trade,
        date,
        pnl,
        rMultiple: numberValue(trade.rMultiple),
        activityKind: "closed"
      };
    })
    .filter((row): row is CalendarDayTradeRow => Boolean(row));
}

function calendarRowStatus(row: CalendarDayTradeRow) {
  return tradeStatusForPnl(row.pnl, row.trade.status !== "OPEN", row.rMultiple);
}

function summarizeCalendarRows(rows: CalendarDayTradeRow[]) {
  const closedTradeIds = new Set(rows.filter((row) => row.activityKind === "closed").map((row) => row.trade.id));
  const partialExitIds = new Set(rows.filter((row) => row.activityKind === "partial_exit").map((row) => row.trade.id));

  return {
    closedTradeCount: closedTradeIds.size,
    partialExitCount: partialExitIds.size,
    activityCount: rows.length
  };
}

function calendarActivityLabel(summary: ReturnType<typeof summarizeCalendarRows>) {
  const parts: string[] = [];
  if (summary.closedTradeCount) parts.push(pluralize(summary.closedTradeCount, "closed trade"));
  if (summary.partialExitCount) parts.push(pluralize(summary.partialExitCount, "trim"));
  return parts.join(" / ");
}

export default function BrandenCalendarPage() {
  const [user, setUser] = useState<TraderUser | null>(null);
  const [trades, setTrades] = useState<TradeLogEntry[]>([]);
  const [activePortfolio, setActivePortfolio] = useState("");
  const [calendarMonth, setCalendarMonth] = useState(new Date().toISOString().slice(0, 7));
  const [selectedCalendarDate, setSelectedCalendarDate] = useState("");
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;

    async function loadPageData() {
      setIsLoading(true);
      setError("");

      const calendarResponse = await fetch("/api/journal/branden/calendar", { cache: "no-store" });
      const calendarData = await calendarResponse.json().catch(() => ({}));

      if (!calendarResponse.ok || !calendarData.user) {
        setError("Sign in to view Calendar.");
        setIsLoading(false);
        return;
      }

      if (cancelled) return;

      setUser(calendarData.user || null);
      setTrades(Array.isArray(calendarData.trades) ? calendarData.trades : []);
      setActivePortfolio(String(calendarData.defaultPortfolio || ""));
      setIsLoading(false);
    }

    loadPageData().catch((loadError) => {
      if (!cancelled) {
        setError(loadError instanceof Error ? loadError.message : "Could not load calendar.");
        setIsLoading(false);
      }
    });

    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!selectedCalendarDate) return;

    function closeOnEscape(event: KeyboardEvent) {
      if (event.key === "Escape") {
        setSelectedCalendarDate("");
      }
    }

    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, [selectedCalendarDate]);

  const brandenTrades = useMemo(() => trades.filter((trade) => trade.userId === "branden" && !trade.hidden), [trades]);
  const filteredTrades = useMemo(
    () => brandenTrades.filter((trade) => !activePortfolio || trade.portfolioTag === activePortfolio),
    [activePortfolio, brandenTrades]
  );
  const calendarSummary = useMemo(() => {
    const [year, month] = calendarMonth.split("-").map(Number);
    const firstDay = new Date(year, month - 1, 1);
    const daysInMonth = new Date(year, month, 0).getDate();
    const startOffset = firstDay.getDay();
    const rowsByDate = Array.from({ length: daysInMonth }, (_, index) => {
      const date = `${calendarMonth}-${String(index + 1).padStart(2, "0")}`;
      return [date, calendarRowsForDate(filteredTrades, date)] as const;
    }).reduce<Record<string, CalendarDayTradeRow[]>>((groups, [date, rows]) => {
      if (rows.length) groups[date] = rows;
      return groups;
    }, {});
    const cells = Array.from({ length: 42 }, (_, index) => {
      const day = index - startOffset + 1;
      const inMonth = day >= 1 && day <= daysInMonth;
      const date = inMonth ? `${calendarMonth}-${String(day).padStart(2, "0")}` : "";
      const dayRows = date ? rowsByDate[date] || [] : [];
      const pnl = dayRows.reduce((total, row) => total + row.pnl, 0);
      const daySummary = summarizeCalendarRows(dayRows);

      return {
        date,
        day,
        inMonth,
        pnl,
        trades: dayRows.map((row) => row.trade),
        rows: dayRows,
        closedTradeCount: daySummary.closedTradeCount,
        partialExitCount: daySummary.partialExitCount,
        activityLabel: calendarActivityLabel(daySummary)
      };
    });
    const weeks = Array.from({ length: 6 }, (_, weekIndex) => {
      const weekCells = cells.slice(weekIndex * 7, weekIndex * 7 + 7);
      const inMonthCells = weekCells.filter((cell) => cell.inMonth && cell.date);
      const pnl = weekCells.reduce((total, cell) => total + cell.pnl, 0);
      const activeDays = weekCells.filter((cell) => cell.trades.length > 0).length;

      return {
        label: inMonthCells.length
          ? formatWeekRange(inMonthCells[0]!.date, inMonthCells[inMonthCells.length - 1]!.date)
          : "",
        pnl,
        activeDays,
        inMonth: inMonthCells.length > 0
      };
    }).filter((week) => week.inMonth);
    const monthRows = Object.values(rowsByDate).flat();
    const monthPnl = monthRows.reduce((total, row) => total + row.pnl, 0);
    const activeDays = Object.values(rowsByDate).filter((dayRows) => dayRows.length > 0).length;
    const closedTradeCount = summarizeCalendarRows(monthRows).closedTradeCount;

    return {
      cells,
      weeks,
      monthPnl,
      activeDays,
      tradeCount: closedTradeCount
    };
  }, [calendarMonth, filteredTrades]);
  const selectedCalendarDay = useMemo(
    () => calendarSummary.cells.find((cell) => cell.date === selectedCalendarDate),
    [calendarSummary.cells, selectedCalendarDate]
  );

  return (
    <>
      <section className="branden-journal-main">
        <div className="section-heading">
          <p className="eyebrow">Branden journal</p>
          <h1>Calendar</h1>
          <p>Daily and weekly realized P&amp;L grouped by exit date.</p>
        </div>

        {isLoading ? <p className="status">Loading calendar...</p> : null}
        {error ? <p className="status error">{error}</p> : null}
        {user ? (
          <section className="trade-calendar-panel">
            <div className="calendar-toolbar">
              <div className="calendar-nav">
                <button className="trade-muted-button" type="button" onClick={() => setCalendarMonth(shiftMonth(calendarMonth, -1))}>
                  ← Prev
                </button>
                <h3>{formatCalendarMonth(calendarMonth)}</h3>
                <button className="trade-muted-button" type="button" onClick={() => setCalendarMonth(shiftMonth(calendarMonth, 1))}>
                  Next →
                </button>
                <button className="trade-muted-button" type="button" onClick={() => setCalendarMonth(new Date().toISOString().slice(0, 7))}>
                  This month
                </button>
              </div>
              <div className="calendar-month-stats">
                <strong className={calendarSummary.monthPnl >= 0 ? "trade-positive" : "trade-negative"}>{money(calendarSummary.monthPnl)}</strong>
                <em>{pluralize(calendarSummary.activeDays, "day")}</em>
                <em>{pluralize(calendarSummary.tradeCount, "closed trade")}</em>
              </div>
            </div>

            <div className="calendar-layout">
              <div className="calendar-scroll">
                <div className="calendar-grid">
                  {["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"].map((day) => (
                    <div className="calendar-weekday" key={day}>{day}</div>
                  ))}
                  {calendarSummary.cells.map((cell, index) => (
                    <button
                      className={[
                        "calendar-day",
                        !cell.inMonth ? "outside" : "",
                        cell.pnl > 0 ? "win" : cell.pnl < 0 ? "loss" : ""
                      ].filter(Boolean).join(" ")}
                      key={`${cell.date || "blank"}-${index}`}
                      type="button"
                      disabled={!cell.inMonth || !cell.trades.length}
                      onClick={() => {
                        if (cell.date && cell.trades.length) setSelectedCalendarDate(cell.date);
                      }}
                    >
                      {cell.inMonth ? <span className="calendar-day-number">{cell.day}</span> : null}
                      {cell.trades.length ? (
                        <span className="calendar-day-content">
                          <strong>{money(cell.pnl)}</strong>
                          <small>{cell.activityLabel || "No closed trades"}</small>
                        </span>
                      ) : null}
                    </button>
                  ))}
                </div>
              </div>
              <aside className="calendar-week-panel">
                {calendarSummary.weeks.map((week) => (
                  <article className="calendar-week-card" key={week.label}>
                    <span>{week.label}</span>
                    <strong className={week.pnl >= 0 ? "trade-positive" : "trade-negative"}>{money(week.pnl)}</strong>
                    <em>{pluralize(week.activeDays, "active day")}</em>
                  </article>
                ))}
              </aside>
            </div>
          </section>
        ) : null}
      </section>

      {selectedCalendarDay?.rows.length ? (
        <div
          className="trade-modal-backdrop"
          role="dialog"
          aria-modal="true"
          aria-label="Calendar day trades"
          onMouseDown={(event) => {
            if (event.target === event.currentTarget) {
              setSelectedCalendarDate("");
            }
          }}
        >
          <section className="trade-entry-form trade-modal calendar-day-modal">
            <div className="trade-detail-actions">
              <div>
                <p className="eyebrow">Calendar day</p>
                <h3>{new Date(`${selectedCalendarDay.date}T00:00:00`).toLocaleDateString("en-US", {
                  month: "long",
                  day: "numeric",
                  year: "numeric"
                })}</h3>
                <p>
                  {money(selectedCalendarDay.pnl)} / {selectedCalendarDay.activityLabel || "No closed trades"}
                </p>
              </div>
              <button className="trade-muted-button" type="button" onClick={() => setSelectedCalendarDate("")}>Close</button>
            </div>
            <div className="calendar-day-modal-summary">
              <article><span>P&L</span><strong className={selectedCalendarDay.pnl >= 0 ? "trade-positive" : "trade-negative"}>{money(selectedCalendarDay.pnl)}</strong></article>
              <article><span>Closed trades</span><strong>{selectedCalendarDay.closedTradeCount}</strong></article>
              <article><span>Trims</span><strong>{selectedCalendarDay.partialExitCount}</strong></article>
              <article><span>Total R</span><strong>{selectedCalendarDay.rows.reduce((total, row) => total + row.rMultiple, 0).toFixed(2)}R</strong></article>
            </div>
            <div className="trade-table-wrap calendar-day-table-wrap">
              <table className="trade-table calendar-day-table">
                <thead>
                  <tr>
                    <th>Symbol</th>
                    <th>Status</th>
                    <th>Side</th>
                    <th>P&L</th>
                    <th>R</th>
                    <th>Notes</th>
                  </tr>
                </thead>
                <tbody>
                  {selectedCalendarDay.rows.map((row) => {
                    const trade = row.trade;
                    const status = calendarRowStatus(row);
                    return (
                      <tr
                        key={`${trade.id}-${row.date}`}
                        onClick={() => {
                          window.location.href = `/journal/branden/dashboard?tradeId=${encodeURIComponent(trade.id)}`;
                        }}
                      >
                        <td><span className="trade-symbol calendar-day-symbol">{trade.symbol}</span></td>
                        <td><span className={`trade-badge ${status.toLowerCase()}`}>{status}</span></td>
                        <td>{trade.side}</td>
                        <td className={row.pnl >= 0 ? "trade-positive" : "trade-negative"}>{money(row.pnl)}</td>
                        <td>{row.rMultiple.toFixed(2)}R</td>
                        <td className="calendar-day-note-cell">{trade.notes || "-"}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </section>
        </div>
      ) : null}
    </>
  );
}
