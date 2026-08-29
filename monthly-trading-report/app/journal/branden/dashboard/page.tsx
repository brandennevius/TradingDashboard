"use client";

import dynamic from "next/dynamic";
import { useEffect, useMemo, useState } from "react";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from "recharts";
import type { SetupChecklistTemplate, TradeLogEntry, TraderUser } from "@/lib/types";
import { hasCompletedTradeReview } from "@/lib/trade-review";

const LegacyTradeDetailHost = dynamic(() => import("@/app/page"), {
  ssr: false
});

type DashboardResponse = {
  user?: TraderUser | null;
  trades?: TradeLogEntry[];
  portfolios?: string[];
  defaultPortfolio?: string;
  setupChecklists?: SetupChecklistTemplate[];
  error?: string;
};

const chartTooltipStyle = {
  backgroundColor: "#fffaf0",
  border: "1px solid rgba(185, 214, 168, 0.95)",
  borderRadius: "10px",
  color: "#2f352d",
  fontWeight: 700
};

function currentDate() {
  const now = new Date();
  return `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, "0")}-${String(now.getDate()).padStart(2, "0")}`;
}

function currentMonthStartDate() {
  const now = new Date();
  return `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, "0")}-01`;
}

function currentYearStartDate() {
  return `${new Date().getFullYear()}-01-01`;
}

function money(value: number) {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0
  }).format(Number.isFinite(value) ? value : 0);
}

function percent(value: number) {
  return `${value >= 0 ? "+" : ""}${value.toFixed(2)}%`;
}

function signedMoney(value: number) {
  return `${value >= 0 ? "+" : ""}${money(value)}`;
}

function average(values: number[]) {
  return values.length ? values.reduce((total, value) => total + value, 0) / values.length : 0;
}

function sum(values: number[]) {
  return values.reduce((total, value) => total + value, 0);
}

function tradePnlDate(trade: TradeLogEntry) {
  return trade.exitDate || trade.entryDate;
}

function inDateRange(value: string, startDate: string, endDate: string) {
  if (!value) return false;
  if (startDate && value < startDate) return false;
  if (endDate && value > endDate) return false;
  return true;
}

function dateFromYmd(value: string) {
  const [year, month, day] = value.split("-").map(Number);
  return new Date(year, (month || 1) - 1, day || 1);
}

function ymdFromDate(value: Date) {
  return `${value.getFullYear()}-${String(value.getMonth() + 1).padStart(2, "0")}-${String(value.getDate()).padStart(2, "0")}`;
}

function addDays(value: Date, days: number) {
  const next = new Date(value);
  next.setDate(next.getDate() + days);
  return next;
}

function weekStartForDate(value: string) {
  const date = dateFromYmd(value);
  const day = date.getDay();
  const mondayOffset = day === 0 ? -6 : 1 - day;
  return ymdFromDate(addDays(date, mondayOffset));
}

function weekEndForStart(weekStart: string) {
  return ymdFromDate(addDays(dateFromYmd(weekStart), 4));
}

function compactDateLabel(value: string) {
  const date = dateFromYmd(value);
  return date.toLocaleDateString("en-US", { month: "short", day: "numeric" });
}

function weeklyLabel(weekStart: string) {
  return `${compactDateLabel(weekStart)}-${compactDateLabel(weekEndForStart(weekStart)).replace(/^[A-Za-z]+ /, "")}`;
}

function rangedExitExecutions(trade: TradeLogEntry, startDate: string, endDate: string) {
  return (trade.executions || []).filter(
    (execution) => execution.type === "EXIT" && inDateRange(execution.date, startDate, endDate)
  );
}

function tradeHasRangeActivity(trade: TradeLogEntry, startDate: string, endDate: string) {
  if (rangedExitExecutions(trade, startDate, endDate).length) return true;
  const activityDate = trade.exitDate || trade.entryDate;
  return inDateRange(activityDate, startDate, endDate);
}

function settledCostBasisForRange(trade: TradeLogEntry, startDate: string, endDate: string) {
  const exits = rangedExitExecutions(trade, startDate, endDate);
  const settledShares = exits.length
    ? exits.reduce((total, execution) => total + Number(execution.shares || 0), 0)
    : trade.status !== "OPEN"
      ? Number(trade.shares || 0)
      : 0;
  return Math.abs(Number(trade.avgEntry || 0) * settledShares);
}

const BREAKEVEN_R_THRESHOLD = 0.1;

function tradeStatusForPnl(pnl: number, closed: boolean, rMultiple = 0): TradeLogEntry["status"] {
  if (!closed) return "OPEN";
  if (Math.abs(rMultiple) < BREAKEVEN_R_THRESHOLD) return "BREAKEVEN";
  if (pnl > 0) return "WIN";
  if (pnl < 0) return "LOSS";
  return "BREAKEVEN";
}

function normalizedTradeStatus(trade: TradeLogEntry): TradeLogEntry["status"] {
  return tradeStatusForPnl(Number(trade.pnl || 0), trade.status !== "OPEN", Number(trade.rMultiple || 0));
}

function countsAsSettledTrade(trade: TradeLogEntry) {
  const hasPartialExits = trade.customTags.some((tag) => tag.trim().toLowerCase() === "partial exits");
  return trade.status !== "OPEN" || (hasPartialExits && Number(trade.pnl || 0) !== 0);
}

function countsAsClosedLifecycleTrade(trade: TradeLogEntry) {
  return trade.status !== "OPEN" && Boolean(trade.exitDate);
}

function tradeForRange(trade: TradeLogEntry, startDate: string, endDate: string) {
  const exits = rangedExitExecutions(trade, startDate, endDate);
  if (!exits.length) return trade;
  const pnl = exits.reduce((total, execution) => total + Number(execution.pnl || 0), 0);
  const rMultiple = trade.risk ? pnl / trade.risk : 0;
  return {
    ...trade,
    pnl,
    rMultiple,
    status: tradeStatusForPnl(pnl, trade.status !== "OPEN", rMultiple),
    exitDate: exits[exits.length - 1]?.date || trade.exitDate
  };
}

function primarySetupName(trade: TradeLogEntry) {
  return trade.setupTags?.[0] || "Unassigned";
}

function effectiveGrade(trade: TradeLogEntry, setupTemplates: SetupChecklistTemplate[]) {
  if (trade.manualGrade) return trade.manualGrade;
  const template = setupTemplates.find((item) => item.setupName === primarySetupName(trade));
  if (!template) return "C";
  const score = (trade.checklistItems || []).reduce((total, item) => total + (item.met ? item.points : 0), 0);
  const band = (template.gradeBands || []).find((gradeBand) => score >= gradeBand.minScore && (gradeBand.maxScore === null || score <= gradeBand.maxScore));
  return band?.label || "C";
}

function tradeNeedsReview(trade: TradeLogEntry, setupTemplates: SetupChecklistTemplate[]) {
  const template = setupTemplates.find((item) => item.setupName === primarySetupName(trade));
  const totalChecklistPoints = template?.groups?.length
    ? template.groups.flatMap((group) => group.criteria || []).reduce((total, item) => total + (Number.isFinite(Number(item.points)) ? Number(item.points) : 0), 0)
    : (trade.checklistItems || []).reduce((total, item) => total + (Number.isFinite(Number(item.points)) ? Number(item.points) : 0), 0);

  return !trade.risk || !hasCompletedTradeReview(trade.reviewSections, trade.notes) || (!trade.screenshots.length && !(trade.chartLinks || []).length) || !totalChecklistPoints;
}

function longestTradeStreak(trades: TradeLogEntry[], status: "WIN" | "LOSS") {
  let longest = 0;
  let current = 0;
  trades.forEach((trade) => {
    if (normalizedTradeStatus(trade) === status) {
      current += 1;
      longest = Math.max(longest, current);
    } else if (normalizedTradeStatus(trade) !== "BREAKEVEN") {
      current = 0;
    }
  });
  return longest;
}

export default function BrandenDashboardPage() {
  const [tradeId, setTradeId] = useState("");
  const [user, setUser] = useState<TraderUser | null>(null);
  const [trades, setTrades] = useState<TradeLogEntry[]>([]);
  const [setupTemplates, setSetupTemplates] = useState<SetupChecklistTemplate[]>([]);
  const [activePortfolio, setActivePortfolio] = useState("");
  const [startDate, setStartDate] = useState(currentMonthStartDate());
  const [endDate, setEndDate] = useState(currentDate());
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    setTradeId(new URLSearchParams(window.location.search).get("tradeId") || "");
  }, []);

  useEffect(() => {
    if (tradeId) return;
    let cancelled = false;

    async function loadDashboard() {
      setIsLoading(true);
      setError("");
      const response = await fetch("/api/journal/branden/trade-log", { cache: "no-store" });
      const data = (await response.json().catch(() => ({}))) as DashboardResponse;

      if (cancelled) return;

      if (!response.ok) {
        setError(data.error || "Could not load dashboard.");
        setIsLoading(false);
        return;
      }

      const nextDefaultPortfolio = String(data.defaultPortfolio || "");
      setUser(data.user || null);
      setTrades(Array.isArray(data.trades) ? data.trades : []);
      setSetupTemplates(Array.isArray(data.setupChecklists) ? data.setupChecklists : []);
      setActivePortfolio(nextDefaultPortfolio);
      setIsLoading(false);
    }

    loadDashboard().catch((loadError) => {
      if (!cancelled) {
        setError(loadError instanceof Error ? loadError.message : "Could not load dashboard.");
        setIsLoading(false);
      }
    });

    return () => {
      cancelled = true;
    };
  }, [tradeId]);

  const visibleTrades = useMemo(() => trades.filter((trade) => trade.userId === "branden" && !trade.hidden), [trades]);
  const filteredTrades = useMemo(
    () =>
      visibleTrades
        .filter((trade) => !activePortfolio || trade.portfolioTag === activePortfolio)
        .filter((trade) => tradeHasRangeActivity(trade, startDate, endDate))
        .map((trade) => tradeForRange(trade, startDate, endDate)),
    [activePortfolio, endDate, startDate, visibleTrades]
  );
  const closedTrades = useMemo(() => filteredTrades.filter(countsAsSettledTrade), [filteredTrades]);
  const openedTrades = useMemo(
    () =>
      visibleTrades
        .filter((trade) => !activePortfolio || trade.portfolioTag === activePortfolio)
        .filter((trade) => inDateRange(trade.entryDate, startDate, endDate)),
    [activePortfolio, endDate, startDate, visibleTrades]
  );
  const closedLifecycleTrades = useMemo(
    () =>
      visibleTrades
        .filter((trade) => !activePortfolio || trade.portfolioTag === activePortfolio)
        .filter((trade) => countsAsClosedLifecycleTrade(trade) && inDateRange(trade.exitDate, startDate, endDate)),
    [activePortfolio, endDate, startDate, visibleTrades]
  );

  const summary = useMemo(() => {
    const wins = closedTrades.filter((trade) => normalizedTradeStatus(trade) === "WIN");
    const losses = closedTrades.filter((trade) => normalizedTradeStatus(trade) === "LOSS");
    const grossWin = wins.reduce((total, trade) => total + trade.pnl, 0);
    const grossLoss = Math.abs(losses.reduce((total, trade) => total + trade.pnl, 0));
    const orderedClosedTrades = [...closedTrades].sort((a, b) => tradePnlDate(a).localeCompare(tradePnlDate(b)));
    const settledCostBasis = closedTrades.reduce((total, trade) => total + settledCostBasisForRange(trade, startDate, endDate), 0);
    const netPnl = closedTrades.reduce((total, trade) => total + trade.pnl, 0);

    return {
      netPnl,
      percentReturn: settledCostBasis ? (netPnl / settledCostBasis) * 100 : 0,
      totalR: closedTrades.reduce((total, trade) => total + trade.rMultiple, 0),
      totalTrades: closedTrades.length,
      winRate: closedTrades.length ? (wins.length / closedTrades.length) * 100 : 0,
      profitFactor: grossLoss ? grossWin / grossLoss : grossWin ? grossWin : 0,
      expectancy: closedTrades.length ? closedTrades.reduce((total, trade) => total + trade.rMultiple, 0) / closedTrades.length : 0,
      avgRWin: wins.length ? average(wins.map((trade) => trade.rMultiple)) : 0,
      avgRLoss: losses.length ? average(losses.map((trade) => trade.rMultiple)) : 0,
      averageWin: wins.length ? average(wins.map((trade) => trade.pnl)) : 0,
      averageLoss: losses.length ? Math.abs(average(losses.map((trade) => trade.pnl))) : 0,
      averageRisk: closedTrades.length ? average(closedTrades.map((trade) => trade.risk || 0)) : 0,
      longestWinStreak: longestTradeStreak(orderedClosedTrades, "WIN"),
      longestLossStreak: longestTradeStreak(orderedClosedTrades, "LOSS"),
      needsReview: filteredTrades.filter((trade) => tradeNeedsReview(trade, setupTemplates)).length
    };
  }, [closedTrades, endDate, filteredTrades, setupTemplates, startDate]);

  const pnlChartData = useMemo(() => {
    let cumulativePnl = 0;
    return [...closedTrades]
      .sort((a, b) => tradePnlDate(a).localeCompare(tradePnlDate(b)))
      .map((trade) => {
        cumulativePnl += trade.pnl;
        return {
          label: tradePnlDate(trade),
          pnl: trade.pnl,
          cumulativePnl,
          cumulativePnlPositive: cumulativePnl >= 0 ? cumulativePnl : 0,
          cumulativePnlNegative: cumulativePnl < 0 ? cumulativePnl : 0
        };
      });
  }, [closedTrades]);

  const rChartData = useMemo(() => {
    let cumulativeR = 0;
    const grouped = closedTrades.reduce<Record<string, number>>((groups, trade) => {
      const date = tradePnlDate(trade);
      groups[date] = (groups[date] || 0) + trade.rMultiple;
      return groups;
    }, {});

    return Object.entries(grouped)
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([label, totalR]) => {
        cumulativeR += totalR;
        return { label, totalR, cumulativeR };
      });
  }, [closedTrades]);

  const rDistributionData = useMemo(() => {
    const buckets = [
      { bucket: "< -2R", min: -Infinity, max: -2 },
      { bucket: "-2 to -1R", min: -2, max: -1 },
      { bucket: "-1 to 0R", min: -1, max: 0 },
      { bucket: "0 to 1R", min: 0, max: 1 },
      { bucket: "1 to 2R", min: 1, max: 2 },
      { bucket: "> 2R", min: 2, max: Infinity }
    ];
    return buckets.map((bucket) => ({
      bucket: bucket.bucket,
      trades: closedTrades.filter((trade) => trade.rMultiple >= bucket.min && trade.rMultiple < bucket.max).length
    }));
  }, [closedTrades]);

  const gradePerformanceData = useMemo(() => {
    const grouped = closedTrades.reduce<Record<string, { trades: number; totalR: number }>>((groups, trade) => {
      const grade = effectiveGrade(trade, setupTemplates);
      groups[grade] = groups[grade] || { trades: 0, totalR: 0 };
      groups[grade].trades += 1;
      groups[grade].totalR += trade.rMultiple;
      return groups;
    }, {});
    return Object.entries(grouped).map(([grade, value]) => ({
      grade,
      trades: value.trades,
      avgR: value.trades ? value.totalR / value.trades : 0
    }));
  }, [closedTrades, setupTemplates]);

  const weeklyFrequencyData = useMemo(() => {
    const grouped: Record<
      string,
      {
        weekStart: string;
        weekEnd: string;
        openedCount: number;
        closedCount: number;
        netPnl: number;
        totalR: number;
        mistakeCount: number;
        needsReviewCount: number;
      }
    > = {};
    const ensureWeek = (weekStart: string) => {
      grouped[weekStart] = grouped[weekStart] || {
        weekStart,
        weekEnd: weekEndForStart(weekStart),
        openedCount: 0,
        closedCount: 0,
        netPnl: 0,
        totalR: 0,
        mistakeCount: 0,
        needsReviewCount: 0
      };
      return grouped[weekStart];
    };

    openedTrades.forEach((trade) => {
      const week = ensureWeek(weekStartForDate(trade.entryDate));
      week.openedCount += 1;
    });

    closedLifecycleTrades.forEach((trade) => {
      const week = ensureWeek(weekStartForDate(trade.exitDate));
      week.closedCount += 1;
      week.netPnl += Number(trade.pnl || 0);
      week.totalR += Number(trade.rMultiple || 0);
      week.mistakeCount += trade.mistakeTags?.length || 0;
      if (tradeNeedsReview(trade, setupTemplates)) week.needsReviewCount += 1;
    });

    return Object.values(grouped)
      .sort((a, b) => a.weekStart.localeCompare(b.weekStart))
      .map((week) => ({
        ...week,
        label: weeklyLabel(week.weekStart),
        avgR: week.closedCount ? week.totalR / week.closedCount : 0,
        reviewCompletion: week.closedCount ? ((week.closedCount - week.needsReviewCount) / week.closedCount) * 100 : 100
      }));
  }, [closedLifecycleTrades, openedTrades, setupTemplates]);

  const weeklyFrequencySummary = useMemo(() => {
    const latest = weeklyFrequencyData[weeklyFrequencyData.length - 1] || null;
    const previous = weeklyFrequencyData[weeklyFrequencyData.length - 2] || null;
    const priorTwo = weeklyFrequencyData.slice(-3, -1);
    const activeWeeks = weeklyFrequencyData.filter((week) => week.openedCount > 0 || week.closedCount > 0).length;
    const closedSample = sum(weeklyFrequencyData.map((week) => week.closedCount));
    const totalR = sum(weeklyFrequencyData.map((week) => week.totalR));
    let cumulativeR = 0;
    let peakR = 0;
    let maxDrawdownR = 0;
    let currentDrawdownR = 0;
    weeklyFrequencyData.forEach((week) => {
      cumulativeR += week.totalR;
      peakR = Math.max(peakR, cumulativeR);
      currentDrawdownR = cumulativeR - peakR;
      maxDrawdownR = Math.min(maxDrawdownR, currentDrawdownR);
    });
    const averageOpened = weeklyFrequencyData.length ? average(weeklyFrequencyData.map((week) => week.openedCount)) : 0;
    const latestOpened = latest?.openedCount || 0;
    const latestAvgR = latest?.avgR || 0;
    const latestReviewCompletion = latest?.reviewCompletion ?? 100;
    const latestMistakes = latest?.mistakeCount || 0;
    const priorAvgR = priorTwo.length ? average(priorTwo.map((week) => week.avgR)) : 0;
    const openedRisingTwoWeeks =
      weeklyFrequencyData.length >= 3 &&
      weeklyFrequencyData[weeklyFrequencyData.length - 3].openedCount < weeklyFrequencyData[weeklyFrequencyData.length - 2].openedCount &&
      weeklyFrequencyData[weeklyFrequencyData.length - 2].openedCount < weeklyFrequencyData[weeklyFrequencyData.length - 1].openedCount;
    const avgRFallingTwoWeeks =
      weeklyFrequencyData.length >= 3 &&
      weeklyFrequencyData[weeklyFrequencyData.length - 3].avgR > weeklyFrequencyData[weeklyFrequencyData.length - 2].avgR &&
      weeklyFrequencyData[weeklyFrequencyData.length - 2].avgR > weeklyFrequencyData[weeklyFrequencyData.length - 1].avgR;
    const tradeCountElevated = averageOpened > 0 && latestOpened >= Math.max(averageOpened * 1.25, averageOpened + 2);
    const drawdownWorsening = weeklyFrequencyData.length >= 2 && currentDrawdownR < 0 && latest && previous && latest.totalR < previous.totalR;
    const noMistakeSpike = !previous || latestMistakes <= Math.max(previous.mistakeCount + 1, previous.mistakeCount * 1.5);
    const reviewCompletionGood = latestReviewCompletion >= 80;

    let status = "NORMAL";
    let reason = "Trade frequency is stable and average R is near or above breakeven.";
    if (closedSample < 12 || activeWeeks < 3) {
      status = "INSUFFICIENT SAMPLE";
      reason = `${closedSample} closed trades across ${activeWeeks} active weeks. Need at least 12 closed trades and 3 active weeks.`;
    } else if (drawdownWorsening && latestAvgR < 0 && tradeCountElevated) {
      status = "PAUSE / REVIEW";
      reason = `Net R drawdown is worsening, latest average R is ${latestAvgR.toFixed(2)}R, and opened trades are elevated versus the weekly average.`;
    } else if (openedRisingTwoWeeks && (latestAvgR < 0 || avgRFallingTwoWeeks)) {
      status = "REDUCE FREQUENCY";
      reason = latestAvgR < 0
        ? `Opened trades rose for 2 weeks while latest average R is negative at ${latestAvgR.toFixed(2)}R.`
        : "Opened trades rose for 2 weeks while average R per trade fell for 2 weeks.";
    } else if (latestAvgR > 0 && totalR > 0 && reviewCompletionGood && noMistakeSpike) {
      status = "CAN INCREASE CAREFULLY";
      reason = `Average R is positive, selected-range net R is ${totalR.toFixed(2)}R, reviews are ${latestReviewCompletion.toFixed(0)}% complete, and mistakes are not spiking.`;
    }

    return {
      latest,
      previous,
      status,
      reason,
      activeWeeks,
      closedSample,
      totalR,
      currentDrawdownR,
      maxDrawdownR,
      averageTradesPerWeek: averageOpened,
      bestWeekR: weeklyFrequencyData.length ? Math.max(...weeklyFrequencyData.map((week) => week.totalR)) : 0,
      worstWeekR: weeklyFrequencyData.length ? Math.min(...weeklyFrequencyData.map((week) => week.totalR)) : 0
    };
  }, [weeklyFrequencyData]);

  if (tradeId) {
    return <LegacyTradeDetailHost />;
  }

  return (
    <div className="branden-journal-content" suppressHydrationWarning>
        <header className="branden-route-header">
          <div>
            <p className="eyebrow">Branden journal</p>
            <h1>Dashboard</h1>
            <span>{isLoading ? "Loading dashboard..." : `${summary.totalTrades} closed trades in view`}</span>
          </div>
        </header>

        {error ? <p className="status error">{error}</p> : null}

        <div className="trade-date-filters trade-toolbar-filters">
          <label>
            Start date
            <input type="date" value={startDate} onChange={(event) => setStartDate(event.target.value)} />
          </label>
          <label>
            End date
            <input type="date" value={endDate} onChange={(event) => setEndDate(event.target.value)} />
          </label>
          <div className="trade-date-quick-filters">
            <button className="trade-muted-button" type="button" onClick={() => { setStartDate(currentMonthStartDate()); setEndDate(currentDate()); }}>
              Month to date
            </button>
            <button className="trade-muted-button" type="button" onClick={() => { setStartDate(currentYearStartDate()); setEndDate(currentDate()); }}>
              Year to date
            </button>
          </div>
        </div>

        {isLoading ? <p className="status">Loading dashboard...</p> : null}

        {!isLoading ? (
          <>
            <div className="trade-chart-rows">
              <article className="trade-chart-panel top-chart trade-frequency-panel">
                <div className="trade-chart-heading trade-frequency-heading">
                  <div>
                    <p className="eyebrow">Behavior check</p>
                    <h3>Trade frequency vs edge</h3>
                    <small>Frequency counts trades opened that week. Edge uses trades closed that week.</small>
                  </div>
                  <span>F</span>
                </div>
                <div className="trade-frequency-kpis">
                  <article>
                    <span>Frequency status</span>
                    <strong>{weeklyFrequencySummary.status}</strong>
                    <small>{weeklyFrequencySummary.reason}</small>
                  </article>
                  <article>
                    <span>Sample</span>
                    <strong>{weeklyFrequencySummary.closedSample} closed</strong>
                    <small>{weeklyFrequencySummary.activeWeeks} active weeks</small>
                  </article>
                  <article>
                    <span>Latest week</span>
                    <strong>{weeklyFrequencySummary.latest ? `${weeklyFrequencySummary.latest.openedCount} opened` : "—"}</strong>
                    <small className={(weeklyFrequencySummary.latest?.totalR || 0) >= 0 ? "trade-positive" : "trade-negative"}>
                      {weeklyFrequencySummary.latest ? `${weeklyFrequencySummary.latest.closedCount} closed · ${weeklyFrequencySummary.latest.totalR.toFixed(2)}R · ${signedMoney(weeklyFrequencySummary.latest.netPnl)}` : "No trades"}
                    </small>
                  </article>
                  <article>
                    <span>Drawdown</span>
                    <strong>
                      {weeklyFrequencySummary.currentDrawdownR.toFixed(2)}R
                    </strong>
                    <small>Max {weeklyFrequencySummary.maxDrawdownR.toFixed(2)}R</small>
                  </article>
                </div>

                <div className="trade-frequency-chart-grid">
                  <section>
                    <div className="trade-frequency-chart-title">
                      <strong>Frequency vs quality</strong>
                      <span>Opened trades vs average R</span>
                    </div>
                    <ResponsiveContainer width="100%" height={260}>
                      <LineChart data={weeklyFrequencyData} margin={{ top: 18, right: 18, bottom: 8, left: 0 }}>
                        <CartesianGrid strokeDasharray="3 5" stroke="rgba(47, 53, 45, 0.14)" vertical={false} />
                        <XAxis dataKey="label" axisLine={false} tickLine={false} tick={{ fill: "#6f7469", fontSize: 11, fontWeight: 800 }} minTickGap={18} />
                        <YAxis
                          yAxisId="trades"
                          allowDecimals={false}
                          axisLine={false}
                          tickLine={false}
                          tick={{ fill: "#6f7469", fontSize: 11 }}
                          width={44}
                        />
                        <YAxis
                          yAxisId="edge"
                          orientation="right"
                          axisLine={false}
                          tickLine={false}
                          tick={{ fill: "#6f7469", fontSize: 11 }}
                          tickFormatter={(value) => `${Number(value).toFixed(1)}R`}
                          width={54}
                        />
                        <ReferenceLine yAxisId="edge" y={0} stroke="rgba(111, 116, 105, 0.34)" strokeDasharray="4 4" />
                        <Tooltip
                          contentStyle={chartTooltipStyle}
                          formatter={(value, name) => {
                            if (name === "openedCount") return [`${Number(value)} opened trades`, "Trades opened"];
                            if (name === "avgR") return [`${Number(value).toFixed(2)}R`, "Avg R / closed trade"];
                            return [String(value), String(name)];
                          }}
                          labelFormatter={(label) => `Week: ${label}`}
                        />
                        <Line yAxisId="trades" type="monotone" dataKey="openedCount" name="Trades opened" stroke="#8c6a4a" strokeWidth={3} dot={{ r: 4, fill: "#8c6a4a", strokeWidth: 0 }} activeDot={{ r: 6, strokeWidth: 0 }} />
                        <Line yAxisId="edge" type="monotone" dataKey="avgR" name="Avg R / closed trade" stroke="#4f7045" strokeWidth={3} dot={{ r: 4, fill: "#4f7045", strokeWidth: 0 }} activeDot={{ r: 6, strokeWidth: 0 }} />
                      </LineChart>
                    </ResponsiveContainer>
                  </section>

                  <section>
                    <div className="trade-frequency-chart-title">
                      <strong>Frequency vs outcome</strong>
                      <span>Opened trades vs weekly net R</span>
                    </div>
                    <ResponsiveContainer width="100%" height={260}>
                      <LineChart data={weeklyFrequencyData} margin={{ top: 18, right: 18, bottom: 8, left: 0 }}>
                        <CartesianGrid strokeDasharray="3 5" stroke="rgba(47, 53, 45, 0.14)" vertical={false} />
                        <XAxis dataKey="label" axisLine={false} tickLine={false} tick={{ fill: "#6f7469", fontSize: 11, fontWeight: 800 }} minTickGap={18} />
                        <YAxis
                          yAxisId="trades"
                          allowDecimals={false}
                          axisLine={false}
                          tickLine={false}
                          tick={{ fill: "#6f7469", fontSize: 11 }}
                          width={44}
                        />
                        <YAxis
                          yAxisId="edge"
                          orientation="right"
                          axisLine={false}
                          tickLine={false}
                          tick={{ fill: "#6f7469", fontSize: 11 }}
                          tickFormatter={(value) => `${Number(value).toFixed(1)}R`}
                          width={54}
                        />
                        <ReferenceLine yAxisId="edge" y={0} stroke="rgba(111, 116, 105, 0.34)" strokeDasharray="4 4" />
                        <Tooltip
                          contentStyle={chartTooltipStyle}
                          formatter={(value, name) => {
                            if (name === "openedCount") return [`${Number(value)} opened trades`, "Trades opened"];
                            if (name === "totalR") return [`${Number(value).toFixed(2)}R`, "Net R from closed trades"];
                            return [String(value), String(name)];
                          }}
                          labelFormatter={(label) => `Week: ${label}`}
                        />
                        <Line yAxisId="trades" type="monotone" dataKey="openedCount" name="Trades opened" stroke="#8c6a4a" strokeWidth={3} dot={{ r: 4, fill: "#8c6a4a", strokeWidth: 0 }} activeDot={{ r: 6, strokeWidth: 0 }} />
                        <Line yAxisId="edge" type="monotone" dataKey="totalR" name="Net R from closed trades" stroke="#4f7045" strokeWidth={3} dot={{ r: 4, fill: "#4f7045", strokeWidth: 0 }} activeDot={{ r: 6, strokeWidth: 0 }} />
                      </LineChart>
                    </ResponsiveContainer>
                  </section>
                </div>

                <div className="trade-frequency-legend">
                  <span><i className="frequency-dot frequency-bar" /> Left axis: trades opened</span>
                  <span><i className="frequency-dot frequency-net-r" /> Right axis: R from closed trades</span>
                </div>
              </article>

              <div className="trade-chart-row trade-chart-row-two">
                <article className="trade-chart-panel top-chart">
                  <div className="trade-chart-heading"><h3>Settled P&amp;L</h3><span>i</span></div>
                  <strong className={summary.netPnl >= 0 ? "trade-positive" : "trade-negative"}>{money(summary.netPnl)}</strong>
                  <span className={summary.percentReturn >= 0 ? "trade-positive" : "trade-negative"}>{percent(summary.percentReturn)} return on settled cost basis</span>
                  <ResponsiveContainer width="100%" height={320}>
                    <AreaChart data={pnlChartData} margin={{ top: 18, right: 12, bottom: 8, left: 4 }}>
                      <defs>
                        <linearGradient id="dashboardSettledPnlPositiveGradient" x1="0" x2="0" y1="0" y2="1">
                          <stop offset="0%" stopColor="#4f7045" stopOpacity={0.78} />
                          <stop offset="72%" stopColor="#83b56d" stopOpacity={0.28} />
                          <stop offset="100%" stopColor="#dff2d8" stopOpacity={0.06} />
                        </linearGradient>
                        <linearGradient id="dashboardSettledPnlNegativeGradient" x1="0" x2="0" y1="0" y2="1">
                          <stop offset="0%" stopColor="#f4d6d1" stopOpacity={0.08} />
                          <stop offset="28%" stopColor="#c8796f" stopOpacity={0.28} />
                          <stop offset="100%" stopColor="#a85757" stopOpacity={0.72} />
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 5" stroke="rgba(47, 53, 45, 0.16)" vertical={false} />
                      <XAxis dataKey="label" axisLine={false} tickLine={false} tick={{ fill: "#6f7469", fontSize: 11 }} minTickGap={22} />
                      <YAxis axisLine={false} tickLine={false} tick={{ fill: "#6f7469", fontSize: 11 }} tickFormatter={(value) => money(Number(value))} width={62} />
                      <ReferenceLine y={0} stroke="rgba(111, 116, 105, 0.34)" strokeDasharray="4 4" />
                      <Tooltip contentStyle={chartTooltipStyle} formatter={(value) => money(Number(value))} labelFormatter={(label) => `Date: ${label}`} />
                      <Area type="monotone" dataKey="cumulativePnlPositive" stroke="none" fill="url(#dashboardSettledPnlPositiveGradient)" baseValue={0} dot={false} activeDot={false} isAnimationActive={false} />
                      <Area type="monotone" dataKey="cumulativePnlNegative" stroke="none" fill="url(#dashboardSettledPnlNegativeGradient)" baseValue={0} dot={false} activeDot={false} isAnimationActive={false} />
                      <Area type="monotone" dataKey="cumulativePnl" stroke="#4f7045" strokeWidth={2.8} fill="transparent" dot={false} activeDot={{ r: 5, strokeWidth: 0, fill: "#4f7045" }} />
                    </AreaChart>
                  </ResponsiveContainer>
                </article>

                <article className="trade-chart-panel top-chart">
                  <div className="trade-chart-heading"><h3>Total R Return</h3><span>R</span></div>
                  <strong className={summary.totalR >= 0 ? "trade-positive" : "trade-negative"}>{summary.totalR.toFixed(2)}R</strong>
                  <ResponsiveContainer width="100%" height={320}>
                    <AreaChart data={rChartData} margin={{ top: 18, right: 12, bottom: 8, left: 4 }}>
                      <defs>
                        <linearGradient id="dashboardFilteredRGradient" x1="0" x2="0" y1="0" y2="1">
                          <stop offset="0%" stopColor="#6f8f5f" stopOpacity={0.86} />
                          <stop offset="70%" stopColor="#6f8f5f" stopOpacity={0.34} />
                          <stop offset="100%" stopColor="#6f8f5f" stopOpacity={0.04} />
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 5" stroke="rgba(47, 53, 45, 0.16)" vertical={false} />
                      <XAxis dataKey="label" axisLine={false} tickLine={false} tick={{ fill: "#6f7469", fontSize: 11 }} minTickGap={22} />
                      <YAxis axisLine={false} tickLine={false} tick={{ fill: "#6f7469", fontSize: 11 }} tickFormatter={(value) => `${Number(value).toFixed(1)}R`} width={54} />
                      <Tooltip contentStyle={chartTooltipStyle} formatter={(value, name) => [`${Number(value).toFixed(2)}R`, name === "cumulativeR" ? "Cumulative R" : "Daily R"]} labelFormatter={(label) => `Trade date: ${label}`} />
                      <Area type="monotone" dataKey="cumulativeR" stroke="#4f7045" strokeWidth={2.4} fill="url(#dashboardFilteredRGradient)" dot={false} activeDot={{ r: 5, strokeWidth: 0 }} />
                      <Line type="monotone" dataKey="totalR" stroke="#8c6a4a" strokeWidth={1.8} dot={false} />
                    </AreaChart>
                  </ResponsiveContainer>
                </article>
              </div>

              <div className="trade-chart-row trade-chart-row-two">
                <article className="trade-chart-panel top-chart">
                  <div className="trade-chart-heading"><h3>R distribution</h3><span>R</span></div>
                  <ResponsiveContainer width="100%" height={320}>
                    <BarChart data={rDistributionData} margin={{ top: 24, right: 16, bottom: 8, left: -18 }}>
                      <CartesianGrid strokeDasharray="3 5" stroke="rgba(47, 53, 45, 0.16)" vertical={false} />
                      <XAxis dataKey="bucket" axisLine={false} tickLine={false} tick={{ fill: "#6f7469", fontSize: 11 }} interval={0} />
                      <YAxis allowDecimals={false} axisLine={false} tickLine={false} tick={{ fill: "#6f7469", fontSize: 11 }} />
                      <Tooltip contentStyle={chartTooltipStyle} />
                      <Bar dataKey="trades" fill="#8c6a4a" radius={[6, 6, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </article>

                <article className="trade-chart-panel top-chart">
                  <div className="trade-chart-heading"><h3>R by checklist grade</h3><span>G</span></div>
                  <ResponsiveContainer width="100%" height={320}>
                    <BarChart data={gradePerformanceData} margin={{ top: 24, right: 16, bottom: 8, left: -12 }}>
                      <CartesianGrid strokeDasharray="3 5" stroke="rgba(47, 53, 45, 0.16)" vertical={false} />
                      <XAxis dataKey="grade" axisLine={false} tickLine={false} tick={{ fill: "#6f7469", fontSize: 11 }} />
                      <YAxis axisLine={false} tickLine={false} tick={{ fill: "#6f7469", fontSize: 11 }} tickFormatter={(value) => `${Number(value).toFixed(1)}R`} width={46} />
                      <Tooltip contentStyle={chartTooltipStyle} formatter={(value, name) => [name === "avgR" ? `${Number(value).toFixed(2)}R` : String(value), name === "avgR" ? "Avg R" : "Trades"]} />
                      <Bar dataKey="avgR" name="Avg R" fill="#6f8f5f" radius={[6, 6, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </article>
              </div>
            </div>
          </>
        ) : null}
      </div>
  );
}
