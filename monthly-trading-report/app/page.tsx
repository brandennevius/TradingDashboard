"use client";

import dynamic from "next/dynamic";
import { CSSProperties, DragEvent, FormEvent, useEffect, useMemo, useRef, useState } from "react";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  PolarAngleAxis,
  PolarGrid,
  PolarRadiusAxis,
  Radar,
  RadarChart,
  ReferenceLine,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from "recharts";
import type {
  ChecklistInputType,
  ChecklistGradeBand,
  MarketCycleEntry,
  MonthlyReport,
  SetupChecklistGroup,
  SetupChecklistTemplate,
  SetupTemplateCriterion,
  TradeChecklistItem,
  TradeLogEntry,
  TradeReviewSections,
  TradeSide,
  TraderUser
} from "@/lib/types";
import { displayTradeReturnPercent, tradeReturnLabel } from "@/lib/trade-return";
import { emptyTradeReviewSections, hasCompletedTradeReview, resolvedTradeReviewSections } from "@/lib/trade-review";
import BrandenSidebar from "./components/BrandenSidebar";

const BottomToBullChecklist = dynamic(() => import("./components/BottomToBullChecklist"), {
  loading: () => <p className="status">Loading Bottom to Bull...</p>,
  ssr: false
});
const MarketRegimeGauge = dynamic(() => import("./components/MarketRegimeGauge"), {
  loading: () => <p className="status">Loading Market Gauge...</p>,
  ssr: false
});
const OpenHeatDashboard = dynamic(() => import("./components/OpenHeatDashboard"), {
  loading: () => <p className="status">Loading open positions...</p>,
  ssr: false
});
const SpyMarketCycleChart = dynamic(() => import("./components/SpyMarketCycleChart"), {
  loading: () => <p className="status">Loading SPY chart...</p>,
  ssr: false
});
const TimeStopMonitor = dynamic(() => import("./components/TimeStopMonitor"), {
  loading: () => <p className="status">Loading time stops...</p>,
  ssr: false
});
const TradePriceChart = dynamic(() => import("./components/TradePriceChart"), {
  loading: () => <p className="status">Loading chart...</p>,
  ssr: false
});

type NumericFormKey = Exclude<keyof MonthlyReport, "id" | "userId" | "month" | "notes" | "createdAt" | "updatedAt">;
type FormState = {
  month: string;
  notes: string;
} & Record<NumericFormKey, number | string>;
type TraderFilter = "both" | "branden" | "cam";
type TradeLogOwner = "branden" | "cam";
type ActiveTab = "entry" | "dashboard" | "manage" | "trades";
type TradeView = "table" | "calendar";
type BrandenJournalSection =
  | "dashboard"
  | "daily-review"
  | "calendar"
  | "market-cycle"
  | "market-regime"
  | "bottom-to-bull"
  | "open-heat"
  | "watchlist"
  | "trade-log"
  | "time-stop"
  | "benchmark"
  | "setup-builder"
  | "portfolios"
  | "settings";
type TradeFilterKey = "symbol" | "side" | "status" | "setup" | "mistake" | "custom" | "grade" | "review";
type BrandenTradeColumnKey =
  | "status"
  | "side"
  | "symbol"
  | "setup"
  | "portfolio"
  | "openDate"
  | "entry"
  | "size"
  | "closeDate"
  | "exit"
  | "stop"
  | "commission"
  | "usedMargin"
  | "takeProfit"
  | "risk"
  | "cost"
  | "netReturn"
  | "r"
  | "mistake"
  | "custom"
  | "grade"
  | "review";
type TradeFilters = Record<TradeFilterKey, string[]> & {
  startDate: string;
  endDate: string;
};
type BrandenColumnPreference = {
  key: BrandenTradeColumnKey;
  visible: boolean;
};
type TradeFormState = {
  importSource: string;
  importRowKey: string;
  symbol: string;
  side: TradeSide;
  entryDate: string;
  exitDate: string;
  avgEntry: number | string;
  exitPrice: number | string;
  shares: number | string;
  risk: number | string;
  pnl: number | string;
  rMultiple: number | string;
  returnPercent: number | string;
  daysInTrade: number | string;
  status: TradeLogEntry["status"];
  openTime: string;
  closeTime: string;
  stopPrice: number | string;
  takeProfitPrice: number | string;
  commission: number | string;
  usedMargin: number | string;
  portfolioTag: string;
  emotion: string;
  tradeQuality: string;
  setupTags: string;
  mistakeTags: string;
  customTags: string;
  manualGrade: string;
  checklistItems: TradeChecklistItem[];
  notes: string;
  reviewSections: TradeReviewSections;
  screenshots: string[];
  chartLinks: string[];
  executions: TradeLogEntry["executions"];
};
type BusyProgress = {
  label: string;
  current: number;
  total: number;
  detail: string;
} | null;
const noFilterSelection = "__none_selected__";
type ExcelTradeDraft = TradeFormState & {
  importedPnl: number;
  importedRMultiple: number;
  importedReturnPercent: number;
  importedDaysInTrade: number;
  importedGrade: string;
};
type MarketCycleFormState = {
  date: string;
  trendDay: number | string;
  phase: string;
  notes: string;
};

const marketCyclePhaseOptions = [
  "Early Uptrend",
  "Middle Uptrend",
  "Late Uptrend",
  "Early Downtrend",
  "Middle Downtrend",
  "Late Downtrend"
] as const;
const manualHiddenTag = "Manually hidden";
const marketCyclePhaseColors: Record<(typeof marketCyclePhaseOptions)[number], string> = {
  "Early Uptrend": "#5B8CFF",
  "Middle Uptrend": "#2BB673",
  "Late Uptrend": "#C6A700",
  "Early Downtrend": "#FF8A3D",
  "Middle Downtrend": "#E35D6A",
  "Late Downtrend": "#7A5AF8"
};

function marketCyclePhaseKey(phase: string) {
  return `phase_${phase.toLowerCase().replace(/[^a-z0-9]+/g, "_")}`;
}

function isEmbeddedBrandenRoute() {
  return typeof window !== "undefined" && window.location.pathname.startsWith("/journal/branden/");
}

function marketCycleLabel(entry: MarketCycleEntry) {
  if (entry.phase.includes("Uptrend")) {
    return "Uptrend";
  }

  if (entry.phase.includes("Downtrend")) {
    return "Downtrend";
  }

  if (entry.trendDay > 0) {
    return "Uptrend";
  }

  if (entry.trendDay < 0) {
    return "Downtrend";
  }

  return "-";
}

const brandenColor = "#6f8f5f";
const camColor = "#8c6a4a";
const chartGridColor = "#c8d9bd";
const chartAxisColor = "#6f7469";
const chartMargin = { top: 18, right: 22, bottom: 8, left: 4 };
const chartTooltipStyle: CSSProperties = {
  background: "rgba(255, 250, 240, 0.96)",
  border: "1px solid #b9d6a8",
  borderRadius: 8,
  boxShadow: "0 16px 34px rgba(86, 100, 72, 0.14)",
  color: "#2f352d"
};
const chartLegendStyle = { color: chartAxisColor, fontWeight: 800 };
const defaultChecklistGradeBands: ChecklistGradeBand[] = [
  { id: "grade-a-plus", label: "A+", minScore: 10, maxScore: null },
  { id: "grade-a", label: "A", minScore: 8, maxScore: 9 },
  { id: "grade-b-plus", label: "B+", minScore: 7, maxScore: 7 },
  { id: "grade-b", label: "B", minScore: 6, maxScore: 6 },
  { id: "grade-c", label: "C", minScore: 0, maxScore: 5 }
];
const emptyForm: FormState = {
  month: new Date().toISOString().slice(0, 7),
  accountSize: 0,
  totalReturn: 0,
  percentReturn: 0,
  netPnl: 0,
  totalPayouts: 0,
  totalTrades: 0,
  winRate: 0,
  avgR: 0,
  totalR: 0,
  avgWinR: 0,
  avgLossR: 0,
  avgWin: 0,
  avgLoss: 0,
  avgRisk: 0,
  currentRiskPercent: 0,
  expectedValueR: 0,
  sharpeRatio: 0,
  avgTradeLength: 0,
  avgSwingLength: 0,
  longestWinStreak: 0,
  longestLossStreak: 0,
  notes: ""
};

const emptyTradeForm: TradeFormState = {
  importSource: "",
  importRowKey: "",
  symbol: "",
  side: "LONG",
  entryDate: new Date().toISOString().slice(0, 10),
  exitDate: "",
  avgEntry: "",
  exitPrice: "",
  shares: "",
  risk: "",
  pnl: 0,
  rMultiple: 0,
  returnPercent: 0,
  daysInTrade: 0,
  status: "OPEN",
  openTime: "",
  closeTime: "",
  stopPrice: "",
  takeProfitPrice: "",
  commission: "",
  usedMargin: "",
  portfolioTag: "",
  emotion: "",
  tradeQuality: "",
  setupTags: "",
  mistakeTags: "",
  customTags: "",
  manualGrade: "",
  checklistItems: [],
  notes: "",
  reviewSections: { ...emptyTradeReviewSections },
  screenshots: [],
  chartLinks: [],
  executions: []
};

function currentMonthStartDate() {
  const now = new Date();
  return `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, "0")}-01`;
}

function currentDate() {
  const now = new Date();
  return `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, "0")}-${String(now.getDate()).padStart(2, "0")}`;
}

function daysAgoDate(days: number) {
  const date = new Date();
  date.setDate(date.getDate() - days);
  return `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, "0")}-${String(date.getDate()).padStart(2, "0")}`;
}

function currentYearStartDate() {
  return `${new Date().getFullYear()}-01-01`;
}

const emptyTradeFilters: TradeFilters = {
  symbol: [],
  side: [],
  status: [],
  setup: [],
  mistake: [],
  custom: [],
  grade: [],
  review: [],
  startDate: currentMonthStartDate(),
  endDate: currentDate()
};

const emptyMarketCycleForm: MarketCycleFormState = {
  date: new Date().toISOString().slice(0, 10),
  trendDay: "",
  phase: "",
  notes: ""
};

const defaultBrandenColumns: BrandenColumnPreference[] = [
  { key: "status", visible: true },
  { key: "side", visible: true },
  { key: "symbol", visible: true },
  { key: "setup", visible: true },
  { key: "portfolio", visible: true },
  { key: "openDate", visible: true },
  { key: "entry", visible: true },
  { key: "size", visible: true },
  { key: "closeDate", visible: true },
  { key: "exit", visible: true },
  { key: "stop", visible: false },
  { key: "commission", visible: false },
  { key: "usedMargin", visible: false },
  { key: "takeProfit", visible: false },
  { key: "risk", visible: false },
  { key: "cost", visible: true },
  { key: "netReturn", visible: true },
  { key: "r", visible: true },
  { key: "mistake", visible: false },
  { key: "custom", visible: true },
  { key: "grade", visible: true },
  { key: "review", visible: true }
];

const fieldGroups = [
  {
    title: "Performance",
    fields: [
      ["accountSize", "Account size", "$"],
      ["netPnl", "Net P&L", "$"],
      ["totalR", "Total net R multiple", "R"]
    ]
  },
  {
    title: "Edge",
    fields: [
      ["totalTrades", "Total trades", ""],
      ["winRate", "Win rate", "%"]
    ]
  },
  {
    title: "Risk",
    fields: [
      ["avgRisk", "Average risk", "$"],
      ["currentRiskPercent", "Current risk", "%"],
      ["avgWinR", "Avg winning R", "R"],
      ["avgLossR", "Avg losing R", "R"]
    ]
  },
  {
    title: "Behavior",
    fields: [["avgTradeLength", "Avg trade length", "days"]]
  }
] as const;

function money(value: number) {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0
  }).format(value || 0);
}

function pct(value: number) {
  return `${(value || 0).toFixed(2)}%`;
}

function monthLabel(value: string) {
  const [year, month] = value.split("-");
  return new Date(Number(year), Number(month) - 1).toLocaleDateString("en-US", {
    month: "short",
    year: "numeric"
  });
}

function numberValue(value: number | string) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : 0;
}

function calculateDerivedMetrics(input: FormState): FormState {
  const accountSize = numberValue(input.accountSize);
  const netPnl = numberValue(input.netPnl);
  const totalTrades = numberValue(input.totalTrades);
  const avgRisk = numberValue(input.avgRisk);
  const avgWinR = numberValue(input.avgWinR);
  const avgLossR = numberValue(input.avgLossR);
  const totalR = numberValue(input.totalR);
  const percentReturn = accountSize ? (netPnl / accountSize) * 100 : 0;
  const avgR = totalTrades ? totalR / totalTrades : 0;
  const winRate = Math.min(Math.max(numberValue(input.winRate) / 100, 0), 1);
  const expectedValueR = winRate * avgWinR + (1 - winRate) * avgLossR;
  const avgWin = avgWinR * avgRisk;
  const avgLoss = Math.abs(avgLossR * avgRisk);

  return {
    ...input,
    accountSize,
    netPnl,
    totalTrades,
    totalR,
    avgRisk,
    avgWinR,
    avgLossR,
    winRate: numberValue(input.winRate),
    currentRiskPercent: numberValue(input.currentRiskPercent),
    avgTradeLength: numberValue(input.avgTradeLength),
    avgSwingLength: numberValue(input.avgSwingLength),
    longestWinStreak: numberValue(input.longestWinStreak),
    longestLossStreak: numberValue(input.longestLossStreak),
    totalPayouts: numberValue(input.totalPayouts),
    totalReturn: netPnl,
    percentReturn,
    avgR,
    expectedValueR,
    avgWin,
    avgLoss,
    sharpeRatio: 0
  };
}

function average(values: number[]) {
  return values.reduce((total, value) => total + value, 0) / Math.max(values.length, 1);
}

function clamp(value: number, min = 0, max = 100) {
  return Math.min(Math.max(value, min), max);
}

function standardDeviation(values: number[]) {
  if (values.length < 2) {
    return 0;
  }

  const mean = average(values);
  const variance = average(values.map((value) => (value - mean) ** 2));
  return Math.sqrt(variance);
}

function returnStabilityScore(reports: MonthlyReport[]) {
  const returns = reports.map((report) => report.percentReturn);
  const deviation = standardDeviation(returns);

  return deviation ? average(returns) / deviation : 0;
}

function aggregateByMonth(reports: MonthlyReport[]) {
  const grouped = new Map<string, MonthlyReport[]>();

  for (const report of reports) {
    grouped.set(report.month, [...(grouped.get(report.month) || []), report]);
  }

  return Array.from(grouped.entries())
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([month, items]) => {
      const sum = (key: keyof MonthlyReport) =>
        items.reduce((total, report) => total + Number(report[key] || 0), 0);
      const avg = (key: keyof MonthlyReport) => sum(key) / Math.max(items.length, 1);

      return {
        month,
        label: monthLabel(month),
        netPnl: sum("netPnl"),
        totalTrades: sum("totalTrades"),
        percentReturn: avg("percentReturn"),
        avgR: avg("avgR"),
        totalR: sum("totalR"),
        winRate: avg("winRate"),
        sharpeRatio: avg("sharpeRatio")
      };
    });
}

function buildTraderSummaries(reports: MonthlyReport[]) {
  return ["branden", "cam"].map((trader) => {
    const items = reports.filter((report) => report.userId === trader);
    const totalNetPnl = items.reduce((total, report) => total + report.netPnl, 0);
    const totalTrades = items.reduce((total, report) => total + report.totalTrades, 0);
    const avgR = items.reduce((total, report) => total + report.avgR, 0) / Math.max(items.length, 1);
    const winRate = items.reduce((total, report) => total + report.winRate, 0) / Math.max(items.length, 1);
    const expectedValueR =
      items.reduce((total, report) => total + report.expectedValueR, 0) / Math.max(items.length, 1);
    const returnStability = returnStabilityScore(items);
    const bestMonth = items.reduce<MonthlyReport | null>(
      (best, report) => (!best || report.netPnl > best.netPnl ? report : best),
      null
    );

    return {
      trader,
      displayName: trader.charAt(0).toUpperCase() + trader.slice(1),
      reports: items.length,
      totalNetPnl,
      totalTrades,
      avgR,
      winRate,
      expectedValueR,
      returnStability,
      bestMonth
    };
  });
}

function buildTraderMonthlyReview(reports: MonthlyReport[]) {
  const months = Array.from(new Set(reports.map((report) => report.month))).sort();

  return months.map((month) => {
    const branden = reports.find((report) => report.month === month && report.userId === "branden");
    const cam = reports.find((report) => report.month === month && report.userId === "cam");

    return {
      month,
      label: monthLabel(month),
      brandenPnl: branden?.netPnl || 0,
      camPnl: cam?.netPnl || 0,
      brandenPercentReturn: branden?.percentReturn || 0,
      camPercentReturn: cam?.percentReturn || 0,
      brandenAvgR: branden?.avgR || 0,
      camAvgR: cam?.avgR || 0,
      brandenTotalR: branden?.totalR || 0,
      camTotalR: cam?.totalR || 0,
      brandenWinRate: branden?.winRate || 0,
      camWinRate: cam?.winRate || 0,
      brandenTrades: branden?.totalTrades || 0,
      camTrades: cam?.totalTrades || 0
    };
  });
}

function showsTrader(filter: TraderFilter, trader: "branden" | "cam") {
  return filter === "both" || filter === trader;
}

function splitTags(value: string) {
  return value
    .split(",")
    .map((tag) => tag.trim())
    .filter(Boolean);
}

function sortedUnique(values: string[]) {
  return Array.from(new Set(values.map((value) => value.trim()).filter(Boolean))).sort((a, b) => a.localeCompare(b));
}

const BREAKEVEN_R_THRESHOLD = 0.1;

function tradeStatus(pnl: number, hasExit: boolean, rMultiple = 0): TradeLogEntry["status"] {
  if (!hasExit) {
    return "OPEN";
  }

  if (Math.abs(rMultiple) < BREAKEVEN_R_THRESHOLD) {
    return "BREAKEVEN";
  }

  if (pnl > 0) {
    return "WIN";
  }

  if (pnl < 0) {
    return "LOSS";
  }

  return "BREAKEVEN";
}

function hasPartialExitTag(tags: string[]) {
  return tags.some((tag) => tag.trim().toLowerCase() === "partial exits");
}

function countsAsSettledTrade(trade: Pick<TradeLogEntry, "status" | "customTags" | "pnl">) {
  return trade.status !== "OPEN" || (hasPartialExitTag(trade.customTags) && numberValue(trade.pnl) !== 0);
}

function normalizedTradeStatus(trade: Pick<TradeLogEntry, "status" | "pnl" | "rMultiple">): TradeLogEntry["status"] {
  return tradeStatus(numberValue(trade.pnl), trade.status !== "OPEN", numberValue(trade.rMultiple));
}

function daysBetween(entryDate: string, exitDate: string) {
  if (!entryDate || !exitDate) {
    return 0;
  }

  const start = new Date(`${entryDate}T00:00:00`);
  const end = new Date(`${exitDate}T00:00:00`);
  const diff = end.getTime() - start.getTime();

  return diff > 0 ? Math.ceil(diff / 86_400_000) : 0;
}

function formatTradeDateTime(dateValue: string, timeValue: string) {
  if (!dateValue) {
    return "Open";
  }

  const time = String(timeValue || "").match(/^(\d{2}):(\d{2})/)?.slice(1);
  const date = new Date(`${dateValue}T${time ? `${time[0]}:${time[1]}:00` : "00:00:00"}`);

  if (Number.isNaN(date.getTime())) {
    return dateValue;
  }

  return date.toLocaleString("en-US", {
    month: "short",
    day: "numeric",
    ...(time
      ? {
          hour: "numeric",
          minute: "2-digit"
        }
      : {})
  });
}

function calculateTradeMetrics(input: TradeFormState) {
  const avgEntry = numberValue(input.avgEntry);
  const exitPrice = numberValue(input.exitPrice);
  const shares = numberValue(input.shares);
  const risk = numberValue(input.risk);
  const hasPricedExecution = Boolean(input.exitDate && avgEntry && exitPrice && shares);
  const hasRealizedPartialExit = hasPartialExitTag(splitTags(input.customTags));
  const direction = input.side === "LONG" ? 1 : -1;
  const fallbackPnl = numberValue(input.pnl);
  const shouldTrustStoredPnl = Boolean(input.importSource);
  const pnl =
    input.status === "OPEN" && !input.exitDate && !hasRealizedPartialExit
      ? 0
      : shouldTrustStoredPnl
        ? fallbackPnl
        : hasPricedExecution
        ? (exitPrice - avgEntry) * shares * direction
        : fallbackPnl;
  const cost = avgEntry * shares;
  const daysInTrade = daysBetween(input.entryDate, input.exitDate) || numberValue(input.daysInTrade);
  const rMultiple = risk ? pnl / risk : numberValue(input.rMultiple);
  const status = input.exitDate
    ? tradeStatus(pnl, true, rMultiple)
    : input.status === "OPEN"
      ? "OPEN"
      : pnl
        ? tradeStatus(pnl, false, rMultiple)
        : input.status;

  return {
    avgEntry,
    exitPrice,
    shares,
    risk,
    pnl,
    rMultiple,
    returnPercent: cost ? (pnl / cost) * 100 : numberValue(input.returnPercent),
    daysInTrade,
    status
  };
}

function calculatedStopPrice(input: TradeFormState) {
  const avgEntry = numberValue(input.avgEntry);
  const shares = numberValue(input.shares);
  const risk = numberValue(input.risk);

  if (!avgEntry || !shares || !risk) {
    return 0;
  }

  const riskPerShare = risk / shares;
  return input.side === "LONG" ? avgEntry - riskPerShare : avgEntry + riskPerShare;
}

function weightedAverage<T>(items: T[], value: (item: T) => number, weight: (item: T) => number) {
  const weighted = items.reduce(
    (total, item) => {
      const itemWeight = Math.max(0, weight(item));
      return {
        value: total.value + value(item) * itemWeight,
        weight: total.weight + itemWeight
      };
    },
    { value: 0, weight: 0 }
  );

  return weighted.weight ? weighted.value / weighted.weight : 0;
}

function checklistScore(items: TradeChecklistItem[], gradeBands: ChecklistGradeBand[]) {
  const total = items.reduce((sum, item) => sum + numberValue(item.points), 0);
  const earned = items.reduce((sum, item) => {
    const maxPoints = numberValue(item.points);
    if ((item.inputType || "boolean") === "points") {
      return sum + Math.max(0, Math.min(maxPoints, numberValue(item.score ?? 0)));
    }

    return sum + (item.met ? maxPoints : 0);
  }, 0);

  return {
    earned,
    total,
    percent: total ? (earned / total) * 100 : 0,
    grade: checklistGrade(earned, gradeBands)
  };
}

function checklistGrade(score: number, gradeBands: ChecklistGradeBand[]) {
  const sortedBands = [...gradeBands].sort((a, b) => b.minScore - a.minScore);
  const match = sortedBands.find((band) => score >= band.minScore && (band.maxScore === null || score <= band.maxScore));

  return match?.label || sortedBands[sortedBands.length - 1]?.label || "Unscored";
}

function effectiveTradeGrade(trade: TradeLogEntry, templates: SetupChecklistTemplate[]) {
  const manualGrade = trade.manualGrade?.trim();
  if (manualGrade) return manualGrade;

  const score = checklistScore(resolvedTradeChecklistItems(trade, templates), tradeGradeBands(trade, templates));
  return score.total ? score.grade : "Unscored";
}

function isSystemHiddenTrade(trade: Pick<TradeLogEntry, "importSource" | "importRowKey">) {
  if (trade.importSource !== "cf-statement-pdf") {
    return false;
  }

  const key = trade.importRowKey || "";
  return (
    key.startsWith("cf-open:") ||
    key.startsWith("cf-transaction:") ||
    key.startsWith("cf-position:") ||
    key.startsWith("cf-position-open:") ||
    key.startsWith("cf-position-unmatched:")
  );
}

function isManualHiddenTrade(trade: Pick<TradeLogEntry, "customTags">) {
  return trade.customTags.includes(manualHiddenTag);
}

function effectiveFormGrade(form: TradeFormState, templates: SetupChecklistTemplate[]) {
  const score = checklistScore(form.checklistItems || [], setupGradeBands(primarySetupName(form.setupTags), templates));
  return score.total ? score.grade : form.manualGrade.trim() || "Unscored";
}

function newChecklistItem(): TradeChecklistItem {
  return {
    id: `criteria-${Date.now()}-${Math.random().toString(16).slice(2)}`,
    criteria: "",
    points: 1,
    met: false,
    score: 0,
    inputType: "boolean",
    groupName: "",
    importTagKey: "",
    importTagValue: ""
  };
}

function newSetupTemplateCriterion(inputType: ChecklistInputType = "boolean"): SetupTemplateCriterion {
  return {
    id: `criteria-${Date.now()}-${Math.random().toString(16).slice(2)}`,
    criteria: "",
    points: 1,
    inputType,
    importTagKey: "",
    importTagValue: ""
  };
}

function newSetupTemplateGroup(name = "New Group"): SetupChecklistGroup {
  return {
    id: `group-${Date.now()}-${Math.random().toString(16).slice(2)}`,
    name,
    criteria: [newSetupTemplateCriterion()]
  };
}

function templateCriteria(template: SetupChecklistTemplate) {
  if (template.groups?.length) {
    return template.groups.flatMap((group) => group.criteria);
  }

  return template.criteria || [];
}

function newSetupTemplate(): SetupChecklistTemplate {
  return {
    id: `setup-template-${Date.now()}-${Math.random().toString(16).slice(2)}`,
    setupName: "",
    description: "",
    gradeBands: defaultChecklistGradeBands,
    criteria: [newSetupTemplateCriterion()],
    groups: [newSetupTemplateGroup("Checklist")]
  };
}

function otcPresetTemplate(): SetupChecklistTemplate {
  const make = (
    criteria: string,
    points: number,
    importTagKey: string,
    importTagValue: string
  ): SetupTemplateCriterion => ({
    id: `criteria-${Date.now()}-${Math.random().toString(16).slice(2)}`,
    criteria,
    points,
    inputType: "boolean",
    importTagKey,
    importTagValue
  });

  const groups = [
    {
      id: `group-tech-${Date.now()}`,
      name: "Technicals",
      criteria: [
        make("Breakout setup confirmed", 2, "Breakout", "Yes"),
        make("Primary trend aligned", 2, "Trend", "Trend"),
        make("Fresh setup / not too extended", 1, "Freshness", "Yes")
      ]
    },
    {
      id: `group-fund-${Date.now()}`,
      name: "Fundamentals",
      criteria: [
        make("Coverage in place", 1, "Coverage", "Yes"),
        make("COT supportive", 1, "COT", "Yes"),
        make("Valuation acceptable", 1, "Valuation", "Yes"),
        make("Seasonality supportive", 1, "Seasonality", "Yes"),
        make("Earnings catalyst > 25%", 2, "Earnings", "Yes")
      ]
    }
  ] satisfies SetupChecklistGroup[];

  return {
    id: `setup-template-${Date.now()}-${Math.random().toString(16).slice(2)}`,
    setupName: "OTC",
    description: "Excel-backed OTC strategy. Imported tag fields can auto-check these rows from the trade sheet.",
    gradeBands: defaultChecklistGradeBands,
    criteria: groups.flatMap((group) => group.criteria),
    groups
  };
}

function canslimPresetTemplate(): SetupChecklistTemplate {
  const make = (criteria: string, points: number, inputType: ChecklistInputType = "boolean"): SetupTemplateCriterion => ({
    id: `criteria-${Date.now()}-${Math.random().toString(16).slice(2)}`,
    criteria,
    points,
    inputType,
    importTagKey: "",
    importTagValue: ""
  });

  const groups = [
    {
      id: `group-fund-${Date.now()}`,
      name: "Fundamentals",
      criteria: [
        make("Current EPS growth >= 25%", 2),
        make("Sales growth >= 20-25%", 2),
        make("Annual EPS growth strong", 1),
        make("Institutional sponsorship improving", 1)
      ]
    },
    {
      id: `group-lead-${Date.now()}`,
      name: "Leadership",
      criteria: [
        make("Leader in leading industry group", 2),
        make("Relative strength near highs", 2),
        make("Price within range of 52-week highs", 1)
      ]
    },
    {
      id: `group-tech-${Date.now()}`,
      name: "Technicals",
      criteria: [
        make("Proper base formed", 2, "points"),
        make("Valid pivot / buy point", 2, "points"),
        make("Breakout volume strong", 2, "points"),
        make("Not extended from pivot", 1)
      ]
    },
    {
      id: `group-market-${Date.now()}`,
      name: "Market",
      criteria: [make("General market in confirmed uptrend", 2)]
    },
    {
      id: `group-manage-${Date.now()}`,
      name: "Trade Management",
      criteria: [
        make("Stop moved to breakeven at 1:1", 1),
        make("Exit plan followed", 2, "points")
      ]
    }
  ] satisfies SetupChecklistGroup[];

  return {
    id: `setup-template-${Date.now()}-${Math.random().toString(16).slice(2)}`,
    setupName: "CANSLIM",
    description: "Website-only growth breakout checklist. Use this for CANSLIM / Minervini / Roppel-style leadership setups.",
    gradeBands: defaultChecklistGradeBands,
    criteria: groups.flatMap((group) => group.criteria),
    groups
  };
}

function primarySetupName(setupTags: string) {
  return splitTags(setupTags)[0] || "";
}

function setupTemplateFor(setupName: string, templates: SetupChecklistTemplate[]) {
  const normalized = setupName.trim().toLowerCase();
  return templates.find((template) => template.setupName.trim().toLowerCase() === normalized) || null;
}

function setupGradeBands(setupName: string, templates: SetupChecklistTemplate[]) {
  return setupTemplateFor(setupName, templates)?.gradeBands?.length
    ? setupTemplateFor(setupName, templates)?.gradeBands || defaultChecklistGradeBands
    : defaultChecklistGradeBands;
}

function tradeGradeBands(trade: TradeLogEntry, templates: SetupChecklistTemplate[]) {
  return setupGradeBands(trade.setupTags[0] || "", templates);
}

function importTagValueForTrade(trade: Pick<TradeLogEntry, "customTags">, key: string) {
  const normalizedKey = key.trim().toLowerCase();

  if (!normalizedKey) {
    return "";
  }

  const match = trade.customTags.find((tag) => tag.toLowerCase().startsWith(`${normalizedKey}:`));
  return match ? match.split(":").slice(1).join(":").trim() : "";
}

function criterionAutoMet(criteria: SetupTemplateCriterion, trade?: Pick<TradeLogEntry, "customTags"> | null) {
  if (!trade || !criteria.importTagKey?.trim()) {
    return false;
  }

  const importedValue = importTagValueForTrade(trade, criteria.importTagKey);

  if (!importedValue) {
    return false;
  }

  if (!criteria.importTagValue?.trim()) {
    return true;
  }

  return importedValue.toLowerCase() === criteria.importTagValue.trim().toLowerCase();
}

function checklistFromSetupTemplate(
  template: SetupChecklistTemplate,
  existingItems: TradeChecklistItem[] = [],
  trade?: Pick<TradeLogEntry, "customTags"> | null
) {
  return (template.groups?.length ? template.groups : [{ id: "group-default", name: "Checklist", criteria: templateCriteria(template) }]).flatMap((group) =>
    group.criteria.map((criteria) => {
    const existing = existingItems.find(
      (item) => item.id === criteria.id || item.criteria.trim().toLowerCase() === criteria.criteria.trim().toLowerCase()
    );
    const autoMet = criterionAutoMet(criteria, trade);
    const inputType = criteria.inputType || "boolean";

    return {
      id: criteria.id,
      criteria: criteria.criteria,
      points: criteria.points,
      met: existing ? Boolean(existing.met) : autoMet,
      score:
        existing && existing.score !== undefined
          ? numberValue(existing.score)
          : inputType === "points"
            ? autoMet
              ? numberValue(criteria.points)
              : 0
            : autoMet
              ? numberValue(criteria.points)
              : 0,
      inputType,
      groupName: group.name,
      importTagKey: criteria.importTagKey || "",
      importTagValue: criteria.importTagValue || ""
    } satisfies TradeChecklistItem;
  }));
}

function resolvedTradeChecklistItems(trade: TradeLogEntry, templates: SetupChecklistTemplate[]) {
  const template = setupTemplateFor(trade.setupTags[0] || "", templates);

  if (!template) {
    return trade.checklistItems || [];
  }

  return checklistFromSetupTemplate(template, trade.checklistItems || [], trade);
}

function tradeToForm(trade: TradeLogEntry, templates: SetupChecklistTemplate[]): TradeFormState {
  return {
    importSource: trade.importSource || "",
    importRowKey: trade.importRowKey || "",
    symbol: trade.symbol,
    side: trade.side,
    entryDate: trade.entryDate,
    exitDate: trade.exitDate,
    openTime: trade.openTime || "",
    closeTime: trade.closeTime || "",
    avgEntry: trade.avgEntry,
    exitPrice: trade.exitPrice || "",
    stopPrice: trade.stopPrice || "",
    takeProfitPrice: trade.takeProfitPrice || "",
    shares: trade.shares,
    commission: trade.commission || "",
    usedMargin: trade.usedMargin || "",
    risk: trade.risk,
    pnl: trade.pnl,
    rMultiple: trade.rMultiple,
    returnPercent: trade.returnPercent,
    daysInTrade: trade.daysInTrade,
    status: trade.status,
    portfolioTag: trade.portfolioTag || "",
    emotion: trade.emotion || "",
    tradeQuality: trade.tradeQuality || "",
    setupTags: trade.setupTags.join(", "),
    mistakeTags: trade.mistakeTags.join(", "),
    customTags: trade.customTags.join(", "),
    manualGrade: trade.manualGrade || "",
    checklistItems: resolvedTradeChecklistItems(trade, templates),
    notes: trade.notes || "",
    reviewSections: resolvedTradeReviewSections(trade.reviewSections, trade.notes),
    screenshots: trade.screenshots,
    chartLinks: trade.chartLinks || [],
    executions: trade.executions || []
  };
}

function tradePayloadFromForm(input: TradeFormState, options: { preserveStatus?: boolean } = {}) {
  const metrics = calculateTradeMetrics(input);

  return {
    ...input,
    importSource: input.importSource.trim(),
    importRowKey: input.importRowKey.trim(),
    symbol: input.symbol,
    openTime: input.openTime,
    closeTime: input.closeTime,
    stopPrice: numberValue(input.stopPrice),
    takeProfitPrice: numberValue(input.takeProfitPrice),
    commission: numberValue(input.commission),
    usedMargin: numberValue(input.usedMargin),
    portfolioTag: input.portfolioTag.trim(),
    emotion: input.emotion.trim(),
    tradeQuality: input.tradeQuality.trim(),
    setupTags: splitTags(input.setupTags),
    mistakeTags: splitTags(input.mistakeTags),
    customTags: splitTags(input.customTags),
    manualGrade: input.manualGrade.trim(),
    executions: input.executions || [],
    checklistItems: input.checklistItems
      .map((item) => ({
        ...item,
        criteria: item.criteria.trim(),
        points: numberValue(item.points),
        met: Boolean(item.met)
      }))
      .filter((item) => item.criteria && item.points > 0),
    ...metrics,
    status: options.preserveStatus ? input.status : metrics.status
  };
}

type BrokerExecution = {
  execDate: string;
  side: "BUY" | "SELL";
  quantity: number;
  positionEffect: string;
  symbol: string;
  price: number;
};

function parseCsvLine(line: string) {
  const values: string[] = [];
  let current = "";
  let inQuotes = false;

  for (let index = 0; index < line.length; index += 1) {
    const character = line[index];
    const nextCharacter = line[index + 1];

    if (character === '"' && inQuotes && nextCharacter === '"') {
      current += '"';
      index += 1;
    } else if (character === '"') {
      inQuotes = !inQuotes;
    } else if (character === "," && !inQuotes) {
      values.push(current.trim());
      current = "";
    } else {
      current += character;
    }
  }

  values.push(current.trim());
  return values;
}

function parseMoneyValue(value: string) {
  const trimmed = value.trim();
  const isNegative = trimmed.startsWith("(") && trimmed.endsWith(")");
  const normalized = trimmed.replace(/[$,()"']/g, "");
  const parsed = Number(normalized);
  return Number.isFinite(parsed) ? parsed * (isNegative ? -1 : 1) : 0;
}

function normalizeStatementDate(value: string) {
  const [datePart] = value.trim().split(/\s+/);
  const [month, day, year] = datePart.split("/").map(Number);
  const fullYear = year < 100 ? 2000 + year : year;

  if (!month || !day || !fullYear) {
    return new Date().toISOString().slice(0, 10);
  }

  return `${fullYear}-${String(month).padStart(2, "0")}-${String(day).padStart(2, "0")}`;
}

function excelDateToIso(value: unknown) {
  if (value instanceof Date && !Number.isNaN(value.getTime())) {
    return value.toISOString().slice(0, 10);
  }

  if (typeof value === "number" && Number.isFinite(value)) {
    const excelEpoch = Date.UTC(1899, 11, 30);
    return new Date(excelEpoch + value * 86_400_000).toISOString().slice(0, 10);
  }

  if (typeof value === "string" && value.trim()) {
    const date = new Date(value);

    if (!Number.isNaN(date.getTime())) {
      return date.toISOString().slice(0, 10);
    }
  }

  return "";
}

function fallbackImportDate(entryDate: string, exitDate: string) {
  return entryDate || exitDate || new Date().toISOString().slice(0, 10);
}

function importedTradePayload(input: ExcelTradeDraft, templates: SetupChecklistTemplate[]) {
  const template = setupTemplateFor(primarySetupName(input.setupTags), templates);

  return {
    ...input,
    importSource: input.importSource.trim(),
    importRowKey: input.importRowKey.trim(),
    symbol: input.symbol,
    avgEntry: 0,
    exitPrice: 0,
    shares: 0,
    risk: numberValue(input.risk),
    pnl: input.importedPnl,
    rMultiple: input.importedRMultiple,
    returnPercent: input.importedReturnPercent,
    daysInTrade: input.importedDaysInTrade,
    status: tradeStatus(input.importedPnl, Boolean(input.exitDate), input.importedRMultiple),
    setupTags: splitTags(input.setupTags),
    mistakeTags: splitTags(input.mistakeTags),
    customTags: splitTags(input.customTags),
    manualGrade: input.importedGrade.trim(),
    checklistItems: template ? checklistFromSetupTemplate(template, [], { customTags: splitTags(input.customTags) }) : [],
    notes: input.notes,
    screenshots: [],
    chartLinks: input.chartLinks
  };
}

async function parseExcelTradeLogWorkbook(file: File): Promise<ExcelTradeDraft[]> {
  const XLSX = await import("xlsx");
  const buffer = await file.arrayBuffer();
  const workbook = XLSX.read(buffer, { type: "array", cellDates: true });
  const sheetName = workbook.SheetNames.includes("Trade Log") ? "Trade Log" : workbook.SheetNames[0];
  const sheet = workbook.Sheets[sheetName];

  if (!sheet) {
    return [];
  }

  const rows = XLSX.utils.sheet_to_json<unknown[]>(sheet, { header: 1, raw: true, defval: "" });
  const headerIndex = rows.findIndex((row) => row.some((cell) => String(cell).trim().toLowerCase() === "asset"));

  if (headerIndex === -1) {
    return [];
  }

  const headers = rows[headerIndex].map((header) => String(header).trim());
  const columnIndex = (name: string) => headers.findIndex((header) => header.toLowerCase() === name.toLowerCase());
  const indexes = {
    asset: columnIndex("Asset"),
    setup: columnIndex("Setup"),
    side: columnIndex("Side"),
    risk: columnIndex("Risk ($)"),
    pnl: columnIndex("P&L"),
    entry: columnIndex("Entry"),
    exit: columnIndex("Exit"),
    mistake: columnIndex("Mistake"),
    grade: columnIndex("Grade"),
    notes: columnIndex("Notes"),
    link: columnIndex("Link"),
    breakout: columnIndex("Breakout"),
    trend: columnIndex("Trend"),
    freshness: columnIndex("Freshness"),
    coverage: columnIndex("Coverage"),
    cot: columnIndex("COT"),
    valuation: columnIndex("Valuation"),
    seasonality: columnIndex("Seasonality"),
    earnings: headers.findIndex((header) => header.toLowerCase().startsWith("earnings"))
  };
  const technicalIndexes = [
    ["Breakout", indexes.breakout],
    ["Trend", indexes.trend],
    ["Freshness", indexes.freshness],
    ["Coverage", indexes.coverage],
    ["COT", indexes.cot],
    ["Valuation", indexes.valuation],
    ["Seasonality", indexes.seasonality],
    ["Earnings", indexes.earnings]
  ] as const;

  return rows
    .slice(headerIndex + 1)
    .map((row, rowOffset) => {
      const symbol = String(row[indexes.asset] || "").trim().toUpperCase();
      const pnl = parseMoneyValue(String(row[indexes.pnl] || ""));
      const risk = parseMoneyValue(String(row[indexes.risk] || ""));

      if (!symbol) {
        return null;
      }

      const rawEntryDate = excelDateToIso(row[indexes.entry]);
      const exitDate = excelDateToIso(row[indexes.exit]);
      const entryDate = fallbackImportDate(rawEntryDate, exitDate);
      const side = String(row[indexes.side] || "").trim().toUpperCase() === "SHORT" ? "SHORT" : "LONG";
      const importedSetup = indexes.setup >= 0 ? String(row[indexes.setup] || "").trim() : "";
      const importedGrade = indexes.grade >= 0 ? String(row[indexes.grade] || "").trim() : "";
      const primarySetup =
        importedSetup ||
        (technicalIndexes.some(([, index]) => index >= 0 && String(row[index] || "").trim()) ? "OTC" : "");
      const technicalTags = technicalIndexes
        .map(([label, index]) => {
          const value = index >= 0 ? String(row[index] || "").trim() : "";
          return value ? `${label}: ${value}` : "";
        })
        .filter(Boolean);
      const importTags = [...technicalTags];
      if (!rawEntryDate) {
        importTags.push("Missing Entry Date");
      }
      const cellAddress = XLSX.utils.encode_cell({ r: headerIndex + 1 + rowOffset, c: indexes.link });
      const linkCell = sheet[cellAddress];
      const chartLink = String((linkCell?.l?.Target || row[indexes.link] || "") as string).trim();

      return {
        ...emptyTradeForm,
        importSource: `excel-trade-log:${String(sheetName || "sheet").trim().toLowerCase()}`,
        importRowKey: String(headerIndex + 2 + rowOffset),
        symbol,
        side,
        entryDate,
        exitDate,
        risk,
        setupTags: primarySetup,
        mistakeTags: String(row[indexes.mistake] || "").trim(),
        customTags: ["Imported", "Excel Trade Log", ...importTags].join(", "),
        manualGrade: importedGrade,
        notes: String(row[indexes.notes] || "").trim(),
        reviewSections: { ...emptyTradeReviewSections },
        chartLinks: chartLink ? [chartLink] : [],
        importedPnl: pnl,
        importedRMultiple: risk ? pnl / risk : 0,
        importedReturnPercent: 0,
        importedDaysInTrade: daysBetween(entryDate, exitDate),
        importedGrade
      } as ExcelTradeDraft;
    })
    .filter(Boolean) as ExcelTradeDraft[];
}

function parseBrokerStatementTrades(csvText: string): TradeFormState[] {
  const lines = csvText.replace(/^\uFEFF/, "").split(/\r?\n/);
  const sectionStart = lines.findIndex((line) => line.trim() === "Account Trade History");

  if (sectionStart === -1) {
    return [];
  }

  const headerIndex = lines.findIndex((line, index) => index > sectionStart && line.includes("Exec Time") && line.includes("Symbol"));

  if (headerIndex === -1) {
    return [];
  }

  const headers = parseCsvLine(lines[headerIndex]).map((header) => header.trim());
  const headerPosition = (name: string) => headers.findIndex((header) => header.toLowerCase() === name.toLowerCase());
  const indexes = {
    execTime: headerPosition("Exec Time"),
    side: headerPosition("Side"),
    quantity: headerPosition("Qty"),
    positionEffect: headerPosition("Pos Effect"),
    symbol: headerPosition("Symbol"),
    price: headerPosition("Price")
  };
  const executions: BrokerExecution[] = [];

  for (let index = headerIndex + 1; index < lines.length; index += 1) {
    const line = lines[index];

    if (!line.trim()) {
      break;
    }

    const values = parseCsvLine(line);
    const symbol = String(values[indexes.symbol] || "").trim().toUpperCase();
    const side = String(values[indexes.side] || "").trim().toUpperCase();
    const positionEffect = String(values[indexes.positionEffect] || "").trim().toUpperCase();
    const price = parseMoneyValue(String(values[indexes.price] || ""));
    const quantity = Math.abs(parseMoneyValue(String(values[indexes.quantity] || "")));

    if (!symbol || !price || !quantity || (side !== "BUY" && side !== "SELL")) {
      continue;
    }

    executions.push({
      execDate: normalizeStatementDate(String(values[indexes.execTime] || "")),
      side: side as "BUY" | "SELL",
      quantity,
      positionEffect,
      symbol,
      price
    });
  }

  const openLots: Record<string, BrokerExecution[]> = {};
  const forms: TradeFormState[] = [];

  executions
    .sort((a, b) => a.execDate.localeCompare(b.execDate))
    .forEach((execution) => {
      const isOpen = execution.positionEffect.includes("OPEN");
      const isClose = execution.positionEffect.includes("CLOSE");

      if (isOpen) {
        openLots[execution.symbol] = openLots[execution.symbol] || [];
        openLots[execution.symbol].push(execution);
        return;
      }

      if (!isClose) {
        return;
      }

      const lot = openLots[execution.symbol]?.shift();
      const side: TradeSide = lot?.side === "SELL" ? "SHORT" : "LONG";
      const entryDate = lot?.execDate || execution.execDate;
      const avgEntry = lot?.price || execution.price;
      const shares = Math.min(lot?.quantity || execution.quantity, execution.quantity);

      forms.push({
        ...emptyTradeForm,
        symbol: execution.symbol,
        side,
        entryDate,
        exitDate: execution.execDate,
        avgEntry,
        exitPrice: execution.price,
        shares,
        risk: "",
        customTags: "Imported, Schwab, Needs review"
      });
    });

  Object.values(openLots)
    .flat()
    .forEach((lot) => {
      forms.push({
        ...emptyTradeForm,
        symbol: lot.symbol,
        side: lot.side === "SELL" ? "SHORT" : "LONG",
        entryDate: lot.execDate,
        avgEntry: lot.price,
        shares: lot.quantity,
        risk: "",
        customTags: "Imported, Schwab, Needs review"
      });
    });

  return forms;
}

function tradeNeedsReview(trade: TradeLogEntry, templates: SetupChecklistTemplate[]) {
  return (
    !numberValue(trade.risk) ||
    !hasCompletedTradeReview(trade.reviewSections, trade.notes) ||
    (!trade.screenshots.length && !(trade.chartLinks || []).length) ||
    !resolvedTradeChecklistItems(trade, templates).length
  );
}

function tradeBadgeClass(status: TradeLogEntry["status"]) {
  return `trade-badge ${status.toLowerCase()}`;
}

function uniqueTags(trades: TradeLogEntry[], key: "setupTags" | "mistakeTags" | "customTags") {
  return Array.from(new Set(trades.flatMap((trade) => trade[key]))).sort((a, b) => a.localeCompare(b));
}

function uniqueSymbols(trades: TradeLogEntry[]) {
  return Array.from(new Set(trades.map((trade) => trade.symbol))).sort((a, b) => a.localeCompare(b));
}

function matchesSelectedFilter(selected: string[], values: string[]) {
  if (selected.includes(noFilterSelection)) {
    return false;
  }

  return !selected.length || values.some((value) => selected.includes(value));
}

function filterCategory(value: string) {
  const [prefix] = value.split(":");
  const trimmed = prefix.trim();
  return value.includes(":") ? trimmed : `__plain__:${value.trim()}`;
}

function matchesCustomTagFilter(selected: string[], values: string[], options: string[]) {
  if (selected.includes(noFilterSelection)) {
    return false;
  }

  if (!selected.length) {
    return true;
  }

  const optionsByCategory = new Map<string, string[]>();
  const selectedByCategory = new Map<string, string[]>();
  const valuesByCategory = new Map<string, string[]>();

  options.forEach((option) => {
    const category = filterCategory(option);
    optionsByCategory.set(category, [...(optionsByCategory.get(category) || []), option]);
  });

  selected.forEach((option) => {
    const category = filterCategory(option);
    selectedByCategory.set(category, [...(selectedByCategory.get(category) || []), option]);
  });

  values.forEach((value) => {
    const category = filterCategory(value);
    valuesByCategory.set(category, [...(valuesByCategory.get(category) || []), value]);
  });

  for (const [category, categoryOptions] of optionsByCategory.entries()) {
    const categorySelected = selectedByCategory.get(category) || [];

    if (!categorySelected.length || categorySelected.length === categoryOptions.length) {
      continue;
    }

    const rowValues = valuesByCategory.get(category) || [];

    if (!rowValues.length || !rowValues.some((value) => categorySelected.includes(value))) {
      return false;
    }
  }

  return true;
}

function filterLabel(selected: string[], totalOptions: number) {
  if (selected.includes(noFilterSelection)) {
    return "None";
  }

  if (!selected.length || selected.length === totalOptions) {
    return "All";
  }

  return `${selected.length} selected`;
}

function longestTradeStreak(trades: TradeLogEntry[], status: "WIN" | "LOSS") {
  let current = 0;
  let longest = 0;

  for (const trade of trades) {
    const tradeStatus = normalizedTradeStatus(trade);
    if (tradeStatus === status) {
      current += 1;
      longest = Math.max(longest, current);
    } else if (tradeStatus === "WIN" || tradeStatus === "LOSS") {
      current = 0;
    }
  }

  return longest;
}

function parseDateKey(value: string) {
  const [year, month, day] = value.split("-").map(Number);
  return new Date(year, month - 1, day || 1);
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

function pluralize(count: number, singular: string) {
  return `${count} ${singular}${count === 1 ? "" : "s"}`;
}

function dayWinRate(trades: TradeLogEntry[]) {
  const closed = trades.filter((trade) => trade.status !== "OPEN");
  const wins = closed.filter((trade) => trade.pnl > 0);
  return closed.length ? (wins.length / closed.length) * 100 : 0;
}

function tradePnlDate(trade: Pick<TradeLogEntry, "status" | "exitDate" | "entryDate" | "customTags" | "executions" | "pnl">) {
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

function dateInRange(value: string, startDate: string, endDate: string) {
  if (!value) {
    return false;
  }

  if (startDate && value < startDate) {
    return false;
  }

  if (endDate && value > endDate) {
    return false;
  }

  return true;
}

function executionsInDateRange(trade: Pick<TradeLogEntry, "executions">, startDate: string, endDate: string) {
  return (trade.executions || []).filter((execution) => execution.type === "EXIT" && dateInRange(execution.date, startDate, endDate));
}

function tradeHasRangeActivity(
  trade: Pick<TradeLogEntry, "entryDate" | "status" | "exitDate" | "customTags" | "executions" | "pnl">,
  startDate: string,
  endDate: string
) {
  if (!startDate && !endDate) {
    return true;
  }

  const rangedExecutions = executionsInDateRange(trade, startDate, endDate);
  if (rangedExecutions.length) {
    return true;
  }

  if (trade.status === "OPEN" || !countsAsSettledTrade(trade)) {
    return dateInRange(trade.entryDate, startDate, endDate);
  }

  return false;
}

function tradeDisplayPnlInRange(
  trade: Pick<TradeLogEntry, "entryDate" | "status" | "customTags" | "executions" | "pnl">,
  startDate: string,
  endDate: string
) {
  if (!startDate && !endDate) {
    return numberValue(trade.pnl);
  }

  const rangedExecutions = executionsInDateRange(trade, startDate, endDate);
  if (rangedExecutions.length) {
    return rangedExecutions.reduce((sum, execution) => sum + numberValue(execution.pnl), 0);
  }

  if (trade.status === "OPEN" || !countsAsSettledTrade(trade)) {
    return dateInRange(trade.entryDate, startDate, endDate) ? numberValue(trade.pnl) : 0;
  }

  return 0;
}

function tradeDisplayExitDateInRange(trade: Pick<TradeLogEntry, "exitDate" | "executions">, startDate: string, endDate: string) {
  if (!startDate && !endDate) {
    return trade.exitDate;
  }

  const rangedExecutions = executionsInDateRange(trade, startDate, endDate);
  return rangedExecutions.map((execution) => execution.date).sort().at(-1) || "";
}

function normalizeBrandenColumns(value: unknown): BrandenColumnPreference[] {
  const source = Array.isArray(value) ? value : [];
  const normalized: BrandenColumnPreference[] = [];
  const seen = new Set<BrandenTradeColumnKey>();

  source.forEach((item) => {
    if (!item || typeof item !== "object") {
      return;
    }

    const key = String((item as { key?: string }).key || "") as BrandenTradeColumnKey;

    if (!defaultBrandenColumns.some((column) => column.key === key)) {
      return;
    }

    if (seen.has(key)) {
      return;
    }

    seen.add(key);
    normalized.push({
      key,
      visible: "visible" in (item as Record<string, unknown>) ? Boolean((item as { visible?: unknown }).visible) : true
    });
  });

  defaultBrandenColumns.forEach((column) => {
    if (!seen.has(column.key)) {
      normalized.push(column);
    }
  });

  return normalized;
}

function csvValue(value: unknown) {
  const raw = Array.isArray(value) ? value.join("; ") : String(value ?? "");
  return `"${raw.replace(/"/g, '""')}"`;
}

export default function Home() {
  const [isBrandenJournalMode, setIsBrandenJournalMode] = useState(false);
  const [user, setUser] = useState<TraderUser | null>(null);
  const [reports, setReports] = useState<MonthlyReport[]>([]);
  const [trades, setTrades] = useState<TradeLogEntry[]>([]);
  const [form, setForm] = useState<FormState>(emptyForm);
  const [tradeForm, setTradeForm] = useState<TradeFormState>(emptyTradeForm);
  const [editTradeForm, setEditTradeForm] = useState<TradeFormState>(emptyTradeForm);
  const [tradeFilters, setTradeFilters] = useState<TradeFilters>(emptyTradeFilters);
  const [marketCycleEntries, setMarketCycleEntries] = useState<MarketCycleEntry[]>([]);
  const [marketCycleForm, setMarketCycleForm] = useState<MarketCycleFormState>(emptyMarketCycleForm);
  const [savedPortfolios, setSavedPortfolios] = useState<string[]>([]);
  const [activePortfolio, setActivePortfolio] = useState("");
  const [defaultPortfolio, setDefaultPortfolio] = useState("");
  const [importPortfolioTag, setImportPortfolioTag] = useState("");
  const excelImportInputRef = useRef<HTMLInputElement | null>(null);
  const cfImportInputRef = useRef<HTMLInputElement | null>(null);
  const journalBackupInputRef = useRef<HTMLInputElement | null>(null);
  const pendingImportPortfolioRef = useRef("");
  const [columnPreferences, setColumnPreferences] = useState<Record<string, BrandenColumnPreference[]>>({});
  const [selectedTradeId, setSelectedTradeId] = useState("");
  const [selectedTradeIds, setSelectedTradeIds] = useState<string[]>([]);
  const [detailNavigationIds, setDetailNavigationIds] = useState<string[]>([]);
  const [selectedHiddenTradeIds, setSelectedHiddenTradeIds] = useState<string[]>([]);
  const [fullscreenScreenshot, setFullscreenScreenshot] = useState<{ src: string; alt: string } | null>(null);
  const [isTradeModalOpen, setIsTradeModalOpen] = useState(false);
  const [newMistakeDraft, setNewMistakeDraft] = useState({ create: "", edit: "" });
  const [newCustomTagDraft, setNewCustomTagDraft] = useState({ create: "", edit: "" });
  const [setupTemplates, setSetupTemplates] = useState<SetupChecklistTemplate[]>([]);
  const [setupTemplateDrafts, setSetupTemplateDrafts] = useState<SetupChecklistTemplate[]>([]);
  const [draggedSetupCriterion, setDraggedSetupCriterion] = useState<{
    templateId: string;
    groupId: string;
    criteriaId: string;
  } | null>(null);
  const [dragOverSetupCriterionId, setDragOverSetupCriterionId] = useState("");
  const [activeTradeOwner, setActiveTradeOwner] = useState<TradeLogOwner>("branden");
  const [tradeView, setTradeView] = useState<TradeView>("table");
  const [login, setLogin] = useState({ userId: "branden", password: "" });
  const [status, setStatus] = useState("");
  const [busyProgress, setBusyProgress] = useState<BusyProgress>(null);
  const [reportsError, setReportsError] = useState("");
  const [tradesError, setTradesError] = useState("");
  const [isLoading, setIsLoading] = useState(true);
  const [chartFilter, setChartFilter] = useState<TraderFilter>("both");
  const [activeTab, setActiveTab] = useState<ActiveTab>("entry");
  const [isPhoneViewport, setIsPhoneViewport] = useState(false);
  const brandenDashboardRef = useRef<HTMLDivElement | null>(null);
  const brandenTradeListRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!fullscreenScreenshot) {
      return;
    }

    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setFullscreenScreenshot(null);
      }
    };

    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, [fullscreenScreenshot]);

  const monthly = useMemo(() => aggregateByMonth(reports), [reports]);
  const traderSummaries = useMemo(() => buildTraderSummaries(reports), [reports]);
  const traderMonthly = useMemo(() => buildTraderMonthlyReview(reports), [reports]);
  const derivedPreview = useMemo(() => calculateDerivedMetrics(form), [form]);
  const myReports = useMemo(() => reports.filter((report) => report.userId === user?.id), [reports, user]);
  const tradeMetrics = useMemo(() => calculateTradeMetrics(tradeForm), [tradeForm]);
  const editTradeMetrics = useMemo(() => calculateTradeMetrics(editTradeForm), [editTradeForm]);
  const editTradeDisplayReturn = useMemo(
    () =>
      displayTradeReturnPercent({
        symbol: editTradeForm.symbol,
        side: editTradeForm.side,
        avgEntry: editTradeMetrics.avgEntry,
        exitPrice: editTradeMetrics.exitPrice,
        shares: editTradeMetrics.shares,
        pnl: editTradeMetrics.pnl,
        fallbackReturnPercent: numberValue(editTradeForm.returnPercent)
      }),
    [editTradeForm, editTradeMetrics]
  );
  const activeOwnerName = activeTradeOwner === "branden" ? "Branden" : "Cam";
  const canEditActiveTradeLog = user?.id === activeTradeOwner;
  const activeOwnerTrades = useMemo(() => trades.filter((trade) => trade.userId === activeTradeOwner), [activeTradeOwner, trades]);
  const visibleOwnerTrades = useMemo(() => activeOwnerTrades.filter((trade) => !trade.hidden), [activeOwnerTrades]);
  const hiddenOwnerTrades = useMemo(
    () =>
      activeOwnerTrades.filter((trade) => trade.hidden),
    [activeOwnerTrades]
  );
  const setupOptions = useMemo(() => uniqueTags(visibleOwnerTrades, "setupTags"), [visibleOwnerTrades]);
  const mistakeOptions = useMemo(() => uniqueTags(visibleOwnerTrades, "mistakeTags"), [visibleOwnerTrades]);
  const customOptions = useMemo(() => uniqueTags(visibleOwnerTrades, "customTags"), [visibleOwnerTrades]);
  const symbolOptions = useMemo(() => uniqueSymbols(visibleOwnerTrades), [visibleOwnerTrades]);
  const portfolioOptions = useMemo(
    () =>
      Array.from(new Set([...savedPortfolios, ...activeOwnerTrades.map((trade) => trade.portfolioTag).filter(Boolean)])).sort((a, b) =>
        a.localeCompare(b)
      ),
    [activeOwnerTrades, savedPortfolios]
  );
  const activeColumnScope = activePortfolio || "__all__";
  const activeBrandenColumns = useMemo(
    () => normalizeBrandenColumns(columnPreferences[activeColumnScope] || columnPreferences.__all__ || defaultBrandenColumns),
    [activeColumnScope, columnPreferences]
  );
  const visibleBrandenColumns = useMemo(
    () => activeBrandenColumns.filter((column) => column.visible),
    [activeBrandenColumns]
  );
  const checklistGradeOptions = useMemo(() => {
    const labels = [
      ...setupTemplates.flatMap((template) => template.gradeBands?.map((band) => band.label) || []),
      ...defaultChecklistGradeBands.map((band) => band.label),
      ...visibleOwnerTrades.map((trade) => trade.manualGrade || "").filter(Boolean),
      ...(visibleOwnerTrades.some((trade) => !trade.manualGrade && !checklistScore(resolvedTradeChecklistItems(trade, setupTemplates), tradeGradeBands(trade, setupTemplates)).total)
        ? ["Unscored"]
        : [])
    ];

    return Array.from(new Set(labels));
  }, [visibleOwnerTrades, setupTemplates]);
  const filteredTrades = useMemo(
    () =>
      visibleOwnerTrades.filter((trade) => {
        const symbolMatch = matchesSelectedFilter(tradeFilters.symbol, [trade.symbol]);
        const sideMatch = matchesSelectedFilter(tradeFilters.side, [trade.side]);
        const statusMatch = matchesSelectedFilter(tradeFilters.status, [normalizedTradeStatus(trade)]);
        const setupMatch = matchesSelectedFilter(tradeFilters.setup, trade.setupTags);
        const mistakeMatch = matchesSelectedFilter(tradeFilters.mistake, trade.mistakeTags);
        const customMatch = matchesCustomTagFilter(tradeFilters.custom, trade.customTags, customOptions);
        const gradeMatch = matchesSelectedFilter(tradeFilters.grade, [effectiveTradeGrade(trade, setupTemplates)]);
        const reviewMatch = matchesSelectedFilter(tradeFilters.review, [tradeNeedsReview(trade, setupTemplates) ? "Needs Review" : "Complete"]);
        const dateMatch = tradeHasRangeActivity(trade, tradeFilters.startDate, tradeFilters.endDate);
        const portfolioMatch = !activePortfolio || trade.portfolioTag === activePortfolio;

        return (
          symbolMatch &&
          sideMatch &&
          statusMatch &&
          setupMatch &&
          mistakeMatch &&
          customMatch &&
          gradeMatch &&
          reviewMatch &&
          portfolioMatch &&
          dateMatch
        );
      }).map((trade) => {
        if (!tradeFilters.startDate && !tradeFilters.endDate) {
          return trade;
        }

        const displayPnl = tradeDisplayPnlInRange(trade, tradeFilters.startDate, tradeFilters.endDate);
        const displayExitDate = tradeDisplayExitDateInRange(trade, tradeFilters.startDate, tradeFilters.endDate);

        return {
          ...trade,
          pnl: displayPnl,
          exitDate: trade.status === "OPEN" ? displayExitDate : displayExitDate || trade.exitDate,
          rMultiple: trade.risk ? displayPnl / trade.risk : 0,
          status: tradeStatus(displayPnl, trade.status !== "OPEN", trade.risk ? displayPnl / trade.risk : 0)
        };
      }),
    [visibleOwnerTrades, setupTemplates, tradeFilters, activePortfolio]
  );
  const selectedTrade = useMemo(
    () => trades.find((trade) => trade.id === selectedTradeId) || null,
    [selectedTradeId, trades]
  );
  const selectedTradeNavigation = useMemo(() => {
    const existingTradeIds = new Set(trades.map((trade) => trade.id));
    const navigationIds = detailNavigationIds.filter((id) => existingTradeIds.has(id));
    const navigationTrades = navigationIds.length
      ? navigationIds.map((id) => trades.find((trade) => trade.id === id)).filter((trade): trade is TradeLogEntry => Boolean(trade))
      : filteredTrades;

    if (!selectedTradeId) {
      return { index: -1, previousTradeId: "", nextTradeId: "", total: navigationTrades.length };
    }

    const index = navigationTrades.findIndex((trade) => trade.id === selectedTradeId);
    if (index === -1) {
      return { index, previousTradeId: "", nextTradeId: "", total: navigationTrades.length };
    }

    return {
      index,
      previousTradeId: index > 0 ? navigationTrades[index - 1].id : "",
      nextTradeId: index < navigationTrades.length - 1 ? navigationTrades[index + 1].id : "",
      total: navigationTrades.length
    };
  }, [detailNavigationIds, filteredTrades, selectedTradeId, trades]);
  const selectedTradeExecutions = useMemo(
    () => {
      if (!selectedTrade) {
        return [];
      }

      if (selectedTrade.executions?.length) {
        return selectedTrade.executions
          .map((execution) => ({
            id: execution.id,
            type: execution.type,
            date: execution.date,
            price: execution.price,
            shares: execution.shares,
            pnl: execution.pnl,
            importSource: execution.source || selectedTrade.importSource
          }))
          .sort((a, b) => `${a.date} ${a.id}`.localeCompare(`${b.date} ${b.id}`));
      }

      return [];
    },
    [selectedTrade]
  );
  const filteredTradeIds = useMemo(() => filteredTrades.map((trade) => trade.id), [filteredTrades]);
  const selectedFilteredTradeIds = useMemo(
    () => selectedTradeIds.filter((id) => filteredTradeIds.includes(id)),
    [filteredTradeIds, selectedTradeIds]
  );
  const selectedFilteredTrades = useMemo(
    () => filteredTrades.filter((trade) => selectedTradeIds.includes(trade.id)),
    [filteredTrades, selectedTradeIds]
  );
  const isAllFilteredTradesSelected = Boolean(filteredTradeIds.length) && selectedFilteredTradeIds.length === filteredTradeIds.length;
  const tradePnlChartData = useMemo(
    () =>
      [...filteredTrades]
        .filter(countsAsSettledTrade)
        .sort((a, b) => tradePnlDate(a).localeCompare(tradePnlDate(b)))
        .reduce<
          {
            label: string;
            symbol: string;
            pnl: number;
            cumulativePnl: number;
            cumulativePnlPositive: number;
            cumulativePnlNegative: number;
          }[]
        >((items, trade) => {
          const last = items[items.length - 1]?.cumulativePnl || 0;
          const cumulativePnl = last + trade.pnl;
          items.push({
            label: tradePnlDate(trade),
            symbol: trade.symbol,
            pnl: trade.pnl,
            cumulativePnl,
            cumulativePnlPositive: Math.max(cumulativePnl, 0),
            cumulativePnlNegative: Math.min(cumulativePnl, 0)
          });
          return items;
        }, []),
    [filteredTrades]
  );
  const tradeDatePerformanceData = useMemo(() => {
    const closed = filteredTrades
      .filter(countsAsSettledTrade)
      .sort((a, b) => (a.exitDate || a.entryDate).localeCompare(b.exitDate || b.entryDate));
    const grouped = closed.reduce<
      Record<string, { date: string; trades: number; wins: number; totalR: number; winningRTotal: number; winningTrades: number }>
    >((groups, trade) => {
      const date = trade.exitDate || trade.entryDate;
      groups[date] = groups[date] || {
        date,
        trades: 0,
        wins: 0,
        totalR: 0,
        winningRTotal: 0,
        winningTrades: 0
      };
      groups[date].trades += 1;
      groups[date].wins += trade.pnl > 0 ? 1 : 0;
      groups[date].totalR += trade.rMultiple;

      if (trade.rMultiple > 0) {
        groups[date].winningRTotal += trade.rMultiple;
        groups[date].winningTrades += 1;
      }

      return groups;
    }, {});
    let cumulativeR = 0;

    return Object.values(grouped)
      .sort((a, b) => a.date.localeCompare(b.date))
      .map((day) => {
        cumulativeR += day.totalR;

        return {
          label: day.date,
          trades: day.trades,
          winRate: day.trades ? (day.wins / day.trades) * 100 : 0,
          totalR: day.totalR,
          cumulativeR,
          avgWinnerR: day.winningTrades ? day.winningRTotal / day.winningTrades : 0
        };
      });
  }, [filteredTrades]);
  const rDistributionData = useMemo(() => {
    const buckets = [
      { bucket: "< -2R", min: Number.NEGATIVE_INFINITY, max: -2 },
      { bucket: "-2 to -1R", min: -2, max: -1 },
      { bucket: "-1 to 0R", min: -1, max: 0 },
      { bucket: "0 to 1R", min: 0, max: 1 },
      { bucket: "1 to 2R", min: 1, max: 2 },
      { bucket: "> 2R", min: 2, max: Number.POSITIVE_INFINITY }
    ];

    return buckets.map((bucket) => ({
      bucket: bucket.bucket,
      trades: filteredTrades.filter((trade) => trade.rMultiple >= bucket.min && trade.rMultiple < bucket.max).length
    }));
  }, [filteredTrades]);
  const tradeScoreData = useMemo(() => {
    const closed = filteredTrades.filter(countsAsSettledTrade);
    const wins = closed.filter((trade) => trade.pnl > 0);
    const losses = closed.filter((trade) => trade.pnl < 0);
    const grossWin = wins.reduce((total, trade) => total + trade.pnl, 0);
    const grossLoss = Math.abs(losses.reduce((total, trade) => total + trade.pnl, 0));
    const avgWin = wins.length ? grossWin / wins.length : 0;
    const avgLoss = losses.length ? grossLoss / losses.length : 0;
    const profitFactor = grossLoss ? grossWin / grossLoss : grossWin ? 4 : 0;
    const winRateScore = closed.length ? (wins.length / closed.length) * 100 : 0;
    const avgWinLossScore = avgLoss ? clamp((avgWin / avgLoss) * 50) : avgWin ? 100 : 0;
    const rValues = closed.map((trade) => trade.rMultiple);
    const consistencyScore = rValues.length > 1 ? clamp(100 - standardDeviation(rValues) * 28) : rValues.length ? 70 : 0;
    const data = [
      { metric: "Win %", score: clamp(winRateScore) },
      { metric: "Profit factor", score: clamp(profitFactor * 25) },
      { metric: "Avg win/loss", score: avgWinLossScore },
      { metric: "Consistency", score: consistencyScore }
    ];

    return {
      data,
      totalScore: average(data.map((item) => item.score))
    };
  }, [filteredTrades]);
  const gradePerformanceData = useMemo(
    () =>
      checklistGradeOptions.map((grade) => {
        const gradeTrades = filteredTrades.filter((trade) => effectiveTradeGrade(trade, setupTemplates) === grade);

        return {
          grade,
          trades: gradeTrades.length,
          avgR: gradeTrades.length ? average(gradeTrades.map((trade) => trade.rMultiple)) : 0,
          netPnl: gradeTrades.reduce((total, trade) => total + trade.pnl, 0)
        };
      }),
    [filteredTrades, setupTemplates, checklistGradeOptions]
  );
  const marketCycleChartData = useMemo(
    () =>
      [...marketCycleEntries]
        .sort((a, b) => a.date.localeCompare(b.date))
        .map((entry) => ({
          label: entry.date,
          trendDay: entry.trendDay,
          phase: entry.phase,
          notes: entry.notes,
          ...Object.fromEntries(
            marketCyclePhaseOptions.map((phase) => [marketCyclePhaseKey(phase), entry.phase === phase ? entry.trendDay : null])
          )
        })),
    [marketCycleEntries]
  );
  const setupScoreData = useMemo(() => {
    const groups = filteredTrades.reduce<Record<string, { earned: number; total: number; trades: number }>>((items, trade) => {
      const setup = trade.setupTags[0] || "No setup";
      const score = checklistScore(resolvedTradeChecklistItems(trade, setupTemplates), tradeGradeBands(trade, setupTemplates));

      if (!score.total) {
        return items;
      }

      items[setup] = items[setup] || { earned: 0, total: 0, trades: 0 };
      items[setup].earned += score.earned;
      items[setup].total += score.total;
      items[setup].trades += 1;
      return items;
    }, {});

    return Object.entries(groups)
      .map(([setup, value]) => ({
        setup,
        ...value,
        averagePoints: value.trades ? value.earned / value.trades : 0
      }))
      .sort((a, b) => b.averagePoints - a.averagePoints);
  }, [filteredTrades, setupTemplates]);
  const closedTrades = useMemo(() => filteredTrades.filter(countsAsSettledTrade), [filteredTrades]);
  const tradeSummary = useMemo(() => {
    const wins = closedTrades.filter((trade) => normalizedTradeStatus(trade) === "WIN");
    const losses = closedTrades.filter((trade) => normalizedTradeStatus(trade) === "LOSS");
    const orderedClosedTrades = [...closedTrades].sort((a, b) =>
      (a.exitDate || a.entryDate).localeCompare(b.exitDate || b.entryDate)
    );
    const netPnl = closedTrades.reduce((total, trade) => total + trade.pnl, 0);
    const grossWin = wins.reduce((total, trade) => total + trade.pnl, 0);
    const grossLoss = Math.abs(losses.reduce((total, trade) => total + trade.pnl, 0));
    const winRate = closedTrades.length ? (wins.length / closedTrades.length) * 100 : 0;
    const expectancy = closedTrades.length ? average(closedTrades.map((trade) => trade.rMultiple)) : 0;
    const avgWinR = wins.length ? average(wins.map((trade) => trade.rMultiple)) : 0;
    const avgLossR = losses.length ? average(losses.map((trade) => trade.rMultiple)) : 0;
    const swingTrades = closedTrades.filter((trade) => trade.daysInTrade > 0);
    const scoredTrades = filteredTrades.filter((trade) => resolvedTradeChecklistItems(trade, setupTemplates).length);
    const needsReview = filteredTrades.filter((trade) => tradeNeedsReview(trade, setupTemplates)).length;
    const avgChecklistScore = scoredTrades.length
      ? average(scoredTrades.map((trade) => checklistScore(resolvedTradeChecklistItems(trade, setupTemplates), tradeGradeBands(trade, setupTemplates)).earned))
      : 0;

    return {
      netPnl,
      profitFactorLabel: grossLoss ? (grossWin / grossLoss).toFixed(2) : grossWin ? "∞" : "-",
      totalTrades: filteredTrades.length,
      totalR: closedTrades.reduce((total, trade) => total + trade.rMultiple, 0),
      winRate,
      expectancy,
      avgWinR,
      avgLossR,
      avgWin: wins.length ? grossWin / wins.length : 0,
      avgLoss: losses.length ? grossLoss / losses.length : 0,
      avgRisk: filteredTrades.length ? average(filteredTrades.map((trade) => trade.risk)) : 0,
      avgChecklistScore,
      needsReview,
      avgTradeLength: closedTrades.length ? average(closedTrades.map((trade) => trade.daysInTrade)) : 0,
      avgSwingLength: swingTrades.length ? average(swingTrades.map((trade) => trade.daysInTrade)) : 0,
      longestWinStreak: longestTradeStreak(orderedClosedTrades, "WIN"),
      longestLossStreak: longestTradeStreak(orderedClosedTrades, "LOSS")
    };
  }, [closedTrades, filteredTrades, setupTemplates]);

  useEffect(() => {
    async function boot() {
      const sessionResponse = await fetch("/api/session");
      const session = await sessionResponse.json();
      setUser(session.user);

      if (session.user) {
        await loadReports();
        await loadTrades();
        await loadSetupTemplates();
        await loadMarketCycleEntries();
        await loadBrandenPortfolios();
        await loadBrandenColumnPreferences();
      }

      setIsLoading(false);
    }

    boot();
  }, []);

  useEffect(() => {
    const query = window.matchMedia("(max-width: 430px)");
    const syncViewport = () => setIsPhoneViewport(query.matches);

    syncViewport();
    query.addEventListener("change", syncViewport);

    return () => query.removeEventListener("change", syncViewport);
  }, []);

  useEffect(() => {
    if (selectedTrade) {
      setEditTradeForm(tradeToForm(selectedTrade, setupTemplates));
    }
  }, [selectedTrade, setupTemplates]);

  useEffect(() => {
    const existing = marketCycleEntries.find((entry) => entry.date === marketCycleForm.date);

    if (!existing) {
      return;
    }

    setMarketCycleForm((current) =>
      current.date === existing.date && current.trendDay === existing.trendDay && current.notes === existing.notes
        && current.phase === existing.phase
        ? current
        : {
            date: existing.date,
            trendDay: existing.trendDay,
            phase: existing.phase,
            notes: existing.notes
          }
    );
  }, [marketCycleEntries, marketCycleForm.date]);

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    setIsBrandenJournalMode(params.get("journal") === "branden" || window.location.pathname === "/journal/branden/dashboard");
  }, []);

  useEffect(() => {
    if (user?.journalOwnerId === "branden") {
      setActiveTradeOwner("branden");
    } else if (user?.id === "branden" || user?.id === "cam") {
      setActiveTradeOwner(user.id);
    }
  }, [user]);

  useEffect(() => {
    if (isBrandenJournalMode) {
      setActiveTradeOwner("branden");
      setActiveTab("trades");
    }
  }, [isBrandenJournalMode]);

  useEffect(() => {
    if (!isBrandenJournalMode || !trades.length) {
      return;
    }

    const params = new URLSearchParams(window.location.search);
    const tradeId = params.get("tradeId") || "";
    const tradeNavKey = params.get("tradeNavKey") || "";

    if (tradeId && trades.some((trade) => trade.id === tradeId)) {
      setSelectedTradeId(tradeId);
    }

    if (tradeNavKey) {
      try {
        const savedIds = JSON.parse(window.sessionStorage.getItem(tradeNavKey) || "[]");
        setDetailNavigationIds(Array.isArray(savedIds) ? savedIds.map(String).filter(Boolean) : []);
      } catch {
        setDetailNavigationIds([]);
      }
    } else {
      setDetailNavigationIds([]);
    }
  }, [isBrandenJournalMode, trades]);

  useEffect(() => {
    if (!isBrandenJournalMode || !selectedTradeId) {
      return;
    }

    const url = new URL(window.location.href);

    if (url.searchParams.get("tradeId") !== selectedTradeId) {
      url.searchParams.set("tradeId", selectedTradeId);
      window.history.replaceState(null, "", `${url.pathname}${url.search}${url.hash}`);
    }
  }, [isBrandenJournalMode, selectedTradeId]);

  useEffect(() => {
    setSelectedTradeId("");
    setSelectedTradeIds([]);
    setTradeFilters(emptyTradeFilters);
    setActivePortfolio("");
  }, [activeTradeOwner]);

  const primaryChartHeight = isPhoneViewport ? 260 : 380;
  const secondaryChartHeight = isPhoneViewport ? 230 : 320;
  const responsiveChartMargin = isPhoneViewport ? { top: 12, right: 4, bottom: 0, left: -28 } : chartMargin;
  const xAxisInterval = isPhoneViewport ? "preserveStartEnd" : 0;

  async function loadReports() {
    const response = await fetch("/api/reports");
    const text = await response.text();
    const data = text ? JSON.parse(text) : {};

    if (!response.ok) {
      setReportsError(data.error || "Could not load reports.");
      setReports([]);
      return;
    }

    setReportsError("");
    setReports(data.reports || []);
  }

  async function loadTrades() {
    const response = await fetch("/api/trades", { cache: "no-store" });
    const text = await response.text();
    const data = text ? JSON.parse(text) : {};

    if (!response.ok) {
      setTradesError(data.error || "Could not load trades.");
      setTrades([]);
      return [] as TradeLogEntry[];
    }

    setTradesError("");
    setTrades(data.trades || []);
    return (data.trades || []) as TradeLogEntry[];
  }

  async function loadSetupTemplates() {
    const response = await fetch("/api/settings/setup-checklists");
    const text = await response.text();
    const data = text ? JSON.parse(text) : {};

    if (!response.ok) {
      return;
    }

    const nextTemplates = data.setupChecklists || [];
    setSetupTemplates(nextTemplates);
    setSetupTemplateDrafts(nextTemplates);
  }

  async function loadMarketCycleEntries() {
    const response = await fetch("/api/settings/market-cycle");
    const text = await response.text();
    const data = text ? JSON.parse(text) : {};

    if (!response.ok) {
      return;
    }

    setMarketCycleEntries(data.entries || []);
  }

  async function loadBrandenPortfolios() {
    const response = await fetch("/api/settings/branden-portfolios");
    const text = await response.text();
    const data = text ? JSON.parse(text) : {};

    if (!response.ok) {
      return;
    }

    const portfolios = Array.isArray(data.portfolios) ? data.portfolios : [];
    const nextDefault = String(data.defaultPortfolio || "");

    setSavedPortfolios(portfolios);
    setDefaultPortfolio(nextDefault);

    if (nextDefault) {
      setActivePortfolio(nextDefault);
    }
  }

  async function loadBrandenColumnPreferences() {
    const response = await fetch("/api/settings/branden-columns");
    const text = await response.text();
    const data = text ? JSON.parse(text) : {};

    if (!response.ok) {
      return;
    }

    const nextPreferences = data.preferences && typeof data.preferences === "object" ? data.preferences : {};
    setColumnPreferences(nextPreferences);
  }

  async function handleLogin(event: FormEvent) {
    event.preventDefault();
    setStatus("");

    const response = await fetch("/api/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(login)
    });

    const data = await response.json();

    if (!response.ok) {
      setStatus(data.error || "Login failed.");
      return;
    }

    setUser(data.user);
    await loadReports();
    await loadTrades();
    await loadSetupTemplates();
    await loadBrandenPortfolios();
    await loadBrandenColumnPreferences();
  }

  async function handleLogout() {
    await fetch("/api/logout", { method: "POST" });
    setUser(null);
    setReports([]);
    setSavedPortfolios([]);
    setColumnPreferences({});
    setTrades([]);
  }

  function updateField(key: keyof FormState, value: string) {
    setForm((current) => ({
      ...current,
      [key]: value
    }));
  }

  function editReport(report: MonthlyReport) {
    setForm({
      month: report.month,
      accountSize: report.accountSize,
      totalReturn: report.totalReturn,
      percentReturn: report.percentReturn,
      netPnl: report.netPnl,
      totalPayouts: report.totalPayouts,
      totalTrades: report.totalTrades,
      winRate: report.winRate,
      avgR: report.avgR,
      totalR: report.totalR,
      avgWinR: report.avgWinR,
      avgLossR: report.avgLossR,
      avgWin: report.avgWin,
      avgLoss: report.avgLoss,
      avgRisk: report.avgRisk,
      currentRiskPercent: report.currentRiskPercent,
      expectedValueR: report.expectedValueR,
      sharpeRatio: report.sharpeRatio,
      avgTradeLength: report.avgTradeLength,
      avgSwingLength: report.avgSwingLength,
      longestWinStreak: report.longestWinStreak,
      longestLossStreak: report.longestLossStreak,
      notes: report.notes
    });
    setActiveTab("entry");
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  async function saveReport(event: FormEvent) {
    event.preventDefault();
    setStatus("Saving report...");

    const response = await fetch("/api/reports", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(calculateDerivedMetrics(form))
    });

    const data = await response.json();

    if (!response.ok) {
      setStatus(data.error || "Could not save report.");
      return;
    }

    setStatus(`${monthLabel(form.month)} saved.`);
    setForm({ ...emptyForm, month: form.month });
    await loadReports();
    setActiveTab("dashboard");
  }

  async function removeReport(report: MonthlyReport) {
    await fetch(`/api/reports/${report.id}`, { method: "DELETE" });
    await loadReports();
  }

  function updateTradeField(key: keyof TradeFormState, value: TradeFormState[keyof TradeFormState]) {
    setTradeForm((current) => ({
      ...current,
      [key]: value
    }));
  }

  function updateEditTradeField(key: keyof TradeFormState, value: TradeFormState[keyof TradeFormState]) {
    setEditTradeForm((current) => ({
      ...current,
      [key]: value
    }));
  }

  function updateTradeReviewField(target: "create" | "edit", key: keyof TradeReviewSections, value: string) {
    const setter = target === "create" ? setTradeForm : setEditTradeForm;
    setter((current) => ({
      ...current,
      reviewSections: { ...current.reviewSections, [key]: value }
    }));
  }

  function updateTradeSetupTags(target: "create" | "edit", value: string) {
    const setter = target === "create" ? setTradeForm : setEditTradeForm;

    setter((current) => {
      const template = setupTemplateFor(primarySetupName(value), setupTemplates);
      const checklistTradeContext = { customTags: splitTags(current.customTags) };

      return {
        ...current,
        setupTags: value,
        checklistItems: template ? checklistFromSetupTemplate(template, current.checklistItems, checklistTradeContext) : current.checklistItems
      };
    });
  }

  function updateTradeMistakeTags(target: "create" | "edit", tags: string[]) {
    const setter = target === "create" ? setTradeForm : setEditTradeForm;
    setter((current) => ({
      ...current,
      mistakeTags: sortedUnique(tags).join(", ")
    }));
  }

  function addTradeMistakeTag(target: "create" | "edit", value: string) {
    const tag = value.trim();
    if (!tag) {
      return;
    }

    const currentTags = splitTags(target === "create" ? tradeForm.mistakeTags : editTradeForm.mistakeTags);
    updateTradeMistakeTags(target, [...currentTags, tag]);
    setNewMistakeDraft((current) => ({ ...current, [target]: "" }));
  }

  function removeTradeMistakeTag(target: "create" | "edit", value: string) {
    const currentTags = splitTags(target === "create" ? tradeForm.mistakeTags : editTradeForm.mistakeTags);
    updateTradeMistakeTags(
      target,
      currentTags.filter((tag) => tag !== value)
    );
  }

  function updateTradeCustomTags(target: "create" | "edit", tags: string[]) {
    const setter = target === "create" ? setTradeForm : setEditTradeForm;
    setter((current) => ({
      ...current,
      customTags: sortedUnique(tags).join(", ")
    }));
  }

  function addTradeCustomTag(target: "create" | "edit", value: string) {
    const tag = value.trim();
    if (!tag) {
      return;
    }

    const currentTags = splitTags(target === "create" ? tradeForm.customTags : editTradeForm.customTags);
    updateTradeCustomTags(target, [...currentTags, tag]);
    setNewCustomTagDraft((current) => ({ ...current, [target]: "" }));
  }

  function removeTradeCustomTag(target: "create" | "edit", value: string) {
    const currentTags = splitTags(target === "create" ? tradeForm.customTags : editTradeForm.customTags);
    updateTradeCustomTags(
      target,
      currentTags.filter((tag) => tag !== value)
    );
  }

  function addChecklistItem(target: "create" | "edit") {
    const setter = target === "create" ? setTradeForm : setEditTradeForm;
    setter((current) => ({
      ...current,
      checklistItems: [...current.checklistItems, newChecklistItem()]
    }));
  }

  function updateChecklistItem(target: "create" | "edit", id: string, updates: Partial<TradeChecklistItem>) {
    const setter = target === "create" ? setTradeForm : setEditTradeForm;
    setter((current) => ({
      ...current,
      checklistItems: current.checklistItems.map((item) => (item.id === id ? { ...item, ...updates } : item))
    }));
  }

  function removeChecklistItem(target: "create" | "edit", id: string) {
    const setter = target === "create" ? setTradeForm : setEditTradeForm;
    setter((current) => ({
      ...current,
      checklistItems: current.checklistItems.filter((item) => item.id !== id)
    }));
  }

  function addSetupTemplate() {
    setStatus("");
    setSetupTemplateDrafts((current) => [...current, newSetupTemplate()]);
  }

  function addPresetSetupTemplate(preset: "OTC" | "CANSLIM") {
    setStatus("");
    setSetupTemplateDrafts((current) => [
      ...current,
      preset === "OTC" ? otcPresetTemplate() : canslimPresetTemplate()
    ]);
  }

  function updateSetupTemplate(id: string, updates: Partial<SetupChecklistTemplate>) {
    setSetupTemplateDrafts((current) =>
      current.map((template) =>
        template.id === id
          ? { ...template, ...updates, criteria: (updates.groups || template.groups || []).flatMap((group) => group.criteria) }
          : template
      )
    );
  }

  function removeSetupTemplate(id: string) {
    setSetupTemplateDrafts((current) => current.filter((template) => template.id !== id));
  }

  function addSetupTemplateGradeBand(templateId: string) {
    setSetupTemplateDrafts((current) =>
      current.map((template) =>
        template.id === templateId
          ? {
              ...template,
              gradeBands: [
                ...(template.gradeBands || []),
                {
                  id: `grade-${Date.now()}-${Math.random().toString(16).slice(2)}`,
                  label: "",
                  minScore: 0,
                  maxScore: null
                }
              ]
            }
          : template
      )
    );
  }

  function updateSetupTemplateGradeBand(templateId: string, bandId: string, updates: Partial<ChecklistGradeBand>) {
    setSetupTemplateDrafts((current) =>
      current.map((template) =>
        template.id === templateId
          ? {
              ...template,
              gradeBands: (template.gradeBands || defaultChecklistGradeBands).map((band) =>
                band.id === bandId ? { ...band, ...updates } : band
              )
            }
          : template
      )
    );
  }

  function removeSetupTemplateGradeBand(templateId: string, bandId: string) {
    setSetupTemplateDrafts((current) =>
      current.map((template) =>
        template.id === templateId
          ? { ...template, gradeBands: (template.gradeBands || []).filter((band) => band.id !== bandId) }
          : template
      )
    );
  }

  function resetSetupTemplateGradeBands(templateId: string) {
    setSetupTemplateDrafts((current) =>
      current.map((template) =>
        template.id === templateId ? { ...template, gradeBands: defaultChecklistGradeBands } : template
      )
    );
  }

  function addSetupTemplateGroup(templateId: string, name = "New Group") {
    setSetupTemplateDrafts((current) =>
      current.map((template) => {
        if (template.id !== templateId) {
          return template;
        }

        const nextGroup = newSetupTemplateGroup(name);
        const groups = [...(template.groups || []), nextGroup];
        return { ...template, groups, criteria: groups.flatMap((group) => group.criteria) };
      })
    );
  }

  function updateSetupTemplateGroup(templateId: string, groupId: string, updates: Partial<SetupChecklistGroup>) {
    setSetupTemplateDrafts((current) =>
      current.map((template) =>
        template.id === templateId
          ? {
              ...template,
              groups: (template.groups || []).map((group) => (group.id === groupId ? { ...group, ...updates } : group)),
              criteria: (template.groups || []).map((group) => (group.id === groupId ? { ...group, ...updates } : group)).flatMap((group) => group.criteria)
            }
          : template
      )
    );
  }

  function removeSetupTemplateGroup(templateId: string, groupId: string) {
    setSetupTemplateDrafts((current) =>
      current.map((template) =>
        template.id === templateId
          ? {
              ...template,
              groups: (template.groups || []).filter((group) => group.id !== groupId),
              criteria: (template.groups || []).filter((group) => group.id !== groupId).flatMap((group) => group.criteria)
            }
          : template
      )
    );
  }

  function addSetupTemplateCriteria(templateId: string, groupId: string, inputType: ChecklistInputType = "boolean") {
    setSetupTemplateDrafts((current) =>
      current.map((template) => {
        if (template.id !== templateId) {
          return template;
        }

        const groups = (template.groups || []).map((group) =>
          group.id === groupId ? { ...group, criteria: [...group.criteria, newSetupTemplateCriterion(inputType)] } : group
        );

        return { ...template, groups, criteria: groups.flatMap((group) => group.criteria) };
      })
    );
  }

  function updateSetupTemplateCriteria(
    templateId: string,
    groupId: string,
    criteriaId: string,
    updates: Partial<SetupTemplateCriterion>
  ) {
    setSetupTemplateDrafts((current) =>
      current.map((template) => {
        if (template.id !== templateId) {
          return template;
        }

        const groups = (template.groups || []).map((group) =>
          group.id === groupId
            ? {
                ...group,
                criteria: group.criteria.map((item) =>
                  item.id === criteriaId ? { ...item, ...updates } : item
                )
              }
            : group
        );

        return { ...template, groups, criteria: groups.flatMap((group) => group.criteria) };
      })
    );
  }

  function removeSetupTemplateCriteria(templateId: string, groupId: string, criteriaId: string) {
    setSetupTemplateDrafts((current) =>
      current.map((template) => {
        if (template.id !== templateId) {
          return template;
        }

        const groups = (template.groups || []).map((group) =>
          group.id === groupId
            ? { ...group, criteria: group.criteria.filter((item) => item.id !== criteriaId) }
            : group
        );

        return { ...template, groups, criteria: groups.flatMap((group) => group.criteria) };
      })
    );
  }

  function dropSetupTemplateCriterion(
    event: DragEvent<HTMLDivElement>,
    targetTemplateId: string,
    targetGroupId: string,
    targetCriteriaId: string
  ) {
    event.preventDefault();
    const source = draggedSetupCriterion;
    setDraggedSetupCriterion(null);
    setDragOverSetupCriterionId("");

    if (!source || source.templateId !== targetTemplateId || source.criteriaId === targetCriteriaId) {
      return;
    }

    setSetupTemplateDrafts((current) =>
      current.map((template) => {
        if (template.id !== targetTemplateId) {
          return template;
        }

        const sourceGroup = (template.groups || []).find((group) => group.id === source.groupId);
        const movedCriterion = sourceGroup?.criteria.find((criterion) => criterion.id === source.criteriaId);
        const targetGroup = (template.groups || []).find((group) => group.id === targetGroupId);
        const targetIndex = targetGroup?.criteria.findIndex((criterion) => criterion.id === targetCriteriaId) ?? -1;

        if (!movedCriterion || targetIndex < 0) {
          return template;
        }

        const groupsWithoutSource = (template.groups || []).map((group) =>
          group.id === source.groupId
            ? { ...group, criteria: group.criteria.filter((criterion) => criterion.id !== source.criteriaId) }
            : group
        );
        const groups = groupsWithoutSource.map((group) => {
          if (group.id !== targetGroupId) {
            return group;
          }

          const criteria = [...group.criteria];
          criteria.splice(targetIndex, 0, movedCriterion);
          return { ...group, criteria };
        });

        return { ...template, groups, criteria: groups.flatMap((group) => group.criteria) };
      })
    );
  }

  async function saveSetupTemplates() {
    const invalidTemplate = setupTemplateDrafts.find((template) => !template.setupName.trim());
    const invalidCriteriaTemplate = setupTemplateDrafts.find(
      (template) =>
        template.setupName.trim() &&
        !(template.groups || []).some((group) =>
          group.criteria.some((criteria) => criteria.criteria.trim() && numberValue(criteria.points) > 0)
        )
    );
    const invalidGroupTemplate = setupTemplateDrafts.find(
      (template) => template.setupName.trim() && !(template.groups || []).some((group) => group.name.trim())
    );
    const invalidGradeTemplate = setupTemplateDrafts.find(
      (template) =>
        template.setupName.trim() &&
        !(template.gradeBands || []).some((band) => band.label.trim() && Number.isFinite(Number(band.minScore)))
    );

    if (!setupTemplateDrafts.length) {
      setStatus("Add at least one setup before saving.");
      return;
    }

    if (invalidTemplate) {
      setStatus("Every setup needs a setup name before saving.");
      return;
    }

    if (invalidCriteriaTemplate) {
      setStatus(`${invalidCriteriaTemplate.setupName} needs at least one valid criteria row with points.`);
      return;
    }

    if (invalidGroupTemplate) {
      setStatus(`${invalidGroupTemplate.setupName} needs at least one named criteria group.`);
      return;
    }

    if (invalidGradeTemplate) {
      setStatus(`${invalidGradeTemplate.setupName} needs at least one valid grade rule.`);
      return;
    }

    setStatus("Saving setup checklists...");
    const response = await fetch("/api/settings/setup-checklists", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ setupChecklists: setupTemplateDrafts })
    });
    const data = await response.json();

    if (!response.ok) {
      setStatus(data.error || "Could not save setup checklists.");
      return;
    }

    setSetupTemplates(data.setupChecklists);
    setSetupTemplateDrafts(data.setupChecklists);
    setStatus("Setup checklists saved.");
  }

  function updateHeaderFilter(key: TradeFilterKey, nextValues: string[]) {
    setTradeFilters((current) => ({
      ...current,
      [key]: nextValues
    }));
  }

  function toggleHeaderFilterValue(key: TradeFilterKey, value: string, options: string[]) {
    const selected = tradeFilters[key];
    const effectiveSelected = selected.includes(noFilterSelection) ? [] : selected.length ? selected : options;

    if (effectiveSelected.length === options.length) {
      updateHeaderFilter(
        key,
        options.filter((option) => option !== value)
      );
      return;
    }

    const nextValues = effectiveSelected.includes(value)
      ? effectiveSelected.filter((item) => item !== value)
      : [...effectiveSelected, value];

    updateHeaderFilter(key, nextValues.length === options.length ? [] : nextValues);
  }

  function renderHeaderFilter(label: string, key: TradeFilterKey, options: string[]) {
    const selected = tradeFilters[key];
    const isNoneSelected = selected.includes(noFilterSelection);
    const isAllSelected = !isNoneSelected && (!selected.length || selected.length === options.length);
    const effectiveSelected = isNoneSelected ? [] : selected.length ? selected : options;

    return (
      <details className={selected.length ? "table-filter active" : "table-filter"}>
        <summary>
          <span>{label}</span>
          <small>{filterLabel(selected, options.length)}</small>
        </summary>
        <div className="table-filter-menu">
          <label>
            <input
              type="checkbox"
              checked={isAllSelected}
              onChange={() => updateHeaderFilter(key, isAllSelected ? [noFilterSelection] : [])}
            />
            Select All
          </label>
          {options.map((option) => (
            <label key={option}>
              <input
                type="checkbox"
                checked={effectiveSelected.includes(option)}
                onChange={() => toggleHeaderFilterValue(key, option, options)}
              />
              {option}
            </label>
          ))}
          {!options.length ? <p>No values</p> : null}
        </div>
      </details>
    );
  }

  function renderBrandenColumnHeader(key: BrandenTradeColumnKey) {
    switch (key) {
      case "status":
        return renderHeaderFilter("Status", "status", ["OPEN", "WIN", "LOSS", "BREAKEVEN"]);
      case "side":
        return renderHeaderFilter("Side", "side", ["LONG", "SHORT"]);
      case "symbol":
        return renderHeaderFilter("Symbol", "symbol", symbolOptions);
      case "setup":
        return renderHeaderFilter("Setup", "setup", setupOptions);
      case "portfolio":
        return "Portfolio";
      case "openDate":
        return "Open Date";
      case "entry":
        return "Entry";
      case "size":
        return "Size";
      case "closeDate":
        return "Close Date";
      case "exit":
        return "Exit";
      case "stop":
        return "Stop";
      case "commission":
        return "Commission";
      case "usedMargin":
        return "Used Margin";
      case "takeProfit":
        return "Take Profit";
      case "risk":
        return "Risk";
      case "cost":
        return "Cost";
      case "netReturn":
        return "Net Return";
      case "r":
        return "R";
      case "mistake":
        return renderHeaderFilter("Mistake", "mistake", mistakeOptions);
      case "custom":
        return renderHeaderFilter("Custom Tags", "custom", customOptions);
      case "grade":
        return renderHeaderFilter("Grade", "grade", checklistGradeOptions);
      case "review":
        return renderHeaderFilter("Review", "review", ["Needs Review", "Complete"]);
      default:
        return "";
    }
  }

  function tradeInlineFormWithUpdate(trade: TradeLogEntry, updates: Partial<TradeFormState>) {
    return {
      ...tradeToForm(trade, setupTemplates),
      ...updates
    };
  }

  function inlineUpdateShouldPreserveStatus(updates: Partial<TradeFormState>) {
    return !Object.keys(updates).some((key) =>
      ["status", "exitDate", "exitPrice", "pnl", "rMultiple", "risk", "avgEntry", "shares", "side"].includes(key)
    );
  }

  async function saveInlineTradeUpdate(trade: TradeLogEntry, updates: Partial<TradeFormState>) {
    if (!canEditActiveTradeLog || trade.userId !== user?.id) {
      return;
    }

    const nextForm = tradeInlineFormWithUpdate(trade, updates);

    if (updates.setupTags !== undefined) {
      const template = setupTemplateFor(primarySetupName(String(updates.setupTags || "")), setupTemplates);
      nextForm.checklistItems = template
        ? checklistFromSetupTemplate(template, [], { customTags: splitTags(nextForm.customTags) })
        : [];
      nextForm.manualGrade = template ? "" : nextForm.manualGrade;
    }

    const response = await fetch(`/api/trades/${trade.id}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(
        tradePayloadFromForm(nextForm, {
          preserveStatus: inlineUpdateShouldPreserveStatus(updates)
        })
      )
    });
    const data = await response.json();

    if (!response.ok) {
      setStatus(data.error || `Could not update ${trade.symbol}.`);
      return;
    }

    setStatus(`${trade.symbol} updated.`);
    await loadTrades();
  }

  function inlineNumberCell(trade: TradeLogEntry, key: keyof TradeFormState, value: number, options: { money?: boolean; step?: string } = {}) {
    if (!canEditActiveTradeLog || trade.userId !== user?.id) {
      return value ? (options.money ? money(value) : value.toString()) : "-";
    }

    return (
      <input
        className="trade-inline-input number"
        type="number"
        inputMode="decimal"
        step={options.step || "0.01"}
        defaultValue={value || ""}
        placeholder="-"
        onClick={(event) => event.stopPropagation()}
        onKeyDown={(event) => {
          if (event.key === "Enter") {
            event.currentTarget.blur();
          }
          if (event.key === "Escape") {
            event.currentTarget.value = value ? String(value) : "";
            event.currentTarget.blur();
          }
        }}
        onBlur={(event) => {
          const nextValue = event.currentTarget.value.trim();
          const parsed = nextValue === "" ? 0 : Number(nextValue);

          if (!Number.isFinite(parsed) || parsed === value) {
            return;
          }

          saveInlineTradeUpdate(trade, { [key]: parsed } as Partial<TradeFormState>);
        }}
      />
    );
  }

  function inlineTextCell(trade: TradeLogEntry, key: keyof TradeFormState, value: string, placeholder = "-") {
    if (!canEditActiveTradeLog || trade.userId !== user?.id) {
      return value || placeholder;
    }

    return (
      <input
        className="trade-inline-input"
        type="text"
        defaultValue={value}
        placeholder={placeholder}
        onClick={(event) => event.stopPropagation()}
        onKeyDown={(event) => {
          if (event.key === "Enter") {
            event.currentTarget.blur();
          }
          if (event.key === "Escape") {
            event.currentTarget.value = value;
            event.currentTarget.blur();
          }
        }}
        onBlur={(event) => {
          const nextValue = event.currentTarget.value.trim();

          if (nextValue === value) {
            return;
          }

          saveInlineTradeUpdate(trade, { [key]: nextValue } as Partial<TradeFormState>);
        }}
      />
    );
  }

  function inlineDateCell(trade: TradeLogEntry, key: "entryDate" | "exitDate", value: string) {
    if (!canEditActiveTradeLog || trade.userId !== user?.id) {
      return value || "-";
    }

    return (
      <input
        className="trade-inline-input date"
        type="text"
        defaultValue={value}
        placeholder="-"
        onClick={(event) => event.stopPropagation()}
        onKeyDown={(event) => {
          if (event.key === "Enter") {
            event.currentTarget.blur();
          }
          if (event.key === "Escape") {
            event.currentTarget.value = value;
            event.currentTarget.blur();
          }
        }}
        onBlur={(event) => {
          const nextValue = event.currentTarget.value;

          if (nextValue === value) {
            return;
          }

          saveInlineTradeUpdate(trade, { [key]: nextValue } as Partial<TradeFormState>);
        }}
      />
    );
  }

  function inlineSelectCell<T extends string>(trade: TradeLogEntry, key: keyof TradeFormState, value: T, options: T[], className = "") {
    if (!canEditActiveTradeLog || trade.userId !== user?.id) {
      return value;
    }

    return (
      <select
        className={["trade-inline-select", className].filter(Boolean).join(" ")}
        value={value}
        onClick={(event) => event.stopPropagation()}
        onChange={(event) => saveInlineTradeUpdate(trade, { [key]: event.currentTarget.value } as Partial<TradeFormState>)}
      >
        {options.map((option) => (
          <option key={option} value={option}>
            {option}
          </option>
        ))}
      </select>
    );
  }

  function setupAssignmentOptions(currentValue = "") {
    return Array.from(
      new Set([
        "",
        ...setupTemplates.map((template) => template.setupName.trim()).filter(Boolean),
        ...setupOptions,
        currentValue.trim()
      ].filter((value) => value !== undefined))
    );
  }

  function inlineSetupCell(trade: TradeLogEntry) {
    const currentSetup = trade.setupTags[0] || "";

    if (!canEditActiveTradeLog || trade.userId !== user?.id) {
      return currentSetup || "-";
    }

    return (
      <select
        className="trade-inline-select setup"
        value={currentSetup}
        onClick={(event) => event.stopPropagation()}
        onChange={(event) => saveInlineTradeUpdate(trade, { setupTags: event.currentTarget.value })}
      >
        {setupAssignmentOptions(currentSetup).map((option) => (
          <option key={option || "__none__"} value={option}>
            {option || "No setup"}
          </option>
        ))}
      </select>
    );
  }

  function renderMistakeTagField(target: "create" | "edit", currentValue: string, disabled = false) {
    const selectedTags = splitTags(currentValue);
    const availableTags = sortedUnique([...mistakeOptions, ...selectedTags]).filter((tag) => !selectedTags.includes(tag));
    const draft = newMistakeDraft[target];

    return (
      <div className="mistake-tag-field">
        <div className="mistake-tag-controls">
          <select
            value=""
            disabled={disabled || !availableTags.length}
            onChange={(event) => addTradeMistakeTag(target, event.target.value)}
          >
            <option value="">{availableTags.length ? "Select mistake" : "No saved mistakes"}</option>
            {availableTags.map((tag) => (
              <option key={tag} value={tag}>
                {tag}
              </option>
            ))}
          </select>
          <input
            value={draft}
            onChange={(event) => setNewMistakeDraft((current) => ({ ...current, [target]: event.target.value }))}
            onKeyDown={(event) => {
              if (event.key === "Enter") {
                event.preventDefault();
                addTradeMistakeTag(target, draft);
              }
            }}
            placeholder="Create new mistake"
            disabled={disabled}
          />
          <button className="trade-muted-button" type="button" onClick={() => addTradeMistakeTag(target, draft)} disabled={disabled || !draft.trim()}>
            Add
          </button>
        </div>
        <div className="mistake-tag-chips">
          {selectedTags.map((tag) => (
            <button key={tag} type="button" onClick={() => removeTradeMistakeTag(target, tag)} disabled={disabled}>
              {tag}
              <span aria-hidden="true">x</span>
            </button>
          ))}
          {!selectedTags.length ? <span>No mistake tagged</span> : null}
        </div>
      </div>
    );
  }

  function renderCustomTagField(target: "create" | "edit", currentValue: string, disabled = false) {
    const selectedTags = splitTags(currentValue);
    const availableTags = sortedUnique([...customOptions, ...selectedTags]).filter((tag) => !selectedTags.includes(tag));
    const draft = newCustomTagDraft[target];

    return (
      <div className="custom-tag-field">
        <div className="custom-tag-controls">
          <select
            value=""
            disabled={disabled || !availableTags.length}
            onChange={(event) => addTradeCustomTag(target, event.target.value)}
          >
            <option value="">{availableTags.length ? "Select tag" : "No saved tags"}</option>
            {availableTags.map((tag) => (
              <option key={tag} value={tag}>
                {tag}
              </option>
            ))}
          </select>
          <input
            value={draft}
            onChange={(event) => setNewCustomTagDraft((current) => ({ ...current, [target]: event.target.value }))}
            onKeyDown={(event) => {
              if (event.key === "Enter") {
                event.preventDefault();
                addTradeCustomTag(target, draft);
              }
            }}
            placeholder="Create tag"
            disabled={disabled}
          />
          <button className="trade-muted-button" type="button" onClick={() => addTradeCustomTag(target, draft)} disabled={disabled || !draft.trim()}>
            Add
          </button>
        </div>
        <div className="custom-tag-chips">
          {selectedTags.map((tag) => (
            <button key={tag} type="button" onClick={() => removeTradeCustomTag(target, tag)} disabled={disabled}>
              {tag}
              <span aria-hidden="true">x</span>
            </button>
          ))}
          {!selectedTags.length ? <span>No custom tags</span> : null}
        </div>
      </div>
    );
  }

  function brandenColumnLabel(key: BrandenTradeColumnKey) {
    switch (key) {
      case "status":
        return "Status";
      case "side":
        return "Side";
      case "symbol":
        return "Symbol";
      case "setup":
        return "Setup";
      case "portfolio":
        return "Portfolio";
      case "openDate":
        return "Open Date";
      case "entry":
        return "Entry";
      case "size":
        return "Size";
      case "closeDate":
        return "Close Date";
      case "exit":
        return "Exit";
      case "stop":
        return "Stop";
      case "commission":
        return "Commission";
      case "usedMargin":
        return "Used Margin";
      case "takeProfit":
        return "Take Profit";
      case "risk":
        return "Risk";
      case "cost":
        return "Cost";
      case "netReturn":
        return "Net Return";
      case "r":
        return "R";
      case "mistake":
        return "Mistake";
      case "custom":
        return "Custom Tags";
      case "grade":
        return "Grade";
      case "review":
        return "Review";
      default:
        return key;
    }
  }

  function renderBrandenTradeCell(column: BrandenTradeColumnKey, trade: TradeLogEntry, grade: string) {
    switch (column) {
      case "status":
        const status = normalizedTradeStatus(trade);
        return canEditActiveTradeLog && trade.userId === user?.id ? (
          inlineSelectCell({ ...trade, status }, "status", status, ["OPEN", "WIN", "LOSS", "BREAKEVEN"], `status ${status.toLowerCase()}`)
        ) : (
          <span className={tradeBadgeClass(status)}>{status}</span>
        );
      case "side":
        return canEditActiveTradeLog && trade.userId === user?.id ? (
          inlineSelectCell(trade, "side", trade.side, ["LONG", "SHORT"], trade.side.toLowerCase())
        ) : (
          <span className={`side-pill ${trade.side.toLowerCase()}`}>{trade.side}</span>
        );
      case "symbol":
        return <span className="trade-symbol">#{trade.symbol}</span>;
      case "setup":
        return inlineSetupCell(trade);
      case "portfolio":
        return inlineTextCell(trade, "portfolioTag", trade.portfolioTag || "", "-");
      case "openDate":
        return inlineDateCell(trade, "entryDate", trade.entryDate);
      case "entry":
        return inlineNumberCell(trade, "avgEntry", trade.avgEntry);
      case "size":
        return inlineNumberCell(trade, "shares", trade.shares, { step: "0.0001" });
      case "closeDate":
        return inlineDateCell(trade, "exitDate", trade.exitDate);
      case "exit":
        return inlineNumberCell(trade, "exitPrice", trade.exitPrice);
      case "stop":
        return inlineNumberCell(trade, "stopPrice", trade.stopPrice);
      case "commission":
        return inlineNumberCell(trade, "commission", trade.commission, { money: true });
      case "usedMargin":
        return inlineNumberCell(trade, "usedMargin", trade.usedMargin, { money: true });
      case "takeProfit":
        return inlineNumberCell(trade, "takeProfitPrice", trade.takeProfitPrice);
      case "risk":
        return inlineNumberCell(trade, "risk", trade.risk, { money: true });
      case "cost":
        return money(trade.avgEntry * trade.shares);
      case "netReturn":
        return inlineNumberCell(trade, "pnl", trade.pnl, { money: true });
      case "r":
        return <span className={trade.rMultiple >= 0 ? "trade-positive" : "trade-negative"}>{trade.rMultiple.toFixed(2)}R</span>;
      case "mistake":
        return canEditActiveTradeLog && trade.userId === user?.id ? (
          inlineTextCell(trade, "mistakeTags", trade.mistakeTags.join(", "), "-")
        ) : trade.mistakeTags.length ? (
          <div className="trade-row-tags">
            {trade.mistakeTags.slice(0, 3).map((tag) => (
              <span key={tag}>{tag}</span>
            ))}
            {trade.mistakeTags.length > 3 ? <span>+{trade.mistakeTags.length - 3}</span> : null}
          </div>
        ) : (
          <span className="trade-row-tag-empty">-</span>
        );
      case "custom":
        return canEditActiveTradeLog && trade.userId === user?.id ? (
          inlineTextCell(trade, "customTags", trade.customTags.join(", "), "-")
        ) : trade.customTags.length ? (
          <div className="trade-row-tags">
            {trade.customTags.slice(0, 3).map((tag) => (
              <span key={tag}>{tag}</span>
            ))}
            {trade.customTags.length > 3 ? <span>+{trade.customTags.length - 3}</span> : null}
          </div>
        ) : (
          <span className="trade-row-tag-empty">-</span>
        );
      case "grade":
        return canEditActiveTradeLog && trade.userId === user?.id ? (
          inlineTextCell(trade, "manualGrade", trade.manualGrade || "", grade || "Grade")
        ) : (
          <span className="grade-pill">{grade}</span>
        );
      case "review":
        return (
          <span className={tradeNeedsReview(trade, setupTemplates) ? "review-pill needs-review" : "review-pill complete"}>
            {tradeNeedsReview(trade, setupTemplates) ? "Needs Review" : "Complete"}
          </span>
        );
      default:
        return "-";
    }
  }

  function renderChecklistEditor(target: "create" | "edit", currentForm: TradeFormState, disabled = false) {
    const selectedSetup = primarySetupName(currentForm.setupTags);
    const score = checklistScore(currentForm.checklistItems, setupGradeBands(selectedSetup, setupTemplates));
    const template = setupTemplateFor(selectedSetup, setupTemplates);
    const hasTemplate = Boolean(template);
    const groupedItems = currentForm.checklistItems.reduce<Record<string, TradeChecklistItem[]>>((groups, item) => {
      const key = item.groupName || "Checklist";
      groups[key] = [...(groups[key] || []), item];
      return groups;
    }, {});

    return (
      <article className="trade-checklist-editor">
        <div className="trade-checklist-heading">
          <div>
            <h3>
              Setup Criteria <span className="review-required-marker" title="Required to clear Needs Review">*</span>
            </h3>
            <span>
              {score.earned}/{score.total} points / {score.total ? score.grade : currentForm.manualGrade || "Unscored"}
            </span>
          </div>
          {!disabled && selectedSetup ? (
            <button
              className="trade-muted-button"
              type="button"
              onClick={() => {
                if (template) {
                  const setter = target === "create" ? setTradeForm : setEditTradeForm;
                  setter((current) => ({ ...current, checklistItems: checklistFromSetupTemplate(template, current.checklistItems) }));
                }
              }}
            >
              Reload setup criteria
            </button>
          ) : null}
        </div>
        {currentForm.checklistItems.length ? (
          <div className="trade-checklist-list">
            {Object.entries(groupedItems).map(([groupName, items]) => (
              <section className="trade-checklist-group" key={groupName}>
                <div className="trade-checklist-group-head">
                  <strong>{groupName}</strong>
                </div>
                {items.map((item) => (
                  <div className="trade-checklist-row" key={item.id}>
                    <label className="trade-checklist-criteria">
                      <span>{item.criteria}</span>
                    </label>
                    {(item.inputType || "boolean") === "points" ? (
                      <label className="trade-checklist-score">
                        <span>Score</span>
                        <input
                          type="number"
                          min="0"
                          max={item.points}
                          step="1"
                          inputMode="numeric"
                          value={String(item.score ?? 0)}
                          onChange={(event) =>
                            updateChecklistItem(target, item.id, {
                              score: Math.max(0, Math.min(numberValue(item.points), numberValue(event.target.value))),
                              met: numberValue(event.target.value) > 0
                            })
                          }
                          disabled={disabled}
                        />
                      </label>
                    ) : (
                      <label className="trade-checklist-met">
                        <input
                          type="checkbox"
                          checked={item.met}
                          onChange={(event) =>
                            updateChecklistItem(target, item.id, {
                              met: event.target.checked,
                              score: event.target.checked ? numberValue(item.points) : 0
                            })
                          }
                          disabled={disabled}
                        />
                        Met
                      </label>
                    )}
                    <label className="trade-checklist-points">
                      <span>Max</span>
                      <input type="number" value={String(item.points)} disabled />
                    </label>
                  </div>
                ))}
              </section>
            ))}
          </div>
        ) : (
          <p className="muted">
            {selectedSetup && !hasTemplate
              ? "No saved criteria exists for this setup yet. Open Setup Builder to define it."
              : "Assign a setup to load that setup's saved criteria."}
          </p>
        )}
      </article>
    );
  }

  async function attachTradeScreenshots(files: FileList | null) {
    if (!files?.length) {
      return;
    }

    const reads = Array.from(files).map(
      (file) =>
        new Promise<string>((resolve, reject) => {
          const reader = new FileReader();
          reader.onload = () => resolve(String(reader.result || ""));
          reader.onerror = () => reject(reader.error);
          reader.readAsDataURL(file);
        })
    );
    const screenshots = await Promise.all(reads);
    setTradeForm((current) => ({ ...current, screenshots: [...current.screenshots, ...screenshots] }));
  }

  async function attachEditTradeScreenshots(files: FileList | null) {
    if (!files?.length || !selectedTrade) {
      return;
    }

    const activeTradeId = selectedTrade.id;
    const uploadedUrls: string[] = [];

    for (const [index, file] of Array.from(files).entries()) {
      setStatus(`Saving screenshot ${index + 1} of ${files.length}...`);
      const formData = new FormData();
      formData.append("file", file);
      const response = await fetch(`/api/trades/${activeTradeId}/screenshots`, {
        method: "POST",
        body: formData
      });
      const data = await response.json().catch(() => ({}));

      if (!response.ok || !data.url) {
        setStatus(data.error || `Could not save screenshot ${index + 1}.`);
        return;
      }

      uploadedUrls.push(String(data.url));
    }

    setEditTradeForm((current) => ({
      ...current,
      screenshots: [...current.screenshots, ...uploadedUrls]
    }));
    setTrades((current) =>
      current.map((trade) =>
        trade.id === activeTradeId
          ? { ...trade, screenshots: [...trade.screenshots, ...uploadedUrls] }
          : trade
      )
    );
    setSelectedTradeId(activeTradeId);
    setStatus(`${uploadedUrls.length} ${uploadedUrls.length === 1 ? "screenshot" : "screenshots"} saved securely.`);
  }

  async function importBrokerStatement(files: FileList | null) {
    const file = files?.[0];

    if (!file) {
      return;
    }

    setStatus("Importing broker statement...");
    const csvText = await file.text();
    const importedForms = parseBrokerStatementTrades(csvText);

    if (!importedForms.length) {
      setStatus("No trades found in that statement.");
      return;
    }

    let saved = 0;
    setBusyProgress({ label: "Importing CSV trades", current: 0, total: importedForms.length, detail: "Preparing import..." });

    try {
      let firstError = "";

      for (const [index, importedForm] of importedForms.entries()) {
        setBusyProgress({
          label: "Importing CSV trades",
          current: index,
          total: importedForms.length,
          detail: importedForm.symbol ? `Saving ${importedForm.symbol}` : "Saving trade"
        });
        const response = await fetch("/api/trades", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(tradePayloadFromForm(importedForm))
        });

        if (response.ok) {
          saved += 1;
        } else if (!firstError) {
          const text = await response.text();
          const data = text ? JSON.parse(text) : {};
          firstError = data.error || `Trade import failed for ${importedForm.symbol || "a row"}.`;
        }

        setBusyProgress({
          label: "Importing CSV trades",
          current: index + 1,
          total: importedForms.length,
          detail: `${saved} saved`
        });
      }

      await loadTrades();
      setTradeView("table");
      setTradeFilters((current) => ({ ...current, review: ["Needs Review"], custom: [] }));
      setStatus(
        saved
          ? `Imported ${saved} trades. ${saved} need review for notes, screenshots, and checklist scoring.${firstError ? ` First error: ${firstError}` : ""}`
          : firstError || "No CSV trades were imported."
      );
    } finally {
      setBusyProgress(null);
    }
  }

  async function importExcelTradeLog(files: FileList | null) {
    const file = files?.[0];

    if (!file) {
      return;
    }

    setStatus(`Selected Excel file: ${file.name}`);

    const targetPortfolio = pendingImportPortfolioRef.current.trim() || importPortfolioTag.trim() || activePortfolio.trim();
    pendingImportPortfolioRef.current = "";

    if (!targetPortfolio) {
      setStatus("Choose, select, or create a portfolio before importing Excel.");
      return;
    }

    if (!savedPortfolios.includes(targetPortfolio)) {
      const savedPortfolioList = await saveBrandenPortfolios([...savedPortfolios, targetPortfolio]);

      if (!savedPortfolioList) {
        return;
      }
    }

    setStatus("Importing Excel trade log...");
    const importedTrades = await parseExcelTradeLogWorkbook(file);

    if (!importedTrades.length) {
      setStatus("No importable trades found in that Excel trade log.");
      return;
    }

    let saved = 0;
    let created = 0;
    let updated = 0;
    setBusyProgress({ label: "Importing Excel trades", current: 0, total: importedTrades.length, detail: "Preparing import..." });

    try {
      let firstError = "";

      for (const [index, importedTrade] of importedTrades.entries()) {
        setBusyProgress({
          label: "Importing Excel trades",
          current: index,
          total: importedTrades.length,
          detail: importedTrade.symbol ? `Saving ${importedTrade.symbol}` : "Saving trade"
        });
        const response = await fetch("/api/trades", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(importedTradePayload({ ...importedTrade, portfolioTag: targetPortfolio }, setupTemplates))
        });

        if (response.ok) {
          const data = await response.json();
          saved += 1;
          if (data.mode === "updated") {
            updated += 1;
          } else {
            created += 1;
          }
        } else if (!firstError) {
          const text = await response.text();
          const data = text ? JSON.parse(text) : {};
          firstError = data.error || `Trade import failed for ${importedTrade.symbol || "a row"}.`;
        }

        setBusyProgress({
          label: "Importing Excel trades",
          current: index + 1,
          total: importedTrades.length,
          detail: `${saved} saved`
        });
      }

      await loadTrades();
      setTradeView("table");
      setTradeFilters(emptyTradeFilters);
      setActivePortfolio(targetPortfolio);
      setStatus(
        saved
          ? `Imported ${saved} Excel trades. New: ${created}. Updated from matching sheet rows: ${updated}.${firstError ? ` First error: ${firstError}` : ""}`
          : firstError || "No Excel trades were imported."
      );
    } finally {
      setBusyProgress(null);
    }
  }

  async function importCfStatement(files: FileList | null) {
    const file = files?.[0];

    if (!file) {
      return;
    }

    setStatus(`Selected CF statement: ${file.name}`);

    const targetPortfolio = pendingImportPortfolioRef.current.trim() || importPortfolioTag.trim() || activePortfolio.trim();
    pendingImportPortfolioRef.current = "";

    if (!targetPortfolio) {
      setStatus("Choose, select, or create a portfolio before CF import.");
      return;
    }

    if (!savedPortfolios.includes(targetPortfolio)) {
      const savedPortfolioList = await saveBrandenPortfolios([...savedPortfolios, targetPortfolio]);

      if (!savedPortfolioList) {
        return;
      }
    }

    setStatus("Importing CF statement...");
    setBusyProgress({ label: "Importing CF statement", current: 0, total: 100, detail: "Preparing upload..." });

    let progressTimer: ReturnType<typeof setInterval> | null = null;
    try {
      const formData = new FormData();
      formData.append("file", file);
      formData.append("portfolioTag", targetPortfolio);
      let simulatedProgress = 8;
      setBusyProgress({ label: "Importing CF statement", current: simulatedProgress, total: 100, detail: `Uploading ${file.name}` });

      progressTimer = setInterval(() => {
        simulatedProgress = Math.min(88, simulatedProgress + (simulatedProgress < 45 ? 7 : 3));
        const detail =
          simulatedProgress < 35
            ? `Uploading ${file.name}`
            : simulatedProgress < 72
              ? "Parsing PDF statement on the server..."
              : "Matching fills and preparing journal rows...";
        setBusyProgress({ label: "Importing CF statement", current: simulatedProgress, total: 100, detail });
      }, 900);

      const response = await fetch("/api/import/cf-statement", {
        method: "POST",
        body: formData
      });
      if (progressTimer) {
        clearInterval(progressTimer);
        progressTimer = null;
      }
      setBusyProgress({ label: "Importing CF statement", current: 90, total: 100, detail: "Reading import result..." });
      const text = await response.text();
      let data: Record<string, unknown> = {};

      if (text) {
        try {
          data = JSON.parse(text);
        } catch {
          data = {
            error: `CF import returned a non-JSON response (${response.status}). ${text.slice(0, 140)}`
          };
        }
      }

      if (!response.ok) {
        setStatus(String(data.error || "Could not import CF statement."));
        return;
      }

      setBusyProgress({ label: "Importing CF statement", current: 95, total: 100, detail: "Refreshing journal..." });
      const nextTrades = await loadTrades();
      setTradeView("table");
      setActivePortfolio(targetPortfolio);
      setTradeFilters(emptyTradeFilters);

      const importedPortfolioTrades = nextTrades.filter((trade) => trade.portfolioTag === targetPortfolio && trade.importSource === "cf-statement-pdf");
      const visibleImportedTrades = importedPortfolioTrades.filter((trade) => !trade.hidden);
      const importedExecutionCount = visibleImportedTrades.reduce((sum, trade) => sum + (trade.executions?.length || 0), 0);
      const hiddenImportedTrades = importedPortfolioTrades.filter((trade) => trade.hidden);

      if (visibleImportedTrades[0]) {
        setSelectedTradeId(visibleImportedTrades[0].id);
      }

      setBusyProgress({
        label: "Importing CF statement",
        current: 100,
        total: 100,
        detail: `${data.imported} rows loaded`
      });
      setStatus(
        `CF import complete. Built ${visibleImportedTrades.length} position rows from ${data.imported} parsed positions/fills. Open: ${data.openTrades}. Closed: ${data.closedTrades}. Needs review: ${data.needsReview}. New: ${data.created}. Updated: ${data.updated}. Stored executions: ${importedExecutionCount}.${data.hiddenLegacyRows ? ` Migrated old transaction rows out of the main log: ${data.hiddenLegacyRows}.` : ""}${data.hiddenSupersededRows ? ` Hid corrected old wrong-side rows: ${data.hiddenSupersededRows}.` : ""}${data.hiddenOpenComponentRows ? ` Collapsed old open fill rows: ${data.hiddenOpenComponentRows}.` : ""}${data.hiddenStaleOpenRows ? ` Moved stale open rows out of the main log: ${data.hiddenStaleOpenRows}.` : ""}${hiddenImportedTrades.length ? ` Hidden/excluded: ${hiddenImportedTrades.length}.` : ""}${data.currentEquity ? ` Current equity saved: ${money(Number(data.currentEquity))}.` : " Current equity was not found in the statement."}`
      );
    } catch (error) {
      setStatus(error instanceof Error ? `CF import failed: ${error.message}` : "CF import failed.");
    } finally {
      if (progressTimer) {
        clearInterval(progressTimer);
      }
      setBusyProgress(null);
    }
  }

  async function choosePortfolioForImport(importLabel: string) {
    const defaultValue = importPortfolioTag.trim() || activePortfolio.trim() || defaultPortfolio.trim();
    const savedList = portfolioOptions.length ? `\n\nSaved portfolios:\n${portfolioOptions.join(", ")}` : "";
    const requestedPortfolio = window.prompt(`${importLabel}: which portfolio should this import save to?${savedList}`, defaultValue);

    if (requestedPortfolio === null) {
      setStatus(`${importLabel} cancelled.`);
      return "";
    }

    const targetPortfolio = requestedPortfolio.trim();

    if (!targetPortfolio) {
      setStatus(`Enter a portfolio name before starting ${importLabel}.`);
      return "";
    }

    if (!savedPortfolios.includes(targetPortfolio)) {
      const savedPortfolioList = await saveBrandenPortfolios([...savedPortfolios, targetPortfolio]);

      if (!savedPortfolioList) {
        return "";
      }
    }

    pendingImportPortfolioRef.current = targetPortfolio;
    setImportPortfolioTag(targetPortfolio);
    setActivePortfolio(targetPortfolio);
    return targetPortfolio;
  }

  async function saveMarketCycleJournalEntry() {
    if (!marketCycleForm.date) {
      setStatus("Choose a date for the market cycle journal entry.");
      return;
    }

    setStatus("Saving market cycle entry...");

    const response = await fetch("/api/settings/market-cycle", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        date: marketCycleForm.date,
        trendDay: numberValue(marketCycleForm.trendDay),
        phase: marketCycleForm.phase,
        notes: marketCycleForm.notes
      })
    });
    const text = await response.text();
    const data = text ? JSON.parse(text) : {};

    if (!response.ok) {
      setStatus(data.error || "Could not save market cycle entry.");
      return;
    }

    setMarketCycleEntries(data.entries || []);
    setStatus(`Market cycle entry saved for ${marketCycleForm.date}.`);
  }

  async function saveTrade(event: FormEvent) {
    event.preventDefault();
    setStatus("Saving trade...");

    if (tradeForm.portfolioTag.trim() && !savedPortfolios.includes(tradeForm.portfolioTag.trim())) {
      const savedPortfolioList = await saveBrandenPortfolios([...savedPortfolios, tradeForm.portfolioTag.trim()]);

      if (!savedPortfolioList) {
        return;
      }
    }

    const response = await fetch("/api/trades", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(tradePayloadFromForm(tradeForm))
    });
    const data = await response.json();

    if (!response.ok) {
      setStatus(data.error || "Could not save trade.");
      return;
    }

    setStatus(`${tradeForm.symbol.toUpperCase()} trade saved.`);
    setTradeForm(emptyTradeForm);
    setIsTradeModalOpen(false);
    await loadTrades();
  }

  async function saveTradeEdit(event: FormEvent) {
    event.preventDefault();

    if (!selectedTrade) {
      return;
    }

    setStatus("Saving trade...");

    if (editTradeForm.portfolioTag.trim() && !savedPortfolios.includes(editTradeForm.portfolioTag.trim())) {
      const savedPortfolioList = await saveBrandenPortfolios([...savedPortfolios, editTradeForm.portfolioTag.trim()]);

      if (!savedPortfolioList) {
        return;
      }
    }

    const response = await fetch(`/api/trades/${selectedTrade.id}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(tradePayloadFromForm(editTradeForm))
    });
    const data = await response.json();

    if (!response.ok) {
      setStatus(data.error || "Could not update trade.");
      return;
    }

    setStatus(`${editTradeForm.symbol.toUpperCase()} trade updated.`);
    await loadTrades();
  }

  async function removeTrade(trade: TradeLogEntry) {
    await fetch(`/api/trades/${trade.id}`, { method: "DELETE" });
    setSelectedTradeId("");
    setSelectedTradeIds((current) => current.filter((id) => id !== trade.id));
    await loadTrades();
  }

  function navigateSelectedTrade(direction: "previous" | "next") {
    const targetId =
      direction === "previous" ? selectedTradeNavigation.previousTradeId : selectedTradeNavigation.nextTradeId;

    if (!targetId) {
      return;
    }

    setSelectedTradeId(targetId);
  }

  function toggleTradeSelection(id: string) {
    setSelectedTradeIds((current) => (current.includes(id) ? current.filter((tradeId) => tradeId !== id) : [...current, id]));
  }

  function toggleAllFilteredTrades(checked: boolean) {
    setSelectedTradeIds((current) => {
      const visible = new Set(filteredTradeIds);

      if (!checked) {
        return current.filter((id) => !visible.has(id));
      }

      return Array.from(new Set([...current, ...filteredTradeIds]));
    });
  }

  function exportFilteredTradesCsv() {
    const headers = [
      "export_row_type",
      "visible_in_table",
      "id",
      "hidden",
      "import_source",
      "import_row_key",
      "portfolio",
      "status",
      "side",
      "symbol",
      "entry_date",
      "open_time",
      "avg_entry",
      "shares",
      "exit_date",
      "close_time",
      "exit_price",
      "stop_price",
      "take_profit_price",
      "risk",
      "pnl",
      "r_multiple",
      "commission",
      "used_margin",
      "return_percent",
      "days_in_trade",
      "setup_tags",
      "mistake_tags",
      "custom_tags",
      "manual_grade",
      "needs_review",
      "review_setup",
      "review_entry",
      "review_exit",
      "review_did_right",
      "review_did_wrong",
      "review_general",
      "notes",
      "screenshots",
      "chart_links"
    ];
    const rows: unknown[][] = [];
    const exportedIds = new Set<string>();

    function addTradeRow(trade: TradeLogEntry, rowType: "table_trade", visibleInTable = false) {
      if (exportedIds.has(`${rowType}:${trade.id}`)) {
        return;
      }

      exportedIds.add(`${rowType}:${trade.id}`);
      const review = resolvedTradeReviewSections(trade.reviewSections, trade.notes);
      rows.push([
        rowType,
        visibleInTable ? "yes" : "no",
        trade.id,
        trade.hidden ? "yes" : "no",
        trade.importSource,
        trade.importRowKey,
        trade.portfolioTag,
        trade.status,
        trade.side,
        trade.symbol,
        trade.entryDate,
        trade.openTime,
        trade.avgEntry,
        trade.shares,
        trade.exitDate,
        trade.closeTime,
        trade.exitPrice,
        trade.stopPrice,
        trade.takeProfitPrice,
        trade.risk,
        trade.pnl,
        trade.rMultiple,
        trade.commission,
        trade.usedMargin,
        trade.returnPercent,
        trade.daysInTrade,
        trade.setupTags,
        trade.mistakeTags,
        trade.customTags,
        trade.manualGrade,
        tradeNeedsReview(trade, setupTemplates) ? "yes" : "no",
        review.setup,
        review.entry,
        review.exit,
        review.didRight,
        review.didWrong,
        review.general,
        trade.notes,
        trade.screenshots,
        trade.chartLinks
      ]);
    }

    filteredTrades.forEach((trade) => {
      addTradeRow(trade, "table_trade", true);
    });

    const csv = [headers, ...rows].map((row) => row.map(csvValue).join(",")).join("\n");
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    const portfolioLabel = activePortfolio ? activePortfolio.replace(/[^a-z0-9_-]+/gi, "-") : "all-portfolios";

    link.href = url;
    link.download = `branden-trade-export-${portfolioLabel}-${new Date().toISOString().slice(0, 10)}.csv`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
    setStatus(`Exported ${filteredTrades.length} table rows.`);
  }

  async function saveBrandenPortfolios(nextPortfolios: string[], nextDefaultPortfolio = defaultPortfolio) {
    const normalizedPortfolios = Array.from(new Set(nextPortfolios.map((name) => name.trim()).filter(Boolean))).sort((a, b) =>
      a.localeCompare(b)
    );
    const normalizedDefault = nextDefaultPortfolio && normalizedPortfolios.includes(nextDefaultPortfolio) ? nextDefaultPortfolio : "";
    const response = await fetch("/api/settings/branden-portfolios", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ portfolios: normalizedPortfolios, defaultPortfolio: normalizedDefault })
    });
    const data = await response.json();

    if (!response.ok) {
      setStatus(data.error || "Could not save portfolios.");
      return null;
    }

    setSavedPortfolios(data.portfolios || []);
    setDefaultPortfolio(data.defaultPortfolio || "");
    return data.portfolios || [];
  }

  async function saveDefaultPortfolio(nextDefaultPortfolio: string) {
    const nextPortfolios =
      nextDefaultPortfolio && !savedPortfolios.includes(nextDefaultPortfolio)
        ? Array.from(new Set([...savedPortfolios, nextDefaultPortfolio])).sort((a, b) => a.localeCompare(b))
        : savedPortfolios;
    const saved = await saveBrandenPortfolios(nextPortfolios, nextDefaultPortfolio);

    if (!saved) {
      return;
    }

    setDefaultPortfolio(nextDefaultPortfolio);
    setActivePortfolio(nextDefaultPortfolio);
    setStatus(nextDefaultPortfolio ? `Default portfolio set to ${nextDefaultPortfolio}.` : "Default portfolio set to all portfolios.");
  }

  async function saveBrandenColumns(nextPreferences: Record<string, BrandenColumnPreference[]>) {
    const response = await fetch("/api/settings/branden-columns", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ preferences: nextPreferences })
    });
    const data = await response.json();

    if (!response.ok) {
      setStatus(data.error || "Could not save column settings.");
      return null;
    }

    const saved = data.preferences && typeof data.preferences === "object" ? data.preferences : {};
    setColumnPreferences(saved);
    return saved;
  }

  async function exportBrandenJournalBackup() {
    setStatus("Preparing full journal backup...");
    const response = await fetch("/api/settings/branden-backup", { cache: "no-store" });
    if (!response.ok) {
      const data = await response.json().catch(() => ({}));
      setStatus(data.error || "Could not export the journal backup.");
      return;
    }
    const blob = await response.blob();
    const disposition = response.headers.get("Content-Disposition") || "";
    const fileName = disposition.match(/filename="([^"]+)"/)?.[1] || `branden-journal-backup-${currentDate()}.json`;
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = fileName;
    link.click();
    URL.revokeObjectURL(url);
    setStatus("Full journal backup exported.");
  }

  async function importBrandenJournalBackup(files: FileList | null) {
    const file = files?.[0];
    if (!file) return;
    const confirmed = window.confirm(
      "Restore this full Branden journal backup?\n\nThis replaces the current Branden trades, reports, settings, watchlists, market-cycle history, and trade screenshots with the contents of the backup file."
    );
    if (!confirmed) return;

    setStatus("Validating and restoring journal backup...");
    try {
      const backup = JSON.parse(await file.text());
      const response = await fetch("/api/settings/branden-backup", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(backup)
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        setStatus(data.error || "Could not import the journal backup.");
        return;
      }
      setStatus(
        `Backup restored: ${data.restored?.trades || 0} trades, ${data.restored?.reports || 0} reports, ${data.restored?.screenshots || 0} screenshots. Reloading...`
      );
      window.setTimeout(() => window.location.reload(), 800);
    } catch (importError) {
      setStatus(importError instanceof Error ? importError.message : "The selected backup file is invalid.");
    }
  }

  async function createBrandenPortfolio() {
    const nextName = importPortfolioTag.trim();

    if (!nextName) {
      setStatus("Enter a portfolio name first.");
      return;
    }

    const nextPortfolios = Array.from(new Set([...savedPortfolios, nextName])).sort((a, b) => a.localeCompare(b));
    const saved = await saveBrandenPortfolios(nextPortfolios);

    if (!saved) {
      return;
    }

    setActivePortfolio(nextName);
    setTradeForm((current) => ({ ...current, portfolioTag: nextName }));
    setStatus(`Portfolio ${nextName} saved.`);
  }

  function clearTradeFilters() {
    setTradeFilters(emptyTradeFilters);
    setActivePortfolio("");
  }

  async function updateActiveBrandenColumns(nextColumns: BrandenColumnPreference[]) {
    const nextPreferences = {
      ...columnPreferences,
      [activeColumnScope]: normalizeBrandenColumns(nextColumns)
    };
    await saveBrandenColumns(nextPreferences);
  }

  async function toggleBrandenColumn(key: BrandenTradeColumnKey) {
    await updateActiveBrandenColumns(
      activeBrandenColumns.map((column) => (column.key === key ? { ...column, visible: !column.visible } : column))
    );
  }

  async function bulkExcludeSelectedTrades() {
    const ownedSelectedTrades = selectedFilteredTrades.filter((trade) => trade.userId === user?.id);

    if (!ownedSelectedTrades.length) {
      setStatus("No selected trades you own can be hidden.");
      return;
    }

    const confirmed = window.confirm(
      `Hide/exclude ${ownedSelectedTrades.length} selected ${ownedSelectedTrades.length === 1 ? "trade" : "trades"}?\n\nHidden trades are removed from the trade log, calendar, charts, stats, and portfolio totals. They are preserved in Hidden trades and can be restored later. This is the safer option for imported CF rows because future imports will not recreate them.`
    );

    if (!confirmed) {
      return;
    }

    let hidden = 0;
    setStatus(`Hiding/excluding ${ownedSelectedTrades.length} trades...`);
    setBusyProgress({ label: "Hiding selected trades", current: 0, total: ownedSelectedTrades.length, detail: "Starting hide/exclude..." });

    try {
      for (const [index, trade] of ownedSelectedTrades.entries()) {
        setBusyProgress({
          label: "Hiding selected trades",
          current: index,
          total: ownedSelectedTrades.length,
          detail: `Hiding ${trade.symbol}`
        });
        const response = await fetch(`/api/trades/${trade.id}`, {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ hidden: true })
        });

        if (response.ok) {
          hidden += 1;
        }

        setBusyProgress({
          label: "Hiding selected trades",
          current: index + 1,
          total: ownedSelectedTrades.length,
          detail: `${hidden} hidden/excluded`
        });
      }

      setSelectedTradeIds((current) => current.filter((id) => !ownedSelectedTrades.some((trade) => trade.id === id)));
      setSelectedTradeId("");
      await loadTrades();
      setStatus(`${hidden} selected ${hidden === 1 ? "trade" : "trades"} hidden/excluded.`);
    } finally {
      setBusyProgress(null);
    }
  }

  async function bulkDeleteSelectedTrades() {
    const ownedSelectedTrades = selectedFilteredTrades.filter((trade) => trade.userId === user?.id);

    if (!ownedSelectedTrades.length) {
      setStatus("No selected trades you own can be permanently deleted.");
      return;
    }

    const importedCount = ownedSelectedTrades.filter((trade) => trade.importSource && trade.importSource !== "manual-combine").length;
    const importedWarning = importedCount
      ? `\n\n${importedCount} selected ${importedCount === 1 ? "row is" : "rows are"} imported. Permanently deleting imported rows means a future import of the same statement can recreate them.`
      : "";
    const confirmed = window.confirm(
      `Permanently delete ${ownedSelectedTrades.length} selected ${ownedSelectedTrades.length === 1 ? "trade" : "trades"}?\n\nThis removes the rows from storage instead of hiding them. This cannot be undone.${importedWarning}`
    );

    if (!confirmed) {
      return;
    }

    let deleted = 0;
    setStatus(`Permanently deleting ${ownedSelectedTrades.length} trades...`);
    setBusyProgress({ label: "Deleting selected trades", current: 0, total: ownedSelectedTrades.length, detail: "Starting permanent delete..." });

    try {
      for (const [index, trade] of ownedSelectedTrades.entries()) {
        setBusyProgress({
          label: "Deleting selected trades",
          current: index,
          total: ownedSelectedTrades.length,
          detail: `Deleting ${trade.symbol}`
        });
        const response = await fetch(`/api/trades/${trade.id}`, { method: "DELETE" });

        if (response.ok) {
          deleted += 1;
        }

        setBusyProgress({
          label: "Deleting selected trades",
          current: index + 1,
          total: ownedSelectedTrades.length,
          detail: `${deleted} permanently deleted`
        });
      }

      setSelectedTradeIds((current) => current.filter((id) => !ownedSelectedTrades.some((trade) => trade.id === id)));
      if (ownedSelectedTrades.some((trade) => trade.id === selectedTradeId)) {
        setSelectedTradeId("");
      }
      await loadTrades();
      setStatus(`${deleted} selected ${deleted === 1 ? "trade" : "trades"} permanently deleted.`);
    } finally {
      setBusyProgress(null);
    }
  }

  async function updateTradeVisibility(tradesToUpdate: TradeLogEntry[], hidden: boolean) {
    const ownedTrades = tradesToUpdate.filter((trade) => trade.userId === user?.id);
    const action = hidden ? "hide" : "unhide";
    const actionLabel = hidden ? "Hiding selected trades" : "Unhiding selected trades";

    if (!ownedTrades.length) {
      setStatus(`No selected trades you own can be ${hidden ? "hidden" : "unhidden"}.`);
      return;
    }

    let updated = 0;
    setStatus(`${hidden ? "Hiding" : "Unhiding"} ${ownedTrades.length} trades...`);
    setBusyProgress({ label: actionLabel, current: 0, total: ownedTrades.length, detail: "Starting update..." });

    try {
      for (const [index, trade] of ownedTrades.entries()) {
        const nextCustomTags = hidden
          ? Array.from(new Set([...trade.customTags, manualHiddenTag]))
          : trade.customTags.filter((tag) => tag !== manualHiddenTag);
        setBusyProgress({
          label: actionLabel,
          current: index,
          total: ownedTrades.length,
          detail: `${hidden ? "Hiding" : "Unhiding"} ${trade.symbol}`
        });
        const response = await fetch(`/api/trades/${trade.id}`, {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ hidden, customTags: nextCustomTags })
        });

        if (response.ok) {
          updated += 1;
        }

        setBusyProgress({
          label: actionLabel,
          current: index + 1,
          total: ownedTrades.length,
          detail: `${updated} ${hidden ? "hidden" : "unhidden"}`
        });
      }

      setSelectedTradeIds((current) => current.filter((id) => !ownedTrades.some((trade) => trade.id === id)));
      setSelectedHiddenTradeIds((current) => current.filter((id) => !ownedTrades.some((trade) => trade.id === id)));
      if (ownedTrades.some((trade) => trade.id === selectedTradeId)) {
        setSelectedTradeId("");
      }
      await loadTrades();
      setStatus(`${updated} selected ${updated === 1 ? "trade" : "trades"} ${action === "hide" ? "hidden" : "unhidden"}.`);
    } finally {
      setBusyProgress(null);
    }
  }

  async function unhideSelectedTrades() {
    const tradesToUnhide = hiddenOwnerTrades.filter((trade) => selectedHiddenTradeIds.includes(trade.id));
    await updateTradeVisibility(tradesToUnhide, false);
  }

  async function deleteSelectedHiddenTrades() {
    const tradesToDelete = hiddenOwnerTrades.filter((trade) => selectedHiddenTradeIds.includes(trade.id) && trade.userId === user?.id);

    if (!tradesToDelete.length) {
      setStatus("No selected hidden trades you own can be permanently deleted.");
      return;
    }

    const confirmed = window.confirm(
      `Permanently delete ${tradesToDelete.length} hidden ${tradesToDelete.length === 1 ? "trade" : "trades"}?\n\nThis removes them from storage. A future broker import can recreate imported rows if the same source trade appears again.`
    );

    if (!confirmed) {
      return;
    }

    let deleted = 0;
    setStatus(`Deleting ${tradesToDelete.length} hidden trades...`);
    setBusyProgress({ label: "Deleting hidden trades", current: 0, total: tradesToDelete.length, detail: "Starting delete..." });

    try {
      for (const [index, trade] of tradesToDelete.entries()) {
        setBusyProgress({
          label: "Deleting hidden trades",
          current: index,
          total: tradesToDelete.length,
          detail: `Deleting ${trade.symbol}`
        });
        const response = await fetch(`/api/trades/${trade.id}`, { method: "DELETE" });

        if (response.ok) {
          deleted += 1;
        }

        setBusyProgress({
          label: "Deleting hidden trades",
          current: index + 1,
          total: tradesToDelete.length,
          detail: `${deleted} deleted`
        });
      }

      setSelectedHiddenTradeIds((current) => current.filter((id) => !tradesToDelete.some((trade) => trade.id === id)));
      if (tradesToDelete.some((trade) => trade.id === selectedTradeId)) {
        setSelectedTradeId("");
      }
      await loadTrades();
      setStatus(`${deleted} hidden ${deleted === 1 ? "trade" : "trades"} permanently deleted.`);
    } finally {
      setBusyProgress(null);
    }
  }

  if (isLoading) {
    return <main className="center-screen">Loading dashboard...</main>;
  }

  if (!user) {
    return (
      <main className="login-shell">
        <section className="login-panel compact-login-panel">
          <form className="login-form" onSubmit={handleLogin}>
            <label>
              Trader
              <select value={login.userId} onChange={(event) => setLogin({ ...login, userId: event.target.value })}>
                <option value="branden">Branden</option>
                <option value="cam">Cam</option>
                <option value="tim">Tim</option>
              </select>
            </label>
            <label>
              Password
              <input
                type="password"
                value={login.password}
                onChange={(event) => setLogin({ ...login, password: event.target.value })}
                placeholder="Enter password"
              />
            </label>
            <button type="submit">Log in</button>
            {status ? <p className="status error">{status}</p> : null}
          </form>
        </section>
      </main>
    );
  }

  const busyPercent = busyProgress?.total
    ? Math.min(100, Math.round((busyProgress.current / busyProgress.total) * 100))
    : 0;

  return (
    <main className="app-shell">
      {busyProgress ? (
        <div className="busy-overlay" role="status" aria-live="polite">
          <section className="busy-panel">
            <span className="busy-eyebrow">Working</span>
            <h2>{busyProgress.label}</h2>
            <p>{busyProgress.detail}</p>
            <div className="busy-meter" aria-label={`${busyPercent}% complete`}>
              <i style={{ width: `${busyPercent}%` }} />
            </div>
            <strong>
              {busyProgress.current} / {busyProgress.total} complete · {busyPercent}%
            </strong>
          </section>
        </div>
      ) : null}
      {!isBrandenJournalMode ? (
        <>
          <header className="topbar">
            <div>
              <p className="eyebrow">Shared trading dashboard</p>
              <h1>Monthly Performance Review</h1>
            </div>
            <div className="top-actions">
              <span>{user.name}</span>
              <button className="ghost-button" type="button" onClick={handleLogout}>
                Log out
              </button>
            </div>
          </header>

          <nav className="app-tabs" aria-label="Monthly report sections">
            <button
              type="button"
              className={activeTab === "entry" ? "active" : ""}
              onClick={() => setActiveTab("entry")}
            >
              Enter Data
            </button>
            <button
              type="button"
              className={activeTab === "dashboard" ? "active" : ""}
              onClick={() => setActiveTab("dashboard")}
            >
              Dashboard
            </button>
            <button
              type="button"
              className={activeTab === "manage" ? "active" : ""}
              onClick={() => setActiveTab("manage")}
            >
              Edit Entries
            </button>
            <button
              type="button"
              className={activeTab === "trades" ? "active" : ""}
              onClick={() => setActiveTab("trades")}
            >
              Trade Logging
            </button>
          </nav>
        </>
      ) : null}

      {reportsError ? <div className="app-alert">{reportsError}</div> : null}
      {tradesError ? <div className="app-alert">{tradesError}</div> : null}

      {activeTab === "entry" ? (
        <section className="entry-layout">
          <form className="report-form" onSubmit={saveReport}>
            <div className="section-heading">
              <div>
                <p className="eyebrow">Monthly input</p>
                <h2>Submit or edit your report</h2>
              </div>
              <label className="month-control">
                Month
                <input
                  type="month"
                  value={form.month}
                  onChange={(event) => updateField("month", event.target.value)}
                />
              </label>
            </div>

            {fieldGroups.map((group) => (
              <fieldset key={group.title}>
                <legend>{group.title}</legend>
                <div className="field-grid">
                  {group.fields.map(([key, label, suffix]) => (
                    <label key={key}>
                      {label}
                      <div className="input-wrap">
                        <input
                          type="number"
                          step="any"
                          inputMode="decimal"
                          value={String(form[key])}
                          onChange={(event) => updateField(key, event.target.value)}
                        />
                        {suffix ? <span>{suffix}</span> : null}
                      </div>
                    </label>
                  ))}
                </div>
              </fieldset>
            ))}

            <label className="notes-field">
              Notes
              <textarea
                value={form.notes}
                onChange={(event) => updateField("notes", event.target.value)}
                placeholder="What mattered this month? Process wins, mistakes, market conditions, rule breaks."
              />
            </label>

            <div className="form-actions">
              <button type="submit">Save monthly report</button>
              <button className="ghost-button" type="button" onClick={() => setForm(emptyForm)}>
                Clear
              </button>
              <button className="ghost-button" type="button" onClick={() => setActiveTab("dashboard")}>
                View dashboard
              </button>
              {status ? <span className="status">{status}</span> : null}
            </div>
          </form>

          <aside className="entry-side-panel">
            <div>
              <p className="eyebrow">Calculated preview</p>
              <h2>Auto-calculated stats</h2>
            </div>
            <dl className="derived-preview-grid">
              <div>
                <dt>% Return</dt>
                <dd>{pct(numberValue(derivedPreview.percentReturn))}</dd>
              </div>
              <div>
                <dt>Total net R multiple</dt>
                <dd>{numberValue(derivedPreview.totalR).toFixed(2)}R</dd>
                <small>Entered manually for the selected monthly report.</small>
              </div>
              <div>
                <dt>Avg R</dt>
                <dd>{numberValue(derivedPreview.avgR).toFixed(2)}R</dd>
              </div>
              <div>
                <dt>Expected value</dt>
                <dd>{numberValue(derivedPreview.expectedValueR).toFixed(2)}R</dd>
              </div>
              <div>
                <dt>Avg win</dt>
                <dd>{money(numberValue(derivedPreview.avgWin))}</dd>
              </div>
              <div>
                <dt>Avg loss</dt>
                <dd>{money(numberValue(derivedPreview.avgLoss))}</dd>
              </div>
            </dl>

            <div>
              <p className="eyebrow">Your entries</p>
              <h2>{user.name}&apos;s saved months</h2>
            </div>
            <div className="entry-month-list">
              {myReports.map((report) => (
                <button key={report.id} type="button" onClick={() => editReport(report)}>
                  <span>{monthLabel(report.month)}</span>
                  <strong>{money(report.netPnl)}</strong>
                </button>
              ))}
              {!myReports.length ? <p className="muted">No saved months yet.</p> : null}
            </div>
          </aside>
        </section>
      ) : activeTab === "dashboard" ? (
        <div className="dashboard-layout">
          <section className="trader-reviews">
            <div className="section-heading">
              <div>
                <p className="eyebrow">Shared review</p>
                <h2>Individual monthly progress</h2>
              </div>
            </div>
            <div className="trader-review-grid">
              {traderSummaries.map((trader) => (
                <article key={trader.trader} className="trader-review-card">
                  <div>
                    <span>{trader.displayName}</span>
                    <strong>{money(trader.totalNetPnl)}</strong>
                  </div>
                  <dl>
                    <div>
                      <dt>Months</dt>
                      <dd>{trader.reports}</dd>
                    </div>
                    <div>
                      <dt>Trades</dt>
                      <dd>{trader.totalTrades}</dd>
                    </div>
                    <div>
                      <dt>Avg R</dt>
                      <dd>{trader.avgR.toFixed(2)}R</dd>
                    </div>
                    <div>
                      <dt>Win rate</dt>
                      <dd>{pct(trader.winRate)}</dd>
                    </div>
                    <div>
                      <dt>EV</dt>
                      <dd>{trader.expectedValueR.toFixed(2)}R</dd>
                    </div>
                    <div>
                      <dt>Stability</dt>
                      <dd>{trader.returnStability.toFixed(2)}</dd>
                    </div>
                  </dl>
                  <small>
                    {trader.bestMonth
                      ? `Best month: ${monthLabel(trader.bestMonth.month)} at ${money(trader.bestMonth.netPnl)}`
                      : `${trader.displayName} has not submitted a report yet`}
                  </small>
                </article>
              ))}
            </div>
          </section>

          <section className="chart-stack">
            <div className="chart-filter-panel">
              <div>
                <p className="eyebrow">Graph filters</p>
                <h2>Chart data</h2>
              </div>
              <div className="segmented-control" aria-label="Chart trader filter">
                {(["both", "branden", "cam"] as const).map((filter) => (
                  <button
                    key={filter}
                    type="button"
                    className={chartFilter === filter ? "active" : ""}
                    onClick={() => setChartFilter(filter)}
                  >
                    {filter === "both" ? "Both" : filter.charAt(0).toUpperCase() + filter.slice(1)}
                  </button>
                ))}
              </div>
            </div>

            <article className="chart-panel tall">
              <div className="section-heading">
                <div>
                  <p className="eyebrow">Trend</p>
                  <h2>Monthly net P&L</h2>
                </div>
              </div>
              <ResponsiveContainer width="100%" height={primaryChartHeight}>
                <LineChart data={traderMonthly} margin={responsiveChartMargin}>
                  <CartesianGrid strokeDasharray="4 6" stroke={chartGridColor} vertical={false} />
                  <XAxis
                    dataKey="label"
                    axisLine={false}
                    interval={xAxisInterval}
                    minTickGap={isPhoneViewport ? 18 : 5}
                    tickLine={false}
                    tick={{ fill: chartAxisColor, fontSize: isPhoneViewport ? 11 : 12 }}
                  />
                  <YAxis
                    axisLine={false}
                    tickLine={false}
                    tick={{ fill: chartAxisColor, fontSize: isPhoneViewport ? 11 : 12 }}
                    tickFormatter={(value) => `$${Number(value) / 1000}k`}
                    width={isPhoneViewport ? 44 : 60}
                  />
                  <Tooltip contentStyle={chartTooltipStyle} formatter={(value) => money(Number(value))} />
                  <Legend iconType="circle" wrapperStyle={chartLegendStyle} />
                  {showsTrader(chartFilter, "branden") ? (
                    <Line
                      type="monotone"
                      dataKey="brandenPnl"
                      name="Branden"
                      stroke={brandenColor}
                      strokeWidth={3}
                      dot={{ r: 4, strokeWidth: 2 }}
                      activeDot={{ r: 7, strokeWidth: 0 }}
                    />
                  ) : null}
                  {showsTrader(chartFilter, "cam") ? (
                    <Line
                      type="monotone"
                      dataKey="camPnl"
                      name="Cam"
                      stroke={camColor}
                      strokeWidth={3}
                      dot={{ r: 4, strokeWidth: 2 }}
                      activeDot={{ r: 7, strokeWidth: 0 }}
                    />
                  ) : null}
                </LineChart>
              </ResponsiveContainer>
            </article>

            <article className="chart-panel">
              <div className="section-heading">
                <div>
                  <p className="eyebrow">Shared review</p>
                  <h2>Monthly return</h2>
                </div>
              </div>
              <ResponsiveContainer width="100%" height={secondaryChartHeight}>
                <LineChart data={traderMonthly} margin={responsiveChartMargin}>
                  <CartesianGrid strokeDasharray="4 6" stroke={chartGridColor} vertical={false} />
                  <XAxis
                    dataKey="label"
                    axisLine={false}
                    interval={xAxisInterval}
                    minTickGap={isPhoneViewport ? 18 : 5}
                    tickLine={false}
                    tick={{ fill: chartAxisColor, fontSize: isPhoneViewport ? 11 : 12 }}
                  />
                  <YAxis
                    axisLine={false}
                    tickLine={false}
                    tick={{ fill: chartAxisColor, fontSize: isPhoneViewport ? 11 : 12 }}
                    tickFormatter={(value) => `${Number(value)}%`}
                    width={isPhoneViewport ? 42 : 60}
                  />
                  <Tooltip contentStyle={chartTooltipStyle} formatter={(value) => pct(Number(value))} />
                  <Legend iconType="circle" wrapperStyle={chartLegendStyle} />
                  {showsTrader(chartFilter, "branden") ? (
                    <Line
                      type="monotone"
                      dataKey="brandenPercentReturn"
                      name="Branden"
                      stroke={brandenColor}
                      strokeWidth={3}
                      dot={{ r: 4, strokeWidth: 2 }}
                      activeDot={{ r: 7, strokeWidth: 0 }}
                    />
                  ) : null}
                  {showsTrader(chartFilter, "cam") ? (
                    <Line
                      type="monotone"
                      dataKey="camPercentReturn"
                      name="Cam"
                      stroke={camColor}
                      strokeWidth={3}
                      dot={{ r: 4, strokeWidth: 2 }}
                      activeDot={{ r: 7, strokeWidth: 0 }}
                    />
                  ) : null}
                </LineChart>
              </ResponsiveContainer>
            </article>

            <article className="chart-panel">
              <div className="section-heading">
                <div>
                  <p className="eyebrow">Edge quality</p>
                  <h2>Average R</h2>
                </div>
              </div>
              <ResponsiveContainer width="100%" height={secondaryChartHeight}>
                <BarChart
                  data={traderMonthly}
                  margin={responsiveChartMargin}
                  barGap={isPhoneViewport ? 3 : 8}
                  barCategoryGap={isPhoneViewport ? "18%" : "28%"}
                >
                  <CartesianGrid strokeDasharray="4 6" stroke={chartGridColor} vertical={false} />
                  <XAxis
                    dataKey="label"
                    axisLine={false}
                    interval={xAxisInterval}
                    minTickGap={isPhoneViewport ? 18 : 5}
                    tickLine={false}
                    tick={{ fill: chartAxisColor, fontSize: isPhoneViewport ? 11 : 12 }}
                  />
                  <YAxis
                    axisLine={false}
                    tickLine={false}
                    tick={{ fill: chartAxisColor, fontSize: isPhoneViewport ? 11 : 12 }}
                    width={isPhoneViewport ? 42 : 60}
                  />
                  <Tooltip contentStyle={chartTooltipStyle} />
                  <Legend iconType="circle" wrapperStyle={chartLegendStyle} />
                  {showsTrader(chartFilter, "branden") ? (
                    <Bar dataKey="brandenAvgR" name="Branden" fill={brandenColor} radius={[6, 6, 0, 0]} />
                  ) : null}
                  {showsTrader(chartFilter, "cam") ? (
                    <Bar dataKey="camAvgR" name="Cam" fill={camColor} radius={[6, 6, 0, 0]} />
                  ) : null}
                </BarChart>
              </ResponsiveContainer>
            </article>

            <article className="chart-panel">
              <div className="section-heading">
                <div>
                  <p className="eyebrow">Execution</p>
                  <h2>Win rate</h2>
                </div>
              </div>
              <ResponsiveContainer width="100%" height={secondaryChartHeight}>
                <BarChart
                  data={traderMonthly}
                  margin={responsiveChartMargin}
                  barGap={isPhoneViewport ? 3 : 8}
                  barCategoryGap={isPhoneViewport ? "18%" : "28%"}
                >
                  <CartesianGrid strokeDasharray="4 6" stroke={chartGridColor} vertical={false} />
                  <XAxis
                    dataKey="label"
                    axisLine={false}
                    interval={xAxisInterval}
                    minTickGap={isPhoneViewport ? 18 : 5}
                    tickLine={false}
                    tick={{ fill: chartAxisColor, fontSize: isPhoneViewport ? 11 : 12 }}
                  />
                  <YAxis
                    axisLine={false}
                    tickLine={false}
                    tick={{ fill: chartAxisColor, fontSize: isPhoneViewport ? 11 : 12 }}
                    tickFormatter={(value) => `${Number(value)}%`}
                    width={isPhoneViewport ? 42 : 60}
                  />
                  <Tooltip contentStyle={chartTooltipStyle} formatter={(value) => pct(Number(value))} />
                  <Legend iconType="circle" wrapperStyle={chartLegendStyle} />
                  {showsTrader(chartFilter, "branden") ? (
                    <Bar dataKey="brandenWinRate" name="Branden" fill={brandenColor} radius={[6, 6, 0, 0]} />
                  ) : null}
                  {showsTrader(chartFilter, "cam") ? (
                    <Bar dataKey="camWinRate" name="Cam" fill={camColor} radius={[6, 6, 0, 0]} />
                  ) : null}
                </BarChart>
              </ResponsiveContainer>
            </article>

            <article className="chart-panel">
              <div className="section-heading">
                <div>
                  <p className="eyebrow">Outcome</p>
                  <h2>Total Net R Multiple</h2>
                </div>
              </div>
              <ResponsiveContainer width="100%" height={secondaryChartHeight}>
                <BarChart
                  data={traderMonthly}
                  margin={responsiveChartMargin}
                  barGap={isPhoneViewport ? 3 : 8}
                  barCategoryGap={isPhoneViewport ? "18%" : "28%"}
                >
                  <CartesianGrid strokeDasharray="4 6" stroke={chartGridColor} vertical={false} />
                  <XAxis
                    dataKey="label"
                    axisLine={false}
                    interval={xAxisInterval}
                    minTickGap={isPhoneViewport ? 18 : 5}
                    tickLine={false}
                    tick={{ fill: chartAxisColor, fontSize: isPhoneViewport ? 11 : 12 }}
                  />
                  <YAxis
                    axisLine={false}
                    tickLine={false}
                    tick={{ fill: chartAxisColor, fontSize: isPhoneViewport ? 11 : 12 }}
                    tickFormatter={(value) => `${Number(value).toFixed(1)}R`}
                    width={isPhoneViewport ? 42 : 60}
                  />
                  <Tooltip contentStyle={chartTooltipStyle} formatter={(value) => `${Number(value).toFixed(2)}R`} />
                  <Legend iconType="circle" wrapperStyle={chartLegendStyle} />
                  {showsTrader(chartFilter, "branden") ? (
                    <Bar dataKey="brandenTotalR" name="Branden" fill={brandenColor} radius={[6, 6, 0, 0]} />
                  ) : null}
                  {showsTrader(chartFilter, "cam") ? (
                    <Bar dataKey="camTotalR" name="Cam" fill={camColor} radius={[6, 6, 0, 0]} />
                  ) : null}
                </BarChart>
              </ResponsiveContainer>
            </article>
          </section>

          <section className="history-section">
            <div className="section-heading">
              <div>
                <p className="eyebrow">Saved reports</p>
                <h2>Month-to-month history</h2>
              </div>
            </div>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Month</th>
                    <th>Trader</th>
                    <th>Net P&L</th>
                    <th>% Return</th>
                    <th>Avg R</th>
                    <th>Total net R</th>
                    <th>EV</th>
                    <th>Avg Win</th>
                    <th>Avg Loss</th>
                    <th>Trades</th>
                    <th>Win rate</th>
                    <th>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {reports.map((report) => (
                    <tr key={report.id}>
                      <td>{monthLabel(report.month)}</td>
                      <td>{report.userId}</td>
                      <td>{money(report.netPnl)}</td>
                      <td>{pct(report.percentReturn)}</td>
                      <td>{report.avgR.toFixed(2)}R</td>
                      <td>{report.totalR.toFixed(2)}R</td>
                      <td>{report.expectedValueR.toFixed(2)}R</td>
                      <td>{money(report.avgWin)}</td>
                      <td>{money(report.avgLoss)}</td>
                      <td>{report.totalTrades}</td>
                      <td>{pct(report.winRate)}</td>
                      <td>
                        {report.userId === user.id ? (
                          <div className="row-actions">
                            <button type="button" onClick={() => editReport(report)}>
                              Edit
                            </button>
                            <button className="danger-button" type="button" onClick={() => removeReport(report)}>
                              Delete
                            </button>
                          </div>
                        ) : (
                          <span className="muted">View only</span>
                        )}
                      </td>
                    </tr>
                  ))}
                  {!reports.length ? (
                    <tr>
                      <td colSpan={12} className="empty-cell">
                        No monthly reports saved yet.
                      </td>
                    </tr>
                  ) : null}
                </tbody>
              </table>
            </div>
            {myReports.length ? (
              <p className="history-note">
                You have {myReports.length} saved {myReports.length === 1 ? "month" : "months"}. Editing a month
                replaces that month&apos;s prior entry.
              </p>
            ) : null}
          </section>
        </div>
      ) : activeTab === "manage" ? (
        <section className="history-section">
          <div className="section-heading">
            <div>
              <p className="eyebrow">Edit entries</p>
              <h2>Update previous months</h2>
            </div>
            <button className="ghost-button" type="button" onClick={() => setActiveTab("entry")}>
              New entry
            </button>
          </div>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Month</th>
                  <th>Trader</th>
                  <th>Net P&L</th>
                  <th>% Return</th>
                  <th>Avg R</th>
                  <th>Total net R</th>
                  <th>EV</th>
                  <th>Avg Win</th>
                  <th>Avg Loss</th>
                  <th>Trades</th>
                  <th>Win rate</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {reports.map((report) => (
                  <tr key={report.id}>
                    <td>{monthLabel(report.month)}</td>
                    <td>{report.userId}</td>
                    <td>{money(report.netPnl)}</td>
                    <td>{pct(report.percentReturn)}</td>
                    <td>{report.avgR.toFixed(2)}R</td>
                    <td>{report.totalR.toFixed(2)}R</td>
                    <td>{report.expectedValueR.toFixed(2)}R</td>
                    <td>{money(report.avgWin)}</td>
                    <td>{money(report.avgLoss)}</td>
                    <td>{report.totalTrades}</td>
                    <td>{pct(report.winRate)}</td>
                    <td>
                      {report.userId === user.id ? (
                        <div className="row-actions">
                          <button type="button" onClick={() => editReport(report)}>
                            Edit
                          </button>
                          <button className="danger-button" type="button" onClick={() => removeReport(report)}>
                            Delete
                          </button>
                        </div>
                      ) : (
                        <span className="muted">View only</span>
                      )}
                    </td>
                  </tr>
                ))}
                {!reports.length ? (
                  <tr>
                    <td colSpan={12} className="empty-cell">
                      No monthly reports saved yet.
                    </td>
                  </tr>
                ) : null}
              </tbody>
            </table>
          </div>
          {myReports.length ? (
            <p className="history-note">
              You have {myReports.length} saved {myReports.length === 1 ? "month" : "months"}. Editing a month opens
              it in the data-entry tab and replaces that month&apos;s prior entry when saved.
            </p>
          ) : null}
        </section>
      ) : selectedTrade ? (
        <section
          className={
            selectedTrade.userId === "branden"
              ? isEmbeddedBrandenRoute()
                ? "branden-journal-content trade-detail-page"
                : "trade-log-shell branden-journal-shell branden-route-shell sidebar-expanded trade-detail-page"
              : "trade-log-shell trade-detail-page"
          }
        >
          {selectedTrade.userId === "branden" && !isEmbeddedBrandenRoute() ? <BrandenSidebar activeHref="/journal/branden/trade-log" /> : null}
          <div className={selectedTrade.userId === "branden" && !isEmbeddedBrandenRoute() ? "branden-journal-content" : undefined}>
          <div className="trade-detail-nav-layer" aria-hidden="false">
            <button
              className="trade-detail-nav-arrow trade-detail-nav-arrow-left"
              type="button"
              onClick={() => navigateSelectedTrade("previous")}
              disabled={!selectedTradeNavigation.previousTradeId}
              aria-label="Previous trade"
              title="Previous trade"
            >
              ‹
            </button>
            <button
              className="trade-detail-nav-arrow trade-detail-nav-arrow-right"
              type="button"
              onClick={() => navigateSelectedTrade("next")}
              disabled={!selectedTradeNavigation.nextTradeId}
              aria-label="Next trade"
              title="Next trade"
            >
              ›
            </button>
          </div>
          <div className="trade-log-header">
            <div>
              <p className="eyebrow">Trade detail</p>
              <h2>#{selectedTrade.symbol}</h2>
            </div>
            <button
              className="trade-muted-button"
              type="button"
              onClick={() => {
                if (selectedTrade.userId === "branden") {
                  const returnTo = new URLSearchParams(window.location.search).get("returnTo");
                  if (returnTo?.startsWith("/journal/branden/trade-log")) {
                    window.location.href = returnTo;
                    return;
                  }
                  window.location.href = selectedTrade.portfolioTag
                    ? `/journal/branden/trade-log?portfolio=${encodeURIComponent(selectedTrade.portfolioTag)}`
                    : "/journal/branden/trade-log";
                  return;
                }

                setSelectedTradeId("");
              }}
            >
              Back to {selectedTrade.userId === "branden" ? "Branden" : "Cam"} Log
            </button>
          </div>

          <form className="trade-detail-layout" onSubmit={saveTradeEdit}>
            <article className="trade-detail-hero">
              <div className="trade-detail-hero-summary">
                <div className="trade-detail-hero-badges">
                  <span className={tradeBadgeClass(editTradeMetrics.status)}>{editTradeMetrics.status}</span>
                  <span className={`side-pill ${editTradeForm.side.toLowerCase()}`}>{editTradeForm.side}</span>
                </div>
                <div className="trade-detail-hero-pnl">
                  <span>Net P&amp;L</span>
                  <strong className={editTradeMetrics.pnl >= 0 ? "trade-positive" : "trade-negative"}>
                    {money(editTradeMetrics.pnl)}
                  </strong>
                </div>
              </div>

              <div className="trade-detail-hero-details">
                <div className="trade-detail-hero-metrics">
                  <div>
                    <span>R multiple</span>
                    <strong>{editTradeMetrics.rMultiple.toFixed(2)}R</strong>
                  </div>
                  <div>
                    <span>{tradeReturnLabel(editTradeForm.symbol)}</span>
                    <strong>{editTradeDisplayReturn === null ? "—" : pct(editTradeDisplayReturn)}</strong>
                  </div>
                  <div>
                    <span>Risk</span>
                    <strong>{money(editTradeMetrics.risk)}</strong>
                  </div>
                </div>

                <div className="trade-detail-timeline">
                  <div>
                    <span>Entry</span>
                    <strong>{formatTradeDateTime(editTradeForm.entryDate, editTradeForm.openTime)}</strong>
                  </div>
                  <span className="trade-detail-timeline-arrow" aria-hidden="true">→</span>
                  <div>
                    <span>Exit</span>
                    <strong>
                      {editTradeForm.exitDate
                        ? formatTradeDateTime(editTradeForm.exitDate, editTradeForm.closeTime)
                        : "Still open"}
                    </strong>
                  </div>
                  <div className="trade-detail-days-held">
                    <span>Time held</span>
                    <strong>{editTradeMetrics.daysInTrade} {editTradeMetrics.daysInTrade === 1 ? "day" : "days"}</strong>
                  </div>
                </div>
              </div>

              {selectedTrade.userId === "cam" ? (
                <p className="trade-detail-hero-note">
                  Checklist: {checklistScore(editTradeForm.checklistItems, setupGradeBands(primarySetupName(editTradeForm.setupTags), setupTemplates)).earned}/
                  {checklistScore(editTradeForm.checklistItems, setupGradeBands(primarySetupName(editTradeForm.setupTags), setupTemplates)).total} points ·{" "}
                  {effectiveFormGrade(editTradeForm, setupTemplates)}
                </p>
              ) : editTradeForm.manualGrade ? (
                <p className="trade-detail-hero-note">Grade: {editTradeForm.manualGrade}</p>
              ) : null}
            </article>

            <div className="trade-detail-main">
              {selectedTradeExecutions.length ? (
                <article className="trade-detail-section">
                  <div className="trade-chart-heading">
                    <h3>Executions</h3>
                    <span>{selectedTradeExecutions.length} fills</span>
                  </div>
                  <p>These are the broker executions that make up this trade.</p>
                  <div className="trade-execution-table-wrap">
                    <table className="trade-execution-table">
                      <thead>
                        <tr>
                          <th>Fill type</th>
                          <th>Date</th>
                          <th>Price</th>
                          <th>Shares</th>
                          <th>P&L</th>
                          <th>Source</th>
                        </tr>
                      </thead>
                      <tbody>
                        {selectedTradeExecutions.map((execution) => (
                          <tr key={execution.id}>
                            <td>{execution.type === "ENTRY" ? "Entry" : "Exit"}</td>
                            <td>{execution.date || "-"}</td>
                            <td>{numberValue(execution.price) ? numberValue(execution.price).toFixed(2) : "-"}</td>
                            <td>{numberValue(execution.shares) || "-"}</td>
                            <td className={numberValue(execution.pnl) >= 0 ? "trade-positive" : "trade-negative"}>{money(execution.pnl)}</td>
                            <td>{execution.importSource === "cf-statement-pdf" ? "CF statement" : execution.importSource || "Manual"}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </article>
              ) : null}

              <article className="trade-detail-section">
                <div className="trade-section-heading-row">
                  <h3>Trade Fields</h3>
                  <span className="trade-review-required-note">
                    <span className="review-required-marker">*</span> Required field
                  </span>
                </div>
                <div className="trade-form-groups">
                  <div className="trade-field-block">
                    <h4>Trade</h4>
                    <div className="trade-form-grid">
                      <label>
                        Asset name
                        <input value={editTradeForm.symbol} disabled />
                      </label>
                      <label>
                        Side
                        <select
                          value={editTradeForm.side}
                          onChange={(event) => updateEditTradeField("side", event.target.value)}
                          disabled={selectedTrade.userId !== user.id}
                        >
                          <option value="LONG">Long</option>
                          <option value="SHORT">Short</option>
                        </select>
                      </label>
                      <label>
                        Entry date
                        <input
                          type="date"
                          value={editTradeForm.entryDate}
                          onChange={(event) => updateEditTradeField("entryDate", event.target.value)}
                          disabled={selectedTrade.userId !== user.id}
                        />
                      </label>
                      <label>
                        Exit date
                        <input
                          type="date"
                          value={editTradeForm.exitDate}
                          onChange={(event) => updateEditTradeField("exitDate", event.target.value)}
                          disabled={selectedTrade.userId !== user.id}
                        />
                      </label>
                      {selectedTrade.userId === "cam" ? (
                        <>
                          <label>
                            Open time
                            <input
                              type="time"
                              value={editTradeForm.openTime}
                              onChange={(event) => updateEditTradeField("openTime", event.target.value)}
                              disabled={selectedTrade.userId !== user.id}
                            />
                          </label>
                          <label>
                            Close time
                            <input
                              type="time"
                              value={editTradeForm.closeTime}
                              onChange={(event) => updateEditTradeField("closeTime", event.target.value)}
                              disabled={selectedTrade.userId !== user.id}
                            />
                          </label>
                        </>
                      ) : null}
                    </div>
                  </div>

                  <div className="trade-field-block">
                    <h4>Prices</h4>
                    <div className="trade-form-grid">
                      <label>
                        Avg entry
                        <input
                          type="number"
                          step="any"
                          inputMode="decimal"
                          value={String(editTradeForm.avgEntry)}
                          onChange={(event) => updateEditTradeField("avgEntry", event.target.value)}
                          disabled={selectedTrade.userId !== user.id}
                        />
                      </label>
                      <label>
                        Exit
                        <input
                          type="number"
                          step="any"
                          inputMode="decimal"
                          value={String(editTradeForm.exitPrice)}
                          onChange={(event) => updateEditTradeField("exitPrice", event.target.value)}
                          disabled={selectedTrade.userId !== user.id}
                        />
                      </label>
                      <label>
                        Stop
                        <input
                          type="number"
                          step="any"
                          inputMode="decimal"
                          value={String(editTradeForm.stopPrice)}
                          onChange={(event) => updateEditTradeField("stopPrice", event.target.value)}
                          disabled={selectedTrade.userId !== user.id}
                        />
                      </label>
                      <label>
                        Take profit
                        <input
                          type="number"
                          step="any"
                          inputMode="decimal"
                          value={String(editTradeForm.takeProfitPrice)}
                          onChange={(event) => updateEditTradeField("takeProfitPrice", event.target.value)}
                          disabled={selectedTrade.userId !== user.id}
                        />
                      </label>
                    </div>
                  </div>

                  <div className="trade-field-block">
                    <h4>Risk &amp; result</h4>
                    <div className="trade-form-grid">
                      <label>
                        Number of shares
                        <input
                          type="number"
                          step="any"
                          inputMode="decimal"
                          value={String(editTradeForm.shares)}
                          onChange={(event) => updateEditTradeField("shares", event.target.value)}
                          disabled={selectedTrade.userId !== user.id}
                        />
                      </label>
                      <label>
                        Risk $ <span className="review-required-marker" title="Required field">*</span>
                        <input
                          type="number"
                          step="any"
                          inputMode="decimal"
                          value={String(editTradeForm.risk)}
                          onChange={(event) => updateEditTradeField("risk", event.target.value)}
                          disabled={selectedTrade.userId !== user.id}
                        />
                      </label>
                      <label>
                        Net P&amp;L
                        <input
                          type="number"
                          step="any"
                          inputMode="decimal"
                          value={String(editTradeForm.pnl)}
                          onChange={(event) => updateEditTradeField("pnl", event.target.value)}
                          disabled={selectedTrade.userId !== user.id}
                        />
                      </label>
                      <label>
                        Commission
                        <input
                          type="number"
                          step="any"
                          inputMode="decimal"
                          value={String(editTradeForm.commission)}
                          onChange={(event) => updateEditTradeField("commission", event.target.value)}
                          disabled={selectedTrade.userId !== user.id}
                        />
                      </label>
                      <label>
                        Used margin
                        <input
                          type="number"
                          step="any"
                          inputMode="decimal"
                          value={String(editTradeForm.usedMargin)}
                          onChange={(event) => updateEditTradeField("usedMargin", event.target.value)}
                          disabled={selectedTrade.userId !== user.id}
                        />
                      </label>
                      {selectedTrade.userId === user.id ? (
                        <div className="trade-inline-actions trade-form-wide">
                          <button
                            className="trade-muted-button"
                            type="button"
                            onClick={() => {
                              const stop = calculatedStopPrice(editTradeForm);
                              if (stop) {
                                updateEditTradeField("stopPrice", Number(stop.toFixed(4)));
                              }
                            }}
                          >
                            Calculate stop from risk
                          </button>
                          <button
                            className="trade-muted-button"
                            type="button"
                            onClick={() => updateEditTradeField("manualGrade", effectiveFormGrade(editTradeForm, setupTemplates))}
                          >
                            Grade from setup
                          </button>
                        </div>
                      ) : null}
                    </div>
                  </div>

                  <div className="trade-field-block">
                    <h4>Review</h4>
                    <div className="trade-form-grid trade-review-field-grid">
                      <label>
                        Setup <span className="review-required-marker" title="Required to clear Needs Review">*</span>
                        <select
                          value={primarySetupName(editTradeForm.setupTags)}
                          onChange={(event) => updateTradeSetupTags("edit", event.target.value)}
                          disabled={selectedTrade.userId !== user.id}
                        >
                          {setupAssignmentOptions(primarySetupName(editTradeForm.setupTags)).map((option) => (
                            <option key={option || "__none__"} value={option}>
                              {option || "No setup"}
                            </option>
                          ))}
                        </select>
                      </label>
                      <label>
                        Grade
                        <input
                          value={editTradeForm.manualGrade}
                          onChange={(event) => updateEditTradeField("manualGrade", event.target.value)}
                          placeholder="A, B+, C"
                          disabled={selectedTrade.userId !== user.id}
                        />
                      </label>
                      <label className="trade-form-wide">
                        Mistake
                        {renderMistakeTagField("edit", editTradeForm.mistakeTags, selectedTrade.userId !== user.id)}
                      </label>
                      {selectedTrade.userId === "cam" ? (
                        <>
                          <label>
                            Portfolio tag
                            <input
                              list="branden-portfolio-options"
                              value={editTradeForm.portfolioTag}
                              onChange={(event) => updateEditTradeField("portfolioTag", event.target.value)}
                              placeholder="Main, IRA, Cash"
                              disabled={selectedTrade.userId !== user.id}
                            />
                          </label>
                          <label>
                            Emotion
                            <input
                              value={editTradeForm.emotion}
                              onChange={(event) => updateEditTradeField("emotion", event.target.value)}
                              placeholder="Calm, rushed, not entered"
                              disabled={selectedTrade.userId !== user.id}
                            />
                          </label>
                          <label>
                            Thesis / trade quality
                            <input
                              value={editTradeForm.tradeQuality}
                              onChange={(event) => updateEditTradeField("tradeQuality", event.target.value)}
                              placeholder="A+ thesis, chased, late exit"
                              disabled={selectedTrade.userId !== user.id}
                            />
                          </label>
                          <label>
                            Custom tags
                            <input
                              value={editTradeForm.customTags}
                              onChange={(event) => updateEditTradeField("customTags", event.target.value)}
                              placeholder="Earnings, Morning"
                              disabled={selectedTrade.userId !== user.id}
                            />
                          </label>
                          <label>
                            Add screenshots
                            <input
                              type="file"
                              accept="image/*"
                              multiple
                              onChange={(event) => attachEditTradeScreenshots(event.target.files)}
                              disabled={selectedTrade.userId !== user.id}
                            />
                          </label>
                        </>
                      ) : null}
                      <label className="trade-form-wide">
                        Chart links <span className="review-required-marker" title="A screenshot or chart link is required to clear Needs Review">*</span>
                        <input
                          value={editTradeForm.chartLinks.join(", ")}
                          onChange={(event) =>
                            updateEditTradeField(
                              "chartLinks",
                              event.target.value
                                .split(",")
                                .map((link) => link.trim())
                                .filter(Boolean)
                            )
                          }
                          placeholder="https://www.tradingview.com/x/..."
                          disabled={selectedTrade.userId !== user.id}
                        />
                      </label>
                    </div>
                  </div>
                </div>
                <div className="trade-review-row-list">
                  {([
                    ["setup", "Setup"],
                    ["entry", "Entry"],
                    ["exit", "Exit"],
                    ["didRight", "What did I do right"],
                    ["didWrong", "What did I do wrong"],
                    ["general", "General review"]
                  ] as Array<[keyof TradeReviewSections, string]>).map(([key, label]) => (
                    <label key={key}>
                      {label}
                      <textarea
                        className="trade-review-row-textarea"
                        rows={2}
                        value={editTradeForm.reviewSections[key]}
                        onChange={(event) => updateTradeReviewField("edit", key, event.target.value)}
                        disabled={selectedTrade.userId !== user.id}
                      />
                    </label>
                  ))}
                  {editTradeForm.notes ? (
                    <label>
                      Legacy notes (preserved)
                      <textarea
                        className="trade-review-row-textarea"
                        rows={3}
                        value={editTradeForm.notes}
                        onChange={(event) => updateEditTradeField("notes", event.target.value)}
                        disabled={selectedTrade.userId !== user.id}
                      />
                    </label>
                  ) : null}
                </div>
              </article>

	              <article className="trade-detail-section">
	                <div className="trade-section-heading-row">
	                  <h3>Screenshots</h3>
                    <span className="trade-review-required-note">
                      <span className="review-required-marker">*</span> screenshot or chart link required
                    </span>
	                  {selectedTrade.userId === user.id ? (
	                    <label className="trade-screenshot-upload">
	                      Upload
	                      <input
	                        type="file"
	                        accept="image/*"
	                        multiple
	                        onChange={(event) => {
	                          attachEditTradeScreenshots(event.target.files);
	                          event.currentTarget.value = "";
	                        }}
	                      />
	                    </label>
	                  ) : null}
	                </div>
	                {editTradeForm.screenshots.length || editTradeForm.chartLinks.length ? (
                  <div className="trade-large-screenshots">
                    {editTradeForm.screenshots.map((screenshot, index) => (
                      <div key={`${screenshot.slice(0, 32)}-${index}`} className="trade-edit-screenshot">
                        <button
                          className="trade-screenshot-preview"
                          type="button"
                          onClick={() =>
                            setFullscreenScreenshot({
                              src: screenshot,
                              alt: `${editTradeForm.symbol || selectedTrade.symbol} screenshot ${index + 1}`
                            })
                          }
                        >
                          <img
                            src={screenshot}
                            alt={`${editTradeForm.symbol || selectedTrade.symbol} screenshot ${index + 1}`}
                            loading="lazy"
                            decoding="async"
                          />
                          <span>View full screen</span>
                        </button>
                        {selectedTrade.userId === user.id ? (
                          <button
                            className="trade-danger-button"
                            type="button"
                            onClick={() =>
                              setEditTradeForm((current) => ({
                                ...current,
                                screenshots: current.screenshots.filter((_, imageIndex) => imageIndex !== index)
                              }))
                            }
                          >
                            Remove image
                          </button>
                        ) : null}
                      </div>
                    ))}
                    {editTradeForm.chartLinks.map((link, index) => (
                      <div key={`${link}-${index}`} className="trade-edit-screenshot trade-chart-link-preview">
                        <a href={link} target="_blank" rel="noreferrer">
                          <img
                            src={link}
                            alt={`${editTradeForm.symbol || selectedTrade.symbol} chart link ${index + 1}`}
                            loading="lazy"
                            decoding="async"
                          />
                          <span>Open chart link</span>
                        </a>
                        {selectedTrade.userId === user.id ? (
                          <button
                            className="trade-danger-button"
                            type="button"
                            onClick={() =>
                              setEditTradeForm((current) => ({
                                ...current,
                                chartLinks: current.chartLinks.filter((_, linkIndex) => linkIndex !== index)
                              }))
                            }
                          >
                            Remove link
                          </button>
                        ) : null}
                      </div>
                    ))}
                  </div>
                ) : (
                  <p>No screenshots or chart links attached.</p>
                )}
              </article>

              {renderChecklistEditor("edit", editTradeForm, selectedTrade.userId !== user.id)}
            </div>

            <aside className="trade-detail-aside">
              <article>
                <h3>Execution</h3>
                <dl>
	                  <div>
	                    <dt>Avg entry</dt>
	                    <dd>{editTradeMetrics.avgEntry.toFixed(2)}</dd>
	                  </div>
	                  <div>
	                    <dt>Exit</dt>
	                    <dd>{editTradeMetrics.exitPrice ? editTradeMetrics.exitPrice.toFixed(2) : "-"}</dd>
	                  </div>
	                  <div>
	                    <dt>Stop</dt>
	                    <dd>{numberValue(editTradeForm.stopPrice) ? numberValue(editTradeForm.stopPrice).toFixed(2) : "-"}</dd>
	                  </div>
	                  <div>
	                    <dt>Take profit</dt>
	                    <dd>{numberValue(editTradeForm.takeProfitPrice) ? numberValue(editTradeForm.takeProfitPrice).toFixed(2) : "-"}</dd>
	                  </div>
	                  <div>
	                    <dt>Shares</dt>
	                    <dd>{editTradeMetrics.shares}</dd>
	                  </div>
	                  <div>
	                    <dt>Commission</dt>
	                    <dd>{numberValue(editTradeForm.commission) ? money(numberValue(editTradeForm.commission)) : "-"}</dd>
	                  </div>
	                  <div>
	                    <dt>Used margin</dt>
	                    <dd>{numberValue(editTradeForm.usedMargin) ? money(numberValue(editTradeForm.usedMargin)) : "-"}</dd>
	                  </div>
	                  <div>
	                    <dt>Cost</dt>
	                    <dd>{money(editTradeMetrics.avgEntry * editTradeMetrics.shares)}</dd>
	                  </div>
                </dl>
              </article>

              <article>
	                <h3>Tags</h3>
	                {renderCustomTagField("edit", editTradeForm.customTags, selectedTrade.userId !== user.id)}
              </article>

              <article>
                <h3>Ownership</h3>
                <dl>
                  <div>
                    <dt>Trader</dt>
                    <dd>{selectedTrade.userId}</dd>
                  </div>
                  <div>
                    <dt>Logged</dt>
                    <dd>{new Date(selectedTrade.createdAt).toLocaleDateString("en-US")}</dd>
                  </div>
                </dl>
	                {selectedTrade.userId === user.id ? (
                  <div className="trade-detail-actions">
                    <button type="submit">Save changes</button>
                    <button className="trade-muted-button" type="button" onClick={() => setEditTradeForm(tradeToForm(selectedTrade, setupTemplates))}>
                      Reset changes
                    </button>
                    <button className="trade-muted-button" type="button" onClick={() => updateTradeVisibility([selectedTrade], true)}>
                      Hide trade
                    </button>
                    <button className="trade-danger-button" type="button" onClick={() => removeTrade(selectedTrade)}>
                      Delete trade
                    </button>
                    {status ? <span className="status">{status}</span> : null}
                  </div>
	                ) : (
                  <p>View only.</p>
                )}
	              </article>
	            </aside>

            <article className="trade-detail-section trade-price-chart-section">
              <div className="trade-chart-heading">
                <h3>{editTradeForm.symbol ? `${editTradeForm.symbol.toUpperCase()} execution chart` : "Execution chart"}</h3>
                <span>Chart</span>
              </div>
              <TradePriceChart
                symbol={editTradeForm.symbol}
                side={editTradeForm.side}
                entryDate={editTradeForm.entryDate}
                exitDate={editTradeForm.exitDate}
                avgEntry={numberValue(editTradeForm.avgEntry)}
                exitPrice={numberValue(editTradeForm.exitPrice)}
                stopPrice={numberValue(editTradeForm.stopPrice)}
                takeProfitPrice={numberValue(editTradeForm.takeProfitPrice)}
              />
            </article>
	          </form>
          </div>
	        </section>
      ) : isBrandenJournalMode ? (
        <section className="trade-log-shell branden-journal-shell branden-route-shell sidebar-expanded">
          <BrandenSidebar
            activeHref="/journal/branden/dashboard"
            accountActions={
              canEditActiveTradeLog
                ? [
                    {
                      key: "import-broker-statement",
                      label: "Import broker statement",
                      icon: "I",
                      disabled: Boolean(busyProgress),
                      onClick: async () => {
                        const targetPortfolio = await choosePortfolioForImport("broker statement import");
                        if (targetPortfolio) {
                          cfImportInputRef.current?.click();
                        }
                      }
                    }
                  ]
                : []
            }
          />

	          <div className="branden-journal-content" ref={brandenDashboardRef}>

            {status ? <p className="status trade-log-status">{status}</p> : null}

          <input
            ref={excelImportInputRef}
            className="trade-file-input"
            type="file"
            accept=".xlsx,.xls,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet,application/vnd.ms-excel"
	            disabled={Boolean(busyProgress) || !canEditActiveTradeLog}
            onChange={(event) => {
              importExcelTradeLog(event.target.files);
              event.currentTarget.value = "";
            }}
          />
          <input
            ref={cfImportInputRef}
            className="trade-file-input"
            type="file"
            accept="application/pdf,.pdf"
            disabled={Boolean(busyProgress) || !canEditActiveTradeLog}
            onChange={(event) => {
              importCfStatement(event.target.files);
              event.currentTarget.value = "";
            }}
          />
          <input
            ref={journalBackupInputRef}
            className="trade-file-input"
            type="file"
            accept="application/json,.json"
            disabled={Boolean(busyProgress) || !canEditActiveTradeLog}
            onChange={(event) => {
              void importBrandenJournalBackup(event.target.files);
              event.currentTarget.value = "";
            }}
          />

          <div className="trade-date-filters trade-toolbar-filters">
                <label>
                  Portfolio view
                  <select value={activePortfolio} onChange={(event) => setActivePortfolio(event.target.value)}>
                    <option value="">All portfolios</option>
                    {portfolioOptions.map((portfolio) => (
                      <option key={portfolio} value={portfolio}>
                        {portfolio}
                      </option>
                    ))}
                  </select>
                </label>
                <label>
                  Start date
                  <input
                    type="date"
                    value={tradeFilters.startDate}
                    onChange={(event) => setTradeFilters({ ...tradeFilters, startDate: event.target.value })}
                  />
                </label>
	                <label>
	                  End date
	                  <input
                    type="date"
                    value={tradeFilters.endDate}
	                    onChange={(event) => setTradeFilters({ ...tradeFilters, endDate: event.target.value })}
	                  />
	                </label>
	                <div className="trade-date-quick-filters">
	                  <button
	                    className="trade-muted-button"
	                    type="button"
	                    onClick={() =>
	                      setTradeFilters((current) => ({
	                        ...current,
	                        startDate: daysAgoDate(29),
	                        endDate: currentDate()
	                      }))
	                    }
	                  >
	                    Last 30 days
	                  </button>
	                  <button
	                    className="trade-muted-button"
	                    type="button"
	                    onClick={() =>
	                      setTradeFilters((current) => ({
	                        ...current,
	                        startDate: currentMonthStartDate(),
	                        endDate: currentDate()
	                      }))
	                    }
	                  >
	                    Month to date
	                  </button>
	                  <button
	                    className="trade-muted-button"
	                    type="button"
	                    onClick={() =>
	                      setTradeFilters((current) => ({
	                        ...current,
	                        startDate: currentYearStartDate(),
	                        endDate: currentDate()
	                      }))
	                    }
	                  >
	                    Year to date
	                  </button>
	                </div>
	              </div>
              <datalist id="branden-portfolio-options">
                {portfolioOptions.map((portfolio) => (
                  <option key={portfolio} value={portfolio} />
                ))}
              </datalist>

              <div className="trade-chart-rows">
                <div className="trade-chart-row trade-chart-row-two">
                  <article className="trade-chart-panel top-chart">
                  <div className="trade-chart-heading">
                    <h3>Settled P&amp;L</h3>
                    <span>i</span>
                  </div>
                  <strong className={tradeSummary.netPnl >= 0 ? "trade-positive" : "trade-negative"}>
                    {money(tradeSummary.netPnl)}
                  </strong>
                  <ResponsiveContainer width="100%" height={320}>
                    <AreaChart data={tradePnlChartData} margin={{ top: 18, right: 12, bottom: 8, left: 4 }}>
                      <defs>
                        <linearGradient id="settledPnlPositiveGradient" x1="0" x2="0" y1="0" y2="1">
                          <stop offset="0%" stopColor="#4f7045" stopOpacity={0.78} />
                          <stop offset="72%" stopColor="#83b56d" stopOpacity={0.28} />
                          <stop offset="100%" stopColor="#dff2d8" stopOpacity={0.06} />
                        </linearGradient>
                        <linearGradient id="settledPnlNegativeGradient" x1="0" x2="0" y1="0" y2="1">
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
                      <Area type="monotone" dataKey="cumulativePnlPositive" name="Positive P&L" stroke="none" fill="url(#settledPnlPositiveGradient)" baseValue={0} dot={false} activeDot={false} isAnimationActive={false} />
                      <Area type="monotone" dataKey="cumulativePnlNegative" name="Negative P&L" stroke="none" fill="url(#settledPnlNegativeGradient)" baseValue={0} dot={false} activeDot={false} isAnimationActive={false} />
                      <Area type="monotone" dataKey="cumulativePnl" name="Cumulative P&L" stroke="#4f7045" strokeWidth={2.8} fill="transparent" dot={false} activeDot={{ r: 5, strokeWidth: 0, fill: "#4f7045" }} />
                    </AreaChart>
                  </ResponsiveContainer>
                </article>

                  <article className="trade-chart-panel top-chart">
                    <div className="trade-chart-heading">
                      <h3>Total R return by trade date</h3>
                      <span>R</span>
                    </div>
                    <strong className={tradeSummary.totalR >= 0 ? "trade-positive" : "trade-negative"}>
                      {tradeSummary.totalR.toFixed(2)}R
                    </strong>
                    <ResponsiveContainer width="100%" height={320}>
                      <AreaChart data={tradeDatePerformanceData} margin={{ top: 18, right: 12, bottom: 8, left: 4 }}>
                        <defs>
                          <linearGradient id="filteredRGradient" x1="0" x2="0" y1="0" y2="1">
                            <stop offset="0%" stopColor="#6f8f5f" stopOpacity={0.86} />
                            <stop offset="70%" stopColor="#6f8f5f" stopOpacity={0.34} />
                            <stop offset="100%" stopColor="#6f8f5f" stopOpacity={0.04} />
                          </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 5" stroke="rgba(47, 53, 45, 0.16)" vertical={false} />
                        <XAxis dataKey="label" axisLine={false} tickLine={false} tick={{ fill: "#6f7469", fontSize: 11 }} minTickGap={22} />
                        <YAxis axisLine={false} tickLine={false} tick={{ fill: "#6f7469", fontSize: 11 }} tickFormatter={(value) => `${Number(value).toFixed(1)}R`} width={54} />
                        <Tooltip
                          contentStyle={chartTooltipStyle}
                          formatter={(value, name) => [
                            `${Number(value).toFixed(2)}R`,
                            name === "cumulativeR" ? "Cumulative R" : "Daily R"
                          ]}
                          labelFormatter={(label) => `Trade date: ${label}`}
                        />
                        <Area type="monotone" dataKey="cumulativeR" name="Cumulative R" stroke="#4f7045" strokeWidth={2.4} fill="url(#filteredRGradient)" dot={false} activeDot={{ r: 5, strokeWidth: 0 }} />
                        <Line type="monotone" dataKey="totalR" name="Daily R" stroke="#8c6a4a" strokeWidth={1.8} dot={false} />
                      </AreaChart>
                    </ResponsiveContainer>
                  </article>
                </div>

                <div className="trade-chart-row trade-chart-row-one">
                  <article className="trade-chart-panel top-chart">
                    <div className="trade-chart-heading">
                      <h3>R distribution</h3>
                      <span>R</span>
                    </div>
                    <ResponsiveContainer width="100%" height={350}>
                      <BarChart data={rDistributionData} margin={{ top: 24, right: 16, bottom: 8, left: -18 }}>
                        <CartesianGrid strokeDasharray="3 5" stroke="rgba(47, 53, 45, 0.16)" vertical={false} />
                        <XAxis dataKey="bucket" axisLine={false} tickLine={false} tick={{ fill: "#6f7469", fontSize: 11 }} interval={0} />
                        <YAxis allowDecimals={false} axisLine={false} tickLine={false} tick={{ fill: "#6f7469", fontSize: 11 }} />
                        <Tooltip contentStyle={chartTooltipStyle} />
                        <Bar dataKey="trades" fill="#8c6a4a" radius={[6, 6, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </article>
                </div>

                <div className="trade-chart-row trade-chart-row-two">
                  <article className="trade-chart-panel top-chart trade-score-card">
                    <div className="trade-chart-heading">
                      <h3>Trade score radar chart</h3>
                      <span>100</span>
                    </div>
                    <ResponsiveContainer width="100%" height={320}>
                      <RadarChart data={tradeScoreData.data} outerRadius="72%">
                        <PolarGrid stroke="rgba(111, 116, 105, 0.28)" />
                        <PolarAngleAxis
                          dataKey="metric"
                          tick={{ fill: "#6f7469", fontSize: 12, fontWeight: 800 }}
                        />
                        <PolarRadiusAxis
                          angle={90}
                          domain={[0, 100]}
                          tick={false}
                          axisLine={false}
                        />
                        <Radar
                          name="Trade score"
                          dataKey="score"
                          stroke="#6f5bd4"
                          fill="#8f7cf2"
                          fillOpacity={0.34}
                          strokeWidth={2.5}
                        />
                        <Tooltip
                          contentStyle={chartTooltipStyle}
                          formatter={(value) => `${Number(value).toFixed(1)} / 100`}
                        />
                      </RadarChart>
                    </ResponsiveContainer>
                    <div className="trade-score-footer">
                      <div>
                        <span>Your Trade Score</span>
                        <strong>{tradeScoreData.totalScore.toFixed(1)}</strong>
                      </div>
                    </div>
                  </article>

                  <article className="trade-chart-panel top-chart">
                    <div className="trade-chart-heading">
                      <h3>R by checklist grade</h3>
                      <span>G</span>
                    </div>
                    <ResponsiveContainer width="100%" height={320}>
                      <BarChart data={gradePerformanceData} margin={{ top: 24, right: 16, bottom: 8, left: -12 }}>
                        <CartesianGrid strokeDasharray="3 5" stroke="rgba(47, 53, 45, 0.16)" vertical={false} />
                        <XAxis dataKey="grade" axisLine={false} tickLine={false} tick={{ fill: "#6f7469", fontSize: 11 }} />
                        <YAxis axisLine={false} tickLine={false} tick={{ fill: "#6f7469", fontSize: 11 }} tickFormatter={(value) => `${Number(value).toFixed(1)}R`} width={46} />
                        <Tooltip
                          contentStyle={chartTooltipStyle}
                          formatter={(value, name) => [
                            name === "avgR" ? `${Number(value).toFixed(2)}R` : String(value),
                            name === "avgR" ? "Avg R" : "Trades"
                          ]}
                        />
                        <Bar dataKey="avgR" name="Avg R" fill="#6f8f5f" radius={[6, 6, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </article>
                </div>

              </div>
          {isTradeModalOpen ? (
            <div className="trade-modal-backdrop" role="dialog" aria-modal="true" aria-label="Add trade">
              <form className="trade-entry-form trade-modal" onSubmit={saveTrade}>
                <div className="trade-panel-heading">
                  <div>
                    <h3>Add trade</h3>
                    <span>
                      {tradeMetrics.status} / {tradeMetrics.rMultiple.toFixed(2)}R / {money(tradeMetrics.pnl)}
                    </span>
                  </div>
                  <button className="trade-muted-button" type="button" onClick={() => setIsTradeModalOpen(false)}>
                    Close
                  </button>
                </div>
                <div className="trade-form-grid">
                  <label>
                    Asset name
                    <input value={tradeForm.symbol} onChange={(event) => updateTradeField("symbol", event.target.value)} placeholder="AAPL" />
                  </label>
                  <label>
                    Side
                    <select value={tradeForm.side} onChange={(event) => updateTradeField("side", event.target.value)}>
                      <option value="LONG">Long</option>
                      <option value="SHORT">Short</option>
                    </select>
                  </label>
                  <label>
                    Entry date
                    <input type="date" value={tradeForm.entryDate} onChange={(event) => updateTradeField("entryDate", event.target.value)} />
                  </label>
                  <label>
                    Open time
                    <input type="time" value={tradeForm.openTime} onChange={(event) => updateTradeField("openTime", event.target.value)} />
                  </label>
                  <label>
                    Exit date
                    <input type="date" value={tradeForm.exitDate} onChange={(event) => updateTradeField("exitDate", event.target.value)} />
                  </label>
                  <label>
                    Close time
                    <input type="time" value={tradeForm.closeTime} onChange={(event) => updateTradeField("closeTime", event.target.value)} />
                  </label>
                  <label>
                    Avg entry
                    <input type="number" step="any" inputMode="decimal" value={String(tradeForm.avgEntry)} onChange={(event) => updateTradeField("avgEntry", event.target.value)} />
                  </label>
                  <label>
                    Exit
                    <input type="number" step="any" inputMode="decimal" value={String(tradeForm.exitPrice)} onChange={(event) => updateTradeField("exitPrice", event.target.value)} />
                  </label>
                  <label>
                    Stop
                    <input type="number" step="any" inputMode="decimal" value={String(tradeForm.stopPrice)} onChange={(event) => updateTradeField("stopPrice", event.target.value)} />
                  </label>
                  <label>
                    Take profit
                    <input type="number" step="any" inputMode="decimal" value={String(tradeForm.takeProfitPrice)} onChange={(event) => updateTradeField("takeProfitPrice", event.target.value)} />
                  </label>
                  <label>
                    Number of shares
                    <input type="number" step="any" inputMode="decimal" value={String(tradeForm.shares)} onChange={(event) => updateTradeField("shares", event.target.value)} />
                  </label>
                  <label>
                    Risk $
                    <input type="number" step="any" inputMode="decimal" value={String(tradeForm.risk)} onChange={(event) => updateTradeField("risk", event.target.value)} />
                  </label>
                  <label>
                    Commission
                    <input type="number" step="any" inputMode="decimal" value={String(tradeForm.commission)} onChange={(event) => updateTradeField("commission", event.target.value)} />
                  </label>
                  <label>
                    Used margin
                    <input type="number" step="any" inputMode="decimal" value={String(tradeForm.usedMargin)} onChange={(event) => updateTradeField("usedMargin", event.target.value)} />
                  </label>
                  <label>
                    Net P&L
                    <input type="number" step="any" inputMode="decimal" value={String(tradeForm.pnl)} onChange={(event) => updateTradeField("pnl", event.target.value)} />
                  </label>
                  <label>
                    Portfolio tag
                    <input
                      list="branden-portfolio-options"
                      value={tradeForm.portfolioTag}
                      onChange={(event) => updateTradeField("portfolioTag", event.target.value)}
                      placeholder="Main, IRA, Cash"
                    />
                  </label>
                  <label>
                    Emotion
                    <input value={tradeForm.emotion} onChange={(event) => updateTradeField("emotion", event.target.value)} placeholder="Calm, rushed, not entered" />
                  </label>
                  <label>
                    Thesis / trade quality
                    <input value={tradeForm.tradeQuality} onChange={(event) => updateTradeField("tradeQuality", event.target.value)} placeholder="A+ thesis, chased, late exit" />
                  </label>
                  <label>
                    Setup
                    <select value={primarySetupName(tradeForm.setupTags)} onChange={(event) => updateTradeSetupTags("create", event.target.value)}>
                      {setupAssignmentOptions(primarySetupName(tradeForm.setupTags)).map((option) => (
                        <option key={option || "__none__"} value={option}>
                          {option || "No setup"}
                        </option>
                      ))}
                    </select>
                  </label>
                  <label>
                    Mistake tags
                    {renderMistakeTagField("create", tradeForm.mistakeTags)}
                  </label>
                  <label>
                    Custom tags
                    {renderCustomTagField("create", tradeForm.customTags)}
                  </label>
                  <label>
                    Grade
                    <input value={tradeForm.manualGrade} onChange={(event) => updateTradeField("manualGrade", event.target.value)} placeholder="A, B+, C" />
                  </label>
                  <label>
                    Screenshots
                    <input type="file" accept="image/*" multiple onChange={(event) => attachTradeScreenshots(event.target.files)} />
                  </label>
                  <label>
                    Chart links
                    <input
                      value={tradeForm.chartLinks.join(", ")}
                      onChange={(event) =>
                        updateTradeField(
                          "chartLinks",
                          event.target.value
                            .split(",")
                            .map((link) => link.trim())
                            .filter(Boolean)
                        )
                      }
                      placeholder="https://www.tradingview.com/x/..."
                    />
                  </label>
                  <div className="trade-inline-actions">
                    <button
                      className="trade-muted-button"
                      type="button"
                      onClick={() => {
                        const stop = calculatedStopPrice(tradeForm);
                        if (stop) {
                          updateTradeField("stopPrice", Number(stop.toFixed(4)));
                        }
                      }}
                    >
                      Calculate stop from risk
                    </button>
                    <button
                      className="trade-muted-button"
                      type="button"
                      onClick={() => updateTradeField("manualGrade", effectiveFormGrade(tradeForm, setupTemplates))}
                    >
                      Grade from setup
                    </button>
                  </div>
                </div>
                {renderChecklistEditor("create", tradeForm)}
                <div className="trade-review-row-list">
                  {([
                    ["setup", "Setup"],
                    ["entry", "Entry"],
                    ["exit", "Exit"],
                    ["didRight", "What did I do right"],
                    ["didWrong", "What did I do wrong"],
                    ["general", "General review"]
                  ] as Array<[keyof TradeReviewSections, string]>).map(([key, label]) => (
                    <label key={key}>
                      {label}
                      <textarea
                        className="trade-review-row-textarea"
                        rows={2}
                        value={tradeForm.reviewSections[key]}
                        onChange={(event) => updateTradeReviewField("create", key, event.target.value)}
                      />
                    </label>
                  ))}
                </div>
                {tradeForm.screenshots.length ? (
                  <div className="trade-screenshot-strip">
                    {tradeForm.screenshots.map((screenshot, index) => (
                      <img
                        key={`${screenshot.slice(0, 32)}-${index}`}
                        src={screenshot}
                        alt={`Trade upload ${index + 1}`}
                        loading="lazy"
                        decoding="async"
                      />
                    ))}
                  </div>
                ) : null}
                <div className="form-actions">
                  <button type="submit">Save trade</button>
                  <button className="trade-muted-button" type="button" onClick={() => setTradeForm(emptyTradeForm)}>
                    Clear
                  </button>
                  {status ? <span className="status">{status}</span> : null}
                </div>
              </form>
            </div>
          ) : null}
          </div>
        </section>
      ) : (
        <section className="trade-journal-launcher">
          <div className="section-heading">
            <div>
              <p className="eyebrow">Trade logging</p>
              <h2>Open the journal you need</h2>
            </div>
          </div>
          <div className="journal-launcher-grid">
            <article className="journal-launch-card">
              <p className="eyebrow">Branden</p>
              <h3>Trade journal</h3>
              <a className="journal-launch-button" href="/branden-journal">
                Open Branden journal
              </a>
            </article>
            <article className="journal-launch-card cam">
              <p className="eyebrow">Cam</p>
              <h3>Cam&apos;s exact journal app</h3>
              <a className="journal-launch-button" href="/cam-journal/index.html">
                Open Cam journal
              </a>
            </article>
          </div>
        </section>
      )}
      {fullscreenScreenshot ? (
        <div
          className="screenshot-lightbox-backdrop"
          role="dialog"
          aria-modal="true"
          aria-label={fullscreenScreenshot.alt}
          onClick={() => setFullscreenScreenshot(null)}
        >
          <button
            className="screenshot-lightbox-close"
            type="button"
            aria-label="Close full screen screenshot"
            onClick={() => setFullscreenScreenshot(null)}
          >
            ×
          </button>
          <img
            src={fullscreenScreenshot.src}
            alt={fullscreenScreenshot.alt}
            onClick={(event) => event.stopPropagation()}
          />
        </div>
      ) : null}
    </main>
  );
}
