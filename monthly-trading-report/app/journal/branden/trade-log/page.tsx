"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import type { DragEvent } from "react";
import type { SetupChecklistTemplate, TradeLogEntry, TraderUser } from "@/lib/types";
import { hasCompletedTradeReview } from "@/lib/trade-review";

type PortfolioSettingsResponse = {
  portfolios?: string[];
  defaultPortfolio?: string;
};

type TradeFilterKey = "status" | "side" | "symbol" | "setup" | "grade" | "review";
type SortDirection = "asc" | "desc";
type TradeColumnKey =
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
type ColumnPreference = {
  key: TradeColumnKey;
  visible: boolean;
};
type TradeSort = {
  key: TradeColumnKey;
  direction: SortDirection;
};

type ReviewExportProgress = {
  open: boolean;
  percent: number;
  title: string;
  detail: string;
};

const filterKeys: TradeFilterKey[] = ["status", "side", "symbol", "setup", "grade", "review"];
const tradeColumnKeys = [
  "status",
  "side",
  "symbol",
  "setup",
  "portfolio",
  "openDate",
  "entry",
  "size",
  "closeDate",
  "exit",
  "stop",
  "commission",
  "usedMargin",
  "takeProfit",
  "risk",
  "cost",
  "netReturn",
  "r",
  "mistake",
  "custom",
  "grade",
  "review"
] as const satisfies TradeColumnKey[];

const defaultColumns: ColumnPreference[] = [
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

function sortedUnique(values: string[]) {
  return Array.from(new Set(values.map((value) => value.trim()).filter(Boolean))).sort((a, b) => a.localeCompare(b));
}

function money(value: number) {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0
  }).format(Number.isFinite(value) ? value : 0);
}

function pct(value: number) {
  return `${(Number.isFinite(value) ? value : 0).toFixed(2)}%`;
}

function avg(values: number[]) {
  return values.length ? values.reduce((total, value) => total + value, 0) / values.length : 0;
}

function localDate() {
  const now = new Date();
  return `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, "0")}-${String(now.getDate()).padStart(2, "0")}`;
}

function daysAgoDate(days: number) {
  const date = new Date();
  date.setDate(date.getDate() - days);
  return `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, "0")}-${String(date.getDate()).padStart(2, "0")}`;
}

function monthStartDate() {
  const now = new Date();
  return `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, "0")}-01`;
}

function yearStartDate() {
  return `${new Date().getFullYear()}-01-01`;
}

function dateInRange(value: string, startDate: string, endDate: string) {
  if (!value) return false;
  if (startDate && value < startDate) return false;
  if (endDate && value > endDate) return false;
  return true;
}

function rangedExitExecutions(trade: TradeLogEntry, startDate: string, endDate: string) {
  return (trade.executions || []).filter(
    (execution) => execution.type === "EXIT" && dateInRange(execution.date, startDate, endDate)
  );
}

function tradeHasRangeActivity(trade: TradeLogEntry, startDate: string, endDate: string) {
  if (rangedExitExecutions(trade, startDate, endDate).length) return true;
  const activityDate = trade.exitDate || trade.entryDate;
  return dateInRange(activityDate, startDate, endDate);
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

function primarySetup(trade: TradeLogEntry) {
  return trade.setupTags[0] || "No setup";
}

function setupTemplateFor(setupName: string, templates: SetupChecklistTemplate[]) {
  return templates.find((template) => template.setupName.trim().toLowerCase() === setupName.trim().toLowerCase());
}

function checklistScore(trade: TradeLogEntry, templates: SetupChecklistTemplate[]) {
  const template = setupTemplateFor(primarySetup(trade), templates);
  const items = template?.criteria?.length ? trade.checklistItems : trade.checklistItems;
  const total = items.reduce((sum, item) => sum + Number(item.points || 0), 0);
  const earned = items.reduce((sum, item) => {
    const points = Number(item.points || 0);
    if ((item.inputType || "boolean") === "points") {
      return sum + Math.max(0, Math.min(points, Number(item.score || 0)));
    }

    return sum + (item.met ? points : 0);
  }, 0);
  const manualGrade = trade.manualGrade?.trim();

  if (!template?.gradeBands?.length || !total) {
    return { earned, total, grade: manualGrade || "Unscored" };
  }

  if (manualGrade) {
    return { earned, total, grade: manualGrade };
  }

  const grade = [...template.gradeBands]
    .sort((a, b) => b.minScore - a.minScore)
    .find((band) => earned >= band.minScore && (band.maxScore === null || earned <= band.maxScore));

  return { earned, total, grade: grade?.label || "Unscored" };
}

function tradeNeedsReview(trade: TradeLogEntry, templates: SetupChecklistTemplate[]) {
  return !trade.risk || !hasCompletedTradeReview(trade.reviewSections, trade.notes) || (!trade.screenshots.length && !(trade.chartLinks || []).length) || !checklistScore(trade, templates).total;
}

function longestStreak(trades: TradeLogEntry[], status: TradeLogEntry["status"]) {
  let longest = 0;
  let current = 0;

  trades.forEach((trade) => {
    const tradeStatus = normalizedTradeStatus(trade);
    if (tradeStatus === status) {
      current += 1;
      longest = Math.max(longest, current);
    } else if (tradeStatus === "WIN" || tradeStatus === "LOSS") {
      current = 0;
    }
  });

  return longest;
}

function csvCell(value: unknown) {
  const text = String(value ?? "");
  return `"${text.replace(/"/g, '""')}"`;
}

function isTradeColumnKey(value: string): value is TradeColumnKey {
  return tradeColumnKeys.includes(value as TradeColumnKey);
}

function defaultSortDirection(key: TradeColumnKey): SortDirection {
  return ["openDate", "closeDate", "entry", "size", "exit", "stop", "commission", "usedMargin", "takeProfit", "risk", "cost", "netReturn", "r"].includes(key)
    ? "desc"
    : "asc";
}

function compareText(a: string, b: string) {
  return a.localeCompare(b, undefined, { numeric: true, sensitivity: "base" });
}

function compareNumber(a: number, b: number) {
  const safeA = Number.isFinite(a) ? a : Number.NEGATIVE_INFINITY;
  const safeB = Number.isFinite(b) ? b : Number.NEGATIVE_INFINITY;
  return safeA - safeB;
}

function gradeSortValue(grade: string) {
  const rank: Record<string, number> = { "A+": 0, A: 1, B: 2, C: 3, D: 4, F: 5, Unscored: 6 };
  return rank[grade] ?? 7;
}

function normalizeColumns(value: unknown): ColumnPreference[] {
  const source = Array.isArray(value) ? value : [];
  const normalized: ColumnPreference[] = [];
  const seen = new Set<TradeColumnKey>();

  source.forEach((item) => {
    if (!item || typeof item !== "object") return;
    const key = String((item as { key?: string }).key || "") as TradeColumnKey;
    if (!defaultColumns.some((column) => column.key === key) || seen.has(key)) return;
    seen.add(key);
    normalized.push({
      key,
      visible: "visible" in (item as Record<string, unknown>) ? Boolean((item as { visible?: unknown }).visible) : true
    });
  });

  defaultColumns.forEach((column) => {
    if (!seen.has(column.key)) normalized.push(column);
  });

  return normalized;
}

function initialTradeLogState() {
  const defaults = {
    filters: {
      status: null,
      side: null,
      symbol: null,
      setup: null,
      grade: null,
      review: null
    } as Record<TradeFilterKey, string[] | null>,
    startDate: monthStartDate(),
    endDate: localDate(),
    portfolio: "",
    symbolSearch: "",
    sort: null as TradeSort | null
  };

  if (typeof window === "undefined") return defaults;

  const params = new URLSearchParams(window.location.search);
  filterKeys.forEach((key) => {
    const raw = params.get(key);
    if (raw === "__none__") {
      defaults.filters[key] = [];
    } else if (raw) {
      defaults.filters[key] = raw.split(",").map((value) => decodeURIComponent(value).trim()).filter(Boolean);
    }
  });
  defaults.startDate = params.get("start") || defaults.startDate;
  defaults.endDate = params.get("end") || defaults.endDate;
  defaults.portfolio = params.get("portfolio") || "";
  defaults.symbolSearch = params.get("q") || "";
  const sortKey = params.get("sort") || "";
  const sortDir = params.get("dir") === "asc" || params.get("dir") === "desc" ? params.get("dir") as SortDirection : null;
  if (isTradeColumnKey(sortKey) && sortDir) {
    defaults.sort = { key: sortKey, direction: sortDir };
  }
  return defaults;
}

function columnLabel(key: TradeColumnKey) {
  const labels: Record<TradeColumnKey, string> = {
    status: "Status",
    side: "Side",
    symbol: "Symbol",
    setup: "Setup",
    portfolio: "Portfolio",
    openDate: "Open date",
    entry: "Entry",
    size: "Size",
    closeDate: "Close date",
    exit: "Exit",
    stop: "Stop",
    commission: "Commission",
    usedMargin: "Used margin",
    takeProfit: "Take profit",
    risk: "Risk",
    cost: "Cost",
    netReturn: "Net return",
    r: "R",
    mistake: "Mistake",
    custom: "Custom tags",
    grade: "Grade",
    review: "Review"
  };
  return labels[key];
}

function tradeSortComparison(
  key: TradeColumnKey,
  a: TradeLogEntry,
  b: TradeLogEntry,
  setupTemplates: SetupChecklistTemplate[]
) {
  const gradeA = checklistScore(a, setupTemplates).grade;
  const gradeB = checklistScore(b, setupTemplates).grade;
  const reviewA = tradeNeedsReview(a, setupTemplates) ? "Needs Review" : "Complete";
  const reviewB = tradeNeedsReview(b, setupTemplates) ? "Needs Review" : "Complete";

  switch (key) {
    case "status": {
      const order: Record<string, number> = { OPEN: 0, WIN: 1, BREAKEVEN: 2, LOSS: 3 };
      return (order[normalizedTradeStatus(a)] ?? 99) - (order[normalizedTradeStatus(b)] ?? 99);
    }
    case "side":
      return compareText(a.side, b.side);
    case "symbol":
      return compareText(a.symbol, b.symbol);
    case "setup":
      return compareText(primarySetup(a), primarySetup(b));
    case "portfolio":
      return compareText(a.portfolioTag || "", b.portfolioTag || "");
    case "openDate":
      return compareText(a.entryDate || "", b.entryDate || "");
    case "entry":
      return compareNumber(a.avgEntry, b.avgEntry);
    case "size":
      return compareNumber(a.shares, b.shares);
    case "closeDate":
      return compareText(a.exitDate || "", b.exitDate || "");
    case "exit":
      return compareNumber(a.exitPrice || 0, b.exitPrice || 0);
    case "stop":
      return compareNumber(a.stopPrice || 0, b.stopPrice || 0);
    case "commission":
      return compareNumber(a.commission, b.commission);
    case "usedMargin":
      return compareNumber(a.usedMargin, b.usedMargin);
    case "takeProfit":
      return compareNumber(a.takeProfitPrice || 0, b.takeProfitPrice || 0);
    case "risk":
      return compareNumber(a.risk, b.risk);
    case "cost":
      return compareNumber(a.avgEntry * a.shares, b.avgEntry * b.shares);
    case "netReturn":
      return compareNumber(a.pnl, b.pnl);
    case "r":
      return compareNumber(a.rMultiple, b.rMultiple);
    case "mistake":
      return compareText(a.mistakeTags.join(", "), b.mistakeTags.join(", "));
    case "custom":
      return compareText(a.customTags.join(", "), b.customTags.join(", "));
    case "grade":
      return gradeSortValue(gradeA) - gradeSortValue(gradeB) || compareText(gradeA, gradeB);
    case "review":
      return compareText(reviewA, reviewB);
    default:
      return 0;
  }
}

export default function BrandenTradeLogPage() {
  const initialStateRef = useRef<ReturnType<typeof initialTradeLogState> | null>(null);
  if (!initialStateRef.current) {
    initialStateRef.current = initialTradeLogState();
  }
  const [user, setUser] = useState<TraderUser | null>(null);
  const [trades, setTrades] = useState<TradeLogEntry[]>([]);
  const [setupTemplates, setSetupTemplates] = useState<SetupChecklistTemplate[]>([]);
  const [activePortfolio, setActivePortfolio] = useState(initialStateRef.current.portfolio);
  const [filters, setFilters] = useState<Record<TradeFilterKey, string[] | null>>(initialStateRef.current.filters);
  const [startDate, setStartDate] = useState(initialStateRef.current.startDate);
  const [endDate, setEndDate] = useState(initialStateRef.current.endDate);
  const [symbolSearch, setSymbolSearch] = useState(initialStateRef.current.symbolSearch);
  const [sort, setSort] = useState<TradeSort | null>(initialStateRef.current.sort);
  const [status, setStatus] = useState("");
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(true);
  const [isExportingReview, setIsExportingReview] = useState(false);
  const [reviewExportProgress, setReviewExportProgress] = useState<ReviewExportProgress>({
    open: false,
    percent: 0,
    title: "",
    detail: ""
  });
  const [columnPreferences, setColumnPreferences] = useState<Record<string, ColumnPreference[]>>({});
  const [draggedColumn, setDraggedColumn] = useState<TradeColumnKey | null>(null);
  const [dragOverColumn, setDragOverColumn] = useState<TradeColumnKey | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function loadPageData() {
      setIsLoading(true);
      setError("");

      const tradeLogResponse = await fetch("/api/journal/branden/trade-log", { cache: "no-store" });
      const tradeLogData = await tradeLogResponse.json().catch(() => ({}));

      if (cancelled) {
        return;
      }

      if (!tradeLogResponse.ok || !tradeLogData.user) {
        setError(tradeLogData.error || "Sign in to view the trade log.");
        setIsLoading(false);
        return;
      }

      setUser(tradeLogData.user);
      setTrades(Array.isArray(tradeLogData.trades) ? tradeLogData.trades : []);
      setSetupTemplates(Array.isArray(tradeLogData.setupChecklists) ? tradeLogData.setupChecklists : []);
      setColumnPreferences(
        tradeLogData.preferences && typeof tradeLogData.preferences === "object"
          ? tradeLogData.preferences
          : {}
      );
      setActivePortfolio((current) => current || String(tradeLogData.defaultPortfolio || ""));
      setIsLoading(false);
    }

    loadPageData().catch((loadError) => {
      if (!cancelled) {
        setError(loadError instanceof Error ? loadError.message : "Could not load trade log.");
        setIsLoading(false);
      }
    });

    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    const params = new URLSearchParams();
    if (startDate !== monthStartDate()) params.set("start", startDate);
    if (endDate !== localDate()) params.set("end", endDate);
    if (activePortfolio) params.set("portfolio", activePortfolio);
    if (symbolSearch.trim()) params.set("q", symbolSearch.trim());
    if (sort) {
      params.set("sort", sort.key);
      params.set("dir", sort.direction);
    }
    filterKeys.forEach((key) => {
      const selected = filters[key];
      if (selected === null) return;
      params.set(key, selected.length ? selected.map(encodeURIComponent).join(",") : "__none__");
    });
    const nextUrl = `${window.location.pathname}${params.toString() ? `?${params.toString()}` : ""}`;
    if (`${window.location.pathname}${window.location.search}` !== nextUrl) {
      window.history.replaceState(null, "", nextUrl);
    }
  }, [activePortfolio, endDate, filters, sort, startDate, symbolSearch]);

  const canEditBrandenJournal = user?.id === "branden" && !user.readOnly;
  const activeColumnScope = activePortfolio || "__all__";
  const activeColumns = useMemo(
    () => normalizeColumns(columnPreferences[activeColumnScope] || columnPreferences.__all__ || defaultColumns),
    [activeColumnScope, columnPreferences]
  );
  const visibleColumns = useMemo(() => activeColumns.filter((column) => column.visible), [activeColumns]);
  const brandenTrades = useMemo(() => trades.filter((trade) => trade.userId === "branden" && !trade.hidden), [trades]);
  const rangeTrades = useMemo(
    () =>
      brandenTrades
        .filter((trade) => !activePortfolio || trade.portfolioTag === activePortfolio)
        .filter((trade) => tradeHasRangeActivity(trade, startDate, endDate))
        .map((trade) => tradeForRange(trade, startDate, endDate)),
    [activePortfolio, brandenTrades, endDate, startDate]
  );
  const filterOptions = useMemo(
    () => ({
      status: sortedUnique(rangeTrades.map((trade) => normalizedTradeStatus(trade))),
      side: sortedUnique(rangeTrades.map((trade) => trade.side)),
      symbol: sortedUnique(rangeTrades.map((trade) => trade.symbol)),
      setup: sortedUnique(rangeTrades.map(primarySetup)),
      grade: sortedUnique(rangeTrades.map((trade) => checklistScore(trade, setupTemplates).grade)),
      review: ["Needs Review", "Complete"]
    }),
    [rangeTrades, setupTemplates]
  );
  const filteredTrades = useMemo(
    () => {
      const tickerSearch = symbolSearch.trim().replace(/^#/, "").toUpperCase();
      const visible = rangeTrades
        .filter((trade) => filters.status === null || filters.status.includes(normalizedTradeStatus(trade)))
        .filter((trade) => filters.side === null || filters.side.includes(trade.side))
        .filter((trade) => filters.symbol === null || filters.symbol.includes(trade.symbol))
        .filter((trade) => !tickerSearch || trade.symbol.toUpperCase().includes(tickerSearch))
        .filter((trade) => filters.setup === null || filters.setup.includes(primarySetup(trade)))
        .filter((trade) => filters.grade === null || filters.grade.includes(checklistScore(trade, setupTemplates).grade))
        .filter(
          (trade) =>
            filters.review === null ||
            filters.review.includes(tradeNeedsReview(trade, setupTemplates) ? "Needs Review" : "Complete")
        );

      return [...visible].sort((a, b) => {
        const comparison = sort
          ? tradeSortComparison(sort.key, a, b, setupTemplates) * (sort.direction === "asc" ? 1 : -1)
          : (b.entryDate || "").localeCompare(a.entryDate || "");
        return comparison || a.symbol.localeCompare(b.symbol) || a.id.localeCompare(b.id);
      });
    },
    [filters, rangeTrades, setupTemplates, sort, symbolSearch]
  );
  const summary = useMemo(() => {
    const settled = filteredTrades.filter(countsAsSettledTrade);
    const wins = settled.filter((trade) => normalizedTradeStatus(trade) === "WIN");
    const losses = settled.filter((trade) => normalizedTradeStatus(trade) === "LOSS");
    const ordered = [...settled].sort((a, b) => (a.exitDate || a.entryDate).localeCompare(b.exitDate || b.entryDate));
    const grossWin = wins.reduce((total, trade) => total + trade.pnl, 0);
    const grossLoss = Math.abs(losses.reduce((total, trade) => total + trade.pnl, 0));
    const swingTrades = settled.filter((trade) => trade.daysInTrade > 0);
    const decisionTrades = wins.length + losses.length;

    return {
      netPnl: settled.reduce((total, trade) => total + trade.pnl, 0),
      profitFactor: grossLoss ? (grossWin / grossLoss).toFixed(2) : grossWin ? "∞" : "-",
      winRate: decisionTrades ? (wins.length / decisionTrades) * 100 : 0,
      expectancy: settled.length ? avg(settled.map((trade) => trade.rMultiple)) : 0,
      avgWinR: avg(wins.map((trade) => trade.rMultiple)),
      avgLossR: avg(losses.map((trade) => trade.rMultiple)),
      totalTrades: filteredTrades.length,
      totalR: settled.reduce((total, trade) => total + trade.rMultiple, 0),
      avgTradeLength: avg(settled.map((trade) => trade.daysInTrade)),
      avgSwingLength: avg(swingTrades.map((trade) => trade.daysInTrade)),
      longestWinStreak: longestStreak(ordered, "WIN"),
      longestLossStreak: longestStreak(ordered, "LOSS"),
      avgWin: avg(wins.map((trade) => trade.pnl)),
      avgLoss: avg(losses.map((trade) => Math.abs(trade.pnl))),
      avgRisk: avg(filteredTrades.map((trade) => trade.risk)),
      needsReview: filteredTrades.filter((trade) => tradeNeedsReview(trade, setupTemplates)).length
    };
  }, [filteredTrades, setupTemplates]);
  function setAllFilterOptions(key: TradeFilterKey) {
    setFilters((current) => ({ ...current, [key]: null }));
  }

  function clearFilterOptions(key: TradeFilterKey) {
    setFilters((current) => ({ ...current, [key]: [] }));
  }

  function toggleFilterOption(key: TradeFilterKey, option: string) {
    setFilters((current) => {
      const options = filterOptions[key];
      const selected = current[key] === null ? [...options] : current[key];
      const next = selected.includes(option)
        ? selected.filter((value) => value !== option)
        : [...selected, option];
      return { ...current, [key]: next.length === options.length ? null : next };
    });
  }

  function filterSummary(key: TradeFilterKey) {
    const selected = filters[key];
    if (selected === null) return "All";
    if (!selected.length) return "None";
    if (selected.length === 1) return selected[0];
    return `${selected.length} selected`;
  }

  function toggleSort(key: TradeColumnKey) {
    setSort((current) => {
      if (current?.key === key) {
        return { key, direction: current.direction === "asc" ? "desc" : "asc" };
      }

      return { key, direction: defaultSortDirection(key) };
    });
  }

  function renderSortButton(key: TradeColumnKey) {
    const isActive = sort?.key === key;
    const direction = isActive ? sort.direction : defaultSortDirection(key);

    return (
      <button
        aria-label={`Sort by ${columnLabel(key)} ${direction === "asc" ? "ascending" : "descending"}`}
        aria-sort={isActive ? (sort.direction === "asc" ? "ascending" : "descending") : "none"}
        className={`trade-sort-button${isActive ? " active" : ""}`}
        onClick={(event) => {
          event.stopPropagation();
          toggleSort(key);
        }}
        onMouseDown={(event) => event.stopPropagation()}
        type="button"
      >
        <span>{columnLabel(key).toUpperCase()}</span>
        <span className="trade-sort-indicator" aria-hidden="true">{isActive ? (sort.direction === "asc" ? "▲" : "▼") : "↕"}</span>
      </button>
    );
  }

  async function saveColumnOrder(nextColumns: ColumnPreference[]) {
    const previousPreferences = columnPreferences;
    const nextPreferences = {
      ...columnPreferences,
      [activeColumnScope]: normalizeColumns(nextColumns)
    };

    setColumnPreferences(nextPreferences);
    const response = await fetch("/api/settings/branden-columns", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ preferences: nextPreferences })
    });
    const data = await response.json().catch(() => ({}));

    if (!response.ok) {
      setColumnPreferences(previousPreferences);
      setStatus(data.error || "Could not save the column order.");
      return;
    }

    setColumnPreferences(data.preferences && typeof data.preferences === "object" ? data.preferences : nextPreferences);
    setStatus("Column order saved.");
  }

  function dropColumn(event: DragEvent<HTMLTableCellElement>, targetKey: TradeColumnKey) {
    event.preventDefault();
    const sourceKey = draggedColumn;
    setDraggedColumn(null);
    setDragOverColumn(null);

    if (!sourceKey || sourceKey === targetKey || !canEditBrandenJournal) return;
    const sourceIndex = activeColumns.findIndex((column) => column.key === sourceKey);
    const targetIndex = activeColumns.findIndex((column) => column.key === targetKey);
    if (sourceIndex < 0 || targetIndex < 0) return;

    const nextColumns = [...activeColumns];
    const [movedColumn] = nextColumns.splice(sourceIndex, 1);
    nextColumns.splice(targetIndex, 0, movedColumn);
    void saveColumnOrder(nextColumns);
  }

  function renderColumnHeader(key: TradeColumnKey) {
    const filterKey = (["status", "side", "symbol", "setup", "grade", "review"] as TradeFilterKey[]).find(
      (candidate) => candidate === key
    );

    if (!filterKey) {
      return renderSortButton(key);
    }

    return (
      <div className="trade-column-filter" onMouseDown={(event) => event.stopPropagation()}>
        {renderSortButton(key)}
        <details className="trade-multi-filter" onClick={(event) => event.stopPropagation()}>
          <summary className={filters[filterKey] === null ? "" : "active"}>
            {filterSummary(filterKey)}
          </summary>
          <div className="trade-multi-filter-menu">
            <div className="trade-multi-filter-actions">
              <button type="button" onClick={() => setAllFilterOptions(filterKey)}>All</button>
              <button type="button" onClick={() => clearFilterOptions(filterKey)}>None</button>
            </div>
            <div className="trade-multi-filter-options">
              {filterOptions[filterKey].map((option) => {
                const selected = filters[filterKey];
                const checked = selected === null || selected.includes(option);
                return (
                  <label key={option}>
                    <input
                      type="checkbox"
                      checked={checked}
                      onChange={() => toggleFilterOption(filterKey, option)}
                    />
                    <span>{option}</span>
                  </label>
                );
              })}
            </div>
          </div>
        </details>
      </div>
    );
  }

  function renderTradeCell(key: TradeColumnKey, trade: TradeLogEntry, grade: string, review: string) {
    switch (key) {
      case "status":
        const status = normalizedTradeStatus(trade);
        return <span className={`trade-badge ${status.toLowerCase()}`}>{status}</span>;
      case "side":
        return <span className={`side-pill ${trade.side.toLowerCase()}`}>{trade.side}</span>;
      case "symbol":
        return <span className="trade-symbol">#{trade.symbol}</span>;
      case "setup":
        return primarySetup(trade);
      case "portfolio":
        return trade.portfolioTag || "-";
      case "openDate":
        return trade.entryDate;
      case "entry":
        return trade.avgEntry ? trade.avgEntry.toFixed(2) : "-";
      case "size":
        return trade.shares || "-";
      case "closeDate":
        return trade.exitDate || "-";
      case "exit":
        return trade.exitPrice ? trade.exitPrice.toFixed(2) : "-";
      case "stop":
        return trade.stopPrice ? trade.stopPrice.toFixed(2) : "-";
      case "commission":
        return money(trade.commission);
      case "usedMargin":
        return money(trade.usedMargin);
      case "takeProfit":
        return trade.takeProfitPrice ? trade.takeProfitPrice.toFixed(2) : "-";
      case "risk":
        return money(trade.risk);
      case "cost":
        return money(trade.avgEntry * trade.shares);
      case "netReturn":
        return <span className={trade.pnl >= 0 ? "trade-positive" : "trade-negative"}>{money(trade.pnl)}</span>;
      case "r":
        return <span className={trade.rMultiple >= 0 ? "trade-positive" : "trade-negative"}>{trade.rMultiple.toFixed(2)}R</span>;
      case "mistake":
        return trade.mistakeTags.length ? (
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
        return trade.customTags.length ? trade.customTags.join(", ") : "-";
      case "grade":
        return <span className="grade-pill">{grade}</span>;
      case "review":
        return <span className={review === "Needs Review" ? "review-pill needs-review" : "review-pill complete"}>{review}</span>;
      default:
        return "-";
    }
  }

  function tradeDetailHref(tradeId: string) {
    const returnTo = `${window.location.pathname}${window.location.search}`;
    let tradeNavKey = "";

    try {
      tradeNavKey = `trade-log-${Date.now()}`;
      window.sessionStorage.setItem(tradeNavKey, JSON.stringify(filteredTrades.map((trade) => trade.id)));
    } catch {
      tradeNavKey = "";
    }

    const params = new URLSearchParams({
      tradeId,
      returnTo
    });
    if (tradeNavKey) {
      params.set("tradeNavKey", tradeNavKey);
    }

    return `/journal/branden/dashboard?${params.toString()}`;
  }

  function exportCsv() {
    const headers = ["status", "side", "symbol", "setup", "open_date", "entry", "close_date", "risk", "net_return", "r", "grade", "review"];
    const rows = filteredTrades.map((trade) => [
      normalizedTradeStatus(trade),
      trade.side,
      trade.symbol,
      primarySetup(trade),
      trade.entryDate,
      trade.avgEntry,
      trade.exitDate,
      trade.risk,
      trade.pnl,
      trade.rMultiple,
      checklistScore(trade, setupTemplates).grade,
      tradeNeedsReview(trade, setupTemplates) ? "Needs Review" : "Complete"
    ]);
    const csv = [headers, ...rows].map((row) => row.map(csvCell).join(",")).join("\n");
    const url = URL.createObjectURL(new Blob([csv], { type: "text/csv;charset=utf-8" }));
    const link = document.createElement("a");
    link.href = url;
    link.download = `branden-trade-log-${new Date().toISOString().slice(0, 10)}.csv`;
    link.click();
    URL.revokeObjectURL(url);
  }

  function updateReviewProgress(percent: number, title: string, detail: string) {
    setReviewExportProgress({ open: true, percent, title, detail });
  }

  async function exportReviewDocx() {
    if (!filteredTrades.length) {
      setStatus("No visible trades to export.");
      return;
    }

    setIsExportingReview(true);
    setStatus("");
    setError("");
    updateReviewProgress(8, "Preparing trade review", `Collecting ${filteredTrades.length} visible trades and setup context.`);
    const progressTimers = [
      window.setTimeout(() => updateReviewProgress(22, "Sending to OpenAI", "Submitting filtered trades, notes, setup criteria, and screenshots."), 500),
      window.setTimeout(() => updateReviewProgress(48, "Reviewing trades", "OpenAI is building the trade-by-trade review."), 1800),
      window.setTimeout(() => updateReviewProgress(68, "Analyzing details", "Checking setup criteria, notes, executions, and chart context."), 4500),
      window.setTimeout(() => updateReviewProgress(82, "Building document", "Formatting the review into a downloadable DOCX."), 9000)
    ];

    try {
      const response = await fetch("/api/journal/branden/trade-log/export-review", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          tradeIds: filteredTrades.map((trade) => trade.id),
          startDate,
          endDate
        })
      });

      if (!response.ok) {
        const data = await response.json().catch(() => ({}));
        throw new Error(data.error || "Could not export the review document. Try again after OpenAI is fixed.");
      }

      updateReviewProgress(92, "Downloading DOCX", "The review is complete. Preparing the file download.");
      const blob = await response.blob();
      const disposition = response.headers.get("Content-Disposition") || "";
      const filenameMatch = disposition.match(/filename="([^"]+)"/);
      const filename = filenameMatch?.[1] || `branden-trade-review-${new Date().toISOString().slice(0, 10)}.docx`;
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = filename;
      link.click();
      URL.revokeObjectURL(url);
      updateReviewProgress(100, "Export complete", `Downloaded review document for ${filteredTrades.length} visible trades.`);
      setStatus(`Exported review document for ${filteredTrades.length} visible trades.`);
    } catch (exportError) {
      setReviewExportProgress({
        open: true,
        percent: 100,
        title: "Export failed",
        detail: exportError instanceof Error ? exportError.message : "Could not export the review document. Try again after OpenAI is fixed."
      });
      setError(exportError instanceof Error ? exportError.message : "Could not export the review document. Try again after OpenAI is fixed.");
      setStatus("");
    } finally {
      progressTimers.forEach(window.clearTimeout);
      setIsExportingReview(false);
    }
  }

  return (
    <div className="branden-journal-content">
        <header className="branden-route-header">
          <div>
            <p className="eyebrow">Branden journal</p>
            <h1>Trade Log</h1>
            <span>{filteredTrades.length} visible rows</span>
          </div>
        </header>

        <section className="branden-route-toolbar">
          <label>
            Start date
            <input type="date" value={startDate} onChange={(event) => setStartDate(event.target.value)} />
          </label>
          <label>
            End date
            <input type="date" value={endDate} onChange={(event) => setEndDate(event.target.value)} />
          </label>
          <div className="trade-date-quick-filters">
            <button
              className="trade-muted-button"
              type="button"
              onClick={() => {
                setStartDate(daysAgoDate(6));
                setEndDate(localDate());
              }}
            >
              Last 7 days
            </button>
            <button
              className="trade-muted-button"
              type="button"
              onClick={() => {
                setStartDate(daysAgoDate(29));
                setEndDate(localDate());
              }}
            >
              Last 30 days
            </button>
            <button
              className="trade-muted-button"
              type="button"
              onClick={() => {
                setStartDate(monthStartDate());
                setEndDate(localDate());
              }}
            >
              Month to date
            </button>
            <button
              className="trade-muted-button"
              type="button"
              onClick={() => {
                setStartDate(yearStartDate());
                setEndDate(localDate());
              }}
            >
              Year to date
            </button>
          </div>
          {user ? <span>Signed in as {user.name}</span> : null}
        </section>

        {status ? <p className="status trade-log-status">{status}</p> : null}
        {error ? <p className="status error">{error}</p> : null}
        {isLoading ? <p className="status">Loading trade log...</p> : null}
        {reviewExportProgress.open ? (
          <div className="review-export-modal-backdrop" role="status" aria-live="polite">
            <div className="review-export-modal">
              <div className="review-export-modal-head">
                <div>
                  <p className="eyebrow">Review export</p>
                  <h3>{reviewExportProgress.title}</h3>
                </div>
                {!isExportingReview ? (
                  <button
                    className="trade-muted-button"
                    type="button"
                    onClick={() => setReviewExportProgress((current) => ({ ...current, open: false }))}
                  >
                    Close
                  </button>
                ) : null}
              </div>
              <p>{reviewExportProgress.detail}</p>
              <div className="review-export-progress-track">
                <div style={{ width: `${reviewExportProgress.percent}%` }} />
              </div>
              <strong>{reviewExportProgress.percent}% complete</strong>
            </div>
          </div>
        ) : null}

        {!isLoading && !error ? (
          <>
            <section className="trade-summary-strip">
              <div className="trade-chart-heading">
                <h3>Filtered stats</h3>
                <span>#</span>
              </div>
              <div className="trade-side-kpis trade-side-kpis-top">
                <article><span>Settled P&L</span><strong className={summary.netPnl >= 0 ? "trade-positive" : "trade-negative"}>{money(summary.netPnl)}</strong></article>
                <article><span>Profit Factor</span><strong>{summary.profitFactor}</strong></article>
                <article><span>Win %</span><strong>{pct(summary.winRate)}</strong></article>
                <article><span>Expectancy</span><strong>{summary.expectancy.toFixed(2)}R</strong></article>
                <article><span>Avg R Win</span><strong>{summary.avgWinR.toFixed(2)}R</strong></article>
                <article><span>Avg R Loss</span><strong>{summary.avgLossR.toFixed(2)}R</strong></article>
              </div>
              <div className="trade-metric-board trade-metric-board-top">
                <div><span>Total Trades</span><strong>{summary.totalTrades}</strong></div>
                <div><span>Total R Multiple</span><strong>{summary.totalR.toFixed(2)}R</strong></div>
                <div><span>Avg Trade Len.</span><strong>{summary.avgTradeLength.toFixed(1)}</strong></div>
                <div><span>Avg Swing Len.</span><strong>{summary.avgSwingLength.toFixed(1)}</strong></div>
                <div><span>Longest Winning Streak</span><strong>{summary.longestWinStreak}</strong></div>
                <div><span>Longest Losing Streak</span><strong>{summary.longestLossStreak}</strong></div>
                <div><span>Average Win</span><strong>{money(summary.avgWin)}</strong></div>
                <div><span>Average Loss</span><strong>{money(summary.avgLoss)}</strong></div>
                <div><span>Average Risk</span><strong>{money(summary.avgRisk)}</strong></div>
                <div><span>Needs Review</span><strong>{summary.needsReview}</strong></div>
              </div>
            </section>

            <div className="trade-workspace trade-list-workspace">
              <div className="trade-main-panel">
                <div className="trade-bulk-actions">
                  <strong>{filteredTrades.length} visible rows</strong>
                  <label className="trade-symbol-search">
                    <span>Search ticker</span>
                    <input
                      type="search"
                      value={symbolSearch}
                      onChange={(event) => setSymbolSearch(event.target.value)}
                      placeholder="Type ticker..."
                      aria-label="Search trades by ticker"
                    />
                  </label>
                  <button className="trade-muted-button" type="button" onClick={exportCsv}>Export CSV</button>
                  <button className="trade-muted-button" type="button" onClick={exportReviewDocx} disabled={isExportingReview || !filteredTrades.length}>
                    {isExportingReview ? "Generating AI review..." : "AI Review .docx"}
                  </button>
                </div>
                <div className="trade-table-wrap">
                  <table className="trade-table">
                    <thead>
                      <tr>
                        {visibleColumns.map((column) => (
                          <th
                            className={[
                              canEditBrandenJournal ? "trade-column-draggable" : "",
                              draggedColumn === column.key ? "dragging" : "",
                              dragOverColumn === column.key ? "drag-over" : ""
                            ].filter(Boolean).join(" ")}
                            draggable={canEditBrandenJournal}
                            key={column.key}
                            onDragStart={(event) => {
                              setDraggedColumn(column.key);
                              event.dataTransfer.effectAllowed = "move";
                              event.dataTransfer.setData("text/plain", column.key);
                            }}
                            onDragEnter={() => setDragOverColumn(column.key)}
                            onDragOver={(event) => {
                              event.preventDefault();
                              event.dataTransfer.dropEffect = "move";
                            }}
                            onDragLeave={(event) => {
                              if (!event.currentTarget.contains(event.relatedTarget as Node | null)) {
                                setDragOverColumn((current) => current === column.key ? null : current);
                              }
                            }}
                            onDrop={(event) => dropColumn(event, column.key)}
                            onDragEnd={() => {
                              setDraggedColumn(null);
                              setDragOverColumn(null);
                            }}
                            title={canEditBrandenJournal ? "Drag to reorder this column" : undefined}
                          >
                            <span className="trade-column-header-content">
                              {canEditBrandenJournal ? <span className="trade-column-grip" aria-hidden="true">⋮⋮</span> : null}
                              {renderColumnHeader(column.key)}
                            </span>
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {filteredTrades.map((trade) => {
                        const grade = checklistScore(trade, setupTemplates).grade;
                        const review = tradeNeedsReview(trade, setupTemplates) ? "Needs Review" : "Complete";
                        return (
                          <tr key={trade.id} onClick={() => { window.location.href = tradeDetailHref(trade.id); }}>
                            {visibleColumns.map((column) => (
                              <td key={column.key}>{renderTradeCell(column.key, trade, grade, review)}</td>
                            ))}
                          </tr>
                        );
                      })}
                      {!filteredTrades.length ? (
                        <tr>
                          <td colSpan={Math.max(visibleColumns.length, 1)} className="trade-empty">No trades match the current filters.</td>
                        </tr>
                      ) : null}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </>
        ) : null}
      </div>
  );
}
