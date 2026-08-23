"use client";

import { useEffect, useMemo, useState } from "react";
import type { TradeLogEntry } from "@/lib/types";

type Props = {
  trades: TradeLogEntry[];
  activePortfolio: string;
  compact?: boolean;
};

type Candle = {
  time: string;
  close: number;
};

type TimeStopStatus = "healthy" | "watch" | "lagging" | "notLeader" | "deadMoney" | "missing";
type SortDirection = "asc" | "desc";
type SortKey = "status" | "symbol" | "entryDate" | "daysHeld" | "stockReturn" | "spyReturn" | "excessReturn" | "spyMultiple" | "rsSlope" | "flags";

type MonitoredPosition = {
  trade: TradeLogEntry;
  daysHeld: number;
  stockReturn: number;
  spyReturn: number;
  excessReturn: number | null;
  spyMultiple: number | null;
  spyRsSlope: number | null;
  status: TimeStopStatus;
  flags: string[];
};

const DEFAULT_TIME_STOP_DAYS = 10;

function isEquityLikeSymbol(symbol: string) {
  const cleaned = symbol.trim().replace(/^#/, "").toUpperCase();
  return Boolean(cleaned) && /^[A-Z]{1,6}$/.test(cleaned) && !["SPY", "QQQ"].includes(cleaned);
}

function pct(value: number | null) {
  if (value === null || !Number.isFinite(value)) {
    return "-";
  }

  return `${value >= 0 ? "+" : ""}${value.toFixed(2)}%`;
}

function multiple(value: number | null) {
  if (value === null || !Number.isFinite(value)) {
    return "-";
  }

  return `${value.toFixed(2)}x`;
}

function statusLabel(status: TimeStopStatus) {
  const labels: Record<TimeStopStatus, string> = {
    healthy: "Healthy",
    watch: "Watch",
    lagging: "Lagging",
    notLeader: "Not leader",
    deadMoney: "Dead money",
    missing: "Missing"
  };
  return labels[status];
}

function returnClass(value: number | null) {
  if (value === null || !Number.isFinite(value) || value === 0) {
    return "";
  }

  return value > 0 ? "time-stop-positive" : "time-stop-negative";
}

function findCandleOnOrBefore(candles: Candle[], date: string) {
  if (!date) {
    return null;
  }

  for (let index = candles.length - 1; index >= 0; index -= 1) {
    if (String(candles[index].time) <= date) {
      return candles[index];
    }
  }

  return null;
}

function candlesFromDate(candles: Candle[], date: string) {
  return candles.filter((candle) => String(candle.time) >= date);
}

function benchmarkReturn(entry: Candle | null, latest: Candle | null) {
  if (!entry?.close || !latest?.close) {
    return 0;
  }

  return ((latest.close - entry.close) / entry.close) * 100;
}

function stockReturn(trade: TradeLogEntry, currentPrice: number) {
  const entry = Number(trade.avgEntry) || 0;

  if (!entry || !currentPrice) {
    return null;
  }

  return trade.side === "SHORT" ? ((entry - currentPrice) / entry) * 100 : ((currentPrice - entry) / entry) * 100;
}

function relativeMultiple(stock: number | null, benchmark: number) {
  if (stock === null || benchmark <= 0) {
    return null;
  }

  return stock / benchmark;
}

function rsLineSlope(stockCandles: Candle[], benchmarkCandles: Candle[], entryDate: string) {
  const tradeStockCandles = candlesFromDate(stockCandles, entryDate);
  const tradeBenchmarkCandles = candlesFromDate(benchmarkCandles, entryDate);
  const points = tradeStockCandles
    .map((stockCandle) => {
      const benchmarkCandle = findCandleOnOrBefore(tradeBenchmarkCandles, stockCandle.time);

      if (!stockCandle.close || !benchmarkCandle?.close) {
        return null;
      }

      return { time: stockCandle.time, ratio: stockCandle.close / benchmarkCandle.close };
    })
    .filter((point): point is { time: string; ratio: number } => Boolean(point));

  if (points.length < 2) {
    return null;
  }

  const latest = points[points.length - 1];
  const prior = points[Math.max(0, points.length - 6)];

  return prior.ratio ? ((latest.ratio - prior.ratio) / prior.ratio) * 100 : null;
}

async function fetchCandles(symbol: string) {
  const response = await fetch(`/api/market-data/${encodeURIComponent(symbol)}?timeframe=1d`, { cache: "no-store" });
  const data = await response.json().catch(() => ({}));

  if (!response.ok) {
    throw new Error(data.error || `Could not load ${symbol}.`);
  }

  return ((data.candles || []) as Candle[]).filter((candle) => {
    const close = Number(candle.close);
    return Boolean(candle.time) && Number.isFinite(close) && close > 0;
  });
}

function statusClass(status: TimeStopStatus) {
  return status.replace(/[A-Z]/g, (letter) => `-${letter.toLowerCase()}`);
}

function statusSeverity(status: TimeStopStatus) {
  return { deadMoney: 5, notLeader: 4, lagging: 3, watch: 2, missing: 1, healthy: 0 }[status];
}

function monitorPosition(
  trade: TradeLogEntry,
  stockCandles: Candle[],
  spyCandles: Candle[],
  timeStopDays: number
): MonitoredPosition {
  const latestStock = stockCandles[stockCandles.length - 1] || null;
  const spyEntry = findCandleOnOrBefore(spyCandles, trade.entryDate);
  const latestSpy = spyCandles[spyCandles.length - 1] || null;
  const daysHeld = candlesFromDate(spyCandles, trade.entryDate).length;
  const currentStockReturn = stockReturn(trade, Number(latestStock?.close) || 0);
  const spyReturn = benchmarkReturn(spyEntry, latestSpy);
  const excessReturn = currentStockReturn === null ? null : currentStockReturn - spyReturn;
  const spyMultiple = relativeMultiple(currentStockReturn, spyReturn);
  const spyRsSlope = rsLineSlope(stockCandles, spyCandles, trade.entryDate);
  const flags: string[] = [];

  if (!latestStock || currentStockReturn === null) {
    flags.push("Missing current price or entry price");
  }

  if (daysHeld > timeStopDays && currentStockReturn !== null) {
    if (spyMultiple !== null && spyMultiple < 1) {
      flags.push("Dead money");
    } else if (spyMultiple !== null && spyMultiple < 2) {
      flags.push("Not meeting leader standard");
    } else if (spyMultiple === null && excessReturn !== null && excessReturn < 0) {
      flags.push("Lagging SPY");
    } else if (currentStockReturn < 0) {
      flags.push("Negative after trigger");
    }
  }

  if (daysHeld > timeStopDays && currentStockReturn !== null && currentStockReturn > 0 && spyRsSlope !== null && Math.abs(spyRsSlope) < 0.25) {
    flags.push("Index-like return with single-stock risk");
  }

  const status: TimeStopStatus = flags.includes("Missing current price or entry price")
    ? "missing"
    : flags.includes("Dead money")
      ? "deadMoney"
      : flags.includes("Not meeting leader standard")
        ? "notLeader"
        : flags.includes("Lagging SPY")
          ? "lagging"
        : flags.length
          ? "watch"
          : "healthy";

  return {
    trade,
    daysHeld,
    stockReturn: currentStockReturn || 0,
    spyReturn,
    excessReturn,
    spyMultiple,
    spyRsSlope,
    status,
    flags
  };
}

export default function TimeStopMonitor({ trades, activePortfolio, compact = false }: Props) {
  const [timeStopDays, setTimeStopDays] = useState(DEFAULT_TIME_STOP_DAYS);
  const [sortKey, setSortKey] = useState<SortKey>("status");
  const [sortDirection, setSortDirection] = useState<SortDirection>("desc");
  const [candlesBySymbol, setCandlesBySymbol] = useState<Record<string, Candle[]>>({});
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }

    const savedDays = Number(window.localStorage.getItem("branden-time-stop-days"));
    if (savedDays) {
      setTimeStopDays(savedDays);
    }
  }, []);

  useEffect(() => {
    if (typeof window !== "undefined") {
      window.localStorage.setItem("branden-time-stop-days", String(timeStopDays || DEFAULT_TIME_STOP_DAYS));
    }
  }, [timeStopDays]);

  const openEquityPositions = useMemo(
    () =>
      trades
        .filter((trade) => trade.status === "OPEN")
        .filter((trade) => !activePortfolio || trade.portfolioTag === activePortfolio)
        .filter((trade) => isEquityLikeSymbol(trade.symbol) && trade.entryDate)
        .sort((a, b) => a.symbol.localeCompare(b.symbol)),
    [trades, activePortfolio]
  );

  const symbols = useMemo(
    () => Array.from(new Set(["SPY", ...openEquityPositions.map((trade) => trade.symbol)])),
    [openEquityPositions]
  );

  useEffect(() => {
    let cancelled = false;

    async function loadCandles() {
      setIsLoading(true);
      setError("");

      try {
        const results = await Promise.allSettled(symbols.map(async (symbol) => [symbol, await fetchCandles(symbol)] as const));

        if (cancelled) {
          return;
        }

        const loaded: Record<string, Candle[]> = {};
        const failures: string[] = [];

        results.forEach((result, index) => {
          const symbol = symbols[index];
          if (result.status === "fulfilled") {
            loaded[result.value[0]] = result.value[1];
          } else {
            failures.push(symbol);
          }
        });

        setCandlesBySymbol(loaded);
        setError(failures.length ? `Could not load candles for ${failures.join(", ")}.` : "");
      } finally {
        if (!cancelled) {
          setIsLoading(false);
        }
      }
    }

    loadCandles();

    return () => {
      cancelled = true;
    };
  }, [symbols.join("|")]);

  const monitoredPositions = useMemo(() => {
    const spyCandles = candlesBySymbol.SPY || [];

    if (!spyCandles.length) {
      return [];
    }

    return openEquityPositions
      .map((trade) => monitorPosition(trade, candlesBySymbol[trade.symbol] || [], spyCandles, timeStopDays))
      .sort((a, b) => {
        return statusSeverity(b.status) - statusSeverity(a.status) || b.daysHeld - a.daysHeld;
      });
  }, [openEquityPositions, candlesBySymbol, timeStopDays]);

  const deadMoneyCount = monitoredPositions.filter((position) => position.status === "deadMoney").length;
  const notLeaderCount = monitoredPositions.filter((position) => position.status === "notLeader").length;
  const watchCount = monitoredPositions.filter((position) => position.status === "watch").length;
  const healthyCount = monitoredPositions.filter((position) => position.status === "healthy").length;
  const averageMultipleValues = monitoredPositions
    .map((position) => position.spyMultiple)
    .filter((value): value is number => value !== null && Number.isFinite(value));
  const averageMultiple = averageMultipleValues.length
    ? averageMultipleValues.reduce((sum, value) => sum + value, 0) / averageMultipleValues.length
    : null;
  const sortedPositions = useMemo(() => {
    const direction = sortDirection === "asc" ? 1 : -1;

    function sortValue(position: MonitoredPosition) {
      switch (sortKey) {
        case "status":
          return statusSeverity(position.status);
        case "symbol":
          return position.trade.symbol;
        case "entryDate":
          return position.trade.entryDate;
        case "daysHeld":
          return position.daysHeld;
        case "stockReturn":
          return position.stockReturn;
        case "spyReturn":
          return position.spyReturn;
        case "excessReturn":
          return position.excessReturn ?? -Infinity;
        case "spyMultiple":
          return position.spyMultiple ?? -Infinity;
        case "rsSlope":
          return position.spyRsSlope ?? -Infinity;
        case "flags":
          return position.flags.join(", ");
        default:
          return "";
      }
    }

    return [...monitoredPositions].sort((a, b) => {
      const aValue = sortValue(a);
      const bValue = sortValue(b);

      if (typeof aValue === "number" && typeof bValue === "number") {
        return (aValue - bValue) * direction;
      }

      return String(aValue).localeCompare(String(bValue)) * direction;
    });
  }, [monitoredPositions, sortDirection, sortKey]);

  function changeSort(nextKey: SortKey) {
    if (sortKey === nextKey) {
      setSortDirection((current) => (current === "asc" ? "desc" : "asc"));
      return;
    }

    setSortKey(nextKey);
    setSortDirection(nextKey === "symbol" || nextKey === "entryDate" || nextKey === "flags" ? "asc" : "desc");
  }

  function sortableHeader(key: SortKey, label: string) {
    return (
      <button className="time-stop-sort-button" type="button" onClick={() => changeSort(key)}>
        {label}
        <span>{sortKey === key ? (sortDirection === "asc" ? "↑" : "↓") : "↕"}</span>
      </button>
    );
  }

  return (
    <section className="time-stop-panel">
      <div className="column-settings-head">
        <div>
          <p className="eyebrow">Time Stop / Opportunity Cost</p>
          <h3>Time Stop Monitor</h3>
          {compact ? <span>Open-position lag check versus SPY.</span> : null}
        </div>
        <div className="time-stop-actions">
          <label>
            Trigger after
            <input
              min={1}
              type="number"
              value={timeStopDays}
              onChange={(event) => setTimeStopDays(Number(event.target.value) || DEFAULT_TIME_STOP_DAYS)}
            />
            trading days
          </label>
        </div>
      </div>

      {!compact ? (
        <div className="time-stop-kpis">
          <article>
            <span>Open equity positions</span>
            <strong>{monitoredPositions.length}</strong>
          </article>
          <article>
            <span>Dead money</span>
            <strong className={deadMoneyCount ? "time-stop-danger" : ""}>{deadMoneyCount}</strong>
          </article>
          <article>
            <span>Not leader standard</span>
            <strong className={notLeaderCount ? "time-stop-warning" : ""}>{notLeaderCount}</strong>
          </article>
          <article>
            <span>Watch</span>
            <strong>{watchCount}</strong>
          </article>
          <article>
            <span>Healthy</span>
            <strong className="time-stop-good">{healthyCount}</strong>
          </article>
          <article>
            <span>Avg relative multiple</span>
            <strong>{multiple(averageMultiple)}</strong>
          </article>
        </div>
      ) : null}

      {isLoading ? <p className="benchmark-note">Loading open position and benchmark candles...</p> : null}
      {error ? <p className="benchmark-error">{error}</p> : null}

      {!compact ? (
        <div className="time-stop-rule-grid">
          <article>
            <strong>Dead money</strong>
            <span>Held &gt; {timeStopDays} trading days and relative multiple &lt; 1.0.</span>
          </article>
          <article>
            <strong>Not meeting leader standard</strong>
            <span>Held &gt; {timeStopDays} trading days and relative multiple &lt; 2.0.</span>
          </article>
          <article>
            <strong>Index-like risk</strong>
            <span>Stock is up, but the RS line is flat while the benchmark does the heavy lifting.</span>
          </article>
        </div>
      ) : (
        <p className="benchmark-note">
          Flags open equity positions held longer than {timeStopDays} trading days that are not clearly outperforming SPY.
          If SPY is negative, the table uses excess return instead of a positive-market multiple.
        </p>
      )}

      <div className="time-stop-table-wrap">
        <table className="time-stop-table">
          <thead>
            <tr>
              <th>{sortableHeader("status", "Status")}</th>
              <th>{sortableHeader("symbol", "Symbol")}</th>
              <th>{sortableHeader("entryDate", "Open date")}</th>
              <th>{sortableHeader("daysHeld", "Days held")}</th>
              <th>{sortableHeader("stockReturn", "Stock return")}</th>
              <th>{sortableHeader("spyReturn", "SPY return")}</th>
              <th>{sortableHeader("excessReturn", "Excess vs SPY")}</th>
              <th>{sortableHeader("spyMultiple", "SPY multiple")}</th>
              <th>{sortableHeader("rsSlope", "RS slope")}</th>
              <th>{sortableHeader("flags", "Flags")}</th>
            </tr>
          </thead>
          <tbody>
            {sortedPositions.map((position) => (
              <tr key={position.trade.id}>
                <td>
                  <span className={`time-stop-status ${statusClass(position.status)}`}>{statusLabel(position.status)}</span>
                </td>
                <td>#{position.trade.symbol}</td>
                <td>{position.trade.entryDate}</td>
                <td>{position.daysHeld}</td>
                <td className={returnClass(position.stockReturn)}>{pct(position.stockReturn)}</td>
                <td className={returnClass(position.spyReturn)}>{pct(position.spyReturn)}</td>
                <td className={returnClass(position.excessReturn)}>{pct(position.excessReturn)}</td>
                <td>{multiple(position.spyMultiple)}</td>
                <td>{pct(position.spyRsSlope)}</td>
                <td>{position.flags.length ? position.flags.join(", ") : "No time stop issues"}</td>
              </tr>
            ))}
            {!monitoredPositions.length ? (
              <tr>
                <td className="trade-empty" colSpan={10}>
                  No open equity positions available for time stop monitoring.
                </td>
              </tr>
            ) : null}
          </tbody>
        </table>
      </div>
    </section>
  );
}
