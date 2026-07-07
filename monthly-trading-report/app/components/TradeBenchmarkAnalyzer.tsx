"use client";

import { useEffect, useMemo, useState } from "react";
import type { TradeLogEntry, TradeStatus } from "@/lib/types";

type Props = {
  trades: TradeLogEntry[];
  activePortfolio: string;
};

type Candle = {
  time: string;
  open: number;
  close: number;
};

type BenchmarkBucketKey = "under" | "zeroOne" | "oneTwo" | "twoThree" | "threeFive" | "fiveTen" | "tenPlus";
type BenchmarkSortKey = "symbol" | "result" | "held" | "stockReturn" | "spyReturn" | "spyMultiple" | "spyPrice";
type SortDirection = "asc" | "desc";

type AnalyzedTrade = {
  trade: TradeLogEntry;
  stockReturn: number;
  spyReturn: number;
  spyMultiple: number | null;
  spyBucket: BenchmarkBucketKey;
  entrySpy: number;
  exitSpy: number;
  missingData: string[];
};

const bucketDefinitions: Array<{ key: BenchmarkBucketKey; label: string; meaning: string }> = [
  { key: "under", label: "< SPY", meaning: "Negative or worse than benchmark" },
  { key: "zeroOne", label: "0-1x", meaning: "Positive, but did not keep up" },
  { key: "oneTwo", label: "1-2x", meaning: "Beat, but weak" },
  { key: "twoThree", label: "2-3x", meaning: "Meets minimum target" },
  { key: "threeFive", label: "3-5x", meaning: "Strong" },
  { key: "fiveTen", label: "5-10x", meaning: "Excellent" },
  { key: "tenPlus", label: "10x+", meaning: "Monster" }
];
const visibleBucketDefinitions = bucketDefinitions;

const outcomeRows: TradeStatus[] = ["WIN", "LOSS", "BREAKEVEN"];
const resultRank: Record<string, number> = { WIN: 0, BREAKEVEN: 1, LOSS: 2 };

function compareText(a: string, b: string) {
  return a.localeCompare(b, undefined, { numeric: true, sensitivity: "base" });
}

function compareNumber(a: number, b: number) {
  const safeA = Number.isFinite(a) ? a : Number.NEGATIVE_INFINITY;
  const safeB = Number.isFinite(b) ? b : Number.NEGATIVE_INFINITY;
  return safeA - safeB;
}

function defaultSortDirection(key: BenchmarkSortKey): SortDirection {
  return ["held", "stockReturn", "spyReturn", "spyMultiple", "spyPrice"].includes(key) ? "desc" : "asc";
}

function benchmarkSortValue(key: BenchmarkSortKey, item: AnalyzedTrade) {
  switch (key) {
    case "symbol":
      return item.trade.symbol;
    case "result":
      return resultRank[item.trade.status] ?? 99;
    case "held":
      return item.trade.exitDate || item.trade.entryDate || "";
    case "stockReturn":
      return item.stockReturn;
    case "spyReturn":
      return item.spyReturn;
    case "spyMultiple":
      return item.spyMultiple ?? Number.NEGATIVE_INFINITY;
    case "spyPrice":
      return item.exitSpy;
    default:
      return "";
  }
}

function compareBenchmarkTrades(key: BenchmarkSortKey, a: AnalyzedTrade, b: AnalyzedTrade) {
  const aValue = benchmarkSortValue(key, a);
  const bValue = benchmarkSortValue(key, b);
  const result =
    typeof aValue === "number" && typeof bValue === "number"
      ? compareNumber(aValue, bValue)
      : compareText(String(aValue), String(bValue));

  return result || compareText(a.trade.symbol, b.trade.symbol) || compareText(a.trade.id, b.trade.id);
}

function pct(value: number) {
  if (!Number.isFinite(value)) {
    return "-";
  }

  return `${value >= 0 ? "+" : ""}${value.toFixed(2)}%`;
}

function multiple(value: number | null) {
  if (value === Number.POSITIVE_INFINITY) {
    return "+∞x";
  }

  if (value === Number.NEGATIVE_INFINITY) {
    return "-∞x";
  }

  if (value === null || !Number.isFinite(value)) {
    return "N/A";
  }

  return `${value.toFixed(2)}x`;
}

function formatPrice(value: number) {
  if (!Number.isFinite(value) || !value) {
    return "-";
  }

  return new Intl.NumberFormat("en-US", {
    maximumFractionDigits: 2
  }).format(value);
}

function isoDaysAgo(days: number) {
  const date = new Date();
  date.setDate(date.getDate() - days);
  return date.toISOString().slice(0, 10);
}

function isEquityLikeSymbol(symbol: string) {
  const cleaned = symbol.trim().replace(/^#/, "").toUpperCase();
  return Boolean(cleaned) && /^[A-Z]{1,6}$/.test(cleaned) && !["SPY", "QQQ"].includes(cleaned);
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

function tradeStockReturn(trade: TradeLogEntry) {
  const entry = Number(trade.avgEntry) || 0;
  const exit = Number(trade.exitPrice) || 0;
  const importedReturn = Number(trade.returnPercent);

  if (entry && exit) {
    return trade.side === "SHORT" ? ((entry - exit) / entry) * 100 : ((exit - entry) / entry) * 100;
  }

  return Number.isFinite(importedReturn) ? importedReturn : 0;
}

function benchmarkPrices(entry: Candle | null, exit: Candle | null, isSameDay: boolean) {
  const entryPrice = isSameDay ? Number(entry?.open) : Number(entry?.close);
  const exitPrice = Number(exit?.close);

  if (!entryPrice || !exitPrice) {
    return { entryPrice: 0, exitPrice: 0, returnPercent: 0 };
  }

  return {
    entryPrice,
    exitPrice,
    returnPercent: ((exitPrice - entryPrice) / entryPrice) * 100
  };
}

function bucketFor(relativeMultiple: number): BenchmarkBucketKey {
  if (relativeMultiple < 0) {
    return "under";
  }

  if (relativeMultiple < 1) {
    return "zeroOne";
  }

  if (relativeMultiple < 2) {
    return "oneTwo";
  }

  if (relativeMultiple < 3) {
    return "twoThree";
  }

  if (relativeMultiple < 5) {
    return "threeFive";
  }

  if (relativeMultiple < 10) {
    return "fiveTen";
  }

  return "tenPlus";
}

function relativeMultiple(stockReturn: number, benchmarkReturn: number) {
  if (benchmarkReturn > 0) {
    return stockReturn / benchmarkReturn;
  }

  if (benchmarkReturn < 0) {
    return (stockReturn - benchmarkReturn) / Math.abs(benchmarkReturn);
  }

  if (stockReturn > 0) {
    return Number.POSITIVE_INFINITY;
  }

  if (stockReturn < 0) {
    return Number.NEGATIVE_INFINITY;
  }

  return 0;
}

async function fetchCandles(symbol: string) {
  const response = await fetch(`/api/market-data/${encodeURIComponent(symbol)}?timeframe=1d`, { cache: "no-store" });
  const data = await response.json().catch(() => ({}));

  if (!response.ok) {
    throw new Error(data.error || `Could not load ${symbol}.`);
  }

  return ((data.candles || []) as Candle[]).filter(
    (candle) =>
      candle.time &&
      Number.isFinite(Number(candle.open)) &&
      Number(candle.open) > 0 &&
      Number.isFinite(Number(candle.close)) &&
      Number(candle.close) > 0
  );
}

function analyzeTrade(trade: TradeLogEntry, spyCandles: Candle[]): AnalyzedTrade {
  const entryDate = trade.entryDate;
  const exitDate = trade.exitDate || trade.entryDate;
  const stockReturn = tradeStockReturn(trade);
  const spyEntry = findCandleOnOrBefore(spyCandles, entryDate);
  const spyExit = findCandleOnOrBefore(spyCandles, exitDate);
  const spyPrices = benchmarkPrices(spyEntry, spyExit, entryDate === exitDate);
  const spyReturn = spyPrices.returnPercent;
  const spyMultiple = relativeMultiple(stockReturn, spyReturn);
  const missingData = [
    !trade.avgEntry && !trade.returnPercent ? "stock return" : "",
    !spyEntry || !spyExit ? "SPY dates" : ""
  ].filter(Boolean);

  return {
    trade,
    stockReturn,
    spyReturn,
    spyMultiple,
    spyBucket: bucketFor(spyMultiple),
    entrySpy: spyPrices.entryPrice,
    exitSpy: spyPrices.exitPrice,
    missingData
  };
}

function emptyBucketCounts() {
  return Object.fromEntries(bucketDefinitions.map((bucket) => [bucket.key, 0])) as Record<BenchmarkBucketKey, number>;
}

function buildBucketMatrix(trades: AnalyzedTrade[]) {
  return outcomeRows.map((status) => {
    const statusTrades = trades.filter((item) => item.trade.status === status);
    const counts = emptyBucketCounts();

    statusTrades.forEach((item) => {
      counts[item.spyBucket] += 1;
    });

    return { status, total: statusTrades.length, counts };
  });
}

function buildDistribution(trades: AnalyzedTrade[]) {
  const counts = emptyBucketCounts();

  trades.forEach((item) => {
    counts[item.spyBucket] += 1;
  });

  const max = Math.max(1, ...Object.values(counts));
  return bucketDefinitions.map((bucket) => ({
    ...bucket,
    count: counts[bucket.key],
    width: `${(counts[bucket.key] / max) * 100}%`
  }));
}

export default function TradeBenchmarkAnalyzer({ trades, activePortfolio }: Props) {
  const [spyCandles, setSpyCandles] = useState<Candle[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState("");
  const [startDate, setStartDate] = useState("");
  const [endDate, setEndDate] = useState("");
  const [sort, setSort] = useState<{ key: BenchmarkSortKey; direction: SortDirection } | null>(null);

  const closedEquityTrades = useMemo(
    () =>
      trades
        .filter((trade) => trade.status !== "OPEN")
        .filter((trade) => !activePortfolio || trade.portfolioTag === activePortfolio)
        .filter((trade) => !startDate || trade.exitDate >= startDate)
        .filter((trade) => !endDate || trade.exitDate <= endDate)
        .filter((trade) => isEquityLikeSymbol(trade.symbol) && trade.entryDate && trade.exitDate)
        .sort((a, b) => b.exitDate.localeCompare(a.exitDate)),
    [trades, activePortfolio, startDate, endDate]
  );

  useEffect(() => {
    let cancelled = false;

    async function loadBenchmarks() {
      setIsLoading(true);
      setError("");

      try {
        const spy = await fetchCandles("SPY");

        if (!cancelled) {
          setSpyCandles(spy);
        }
      } catch (loadError) {
        if (!cancelled) {
          setError(loadError instanceof Error ? loadError.message : "Could not load benchmark candles.");
        }
      } finally {
        if (!cancelled) {
          setIsLoading(false);
        }
      }
    }

    loadBenchmarks();

    return () => {
      cancelled = true;
    };
  }, []);

  const analyzedTrades = useMemo(() => {
    if (!spyCandles.length) {
      return [];
    }

    return closedEquityTrades.map((trade) => analyzeTrade(trade, spyCandles));
  }, [closedEquityTrades, spyCandles]);

  const targetHitsSpy = analyzedTrades.filter((item) => item.spyMultiple !== null && item.spyMultiple >= 2).length;
  const spyMultipleTrades = analyzedTrades.filter(
    (item) => item.spyMultiple !== null && Number.isFinite(item.spyMultiple)
  );
  const averageSpyMultiple = spyMultipleTrades.length
    ? spyMultipleTrades.reduce((sum, item) => sum + Number(item.spyMultiple), 0) / spyMultipleTrades.length
    : 0;
  const matrix = buildBucketMatrix(analyzedTrades);
  const distribution = buildDistribution(analyzedTrades);
  const displayedTrades = useMemo(
    () =>
      [...analyzedTrades].sort((a, b) => {
        if (!sort) {
          return compareText(b.trade.exitDate || b.trade.entryDate || "", a.trade.exitDate || a.trade.entryDate || "");
        }

        return compareBenchmarkTrades(sort.key, a, b) * (sort.direction === "asc" ? 1 : -1);
      }),
    [analyzedTrades, sort]
  );

  function toggleSort(key: BenchmarkSortKey) {
    setSort((current) => {
      if (current?.key === key) {
        return { key, direction: current.direction === "asc" ? "desc" : "asc" };
      }

      return { key, direction: defaultSortDirection(key) };
    });
  }

  function sortHeader(key: BenchmarkSortKey, label: string) {
    const active = sort?.key === key;
    const direction = active ? sort.direction : defaultSortDirection(key);

    return (
      <button
        aria-label={`Sort by ${label} ${direction === "asc" ? "ascending" : "descending"}`}
        aria-sort={active ? (sort.direction === "asc" ? "ascending" : "descending") : "none"}
        className={`benchmark-sort-button${active ? " active" : ""}`}
        onClick={() => toggleSort(key)}
        type="button"
      >
        <span>{label}</span>
        <span aria-hidden="true">{active ? (sort.direction === "asc" ? "▲" : "▼") : "↕"}</span>
      </button>
    );
  }

  return (
    <section className="benchmark-panel">
      <div className="column-settings-head">
        <div>
          <p className="eyebrow">Stock vs S&P Benchmark Tracker</p>
          <h3>Trade Benchmark Analyzer</h3>
        </div>
        <span className="benchmark-static-pill">SPY</span>
      </div>

      <div className="benchmark-filter-bar">
        <label>
          Closed from
          <input type="date" value={startDate} onChange={(event) => setStartDate(event.target.value)} />
        </label>
        <label>
          Closed through
          <input type="date" value={endDate} onChange={(event) => setEndDate(event.target.value)} />
        </label>
        <button
          type="button"
          onClick={() => {
            setStartDate(isoDaysAgo(7));
            setEndDate(new Date().toISOString().slice(0, 10));
          }}
        >
          Last week
        </button>
        <button
          type="button"
          onClick={() => {
            setStartDate(isoDaysAgo(30));
            setEndDate(new Date().toISOString().slice(0, 10));
          }}
        >
          Last month
        </button>
        <button
          type="button"
          onClick={() => {
            setStartDate("");
            setEndDate("");
          }}
        >
          Clear dates
        </button>
      </div>

      <div className="benchmark-kpis">
        <article>
          <span>Closed equity trades</span>
          <strong>{analyzedTrades.length}</strong>
        </article>
        <article>
          <span>2x+ vs SPY</span>
          <strong>{targetHitsSpy}</strong>
        </article>
        <article>
          <span>Avg SPY relative score</span>
          <strong>{multiple(averageSpyMultiple || null)}</strong>
        </article>
      </div>

      {isLoading ? <p className="benchmark-note">Loading SPY benchmark candles...</p> : null}
      {error ? <p className="benchmark-error">{error}</p> : null}
      <p className="benchmark-note">
        Same-day trades use SPY open-to-close. Multi-day trades use the entry-day close through the exit-day close.
        SPY above 0%: stock return ÷ SPY return. SPY below 0%: excess return ÷ the size of SPY&apos;s decline.
      </p>

      <div className="benchmark-grid">
        <article className="benchmark-card">
          <div className="benchmark-card-head">
            <h4>SPY outcome bucket matrix</h4>
            <span>{activePortfolio ? activePortfolio : "All portfolios"}</span>
          </div>
          <div className="benchmark-table-wrap">
            <table className="benchmark-table">
              <thead>
                <tr>
                  <th>Result</th>
                  <th>Total</th>
                  {visibleBucketDefinitions.map((bucket) => (
                    <th key={bucket.key}>{bucket.label}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {matrix.map((row) => (
                  <tr key={row.status}>
                    <td>{row.status}</td>
                    <td>{row.total}</td>
                    {visibleBucketDefinitions.map((bucket) => (
                      <td key={bucket.key}>{row.counts[bucket.key] || ""}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </article>

        <article className="benchmark-card">
          <div className="benchmark-card-head">
            <h4>SPY relative multiple distribution</h4>
            <span>Target: 2-3x minimum</span>
          </div>
          <div className="benchmark-bars">
            {distribution.map((bucket) => (
              <div className="benchmark-bar-row" key={bucket.key}>
                <div>
                  <strong>{bucket.label}</strong>
                  <span>{bucket.meaning}</span>
                </div>
                <i>
                  <b style={{ width: bucket.width }} />
                </i>
                <em>{bucket.count}</em>
              </div>
            ))}
          </div>
        </article>
      </div>

      <article className="benchmark-card">
        <div className="benchmark-card-head">
          <h4>Filtered trades</h4>
          <span>{analyzedTrades.length} shown</span>
        </div>
        <div className="benchmark-table-wrap">
          <table className="benchmark-table benchmark-detail-table">
            <thead>
              <tr>
                <th>{sortHeader("symbol", "Symbol")}</th>
                <th>{sortHeader("result", "Result")}</th>
                <th>{sortHeader("held", "Held")}</th>
                <th>{sortHeader("stockReturn", "Stock return")}</th>
                <th>{sortHeader("spyReturn", "SPY return")}</th>
                <th>{sortHeader("spyMultiple", "SPY relative score")}</th>
                <th>{sortHeader("spyPrice", "SPY entry/exit")}</th>
              </tr>
            </thead>
            <tbody>
              {displayedTrades.map((item) => (
                <tr key={item.trade.id}>
                  <td>{item.trade.symbol}</td>
                  <td>{item.trade.status}</td>
                  <td>
                    {item.trade.entryDate} to {item.trade.exitDate}
                  </td>
                  <td>{pct(item.stockReturn)}</td>
                  <td>{pct(item.spyReturn)}</td>
                  <td>{multiple(item.spyMultiple)}</td>
                  <td>
                    {formatPrice(item.entrySpy)} / {formatPrice(item.exitSpy)}
                  </td>
                </tr>
              ))}
              {!analyzedTrades.length ? (
                <tr>
                  <td className="trade-empty" colSpan={7}>
                    No closed equity trades available for benchmark analysis.
                  </td>
                </tr>
              ) : null}
            </tbody>
          </table>
        </div>
      </article>
    </section>
  );
}
