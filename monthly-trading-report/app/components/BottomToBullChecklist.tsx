"use client";

import { useEffect, useMemo, useState } from "react";

type Candle = {
  time: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume?: number;
};
type IndicatorCandle = Candle & {
  ema8: number | null;
  ema21: number | null;
  sma50: number | null;
  sma200: number | null;
};
type ChecklistItem = {
  step: number;
  condition: string;
  passed: boolean;
  triggerDate: string;
  detail: string;
};
type SymbolChecklist = {
  symbol: string;
  latestDate: string;
  score: number;
  total: number;
  items: ChecklistItem[];
};

const symbols = ["SPY"];
const checklistLabels = [
  "Regain 8EMA",
  "Regain 21EMA",
  "O'Neil follow-through day, 4+ days off low",
  "No distribution day 4 days after FTD",
  "8EMA crosses above 21EMA",
  "Regain 50SMA",
  "21EMA crosses above 50SMA",
  "Regain 200SMA",
  "50SMA slope turns positive",
  "50SMA crosses above 200SMA"
];

function average(values: number[]) {
  return values.length ? values.reduce((total, value) => total + value, 0) / values.length : 0;
}

function buildEmaSeries(values: number[], period: number) {
  const series: (number | null)[] = Array(values.length).fill(null);

  if (values.length < period) {
    return series;
  }

  const multiplier = 2 / (period + 1);
  let current = average(values.slice(0, period));
  series[period - 1] = current;

  for (let index = period; index < values.length; index += 1) {
    current = (values[index] - current) * multiplier + current;
    series[index] = current;
  }

  return series;
}

function buildSmaSeries(values: number[], period: number) {
  return values.map((_, index) => {
    if (index < period - 1) {
      return null;
    }

    return average(values.slice(index - period + 1, index + 1));
  });
}

function withIndicators(candles: Candle[]): IndicatorCandle[] {
  const closes = candles.map((candle) => candle.close);
  const ema8 = buildEmaSeries(closes, 8);
  const ema21 = buildEmaSeries(closes, 21);
  const sma50 = buildSmaSeries(closes, 50);
  const sma200 = buildSmaSeries(closes, 200);

  return candles.map((candle, index) => ({
    ...candle,
    ema8: ema8[index],
    ema21: ema21[index],
    sma50: sma50[index],
    sma200: sma200[index]
  }));
}

function triggerDate(candles: IndicatorCandle[], condition: (candle: IndicatorCandle, index: number) => boolean) {
  const latest = candles[candles.length - 1];

  if (!latest || !condition(latest, candles.length - 1)) {
    return "";
  }

  for (let index = candles.length - 1; index >= 0; index -= 1) {
    const currentPassed = condition(candles[index], index);
    const previousPassed = index > 0 ? condition(candles[index - 1], index - 1) : false;

    if (currentPassed && !previousPassed) {
      return candles[index].time;
    }
  }

  return latest.time;
}

function latestLowIndex(candles: IndicatorCandle[], lookback = 90) {
  const start = Math.max(0, candles.length - lookback);
  let lowIndex = start;

  for (let index = start + 1; index < candles.length; index += 1) {
    if (candles[index].low < candles[lowIndex].low) {
      lowIndex = index;
    }
  }

  return lowIndex;
}

function isDistributionDay(candles: IndicatorCandle[], index: number) {
  if (index <= 0) {
    return false;
  }

  const candle = candles[index];
  const previous = candles[index - 1];
  const changePercent = ((candle.close - previous.close) / previous.close) * 100;
  const higherVolume = (candle.volume || 0) > (previous.volume || 0);

  return changePercent <= -0.2 && higherVolume;
}

function findFollowThroughDay(candles: IndicatorCandle[]) {
  const lowIndex = latestLowIndex(candles);

  for (let index = lowIndex + 4; index < candles.length; index += 1) {
    const candle = candles[index];
    const previous = candles[index - 1];
    const gainPercent = ((candle.close - previous.close) / previous.close) * 100;
    const volumeConfirms = !candle.volume || !previous.volume || candle.volume > previous.volume;

    if (gainPercent >= 1.25 && candle.close > candle.open && volumeConfirms) {
      return index;
    }
  }

  return -1;
}

function conditionRow(
  step: number,
  passed: boolean,
  trigger: string,
  detail = ""
): ChecklistItem {
  return {
    step,
    condition: checklistLabels[step - 1],
    passed,
    triggerDate: passed ? trigger : "",
    detail
  };
}

function buildChecklist(symbol: string, rawCandles: Candle[]): SymbolChecklist | null {
  const candles = withIndicators(rawCandles).filter((candle) => candle.close);
  const latest = candles[candles.length - 1];

  if (!latest || candles.length < 200) {
    return null;
  }

  const ftdIndex = findFollowThroughDay(candles);
  const ftdDate = ftdIndex >= 0 ? candles[ftdIndex].time : "";
  const ftdReviewWindowComplete = ftdIndex >= 0 && candles.length - 1 >= ftdIndex + 4;
  const noDistributionAfterFtd =
    ftdReviewWindowComplete &&
    candles.slice(ftdIndex + 1, ftdIndex + 5).every((_, offset) => !isDistributionDay(candles, ftdIndex + 1 + offset));
  const currentIndex = candles.length - 1;
  const sma50SlopePassed = Boolean(latest.sma50 && candles[currentIndex - 5]?.sma50 && latest.sma50 > Number(candles[currentIndex - 5].sma50));

  const items = [
    conditionRow(1, Boolean(latest.ema8 && latest.close > latest.ema8), triggerDate(candles, (candle) => Boolean(candle.ema8 && candle.close > candle.ema8))),
    conditionRow(2, Boolean(latest.ema21 && latest.close > latest.ema21), triggerDate(candles, (candle) => Boolean(candle.ema21 && candle.close > candle.ema21))),
    conditionRow(3, ftdIndex >= 0, ftdDate, "Requires 1.25%+ price thrust 4+ sessions off latest 90-day low; volume confirms when available."),
    conditionRow(4, noDistributionAfterFtd, noDistributionAfterFtd ? candles[ftdIndex + 4].time : "", ftdReviewWindowComplete ? "" : "Waiting for 4 sessions after FTD."),
    conditionRow(5, Boolean(latest.ema8 && latest.ema21 && latest.ema8 > latest.ema21), triggerDate(candles, (candle) => Boolean(candle.ema8 && candle.ema21 && candle.ema8 > candle.ema21))),
    conditionRow(6, Boolean(latest.sma50 && latest.close > latest.sma50), triggerDate(candles, (candle) => Boolean(candle.sma50 && candle.close > candle.sma50))),
    conditionRow(7, Boolean(latest.ema21 && latest.sma50 && latest.ema21 > latest.sma50), triggerDate(candles, (candle) => Boolean(candle.ema21 && candle.sma50 && candle.ema21 > candle.sma50))),
    conditionRow(8, Boolean(latest.sma200 && latest.close > latest.sma200), triggerDate(candles, (candle) => Boolean(candle.sma200 && candle.close > candle.sma200))),
    conditionRow(9, sma50SlopePassed, triggerDate(candles, (candle, index) => Boolean(candle.sma50 && candles[index - 5]?.sma50 && candle.sma50 > Number(candles[index - 5].sma50)))),
    conditionRow(10, Boolean(latest.sma50 && latest.sma200 && latest.sma50 > latest.sma200), triggerDate(candles, (candle) => Boolean(candle.sma50 && candle.sma200 && candle.sma50 > candle.sma200)))
  ];

  return {
    symbol,
    latestDate: latest.time,
    score: items.filter((item) => item.passed).length,
    total: items.length,
    items
  };
}

async function fetchCandles(symbol: string) {
  const response = await fetch(`/api/market-data/${encodeURIComponent(symbol)}?timeframe=1d`, { cache: "no-store" });
  const data = await response.json();

  if (!response.ok) {
    throw new Error(data.error || `Could not load ${symbol}.`);
  }

  return (data.candles || []) as Candle[];
}

export default function BottomToBullChecklist() {
  const [checklists, setChecklists] = useState<SymbolChecklist[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;

    async function loadChecklists() {
      setIsLoading(true);
      setError("");

      try {
        const results = await Promise.all(symbols.map(async (symbol) => buildChecklist(symbol, await fetchCandles(symbol))));

        if (!cancelled) {
          setChecklists(results.filter((item): item is SymbolChecklist => Boolean(item)));
        }
      } catch (loadError) {
        if (!cancelled) {
          setError(loadError instanceof Error ? loadError.message : "Could not load Bottom to Bull checklist.");
        }
      } finally {
        if (!cancelled) {
          setIsLoading(false);
        }
      }
    }

    loadChecklists();

    return () => {
      cancelled = true;
    };
  }, []);

  const aggregate = useMemo(() => {
    const total = checklists.reduce((sum, item) => sum + item.total, 0);
    const score = checklists.reduce((sum, item) => sum + item.score, 0);

    return {
      score,
      total,
      percent: total ? Math.round((score / total) * 100) : 0
    };
  }, [checklists]);

  const primaryChecklist = checklists[0];
  const passedItems = primaryChecklist?.items.filter((item) => item.passed) ?? [];
  const pendingItems = primaryChecklist?.items.filter((item) => !item.passed) ?? [];
  const nextPendingItem = pendingItems[0];

  return (
    <section className="bottom-bull-panel">
      <div className="bottom-bull-heading">
        <div>
          <p className="eyebrow">Market recovery</p>
          <h3>Bottom to Bull Checklist</h3>
        </div>
        <div className="bottom-bull-score">
          <span>Market Recovery Score</span>
          <strong>{aggregate.score}/{aggregate.total || 10}</strong>
        </div>
      </div>

      {isLoading ? <p className="muted">Loading SPY recovery checklist...</p> : null}
      {error ? <p className="status error">{error}</p> : null}

      {!isLoading && !error ? (
        <>
          <div className="bottom-bull-summary">
            <strong>{aggregate.percent}% complete</strong>
            <div>
              <i style={{ width: `${aggregate.percent}%` }} />
            </div>
            <p>Calculated from current daily candles, moving-average regains/crosses, FTD proxy, and post-FTD distribution check.</p>
          </div>

          <div className="bottom-bull-layout">
            <div className="bottom-bull-grid">
              {checklists.map((checklist) => (
                <article className="bottom-bull-card" key={checklist.symbol}>
                  <div className="bottom-bull-card-head">
                    <div>
                      <span className="eyebrow">{checklist.symbol}</span>
                      <h4>{checklist.score}/{checklist.total} complete</h4>
                    </div>
                    <span>{checklist.latestDate}</span>
                  </div>
                  <div className="bottom-bull-table-wrap">
                    <table className="bottom-bull-table">
                      <thead>
                        <tr>
                          <th>Step</th>
                          <th>Condition</th>
                          <th>Status</th>
                          <th>Trigger date</th>
                        </tr>
                      </thead>
                      <tbody>
                        {checklist.items.map((item) => (
                          <tr key={item.step}>
                            <td>{item.step}</td>
                            <td>
                              <strong>{item.condition}</strong>
                              {item.detail ? <span>{item.detail}</span> : null}
                            </td>
                            <td>
                              <span className={item.passed ? "bottom-bull-pass" : "bottom-bull-fail"}>{item.passed ? "Pass" : "Fail"}</span>
                            </td>
                            <td>{item.triggerDate || "-"}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </article>
              ))}
            </div>

            <aside className="bottom-bull-side-panel" aria-label="Market recovery summary">
              <article className="bottom-bull-side-card bottom-bull-side-card-strong">
                <span className="eyebrow">Recovery state</span>
                <strong>{aggregate.percent}% complete</strong>
                <p>{aggregate.score} of {aggregate.total || 10} recovery conditions are currently passing.</p>
              </article>
              <article className="bottom-bull-side-card">
                <span className="eyebrow">Next step</span>
                {nextPendingItem ? (
                  <>
                    <strong>{nextPendingItem.condition}</strong>
                    <p>{nextPendingItem.detail || "Waiting for this condition to trigger."}</p>
                  </>
                ) : (
                  <>
                    <strong>All steps passing</strong>
                    <p>SPY currently meets the full recovery checklist.</p>
                  </>
                )}
              </article>
              <article className="bottom-bull-side-card">
                <span className="eyebrow">Step breakdown</span>
                <div className="bottom-bull-mini-stats">
                  <span><strong>{passedItems.length}</strong> passed</span>
                  <span><strong>{pendingItems.length}</strong> pending</span>
                </div>
              </article>
            </aside>
          </div>
        </>
      ) : null}
    </section>
  );
}
