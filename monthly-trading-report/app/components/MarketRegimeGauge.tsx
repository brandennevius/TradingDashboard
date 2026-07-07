"use client";

import { useEffect, useMemo, useState } from "react";

type Candle = {
  time: string;
  open: number;
  high: number;
  low: number;
  close: number;
};
type TrendState = "Down" | "Neutral" | "Up";
type GaugeState = "Protect" | "Neutral" | "Grow";
type ExtensionState = "Normal" | "Caution" | "Extended";
type SymbolRegime = {
  symbol: string;
  close: number;
  date: string;
  ema21: number;
  sma50: number;
  sma200: number;
  shortTerm: TrendState;
  mediumTerm: TrendState;
  longTerm: TrendState;
  rawShortTerm?: TrendState;
  rawMediumTerm?: TrendState;
  rawLongTerm?: TrendState;
  above21Percent: number;
  above50Percent: number;
  extension: ExtensionState;
};
type GaugeComponent = {
  label: string;
  detail: string;
  state: GaugeState;
  previousState?: GaugeState;
  pendingState?: GaugeState;
};

const indexSymbols = ["SPY", "QQQ", "IWM"];
const defaultLeaderWatchlist = ["NVDA", "MSFT", "META", "AMZN", "GOOGL", "AVGO", "TSLA", "AMD", "PLTR", "CRWD", "COIN", "APP"];
const gaugeStates = ["Protect", "Neutral", "Grow"] as const;
const gaugeRank: Record<GaugeState, number> = { Protect: 0, Neutral: 1, Grow: 2 };

function average(values: number[]) {
  return values.length ? values.reduce((total, value) => total + value, 0) / values.length : 0;
}

function sma(values: number[], period: number) {
  if (values.length < period) {
    return 0;
  }

  return average(values.slice(-period));
}

function ema(values: number[], period: number) {
  if (values.length < period) {
    return 0;
  }

  const multiplier = 2 / (period + 1);
  let current = average(values.slice(0, period));

  for (let index = period; index < values.length; index += 1) {
    current = (values[index] - current) * multiplier + current;
  }

  return current;
}

function percentAbove(value: number, basis: number) {
  return basis ? ((value - basis) / basis) * 100 : 0;
}

function trendState(close: number, averageValue: number): TrendState {
  if (!averageValue) {
    return "Neutral";
  }

  const distance = percentAbove(close, averageValue);

  if (distance > 0.5) {
    return "Up";
  }

  if (distance < -0.5) {
    return "Down";
  }

  return "Neutral";
}

function extensionState(above21Percent: number, above50Percent: number): ExtensionState {
  if (above21Percent >= 8 || above50Percent >= 15) {
    return "Extended";
  }

  if (above21Percent >= 4 || above50Percent >= 8) {
    return "Caution";
  }

  return "Normal";
}

function gaugeStateFromScore(score: number): GaugeState {
  if (score >= 67) {
    return "Grow";
  }

  if (score >= 40) {
    return "Neutral";
  }

  return "Protect";
}

function scoreTrend(state: TrendState) {
  if (state === "Up") {
    return 100;
  }

  if (state === "Neutral") {
    return 50;
  }

  return 0;
}

function scoreExtension(state: ExtensionState) {
  if (state === "Extended") {
    return 20;
  }

  if (state === "Caution") {
    return 55;
  }

  return 85;
}

function buildRegime(symbol: string, candles: Candle[]): SymbolRegime | null {
  const closes = candles.map((candle) => candle.close).filter(Number.isFinite);
  const latest = candles[candles.length - 1];

  if (!latest || closes.length < 200) {
    return null;
  }

  const ema21 = ema(closes, 21);
  const sma50 = sma(closes, 50);
  const sma200 = sma(closes, 200);
  const above21Percent = percentAbove(latest.close, ema21);
  const above50Percent = percentAbove(latest.close, sma50);

  return {
    symbol,
    close: latest.close,
    date: latest.time,
    ema21,
    sma50,
    sma200,
    shortTerm: trendState(latest.close, ema21),
    mediumTerm: trendState(latest.close, sma50),
    longTerm: trendState(latest.close, sma200),
    above21Percent,
    above50Percent,
    extension: extensionState(above21Percent, above50Percent)
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

function stateClass(value: string) {
  return value.toLowerCase().replace(/[^a-z0-9]+/g, "-");
}

function formatPercent(value: number) {
  return `${value >= 0 ? "+" : ""}${value.toFixed(1)}%`;
}

function stateTransition(fromState: GaugeState | undefined, toState: GaugeState | undefined, kind: "confirmed" | "pending") {
  if (!fromState || !toState || fromState === toState) {
    return null;
  }

  const fromRank = gaugeRank[fromState];
  const toRank = gaugeRank[toState];
  const boundary = toRank > fromRank ? toRank * (100 / 3) : fromRank * (100 / 3);
  const width = Math.abs(toRank - fromRank) > 1 ? 18 : 14;
  const left = Math.max(3, Math.min(97 - width, boundary - width / 2));

  return {
    direction: toRank > fromRank ? "upgrade" : "downgrade",
    label: kind === "pending" ? `First close points from ${fromState} toward ${toState}` : `Moved from ${fromState} to ${toState}`,
    style: { left: `${left}%`, width: `${width}%` },
    kind
  };
}

export default function MarketRegimeGauge() {
  const [indexRegimes, setIndexRegimes] = useState<SymbolRegime[]>([]);
  const [leaderRegimes, setLeaderRegimes] = useState<SymbolRegime[]>([]);
  const [apiComponents, setApiComponents] = useState<GaugeComponent[]>([]);
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;

    async function loadRegime() {
      setIsLoading(true);
      setError("");

      try {
        const response = await fetch("/api/market-gauge", { cache: "no-store" });
        const data = await response.json().catch(() => ({}));

        if (!response.ok) {
          throw new Error(data.error || "Could not load market gauge data.");
        }

        if (cancelled) {
          return;
        }

        setIndexRegimes(Array.isArray(data.indexRegimes) ? data.indexRegimes : []);
        setLeaderRegimes(Array.isArray(data.leaderRegimes) ? data.leaderRegimes : []);
        setApiComponents(Array.isArray(data.components) ? data.components : []);
      } catch (loadError) {
        if (!cancelled) {
          setError(loadError instanceof Error ? loadError.message : "Could not load market gauge data.");
        }
      } finally {
        if (!cancelled) {
          setIsLoading(false);
        }
      }
    }

    loadRegime();

    return () => {
      cancelled = true;
    };
  }, []);

  const leadership = useMemo(() => {
    const above21 = leaderRegimes.filter((item) => item.shortTerm === "Up").length;
    const percentAbove21 = leaderRegimes.length ? (above21 / leaderRegimes.length) * 100 : 0;
    const state = gaugeStateFromScore(percentAbove21);

    return {
      above21,
      total: leaderRegimes.length,
      percentAbove21,
      state
    };
  }, [leaderRegimes]);

  const components = useMemo(() => {
    if (apiComponents.length) {
      return apiComponents;
    }

    const shortScore = average(indexRegimes.map((item) => scoreTrend(item.shortTerm)));
    const mediumScore = average(indexRegimes.map((item) => scoreTrend(item.mediumTerm)));
    const longScore = average(indexRegimes.map((item) => scoreTrend(item.longTerm)));
    const extensionScore = average(indexRegimes.map((item) => scoreExtension(item.extension)));

    return [
      { label: "Market leaders", detail: `${leadership.above21}/${leadership.total} leaders above 21EMA`, state: leadership.state },
      { label: "Short term", detail: "Indexes vs 21EMA", state: gaugeStateFromScore(shortScore) },
      { label: "Medium term", detail: "Indexes vs 50SMA", state: gaugeStateFromScore(mediumScore) },
      { label: "Long term", detail: "Indexes vs 200SMA", state: gaugeStateFromScore(longScore) },
      { label: "Extension warning", detail: "% above 21EMA / 50SMA", state: gaugeStateFromScore(extensionScore) }
    ];
  }, [apiComponents, indexRegimes, leadership]);

  const overallState = useMemo(() => gaugeStateFromScore(average(components.map((component) => {
    if (component.state === "Grow") {
      return 100;
    }
    if (component.state === "Neutral") {
      return 50;
    }
    return 0;
  }))), [components]);

  return (
    <section className="market-regime-panel">
      <div className="trade-chart-heading">
        <div>
          <p className="eyebrow">Market gauge</p>
          <h3>Market Gauge</h3>
        </div>
        <span className={`regime-pill ${stateClass(overallState)}`}>{overallState}</span>
      </div>

      {isLoading ? <p className="muted">Loading market gauge data...</p> : null}
      {error ? <p className="status error">{error}</p> : null}

      {!isLoading && !error ? (
        <>
          <div className="growtection-grid">
            {components.map((component) => (
              <article className="growtection-row" key={component.label}>
                <div>
                  <strong>{component.label}</strong>
                  <span>{component.detail}</span>
                </div>
                <div className="growtection-state-cells" aria-label={`${component.label} state`}>
                  {gaugeStates.map((state) => (
                    <span key={state} className={component.state === state ? `active ${stateClass(state)}` : ""}>
                      {state}
                    </span>
                  ))}
                  {(() => {
                    const transition =
                      stateTransition(component.state, component.pendingState, "pending") ||
                      stateTransition(component.previousState, component.state, "confirmed");
                    return transition ? (
                      <i
                        aria-label={`${component.label}: ${transition.label}`}
                        className={`growtection-change ${transition.direction} ${transition.kind}`}
                        role="img"
                        style={transition.style}
                        title={transition.label}
                      />
                    ) : null;
                  })()}
                </div>
              </article>
            ))}
          </div>

          <div className="regime-kpi-grid">
            <article>
              <span>Leadership health</span>
              <strong>{leadership.percentAbove21.toFixed(0)}%</strong>
              <p>{leadership.above21} of {leadership.total} watchlist leaders above their 21EMA</p>
            </article>
            <article>
              <span>Index short term</span>
              <strong>{indexRegimes.filter((item) => item.shortTerm === "Up").length}/{indexRegimes.length}</strong>
              <p>SPY, QQQ, and IWM above 21EMA</p>
            </article>
            <article>
              <span>Extension</span>
              <strong>{indexRegimes.filter((item) => item.extension === "Extended").length ? "Watch" : "OK"}</strong>
              <p>Flags if indexes are stretched above 21EMA or 50SMA</p>
            </article>
          </div>

          <div className="regime-table-wrap">
            <table className="regime-table">
              <thead>
                <tr>
                  <th>Index</th>
                  <th>Close</th>
                  <th>21EMA</th>
                  <th>50SMA</th>
                  <th>200SMA</th>
                  <th>Short</th>
                  <th>Medium</th>
                  <th>Long</th>
                  <th>Extension</th>
                </tr>
              </thead>
              <tbody>
                {indexRegimes.map((item) => (
                  <tr key={item.symbol}>
                    <td>{item.symbol}</td>
                    <td>{item.close.toFixed(2)}</td>
                    <td>{item.ema21.toFixed(2)} <span>{formatPercent(item.above21Percent)}</span></td>
                    <td>{item.sma50.toFixed(2)} <span>{formatPercent(item.above50Percent)}</span></td>
                    <td>{item.sma200.toFixed(2)}</td>
                    <td><span className={`regime-mini-pill ${stateClass(item.shortTerm)}`}>{item.shortTerm}</span></td>
                    <td><span className={`regime-mini-pill ${stateClass(item.mediumTerm)}`}>{item.mediumTerm}</span></td>
                    <td><span className={`regime-mini-pill ${stateClass(item.longTerm)}`}>{item.longTerm}</span></td>
                    <td><span className={`regime-mini-pill ${stateClass(item.extension)}`}>{item.extension}</span></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="leader-health-list">
            {leaderRegimes.map((item) => (
              <span key={item.symbol} className={item.shortTerm === "Up" ? "healthy" : item.shortTerm === "Neutral" ? "neutral" : "weak"}>
                {item.symbol}
              </span>
            ))}
          </div>
        </>
      ) : null}
    </section>
  );
}
