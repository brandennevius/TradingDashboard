"use client";

import { CSSProperties, useEffect, useMemo, useRef, useState } from "react";
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import SpyMarketCycleChart from "@/app/components/SpyMarketCycleChart";
import type { MarketCycleEntry, TraderUser } from "@/lib/types";

type PortfolioSettingsResponse = {
  portfolios?: string[];
  defaultPortfolio?: string;
};

type MarketCycleFormState = {
  date: string;
  trendDay: number | string;
  phase: string;
  notes: string;
};

type SpyCandle = {
  time: string;
  close: number;
};

type MarketCycleSuggestion = {
  date: string;
  trendDay: number;
  phase: (typeof marketCyclePhaseOptions)[number];
  reason: string;
};

const marketCyclePhaseOptions = [
  "Early Uptrend",
  "Middle Uptrend",
  "Late Uptrend",
  "Early Downtrend",
  "Middle Downtrend",
  "Late Downtrend"
] as const;

const marketCyclePhaseColors: Record<(typeof marketCyclePhaseOptions)[number], string> = {
  "Early Uptrend": "#5B8CFF",
  "Middle Uptrend": "#2BB673",
  "Late Uptrend": "#C6A700",
  "Early Downtrend": "#FF8A3D",
  "Middle Downtrend": "#E35D6A",
  "Late Downtrend": "#7A5AF8"
};

const chartTooltipStyle: CSSProperties = {
  background: "rgba(255, 250, 240, 0.96)",
  border: "1px solid #b9d6a8",
  borderRadius: 8,
  boxShadow: "0 16px 34px rgba(86, 100, 72, 0.14)",
  color: "#2f352d"
};

const emptyMarketCycleForm: MarketCycleFormState = {
  date: new Date().toISOString().slice(0, 10),
  trendDay: "",
  phase: "",
  notes: ""
};

function numberValue(value: unknown) {
  const number = Number(value);
  return Number.isFinite(number) ? number : 0;
}

function movingAverage(values: number[], period: number, type: "ema" | "sma") {
  const output = Array<number | null>(values.length).fill(null);
  if (!values.length || period <= 0) return output;

  if (type === "sma") {
    for (let index = period - 1; index < values.length; index += 1) {
      const slice = values.slice(index - period + 1, index + 1);
      output[index] = slice.reduce((sum, value) => sum + value, 0) / period;
    }
    return output;
  }

  const multiplier = 2 / (period + 1);
  let ema = 0;
  values.forEach((value, index) => {
    if (index < period - 1) return;
    if (index === period - 1) {
      ema = values.slice(0, period).reduce((sum, item) => sum + item, 0) / period;
    } else {
      ema = (value - ema) * multiplier + ema;
    }
    output[index] = ema;
  });
  return output;
}

function suggestMarketCycle(candles: SpyCandle[], selectedDate: string): MarketCycleSuggestion | null {
  const validCandles = candles
    .filter((candle) => /^\d{4}-\d{2}-\d{2}$/.test(candle.time) && Number.isFinite(Number(candle.close)))
    .sort((a, b) => a.time.localeCompare(b.time));
  if (!validCandles.length || !selectedDate) return null;

  const index = validCandles.findLastIndex((candle) => candle.time <= selectedDate);
  if (index < 0) return null;

  const closes = validCandles.map((candle) => Number(candle.close));
  const ema8 = movingAverage(closes, 8, "ema");
  const ema21 = movingAverage(closes, 21, "ema");
  const sma50 = movingAverage(closes, 50, "sma");
  const currentEma21 = ema21[index];

  if (!currentEma21) return null;

  const directionFor = (rowIndex: number) => {
    const rowEma21 = ema21[rowIndex];
    if (!rowEma21) return "none";
    return closes[rowIndex] >= rowEma21 ? "up" : "down";
  };

  const direction = directionFor(index);
  if (direction === "none") return null;

  let count = 0;
  for (let rowIndex = index; rowIndex >= 0; rowIndex -= 1) {
    if (directionFor(rowIndex) !== direction) break;
    count += 1;
  }

  const close = closes[index];
  const currentEma8 = ema8[index];
  const currentSma50 = sma50[index];
  const ema8Lookback = ema8[Math.max(0, index - 3)];
  const ema21Lookback = ema21[Math.max(0, index - 5)];
  const ema8Slope = currentEma8 && ema8Lookback ? currentEma8 - ema8Lookback : 0;
  const ema21Slope = currentEma21 && ema21Lookback ? currentEma21 - ema21Lookback : 0;
  const distanceFrom21Pct = Math.abs((close - currentEma21) / currentEma21) * 100;

  let phase: MarketCycleSuggestion["phase"];
  let reason = "";

  if (direction === "up") {
    if (count <= 3 || !currentEma8 || currentEma8 < currentEma21 || Boolean(currentSma50 && close < currentSma50) || ema8Slope <= 0) {
      phase = "Early Uptrend";
      reason = "SPY is reclaiming/holding the 21 EMA, but the trend is still early or not fully stacked.";
    } else if (count >= 18 && (distanceFrom21Pct >= 4 || ema8Slope <= 0 || close < currentEma8)) {
      phase = "Late Uptrend";
      reason = "SPY remains in an uptrend, but the move is extended or momentum is starting to flatten.";
    } else {
      phase = "Middle Uptrend";
      reason = "SPY is above the 8/21/50 area with a constructive moving-average structure.";
    }

    return {
      date: validCandles[index].time,
      trendDay: count,
      phase,
      reason
    };
  }

  if (count <= 3 || !currentEma8 || currentEma8 > currentEma21 || Boolean(currentSma50 && close > currentSma50) || ema8Slope >= 0) {
    phase = "Early Downtrend";
    reason = "SPY is losing/holding below the 21 EMA, but the downtrend is still early or near the 50 SMA.";
  } else if (count >= 18 && (distanceFrom21Pct >= 4 || ema8Slope >= 0 || close > currentEma8)) {
    phase = "Late Downtrend";
    reason = "SPY remains in a downtrend, but the move is extended or downside momentum is maturing.";
  } else {
    phase = "Middle Downtrend";
    reason = "SPY is below the 8/21 area with a confirmed bearish moving-average structure.";
  }

  return {
    date: validCandles[index].time,
    trendDay: -count,
    phase,
    reason
  };
}

function marketCyclePhaseKey(phase: string) {
  return `phase_${phase.toLowerCase().replace(/[^a-z0-9]+/g, "_")}`;
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

export default function BrandenMarketCyclePage() {
  const [user, setUser] = useState<TraderUser | null>(null);
  const [marketCycleEntries, setMarketCycleEntries] = useState<MarketCycleEntry[]>([]);
  const [marketCycleForm, setMarketCycleForm] = useState<MarketCycleFormState>(emptyMarketCycleForm);
  const [spyCandles, setSpyCandles] = useState<SpyCandle[]>([]);
  const [savedPortfolios, setSavedPortfolios] = useState<string[]>([]);
  const [defaultPortfolio, setDefaultPortfolio] = useState("");
  const [status, setStatus] = useState("");
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(true);
  const [isImporting, setIsImporting] = useState(false);
  const cfImportInputRef = useRef<HTMLInputElement | null>(null);
  const pendingImportPortfolioRef = useRef("");

  const canEditBrandenJournal = user?.id === "branden" && !user.readOnly;
  const sidebarActions = canEditBrandenJournal
    ? [
        {
          key: "import-broker-statement",
          label: "Import broker statement",
          icon: "I",
          disabled: isImporting,
          onClick: async () => {
            const targetPortfolio = await choosePortfolioForImport();
            if (targetPortfolio) {
              cfImportInputRef.current?.click();
            }
          }
        }
      ]
    : [];
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
  const marketCycleSuggestion = useMemo(
    () => suggestMarketCycle(spyCandles, marketCycleForm.date),
    [marketCycleForm.date, spyCandles]
  );

  useEffect(() => {
    let cancelled = false;

    async function loadPageData() {
      setIsLoading(true);
      setError("");

      const [entriesResponse, portfoliosResponse] = await Promise.all([
        fetch("/api/settings/market-cycle", { cache: "no-store" }),
        fetch("/api/settings/branden-portfolios", { cache: "no-store" })
      ]);
      const entriesData = await entriesResponse.json().catch(() => ({}));
      const portfoliosData = (await portfoliosResponse.json().catch(() => ({}))) as PortfolioSettingsResponse;

      if (cancelled) {
        return;
      }

      if (!entriesResponse.ok || !entriesData.user) {
        setError(entriesData.error || "Sign in to view Market Cycle.");
      } else {
        setMarketCycleEntries(Array.isArray(entriesData.entries) ? entriesData.entries : []);
        setUser(entriesData.user);
      }

      if (portfoliosResponse.ok) {
        setSavedPortfolios(Array.isArray(portfoliosData.portfolios) ? portfoliosData.portfolios : []);
        setDefaultPortfolio(String(portfoliosData.defaultPortfolio || ""));
      }

      setIsLoading(false);
    }

    loadPageData().catch((loadError) => {
      if (!cancelled) {
        setError(loadError instanceof Error ? loadError.message : "Could not load market cycle.");
        setIsLoading(false);
      }
    });

    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    let cancelled = false;

    async function loadSpyCandles() {
      const response = await fetch("/api/market-data/spy?timeframe=1d", { cache: "no-store" });
      const data = await response.json().catch(() => ({}));

      if (cancelled) return;

      setSpyCandles(
        response.ok && Array.isArray(data.candles)
          ? data.candles
              .map((candle: { time?: unknown; close?: unknown }) => ({
                time: String(candle.time || ""),
                close: numberValue(candle.close)
              }))
              .filter((candle: SpyCandle) => candle.time && candle.close > 0)
          : []
      );
    }

    loadSpyCandles().catch(() => {
      if (!cancelled) setSpyCandles([]);
    });

    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    const existing = marketCycleEntries.find((entry) => entry.date === marketCycleForm.date);

    if (!existing) {
      return;
    }

    setMarketCycleForm((current) =>
      current.date === existing.date &&
      current.trendDay === existing.trendDay &&
      current.notes === existing.notes &&
      current.phase === existing.phase
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
    if (!marketCycleSuggestion) return;
    const existing = marketCycleEntries.find((entry) => entry.date === marketCycleForm.date);
    if (existing) return;

    setMarketCycleForm((current) => {
      if (
        current.date !== marketCycleForm.date ||
        (current.trendDay === marketCycleSuggestion.trendDay && current.phase === marketCycleSuggestion.phase)
      ) {
        return current;
      }

      return {
        ...current,
        trendDay: marketCycleSuggestion.trendDay,
        phase: marketCycleSuggestion.phase
      };
    });
  }, [marketCycleEntries, marketCycleForm.date, marketCycleSuggestion]);

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
    const data = await response.json().catch(() => ({}));

    if (!response.ok) {
      setStatus(data.error || "Could not save market cycle entry.");
      return;
    }

    setMarketCycleEntries(data.entries || []);
    setStatus(`Market cycle entry saved for ${marketCycleForm.date}.`);
  }

  async function choosePortfolioForImport() {
    const defaultValue = defaultPortfolio.trim() || savedPortfolios[0] || "";
    const targetPortfolio = window.prompt("Portfolio for broker statement import", defaultValue);

    if (targetPortfolio === null) {
      return "";
    }

    const normalized = targetPortfolio.trim();

    if (!normalized) {
      setStatus("Choose a portfolio before importing a broker statement.");
      return "";
    }

    pendingImportPortfolioRef.current = normalized;
    return normalized;
  }

  async function importBrokerStatement(files: FileList | null) {
    if (!files?.length || !canEditBrandenJournal) {
      return;
    }

    const targetPortfolio = pendingImportPortfolioRef.current.trim();

    if (!targetPortfolio) {
      setStatus("Choose a portfolio before importing a broker statement.");
      return;
    }

    setIsImporting(true);
    setStatus("Importing broker statement...");

    const formData = new FormData();
    formData.append("file", files[0]);
    formData.append("portfolioTag", targetPortfolio);

    try {
      const response = await fetch("/api/import/cf-statement", {
        method: "POST",
        body: formData
      });
      const data = await response.json().catch(() => ({}));

      if (!response.ok) {
        setStatus(data.error || "Could not import broker statement.");
        return;
      }

      setStatus(
        `Broker statement imported. Open: ${data.openTrades}. Closed: ${data.closedTrades}. Updated: ${data.updated}.`
      );
    } finally {
      pendingImportPortfolioRef.current = "";
      setIsImporting(false);
    }
  }

  return (
    <div className="branden-journal-content">
        <input
          ref={cfImportInputRef}
          className="trade-file-input"
          type="file"
          accept="application/pdf,.pdf"
          disabled={isImporting || !canEditBrandenJournal}
          onChange={(event) => {
            importBrokerStatement(event.target.files);
            event.currentTarget.value = "";
          }}
        />

        <header className="branden-route-header">
          <div>
            <p className="eyebrow">Branden journal</p>
            <h1>Market Cycle</h1>
            <span>{marketCycleEntries.length ? `${marketCycleEntries.length} saved days` : "No saved days yet"}</span>
          </div>
        </header>

        {status ? <p className="status trade-log-status">{status}</p> : null}
        {error ? <p className="status error">{error}</p> : null}
        {isLoading ? <p className="status">Loading market cycle...</p> : null}

        {!isLoading && !error ? (
          <section className="market-cycle-panel">
            <div className="trade-chart-heading">
              <h3>Market cycle journal</h3>
              <span>{canEditBrandenJournal ? "Editable" : "Read only"}</span>
            </div>
            <div className="market-cycle-layout">
              <div className="market-cycle-form">
                <label>
                  Date
                  <input
                    type="date"
                    value={marketCycleForm.date}
                    onChange={(event) => setMarketCycleForm((current) => ({ ...current, date: event.target.value }))}
                    disabled={!canEditBrandenJournal}
                  />
                </label>
                <label>
                  Trend day
                  <input
                    type="number"
                    step="1"
                    value={marketCycleForm.trendDay}
                    onChange={(event) => setMarketCycleForm((current) => ({ ...current, trendDay: event.target.value }))}
                    placeholder="-31 or 1"
                    disabled={!canEditBrandenJournal}
                  />
                </label>
                <label>
                  Phase
                  <select
                    value={marketCycleForm.phase}
                    onChange={(event) => setMarketCycleForm((current) => ({ ...current, phase: event.target.value }))}
                    disabled={!canEditBrandenJournal}
                  >
                    <option value="">Select phase</option>
                    {marketCyclePhaseOptions.map((phase) => (
                      <option key={phase} value={phase}>
                        {phase}
                      </option>
                    ))}
                  </select>
                </label>
                {marketCycleSuggestion ? (
                  <div className="market-cycle-suggestion">
                    <span className="eyebrow">Auto suggestion</span>
                    <strong>
                      {marketCycleSuggestion.trendDay > 0 ? `+${marketCycleSuggestion.trendDay}` : marketCycleSuggestion.trendDay} ·{" "}
                      {marketCycleSuggestion.phase}
                    </strong>
                    <p>
                      Based on SPY through {marketCycleSuggestion.date}. {marketCycleSuggestion.reason}
                    </p>
                    {canEditBrandenJournal ? (
                      <button
                        className="trade-muted-button"
                        type="button"
                        onClick={() =>
                          setMarketCycleForm((current) => ({
                            ...current,
                            trendDay: marketCycleSuggestion.trendDay,
                            phase: marketCycleSuggestion.phase
                          }))
                        }
                      >
                        Use suggestion
                      </button>
                    ) : null}
                  </div>
                ) : null}
                <label className="market-cycle-notes">
                  Market notes
                  <textarea
                    value={marketCycleForm.notes}
                    onChange={(event) => setMarketCycleForm((current) => ({ ...current, notes: event.target.value }))}
                    placeholder="Action, themes, key stocks, context."
                    disabled={!canEditBrandenJournal}
                  />
                </label>
                {canEditBrandenJournal ? (
                  <div className="market-cycle-actions">
                    <button type="button" className="trade-import-button" onClick={saveMarketCycleJournalEntry}>
                      Save market cycle
                    </button>
                  </div>
                ) : null}
              </div>

              <div className="market-cycle-chart-wrap">
                <strong>{marketCycleEntries.length ? `${marketCycleEntries.length} saved days` : "No saved days yet"}</strong>
                <div className="market-cycle-legend">
                  {marketCyclePhaseOptions.map((phase) => (
                    <span key={phase}>
                      <i style={{ backgroundColor: marketCyclePhaseColors[phase] }} />
                      {phase}
                    </span>
                  ))}
                </div>
                <ResponsiveContainer width="100%" height={240}>
                  <LineChart data={marketCycleChartData} margin={{ top: 18, right: 12, bottom: 8, left: 4 }}>
                    <CartesianGrid strokeDasharray="3 5" stroke="rgba(47, 53, 45, 0.16)" vertical={false} />
                    <XAxis dataKey="label" axisLine={false} tickLine={false} tick={{ fill: "#6f7469", fontSize: 11 }} minTickGap={22} />
                    <YAxis axisLine={false} tickLine={false} tick={{ fill: "#6f7469", fontSize: 11 }} width={48} />
                    <Tooltip
                      contentStyle={chartTooltipStyle}
                      formatter={(value) => [String(value), "Trend day"]}
                      labelFormatter={(label, payload) =>
                        `Date: ${label}${payload?.[0]?.payload?.phase ? ` | ${payload[0].payload.phase}` : ""}${payload?.[0]?.payload?.notes ? ` | ${payload[0].payload.notes}` : ""}`
                      }
                    />
                    <Line type="monotone" dataKey="trendDay" stroke="rgba(111, 116, 105, 0.28)" strokeWidth={2} dot={false} activeDot={false} />
                    {marketCyclePhaseOptions.map((phase) => (
                      <Line
                        key={phase}
                        type="monotone"
                        dataKey={marketCyclePhaseKey(phase)}
                        stroke={marketCyclePhaseColors[phase]}
                        strokeWidth={3}
                        dot={{ r: 4, fill: marketCyclePhaseColors[phase], stroke: "#fffaf0", strokeWidth: 1.5 }}
                        activeDot={{ r: 6, fill: marketCyclePhaseColors[phase], stroke: "#fffaf0", strokeWidth: 2 }}
                        connectNulls={false}
                      />
                    ))}
                  </LineChart>
                </ResponsiveContainer>
                <div className="market-cycle-tv-chart">
                  <div className="trade-chart-heading">
                    <h3>SPY with cycle markers</h3>
                    <span>SPY</span>
                  </div>
                  <SpyMarketCycleChart />
                </div>
              </div>
            </div>
            <div className="market-cycle-log">
              <div className="trade-chart-heading">
                <h3>Saved entries</h3>
                <span>{marketCycleEntries.length}</span>
              </div>
              <div className="market-cycle-table-wrap">
                <table className="market-cycle-table">
                  <thead>
                    <tr>
                      <th>Date</th>
                      <th>Cycle</th>
                      <th>Trend day</th>
                      <th>Phase</th>
                      <th>Note</th>
                    </tr>
                  </thead>
                  <tbody>
                    {[...marketCycleEntries].reverse().map((entry) => (
                      <tr key={entry.id}>
                        <td>{entry.date}</td>
                        <td>
                          <span className={`market-cycle-chip ${marketCycleLabel(entry).toLowerCase()}`}>{marketCycleLabel(entry)}</span>
                        </td>
                        <td>{entry.trendDay > 0 ? `+${entry.trendDay}` : entry.trendDay}</td>
                        <td>{entry.phase || "-"}</td>
                        <td>{entry.notes || "-"}</td>
                      </tr>
                    ))}
                    {!marketCycleEntries.length ? (
                      <tr>
                        <td className="trade-empty" colSpan={5}>
                          No market cycle entries saved yet.
                        </td>
                      </tr>
                    ) : null}
                  </tbody>
                </table>
              </div>
            </div>
          </section>
        ) : null}
      </div>
  );
}
