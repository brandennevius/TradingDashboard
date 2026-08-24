"use client";

import Link from "next/link";
import { FormEvent, useEffect, useMemo, useState } from "react";

type ExitStrategySummary = {
  strategy: string;
  category: string;
  trades: number;
  totalR: number;
  averageR: number;
  medianR: number;
  winRate: number;
  maxDrawdownR: number;
  markedOpen: number;
  deltaVsActualR: number;
  earlierDeltaVsActualR: number;
  recentDeltaVsActualR: number;
  earlierAverageR: number;
  recentAverageR: number;
  performanceToDrawdown: number;
};

type ExitAnalysisResponse = {
  error?: string;
  options?: { setups?: string[]; portfolios?: string[] };
  coverage?: {
    closedTrades: number;
    analyzedTrades: number;
    missingRisk: number;
    incompleteExecutions: number;
    unsupportedSymbols: number;
    noMarketData: number;
  };
  metrics?: {
    actualTotalR: number;
    averageMfeR: number;
    averageMaeR: number;
    averageGivebackR: number;
    averageCaptureRate: number;
    reached2R: number;
    reached3R: number;
    highGiveback: number;
    halfAt3RTotal: number;
    halfAt3RDelta: number;
  };
  summaries?: ExitStrategySummary[];
  recommendation?: {
    suggested: ExitStrategySummary | null;
    bestRaw: ExitStrategySummary | null;
    bestRiskAdjusted: ExitStrategySummary | null;
    rationale: string;
  };
};

type AnalysisFilters = {
  startDate: string;
  endDate: string;
  setup: string;
  portfolio: string;
};

const emptyFilters: AnalysisFilters = { startDate: "", endDate: "", setup: "", portfolio: "" };

function localDate() {
  const date = new Date();
  return `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, "0")}-${String(date.getDate()).padStart(2, "0")}`;
}

function daysAgo(days: number) {
  const date = new Date();
  date.setDate(date.getDate() - days);
  return `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, "0")}-${String(date.getDate()).padStart(2, "0")}`;
}

function yearStart() {
  return `${new Date().getFullYear()}-01-01`;
}

function rValue(value: number) {
  return `${value >= 0 ? "+" : ""}${value.toFixed(2)}R`;
}

function percentValue(value: number) {
  return `${(value * 100).toFixed(1)}%`;
}

export default function BrandenExitAnalysisPage() {
  const [draftFilters, setDraftFilters] = useState<AnalysisFilters>(emptyFilters);
  const [appliedFilters, setAppliedFilters] = useState<AnalysisFilters>(emptyFilters);
  const [data, setData] = useState<ExitAnalysisResponse>({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    const controller = new AbortController();
    const params = new URLSearchParams();
    if (appliedFilters.startDate) params.set("start", appliedFilters.startDate);
    if (appliedFilters.endDate) params.set("end", appliedFilters.endDate);
    if (appliedFilters.setup) params.set("setup", appliedFilters.setup);
    if (appliedFilters.portfolio) params.set("portfolio", appliedFilters.portfolio);

    setLoading(true);
    setError("");
    fetch(`/api/journal/branden/exit-analysis?${params.toString()}`, { cache: "no-store", signal: controller.signal })
      .then(async (response) => {
        const payload = (await response.json().catch(() => ({}))) as ExitAnalysisResponse;
        if (!response.ok) throw new Error(payload.error || "Could not load exit analysis.");
        setData(payload);
      })
      .catch((loadError) => {
        if (loadError instanceof DOMException && loadError.name === "AbortError") return;
        setError(loadError instanceof Error ? loadError.message : "Could not load exit analysis.");
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });

    return () => controller.abort();
  }, [appliedFilters]);

  const summaries = data.summaries || [];
  const coverage = data.coverage;
  const metrics = data.metrics;
  const recommendation = data.recommendation;
  const displayedSummaries = useMemo(() => [...summaries].sort((a, b) => {
    if (a.strategy === "Actual") return -1;
    if (b.strategy === "Actual") return 1;
    if (a.strategy === recommendation?.suggested?.strategy) return -1;
    if (b.strategy === recommendation?.suggested?.strategy) return 1;
    return b.totalR - a.totalR;
  }), [summaries, recommendation?.suggested?.strategy]);

  function applyFilters(event: FormEvent) {
    event.preventDefault();
    setAppliedFilters({ ...draftFilters });
  }

  function setRange(startDate: string, endDate: string) {
    const next = { ...draftFilters, startDate, endDate };
    setDraftFilters(next);
    setAppliedFilters(next);
  }

  return (
    <div className="branden-journal-content exit-lab-page">
      <header className="branden-route-header">
        <div>
          <p className="eyebrow">Branden journal</p>
          <h1>Exit Analysis</h1>
          <span>Aggregate closed-trade exit strategy comparison</span>
        </div>
        <nav className="branden-route-nav" aria-label="Branden journal pages">
          <Link href="/journal/branden/dashboard">Dashboard</Link>
          <Link href="/journal/branden/trade-log">Trade Log</Link>
          <Link className="active" href="/journal/branden/exit-analysis">Exit Analysis</Link>
        </nav>
      </header>

      <form className="exit-lab-filters" onSubmit={applyFilters}>
        <label>
          Start date
          <input
            type="date"
            value={draftFilters.startDate}
            onChange={(event) => setDraftFilters((current) => ({ ...current, startDate: event.target.value }))}
          />
        </label>
        <label>
          End date
          <input
            type="date"
            value={draftFilters.endDate}
            onChange={(event) => setDraftFilters((current) => ({ ...current, endDate: event.target.value }))}
          />
        </label>
        <label>
          Setup
          <select
            value={draftFilters.setup}
            onChange={(event) => setDraftFilters((current) => ({ ...current, setup: event.target.value }))}
          >
            <option value="">All setups</option>
            {(data.options?.setups || []).map((setup) => <option key={setup} value={setup}>{setup}</option>)}
          </select>
        </label>
        <label>
          Portfolio
          <select
            value={draftFilters.portfolio}
            onChange={(event) => setDraftFilters((current) => ({ ...current, portfolio: event.target.value }))}
          >
            <option value="">All portfolios</option>
            {(data.options?.portfolios || []).map((portfolio) => <option key={portfolio} value={portfolio}>{portfolio}</option>)}
          </select>
        </label>
        <button type="submit">Analyze</button>
        <div className="exit-lab-range-buttons">
          <button className="trade-muted-button" type="button" onClick={() => setRange(daysAgo(30), localDate())}>30 days</button>
          <button className="trade-muted-button" type="button" onClick={() => setRange(daysAgo(90), localDate())}>90 days</button>
          <button className="trade-muted-button" type="button" onClick={() => setRange(yearStart(), localDate())}>YTD</button>
          <button className="trade-muted-button" type="button" onClick={() => setRange("", "")}>All</button>
        </div>
      </form>

      {loading ? <section className="exit-lab-loading"><strong>Analyzing closed trades...</strong><span>Loading executions and cached daily market data.</span></section> : null}
      {error ? <p className="status error">{error}</p> : null}

      {!loading && !error && coverage && metrics ? (
        <>
          <section className="exit-lab-coverage">
            <div><span>Closed trades</span><strong>{coverage.closedTrades}</strong></div>
            <div><span>Analyzed</span><strong>{coverage.analyzedTrades}</strong></div>
            <div><span>Actual total</span><strong className={metrics.actualTotalR >= 0 ? "trade-positive" : "trade-negative"}>{rValue(metrics.actualTotalR)}</strong></div>
            <div><span>Average MFE</span><strong>{rValue(metrics.averageMfeR)}</strong></div>
            <div><span>Average MAE</span><strong>{rValue(metrics.averageMaeR)}</strong></div>
            <div><span>Average giveback</span><strong>{rValue(metrics.averageGivebackR)}</strong></div>
            <div><span>Average captured</span><strong>{percentValue(metrics.averageCaptureRate)}</strong></div>
          </section>

          <section className="exit-lab-insight">
            <div>
              <p className="eyebrow">Profit protection test</p>
              <h2>50% at 3R</h2>
              <strong className={metrics.halfAt3RDelta >= 0 ? "trade-positive" : "trade-negative"}>{rValue(metrics.halfAt3RTotal)}</strong>
              <span>{rValue(metrics.halfAt3RDelta)} versus actual across {metrics.reached3R} trades that reached 3R.</span>
            </div>
            <dl>
              <div><dt>Reached 2R</dt><dd>{metrics.reached2R}</dd></div>
              <div><dt>Reached 3R</dt><dd>{metrics.reached3R}</dd></div>
              <div><dt>High giveback</dt><dd>{metrics.highGiveback}</dd></div>
            </dl>
          </section>

          {recommendation ? (
            <section className="exit-lab-recommendation">
              <div className="exit-lab-recommendation-copy">
                <p className="eyebrow">Historical fit</p>
                <h2>{recommendation.suggested?.strategy || "Keep actual exits as baseline"}</h2>
                <p>{recommendation.rationale}</p>
                <span>Use this as a forward-test candidate, not proof that it will outperform future trades.</span>
              </div>
              <div className="exit-lab-picks">
                <div>
                  <span>Suggested test</span>
                  <strong>{recommendation.suggested?.strategy || "No robust alternative"}</strong>
                  <small>{recommendation.suggested ? `${rValue(recommendation.suggested.totalR)} total · ${rValue(recommendation.suggested.deltaVsActualR)} vs actual` : "Actual exits remain the benchmark"}</small>
                </div>
                <div>
                  <span>Best raw return</span>
                  <strong>{recommendation.bestRaw?.strategy || "—"}</strong>
                  <small>{recommendation.bestRaw ? `${rValue(recommendation.bestRaw.totalR)} · ${recommendation.bestRaw.markedOpen} unresolved` : "—"}</small>
                </div>
                <div>
                  <span>Best R / drawdown</span>
                  <strong>{recommendation.bestRiskAdjusted?.strategy || "—"}</strong>
                  <small>{recommendation.bestRiskAdjusted ? `${recommendation.bestRiskAdjusted.performanceToDrawdown.toFixed(2)} ratio · ${rValue(recommendation.bestRiskAdjusted.maxDrawdownR)} drawdown` : "—"}</small>
                </div>
              </div>
            </section>
          ) : null}

          <section className="exit-lab-results">
            <div className="trade-chart-heading">
              <div>
                <p className="eyebrow">All analyzed trades</p>
                <h2>Strategy comparison</h2>
              </div>
              <span>Earlier and recent compare each strategy with your actual exits in each half of the selected trades.</span>
            </div>
            {summaries.length ? (
              <div className="exit-lab-table-wrap">
                <table className="exit-lab-table">
                  <thead>
                    <tr>
                      <th>Exit method</th>
                      <th>Type</th>
                      <th>Total R</th>
                      <th>Vs actual</th>
                      <th>Earlier</th>
                      <th>Recent</th>
                      <th>Average R</th>
                      <th>Median R</th>
                      <th>Win rate</th>
                      <th>Max drawdown</th>
                      <th>Unresolved</th>
                    </tr>
                  </thead>
                  <tbody>
                    {displayedSummaries.map((summary) => (
                      <tr key={summary.strategy} className={summary.strategy === "Actual" ? "actual" : summary.strategy === recommendation?.suggested?.strategy ? "suggested" : ""}>
                        <td>{summary.strategy}</td>
                        <td>{summary.category}</td>
                        <td className={summary.totalR >= 0 ? "trade-positive" : "trade-negative"}>{rValue(summary.totalR)}</td>
                        <td className={summary.deltaVsActualR >= 0 ? "trade-positive" : "trade-negative"}>{summary.strategy === "Actual" ? "Baseline" : rValue(summary.deltaVsActualR)}</td>
                        <td className={summary.earlierDeltaVsActualR >= 0 ? "trade-positive" : "trade-negative"}>{summary.strategy === "Actual" ? "Baseline" : rValue(summary.earlierDeltaVsActualR)}</td>
                        <td className={summary.recentDeltaVsActualR >= 0 ? "trade-positive" : "trade-negative"}>{summary.strategy === "Actual" ? "Baseline" : rValue(summary.recentDeltaVsActualR)}</td>
                        <td>{rValue(summary.averageR)}</td>
                        <td>{rValue(summary.medianR)}</td>
                        <td>{percentValue(summary.winRate)}</td>
                        <td className="trade-negative">{rValue(summary.maxDrawdownR)}</td>
                        <td>{summary.markedOpen}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : <p>No eligible closed trades match these filters.</p>}
            <p className="exit-lab-method">
              Tests include fixed R and percentage targets, 8/21EMA closes, 5/10/15-session exits, 5/8/10% trailing stops, 2x/3x ATR trails, and partial-profit hybrids. Alternatives use the maximum saved position size, saved initial risk, a 1R stop, daily bars, and a 60-session limit. Unresolved alternatives are marked at the latest close. Same-bar stop and target hits assume the stop occurred first; newly raised trailing stops apply on the next bar.
            </p>
          </section>

          <section className="exit-lab-exclusions">
            <strong>Coverage notes</strong>
            <span>{coverage.missingRisk} missing risk</span>
            <span>{coverage.incompleteExecutions} incomplete executions</span>
            <span>{coverage.unsupportedSymbols} unsupported market symbols</span>
            <span>{coverage.noMarketData} missing market histories</span>
          </section>
        </>
      ) : null}
    </div>
  );
}
