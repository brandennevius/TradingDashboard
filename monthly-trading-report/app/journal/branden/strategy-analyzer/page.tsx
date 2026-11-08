"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import { usePortfolioSelection } from "@/lib/portfolio-selection";
import {
  analyzeStrategy,
  strategySetupOptions,
  strategyTradeR,
  type CriterionFilter,
  type CriterionFilters,
  type StrategySummary
} from "@/lib/strategy-analyzer";
import { tradeChecklistScore } from "@/lib/trade-review";
import type { SetupChecklistTemplate, TradeLogEntry } from "@/lib/types";
import "./strategy-analyzer.css";

const money = (value: number | null) =>
  value === null
    ? "—"
    : value.toLocaleString("en-US", {
        style: "currency",
        currency: "USD",
        maximumFractionDigits: 2
      });
const percent = (value: number | null) =>
  value === null ? "—" : `${(value * 100).toFixed(1)}%`;
const rValue = (value: number | null) =>
  value === null ? "—" : `${value >= 0 ? "+" : ""}${value.toFixed(2)}R`;
const difference = (value: number | null) =>
  value === null
    ? "—"
    : `${value >= 0 ? "+" : ""}${(value * 100).toFixed(1)} pp`;
const sampleSize = 5;
const pageSize = 25;
type PageData = {
  trades: TradeLogEntry[];
  templates: SetupChecklistTemplate[];
  portfolios: string[];
};

function CohortResult({ summary }: { summary: StrategySummary }) {
  return (
    <div className="sa-cohort-result">
      <div>
        <strong>{percent(summary.winRate)}</strong>
        <span>
          {summary.count} {summary.count === 1 ? "trade" : "trades"}
        </span>
      </div>
      <div className="sa-rate-track" aria-hidden="true">
        <span style={{ width: `${(summary.winRate || 0) * 100}%` }} />
      </div>
      <span>
        {rValue(summary.averageR)} avg · R available for {summary.rCount}/
        {summary.count}
      </span>
      {summary.count > 0 && summary.count < sampleSize ? (
        <small>Small sample</small>
      ) : null}
    </div>
  );
}

export default function StrategyAnalyzerPage() {
  const [data, setData] = useState<PageData>({
    trades: [],
    templates: [],
    portfolios: []
  });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [attempt, setAttempt] = useState(0);
  const [setup, setSetup] = useState("");
  const [portfolio, selectPortfolio, initializePortfolio] =
    usePortfolioSelection();
  const [start, setStart] = useState("");
  const [end, setEnd] = useState("");
  const [side, setSide] = useState("");
  const [filters, setFilters] = useState<CriterionFilters>({});
  const [sort, setSort] = useState("checklist");
  const [outcome, setOutcome] = useState("");
  const [symbol, setSymbol] = useState("");
  const [pageNumber, setPageNumber] = useState(1);

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    setError("");
    fetch("/api/journal/branden/trade-log", {
      cache: "no-store",
      signal: controller.signal
    })
      .then(async (response) => {
        const payload = await response.json();
        if (!response.ok || !payload.user)
          throw new Error(
            payload.error || "Sign in to load Strategy Analyzer."
          );
        if (controller.signal.aborted) return;
        const next = {
          trades: payload.trades || [],
          templates: payload.setupChecklists || [],
          portfolios: payload.portfolios || []
        };
        setData(next);
        const options = strategySetupOptions(next.trades, next.templates);
        setSetup(
          (current) =>
            current ||
            options.find((option) => option.count > 0)?.name ||
            options[0]?.name ||
            ""
        );
        initializePortfolio(String(payload.defaultPortfolio || ""));
      })
      .catch((loadError) => {
        if (!controller.signal.aborted)
          setError(
            loadError instanceof Error
              ? loadError.message
              : "Could not load Strategy Analyzer."
          );
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, [attempt, initializePortfolio]);

  const options = useMemo(
    () => strategySetupOptions(data.trades, data.templates),
    [data]
  );
  const portfolios = useMemo(
    () =>
      Array.from(
        new Set(
          [
            ...data.portfolios,
            ...data.trades
              .filter((trade) => trade.userId === "branden")
              .map((trade) => trade.portfolioTag),
            portfolio
          ].filter(Boolean)
        )
      ).sort(),
    [data, portfolio]
  );
  const invalidRange = Boolean(start && end && start > end);
  const analysis = useMemo(
    () =>
      analyzeStrategy(
        data.trades,
        data.templates,
        { setup, portfolio, start, end, side },
        filters
      ),
    [data, setup, portfolio, start, end, side, filters]
  );
  const comparisonRows = useMemo(
    () =>
      [...analysis.comparisons].sort((a, b) => {
        if (sort === "win-rate")
          return (
            (b.winRateDifference === null
              ? -1
              : Math.abs(b.winRateDifference)) -
            (a.winRateDifference === null ? -1 : Math.abs(a.winRateDifference))
          );
        if (sort === "average-r")
          return (
            (b.averageRDifference === null
              ? -1
              : Math.abs(b.averageRDifference)) -
            (a.averageRDifference === null
              ? -1
              : Math.abs(a.averageRDifference))
          );
        return 0;
      }),
    [analysis.comparisons, sort]
  );
  const detailTrades = useMemo(
    () =>
      analysis.matching
        .filter(
          (trade) =>
            (!symbol.trim() ||
              trade.symbol
                .toLowerCase()
                .includes(symbol.trim().toLowerCase())) &&
            (!outcome ||
              (outcome === "win"
                ? trade.pnl > 0
                : outcome === "loss"
                  ? trade.pnl < 0
                  : trade.pnl === 0))
        )
        .sort(
          (a, b) =>
            b.entryDate.localeCompare(a.entryDate) ||
            a.symbol.localeCompare(b.symbol)
        ),
    [analysis.matching, symbol, outcome]
  );
  const pages = Math.max(1, Math.ceil(detailTrades.length / pageSize));
  const currentPage = Math.min(pageNumber, pages);
  const visibleTrades = detailTrades.slice(
    (currentPage - 1) * pageSize,
    currentPage * pageSize
  );
  const summary = analysis.summary.matching;

  function updateFilter(key: string, value: CriterionFilter) {
    setFilters((current) => ({ ...current, [key]: value }));
    setPageNumber(1);
  }

  return (
    <div className="branden-journal-content strategy-analyzer">
      <header className="branden-route-header">
        <div>
          <p className="eyebrow">Branden journal · Research</p>
          <h1>Strategy Analyzer</h1>
          <span>Find the conditions behind your results.</span>
        </div>
        <Link
          className="sa-secondary-link"
          href="/journal/branden/setup-builder"
        >
          Open Setup Builder →
        </Link>
      </header>
      <p className="sa-intro">
        A grade tells you how many boxes you checked. Explore which boxes are
        associated with better outcomes, including winners graded low or
        incomplete.
      </p>

      {loading ? (
        <div className="sa-empty" role="status">
          Loading saved trades and setup criteria…
        </div>
      ) : error ? (
        <div className="sa-empty" role="alert">
          <h2>Could not load the analyzer</h2>
          <p>{error}</p>
          <button
            type="button"
            onClick={() => setAttempt((value) => value + 1)}
          >
            Try again
          </button>
        </div>
      ) : !options.length ? (
        <div className="sa-empty">
          <h2>Start with a setup and recorded trades</h2>
          <p>
            Create a setup in Setup Builder, assign it to trades, and save their
            checklist answers. The analyzer will compare those saved answers
            here.
          </p>
          <Link href="/journal/branden/setup-builder">
            Go to Setup Builder →
          </Link>
        </div>
      ) : (
        <>
          <section className="sa-panel sa-scope" aria-label="Analysis scope">
            <div className="sa-section-heading">
              <div>
                <span className="sa-step">01 · Choose your sample</span>
                <h2>One setup. Its recorded history.</h2>
              </div>
              <span className="sa-badge">Closed trades · All grades</span>
            </div>
            <div className="sa-scope-fields">
              <label className="sa-setup-picker">
                Setup / version
                <select
                  value={setup}
                  onChange={(event) => {
                    setSetup(event.target.value);
                    setFilters({});
                    setPageNumber(1);
                  }}
                >
                  {options.map((option) => (
                    <option key={option.name} value={option.name}>
                      {option.name}
                      {option.archived ? " · archived" : ""}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                Portfolio
                <select
                  value={portfolio}
                  onChange={(event) => {
                    selectPortfolio(event.target.value);
                    setPageNumber(1);
                  }}
                >
                  <option value="">All portfolios</option>
                  {portfolios.map((name) => (
                    <option key={name}>{name}</option>
                  ))}
                </select>
              </label>
              <label>
                Entry from
                <input
                  type="date"
                  aria-invalid={invalidRange}
                  value={start}
                  onChange={(event) => {
                    setStart(event.target.value);
                    setPageNumber(1);
                  }}
                />
              </label>
              <label>
                Entry through
                <input
                  type="date"
                  aria-invalid={invalidRange}
                  value={end}
                  onChange={(event) => {
                    setEnd(event.target.value);
                    setPageNumber(1);
                  }}
                />
              </label>
              <label>
                Direction
                <select
                  value={side}
                  onChange={(event) => {
                    setSide(event.target.value);
                    setPageNumber(1);
                  }}
                >
                  <option value="">Long & short</option>
                  <option value="LONG">Long</option>
                  <option value="SHORT">Short</option>
                </select>
              </label>
            </div>
            <p className="sa-help">
              Setup uses the trade’s primary setup tag. Versions stay separate;
              archived setups remain available. Date limits include both
              endpoints.
            </p>
          </section>
          {invalidRange ? (
            <div className="sa-empty" role="alert">
              Entry from must be on or before Entry through.
            </div>
          ) : (
            <>
              <section
                className="sa-metrics"
                aria-label="Matching trade performance"
                aria-live="polite"
              >
                <article>
                  <span>Matching trades</span>
                  <strong>
                    {summary.count}
                    <small> / {analysis.baseline.length}</small>
                  </strong>
                  <p>
                    {analysis.activeFilters
                      ? `${analysis.activeFilters} criteria filters applied`
                      : "All trades in this sample"}
                  </p>
                </article>
                <article>
                  <span>Win rate</span>
                  <strong>{percent(summary.winRate)}</strong>
                  <p>
                    {summary.wins} wins · {summary.losses} losses ·{" "}
                    {summary.breakeven} flat
                  </p>
                </article>
                <article>
                  <span>Net P/L</span>
                  <strong className={summary.netPnl < 0 ? "sa-negative" : ""}>
                    {money(summary.netPnl)}
                  </strong>
                  <p>{money(summary.averagePnl)} average per trade</p>
                </article>
                <article>
                  <span>Average R</span>
                  <strong>{rValue(summary.averageR)}</strong>
                  <p>
                    Risk recorded for {summary.rCount}/{summary.count} trades
                  </p>
                </article>
              </section>

              <section
                className="sa-panel"
                aria-labelledby="sa-criteria-heading"
              >
                <div className="sa-section-heading">
                  <div>
                    <span className="sa-step">02 · Dissect the criteria</span>
                    <h2 id="sa-criteria-heading">
                      What changes when this condition is met?
                    </h2>
                  </div>
                  <div className="sa-heading-actions">
                    <label>
                      Sort by
                      <select
                        value={sort}
                        onChange={(event) => setSort(event.target.value)}
                      >
                        <option value="checklist">Checklist order</option>
                        <option value="win-rate">Largest win-rate gap</option>
                        <option value="average-r">Largest average-R gap</option>
                      </select>
                    </label>
                    <button
                      type="button"
                      className="sa-secondary"
                      disabled={!analysis.activeFilters}
                      onClick={() => {
                        setFilters({});
                        setPageNumber(1);
                      }}
                    >
                      Clear criteria filters
                    </button>
                  </div>
                </div>
                <p className="sa-help">
                  Choose <strong>Met</strong> or <strong>Not met</strong> on any
                  rows to combine conditions. Every selected condition must
                  match. For point rows, Met means full points; Not met includes
                  zero and partial points. Scroll the table horizontally to see
                  all columns.
                </p>
                {analysis.criteria.length ? (
                  <div
                    className="sa-table-scroll"
                    tabIndex={0}
                    role="region"
                    aria-label="Criterion comparisons, scroll horizontally"
                  >
                    <table
                      className="sa-criteria-table"
                      aria-label="Criterion comparisons"
                    >
                      <thead>
                        <tr>
                          <th scope="col">Saved criterion</th>
                          <th scope="col">Require in sample</th>
                          <th scope="col">Met / full points</th>
                          <th scope="col">Not met / below full</th>
                          <th scope="col">Met − not met</th>
                          <th scope="col">Not recorded</th>
                        </tr>
                      </thead>
                      <tbody>
                        {comparisonRows.map(
                          ({
                            criterion,
                            met,
                            notMet,
                            unknown,
                            winRateDifference,
                            averageRDifference
                          }) => (
                            <tr
                              key={criterion.key}
                              className={
                                filters[criterion.key] &&
                                filters[criterion.key] !== "any"
                                  ? "sa-filtered-row"
                                  : ""
                              }
                            >
                              <th scope="row">
                                <span className="sa-group-label">
                                  {criterion.group}
                                </span>
                                <strong>{criterion.label}</strong>
                                <small>
                                  {criterion.inputType === "points"
                                    ? `Scored · max ${criterion.points} points`
                                    : "Yes / no"}
                                </small>
                              </th>
                              <td>
                                <select
                                  aria-label={`Require ${criterion.group}: ${criterion.label}${criterion.inputType === "points" ? ` (max ${criterion.points})` : ""}`}
                                  value={filters[criterion.key] || "any"}
                                  onChange={(event) =>
                                    updateFilter(
                                      criterion.key,
                                      event.target.value as CriterionFilter
                                    )
                                  }
                                >
                                  <option value="any">Any answer</option>
                                  <option value="met">
                                    {criterion.inputType === "points"
                                      ? "Full points"
                                      : "Met"}
                                  </option>
                                  <option value="not-met">
                                    {criterion.inputType === "points"
                                      ? "Below full points"
                                      : "Not met"}
                                  </option>
                                  {criterion.inputType === "points" ? (
                                    <option value="partial">
                                      Partial points only
                                    </option>
                                  ) : null}
                                  <option value="unknown">Not recorded</option>
                                </select>
                              </td>
                              <td>
                                <CohortResult summary={met} />
                              </td>
                              <td>
                                <CohortResult summary={notMet} />
                              </td>
                              <td>
                                <strong
                                  className={
                                    winRateDifference === null
                                      ? ""
                                      : winRateDifference < 0
                                        ? "sa-negative"
                                        : "sa-positive"
                                  }
                                >
                                  {difference(winRateDifference)}
                                </strong>
                                <span className="sa-cell-detail">
                                  {rValue(averageRDifference)} avg R gap
                                </span>
                              </td>
                              <td>
                                <span className="sa-unknown-count">
                                  {unknown}
                                </span>
                              </td>
                            </tr>
                          )
                        )}
                      </tbody>
                    </table>
                  </div>
                ) : (
                  <div className="sa-empty">
                    <h3>No criteria recorded for this setup</h3>
                    <p>
                      Save checklist answers on your trades to start comparing
                      individual conditions.
                    </p>
                  </div>
                )}
                <p className="sa-help sa-footnote">
                  Each row holds your other selected criteria fixed, then
                  compares this row’s met and not-met groups. Its own filter is
                  ignored for that comparison. “Not recorded” is excluded from
                  both groups.
                </p>
              </section>

              <section className="sa-panel" aria-labelledby="sa-subset-heading">
                <div className="sa-section-heading">
                  <div>
                    <span className="sa-step">03 · Compare your subset</span>
                    <h2 id="sa-subset-heading">
                      Does this combination stand out?
                    </h2>
                  </div>
                </div>
                <div
                  className="sa-active-filters"
                  aria-label="Selected criteria"
                >
                  {analysis.activeFilters ? (
                    analysis.criteria
                      .filter(
                        (item) =>
                          filters[item.key] && filters[item.key] !== "any"
                      )
                      .map((item) => (
                        <button
                          type="button"
                          key={item.key}
                          className="sa-filter-chip"
                          onClick={() => updateFilter(item.key, "any")}
                          aria-label={`Remove filter ${item.group}: ${item.label}`}
                        >
                          {item.label}:{" "}
                          {filters[item.key] === "met"
                            ? "met / full"
                            : filters[item.key] === "not-met"
                              ? "not met / below full"
                              : filters[item.key] === "partial"
                                ? "partial"
                                : "not recorded"}{" "}
                          <span aria-hidden="true">×</span>
                        </button>
                      ))
                  ) : (
                    <span className="sa-help">
                      No criteria selected. Use the row filters above to explore
                      a combination.
                    </span>
                  )}
                </div>
                <div
                  className="sa-table-scroll"
                  tabIndex={0}
                  role="region"
                  aria-label="Subset performance, scroll horizontally"
                >
                  <table
                    className="sa-subset-table"
                    aria-label="Subset performance"
                  >
                    <thead>
                      <tr>
                        <th scope="col">Trade group</th>
                        <th scope="col">Trades</th>
                        <th scope="col">Win rate</th>
                        <th scope="col">Net P/L</th>
                        <th scope="col">Avg P/L</th>
                        <th scope="col">Avg R</th>
                        <th scope="col">R coverage</th>
                      </tr>
                    </thead>
                    <tbody>
                      {(
                        [
                          ["All scoped trades", analysis.summary.baseline],
                          ["Matching criteria", summary],
                          ["Other scoped trades", analysis.summary.others]
                        ] as const
                      ).map(([label, value]) => (
                        <tr key={label}>
                          <th scope="row">{label}</th>
                          <td>{value.count}</td>
                          <td>{percent(value.winRate)}</td>
                          <td>{money(value.netPnl)}</td>
                          <td>{money(value.averagePnl)}</td>
                          <td>{rValue(value.averageR)}</td>
                          <td>
                            {value.rCount}/{value.count}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                {summary.count > 0 && summary.count < sampleSize ? (
                  <p className="sa-sample-note">
                    Only {summary.count} matching trades. Treat this as a lead
                    to investigate, rather than a reason to change a setup yet.
                  </p>
                ) : null}
                {!summary.count ? (
                  <p className="sa-sample-note">
                    {analysis.baseline.length
                      ? "No trades match this combination. Relax a criterion or include not-recorded answers."
                      : "No closed trades match this setup and scope. Try another version, portfolio, or date range."}
                  </p>
                ) : null}
              </section>

              <section className="sa-panel" aria-labelledby="sa-trades-heading">
                <div className="sa-section-heading">
                  <div>
                    <span className="sa-step">04 · Inspect the evidence</span>
                    <h2 id="sa-trades-heading">Matching trades</h2>
                  </div>
                  <div className="sa-heading-actions">
                    <label>
                      Find symbol
                      <input
                        type="search"
                        placeholder="e.g. IONQ"
                        value={symbol}
                        onChange={(event) => {
                          setSymbol(event.target.value);
                          setPageNumber(1);
                        }}
                      />
                    </label>
                    <label>
                      Show outcome
                      <select
                        value={outcome}
                        onChange={(event) => {
                          setOutcome(event.target.value);
                          setPageNumber(1);
                        }}
                      >
                        <option value="">All outcomes</option>
                        <option value="win">Winners</option>
                        <option value="loss">Losers</option>
                        <option value="flat">Breakeven</option>
                      </select>
                    </label>
                  </div>
                </div>
                <p className="sa-help">
                  Symbol and outcome narrow this list only; performance figures
                  above still include all matching outcomes. Open a symbol in a
                  new tab to review the original trade.
                </p>
                {detailTrades.length ? (
                  <>
                    <div
                      className="sa-table-scroll"
                      tabIndex={0}
                      role="region"
                      aria-label="Matching trades and saved answers, scroll horizontally"
                    >
                      <table
                        className="sa-trades-table"
                        aria-label="Matching trades"
                      >
                        <thead>
                          <tr>
                            <th scope="col">Trade / entry</th>
                            <th scope="col">Outcome</th>
                            <th scope="col">Saved grade</th>
                            <th scope="col">Net P/L</th>
                            <th scope="col">R</th>
                            {analysis.criteria.map((item) => (
                              <th
                                className="sa-matrix-heading"
                                scope="col"
                                key={item.key}
                                title={`${item.group}: ${item.label}`}
                              >
                                <span>{item.label}</span>
                              </th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {visibleTrades.map((trade) => (
                            <tr key={trade.id}>
                              <th scope="row">
                                <Link
                                  href={`/journal/branden/dashboard?${new URLSearchParams({ tradeId: trade.id })}`}
                                  target="_blank"
                                  rel="noreferrer"
                                  aria-label={`Review ${trade.symbol} (opens in a new tab)`}
                                >
                                  {trade.symbol}
                                </Link>
                                <small>
                                  {trade.entryDate} · {trade.side.toLowerCase()}
                                </small>
                              </th>
                              <td>
                                <span
                                  className={`sa-outcome ${trade.pnl > 0 ? "sa-win" : trade.pnl < 0 ? "sa-loss" : ""}`}
                                >
                                  {trade.pnl > 0
                                    ? "Win"
                                    : trade.pnl < 0
                                      ? "Loss"
                                      : "Flat"}
                                </span>
                              </td>
                              <td>
                                {
                                  tradeChecklistScore(trade, data.templates)
                                    .grade
                                }
                              </td>
                              <td>{money(trade.pnl)}</td>
                              <td>{rValue(strategyTradeR(trade))}</td>
                              {analysis.criteria.map((item) => {
                                const answer = analysis.answers
                                  .get(trade.id)
                                  ?.get(item.key);
                                const state = answer?.state || "unknown";
                                return (
                                  <td className="sa-matrix-cell" key={item.key}>
                                    <span
                                      className={`sa-answer sa-answer-${state}`}
                                      title={`${item.label}: ${state === "unknown" ? "Not recorded" : state}`}
                                    >
                                      {state === "unknown"
                                        ? "—"
                                        : item.inputType === "points"
                                          ? `${answer?.score}/${item.points}`
                                          : state === "met"
                                            ? "✓ Met"
                                            : "× Not met"}
                                    </span>
                                  </td>
                                );
                              })}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                    <div className="sa-pagination">
                      <span>
                        {detailTrades.length} trades · Page {currentPage} of{" "}
                        {pages}
                      </span>
                      <div>
                        <button
                          type="button"
                          className="sa-secondary"
                          disabled={currentPage <= 1}
                          onClick={() => setPageNumber(currentPage - 1)}
                        >
                          Previous
                        </button>
                        <button
                          type="button"
                          className="sa-secondary"
                          disabled={currentPage >= pages}
                          onClick={() => setPageNumber(currentPage + 1)}
                        >
                          Next
                        </button>
                      </div>
                    </div>
                  </>
                ) : (
                  <p className="sa-empty">
                    No matching trades to show. Try broadening the criteria,
                    symbol, or outcome filters.
                  </p>
                )}
              </section>

              <details className="sa-panel sa-method">
                <summary>How to read this analysis</summary>
                <div>
                  <p>
                    This is a comparison of recorded outcomes, not proof that a
                    criterion causes wins. Conditions can overlap, and repeated
                    filtering can make a small sample look stronger than it is.
                    Confirm ideas on future trades before revising a setup.
                  </p>
                  <p>
                    Win rate = positive-P/L trades ÷ all trades in the group,
                    including breakeven. P/L uses the journal’s saved P/L
                    without subtracting commission again. R = saved P/L ÷ saved
                    positive risk; trades without valid risk remain in P/L and
                    win rate, but are excluded from average R.
                  </p>
                  <p>
                    Only visible Branden trades with a closed status are
                    included. Hidden trades and merged child trades are
                    excluded. No minimum grade is required. Checklist answers
                    are read from each trade; current templates never fill in
                    missing historical answers. Identical saved labels, groups,
                    input types, and maximum points are compared within the
                    selected setup version; changed definitions stay separate.
                  </p>
                  <p>
                    In the current scope: {analysis.coverage.open} open trades
                    excluded; {analysis.coverage.invalidPnl} trades with invalid
                    P/L excluded; {analysis.coverage.noChecklist} closed trades
                    without any saved checklist. Empty, invalid, or conflicting
                    answers appear as not recorded. Groups with fewer than five
                    trades carry a small-sample label.
                  </p>
                </div>
              </details>
            </>
          )}
        </>
      )}
    </div>
  );
}
