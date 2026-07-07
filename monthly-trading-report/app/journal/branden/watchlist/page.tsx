"use client";

import { useEffect, useMemo, useState } from "react";
import TradePriceChart from "@/app/components/TradePriceChart";
import type {
  ChecklistGradeBand,
  SetupChecklistTemplate,
  TradeChecklistItem,
  TraderUser,
  WatchlistItem,
  WeeklyWatchlist
} from "@/lib/types";

type AiWatchlistReview = {
  verdict: "Actionable" | "Watch" | "Pass" | "Manage Existing";
  buyPlan?: {
    recommendation: "Buy" | "Starter" | "Add" | "Wait" | "No Trade";
    primaryBuyLevel: string;
    starterBuyLevel: string;
    addOnBuyLevel: string;
    stopLevel: string;
    noTradeReason: string;
    canslimRule: string;
  };
  tradeDeskVerdict?: string;
  technicalSetupGrade?: string;
  canslimQualityGrade?: string;
  entryQuality?: string;
  riskQuality?: string;
  actionPlan?: string;
  addTrigger?: string;
  invalidationSignal?: string;
  positionSizeGuidance?: string;
  whatWouldUpgradeThis?: string[];
  whatWouldKillThisSetup?: string[];
  decisionSummary?: string;
  checklistGradeContext?: string;
  independentCanslimAssessment?: string;
  valueAddInsight?: string;
  contradictionFlags?: string[];
  positionSizingImplication?: string;
  gradeRead: string;
  setupRead: string;
  entryRead: string;
  riskRead: string;
  chartAnalysis: {
    visibleText: string[];
    patternRead: string;
    keyLevels: string[];
    relativeStrengthRead: string;
    volumeRead: string;
    modelComparison: string;
    confidence: "low" | "medium" | "high";
  };
  modelExamplesUsed: string[];
  missingEvidence: string[];
  actionItems: string[];
};

const defaultChecklistGradeBands: ChecklistGradeBand[] = [
  { id: "grade-a-plus", label: "A+", minScore: 10, maxScore: null },
  { id: "grade-a", label: "A", minScore: 8, maxScore: 9 },
  { id: "grade-b-plus", label: "B+", minScore: 7, maxScore: 7 },
  { id: "grade-b", label: "B", minScore: 6, maxScore: 6 },
  { id: "grade-c", label: "C", minScore: 0, maxScore: 5 }
];

function numberValue(value: unknown) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : 0;
}

function checklistGrade(score: number, gradeBands: ChecklistGradeBand[]) {
  const sortedBands = [...gradeBands].sort((a, b) => b.minScore - a.minScore);
  const match = sortedBands.find((band) => score >= band.minScore && (band.maxScore === null || score <= band.maxScore));
  return match?.label || sortedBands[sortedBands.length - 1]?.label || "Unscored";
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
    grade: total ? checklistGrade(earned, gradeBands) : "Unscored"
  };
}

function setupTemplateFor(setupName: string, templates: SetupChecklistTemplate[]) {
  return templates.find((template) => template.setupName.trim().toLowerCase() === setupName.trim().toLowerCase());
}

function setupGradeBands(setupName: string, templates: SetupChecklistTemplate[]) {
  return setupTemplateFor(setupName, templates)?.gradeBands?.length
    ? setupTemplateFor(setupName, templates)?.gradeBands || defaultChecklistGradeBands
    : defaultChecklistGradeBands;
}

function checklistFromTemplate(template: SetupChecklistTemplate): TradeChecklistItem[] {
  const groups = template.groups?.length
    ? template.groups
    : [{ id: "default", name: "Checklist", criteria: template.criteria || [] }];
  return groups.flatMap((group) =>
    group.criteria.map((criterion) => ({
      id: criterion.id,
      criteria: criterion.criteria,
      points: criterion.points,
      met: false,
      score: 0,
      inputType: criterion.inputType || "boolean",
      groupName: group.name,
      importTagKey: criterion.importTagKey || "",
      importTagValue: criterion.importTagValue || ""
    }))
  );
}

function newItem(): WatchlistItem {
  const now = new Date().toISOString();
  return {
    id: `watchlist-item-${crypto.randomUUID()}`,
    symbol: "",
    side: "LONG",
    setupTag: "",
    setupGrade: "",
    checklistItems: [],
    plannedEntry: 0,
    stopPrice: 0,
    takeProfitPrice: 0,
    entryCriteria: "",
    entryNotes: "",
    invalidation: "",
    notes: "",
    screenshots: [],
    chartLinks: [],
    createdAt: now,
    updatedAt: now
  };
}

function weekRange(watchlist: WeeklyWatchlist) {
  const format = (value: string) =>
    new Intl.DateTimeFormat("en-US", { month: "short", day: "numeric", timeZone: "UTC" }).format(
      new Date(`${value}T12:00:00Z`)
    );
  return `${format(watchlist.startDate)}–${format(watchlist.endDate)}, ${watchlist.year}`;
}

function isoWeek(date = new Date()) {
  const utc = new Date(Date.UTC(date.getFullYear(), date.getMonth(), date.getDate()));
  const day = utc.getUTCDay() || 7;
  utc.setUTCDate(utc.getUTCDate() + 4 - day);
  const year = utc.getUTCFullYear();
  const yearStart = new Date(Date.UTC(year, 0, 1));
  const weekNumber = Math.ceil(((utc.getTime() - yearStart.getTime()) / 86400000 + 1) / 7);
  const monday = new Date(utc);
  monday.setUTCDate(utc.getUTCDate() - ((utc.getUTCDay() || 7) - 1));
  const friday = new Date(monday);
  friday.setUTCDate(monday.getUTCDate() + 4);
  return {
    year,
    weekNumber,
    weekKey: `${year}-W${String(weekNumber).padStart(2, "0")}`,
    startDate: monday.toISOString().slice(0, 10),
    endDate: friday.toISOString().slice(0, 10)
  };
}

function nextWeekAfter(watchlist: WeeklyWatchlist | null, watchlists: WeeklyWatchlist[]) {
  const anchor =
    watchlist ||
    [...watchlists].sort((a, b) => b.weekKey.localeCompare(a.weekKey))[0] ||
    null;
  const nextDate = anchor
    ? new Date(`${anchor.startDate}T12:00:00Z`)
    : new Date();
  nextDate.setUTCDate(nextDate.getUTCDate() + 7);
  return isoWeek(nextDate);
}

function createWatchlist(userId: string, week: ReturnType<typeof isoWeek>): WeeklyWatchlist {
  const now = new Date().toISOString();
  return {
    id: `${userId}-${week.weekKey}`,
    userId,
    ...week,
    title: `W${week.weekNumber} Watchlist`,
    items: [],
    createdAt: now,
    updatedAt: now
  };
}

export default function WatchlistPage() {
  const [user, setUser] = useState<TraderUser | null>(null);
  const [watchlists, setWatchlists] = useState<WeeklyWatchlist[]>([]);
  const [templates, setTemplates] = useState<SetupChecklistTemplate[]>([]);
  const [activeWeekKey, setActiveWeekKey] = useState("");
  const [selectedItemId, setSelectedItemId] = useState("");
  const [status, setStatus] = useState("");
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(true);
  const [isSaving, setIsSaving] = useState(false);
  const [isReviewingSetup, setIsReviewingSetup] = useState(false);
  const [aiReviewByItemId, setAiReviewByItemId] = useState<Record<string, AiWatchlistReview>>({});
  const [maximizedScreenshot, setMaximizedScreenshot] = useState("");

  useEffect(() => {
    let cancelled = false;
    async function load() {
      const [watchlistsResponse, templatesResponse] = await Promise.all([
        fetch("/api/watchlists", { cache: "no-store" }),
        fetch("/api/settings/setup-checklists", { cache: "no-store" })
      ]);
      const watchlistData = await watchlistsResponse.json().catch(() => ({}));
      const templateData = await templatesResponse.json().catch(() => ({}));
      if (cancelled) return;
      if (!watchlistsResponse.ok) {
        setError(watchlistData.error || "Could not load watchlists.");
      } else {
        const loaded = Array.isArray(watchlistData.watchlists) ? watchlistData.watchlists : [];
        setWatchlists(loaded);
        setActiveWeekKey(watchlistData.currentWeekKey || loaded[0]?.weekKey || "");
        setUser(watchlistData.user || null);
      }
      setTemplates(Array.isArray(templateData.setupChecklists) ? templateData.setupChecklists : []);
      setIsLoading(false);
    }
    load();
    return () => {
      cancelled = true;
    };
  }, []);

  const activeWatchlist = useMemo(
    () => watchlists.find((watchlist) => watchlist.weekKey === activeWeekKey) || null,
    [activeWeekKey, watchlists]
  );
  const selectedItem = activeWatchlist?.items.find((item) => item.id === selectedItemId) || activeWatchlist?.items[0] || null;
  const canEdit = Boolean(user && !user.readOnly);

  useEffect(() => {
    if (!activeWatchlist?.items.some((item) => item.id === selectedItemId)) {
      setSelectedItemId(activeWatchlist?.items[0]?.id || "");
    }
  }, [activeWatchlist, selectedItemId]);

  function updateActiveWatchlist(updater: (watchlist: WeeklyWatchlist) => WeeklyWatchlist) {
    setWatchlists((current) =>
      current.map((watchlist) => (watchlist.weekKey === activeWeekKey ? updater(watchlist) : watchlist))
    );
  }

  function updateItem(itemId: string, updates: Partial<WatchlistItem>) {
    updateActiveWatchlist((watchlist) => ({
      ...watchlist,
      updatedAt: new Date().toISOString(),
      items: watchlist.items.map((item) =>
        item.id === itemId ? { ...item, ...updates, updatedAt: new Date().toISOString() } : item
      )
    }));
  }

  function chooseSetup(item: WatchlistItem, setupTag: string) {
    const template = setupTemplateFor(setupTag, templates);
    updateItem(item.id, {
      setupTag,
      setupGrade: "",
      checklistItems: template ? checklistFromTemplate(template) : []
    });
  }

  function gradeSelectedSetup(item: WatchlistItem) {
    const score = checklistScore(item.checklistItems, setupGradeBands(item.setupTag, templates));
    updateItem(item.id, { setupGrade: score.grade });
  }

  async function addScreenshots(item: WatchlistItem, files: FileList | null) {
    if (!files?.length) return;
    const images = Array.from(files).filter((file) => file.type.startsWith("image/"));
    if (images.some((file) => file.size > 3_500_000)) {
      setStatus("Each screenshot must be smaller than 3.5 MB.");
      return;
    }
    setStatus(`Uploading ${images.length} screenshot${images.length === 1 ? "" : "s"}...`);
    const urls: string[] = [];
    for (const file of images) {
      const formData = new FormData();
      formData.set("file", file);
      formData.set("itemId", item.id);
      const response = await fetch("/api/watchlists/screenshots", {
        method: "POST",
        body: formData
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        setStatus(data.error || "Could not upload screenshot.");
        return;
      }
      if (data.url) urls.push(String(data.url));
    }
    updateItem(item.id, { screenshots: [...item.screenshots, ...urls] });
    setStatus("Screenshots uploaded. Save current ticker to persist them.");
  }

  async function save() {
    if (!activeWatchlist || !selectedItem) {
      setStatus("Select a ticker to save.");
      return;
    }
    setIsSaving(true);
    setStatus(`Saving ${selectedItem.symbol || "ticker"}...`);
    await saveWatchlistItem(activeWatchlist, selectedItem);
    setIsSaving(false);
  }

  async function saveWatchlistItem(watchlist: WeeklyWatchlist, item: WatchlistItem) {
    const response = await fetch("/api/watchlists", {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ weekKey: watchlist.weekKey, watchlist, item })
    });
    const data = await response.json().catch(() => ({}));
    if (response.ok) {
      if (data.item) {
        setWatchlists((current) =>
          current.map((savedWatchlist) =>
            savedWatchlist.weekKey === watchlist.weekKey
              ? {
                  ...savedWatchlist,
                  updatedAt: data.watchlist?.updatedAt || savedWatchlist.updatedAt,
                  items: savedWatchlist.items.some((currentItem) => currentItem.id === item.id)
                    ? savedWatchlist.items.map((currentItem) => (currentItem.id === item.id ? data.item : currentItem))
                    : [...savedWatchlist.items, data.item]
                }
              : savedWatchlist
          )
        );
      } else if (data.watchlist) {
        setWatchlists((current) =>
          current.some((savedWatchlist) => savedWatchlist.weekKey === data.watchlist.weekKey)
            ? current.map((savedWatchlist) => (savedWatchlist.weekKey === data.watchlist.weekKey ? data.watchlist : savedWatchlist))
            : [data.watchlist, ...current].sort((a, b) => b.weekKey.localeCompare(a.weekKey))
        );
      }
      setStatus(`${item.symbol || "Ticker"} saved.`);
    } else {
      setStatus(data.error || "Could not save ticker.");
    }
  }

  async function reviewSetup(item: WatchlistItem) {
    setIsReviewingSetup(true);
    setStatus(`Reviewing ${item.symbol || "setup"} with AI...`);
    setError("");
    try {
      const response = await fetch("/api/journal/branden/watchlist/review", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ item })
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(data.error || "Could not review this setup.");
      }
      const review = data.review as AiWatchlistReview;
      setAiReviewByItemId((current) => ({ ...current, [item.id]: review }));
      const now = new Date().toISOString();
      const nextWatchlists = watchlists.map((watchlist) =>
        watchlist.weekKey === activeWeekKey
          ? {
              ...watchlist,
              updatedAt: now,
              items: watchlist.items.map((currentItem) =>
                currentItem.id === item.id ? { ...currentItem, aiReview: review as unknown as Record<string, unknown>, updatedAt: now } : currentItem
              )
            }
          : watchlist
      );
      const updatedWatchlist = nextWatchlists.find((watchlist) => watchlist.weekKey === activeWeekKey);
      const updatedItem = updatedWatchlist?.items.find((currentItem) => currentItem.id === item.id);
      setWatchlists(nextWatchlists);
      if (updatedWatchlist && updatedItem) {
        await saveWatchlistItem(updatedWatchlist, updatedItem);
      }
      const context = data.context || {};
      setStatus(
        `AI review complete. Used ${context.modelExampleCount || 0} model examples and ${context.strategyKnowledgeCount || 0} strategy sources.`
      );
    } catch (reviewError) {
      setError(reviewError instanceof Error ? reviewError.message : "Could not review this setup.");
      setStatus("");
    } finally {
      setIsReviewingSetup(false);
    }
  }

  function addTicker() {
    if (!activeWatchlist) return;
    const item = newItem();
    updateActiveWatchlist((watchlist) => ({ ...watchlist, items: [...watchlist.items, item] }));
    setSelectedItemId(item.id);
  }

  function createNextWeek() {
    if (!user) return;
    const next = nextWeekAfter(activeWatchlist, watchlists);
    const existing = watchlists.find((watchlist) => watchlist.weekKey === next.weekKey);
    if (existing) {
      setActiveWeekKey(existing.weekKey);
      setStatus(`${existing.title} already exists.`);
      return;
    }
    const created = createWatchlist(user.journalOwnerId || user.id, next);
    setWatchlists((current) => [created, ...current].sort((a, b) => b.weekKey.localeCompare(a.weekKey)));
    setActiveWeekKey(created.weekKey);
    setSelectedItemId("");
    setStatus(`${created.title} created. Save watchlist to keep it.`);
  }

  return (
    <div className="branden-journal-content watchlist-page">
        <header className="branden-route-header">
          <div>
            <p className="eyebrow">Weekly preparation</p>
            <h1>{activeWatchlist?.title || "Watchlist"}</h1>
            <span>{activeWatchlist ? weekRange(activeWatchlist) : "Loading current trading week"}</span>
          </div>
        </header>

        <section className="branden-route-toolbar watchlist-toolbar">
          <label>
            Trading week
            <select value={activeWeekKey} onChange={(event) => setActiveWeekKey(event.target.value)}>
              {watchlists.map((watchlist) => (
                <option value={watchlist.weekKey} key={watchlist.weekKey}>
                  {watchlist.title} · {weekRange(watchlist)}
                </option>
              ))}
            </select>
          </label>
          <button type="button" onClick={createNextWeek} disabled={!canEdit}>Create next week</button>
          <button type="button" onClick={addTicker} disabled={!canEdit || !activeWatchlist}>Add ticker</button>
          <button type="button" onClick={save} disabled={!canEdit || isSaving || !selectedItem}>{isSaving ? "Saving..." : "Save current ticker"}</button>
          {status ? <span className="status">{status}</span> : null}
        </section>

        {isLoading ? <p className="status">Loading watchlist...</p> : null}
        {error ? <p className="status error">{error}</p> : null}

        {!isLoading && activeWatchlist ? (
          <div className="watchlist-workspace">
            <aside className="watchlist-ticker-list">
              <div className="trade-chart-heading"><h3>Tickers</h3><span>{activeWatchlist.items.length}</span></div>
              {activeWatchlist.items.map((item) => (
                <button
                  className={selectedItem?.id === item.id ? "active" : ""}
                  key={item.id}
                  type="button"
                  onClick={() => setSelectedItemId(item.id)}
                >
                  <strong>{item.symbol || "New ticker"}</strong>
                  <span>{item.setupTag || "No setup"}</span>
                </button>
              ))}
              {!activeWatchlist.items.length ? <p className="muted">Add the first ticker for this week.</p> : null}
            </aside>

            {selectedItem ? (
              <section className="watchlist-detail">
                {(() => {
                  const setupScore = checklistScore(selectedItem.checklistItems, setupGradeBands(selectedItem.setupTag, templates));
                  const aiReview = aiReviewByItemId[selectedItem.id] || (selectedItem.aiReview as AiWatchlistReview | undefined);
                  return (
                    <>
                <div className="watchlist-detail-heading">
                  <div>
                    <p className="eyebrow">{activeWatchlist.title}</p>
                    <h2>{selectedItem.symbol || "New ticker"}</h2>
                    {selectedItem.setupGrade ? <span className="grade-pill">{selectedItem.setupGrade}</span> : null}
                  </div>
                  {canEdit ? (
                    <div className="trade-checklist-heading-actions">
                      <button
                        className="trade-muted-button"
                        type="button"
                        onClick={() => reviewSetup(selectedItem)}
                        disabled={isReviewingSetup}
                      >
                        {isReviewingSetup ? "Reviewing..." : "AI setup review"}
                      </button>
                      <button
                        className="trade-danger-button"
                        type="button"
                        onClick={() => {
                          updateActiveWatchlist((watchlist) => ({
                            ...watchlist,
                            items: watchlist.items.filter((item) => item.id !== selectedItem.id)
                          }));
                          setSelectedItemId("");
                        }}
                      >
                        Remove ticker
                      </button>
                    </div>
                  ) : null}
                </div>

                {aiReview ? (
                  <article className="watchlist-notes-card ai-review-card">
                    <div className="ai-review-heading">
                      <div>
                        <h3>AI Setup Review</h3>
                        <span>{aiReview.verdict} · confidence {aiReview.chartAnalysis.confidence}</span>
                      </div>
                    </div>
                    <div className="ai-review-badges">
                      {aiReview.tradeDeskVerdict ? <span>Decision: {aiReview.tradeDeskVerdict}</span> : null}
                      {aiReview.technicalSetupGrade ? <span>Technical: {aiReview.technicalSetupGrade}</span> : null}
                      {aiReview.canslimQualityGrade ? <span>CANSLIM: {aiReview.canslimQualityGrade}</span> : null}
                    </div>
                    {aiReview.decisionSummary ? <p className="ai-review-summary">{aiReview.decisionSummary}</p> : null}
                    {aiReview.buyPlan ? (
                      <div className="ai-buy-plan">
                        <div className="ai-buy-plan-head">
                          <span>Buy plan</span>
                          <strong>{aiReview.buyPlan.recommendation}</strong>
                        </div>
                        <div className="ai-review-grid">
                          <div><span>Primary</span><strong>{aiReview.buyPlan.primaryBuyLevel || "not defined"}</strong></div>
                          <div><span>Starter</span><strong>{aiReview.buyPlan.starterBuyLevel || "not defined"}</strong></div>
                          <div><span>Add</span><strong>{aiReview.buyPlan.addOnBuyLevel || "not defined"}</strong></div>
                          <div><span>Stop</span><strong>{aiReview.buyPlan.stopLevel || "not defined"}</strong></div>
                          <div><span>Rule</span><strong>{aiReview.buyPlan.canslimRule || "not defined"}</strong></div>
                          <div><span>No trade</span><strong>{aiReview.buyPlan.noTradeReason || "n/a"}</strong></div>
                        </div>
                      </div>
                    ) : null}
                    <div className="ai-review-grid">
                      {aiReview.actionPlan ? <div><span>Action</span><strong>{aiReview.actionPlan}</strong></div> : null}
                      {aiReview.invalidationSignal ? <div><span>Invalid</span><strong>{aiReview.invalidationSignal}</strong></div> : null}
                      {aiReview.positionSizeGuidance ? <div><span>Size</span><strong>{aiReview.positionSizeGuidance}</strong></div> : null}
                    </div>
                    <div className="ai-review-two-column">
                      {aiReview.whatWouldUpgradeThis?.length ? (
                        <div>
                          <h4>Upgrade</h4>
                          <ul>{aiReview.whatWouldUpgradeThis.map((item) => <li key={item}>{item}</li>)}</ul>
                        </div>
                      ) : null}
                      {aiReview.whatWouldKillThisSetup?.length ? (
                        <div>
                          <h4>Kill</h4>
                          <ul>{aiReview.whatWouldKillThisSetup.map((item) => <li key={item}>{item}</li>)}</ul>
                        </div>
                      ) : null}
                      {aiReview.contradictionFlags?.length ? (
                        <div>
                          <h4>Tension</h4>
                          <ul>{aiReview.contradictionFlags.map((item) => <li key={item}>{item}</li>)}</ul>
                        </div>
                      ) : null}
                    </div>
                    <details className="ai-review-details">
                      <summary>Detailed read</summary>
                      {aiReview.checklistGradeContext ? <p><strong>Checklist:</strong> {aiReview.checklistGradeContext}</p> : null}
                      {aiReview.independentCanslimAssessment ? <p><strong>CANSLIM:</strong> {aiReview.independentCanslimAssessment}</p> : null}
                      {aiReview.valueAddInsight ? <p><strong>Value-add:</strong> {aiReview.valueAddInsight}</p> : null}
                      <p><strong>Grade:</strong> {aiReview.gradeRead}</p>
                      <p><strong>Setup:</strong> {aiReview.setupRead}</p>
                      <p><strong>Entry:</strong> {aiReview.entryRead}</p>
                      <p><strong>Risk:</strong> {aiReview.riskRead}</p>
                      <p><strong>Pattern:</strong> {aiReview.chartAnalysis.patternRead}</p>
                      <p><strong>Volume:</strong> {aiReview.chartAnalysis.volumeRead}</p>
                      <p><strong>RS:</strong> {aiReview.chartAnalysis.relativeStrengthRead}</p>
                      <p><strong>Models:</strong> {aiReview.chartAnalysis.modelComparison}</p>
                      {aiReview.chartAnalysis.keyLevels.length ? <p><strong>Levels:</strong> {aiReview.chartAnalysis.keyLevels.join(", ")}</p> : null}
                      {aiReview.modelExamplesUsed.length ? <p><strong>Examples:</strong> {aiReview.modelExamplesUsed.join(", ")}</p> : null}
                      {aiReview.missingEvidence.length ? <p><strong>Missing:</strong> {aiReview.missingEvidence.join(", ")}</p> : null}
                      {aiReview.actionItems.length ? <p><strong>Actions:</strong> {aiReview.actionItems.join("; ")}</p> : null}
                    </details>
                  </article>
                ) : null}

                <div className="watchlist-fields watchlist-primary-fields">
                  <label>Symbol<input value={selectedItem.symbol} disabled={!canEdit} onChange={(event) => updateItem(selectedItem.id, { symbol: event.target.value.toUpperCase() })} /></label>
                  <label>Bias<select value={selectedItem.side} disabled={!canEdit} onChange={(event) => updateItem(selectedItem.id, { side: event.target.value === "SHORT" ? "SHORT" : "LONG" })}><option>LONG</option><option>SHORT</option></select></label>
                  <label>Setup<select value={selectedItem.setupTag} disabled={!canEdit} onChange={(event) => chooseSetup(selectedItem, event.target.value)}><option value="">Select setup</option>{templates.map((template) => <option key={template.id} value={template.setupName}>{template.setupName}</option>)}</select></label>
                  <label>Planned entry<input type="number" step="any" value={selectedItem.plannedEntry || ""} disabled={!canEdit} onChange={(event) => updateItem(selectedItem.id, { plannedEntry: Number(event.target.value) })} /></label>
                  <label>Stop<input type="number" step="any" value={selectedItem.stopPrice || ""} disabled={!canEdit} onChange={(event) => updateItem(selectedItem.id, { stopPrice: Number(event.target.value) })} /></label>
                  <label>Target<input type="number" step="any" value={selectedItem.takeProfitPrice || ""} disabled={!canEdit} onChange={(event) => updateItem(selectedItem.id, { takeProfitPrice: Number(event.target.value) })} /></label>
                </div>

                <div className="watchlist-two-column">
                  <article className="trade-checklist-editor">
                    <div className="trade-checklist-heading">
                      <div>
                        <h3>Setup Criteria</h3>
                        <span>
                          {setupScore.total
                            ? `${setupScore.earned}/${setupScore.total} points / ${setupScore.grade}`
                            : selectedItem.setupGrade || "From Setup Builder"}
                        </span>
                      </div>
                      <div className="trade-checklist-heading-actions">
                        {selectedItem.checklistItems.length && canEdit ? (
                          <button className="trade-muted-button" type="button" onClick={() => gradeSelectedSetup(selectedItem)}>
                            Grade from setup
                          </button>
                        ) : null}
                        {selectedItem.setupTag && canEdit ? <button className="trade-muted-button" type="button" onClick={() => chooseSetup(selectedItem, selectedItem.setupTag)}>Reload criteria</button> : null}
                      </div>
                    </div>
                    {selectedItem.checklistItems.length ? (
                      <div className="trade-checklist-list">
                        {Object.entries(
                          selectedItem.checklistItems.reduce<Record<string, TradeChecklistItem[]>>((groups, item) => {
                            const group = item.groupName || "Checklist";
                            groups[group] = [...(groups[group] || []), item];
                            return groups;
                          }, {})
                        ).map(([group, items]) => (
                          <section className="trade-checklist-group" key={group}>
                            <div className="trade-checklist-group-head"><strong>{group}</strong></div>
                            {items.map((item) => (
                              <div className="trade-checklist-row" key={item.id}>
                                <label className="trade-checklist-criteria"><span>{item.criteria}</span></label>
                                {(item.inputType || "boolean") === "points" ? (
                                  <label className="trade-checklist-score">Score<input type="number" min="0" max={item.points} value={item.score || 0} disabled={!canEdit} onChange={(event) => updateItem(selectedItem.id, { checklistItems: selectedItem.checklistItems.map((current) => current.id === item.id ? { ...current, score: Math.min(item.points, Math.max(0, Number(event.target.value))), met: Number(event.target.value) > 0 } : current) })} /></label>
                                ) : (
                                  <label className="trade-checklist-met"><input type="checkbox" checked={item.met} disabled={!canEdit} onChange={(event) => updateItem(selectedItem.id, { checklistItems: selectedItem.checklistItems.map((current) => current.id === item.id ? { ...current, met: event.target.checked, score: event.target.checked ? item.points : 0 } : current) })} />Met</label>
                                )}
                                <span className="watchlist-points">{item.points} pts</span>
                              </div>
                            ))}
                          </section>
                        ))}
                      </div>
                    ) : <p className="muted">Select a setup to load its criteria.</p>}
                  </article>

                  <article className="watchlist-notes-card watchlist-plan-card">
                    <div>
                      <h3>Trade Plan</h3>
                    </div>
                    <label>Trigger<textarea value={selectedItem.entryCriteria} disabled={!canEdit} onChange={(event) => updateItem(selectedItem.id, { entryCriteria: event.target.value })} placeholder="What must happen before this is actionable?" /></label>
                    <label>Plan<textarea value={selectedItem.entryNotes} disabled={!canEdit} onChange={(event) => updateItem(selectedItem.id, { entryNotes: event.target.value })} placeholder="Execution, sizing, add rule, target behavior..." /></label>
                    <label>Invalidation<textarea value={selectedItem.invalidation || ""} disabled={!canEdit} onChange={(event) => updateItem(selectedItem.id, { invalidation: event.target.value })} placeholder="What kills this idea?" /></label>
                    <label>Notes<textarea value={selectedItem.notes} disabled={!canEdit} onChange={(event) => updateItem(selectedItem.id, { notes: event.target.value })} placeholder="Optional context, fundamentals, market notes..." /></label>
                    <details className="watchlist-links-panel">
                      <summary>Links</summary>
                      <label>Chart / reference links<textarea value={selectedItem.chartLinks.join("\n")} disabled={!canEdit} onChange={(event) => updateItem(selectedItem.id, { chartLinks: event.target.value.split(/\n|,/).map((value) => value.trim()).filter(Boolean) })} placeholder="One URL per line" /></label>
                    </details>
                  </article>
                </div>

                <article className="watchlist-screenshots">
                  <div className="trade-chart-heading">
                    <h3>Screenshots</h3>
                    {canEdit ? <label className="watchlist-upload">Add screenshots<input type="file" accept="image/*" multiple onChange={(event) => addScreenshots(selectedItem, event.target.files)} /></label> : null}
                  </div>
                  <div className="watchlist-screenshot-grid">
                    {selectedItem.screenshots.map((screenshot, index) => (
                      <figure key={`${selectedItem.id}-${index}`}>
                        <button
                          className="watchlist-screenshot-preview"
                          type="button"
                          onClick={() => setMaximizedScreenshot(screenshot)}
                          aria-label={`Maximize ${selectedItem.symbol || "watchlist"} screenshot ${index + 1}`}
                        >
                          <img
                            src={screenshot}
                            alt={`${selectedItem.symbol} watchlist screenshot ${index + 1}`}
                            loading="lazy"
                            decoding="async"
                          />
                        </button>
                        {canEdit ? <button type="button" onClick={() => updateItem(selectedItem.id, { screenshots: selectedItem.screenshots.filter((_, imageIndex) => imageIndex !== index) })}>Remove</button> : null}
                      </figure>
                    ))}
                    {!selectedItem.screenshots.length ? <p className="muted">No screenshots attached.</p> : null}
                  </div>
                </article>

                <article className="trade-detail-section trade-price-chart-section">
                  <div className="trade-chart-heading"><h3>{selectedItem.symbol ? `${selectedItem.symbol} planning chart` : "Planning chart"}</h3><span>Live market data</span></div>
                  <TradePriceChart
                    symbol={selectedItem.symbol}
                    side={selectedItem.side}
                    entryDate=""
                    exitDate=""
                    avgEntry={selectedItem.plannedEntry}
                    exitPrice={0}
                    stopPrice={selectedItem.stopPrice}
                    takeProfitPrice={selectedItem.takeProfitPrice}
                  />
                </article>
                    </>
                  );
                })()}
              </section>
            ) : <section className="empty-state">Add a ticker to begin this week’s plan.</section>}
          </div>
        ) : null}

        {maximizedScreenshot ? (
          <div
            className="watchlist-screenshot-lightbox"
            role="dialog"
            aria-modal="true"
            aria-label="Maximized watchlist screenshot"
            onClick={() => setMaximizedScreenshot("")}
          >
            <button
              className="watchlist-lightbox-close"
              type="button"
              onClick={() => setMaximizedScreenshot("")}
              aria-label="Close maximized screenshot"
            >
              ×
            </button>
            <img
              src={maximizedScreenshot}
              alt="Maximized watchlist screenshot"
              onClick={(event) => event.stopPropagation()}
            />
          </div>
        ) : null}
      </div>
  );
}
