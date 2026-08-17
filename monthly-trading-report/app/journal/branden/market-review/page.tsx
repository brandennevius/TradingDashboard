"use client";

import { FormEvent, useEffect, useMemo, useRef, useState } from "react";

type ReviewStatus = "QUEUED" | "RUNNING" | "NEEDS_REVIEW" | "FAILED" | "COMPLETED";
type ReviewRun = {
  run_id: string;
  session_date: string;
  status: ReviewStatus;
  attempt: number;
  source_hashes: {
    marketsurge_pdf_sha256: string;
    snapshot_json_sha256: string;
    snapshot_markdown_sha256: string;
  };
  marketsurge_pdf_filename: string;
  marketsurge_pdf_size_bytes: number;
  marketsurge_pdf_page_count: number;
  source_expires_at: string;
  source_deleted_at: string | null;
  github_run_id: string | null;
  github_workflow_url: string | null;
  ocr: Record<string, unknown> | null;
  error: { code: string; message: string } | null;
  delivery_status: "NOT_REQUESTED" | "PENDING" | "SENDING" | "SENT" | "FAILED";
  delivery_error: string | null;
  artifacts: Array<{ kind: "pdf" | "markdown" | "json"; filename: string; download_url: string; sha256: string }>;
  created_at: string;
  updated_at: string;
  completed_at: string | null;
};

function nyDate() {
  return new Intl.DateTimeFormat("en-CA", { timeZone: "America/New_York" }).format(new Date());
}

function shortHash(value: string) {
  return value ? `${value.slice(0, 10)}…${value.slice(-8)}` : "—";
}

function formatTime(value: string | null) {
  if (!value) return "—";
  return new Intl.DateTimeFormat("en-US", { dateStyle: "medium", timeStyle: "short" }).format(new Date(value));
}

function errorMessage(data: { error?: string; code?: string }) {
  return `${data.code ? `${data.code}: ` : ""}${data.error || "Market review request failed."}`;
}

export default function MarketReviewPage() {
  const [sessionDate, setSessionDate] = useState(nyDate);
  const [pdf, setPdf] = useState<File | null>(null);
  const [runs, setRuns] = useState<ReviewRun[]>([]);
  const [selectedRunId, setSelectedRunId] = useState("");
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [actionRunId, setActionRunId] = useState("");
  const [error, setError] = useState("");
  const [correctionsText, setCorrectionsText] = useState("[]");
  const fileRef = useRef<HTMLInputElement | null>(null);

  async function loadRuns(silent = false) {
    if (!silent) setLoading(true);
    try {
      const response = await fetch("/api/journal/branden/market-review", { cache: "no-store" });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(errorMessage(data));
      const nextRuns = Array.isArray(data.runs) ? data.runs as ReviewRun[] : [];
      setRuns(nextRuns);
      setSelectedRunId((current) => current || nextRuns[0]?.run_id || "");
      setError("");
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : "Could not load market review runs.");
    } finally {
      if (!silent) setLoading(false);
    }
  }

  useEffect(() => {
    loadRuns();
  }, []);

  const hasActiveRun = runs.some((run) => run.status === "QUEUED" || run.status === "RUNNING");
  useEffect(() => {
    if (!hasActiveRun) return;
    const timer = window.setInterval(() => loadRuns(true), 8000);
    return () => window.clearInterval(timer);
  }, [hasActiveRun]);

  const selectedRun = useMemo(() => runs.find((run) => run.run_id === selectedRunId) || runs[0] || null, [runs, selectedRunId]);

  useEffect(() => {
    const items = selectedRun?.ocr && Array.isArray(selectedRun.ocr.items) ? selectedRun.ocr.items : [];
    setCorrectionsText(JSON.stringify(items, null, 2));
  }, [selectedRun?.run_id, selectedRun?.ocr]);

  async function startReview(event: FormEvent) {
    event.preventDefault();
    if (!pdf) {
      setError("Choose the exact-session MarketSurge screenshot PDF.");
      return;
    }
    setSubmitting(true);
    setError("");
    try {
      const formData = new FormData();
      formData.append("session_date", sessionDate);
      formData.append("marketsurge_pdf", pdf);
      const response = await fetch("/api/journal/branden/market-review", { method: "POST", body: formData });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(errorMessage(data));
      const run = data.run as ReviewRun;
      setRuns((current) => [run, ...current.filter((item) => item.run_id !== run.run_id)]);
      setSelectedRunId(run.run_id);
      setPdf(null);
      if (fileRef.current) fileRef.current.value = "";
    } catch (submitError) {
      setError(submitError instanceof Error ? submitError.message : "Could not start the market review.");
    } finally {
      setSubmitting(false);
    }
  }

  async function retryRun(run: ReviewRun) {
    setActionRunId(run.run_id);
    setError("");
    try {
      const response = await fetch(`/api/journal/branden/market-review/runs/${run.run_id}/retry`, { method: "POST" });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(errorMessage(data));
      setRuns((current) => current.map((item) => item.run_id === run.run_id ? data.run : item));
    } catch (retryError) {
      setError(retryError instanceof Error ? retryError.message : "Could not retry this run.");
    } finally {
      setActionRunId("");
    }
  }

  async function saveCorrections(run: ReviewRun) {
    setActionRunId(run.run_id);
    setError("");
    try {
      const corrections = JSON.parse(correctionsText);
      if (!Array.isArray(corrections)) throw new Error("OCR corrections must be a JSON array.");
      const response = await fetch(`/api/journal/branden/market-review/runs/${run.run_id}/ocr-corrections`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ expected_version: Number(run.ocr?.version || 0), corrections })
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(errorMessage(data));
      setRuns((current) => current.map((item) => item.run_id === run.run_id ? data.run : item));
    } catch (correctionError) {
      setError(correctionError instanceof Error ? correctionError.message : "Could not save OCR corrections.");
    } finally {
      setActionRunId("");
    }
  }

  return (
    <section className="market-review-page">
      <header className="market-review-hero">
        <div>
          <p className="eyebrow">Campus Fund</p>
          <h1>Daily Market &amp; Portfolio Review</h1>
          <p>Freeze the exact portfolio snapshot and MarketSurge PDF for one completed session, then follow the audited pipeline run.</p>
        </div>
        <span>Dashboard-driven · no routine Codex dependency</span>
      </header>

      <form className="market-review-start-card" onSubmit={startReview}>
        <div>
          <p className="eyebrow">Start one review</p>
          <h2>Exact-session source packet</h2>
          <p>The dashboard generates the existing daily snapshot internally. It is not emailed as an intermediate step.</p>
        </div>
        <label>
          Completed session
          <input type="date" required value={sessionDate} onChange={(event) => setSessionDate(event.target.value)} />
        </label>
        <label>
          MarketSurge screenshot PDF
          <input
            ref={fileRef}
            type="file"
            required
            accept="application/pdf,.pdf"
            onChange={(event) => setPdf(event.target.files?.[0] || null)}
          />
          <small>{pdf ? `${pdf.name} · ${(pdf.size / 1024 / 1024).toFixed(2)} MB` : "PDF only · maximum 20 MB · 1–75 pages"}</small>
        </label>
        <button type="submit" disabled={submitting}>{submitting ? "Validating and queueing…" : "Generate Review"}</button>
      </form>

      {error ? <div className="market-review-error" role="alert">{error}</div> : null}

      <div className="market-review-workspace">
        <aside className="market-review-run-list">
          <div className="market-review-section-heading">
            <div><p className="eyebrow">Run history</p><h2>Reviews</h2></div>
            <button type="button" onClick={() => loadRuns()} disabled={loading}>Refresh</button>
          </div>
          {loading ? <p>Loading reviews…</p> : runs.length ? runs.map((run) => (
            <button
              type="button"
              key={run.run_id}
              className={run.run_id === selectedRun?.run_id ? "active" : ""}
              onClick={() => setSelectedRunId(run.run_id)}
            >
              <span><strong>{run.session_date}</strong><small>Attempt {run.attempt}</small></span>
              <em className={`market-review-status ${run.status.toLowerCase()}`}>{run.status.replace("_", " ")}</em>
            </button>
          )) : <p>No market review runs yet.</p>}
        </aside>

        <article className="market-review-run-detail">
          {selectedRun ? (
            <>
              <div className="market-review-section-heading">
                <div><p className="eyebrow">Selected run</p><h2>{selectedRun.session_date}</h2></div>
                <em className={`market-review-status ${selectedRun.status.toLowerCase()}`}>{selectedRun.status.replace("_", " ")}</em>
              </div>

              <dl className="market-review-facts">
                <div><dt>Run ID</dt><dd>{selectedRun.run_id}</dd></div>
                <div><dt>Attempt</dt><dd>{selectedRun.attempt}</dd></div>
                <div><dt>Created</dt><dd>{formatTime(selectedRun.created_at)}</dd></div>
                <div><dt>Updated</dt><dd>{formatTime(selectedRun.updated_at)}</dd></div>
                <div><dt>MarketSurge PDF</dt><dd>{selectedRun.marketsurge_pdf_page_count} pages · {(selectedRun.marketsurge_pdf_size_bytes / 1024 / 1024).toFixed(2)} MB</dd></div>
                <div><dt>Temporary source</dt><dd>{selectedRun.source_deleted_at ? `Deleted ${formatTime(selectedRun.source_deleted_at)}` : `Expires ${formatTime(selectedRun.source_expires_at)}`}</dd></div>
                <div><dt>Workflow run</dt><dd>{selectedRun.github_workflow_url ? <a href={selectedRun.github_workflow_url} target="_blank" rel="noreferrer">{selectedRun.github_run_id || "Open workflow"}</a> : "Waiting for callback"}</dd></div>
                <div><dt>Email delivery</dt><dd>{selectedRun.delivery_status}{selectedRun.delivery_error ? ` — ${selectedRun.delivery_error}` : ""}</dd></div>
              </dl>

              <section className="market-review-hashes">
                <h3>Frozen source hashes</h3>
                <code>MarketSurge {shortHash(selectedRun.source_hashes.marketsurge_pdf_sha256)}</code>
                <code>Snapshot JSON {shortHash(selectedRun.source_hashes.snapshot_json_sha256)}</code>
                <code>Snapshot Markdown {shortHash(selectedRun.source_hashes.snapshot_markdown_sha256)}</code>
              </section>

              {selectedRun.error ? (
                <section className="market-review-run-error">
                  <strong>{selectedRun.error.code}</strong>
                  <p>{selectedRun.error.message}</p>
                </section>
              ) : null}

              {selectedRun.ocr ? (
                <section className="market-review-ocr">
                  <div><p className="eyebrow">OCR review</p><h3>{String(selectedRun.ocr.status || "RETURNED")}</h3></div>
                  <pre>{JSON.stringify(selectedRun.ocr, null, 2)}</pre>
                  {selectedRun.status === "NEEDS_REVIEW" ? (
                    <>
                      <label>Corrections JSON<textarea rows={8} value={correctionsText} onChange={(event) => setCorrectionsText(event.target.value)} /></label>
                      <button type="button" disabled={actionRunId === selectedRun.run_id} onClick={() => saveCorrections(selectedRun)}>Save OCR Corrections</button>
                    </>
                  ) : null}
                </section>
              ) : null}

              {selectedRun.artifacts.length ? (
                <section className="market-review-results">
                  <p className="eyebrow">Generated results</p>
                  <div>{selectedRun.artifacts.map((artifact) => (
                    <a key={artifact.kind} href={artifact.download_url}>Download {artifact.kind === "json" ? "JSON" : artifact.kind === "pdf" ? "PDF" : "Markdown"}</a>
                  ))}</div>
                </section>
              ) : null}

              {(selectedRun.status === "FAILED" || selectedRun.status === "NEEDS_REVIEW") && !selectedRun.source_deleted_at ? (
                <button className="market-review-retry" type="button" disabled={actionRunId === selectedRun.run_id} onClick={() => retryRun(selectedRun)}>
                  {actionRunId === selectedRun.run_id ? "Queueing retry…" : `Retry with frozen sources (attempt ${selectedRun.attempt + 1})`}
                </button>
              ) : null}
            </>
          ) : <div className="market-review-empty"><h2>Select or start a review</h2><p>Every run is tied to its explicit ID, session, attempt, and source hashes.</p></div>}
        </article>
      </div>
    </section>
  );
}
