"use client";

import { upload } from "@vercel/blob/client";
import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import {
  MARKET_REVIEW_MAX_SOURCE_PDF_BYTES,
  buildMarketReviewCreatePayload,
  marketReviewBlobPathnameValue,
  type MarketReviewBlobReference,
  type MarketReviewUploadDescriptor
} from "@/lib/market-review-upload-shared";
import {
  buildMarketReviewOcrCorrections,
  canRetryMarketReview,
  hasSavedV2MarketReviewCorrections,
  normalizeMarketReviewTicker,
  parseMarketReviewOcrReview
} from "@/lib/market-review-ocr-ui";

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
    market_gauge_json_sha256: string;
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

async function fileSha256(file: File) {
  const digest = await crypto.subtle.digest("SHA-256", await file.arrayBuffer());
  return Array.from(new Uint8Array(digest)).map((value) => value.toString(16).padStart(2, "0")).join("");
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
  const [ocrResolutions, setOcrResolutions] = useState<Record<string, string>>({});
  const [reviewedOcrPages, setReviewedOcrPages] = useState<Record<number, boolean>>({});
  const [previewPage, setPreviewPage] = useState<number | null>(null);
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
  const selectedRunHasLegacyCorrections = selectedRun?.ocr?.status === "CORRECTED"
    && selectedRun.ocr.schema_version !== "marketsurge_ocr_v2";
  const ocrReviewPages = useMemo(() => parseMarketReviewOcrReview(selectedRun?.ocr), [selectedRun?.ocr]);
  const ocrReviewResult = useMemo(
    () => buildMarketReviewOcrCorrections(ocrReviewPages, ocrResolutions, reviewedOcrPages),
    [ocrReviewPages, ocrResolutions, reviewedOcrPages]
  );
  const hasSavedV2Corrections = hasSavedV2MarketReviewCorrections(selectedRun?.ocr);
  const hasFrozenMarketGauge = Boolean(selectedRun?.source_hashes.market_gauge_json_sha256);
  const retryAllowed = selectedRun ? hasFrozenMarketGauge && canRetryMarketReview(selectedRun.status, selectedRun.ocr) : false;

  useEffect(() => {
    setOcrResolutions({});
    setReviewedOcrPages({});
    setPreviewPage(null);
  }, [selectedRun?.run_id, selectedRun?.ocr]);

  async function startReview(event: FormEvent) {
    event.preventDefault();
    if (!pdf) {
      setError("Choose the exact-session MarketSurge screenshot PDF.");
      return;
    }
    if (pdf.type.toLowerCase() !== "application/pdf" || !pdf.name.toLowerCase().endsWith(".pdf")) {
      setError("Choose a PDF file with MIME type application/pdf.");
      return;
    }
    if (!pdf.size || pdf.size > MARKET_REVIEW_MAX_SOURCE_PDF_BYTES) {
      setError("The MarketSurge PDF must be between 1 byte and 20 MB.");
      return;
    }
    setSubmitting(true);
    setError("");
    try {
      const descriptor: MarketReviewUploadDescriptor = {
        upload_id: crypto.randomUUID(),
        session_date: sessionDate,
        filename: pdf.name,
        content_type: pdf.type.toLowerCase(),
        size_bytes: pdf.size,
        sha256: await fileSha256(pdf)
      };
      const blob = await upload(marketReviewBlobPathnameValue(descriptor), pdf, {
        access: "private",
        contentType: "application/pdf",
        handleUploadUrl: "/api/journal/branden/market-review/upload",
        clientPayload: JSON.stringify(descriptor),
        multipart: true
      });
      const reference: MarketReviewBlobReference = {
        ...descriptor,
        blob_url: blob.url,
        blob_pathname: blob.pathname,
        blob_content_type: blob.contentType
      };
      const response = await fetch("/api/journal/branden/market-review", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(buildMarketReviewCreatePayload(reference))
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        await fetch("/api/journal/branden/market-review/upload", {
          method: "DELETE",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ marketsurge_pdf: reference })
        }).catch(() => undefined);
        throw new Error(errorMessage(data));
      }
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
      if (!ocrReviewResult.ready) throw new Error(ocrReviewResult.errors[0] || "Complete every OCR review item before saving.");
      const response = await fetch(`/api/journal/branden/market-review/runs/${run.run_id}/ocr-corrections`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ expected_version: Number(run.ocr?.version || 0), corrections: ocrReviewResult.corrections })
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
        <button type="submit" disabled={submitting}>{submitting ? "Uploading and validating…" : "Generate Review"}</button>
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
                <code>Market Gauge {shortHash(selectedRun.source_hashes.market_gauge_json_sha256)}</code>
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
                  {selectedRun.status === "NEEDS_REVIEW" && !hasSavedV2Corrections ? (
                    <div className="market-review-ocr-workflow">
                      <p className="market-review-ocr-instructions">
                        Resolve every ambiguous symbol against the frozen PDF, then explicitly review each page. The dashboard builds the v2 correction packet; raw JSON editing is not required.
                      </p>
                      <div className="market-review-ocr-progress">
                        <strong>{ocrReviewPages.reduce((total, page) => total + page.reviewRows.length, 0)} ambiguous rows</strong>
                        <span>{Object.values(reviewedOcrPages).filter(Boolean).length} of {ocrReviewPages.length} pages reviewed</span>
                      </div>
                      {ocrReviewPages.map((page) => (
                        <section className="market-review-ocr-page" key={page.pdfPage}>
                          <header>
                            <div>
                              <p>Page {page.pdfPage}</p>
                              <h4>{page.label}</h4>
                              <small>{page.acceptedTickers.length} symbols accepted automatically · {page.reviewRows.length} need review</small>
                            </div>
                            <button type="button" className="market-review-evidence-toggle" onClick={() => setPreviewPage((current) => current === page.pdfPage ? null : page.pdfPage)}>
                              {previewPage === page.pdfPage ? "Hide evidence" : "View evidence"}
                            </button>
                          </header>
                          {previewPage === page.pdfPage ? (
                            <div className="market-review-evidence-preview">
                              <object
                                aria-label={`Frozen MarketSurge PDF page ${page.pdfPage}`}
                                data={`/api/journal/branden/market-review/runs/${selectedRun.run_id}/evidence#page=${page.pdfPage}&view=FitH`}
                                type="application/pdf"
                              >
                                <a href={`/api/journal/branden/market-review/runs/${selectedRun.run_id}/evidence#page=${page.pdfPage}`} target="_blank" rel="noreferrer">Open frozen PDF at page {page.pdfPage}</a>
                              </object>
                            </div>
                          ) : null}
                          {page.reviewRows.length ? (
                            <div className="market-review-ocr-rows">
                              {page.reviewRows.map((row) => {
                                const normalized = normalizeMarketReviewTicker(ocrResolutions[row.key]);
                                return (
                                  <div className={`market-review-ocr-row ${normalized ? "resolved" : ""}`} key={row.key}>
                                    <dl>
                                      <div><dt>Rank</dt><dd>{row.rank ?? "Not detected"}</dd></div>
                                      <div><dt>Raw OCR</dt><dd>{row.rawText || "Nothing detected"}</dd></div>
                                      <div><dt>Candidate</dt><dd>{row.candidateTicker || "None"}</dd></div>
                                      <div><dt>Confidence</dt><dd>{row.confidence === null ? "Unavailable" : `${row.confidence.toFixed(2)}%`}</dd></div>
                                      <div><dt>Reason</dt><dd>{row.reason.replaceAll("_", " ")}</dd></div>
                                    </dl>
                                    <label>
                                      Verified ticker
                                      <input
                                        aria-label={`Verified ticker for page ${page.pdfPage} rank ${row.rank ?? "unknown"}`}
                                        autoCapitalize="characters"
                                        maxLength={10}
                                        pattern="[A-Za-z][A-Za-z0-9.-]{0,9}"
                                        placeholder={row.candidateTicker ? `Check ${row.candidateTicker} against the PDF` : "Enter the ticker shown in the PDF"}
                                        value={ocrResolutions[row.key] || ""}
                                        onChange={(event) => setOcrResolutions((current) => ({ ...current, [row.key]: event.target.value.toUpperCase() }))}
                                      />
                                      <small>{normalized ? `Validated as ${normalized}` : "Required · ticker format only"}</small>
                                    </label>
                                  </div>
                                );
                              })}
                            </div>
                          ) : <p className="market-review-ocr-clear">No ambiguous rows were returned for this page.</p>}
                          <label className="market-review-ocr-confirm">
                            <input
                              type="checkbox"
                              checked={Boolean(reviewedOcrPages[page.pdfPage])}
                              disabled={page.reviewRows.some((row) => !normalizeMarketReviewTicker(ocrResolutions[row.key]))}
                              onChange={(event) => setReviewedOcrPages((current) => ({ ...current, [page.pdfPage]: event.target.checked }))}
                            />
                            I checked page {page.pdfPage}, its section, and every ambiguous ticker against the frozen evidence.
                          </label>
                        </section>
                      ))}
                      {!ocrReviewResult.ready ? <p className="market-review-ocr-blocker">Save remains disabled until all ambiguous rows are valid and all {ocrReviewPages.length} pages are reviewed.</p> : null}
                      <button type="button" disabled={!ocrReviewResult.ready || actionRunId === selectedRun.run_id} onClick={() => saveCorrections(selectedRun)}>Save reviewed OCR corrections</button>
                    </div>
                  ) : hasSavedV2Corrections ? (
                    <p className="market-review-ocr-saved">Valid v2 corrections are saved. Retry is now available and will use the frozen PDF and reviewed correction packet.</p>
                  ) : null}
                  <details className="market-review-ocr-diagnostics">
                    <summary>Raw OCR diagnostics</summary>
                    <pre>{JSON.stringify(selectedRun.ocr, null, 2)}</pre>
                  </details>
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
                <>
                  {!hasFrozenMarketGauge ? (
                    <p className="market-review-retry-note">
                      This legacy run has no frozen exact-session Market Gauge source. Start a new review; retry is disabled so source packets cannot be mixed.
                    </p>
                  ) : null}
                  {selectedRunHasLegacyCorrections ? (
                    <p className="market-review-retry-note">
                      Legacy OCR corrections will be ignored. Retry will rerun the corrected OCR parser from the frozen PDF.
                    </p>
                  ) : null}
                  {hasFrozenMarketGauge && selectedRun.status === "NEEDS_REVIEW" && !retryAllowed ? (
                    <p className="market-review-retry-note">Retry is locked until every OCR page is reviewed and a valid v2 correction packet is saved.</p>
                  ) : null}
                  <button className="market-review-retry" type="button" disabled={!retryAllowed || actionRunId === selectedRun.run_id} onClick={() => retryRun(selectedRun)}>
                    {actionRunId === selectedRun.run_id
                      ? "Queueing retry…"
                      : selectedRunHasLegacyCorrections
                        ? `Rerun OCR from frozen PDF (attempt ${selectedRun.attempt + 1})`
                        : `Retry with frozen sources (attempt ${selectedRun.attempt + 1})`}
                  </button>
                </>
              ) : null}
            </>
          ) : <div className="market-review-empty"><h2>Select or start a review</h2><p>Every run is tied to its explicit ID, session, attempt, and source hashes.</p></div>}
        </article>
      </div>
    </section>
  );
}
