import crypto from "node:crypto";
import { del, get, list } from "@vercel/blob";
import { generateDailyPortfolioSnapshot, SnapshotValidationError } from "./daily-portfolio-snapshot-server";
import {
  MARKET_REVIEW_CALLBACK_SCHEMA_VERSION,
  MARKET_REVIEW_SOURCE_RETENTION_MS,
  MarketReviewValidationError,
  requireSessionDate,
  sha256,
  signMarketReviewToken,
  sourceHashForKind,
  validateMarketSurgePdf,
  validateMarketReviewOcrCorrections,
  validateWorkerInputCorrelation,
  verifyMarketReviewToken,
  type MarketReviewCallbackPayload,
  type MarketReviewRun,
  type MarketReviewSourceKind,
  type MarketReviewWorkerCorrelation
} from "./market-review-contract";
import {
  MARKET_REVIEW_BLOB_PREFIX,
  selectExpiredOrphanMarketReviewBlobs,
  validateMarketReviewBlobReference,
  type MarketReviewBlobReference
} from "./market-review-upload";
import {
  applyMarketReviewCallback,
  createMarketReviewRun,
  failMarketReviewRun,
  getMarketReviewRun,
  getMarketReviewSource,
  listMarketReviewSourceBlobPathnames,
  queueMarketReviewRetry,
  recordMarketReviewDispatch,
  cleanupMarketReviewSources
} from "./market-review-store";
import type { SourceRecord } from "./market-review-store";

type DispatchFetch = typeof fetch;

export type MarketReviewGithubConfig = {
  token: string;
  repository: string;
  workflow: string;
  ref: string;
};

export function getMarketReviewGithubConfig(): MarketReviewGithubConfig {
  return {
    token: String(process.env.MARKET_REVIEW_GITHUB_TOKEN || "").trim(),
    repository: String(process.env.MARKET_REVIEW_GITHUB_REPOSITORY || "brandennevius/DailyMarketChartPipeline").trim(),
    workflow: String(process.env.MARKET_REVIEW_GITHUB_WORKFLOW || "daily-review.yml").trim(),
    ref: String(process.env.MARKET_REVIEW_GITHUB_REF || "main").trim()
  };
}

export function resolveMarketReviewBaseUrl(requestUrl: string) {
  const configured = String(process.env.MARKET_REVIEW_BASE_URL || "").trim().replace(/\/$/, "");
  if (configured) return configured;
  const url = new URL(requestUrl);
  return `${url.protocol}//${url.host}`;
}

export async function countPdfPages(data: Buffer) {
  const { PDFParse } = await import("pdf-parse");
  const parser = new PDFParse({ data });
  try {
    const info = await parser.getInfo();
    return info.total;
  } finally {
    await parser.destroy();
  }
}

export async function readAndValidateMarketReviewBlob(
  input: MarketReviewBlobReference,
  getImpl: typeof get = get,
  countPages: (data: Buffer) => Promise<number> = countPdfPages
) {
  const reference = validateMarketReviewBlobReference(input, input.session_date);
  const result = await getImpl(reference.blob_url, { access: "private", useCache: false });
  if (!result || result.statusCode !== 200 || !result.stream) {
    throw new MarketReviewValidationError("MARKETSURGE_BLOB_NOT_FOUND", "The uploaded MarketSurge PDF is no longer available in private storage.");
  }
  if (
    result.blob.url !== reference.blob_url ||
    result.blob.pathname !== reference.blob_pathname ||
    result.blob.contentType?.toLowerCase() !== "application/pdf" ||
    result.blob.size !== reference.size_bytes
  ) {
    throw new MarketReviewValidationError("MARKETSURGE_BLOB_METADATA_MISMATCH", "The private Blob metadata does not match the submitted MarketSurge PDF reference.");
  }
  const data = Buffer.from(await new Response(result.stream).arrayBuffer());
  const validated = await validateMarketSurgePdf({
    mimeType: reference.content_type,
    filename: reference.filename,
    data,
    countPages
  });
  if (validated.sha256 !== reference.sha256 || validated.sizeBytes !== reference.size_bytes) {
    throw new MarketReviewValidationError("MARKETSURGE_BLOB_HASH_MISMATCH", "The stored MarketSurge PDF bytes do not match the browser SHA-256 and size.");
  }
  return { reference, data, ...validated };
}

function sourceUrl(baseUrl: string, run: MarketReviewRun, kind: MarketReviewSourceKind, sha: string) {
  const token = signMarketReviewToken({
    scope: "source",
    review_run_id: run.run_id,
    session_date: run.session_date,
    attempt: run.attempt,
    source_hashes: run.source_hashes,
    source_kind: kind,
    source_sha256: sha
  });
  return `${baseUrl}/api/journal/branden/market-review/runs/${run.run_id}/source/${kind}?token=${encodeURIComponent(token)}`;
}

export function buildMarketReviewWorkerDispatch(run: MarketReviewRun, baseUrl: string) {
  return {
    review_run_id: run.run_id,
    session_date: run.session_date,
    attempt: String(run.attempt),
    marketsurge_pdf_sha256: run.source_hashes.marketsurge_pdf_sha256,
    snapshot_json_sha256: run.source_hashes.snapshot_json_sha256,
    snapshot_markdown_sha256: run.source_hashes.snapshot_markdown_sha256,
    market_gauge_json_sha256: run.source_hashes.market_gauge_json_sha256,
    worker_input_url: `${baseUrl}/api/journal/branden/market-review/runs/${run.run_id}/worker-input`,
    worker_callback_url: `${baseUrl}/api/journal/branden/market-review/runs/${run.run_id}/worker-callback`
  };
}

export async function dispatchMarketReviewRun(run: MarketReviewRun, baseUrl: string, fetchImpl: DispatchFetch = fetch) {
  const config = getMarketReviewGithubConfig();
  if (!config.token) throw new MarketReviewValidationError("GITHUB_DISPATCH_NOT_CONFIGURED", "MARKET_REVIEW_GITHUB_TOKEN is not configured.");
  const response = await fetchImpl(
    `https://api.github.com/repos/${config.repository}/actions/workflows/${encodeURIComponent(config.workflow)}/dispatches`,
    {
      method: "POST",
      headers: {
        Accept: "application/vnd.github+json",
        Authorization: `Bearer ${config.token}`,
        "Content-Type": "application/json",
        "X-GitHub-Api-Version": "2022-11-28"
      },
      body: JSON.stringify({ ref: config.ref, inputs: buildMarketReviewWorkerDispatch(run, baseUrl) })
    }
  );
  if (response.status !== 204) {
    const message = (await response.text()).slice(0, 500);
    throw new MarketReviewValidationError("GITHUB_DISPATCH_FAILED", `GitHub workflow dispatch failed (${response.status}).`, { response: message });
  }
  await recordMarketReviewDispatch(run.run_id, run.attempt);
}

export async function fetchExactSessionMarketGauge(
  baseUrl: string,
  session: string,
  fetchImpl: DispatchFetch = fetch
) {
  const response = await fetchImpl(`${baseUrl}/api/market-gauge?session_date=${encodeURIComponent(session)}`, {
    cache: "no-store",
    headers: { Accept: "application/json" }
  });
  const payload = await response.json().catch(() => null) as Record<string, unknown> | null;
  if (!response.ok || payload?.schema_version !== "dashboard_market_gauge_v1" || payload.session_date !== session) {
    throw new MarketReviewValidationError(
      "MARKET_GAUGE_SOURCE_INVALID",
      "The exact-session Dashboard Market Gauge could not be frozen for this review.",
      { status: response.status, code: payload?.code || null }
    );
  }
  return Buffer.from(`${JSON.stringify(payload, null, 2)}\n`, "utf8");
}

export function marketReviewSnapshotError(error: unknown) {
  if (error instanceof SnapshotValidationError) {
    return new MarketReviewValidationError(error.code, error.message, error.diagnostic);
  }
  return error;
}

export async function createAndDispatchMarketReview(input: {
  session: string;
  pdfBlob: MarketReviewBlobReference;
  baseUrl: string;
  now?: Date;
  countPages?: (data: Buffer) => Promise<number>;
  dispatch?: typeof dispatchMarketReviewRun;
  getBlob?: typeof get;
  deleteBlob?: typeof del;
  fetchMarketGauge?: typeof fetchExactSessionMarketGauge;
}) {
  const session = requireSessionDate(input.session);
  const now = input.now || new Date();
  const reference = validateMarketReviewBlobReference(input.pdfBlob, session);
  let runCreated = false;
  try {
    const pdf = await readAndValidateMarketReviewBlob(reference, input.getBlob || get, input.countPages || countPdfPages);
    const snapshotResult = await generateDailyPortfolioSnapshot({
      session,
      writeExports: false,
      dependencies: input.now ? { now: () => now } : undefined
    }).catch((error) => {
      throw marketReviewSnapshotError(error);
    });
    if (snapshotResult.snapshot.metadata.requested_session !== session) {
      throw new MarketReviewValidationError("SNAPSHOT_SESSION_MISMATCH", "The generated portfolio snapshot did not preserve the requested session.");
    }
    const snapshotJson = Buffer.from(`${JSON.stringify(snapshotResult.snapshot, null, 2)}\n`, "utf8");
    const snapshotMarkdown = Buffer.from(snapshotResult.markdown, "utf8");
    const marketGaugeJson = await (input.fetchMarketGauge || fetchExactSessionMarketGauge)(input.baseUrl, session);
    const sourceHashes = {
      marketsurge_pdf_sha256: pdf.sha256,
      snapshot_json_sha256: sha256(snapshotJson),
      snapshot_markdown_sha256: sha256(snapshotMarkdown),
      market_gauge_json_sha256: sha256(marketGaugeJson)
    };
    const config = getMarketReviewGithubConfig();
    const runId = crypto.randomUUID();
    const sourceExpiresAt = new Date(now.getTime() + MARKET_REVIEW_SOURCE_RETENTION_MS).toISOString();
    let run = await createMarketReviewRun({
      runId,
      sessionDate: session,
      sourceHashes,
      pdfFilename: reference.filename,
      pdfSizeBytes: pdf.sizeBytes,
      pdfPageCount: pdf.pageCount,
      sourceExpiresAt,
      githubRepository: config.repository,
      githubWorkflow: config.workflow,
      githubRef: config.ref,
      sources: [
        {
          kind: "marketsurge_pdf",
          filename: reference.filename,
          mediaType: "application/pdf",
          sha256: sourceHashes.marketsurge_pdf_sha256,
          blob: { url: reference.blob_url, pathname: reference.blob_pathname },
          sizeBytes: pdf.sizeBytes
        },
        { kind: "snapshot_json", filename: `daily-portfolio-snapshot-${session}.json`, mediaType: "application/json", sha256: sourceHashes.snapshot_json_sha256, data: snapshotJson },
        { kind: "snapshot_markdown", filename: `daily-portfolio-snapshot-${session}.md`, mediaType: "text/markdown; charset=utf-8", sha256: sourceHashes.snapshot_markdown_sha256, data: snapshotMarkdown },
        { kind: "market_gauge_json", filename: `market-gauge-${session}.json`, mediaType: "application/json", sha256: sourceHashes.market_gauge_json_sha256, data: marketGaugeJson }
      ]
    });
    runCreated = true;
    try {
      await (input.dispatch || dispatchMarketReviewRun)(run, input.baseUrl);
    } catch (error) {
      const code = error instanceof MarketReviewValidationError ? error.code : "GITHUB_DISPATCH_FAILED";
      const message = error instanceof Error ? error.message : "GitHub workflow dispatch failed.";
      run = await failMarketReviewRun(run.run_id, code, message, error instanceof MarketReviewValidationError ? error.details : undefined);
    }
    return run;
  } catch (error) {
    if (!runCreated) {
      await (input.deleteBlob || del)(reference.blob_url).catch(() => undefined);
    }
    throw error;
  }
}

export async function retryAndDispatchMarketReview(runId: string, baseUrl: string, dispatch: typeof dispatchMarketReviewRun = dispatchMarketReviewRun) {
  const existing = await getMarketReviewRun(runId);
  if (!existing) throw new MarketReviewValidationError("REVIEW_RUN_NOT_FOUND", "The requested market review run does not exist.");
  if (existing.ocr?.status === "CORRECTED" && existing.ocr.schema_version === "marketsurge_ocr_v2") {
    const correctionSource = await getMarketReviewSource(runId, "ocr_corrections_json");
    if (!dispatchableMarketReviewOcrCorrections(existing, correctionSource)) {
      throw new MarketReviewValidationError(
        "OCR_CORRECTIONS_INVALID",
        "Saved OCR corrections are invalid or no longer correlate with this run. Edit and resave them before retrying."
      );
    }
  }
  let run = await queueMarketReviewRetry(runId);
  try {
    await dispatch(run, baseUrl);
  } catch (error) {
    const code = error instanceof MarketReviewValidationError ? error.code : "GITHUB_DISPATCH_FAILED";
    const message = error instanceof Error ? error.message : "GitHub workflow dispatch failed.";
    run = await failMarketReviewRun(run.run_id, code, message, error instanceof MarketReviewValidationError ? error.details : undefined);
  }
  return run;
}

export async function cleanupExpiredMarketReviewSources(now = new Date()) {
  const databaseCleanup = await cleanupMarketReviewSources(now);
  const referenced = await listMarketReviewSourceBlobPathnames();
  const blobs: Array<{ pathname: string; uploadedAt: Date }> = [];
  let cursor: string | undefined;
  do {
    const page = await list({ prefix: `${MARKET_REVIEW_BLOB_PREFIX}/`, cursor, limit: 1000 });
    blobs.push(...page.blobs.map((blob) => ({ pathname: blob.pathname, uploadedAt: blob.uploadedAt })));
    cursor = page.hasMore ? page.cursor : undefined;
  } while (cursor);
  const orphanPathnames = selectExpiredOrphanMarketReviewBlobs(blobs, referenced, now.getTime());
  if (orphanPathnames.length) await del(orphanPathnames);
  return { ...databaseCleanup, deletedOrphanBlobCount: orphanPathnames.length };
}

export function verifyDashboardWorkerSecret(supplied: string) {
  const expected = String(process.env.DASHBOARD_WORKER_SECRET || "");
  if (!expected) throw new MarketReviewValidationError("WORKER_AUTH_NOT_CONFIGURED", "DASHBOARD_WORKER_SECRET is not configured.");
  const left = crypto.createHash("sha256").update(supplied).digest();
  const right = crypto.createHash("sha256").update(expected).digest();
  if (!supplied || !crypto.timingSafeEqual(left, right)) {
    throw new MarketReviewValidationError("WORKER_AUTH_INVALID", "Worker authentication failed.");
  }
}

export function dispatchableMarketReviewOcrCorrections(
  run: MarketReviewRun,
  source: SourceRecord | null
): SourceRecord | null {
  if (!source?.data || run.ocr?.status !== "CORRECTED") return null;
  if (
    sha256(source.data) !== source.sha256
    || String(run.ocr.correction_sha256 || "") !== source.sha256
  ) return null;
  try {
    const payload = JSON.parse(source.data.toString("utf8")) as Record<string, unknown>;
    if (payload.schema_version !== "marketsurge_ocr_corrections_v2") return null;
    if (Number(payload.expected_version) + 1 !== Number(run.ocr.version)) return null;
    validateMarketReviewOcrCorrections(payload.corrections);
    return source;
  } catch {
    return null;
  }
}

export async function getMarketReviewWorkerInput(
  runId: string,
  workerSecret: string,
  baseUrl: string,
  correlation: MarketReviewWorkerCorrelation
) {
  const run = await getMarketReviewRun(runId);
  if (!run) throw new MarketReviewValidationError("REVIEW_RUN_NOT_FOUND", "The requested market review run does not exist.");
  verifyDashboardWorkerSecret(workerSecret);
  validateWorkerInputCorrelation(run, correlation);
  if (run.source_deleted_at || new Date(run.source_expires_at).getTime() <= Date.now()) {
    throw new MarketReviewValidationError("SOURCE_PACKET_UNAVAILABLE", "The exact review source packet has expired or was deleted.");
  }
  const corrections = dispatchableMarketReviewOcrCorrections(
    run,
    await getMarketReviewSource(run.run_id, "ocr_corrections_json")
  );
  return {
    schema_version: "campus-fund-market-review-worker-input-v1",
    review_run_id: run.run_id,
    session_date: run.session_date,
    attempt: run.attempt,
    source_hashes: run.source_hashes,
    portfolio_snapshot: {
      download_url: sourceUrl(baseUrl, run, "snapshot_json", run.source_hashes.snapshot_json_sha256),
      sha256: run.source_hashes.snapshot_json_sha256,
      markdown_download_url: sourceUrl(baseUrl, run, "snapshot_markdown", run.source_hashes.snapshot_markdown_sha256),
      markdown_sha256: run.source_hashes.snapshot_markdown_sha256
    },
    marketsurge_pdf: {
      download_url: sourceUrl(baseUrl, run, "marketsurge_pdf", run.source_hashes.marketsurge_pdf_sha256),
      sha256: run.source_hashes.marketsurge_pdf_sha256,
      page_count: run.marketsurge_pdf_page_count,
      filename: run.marketsurge_pdf_filename
    },
    market_gauge: {
      download_url: sourceUrl(baseUrl, run, "market_gauge_json", run.source_hashes.market_gauge_json_sha256),
      sha256: run.source_hashes.market_gauge_json_sha256
    },
    ocr_corrections: corrections
      ? {
          download_url: sourceUrl(baseUrl, run, "ocr_corrections_json", corrections.sha256),
          sha256: corrections.sha256,
          version: Number(run.ocr?.version || 0)
        }
      : null,
    callback: {
      url: `${baseUrl}/api/journal/branden/market-review/runs/${run.run_id}/worker-callback`,
      schema_version: MARKET_REVIEW_CALLBACK_SCHEMA_VERSION,
      token: signMarketReviewToken({
        scope: "callback",
        review_run_id: run.run_id,
        session_date: run.session_date,
        attempt: run.attempt,
        source_hashes: run.source_hashes
      })
    }
  };
}

export async function authorizeMarketReviewSource(runId: string, kind: MarketReviewSourceKind, token: string) {
  const run = await getMarketReviewRun(runId);
  if (!run) throw new MarketReviewValidationError("REVIEW_RUN_NOT_FOUND", "The requested market review run does not exist.");
  const source = await getMarketReviewSource(runId, kind);
  if (!source) throw new MarketReviewValidationError("SOURCE_NOT_FOUND", "The requested exact source does not exist or has been deleted.");
  const expectedHash = sourceHashForKind(run.source_hashes, kind) || source.sha256;
  verifyMarketReviewToken(token, {
    scope: "source",
    review_run_id: run.run_id,
    session_date: run.session_date,
    attempt: run.attempt,
    source_hashes: run.source_hashes,
    source_kind: kind,
    source_sha256: expectedHash
  });
  if (source.sha256 !== expectedHash || new Date(source.expiresAt).getTime() <= Date.now()) {
    throw new MarketReviewValidationError("SOURCE_CORRELATION_MISMATCH", "The stored source hash or expiry does not match the signed request.");
  }
  return source;
}

export async function acceptMarketReviewCallback(runId: string, token: string, payload: MarketReviewCallbackPayload) {
  const run = await getMarketReviewRun(runId);
  if (!run) throw new MarketReviewValidationError("REVIEW_RUN_NOT_FOUND", "The requested market review run does not exist.");
  verifyMarketReviewToken(token, {
    scope: "callback",
    review_run_id: run.run_id,
    session_date: run.session_date,
    attempt: run.attempt,
    source_hashes: run.source_hashes
  });
  return applyMarketReviewCallback(payload);
}
