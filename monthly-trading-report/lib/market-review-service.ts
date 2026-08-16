import crypto from "node:crypto";
import { generateDailyPortfolioSnapshot } from "./daily-portfolio-snapshot-server";
import {
  MARKET_REVIEW_CALLBACK_SCHEMA_VERSION,
  MARKET_REVIEW_SOURCE_RETENTION_MS,
  MarketReviewValidationError,
  requireSessionDate,
  sha256,
  signMarketReviewToken,
  sourceHashForKind,
  validateMarketSurgePdf,
  validateWorkerInputCorrelation,
  verifyMarketReviewToken,
  type MarketReviewCallbackPayload,
  type MarketReviewRun,
  type MarketReviewSourceKind,
  type MarketReviewWorkerCorrelation
} from "./market-review-contract";
import {
  applyMarketReviewCallback,
  createMarketReviewRun,
  failMarketReviewRun,
  getMarketReviewRun,
  getMarketReviewSource,
  queueMarketReviewRetry,
  recordMarketReviewDispatch
} from "./market-review-store";

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
    workflow: String(process.env.MARKET_REVIEW_GITHUB_WORKFLOW || "dashboard-review.yml").trim(),
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

export async function createAndDispatchMarketReview(input: {
  session: string;
  pdfFilename: string;
  pdfMimeType: string;
  pdfData: Buffer;
  baseUrl: string;
  now?: Date;
  countPages?: (data: Buffer) => Promise<number>;
  dispatch?: typeof dispatchMarketReviewRun;
}) {
  const session = requireSessionDate(input.session);
  const now = input.now || new Date();
  const pdf = await validateMarketSurgePdf({
    mimeType: input.pdfMimeType,
    filename: input.pdfFilename,
    data: input.pdfData,
    countPages: input.countPages || countPdfPages
  });
  const snapshotResult = await generateDailyPortfolioSnapshot({
    session,
    writeExports: false,
    dependencies: input.now ? { now: () => now } : undefined
  });
  if (snapshotResult.snapshot.metadata.requested_session !== session) {
    throw new MarketReviewValidationError("SNAPSHOT_SESSION_MISMATCH", "The generated portfolio snapshot did not preserve the requested session.");
  }
  const snapshotJson = Buffer.from(`${JSON.stringify(snapshotResult.snapshot, null, 2)}\n`, "utf8");
  const snapshotMarkdown = Buffer.from(snapshotResult.markdown, "utf8");
  const sourceHashes = {
    marketsurge_pdf_sha256: pdf.sha256,
    snapshot_json_sha256: sha256(snapshotJson),
    snapshot_markdown_sha256: sha256(snapshotMarkdown)
  };
  const config = getMarketReviewGithubConfig();
  const runId = crypto.randomUUID();
  const sourceExpiresAt = new Date(now.getTime() + MARKET_REVIEW_SOURCE_RETENTION_MS).toISOString();
  let run = await createMarketReviewRun({
    runId,
    sessionDate: session,
    sourceHashes,
    pdfFilename: input.pdfFilename,
    pdfSizeBytes: pdf.sizeBytes,
    pdfPageCount: pdf.pageCount,
    sourceExpiresAt,
    githubRepository: config.repository,
    githubWorkflow: config.workflow,
    githubRef: config.ref,
    sources: [
      { kind: "marketsurge_pdf", filename: input.pdfFilename, mediaType: "application/pdf", sha256: sourceHashes.marketsurge_pdf_sha256, data: input.pdfData },
      { kind: "snapshot_json", filename: `daily-portfolio-snapshot-${session}.json`, mediaType: "application/json", sha256: sourceHashes.snapshot_json_sha256, data: snapshotJson },
      { kind: "snapshot_markdown", filename: `daily-portfolio-snapshot-${session}.md`, mediaType: "text/markdown; charset=utf-8", sha256: sourceHashes.snapshot_markdown_sha256, data: snapshotMarkdown }
    ]
  });
  try {
    await (input.dispatch || dispatchMarketReviewRun)(run, input.baseUrl);
  } catch (error) {
    const code = error instanceof MarketReviewValidationError ? error.code : "GITHUB_DISPATCH_FAILED";
    const message = error instanceof Error ? error.message : "GitHub workflow dispatch failed.";
    run = await failMarketReviewRun(run.run_id, code, message, error instanceof MarketReviewValidationError ? error.details : undefined);
  }
  return run;
}

export async function retryAndDispatchMarketReview(runId: string, baseUrl: string, dispatch: typeof dispatchMarketReviewRun = dispatchMarketReviewRun) {
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

export function verifyDashboardWorkerSecret(supplied: string) {
  const expected = String(process.env.DASHBOARD_WORKER_SECRET || "");
  if (!expected) throw new MarketReviewValidationError("WORKER_AUTH_NOT_CONFIGURED", "DASHBOARD_WORKER_SECRET is not configured.");
  const left = crypto.createHash("sha256").update(supplied).digest();
  const right = crypto.createHash("sha256").update(expected).digest();
  if (!supplied || !crypto.timingSafeEqual(left, right)) {
    throw new MarketReviewValidationError("WORKER_AUTH_INVALID", "Worker authentication failed.");
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
  const corrections = await getMarketReviewSource(run.run_id, "ocr_corrections_json");
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
