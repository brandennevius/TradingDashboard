import crypto from "node:crypto";
import { MARKET_REVIEW_MAX_SOURCE_PDF_BYTES } from "./market-review-upload-shared";

export { MARKET_REVIEW_MAX_SOURCE_PDF_BYTES } from "./market-review-upload-shared";

export const MARKET_REVIEW_SCHEMA_VERSION = "campus-fund-market-review-v1" as const;
export const MARKET_REVIEW_CALLBACK_SCHEMA_VERSION = "campus-fund-market-review-callback-v1" as const;
export const MARKET_REVIEW_MAX_SOURCE_PAGES = 75;
export const MARKET_REVIEW_SOURCE_RETENTION_MS = 24 * 60 * 60 * 1000;
export const MARKET_REVIEW_TOKEN_TTL_SECONDS = 2 * 60 * 60;

export type MarketReviewStatus = "QUEUED" | "RUNNING" | "NEEDS_REVIEW" | "FAILED" | "COMPLETED";
export type MarketReviewDeliveryStatus = "NOT_REQUESTED" | "PENDING" | "SENDING" | "SENT" | "FAILED";
export type MarketReviewSourceKind = "marketsurge_pdf" | "snapshot_json" | "snapshot_markdown" | "market_gauge_json" | "ocr_corrections_json";
export type MarketReviewArtifactKind = "pdf" | "markdown" | "json";
export type MarketReviewCallbackEventType = "RUNNING" | "OCR_REVIEW_REQUIRED" | "FAILED" | "RESULTS_REGISTERED" | "DELIVERY_STATUS";

export type MarketReviewSourceHashes = {
  marketsurge_pdf_sha256: string;
  snapshot_json_sha256: string;
  snapshot_markdown_sha256: string;
  market_gauge_json_sha256: string;
};

export type MarketReviewArtifactInput = {
  kind: MarketReviewArtifactKind;
  filename: string;
  media_type: string;
  sha256: string;
  size_bytes: number;
  content_base64: string;
};

export type MarketReviewCallbackPayload = {
  schema_version: typeof MARKET_REVIEW_CALLBACK_SCHEMA_VERSION;
  event_id: string;
  event_type: MarketReviewCallbackEventType;
  review_run_id: string;
  session_date: string;
  attempt: number;
  source_hashes: MarketReviewSourceHashes;
  github?: {
    repository?: string;
    workflow_run_id?: string;
    workflow_run_attempt?: number;
    workflow_url?: string;
  };
  error?: { code: string; message: string; details?: unknown };
  ocr?: { status: string; version?: number; items?: unknown[]; message?: string };
  audit?: { status: "PASS" | "FAIL"; packet_sha256?: string; evidence?: unknown };
  delivery?: { status: MarketReviewDeliveryStatus; error?: string | null };
  artifacts?: MarketReviewArtifactInput[];
};

export type MarketReviewRun = {
  run_id: string;
  schema_version: typeof MARKET_REVIEW_SCHEMA_VERSION;
  session_date: string;
  status: MarketReviewStatus;
  attempt: number;
  source_hashes: MarketReviewSourceHashes;
  marketsurge_pdf_filename: string;
  marketsurge_pdf_size_bytes: number;
  marketsurge_pdf_page_count: number;
  source_expires_at: string;
  source_deleted_at: string | null;
  github_repository: string;
  github_workflow: string;
  github_ref: string;
  github_run_id: string | null;
  github_run_attempt: number | null;
  github_workflow_url: string | null;
  ocr: Record<string, unknown> | null;
  error: { code: string; message: string; details?: unknown } | null;
  delivery_status: MarketReviewDeliveryStatus;
  delivery_error: string | null;
  artifacts: Array<{
    kind: MarketReviewArtifactKind;
    filename: string;
    media_type: string;
    sha256: string;
    size_bytes: number;
    download_url: string;
  }>;
  created_at: string;
  updated_at: string;
  completed_at: string | null;
};

export type MarketReviewTokenClaims = {
  version: 1;
  scope: "source" | "callback";
  review_run_id: string;
  session_date: string;
  attempt: number;
  source_hashes: MarketReviewSourceHashes;
  source_kind?: MarketReviewSourceKind;
  source_sha256?: string;
  exp: number;
};

export type MarketReviewWorkerCorrelation = {
  attempt: number;
  source_hashes: MarketReviewSourceHashes;
};

export class MarketReviewValidationError extends Error {
  constructor(public readonly code: string, message: string, public readonly details?: unknown) {
    super(message);
    this.name = "MarketReviewValidationError";
  }
}

export function sha256(data: Buffer | string) {
  return crypto.createHash("sha256").update(data).digest("hex");
}

export function isSha256(value: unknown): value is string {
  return typeof value === "string" && /^[a-f0-9]{64}$/.test(value);
}

export function requireSessionDate(value: unknown) {
  const session = String(value || "").trim();
  if (!/^\d{4}-\d{2}-\d{2}$/.test(session)) {
    throw new MarketReviewValidationError("SESSION_DATE_INVALID", "Choose a session in YYYY-MM-DD format.");
  }
  const [year, month, day] = session.split("-").map(Number);
  const parsed = new Date(Date.UTC(year, month - 1, day));
  if (parsed.getUTCFullYear() !== year || parsed.getUTCMonth() !== month - 1 || parsed.getUTCDate() !== day) {
    throw new MarketReviewValidationError("SESSION_DATE_INVALID", "Choose a valid calendar session.");
  }
  return session;
}

const MARKET_REVIEW_OCR_LABELS = new Set([
  "Breaking Out Today",
  "Recent Breakouts",
  "Tight Areas",
  "Near Pivot",
  "Power from Pivot",
  "Top Rated Stocks",
  "BRANDENS WATCHLIST"
]);
const MARKET_REVIEW_OCR_EXCLUDED_TOKENS = new Set([
  "FAVORITES", "MARKET", "MARKETS", "NAME", "SCREENS", "STOCK", "STOCKS", "SYMBOL", "TICKER"
]);

export type MarketReviewOcrCorrection = {
  pdf_page: number;
  label: string;
  tickers: string[];
  reviewed: true;
};

export function validateMarketReviewOcrCorrections(value: unknown): MarketReviewOcrCorrection[] {
  if (!Array.isArray(value) || !value.length || value.length > MARKET_REVIEW_MAX_SOURCE_PAGES) {
    throw new MarketReviewValidationError(
      "OCR_CORRECTIONS_INVALID",
      `Submit one reviewed correction per PDF page, up to ${MARKET_REVIEW_MAX_SOURCE_PAGES} pages.`
    );
  }
  const pages = new Set<number>();
  return value.map((raw) => {
    if (!raw || typeof raw !== "object") {
      throw new MarketReviewValidationError("OCR_CORRECTIONS_INVALID", "Every OCR correction must be an object.");
    }
    const item = raw as Record<string, unknown>;
    const pdfPage = item.pdf_page;
    const label = item.label;
    const tickers = item.tickers;
    if (
      !Number.isInteger(pdfPage)
      || Number(pdfPage) < 1
      || pages.has(Number(pdfPage))
      || typeof label !== "string"
      || !MARKET_REVIEW_OCR_LABELS.has(label)
      || !Array.isArray(tickers)
      || item.reviewed !== true
    ) {
      throw new MarketReviewValidationError(
        "OCR_CORRECTIONS_INVALID",
        "Each correction requires a unique pdf_page, a known section label, a ticker array, and reviewed=true."
      );
    }
    pages.add(Number(pdfPage));
    const normalizedTickers: string[] = [];
    for (const rawTicker of tickers) {
      const ticker = String(rawTicker || "").trim().toUpperCase();
      if (!/^[A-Z][A-Z0-9.-]{0,9}$/.test(ticker) || MARKET_REVIEW_OCR_EXCLUDED_TOKENS.has(ticker)) {
        throw new MarketReviewValidationError(
          "OCR_CORRECTIONS_INVALID",
          `Page ${pdfPage} contains an invalid or navigation ticker token.`
        );
      }
      if (!normalizedTickers.includes(ticker)) normalizedTickers.push(ticker);
    }
    return { pdf_page: Number(pdfPage), label, tickers: normalizedTickers, reviewed: true };
  });
}

export async function validateMarketSurgePdf(input: {
  mimeType: string;
  filename: string;
  data: Buffer;
  countPages: (data: Buffer) => Promise<number>;
}) {
  if (input.mimeType.toLowerCase() !== "application/pdf") {
    throw new MarketReviewValidationError("MARKETSURGE_PDF_MIME_INVALID", "The MarketSurge source must have MIME type application/pdf.");
  }
  if (!input.filename.toLowerCase().endsWith(".pdf")) {
    throw new MarketReviewValidationError("MARKETSURGE_PDF_EXTENSION_INVALID", "The MarketSurge source filename must end in .pdf.");
  }
  if (!input.data.length || input.data.length > MARKET_REVIEW_MAX_SOURCE_PDF_BYTES) {
    throw new MarketReviewValidationError(
      "MARKETSURGE_PDF_SIZE_INVALID",
      `The MarketSurge PDF must be between 1 byte and ${MARKET_REVIEW_MAX_SOURCE_PDF_BYTES} bytes.`
    );
  }
  if (input.data.subarray(0, 5).toString("ascii") !== "%PDF-") {
    throw new MarketReviewValidationError("MARKETSURGE_PDF_MAGIC_INVALID", "The uploaded file does not have a valid PDF signature.");
  }
  let pageCount: number;
  try {
    pageCount = await input.countPages(input.data);
  } catch (error) {
    throw new MarketReviewValidationError(
      "MARKETSURGE_PDF_PARSE_FAILED",
      error instanceof Error ? `The MarketSurge PDF could not be parsed: ${error.message}` : "The MarketSurge PDF could not be parsed."
    );
  }
  if (!Number.isInteger(pageCount) || pageCount < 1 || pageCount > MARKET_REVIEW_MAX_SOURCE_PAGES) {
    throw new MarketReviewValidationError(
      "MARKETSURGE_PDF_PAGE_COUNT_INVALID",
      `The MarketSurge PDF must contain between 1 and ${MARKET_REVIEW_MAX_SOURCE_PAGES} pages.`
    );
  }
  return { pageCount, sha256: sha256(input.data), sizeBytes: input.data.length };
}

function base64UrlEncode(value: Buffer | string) {
  return Buffer.from(value).toString("base64url");
}

function tokenSecret() {
  const secret = process.env.MARKET_REVIEW_TOKEN_SECRET || process.env.APP_SECRET;
  if (!secret || secret === "local-dev-secret-change-before-deploying") {
    if (process.env.NODE_ENV === "production") {
      throw new Error("MARKET_REVIEW_TOKEN_SECRET or APP_SECRET is required in production.");
    }
    return "local-market-review-secret-change-before-deploying";
  }
  return secret;
}

export function signMarketReviewToken(claims: Omit<MarketReviewTokenClaims, "version" | "exp"> & { exp?: number }) {
  const payload: MarketReviewTokenClaims = {
    ...claims,
    version: 1,
    exp: claims.exp || Math.floor(Date.now() / 1000) + MARKET_REVIEW_TOKEN_TTL_SECONDS
  };
  const encoded = base64UrlEncode(JSON.stringify(payload));
  const signature = crypto.createHmac("sha256", tokenSecret()).update(encoded).digest("base64url");
  return `${encoded}.${signature}`;
}

export function verifyMarketReviewToken(token: string, expected: Partial<MarketReviewTokenClaims>, now = Date.now()) {
  const [encoded, signature] = token.split(".");
  if (!encoded || !signature) throw new MarketReviewValidationError("TOKEN_INVALID", "The review token is malformed.");
  const actual = crypto.createHmac("sha256", tokenSecret()).update(encoded).digest();
  const supplied = Buffer.from(signature, "base64url");
  if (actual.length !== supplied.length || !crypto.timingSafeEqual(actual, supplied)) {
    throw new MarketReviewValidationError("TOKEN_INVALID", "The review token signature is invalid.");
  }
  let claims: MarketReviewTokenClaims;
  try {
    claims = JSON.parse(Buffer.from(encoded, "base64url").toString("utf8")) as MarketReviewTokenClaims;
  } catch {
    throw new MarketReviewValidationError("TOKEN_INVALID", "The review token payload is invalid.");
  }
  if (claims.version !== 1 || !Number.isFinite(claims.exp) || claims.exp * 1000 < now) {
    throw new MarketReviewValidationError("TOKEN_EXPIRED", "The review token has expired.");
  }
  for (const [key, value] of Object.entries(expected)) {
    if (value === undefined) continue;
    const actualValue = claims[key as keyof MarketReviewTokenClaims];
    if (typeof value === "object") {
      const left = actualValue as Partial<MarketReviewSourceHashes> | undefined;
      const right = value as Partial<MarketReviewSourceHashes>;
      if (
        left?.marketsurge_pdf_sha256 !== right.marketsurge_pdf_sha256 ||
        left?.snapshot_json_sha256 !== right.snapshot_json_sha256 ||
        left?.snapshot_markdown_sha256 !== right.snapshot_markdown_sha256 ||
        left?.market_gauge_json_sha256 !== right.market_gauge_json_sha256
      ) {
        throw new MarketReviewValidationError("TOKEN_CORRELATION_MISMATCH", `The review token does not match ${key}.`);
      }
    } else if (actualValue !== value) {
      throw new MarketReviewValidationError("TOKEN_CORRELATION_MISMATCH", `The review token does not match ${key}.`);
    }
  }
  return claims;
}

export function sourceHashForKind(hashes: MarketReviewSourceHashes, kind: MarketReviewSourceKind) {
  if (kind === "marketsurge_pdf") return hashes.marketsurge_pdf_sha256;
  if (kind === "snapshot_json") return hashes.snapshot_json_sha256;
  if (kind === "snapshot_markdown") return hashes.snapshot_markdown_sha256;
  if (kind === "market_gauge_json") return hashes.market_gauge_json_sha256;
  return undefined;
}

const allowedTransitions: Record<MarketReviewStatus, ReadonlySet<MarketReviewStatus>> = {
  QUEUED: new Set(["QUEUED", "RUNNING", "NEEDS_REVIEW", "FAILED", "COMPLETED"]),
  RUNNING: new Set(["RUNNING", "NEEDS_REVIEW", "FAILED", "COMPLETED"]),
  NEEDS_REVIEW: new Set(["NEEDS_REVIEW", "QUEUED", "FAILED", "COMPLETED"]),
  FAILED: new Set(["QUEUED"]),
  COMPLETED: new Set()
};

export function assertMarketReviewTransition(from: MarketReviewStatus, to: MarketReviewStatus) {
  if (!allowedTransitions[from]?.has(to)) {
    throw new MarketReviewValidationError("STATUS_TRANSITION_INVALID", `A market review cannot move from ${from} to ${to}.`);
  }
}

export function callbackStatus(eventType: MarketReviewCallbackEventType): MarketReviewStatus | null {
  if (eventType === "RUNNING") return "RUNNING";
  if (eventType === "OCR_REVIEW_REQUIRED") return "NEEDS_REVIEW";
  if (eventType === "FAILED") return "FAILED";
  if (eventType === "RESULTS_REGISTERED") return "COMPLETED";
  return null;
}

export function validateCallbackCorrelation(run: MarketReviewRun, payload: MarketReviewCallbackPayload) {
  if (payload.schema_version !== MARKET_REVIEW_CALLBACK_SCHEMA_VERSION) {
    throw new MarketReviewValidationError("CALLBACK_SCHEMA_INVALID", "The callback schema version is not supported.");
  }
  if (!Number.isInteger(payload.attempt) || payload.attempt < 1) {
    throw new MarketReviewValidationError("CALLBACK_ATTEMPT_INVALID", "Callback attempt must be a positive integer.");
  }
  if (!payload.source_hashes || !isSha256(payload.source_hashes.marketsurge_pdf_sha256)
    || !isSha256(payload.source_hashes.snapshot_json_sha256)
    || !isSha256(payload.source_hashes.snapshot_markdown_sha256)
    || !isSha256(payload.source_hashes.market_gauge_json_sha256)) {
    throw new MarketReviewValidationError("CALLBACK_SOURCE_HASHES_INVALID", "Callback source hashes must contain all four SHA-256 values.");
  }
  if (!new Set<MarketReviewCallbackEventType>(["RUNNING", "OCR_REVIEW_REQUIRED", "FAILED", "RESULTS_REGISTERED", "DELIVERY_STATUS"]).has(payload.event_type)) {
    throw new MarketReviewValidationError("CALLBACK_EVENT_INVALID", "The callback event type is not supported.");
  }
  if (payload.review_run_id !== run.run_id || payload.session_date !== run.session_date || payload.attempt !== run.attempt) {
    throw new MarketReviewValidationError("CALLBACK_CORRELATION_MISMATCH", "Callback run, session, or attempt does not match the stored review.");
  }
  if (
    payload.source_hashes.marketsurge_pdf_sha256 !== run.source_hashes.marketsurge_pdf_sha256 ||
    payload.source_hashes.snapshot_json_sha256 !== run.source_hashes.snapshot_json_sha256 ||
    payload.source_hashes.snapshot_markdown_sha256 !== run.source_hashes.snapshot_markdown_sha256 ||
    payload.source_hashes.market_gauge_json_sha256 !== run.source_hashes.market_gauge_json_sha256
  ) {
    throw new MarketReviewValidationError("CALLBACK_SOURCE_HASH_MISMATCH", "Callback source hashes do not match the stored review sources.");
  }
  if (!payload.event_id || payload.event_id.length > 160) {
    throw new MarketReviewValidationError("CALLBACK_EVENT_ID_INVALID", "A bounded callback event_id is required.");
  }
  const next = callbackStatus(payload.event_type);
  if (next) assertMarketReviewTransition(run.status, next);
  if (payload.event_type === "FAILED" && (!payload.error?.code || !payload.error.message)) {
    throw new MarketReviewValidationError("CALLBACK_ERROR_REQUIRED", "FAILED callbacks require an error code and message.");
  }
  if (payload.event_type === "OCR_REVIEW_REQUIRED" && !payload.ocr?.status) {
    throw new MarketReviewValidationError("CALLBACK_OCR_REQUIRED", "OCR_REVIEW_REQUIRED callbacks require OCR review details.");
  }
  if (payload.event_type === "RESULTS_REGISTERED") {
    if (payload.audit?.status !== "PASS") {
      throw new MarketReviewValidationError("STRICT_AUDIT_REQUIRED", "Results cannot be registered until the pipeline strict audit passes.");
    }
    const kinds = new Set((payload.artifacts || []).map((artifact) => artifact.kind));
    if (kinds.size !== 3 || !kinds.has("pdf") || !kinds.has("markdown") || !kinds.has("json")) {
      throw new MarketReviewValidationError("RESULT_ARTIFACTS_INCOMPLETE", "PDF, Markdown, and JSON result artifacts are all required.");
    }
    const packet = (payload.artifacts || []).find((artifact) => artifact.kind === "json");
    if (!isSha256(payload.audit.packet_sha256) || payload.audit.packet_sha256 !== packet?.sha256) {
      throw new MarketReviewValidationError("STRICT_AUDIT_PACKET_HASH_MISMATCH", "The strict-audit packet hash must match the registered JSON packet.");
    }
  }
  if (payload.event_type === "DELIVERY_STATUS") {
    if (run.status !== "COMPLETED") {
      throw new MarketReviewValidationError("DELIVERY_STATUS_NOT_ALLOWED", "Delivery status can be updated only after report completion.");
    }
    if (!payload.delivery?.status) {
      throw new MarketReviewValidationError("DELIVERY_STATUS_REQUIRED", "DELIVERY_STATUS callbacks require delivery details.");
    }
  }
}

export function validateWorkerInputCorrelation(run: MarketReviewRun, correlation: MarketReviewWorkerCorrelation) {
  if (!Number.isInteger(correlation.attempt) || correlation.attempt < 1) {
    throw new MarketReviewValidationError("WORKER_ATTEMPT_INVALID", "Worker attempt must be a positive integer.");
  }
  if (!correlation.source_hashes || !isSha256(correlation.source_hashes.marketsurge_pdf_sha256)
    || !isSha256(correlation.source_hashes.snapshot_json_sha256)
    || !isSha256(correlation.source_hashes.snapshot_markdown_sha256)
    || !isSha256(correlation.source_hashes.market_gauge_json_sha256)) {
    throw new MarketReviewValidationError("WORKER_SOURCE_HASHES_INVALID", "Worker request must present all four source SHA-256 values.");
  }
  if (correlation.attempt !== run.attempt) {
    throw new MarketReviewValidationError("WORKER_ATTEMPT_MISMATCH", "Worker attempt does not match the current review attempt.");
  }
  if (correlation.source_hashes.marketsurge_pdf_sha256 !== run.source_hashes.marketsurge_pdf_sha256
    || correlation.source_hashes.snapshot_json_sha256 !== run.source_hashes.snapshot_json_sha256
    || correlation.source_hashes.snapshot_markdown_sha256 !== run.source_hashes.snapshot_markdown_sha256
    || correlation.source_hashes.market_gauge_json_sha256 !== run.source_hashes.market_gauge_json_sha256) {
    throw new MarketReviewValidationError("WORKER_SOURCE_HASH_MISMATCH", "Worker source hashes do not match the frozen review sources.");
  }
}

export function deriveCallbackUpdate(run: MarketReviewRun, payload: MarketReviewCallbackPayload, duplicate = false) {
  if (duplicate) {
    return {
      duplicate: true,
      status: run.status,
      deliveryStatus: run.delivery_status,
      deliveryError: run.delivery_error,
      deleteSources: false
    };
  }
  validateCallbackCorrelation(run, payload);
  return {
    duplicate: false,
    status: callbackStatus(payload.event_type) || run.status,
    deliveryStatus: payload.delivery?.status || run.delivery_status,
    deliveryError: payload.delivery?.error ?? run.delivery_error,
    deleteSources: payload.event_type === "RESULTS_REGISTERED"
  };
}

export function decodeAndValidateResultArtifacts(inputs: MarketReviewArtifactInput[]) {
  const totalLimit = 35 * 1024 * 1024;
  let total = 0;
  const seen = new Set<MarketReviewArtifactKind>();
  return inputs.map((input) => {
    if (seen.has(input.kind)) throw new MarketReviewValidationError("RESULT_ARTIFACT_DUPLICATE", `Duplicate ${input.kind} result artifact.`);
    seen.add(input.kind);
    if (!isSha256(input.sha256) || !Number.isInteger(input.size_bytes) || input.size_bytes < 1) {
      throw new MarketReviewValidationError("RESULT_ARTIFACT_METADATA_INVALID", `Invalid metadata for ${input.kind} result artifact.`);
    }
    const data = Buffer.from(input.content_base64, "base64");
    total += data.length;
    if (data.length !== input.size_bytes || sha256(data) !== input.sha256) {
      throw new MarketReviewValidationError("RESULT_ARTIFACT_HASH_MISMATCH", `${input.kind} result bytes do not match the registered size or SHA-256.`);
    }
    if (input.kind === "pdf" && data.subarray(0, 5).toString("ascii") !== "%PDF-") {
      throw new MarketReviewValidationError("RESULT_PDF_MAGIC_INVALID", "The result PDF does not have a valid PDF signature.");
    }
    if (input.kind === "json") {
      try {
        JSON.parse(data.toString("utf8"));
      } catch {
        throw new MarketReviewValidationError("RESULT_JSON_INVALID", "The result JSON artifact is not valid JSON.");
      }
    }
    return { ...input, data };
  }).map((artifact) => {
    if (total > totalLimit) throw new MarketReviewValidationError("RESULT_ARTIFACTS_TOO_LARGE", "Combined result artifacts exceed 35 MB.");
    return artifact;
  });
}

export function shouldDeleteMarketReviewSources(run: Pick<MarketReviewRun, "status" | "source_deleted_at" | "source_expires_at">, now = Date.now()) {
  if (run.source_deleted_at) return false;
  if (run.status === "COMPLETED") return true;
  return new Date(run.source_expires_at).getTime() <= now;
}
