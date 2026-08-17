import { MARKET_REVIEW_MAX_SOURCE_PDF_BYTES, MarketReviewValidationError, isSha256, requireSessionDate } from "./market-review-contract";
import {
  MARKET_REVIEW_BLOB_PREFIX,
  marketReviewBlobPathnameValue,
  type MarketReviewBlobReference,
  type MarketReviewUploadDescriptor
} from "./market-review-upload-shared";

export { MARKET_REVIEW_BLOB_PREFIX } from "./market-review-upload-shared";
export type { MarketReviewBlobReference, MarketReviewUploadDescriptor } from "./market-review-upload-shared";
export const MARKET_REVIEW_UPLOAD_TOKEN_TTL_MS = 10 * 60 * 1000;

function isUuid(value: unknown) {
  return typeof value === "string" && /^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(value);
}

export function validateMarketReviewUploadDescriptor(value: unknown): MarketReviewUploadDescriptor {
  if (!value || typeof value !== "object") {
    throw new MarketReviewValidationError("MARKETSURGE_UPLOAD_METADATA_INVALID", "MarketSurge upload metadata is required.");
  }
  const input = value as Partial<MarketReviewUploadDescriptor>;
  if (!isUuid(input.upload_id)) {
    throw new MarketReviewValidationError("MARKETSURGE_UPLOAD_ID_INVALID", "A valid MarketSurge upload ID is required.");
  }
  const sessionDate = requireSessionDate(input.session_date);
  const filename = String(input.filename || "").trim();
  if (!filename || filename.length > 240 || !filename.toLowerCase().endsWith(".pdf")) {
    throw new MarketReviewValidationError("MARKETSURGE_PDF_EXTENSION_INVALID", "The MarketSurge source filename must end in .pdf.");
  }
  const contentType = String(input.content_type || "").toLowerCase();
  if (contentType !== "application/pdf") {
    throw new MarketReviewValidationError("MARKETSURGE_PDF_MIME_INVALID", "The MarketSurge source must have MIME type application/pdf.");
  }
  const sizeBytes = Number(input.size_bytes);
  if (!Number.isInteger(sizeBytes) || sizeBytes < 1 || sizeBytes > MARKET_REVIEW_MAX_SOURCE_PDF_BYTES) {
    throw new MarketReviewValidationError(
      "MARKETSURGE_PDF_SIZE_INVALID",
      `The MarketSurge PDF must be between 1 byte and ${MARKET_REVIEW_MAX_SOURCE_PDF_BYTES} bytes.`
    );
  }
  const hash = String(input.sha256 || "").toLowerCase();
  if (!isSha256(hash)) {
    throw new MarketReviewValidationError("MARKETSURGE_PDF_HASH_INVALID", "A valid MarketSurge PDF SHA-256 is required.");
  }
  return {
    upload_id: String(input.upload_id).toLowerCase(),
    session_date: sessionDate,
    filename,
    content_type: contentType,
    size_bytes: sizeBytes,
    sha256: hash
  };
}

export function marketReviewBlobPathname(input: MarketReviewUploadDescriptor) {
  const descriptor = validateMarketReviewUploadDescriptor(input);
  return marketReviewBlobPathnameValue(descriptor);
}

export function parseMarketReviewUploadClientPayload(value: string | null) {
  if (!value) throw new MarketReviewValidationError("MARKETSURGE_UPLOAD_METADATA_INVALID", "MarketSurge upload metadata is required.");
  try {
    return validateMarketReviewUploadDescriptor(JSON.parse(value));
  } catch (error) {
    if (error instanceof MarketReviewValidationError) throw error;
    throw new MarketReviewValidationError("MARKETSURGE_UPLOAD_METADATA_INVALID", "MarketSurge upload metadata must be valid JSON.");
  }
}

export function requireMarketReviewCreateJson(contentType: string | null) {
  if (!(contentType || "").toLowerCase().startsWith("application/json")) {
    throw new MarketReviewValidationError(
      "MARKETSURGE_DIRECT_UPLOAD_REQUIRED",
      "Upload the MarketSurge PDF directly to private storage before creating the review."
    );
  }
}

export function validateMarketReviewBlobReference(value: unknown, expectedSession?: string): MarketReviewBlobReference {
  if (!value || typeof value !== "object") {
    throw new MarketReviewValidationError("MARKETSURGE_BLOB_REFERENCE_INVALID", "A stored MarketSurge PDF reference is required.");
  }
  const input = value as Partial<MarketReviewBlobReference>;
  const descriptor = validateMarketReviewUploadDescriptor(input);
  if (expectedSession && descriptor.session_date !== requireSessionDate(expectedSession)) {
    throw new MarketReviewValidationError("MARKETSURGE_UPLOAD_SESSION_MISMATCH", "The stored PDF session does not match the requested review session.");
  }
  const blobUrl = String(input.blob_url || "").trim();
  let parsedUrl: URL;
  try {
    parsedUrl = new URL(blobUrl);
  } catch {
    throw new MarketReviewValidationError("MARKETSURGE_BLOB_URL_INVALID", "The stored MarketSurge PDF URL is invalid.");
  }
  if (parsedUrl.protocol !== "https:" || !/^[a-z0-9-]+\.private\.blob\.vercel-storage\.com$/i.test(parsedUrl.hostname)) {
    throw new MarketReviewValidationError("MARKETSURGE_BLOB_URL_INVALID", "The MarketSurge PDF must be stored in the configured private Vercel Blob store.");
  }
  const blobPathname = String(input.blob_pathname || "").replace(/^\/+/, "");
  const expectedPathname = marketReviewBlobPathname(descriptor);
  if (blobPathname !== expectedPathname || decodeURIComponent(parsedUrl.pathname.replace(/^\/+/, "")) !== expectedPathname) {
    throw new MarketReviewValidationError("MARKETSURGE_BLOB_PATH_MISMATCH", "The stored MarketSurge PDF path does not match its session, upload ID, and hash.");
  }
  const blobContentType = String(input.blob_content_type || "").toLowerCase();
  if (blobContentType !== "application/pdf") {
    throw new MarketReviewValidationError("MARKETSURGE_PDF_MIME_INVALID", "The stored MarketSurge source must have MIME type application/pdf.");
  }
  return {
    ...descriptor,
    blob_url: blobUrl,
    blob_pathname: blobPathname,
    blob_content_type: blobContentType
  };
}

export function selectExpiredOrphanMarketReviewBlobs(
  blobs: Array<{ pathname: string; uploadedAt: Date }>,
  referencedPathnames: ReadonlySet<string>,
  now = Date.now()
) {
  const cutoff = now - 24 * 60 * 60 * 1000;
  return blobs
    .filter((blob) => blob.pathname.startsWith(`${MARKET_REVIEW_BLOB_PREFIX}/`))
    .filter((blob) => !referencedPathnames.has(blob.pathname) && blob.uploadedAt.getTime() <= cutoff)
    .map((blob) => blob.pathname);
}
