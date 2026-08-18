import {
  MarketReviewValidationError,
  isSha256,
  type MarketReviewArtifactKind,
  type MarketReviewSourceKind
} from "./market-review-contract";

const SOURCE_MEDIA_TYPES: Record<MarketReviewSourceKind, readonly string[]> = {
  marketsurge_pdf: ["application/pdf"],
  snapshot_json: ["application/json"],
  snapshot_markdown: ["text/markdown", "text/markdown; charset=utf-8"],
  market_gauge_json: ["application/json"],
  ocr_corrections_json: ["application/json"]
};

const ARTIFACT_MEDIA_TYPES: Record<MarketReviewArtifactKind, readonly string[]> = {
  pdf: ["application/pdf"],
  markdown: ["text/markdown", "text/markdown; charset=utf-8"],
  json: ["application/json"]
};

function normalizeMediaType(value: string) {
  return value.trim().toLowerCase().replace(/\s*;\s*/g, "; ");
}

function requireExpectedMediaType(value: string, expected: readonly string[], code: string) {
  const normalized = normalizeMediaType(value);
  const allowed = expected.map(normalizeMediaType);
  if (!allowed.includes(normalized)) {
    throw new MarketReviewValidationError(code, "Stored market-review media type does not match the requested download kind.");
  }
  return allowed.includes("text/markdown; charset=utf-8") ? "text/markdown; charset=utf-8" : allowed[0];
}

export function marketReviewSourceMediaType(kind: MarketReviewSourceKind, storedMediaType: string) {
  return requireExpectedMediaType(storedMediaType, SOURCE_MEDIA_TYPES[kind], "SOURCE_CORRELATION_MISMATCH");
}

export function marketReviewArtifactMediaType(kind: MarketReviewArtifactKind, storedMediaType: string) {
  return requireExpectedMediaType(storedMediaType, ARTIFACT_MEDIA_TYPES[kind], "RESULT_ARTIFACT_METADATA_INVALID");
}

function filenameBasename(value: string) {
  const pathSegments = value.replace(/\\/g, "/").split("/");
  return pathSegments.at(-1) || "download";
}

function replaceInvalidUnicode(value: string) {
  let output = "";
  for (const character of value) {
    const codePoint = character.codePointAt(0) || 0;
    output += codePoint >= 0xd800 && codePoint <= 0xdfff ? "\ufffd" : character;
  }
  return output;
}

function safeExtendedFilename(value: string) {
  const sanitized = replaceInvalidUnicode(filenameBasename(value))
    .replace(/[\u0000-\u001f\u007f-\u009f]/gu, "_");
  return Array.from(sanitized).slice(0, 240).join("") || "download";
}

export function asciiDownloadFilename(value: string) {
  const extended = safeExtendedFilename(value);
  const extensionMatch = extended.match(/(\.[A-Za-z0-9]{1,10})$/);
  const extension = extensionMatch?.[1] || "";
  const rawStem = extension ? extended.slice(0, -extension.length) : extended;
  const stem = rawStem
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/[^A-Za-z0-9._ -]/g, "_")
    .replace(/[\s_]+/g, "_")
    .replace(/^[._ -]+|[._ -]+$/g, "")
    .slice(0, 180);
  return `${stem || "download"}${extension.toLowerCase()}`;
}

function encodeRfc5987(value: string) {
  return encodeURIComponent(value).replace(/[!'()*]/g, (character) => `%${character.charCodeAt(0).toString(16).toUpperCase()}`);
}

export function marketReviewContentDisposition(filename: string, disposition: "attachment" | "inline" = "attachment") {
  const extended = safeExtendedFilename(filename);
  const fallback = asciiDownloadFilename(extended);
  return `${disposition}; filename="${fallback}"; filename*=UTF-8''${encodeRfc5987(extended)}`;
}

export function buildMarketReviewDownloadResponse(
  body: BodyInit,
  input: { filename: string; contentType: string; sizeBytes: number; sha256: string; disposition?: "attachment" | "inline" }
) {
  if (!Number.isSafeInteger(input.sizeBytes) || input.sizeBytes < 0 || !isSha256(input.sha256)) {
    throw new MarketReviewValidationError("DOWNLOAD_METADATA_INVALID", "Stored market-review download size or SHA-256 is invalid.");
  }
  if (/[^\x20-\x7e]/.test(input.contentType) || /[\r\n]/.test(input.contentType)) {
    throw new MarketReviewValidationError("DOWNLOAD_MEDIA_TYPE_INVALID", "Stored market-review media type is unsafe.");
  }
  return new Response(body, {
    headers: {
      "Content-Type": input.contentType,
      "Content-Length": String(input.sizeBytes),
      "Content-Disposition": marketReviewContentDisposition(input.filename, input.disposition),
      "Cache-Control": "private, no-store",
      "X-Content-SHA256": input.sha256,
      "X-Content-Type-Options": "nosniff"
    }
  });
}
