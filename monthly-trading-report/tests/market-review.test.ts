import assert from "node:assert/strict";
import test from "node:test";
import {
  MARKET_REVIEW_CALLBACK_SCHEMA_VERSION,
  MARKET_REVIEW_MAX_SOURCE_PDF_BYTES,
  MarketReviewValidationError,
  assertMarketReviewTransition,
  decodeAndValidateResultArtifacts,
  deriveCallbackUpdate,
  sha256,
  shouldDeleteMarketReviewSources,
  signMarketReviewToken,
  validateCallbackCorrelation,
  validateMarketSurgePdf,
  validateMarketReviewOcrCorrections,
  validateWorkerInputCorrelation,
  verifyMarketReviewToken,
  type MarketReviewCallbackPayload,
  type MarketReviewRun
} from "../lib/market-review-contract";
import {
  asciiDownloadFilename,
  buildMarketReviewDownloadResponse,
  marketReviewArtifactMediaType,
  marketReviewContentDisposition,
  marketReviewSourceMediaType
} from "../lib/market-review-download";
import {
  buildMarketReviewWorkerDispatch,
  dispatchableMarketReviewOcrCorrections,
  fetchExactSessionMarketGauge,
  getMarketReviewGithubConfig,
  readAndValidateMarketReviewBlob,
  verifyDashboardWorkerSecret,
  marketReviewSnapshotError
} from "../lib/market-review-service";
import { SnapshotValidationError } from "../lib/daily-portfolio-snapshot-server";
import type { SourceRecord } from "../lib/market-review-store";
import {
  marketReviewBlobPathname,
  parseMarketReviewUploadClientPayload,
  requireMarketReviewCreateJson,
  selectExpiredOrphanMarketReviewBlobs,
  validateMarketReviewBlobReference
} from "../lib/market-review-upload";
import { buildMarketReviewCreatePayload, type MarketReviewBlobReference } from "../lib/market-review-upload-shared";
import { serializeMarketReviewSessionDate } from "../lib/market-review-store";
import {
  buildMarketReviewOcrCorrections,
  canRetryMarketReview,
  hasSavedV2MarketReviewCorrections,
  marketReviewOcrRowKey,
  parseMarketReviewOcrReview
} from "../lib/market-review-ocr-ui";

const hashes = {
  marketsurge_pdf_sha256: "a".repeat(64),
  snapshot_json_sha256: "b".repeat(64),
  snapshot_markdown_sha256: "c".repeat(64),
  market_gauge_json_sha256: "d".repeat(64)
};

test("exact-session market gauge source is canonical and rejects mismatched sessions", async () => {
  const okFetch = async () => new Response(JSON.stringify({
    schema_version: "dashboard_market_gauge_v1",
    session_date: "2026-08-14",
    overall_state: "Neutral"
  }), { status: 200, headers: { "Content-Type": "application/json" } });
  const bytes = await fetchExactSessionMarketGauge("https://dashboard.example", "2026-08-14", okFetch as typeof fetch);
  assert.equal(JSON.parse(bytes.toString("utf8")).session_date, "2026-08-14");

  const mismatchFetch = async () => new Response(JSON.stringify({
    schema_version: "dashboard_market_gauge_v1",
    session_date: "2026-08-13"
  }), { status: 200, headers: { "Content-Type": "application/json" } });
  await assert.rejects(
    fetchExactSessionMarketGauge("https://dashboard.example", "2026-08-14", mismatchFetch as typeof fetch),
    (error: unknown) => error instanceof MarketReviewValidationError && error.code === "MARKET_GAUGE_SOURCE_INVALID"
  );
});

test("snapshot validation failures retain their machine-readable code at the market-review boundary", () => {
  const mapped = marketReviewSnapshotError(new SnapshotValidationError(
    "CURRENT_PRICES_INVALID",
    "Exact-session price evidence is unavailable."
  ));

  assert(mapped instanceof MarketReviewValidationError);
  assert.equal(mapped.code, "CURRENT_PRICES_INVALID");
  assert.equal(mapped.details, undefined);
  const unrelated = new Error("unrelated");
  assert.equal(marketReviewSnapshotError(unrelated), unrelated);
});

function run(overrides: Partial<MarketReviewRun> = {}): MarketReviewRun {
  return {
    run_id: "11111111-1111-4111-8111-111111111111",
    schema_version: "campus-fund-market-review-v1",
    session_date: "2026-08-14",
    status: "QUEUED",
    attempt: 1,
    source_hashes: hashes,
    marketsurge_pdf_filename: "MarketSurge.pdf",
    marketsurge_pdf_size_bytes: 1000,
    marketsurge_pdf_page_count: 4,
    source_expires_at: "2026-08-16T00:00:00.000Z",
    source_deleted_at: null,
    github_repository: "brandennevius/DailyMarketChartPipeline",
    github_workflow: "daily-review.yml",
    github_ref: "main",
    github_run_id: null,
    github_run_attempt: null,
    github_workflow_url: null,
    ocr: null,
    error: null,
    delivery_status: "NOT_REQUESTED",
    delivery_error: null,
    artifacts: [],
    created_at: "2026-08-15T00:00:00.000Z",
    updated_at: "2026-08-15T00:00:00.000Z",
    completed_at: null,
    ...overrides
  };
}

function callback(eventType: MarketReviewCallbackPayload["event_type"], overrides: Partial<MarketReviewCallbackPayload> = {}): MarketReviewCallbackPayload {
  return {
    schema_version: MARKET_REVIEW_CALLBACK_SCHEMA_VERSION,
    event_id: `event-${eventType.toLowerCase()}`,
    event_type: eventType,
    review_run_id: run().run_id,
    session_date: run().session_date,
    attempt: 1,
    source_hashes: hashes,
    ...overrides
  };
}

test("MarketSurge PDF validation checks MIME, magic, page count, and size", async () => {
  const valid = Buffer.from("%PDF-1.7\nbody");
  assert.deepEqual(await validateMarketSurgePdf({ mimeType: "application/pdf", filename: "scan.pdf", data: valid, countPages: async () => 3 }), {
    pageCount: 3,
    sha256: sha256(valid),
    sizeBytes: valid.length
  });

  await assert.rejects(
    validateMarketSurgePdf({ mimeType: "image/png", filename: "scan.pdf", data: valid, countPages: async () => 3 }),
    (error: unknown) => error instanceof MarketReviewValidationError && error.code === "MARKETSURGE_PDF_MIME_INVALID"
  );
  await assert.rejects(
    validateMarketSurgePdf({ mimeType: "application/pdf", filename: "scan.pdf", data: Buffer.from("not-pdf"), countPages: async () => 3 }),
    (error: unknown) => error instanceof MarketReviewValidationError && error.code === "MARKETSURGE_PDF_MAGIC_INVALID"
  );
  await assert.rejects(
    validateMarketSurgePdf({ mimeType: "application/pdf", filename: "scan.pdf", data: valid, countPages: async () => 0 }),
    (error: unknown) => error instanceof MarketReviewValidationError && error.code === "MARKETSURGE_PDF_PAGE_COUNT_INVALID"
  );
  await assert.rejects(
    validateMarketSurgePdf({ mimeType: "application/pdf", filename: "scan.pdf", data: Buffer.alloc(MARKET_REVIEW_MAX_SOURCE_PDF_BYTES + 1), countPages: async () => 1 }),
    (error: unknown) => error instanceof MarketReviewValidationError && error.code === "MARKETSURGE_PDF_SIZE_INVALID"
  );
});

const attemptFourOcrFixture = {
  schema_version: "marketsurge_ocr_v2",
  status: "NEEDS_REVIEW",
  version: 4,
  items: [
    { pdf_page: 1, label: "Breaking Out Today", tickers: ["CVE"], review_rows: [] },
    {
      pdf_page: 2,
      label: "Recent Breakouts",
      tickers: ["LNVGY", "NESR"],
      review_rows: [
        { rank: 1, raw_text: "FEL", candidate_ticker: "FEL", confidence: 67.56, reason: "LOW_CONFIDENCE_SYMBOL" },
        { rank: 2, raw_text: ">)", candidate_ticker: ">)", confidence: 50.34, reason: "SYMBOL_TOKEN_INVALID" },
        { rank: 9, raw_text: "ATl", candidate_ticker: "ATL", confidence: 73.67, reason: "SYMBOL_CASE_AMBIGUOUS" }
      ]
    },
    { pdf_page: 3, label: "Tight Areas", tickers: ["PCAR"], review_rows: [] },
    { pdf_page: 4, label: "Near Pivot", tickers: ["NVDA"], review_rows: [] },
    { pdf_page: 5, label: "Power from Pivot", tickers: ["QNST"], review_rows: [{ rank: 40, raw_text: "S)", candidate_ticker: "S)", confidence: 41.67, reason: "SYMBOL_TOKEN_INVALID" }] },
    {
      pdf_page: 6,
      label: "Power from Pivot",
      tickers: ["CDNA"],
      review_rows: [
        { rank: 23, raw_text: "FET,", candidate_ticker: "FET", confidence: 62.23, reason: "LOW_CONFIDENCE_SYMBOL" },
        { rank: 26, raw_text: "Oll", candidate_ticker: "OLL", confidence: 91.81, reason: "SYMBOL_CASE_AMBIGUOUS" }
      ]
    },
    {
      pdf_page: 7,
      label: "Top Rated Stocks",
      tickers: ["ATEYY"],
      review_rows: [
        { rank: 11, raw_text: null, candidate_ticker: null, confidence: null, reason: "SYMBOL_CELL_NOT_DETECTED" },
        { rank: 16, raw_text: "DGIl", candidate_ticker: "DGIL", confidence: 68.15, reason: "SYMBOL_CASE_AMBIGUOUS" }
      ]
    },
    { pdf_page: 8, label: "BRANDENS WATCHLIST", tickers: ["CPRT"], review_rows: [] },
    { pdf_page: 9, label: "Top Rated Stocks", tickers: ["ATI"], review_rows: [] }
  ]
};

test("structured OCR review parses all attempt-four pages and ambiguous rows", () => {
  const pages = parseMarketReviewOcrReview(attemptFourOcrFixture);
  assert.equal(pages.length, 9);
  assert.equal(pages.reduce((total, page) => total + page.reviewRows.length, 0), 8);
  assert.deepEqual(pages.filter((page) => page.reviewRows.length).map((page) => [page.pdfPage, page.reviewRows.map((row) => row.rank)]), [
    [2, [1, 2, 9]],
    [5, [40]],
    [6, [23, 26]],
    [7, [11, 16]]
  ]);
  assert.deepEqual(pages[1].reviewRows[0], {
    key: "2:1:0",
    pdfPage: 2,
    label: "Recent Breakouts",
    rank: 1,
    rawText: "FEL",
    candidateTicker: "FEL",
    confidence: 67.56,
    reason: "LOW_CONFIDENCE_SYMBOL"
  });
});

test("structured OCR review remains fail-closed until every row and page is reviewed", () => {
  const pages = parseMarketReviewOcrReview(attemptFourOcrFixture);
  const incomplete = buildMarketReviewOcrCorrections(pages, { [marketReviewOcrRowKey(2, 1, 0)]: "FET" }, { 1: true });
  assert.equal(incomplete.ready, false);
  assert.ok(incomplete.errors.some((error) => error.includes("Page 2 rank 2")));
  assert.ok(incomplete.errors.some((error) => error.includes("Page 9 has not been marked reviewed")));

  const resolutions = {
    [marketReviewOcrRowKey(2, 1, 0)]: "FET",
    [marketReviewOcrRowKey(2, 2, 1)]: "P",
    [marketReviewOcrRowKey(2, 9, 2)]: "ATI",
    [marketReviewOcrRowKey(5, 40, 0)]: "S",
    [marketReviewOcrRowKey(6, 23, 0)]: "FET",
    [marketReviewOcrRowKey(6, 26, 1)]: "OII",
    [marketReviewOcrRowKey(7, 11, 0)]: "P",
    [marketReviewOcrRowKey(7, 16, 1)]: "DGII"
  };
  const reviewedPages = Object.fromEntries(pages.map((page) => [page.pdfPage, true]));
  const complete = buildMarketReviewOcrCorrections(pages, resolutions, reviewedPages);
  assert.equal(complete.ready, true);
  assert.deepEqual(complete.errors, []);
  assert.deepEqual(complete.corrections.find((item) => item.pdf_page === 2), {
    pdf_page: 2,
    label: "Recent Breakouts",
    tickers: ["LNVGY", "NESR", "FET", "P", "ATI"],
    reviewed: true
  });
  assert.deepEqual(complete.corrections.find((item) => item.pdf_page === 7)?.tickers, ["ATEYY", "P", "DGII"]);
});

test("NEEDS_REVIEW retry stays locked until valid v2 corrections are saved", () => {
  assert.equal(canRetryMarketReview("NEEDS_REVIEW", attemptFourOcrFixture), false);
  assert.equal(canRetryMarketReview("NEEDS_REVIEW", { ...attemptFourOcrFixture, status: "CORRECTED", corrections: [{ reviewed: true }] }), true);
  assert.equal(canRetryMarketReview("NEEDS_REVIEW", { status: "CORRECTED", schema_version: "marketsurge_ocr_v1", corrections: [{ reviewed: true }] }), false);
  assert.equal(canRetryMarketReview("FAILED", null), true);
  assert.equal(hasSavedV2MarketReviewCorrections({ ...attemptFourOcrFixture, status: "CORRECTED", corrections: [{ reviewed: false }] }), false);
});

function uploadReference(overrides: Partial<MarketReviewBlobReference> = {}): MarketReviewBlobReference {
  const descriptor = {
    upload_id: "22222222-2222-4222-8222-222222222222",
    session_date: "2026-08-14",
    filename: "MarketSurge.pdf",
    content_type: "application/pdf",
    size_bytes: 15 * 1024 * 1024,
    sha256: "d".repeat(64),
    ...overrides
  };
  const pathname = marketReviewBlobPathname(descriptor);
  return {
    ...descriptor,
    blob_url: `https://test-store.private.blob.vercel-storage.com/${pathname}`,
    blob_pathname: pathname,
    blob_content_type: "application/pdf"
  };
}

test("direct-upload metadata binds session, UUID, PDF hash, and private Blob path", () => {
  const reference = uploadReference();
  assert.deepEqual(parseMarketReviewUploadClientPayload(JSON.stringify(reference)), {
    upload_id: reference.upload_id,
    session_date: reference.session_date,
    filename: reference.filename,
    content_type: reference.content_type,
    size_bytes: reference.size_bytes,
    sha256: reference.sha256
  });
  assert.deepEqual(validateMarketReviewBlobReference(reference, "2026-08-14"), reference);
  assert.throws(
    () => validateMarketReviewBlobReference({ ...reference, blob_pathname: "market-review/source/other.pdf" }),
    (error: unknown) => error instanceof MarketReviewValidationError && error.code === "MARKETSURGE_BLOB_PATH_MISMATCH"
  );
  assert.throws(
    () => validateMarketReviewBlobReference({ ...reference, blob_url: `https://public.example/${reference.blob_pathname}` }),
    (error: unknown) => error instanceof MarketReviewValidationError && error.code === "MARKETSURGE_BLOB_URL_INVALID"
  );
});

test("large direct upload creates a small reference-only review request", () => {
  const payload = buildMarketReviewCreatePayload(uploadReference());
  const serialized = JSON.stringify(payload);
  assert(serialized.length < 2_000);
  assert(!serialized.includes("content_base64"));
  assert(!serialized.includes("%PDF-"));
  assert.equal(payload.marketsurge_pdf.size_bytes, 15 * 1024 * 1024);
  assert(!("account_scope" in payload));
  assert(!("consumer" in payload));
  assert.doesNotThrow(() => requireMarketReviewCreateJson("application/json; charset=utf-8"));
  assert.throws(
    () => requireMarketReviewCreateJson("multipart/form-data; boundary=large-pdf"),
    (error: unknown) => error instanceof MarketReviewValidationError && error.code === "MARKETSURGE_DIRECT_UPLOAD_REQUIRED"
  );
});

test("server re-reads private Blob and verifies metadata, PDF bytes, size, pages, and SHA-256", async () => {
  const data = Buffer.from("%PDF-1.7\nbody");
  const reference = uploadReference({ size_bytes: data.length, sha256: sha256(data) });
  const getBlob = (async () => ({
    statusCode: 200,
    stream: new Response(data).body!,
    headers: new Headers(),
    blob: {
      url: reference.blob_url,
      downloadUrl: `${reference.blob_url}?download=1`,
      pathname: reference.blob_pathname,
      contentType: "application/pdf",
      contentDisposition: "attachment",
      cacheControl: "public, max-age=60",
      etag: "test-etag",
      size: data.length,
      uploadedAt: new Date("2026-08-14T21:00:00Z")
    }
  })) as unknown as Parameters<typeof readAndValidateMarketReviewBlob>[1];
  const result = await readAndValidateMarketReviewBlob(reference, getBlob, async () => 6);
  assert.equal(result.pageCount, 6);
  assert.equal(result.sha256, sha256(data));
  await assert.rejects(
    readAndValidateMarketReviewBlob({ ...reference, sha256: "e".repeat(64), blob_url: reference.blob_url.replace(reference.sha256, "e".repeat(64)), blob_pathname: reference.blob_pathname.replace(reference.sha256, "e".repeat(64)) }, getBlob, async () => 6),
    (error: unknown) => error instanceof MarketReviewValidationError && ["MARKETSURGE_BLOB_METADATA_MISMATCH", "MARKETSURGE_BLOB_HASH_MISMATCH"].includes(error.code)
  );
});

test("market-review downloads encode Unicode filenames without invalid response-header bytes", async () => {
  const filename = "MarketSurge Fri Aug 14\u202f2026.pdf";
  const data = Buffer.from("%PDF-1.7\nbody");
  const digest = sha256(data);
  const disposition = marketReviewContentDisposition(filename);
  assert.equal(asciiDownloadFilename(filename), "MarketSurge_Fri_Aug_14_2026.pdf");
  assert.match(disposition, /^attachment; filename="MarketSurge_Fri_Aug_14_2026\.pdf"; filename\*=UTF-8''/);
  assert(disposition.includes("%E2%80%AF"));
  assert([...disposition].every((character) => character.charCodeAt(0) <= 0x7f));
  assert.doesNotThrow(() => new Headers({ "Content-Disposition": disposition }));

  const response = buildMarketReviewDownloadResponse(new Uint8Array(data), {
    filename,
    contentType: marketReviewSourceMediaType("marketsurge_pdf", "application/pdf"),
    sizeBytes: data.length,
    sha256: digest,
    disposition: "inline"
  });
  assert.match(response.headers.get("content-disposition") || "", /^inline; filename="MarketSurge_Fri_Aug_14_2026\.pdf";/);
  assert.equal(response.headers.get("x-content-sha256"), digest);
  assert.equal(sha256(Buffer.from(await response.arrayBuffer())), digest);
});

test("market-review download headers sanitize non-ASCII, path traversal, and header injection", () => {
  const unicode = marketReviewContentDisposition("../../r\u00e9sum\u00e9 \u5e02\u5834.pdf");
  assert.match(unicode, /^attachment; filename="resume.pdf"; filename\*=UTF-8''/);
  assert(unicode.includes("r%C3%A9sum%C3%A9%20%E5%B8%82%E5%A0%B4.pdf"));
  assert([...unicode].every((character) => character.charCodeAt(0) <= 0x7f));

  const injected = marketReviewContentDisposition("report.pdf\r\nX-Injected: yes.pdf");
  assert(!injected.includes("\r"));
  assert(!injected.includes("\n"));
  assert.doesNotThrow(() => new Headers({ "Content-Disposition": injected }));
  const headers = new Headers({ "Content-Disposition": injected });
  assert.equal(headers.get("x-injected"), null);
  assert.throws(
    () => buildMarketReviewDownloadResponse("x", {
      filename: "x.pdf",
      contentType: "application/pdf\r\nX-Injected: yes",
      sizeBytes: 1,
      sha256: "a".repeat(64)
    }),
    (error: unknown) => error instanceof MarketReviewValidationError && error.code === "DOWNLOAD_MEDIA_TYPE_INVALID"
  );
});

test("market-review download media types are fixed by source or artifact kind", () => {
  assert.equal(marketReviewSourceMediaType("snapshot_markdown", "text/markdown"), "text/markdown; charset=utf-8");
  assert.equal(marketReviewArtifactMediaType("json", "application/json"), "application/json");
  assert.throws(
    () => marketReviewArtifactMediaType("pdf", "text/plain\r\nX-Injected: yes"),
    (error: unknown) => error instanceof MarketReviewValidationError && error.code === "RESULT_ARTIFACT_METADATA_INVALID"
  );
});

test("market-review session dates stay canonical for worker dispatch", () => {
  assert.equal(run({ session_date: "2026-08-14" }).session_date, "2026-08-14");
  assert.equal(buildMarketReviewWorkerDispatch(run(), "https://dashboard.example").session_date, "2026-08-14");
  assert.throws(
    () => validateMarketReviewBlobReference(uploadReference({ session_date: "Fri Aug 14" })),
    (error: unknown) => error instanceof MarketReviewValidationError && error.code === "SESSION_DATE_INVALID"
  );
});

test("database DATE values remain canonical in the worker contract", () => {
  assert.equal(serializeMarketReviewSessionDate("2026-08-14"), "2026-08-14");
  assert.equal(serializeMarketReviewSessionDate("2026-08-14T00:00:00.000Z"), "2026-08-14");
  assert.equal(serializeMarketReviewSessionDate(new Date("2026-08-14T00:00:00.000Z")), "2026-08-14");
  assert.equal(
    serializeMarketReviewSessionDate("Fri Aug 14 2026 00:00:00 GMT+0000 (Coordinated Universal Time)"),
    "2026-08-14"
  );
});

test("OCR corrections require explicit v2-reviewed pages and reject navigation tokens", () => {
  assert.deepEqual(validateMarketReviewOcrCorrections([
    { pdf_page: 2, label: "Recent Breakouts", tickers: ["fet", "P", "FET"], reviewed: true }
  ]), [
    { pdf_page: 2, label: "Recent Breakouts", tickers: ["FET", "P"], reviewed: true }
  ]);
  for (const corrections of [
    [{ pdf_page: 2, label: "Recent Breakouts", tickers: ["FET"], reviewed: false }],
    [{ pdf_page: 2, label: "Recent Breakouts", tickers: ["FAVORITES"], reviewed: true }],
    [
      { pdf_page: 2, label: "Recent Breakouts", tickers: ["FET"], reviewed: true },
      { pdf_page: 2, label: "Recent Breakouts", tickers: ["P"], reviewed: true }
    ]
  ]) {
    assert.throws(
      () => validateMarketReviewOcrCorrections(corrections),
      (error: unknown) => error instanceof MarketReviewValidationError && error.code === "OCR_CORRECTIONS_INVALID"
    );
  }
});

function correctionSource(data: Buffer): SourceRecord {
  return {
    kind: "ocr_corrections_json",
    filename: "ocr-corrections.json",
    mediaType: "application/json",
    sha256: sha256(data),
    sizeBytes: data.length,
    data,
    storageUrl: null,
    storagePathname: null,
    expiresAt: "2026-08-16T00:00:00.000Z"
  };
}

test("legacy v1 OCR corrections are omitted so retry reruns OCR from the frozen PDF", () => {
  const data = Buffer.from(JSON.stringify({
    expected_version: 0,
    corrections: [{ pdf_page: 1, label: "Near Pivot", tickers: ["FAVORITES"] }]
  }));
  const source = correctionSource(data);
  const correctedRun = run({
    status: "FAILED",
    attempt: 4,
    ocr: { status: "CORRECTED", version: 1, correction_sha256: source.sha256 }
  });
  assert.equal(dispatchableMarketReviewOcrCorrections(correctedRun, source), null);
});

test("reviewed v2 OCR corrections remain dispatchable with exact hash and version correlation", () => {
  const data = Buffer.from(JSON.stringify({
    schema_version: "marketsurge_ocr_corrections_v2",
    expected_version: 1,
    corrections: [
      { pdf_page: 2, label: "Recent Breakouts", tickers: ["FET", "P"], reviewed: true }
    ]
  }));
  const source = correctionSource(data);
  const correctedRun = run({
    status: "FAILED",
    attempt: 4,
    ocr: {
      status: "CORRECTED",
      schema_version: "marketsurge_ocr_v2",
      version: 2,
      correction_sha256: source.sha256
    }
  });
  assert.equal(dispatchableMarketReviewOcrCorrections(correctedRun, source), source);
  assert.equal(
    dispatchableMarketReviewOcrCorrections(
      { ...correctedRun, ocr: { ...correctedRun.ocr, correction_sha256: "f".repeat(64) } },
      source
    ),
    null
  );
});

test("orphan Blob cleanup keeps active references and waits 24 hours", () => {
  const now = Date.parse("2026-08-16T00:00:00Z");
  const oldOrphan = "market-review/source/2026-08-14/orphan/file.pdf";
  const active = "market-review/source/2026-08-14/active/file.pdf";
  assert.deepEqual(selectExpiredOrphanMarketReviewBlobs([
    { pathname: oldOrphan, uploadedAt: new Date("2026-08-14T23:59:59Z") },
    { pathname: active, uploadedAt: new Date("2026-08-14T00:00:00Z") },
    { pathname: "market-review/source/2026-08-15/new/file.pdf", uploadedAt: new Date("2026-08-15T12:00:00Z") }
  ], new Set([active]), now), [oldOrphan]);
});

test("signed callback tokens bind run, session, attempt, hashes, and expiry", { concurrency: false }, () => {
  const previous = process.env.MARKET_REVIEW_TOKEN_SECRET;
  process.env.MARKET_REVIEW_TOKEN_SECRET = "test-market-review-token-secret";
  try {
    const token = signMarketReviewToken({
      scope: "callback",
      review_run_id: run().run_id,
      session_date: run().session_date,
      attempt: 1,
      source_hashes: hashes,
      exp: 2_000_000_000
    });
    const claims = verifyMarketReviewToken(token, {
      scope: "callback",
      review_run_id: run().run_id,
      session_date: run().session_date,
      attempt: 1,
      source_hashes: hashes
    }, 1_900_000_000_000);
    assert.equal(claims.attempt, 1);
    assert.throws(
      () => verifyMarketReviewToken(token, { attempt: 2 }, 1_900_000_000_000),
      (error: unknown) => error instanceof MarketReviewValidationError && error.code === "TOKEN_CORRELATION_MISMATCH"
    );
    assert.throws(
      () => verifyMarketReviewToken(token, {}, 2_100_000_000_000),
      (error: unknown) => error instanceof MarketReviewValidationError && error.code === "TOKEN_EXPIRED"
    );
  } finally {
    if (previous === undefined) delete process.env.MARKET_REVIEW_TOKEN_SECRET;
    else process.env.MARKET_REVIEW_TOKEN_SECRET = previous;
  }
});

test("worker shared secret is presented outside public dispatch inputs", { concurrency: false }, () => {
  const previous = process.env.DASHBOARD_WORKER_SECRET;
  process.env.DASHBOARD_WORKER_SECRET = "github-actions-secret";
  try {
    assert.doesNotThrow(() => verifyDashboardWorkerSecret("github-actions-secret"));
    assert.throws(
      () => verifyDashboardWorkerSecret("wrong"),
      (error: unknown) => error instanceof MarketReviewValidationError && error.code === "WORKER_AUTH_INVALID"
    );
  } finally {
    if (previous === undefined) delete process.env.DASHBOARD_WORKER_SECRET;
    else process.env.DASHBOARD_WORKER_SECRET = previous;
  }

  const inputs = buildMarketReviewWorkerDispatch(run(), "https://dashboard.example");
  assert.deepEqual(Object.keys(inputs).sort(), [
    "attempt",
    "market_gauge_json_sha256",
    "marketsurge_pdf_sha256",
    "review_run_id",
    "session_date",
    "snapshot_json_sha256",
    "snapshot_markdown_sha256",
    "worker_callback_url",
    "worker_input_url"
  ]);
  const serialized = JSON.stringify(inputs);
  assert(!serialized.includes("token"));
  assert(!serialized.includes("secret"));
  assert(!serialized.includes("account_scope"));
  assert(!serialized.includes("consumer"));
  assert(!serialized.includes("chart_manifest"));
});

test("GitHub workflow defaults to daily-review.yml and preserves the environment override", { concurrency: false }, () => {
  const previous = process.env.MARKET_REVIEW_GITHUB_WORKFLOW;
  try {
    delete process.env.MARKET_REVIEW_GITHUB_WORKFLOW;
    assert.equal(getMarketReviewGithubConfig().workflow, "daily-review.yml");
    process.env.MARKET_REVIEW_GITHUB_WORKFLOW = "custom-review.yml";
    assert.equal(getMarketReviewGithubConfig().workflow, "custom-review.yml");
  } finally {
    if (previous === undefined) delete process.env.MARKET_REVIEW_GITHUB_WORKFLOW;
    else process.env.MARKET_REVIEW_GITHUB_WORKFLOW = previous;
  }
});

test("callbacks require exact run, date, attempt, and all four source hashes", () => {
  assert.doesNotThrow(() => validateCallbackCorrelation(run(), callback("RUNNING")));
  assert.throws(
    () => validateCallbackCorrelation(run(), callback("RUNNING", { session_date: "2026-08-13" })),
    (error: unknown) => error instanceof MarketReviewValidationError && error.code === "CALLBACK_CORRELATION_MISMATCH"
  );
  assert.throws(
    () => validateCallbackCorrelation(run(), callback("RUNNING", { source_hashes: { ...hashes, market_gauge_json_sha256: "e".repeat(64) } })),
    (error: unknown) => error instanceof MarketReviewValidationError && error.code === "CALLBACK_SOURCE_HASH_MISMATCH"
  );
});

test("worker input rejects delayed attempts and any source hash mismatch", () => {
  assert.doesNotThrow(() => validateWorkerInputCorrelation(run(), { attempt: 1, source_hashes: hashes }));
  assert.throws(
    () => validateWorkerInputCorrelation(run({ attempt: 2 }), { attempt: 1, source_hashes: hashes }),
    (error: unknown) => error instanceof MarketReviewValidationError && error.code === "WORKER_ATTEMPT_MISMATCH"
  );
  assert.throws(
    () => validateWorkerInputCorrelation(run(), {
      attempt: 1,
      source_hashes: { ...hashes, snapshot_json_sha256: "d".repeat(64) }
    }),
    (error: unknown) => error instanceof MarketReviewValidationError && error.code === "WORKER_SOURCE_HASH_MISMATCH"
  );
});

test("callback lifecycle is idempotent and delivery failure cannot rebuild a completed report", () => {
  const running = deriveCallbackUpdate(run(), callback("RUNNING"));
  assert.equal(running.status, "RUNNING");
  const duplicate = deriveCallbackUpdate(run({ status: "COMPLETED", delivery_status: "SENT" }), callback("RUNNING"), true);
  assert.equal(duplicate.status, "COMPLETED");
  assert.equal(duplicate.deliveryStatus, "SENT");

  const deliveryFailure = deriveCallbackUpdate(
    run({ status: "COMPLETED", delivery_status: "PENDING" }),
    callback("DELIVERY_STATUS", { delivery: { status: "FAILED", error: "SMTP unavailable" } })
  );
  assert.equal(deliveryFailure.status, "COMPLETED");
  assert.equal(deliveryFailure.deliveryStatus, "FAILED");
  assert.equal(deliveryFailure.deleteSources, false);
  assert.throws(
    () => deriveCallbackUpdate(run({ status: "RUNNING" }), callback("DELIVERY_STATUS", { delivery: { status: "SENT" } })),
    (error: unknown) => error instanceof MarketReviewValidationError && error.code === "DELIVERY_STATUS_NOT_ALLOWED"
  );
  assert.throws(() => assertMarketReviewTransition("COMPLETED", "QUEUED"));
  assert.throws(
    () => deriveCallbackUpdate(run({ status: "COMPLETED" }), callback("RESULTS_REGISTERED", {
      audit: { status: "PASS" },
      artifacts: [
        { kind: "pdf", filename: "x.pdf", media_type: "application/pdf", sha256: "a".repeat(64), size_bytes: 1, content_base64: "eA==" },
        { kind: "markdown", filename: "x.md", media_type: "text/markdown", sha256: "b".repeat(64), size_bytes: 1, content_base64: "eA==" },
        { kind: "json", filename: "x.json", media_type: "application/json", sha256: "c".repeat(64), size_bytes: 1, content_base64: "eA==" }
      ]
    })),
    (error: unknown) => error instanceof MarketReviewValidationError && error.code === "STATUS_TRANSITION_INVALID"
  );
});

test("strict completed callback requires and verifies PDF, Markdown, and JSON artifacts", () => {
  const pdf = Buffer.from("%PDF-1.7\n%%EOF");
  const markdown = Buffer.from("# Review\n");
  const packet = Buffer.from('{"session_date":"2026-08-14"}\n');
  const artifacts = [
    { kind: "pdf" as const, filename: "review.pdf", media_type: "application/pdf", sha256: sha256(pdf), size_bytes: pdf.length, content_base64: pdf.toString("base64") },
    { kind: "markdown" as const, filename: "review.md", media_type: "text/markdown", sha256: sha256(markdown), size_bytes: markdown.length, content_base64: markdown.toString("base64") },
    { kind: "json" as const, filename: "review.json", media_type: "application/json", sha256: sha256(packet), size_bytes: packet.length, content_base64: packet.toString("base64") }
  ];
  const payload = callback("RESULTS_REGISTERED", { audit: { status: "PASS", packet_sha256: sha256(packet) }, artifacts });
  assert.doesNotThrow(() => validateCallbackCorrelation(run({ status: "RUNNING" }), payload));
  assert.deepEqual(decodeAndValidateResultArtifacts(artifacts).map((item) => item.kind), ["pdf", "markdown", "json"]);
  assert.throws(
    () => validateCallbackCorrelation(run({ status: "RUNNING" }), callback("RESULTS_REGISTERED", {
      audit: { status: "PASS", packet_sha256: "f".repeat(64) }, artifacts
    })),
    (error: unknown) => error instanceof MarketReviewValidationError && error.code === "STRICT_AUDIT_PACKET_HASH_MISMATCH"
  );
  assert.throws(
    () => decodeAndValidateResultArtifacts([{ ...artifacts[0], sha256: "f".repeat(64) }, artifacts[1], artifacts[2]]),
    (error: unknown) => error instanceof MarketReviewValidationError && error.code === "RESULT_ARTIFACT_HASH_MISMATCH"
  );
});

test("source deletion is immediate after completion and bounded to 24-hour expiry otherwise", () => {
  assert.equal(shouldDeleteMarketReviewSources(run({ status: "COMPLETED" }), Date.parse("2026-08-15T01:00:00Z")), true);
  assert.equal(shouldDeleteMarketReviewSources(run({ status: "FAILED", source_expires_at: "2026-08-16T00:00:00Z" }), Date.parse("2026-08-15T23:59:59Z")), false);
  assert.equal(shouldDeleteMarketReviewSources(run({ status: "FAILED", source_expires_at: "2026-08-16T00:00:00Z" }), Date.parse("2026-08-16T00:00:00Z")), true);
  assert.equal(shouldDeleteMarketReviewSources(run({ status: "COMPLETED", source_deleted_at: "2026-08-15T00:30:00Z" }), Date.parse("2026-08-16T00:00:00Z")), false);
});
