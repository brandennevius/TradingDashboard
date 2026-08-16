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
  validateWorkerInputCorrelation,
  verifyMarketReviewToken,
  type MarketReviewCallbackPayload,
  type MarketReviewRun
} from "../lib/market-review-contract";
import { buildMarketReviewWorkerDispatch, verifyDashboardWorkerSecret } from "../lib/market-review-service";

const hashes = {
  marketsurge_pdf_sha256: "a".repeat(64),
  snapshot_json_sha256: "b".repeat(64),
  snapshot_markdown_sha256: "c".repeat(64)
};

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
    github_workflow: "dashboard-review.yml",
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

test("callbacks require exact run, date, attempt, and all three source hashes", () => {
  assert.doesNotThrow(() => validateCallbackCorrelation(run(), callback("RUNNING")));
  assert.throws(
    () => validateCallbackCorrelation(run(), callback("RUNNING", { session_date: "2026-08-13" })),
    (error: unknown) => error instanceof MarketReviewValidationError && error.code === "CALLBACK_CORRELATION_MISMATCH"
  );
  assert.throws(
    () => validateCallbackCorrelation(run(), callback("RUNNING", { source_hashes: { ...hashes, snapshot_markdown_sha256: "d".repeat(64) } })),
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
