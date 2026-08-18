import type { PoolClient } from "pg";
import { del } from "@vercel/blob";
import { getPool } from "./store";
import {
  MARKET_REVIEW_SCHEMA_VERSION,
  MarketReviewValidationError,
  assertMarketReviewTransition,
  canSaveMarketReviewOcrCorrections,
  callbackStatus,
  decodeAndValidateResultArtifacts,
  deriveCallbackUpdate,
  requireSessionDate,
  shouldDeleteMarketReviewSources,
  type MarketReviewArtifactKind,
  type MarketReviewCallbackPayload,
  type MarketReviewDeliveryStatus,
  type MarketReviewRun,
  type MarketReviewSourceHashes,
  type MarketReviewSourceKind,
  type MarketReviewStatus
} from "./market-review-contract";

export const REQUIRED_RETRY_SOURCE_KINDS = [
  "marketsurge_pdf", "snapshot_json", "snapshot_markdown", "market_gauge_json"
] as const;

type CreateRunInput = {
  runId: string;
  sessionDate: string;
  sourceHashes: MarketReviewSourceHashes;
  pdfFilename: string;
  pdfSizeBytes: number;
  pdfPageCount: number;
  sourceExpiresAt: string;
  githubRepository: string;
  githubWorkflow: string;
  githubRef: string;
  sources: Array<{
    kind: MarketReviewSourceKind;
    filename: string;
    mediaType: string;
    sha256: string;
    data?: Buffer;
    blob?: { url: string; pathname: string };
    sizeBytes?: number;
  }>;
};

export type SourceRecord = {
  kind: MarketReviewSourceKind;
  filename: string;
  mediaType: string;
  sha256: string;
  sizeBytes: number;
  data: Buffer | null;
  storageUrl: string | null;
  storagePathname: string | null;
  expiresAt: string;
};

type ArtifactRecord = {
  kind: MarketReviewArtifactKind;
  filename: string;
  mediaType: string;
  sha256: string;
  sizeBytes: number;
  data: Buffer;
};

type RunRow = Record<string, unknown>;

let schemaPromise: Promise<void> | null = null;

function database() {
  const db = getPool();
  if (!db) throw new Error("DATABASE_URL is required for Campus Fund market reviews.");
  return db;
}

export function ensureMarketReviewSchema() {
  if (!schemaPromise) {
    schemaPromise = (async () => {
      const db = database();
      await db.query(`
        create table if not exists market_review_runs (
          run_id uuid primary key,
          schema_version text not null,
          session_date date not null,
          status text not null,
          attempt integer not null default 1,
          marketsurge_pdf_sha256 text not null,
          snapshot_json_sha256 text not null,
          snapshot_markdown_sha256 text not null,
          market_gauge_json_sha256 text,
          marketsurge_pdf_filename text not null,
          marketsurge_pdf_size_bytes integer not null,
          marketsurge_pdf_page_count integer not null,
          source_expires_at timestamptz not null,
          source_deleted_at timestamptz,
          github_repository text not null,
          github_workflow text not null,
          github_ref text not null,
          github_run_id text,
          github_run_attempt integer,
          github_workflow_url text,
          ocr jsonb,
          error jsonb,
          delivery_status text not null default 'NOT_REQUESTED',
          delivery_error text,
          created_at timestamptz not null default now(),
          updated_at timestamptz not null default now(),
          completed_at timestamptz,
          check (status in ('QUEUED','RUNNING','NEEDS_REVIEW','FAILED','COMPLETED')),
          check (delivery_status in ('NOT_REQUESTED','PENDING','SENDING','SENT','FAILED'))
        )
      `);
      await db.query("alter table market_review_runs add column if not exists market_gauge_json_sha256 text");
      await db.query(`
        create table if not exists market_review_sources (
          run_id uuid not null references market_review_runs(run_id) on delete cascade,
          kind text not null,
          filename text not null,
          media_type text not null,
          sha256 text not null,
          size_bytes integer not null,
          data bytea not null,
          expires_at timestamptz not null,
          created_at timestamptz not null default now(),
          primary key (run_id, kind)
        )
      `);
      await db.query("alter table market_review_sources add column if not exists storage_url text");
      await db.query("alter table market_review_sources add column if not exists storage_pathname text");
      await db.query("alter table market_review_sources alter column data drop not null");
      await db.query(`
        create table if not exists market_review_artifacts (
          run_id uuid not null references market_review_runs(run_id) on delete cascade,
          kind text not null,
          filename text not null,
          media_type text not null,
          sha256 text not null,
          size_bytes integer not null,
          data bytea not null,
          created_at timestamptz not null default now(),
          primary key (run_id, kind)
        )
      `);
      await db.query(`
        create table if not exists market_review_events (
          event_id text primary key,
          run_id uuid not null references market_review_runs(run_id) on delete cascade,
          event_type text not null,
          status text not null,
          attempt integer not null,
          payload jsonb not null default '{}'::jsonb,
          created_at timestamptz not null default now()
        )
      `);
      await db.query("create index if not exists market_review_runs_session_idx on market_review_runs(session_date desc, created_at desc)");
      await db.query("create index if not exists market_review_events_run_idx on market_review_events(run_id, created_at)");
    })().catch((error) => {
      schemaPromise = null;
      throw error;
    });
  }
  return schemaPromise;
}

function iso(value: unknown) {
  return value ? new Date(String(value)).toISOString() : null;
}

function parseJson<T>(value: unknown, fallback: T): T {
  if (value && typeof value === "object") return value as T;
  if (typeof value === "string") {
    try {
      return JSON.parse(value) as T;
    } catch {
      return fallback;
    }
  }
  return fallback;
}

export function serializeMarketReviewSessionDate(value: unknown) {
  const text = typeof value === "string" ? value.trim() : "";
  const canonical = text.match(/^(\d{4}-\d{2}-\d{2})(?:$|[T\s])/);
  if (canonical) return requireSessionDate(canonical[1]);

  const date = value instanceof Date ? value : new Date(String(value));
  if (Number.isNaN(date.getTime())) {
    throw new MarketReviewValidationError("SESSION_DATE_INVALID", "Stored market-review session date is invalid.");
  }
  return requireSessionDate(
    `${date.getUTCFullYear()}-${String(date.getUTCMonth() + 1).padStart(2, "0")}-${String(date.getUTCDate()).padStart(2, "0")}`
  );
}

function rowToRun(row: RunRow, artifacts: MarketReviewRun["artifacts"] = []): MarketReviewRun {
  return {
    run_id: String(row.run_id),
    schema_version: MARKET_REVIEW_SCHEMA_VERSION,
    session_date: serializeMarketReviewSessionDate(row.session_date),
    status: String(row.status) as MarketReviewStatus,
    attempt: Number(row.attempt),
    source_hashes: {
      marketsurge_pdf_sha256: String(row.marketsurge_pdf_sha256),
      snapshot_json_sha256: String(row.snapshot_json_sha256),
      snapshot_markdown_sha256: String(row.snapshot_markdown_sha256),
      market_gauge_json_sha256: row.market_gauge_json_sha256 ? String(row.market_gauge_json_sha256) : ""
    },
    marketsurge_pdf_filename: String(row.marketsurge_pdf_filename),
    marketsurge_pdf_size_bytes: Number(row.marketsurge_pdf_size_bytes),
    marketsurge_pdf_page_count: Number(row.marketsurge_pdf_page_count),
    source_expires_at: iso(row.source_expires_at) || "",
    source_deleted_at: iso(row.source_deleted_at),
    github_repository: String(row.github_repository),
    github_workflow: String(row.github_workflow),
    github_ref: String(row.github_ref),
    github_run_id: row.github_run_id ? String(row.github_run_id) : null,
    github_run_attempt: row.github_run_attempt === null || row.github_run_attempt === undefined ? null : Number(row.github_run_attempt),
    github_workflow_url: row.github_workflow_url ? String(row.github_workflow_url) : null,
    ocr: parseJson<Record<string, unknown> | null>(row.ocr, null),
    error: parseJson<MarketReviewRun["error"]>(row.error, null),
    delivery_status: String(row.delivery_status) as MarketReviewDeliveryStatus,
    delivery_error: row.delivery_error ? String(row.delivery_error) : null,
    artifacts,
    created_at: iso(row.created_at) || "",
    updated_at: iso(row.updated_at) || "",
    completed_at: iso(row.completed_at)
  };
}

async function artifactSummaries(client: PoolClient, runIds: string[]) {
  const output = new Map<string, MarketReviewRun["artifacts"]>();
  if (!runIds.length) return output;
  const result = await client.query(
    "select run_id, kind, filename, media_type, sha256, size_bytes from market_review_artifacts where run_id = any($1::uuid[]) order by kind",
    [runIds]
  );
  for (const row of result.rows) {
    const runId = String(row.run_id);
    const values = output.get(runId) || [];
    const kind = String(row.kind) as MarketReviewArtifactKind;
    values.push({
      kind,
      filename: String(row.filename),
      media_type: String(row.media_type),
      sha256: String(row.sha256),
      size_bytes: Number(row.size_bytes),
      download_url: `/api/journal/branden/market-review/runs/${runId}/artifacts/${kind}`
    });
    output.set(runId, values);
  }
  return output;
}

async function withTransaction<T>(task: (client: PoolClient) => Promise<T>) {
  const client = await database().connect();
  try {
    await client.query("begin");
    const result = await task(client);
    await client.query("commit");
    return result;
  } catch (error) {
    await client.query("rollback");
    throw error;
  } finally {
    client.release();
  }
}

async function lockedRun(client: PoolClient, runId: string) {
  const result = await client.query("select * from market_review_runs where run_id = $1 for update", [runId]);
  if (!result.rowCount) throw new MarketReviewValidationError("REVIEW_RUN_NOT_FOUND", "The requested market review run does not exist.");
  const artifacts = await artifactSummaries(client, [runId]);
  return rowToRun(result.rows[0], artifacts.get(runId) || []);
}

async function deleteSourcesForRun(client: PoolClient, runId: string) {
  const stored = await client.query(
    "select storage_url from market_review_sources where run_id = $1 and storage_url is not null",
    [runId]
  );
  const blobUrls = stored.rows.map((row) => String(row.storage_url)).filter(Boolean);
  if (blobUrls.length) await del(blobUrls);
  await client.query("delete from market_review_sources where run_id = $1", [runId]);
}

export async function createMarketReviewRun(input: CreateRunInput) {
  await ensureMarketReviewSchema();
  return withTransaction(async (client) => {
    await client.query(
      `insert into market_review_runs (
        run_id, schema_version, session_date, status, attempt,
        marketsurge_pdf_sha256, snapshot_json_sha256, snapshot_markdown_sha256, market_gauge_json_sha256,
        marketsurge_pdf_filename, marketsurge_pdf_size_bytes, marketsurge_pdf_page_count,
        source_expires_at, github_repository, github_workflow, github_ref
      ) values ($1,$2,$3,'QUEUED',1,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14)`,
      [
        input.runId,
        MARKET_REVIEW_SCHEMA_VERSION,
        input.sessionDate,
        input.sourceHashes.marketsurge_pdf_sha256,
        input.sourceHashes.snapshot_json_sha256,
        input.sourceHashes.snapshot_markdown_sha256,
        input.sourceHashes.market_gauge_json_sha256,
        input.pdfFilename,
        input.pdfSizeBytes,
        input.pdfPageCount,
        input.sourceExpiresAt,
        input.githubRepository,
        input.githubWorkflow,
        input.githubRef
      ]
    );
    for (const source of input.sources) {
      if ((!source.data && !source.blob) || (source.data && source.blob)) {
        throw new MarketReviewValidationError("SOURCE_STORAGE_INVALID", "Each review source must use exactly one storage mechanism.");
      }
      const sourceSizeBytes = source.data?.length ?? source.sizeBytes;
      if (!Number.isInteger(sourceSizeBytes) || Number(sourceSizeBytes) < 1) {
        throw new MarketReviewValidationError("SOURCE_STORAGE_INVALID", "Each review source must have a positive byte size.");
      }
      await client.query(
        `insert into market_review_sources (
          run_id, kind, filename, media_type, sha256, size_bytes, data, expires_at, storage_url, storage_pathname
        ) values ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10)`,
        [
          input.runId,
          source.kind,
          source.filename,
          source.mediaType,
          source.sha256,
          sourceSizeBytes,
          source.data || null,
          input.sourceExpiresAt,
          source.blob?.url || null,
          source.blob?.pathname || null
        ]
      );
    }
    await client.query(
      `insert into market_review_events (event_id, run_id, event_type, status, attempt, payload)
       values ($1,$2,'CREATED','QUEUED',1,$3::jsonb)`,
      [`${input.runId}:created:1`, input.runId, JSON.stringify({ source_hashes: input.sourceHashes })]
    );
    return lockedRun(client, input.runId);
  });
}

export async function getMarketReviewRun(runId: string) {
  await ensureMarketReviewSchema();
  const client = await database().connect();
  try {
    const result = await client.query("select * from market_review_runs where run_id = $1", [runId]);
    if (!result.rowCount) return null;
    const artifacts = await artifactSummaries(client, [runId]);
    return rowToRun(result.rows[0], artifacts.get(runId) || []);
  } finally {
    client.release();
  }
}

export async function listMarketReviewRuns(limit = 25) {
  await ensureMarketReviewSchema();
  const client = await database().connect();
  try {
    const result = await client.query("select * from market_review_runs order by created_at desc limit $1", [Math.max(1, Math.min(limit, 100))]);
    const runIds = result.rows.map((row) => String(row.run_id));
    const artifacts = await artifactSummaries(client, runIds);
    return result.rows.map((row) => rowToRun(row, artifacts.get(String(row.run_id)) || []));
  } finally {
    client.release();
  }
}

export async function listMarketReviewSourceBlobPathnames() {
  await ensureMarketReviewSchema();
  const result = await database().query(
    "select storage_pathname from market_review_sources where storage_pathname is not null"
  );
  return new Set(result.rows.map((row) => String(row.storage_pathname)).filter(Boolean));
}

export async function getMarketReviewSource(runId: string, kind: MarketReviewSourceKind): Promise<SourceRecord | null> {
  await ensureMarketReviewSchema();
  const result = await database().query(
    `select kind, filename, media_type, sha256, size_bytes, data, expires_at, storage_url, storage_pathname
     from market_review_sources where run_id = $1 and kind = $2`,
    [runId, kind]
  );
  if (!result.rowCount) return null;
  const row = result.rows[0];
  return {
    kind: String(row.kind) as MarketReviewSourceKind,
    filename: String(row.filename),
    mediaType: String(row.media_type),
    sha256: String(row.sha256),
    sizeBytes: Number(row.size_bytes),
    data: row.data ? Buffer.from(row.data) : null,
    storageUrl: row.storage_url ? String(row.storage_url) : null,
    storagePathname: row.storage_pathname ? String(row.storage_pathname) : null,
    expiresAt: iso(row.expires_at) || ""
  };
}

export async function getMarketReviewArtifact(runId: string, kind: MarketReviewArtifactKind): Promise<ArtifactRecord | null> {
  await ensureMarketReviewSchema();
  const result = await database().query(
    "select kind, filename, media_type, sha256, size_bytes, data from market_review_artifacts where run_id = $1 and kind = $2",
    [runId, kind]
  );
  if (!result.rowCount) return null;
  const row = result.rows[0];
  return {
    kind: String(row.kind) as MarketReviewArtifactKind,
    filename: String(row.filename),
    mediaType: String(row.media_type),
    sha256: String(row.sha256),
    sizeBytes: Number(row.size_bytes),
    data: Buffer.from(row.data)
  };
}

export async function recordMarketReviewDispatch(runId: string, attempt: number) {
  await ensureMarketReviewSchema();
  await database().query(
    `insert into market_review_events (event_id, run_id, event_type, status, attempt, payload)
     values ($1,$2,'DISPATCHED','QUEUED',$3,'{}'::jsonb) on conflict (event_id) do nothing`,
    [`${runId}:dispatched:${attempt}`, runId, attempt]
  );
}

export async function failMarketReviewRun(runId: string, code: string, message: string, details?: unknown) {
  await ensureMarketReviewSchema();
  return withTransaction(async (client) => {
    const run = await lockedRun(client, runId);
    assertMarketReviewTransition(run.status, "FAILED");
    await client.query(
      "update market_review_runs set status = 'FAILED', error = $2::jsonb, updated_at = now() where run_id = $1",
      [runId, JSON.stringify({ code, message, details })]
    );
    await client.query(
      `insert into market_review_events (event_id, run_id, event_type, status, attempt, payload)
       values ($1,$2,'DISPATCH_FAILED','FAILED',$3,$4::jsonb) on conflict (event_id) do nothing`,
      [`${runId}:dispatch-failed:${run.attempt}`, runId, run.attempt, JSON.stringify({ code, message })]
    );
    return lockedRun(client, runId);
  });
}

export async function applyMarketReviewCallback(payload: MarketReviewCallbackPayload) {
  await ensureMarketReviewSchema();
  const decodedArtifacts = payload.event_type === "RESULTS_REGISTERED"
    ? decodeAndValidateResultArtifacts(payload.artifacts || [])
    : [];
  return withTransaction(async (client) => {
    const run = await lockedRun(client, payload.review_run_id);
    const duplicate = await client.query("select 1 from market_review_events where event_id = $1", [payload.event_id]);
    if (duplicate.rowCount) return { run, duplicate: true };

    const update = deriveCallbackUpdate(run, payload);
    const nextStatus = callbackStatus(payload.event_type);
    if (payload.github?.repository && payload.github.repository !== run.github_repository) {
      throw new MarketReviewValidationError("CALLBACK_REPOSITORY_MISMATCH", "Callback GitHub repository does not match the dispatched repository.");
    }

    if (payload.event_type === "RESULTS_REGISTERED") {
      for (const artifact of decodedArtifacts) {
        await client.query(
          `insert into market_review_artifacts (run_id, kind, filename, media_type, sha256, size_bytes, data)
           values ($1,$2,$3,$4,$5,$6,$7)
           on conflict (run_id, kind) do update set
             filename = excluded.filename, media_type = excluded.media_type, sha256 = excluded.sha256,
             size_bytes = excluded.size_bytes, data = excluded.data, created_at = now()`,
          [run.run_id, artifact.kind, artifact.filename, artifact.media_type, artifact.sha256, artifact.size_bytes, artifact.data]
        );
      }
      await deleteSourcesForRun(client, run.run_id);
    }

    const error = payload.event_type === "FAILED" ? payload.error : null;
    const completedAt = payload.event_type === "RESULTS_REGISTERED" ? new Date().toISOString() : run.completed_at;
    await client.query(
      `update market_review_runs set
        status = $2,
        github_run_id = coalesce($3, github_run_id),
        github_run_attempt = coalesce($4, github_run_attempt),
        github_workflow_url = coalesce($5, github_workflow_url),
        ocr = coalesce($6::jsonb, ocr),
        error = $7::jsonb,
        delivery_status = $8,
        delivery_error = $9,
        source_deleted_at = case when $10::boolean then now() else source_deleted_at end,
        completed_at = $11,
        updated_at = now()
       where run_id = $1`,
      [
        run.run_id,
        nextStatus || run.status,
        payload.github?.workflow_run_id || null,
        payload.github?.workflow_run_attempt || null,
        payload.github?.workflow_url || null,
        payload.ocr ? JSON.stringify(payload.ocr) : null,
        JSON.stringify(error),
        update.deliveryStatus,
        update.deliveryError,
        payload.event_type === "RESULTS_REGISTERED",
        completedAt
      ]
    );
    await client.query(
      `insert into market_review_events (event_id, run_id, event_type, status, attempt, payload)
       values ($1,$2,$3,$4,$5,$6::jsonb)`,
      [payload.event_id, run.run_id, payload.event_type, nextStatus || run.status, run.attempt, JSON.stringify(payload)]
    );
    return { run: await lockedRun(client, run.run_id), duplicate: false };
  });
}

export async function queueMarketReviewRetry(runId: string) {
  await ensureMarketReviewSchema();
  return withTransaction(async (client) => {
    const run = await lockedRun(client, runId);
    if (run.status !== "FAILED" && run.status !== "NEEDS_REVIEW") {
      throw new MarketReviewValidationError("RETRY_NOT_ALLOWED", "Only failed or needs-review runs can be retried.");
    }
    if (run.source_deleted_at || new Date(run.source_expires_at).getTime() <= Date.now()) {
      throw new MarketReviewValidationError("RETRY_SOURCE_UNAVAILABLE", "The exact source packet has expired or was deleted; start a new review.");
    }
    const sourceCount = await client.query(
      "select count(*)::int as count from market_review_sources where run_id = $1 and kind = any($2::text[])",
      [runId, REQUIRED_RETRY_SOURCE_KINDS]
    );
    if (Number(sourceCount.rows[0]?.count) !== REQUIRED_RETRY_SOURCE_KINDS.length) {
      throw new MarketReviewValidationError("RETRY_SOURCE_INCOMPLETE", "The exact four-source packet is incomplete; start a new review.");
    }
    const nextAttempt = run.attempt + 1;
    await client.query(
      `update market_review_runs set status = 'QUEUED', attempt = $2, error = null,
       delivery_status = 'NOT_REQUESTED', delivery_error = null, github_run_id = null,
       github_run_attempt = null, github_workflow_url = null, updated_at = now()
       where run_id = $1`,
      [runId, nextAttempt]
    );
    await client.query(
      `insert into market_review_events (event_id, run_id, event_type, status, attempt, payload)
       values ($1,$2,'RETRY_QUEUED','QUEUED',$3,'{}'::jsonb)`,
      [`${runId}:retry:${nextAttempt}`, runId, nextAttempt]
    );
    return lockedRun(client, runId);
  });
}

export async function saveMarketReviewOcrCorrections(runId: string, input: { expectedVersion: number; corrections: unknown[]; data: Buffer; sha256: string }) {
  await ensureMarketReviewSchema();
  return withTransaction(async (client) => {
    const run = await lockedRun(client, runId);
    if (!canSaveMarketReviewOcrCorrections(run.status)) {
      throw new MarketReviewValidationError("OCR_CORRECTION_NOT_ALLOWED", "OCR corrections are accepted only while a run needs review or after an OCR-related failure.");
    }
    const storedVersion = Number(run.ocr?.version || 0);
    if (storedVersion !== input.expectedVersion) throw new MarketReviewValidationError("OCR_VERSION_CONFLICT", "OCR review changed; reload before submitting corrections.");
    await client.query(
      `insert into market_review_sources (run_id, kind, filename, media_type, sha256, size_bytes, data, expires_at)
       values ($1,'ocr_corrections_json','ocr-corrections.json','application/json',$2,$3,$4,$5)
       on conflict (run_id, kind) do update set sha256 = excluded.sha256, size_bytes = excluded.size_bytes,
       data = excluded.data, storage_url = null, storage_pathname = null,
       expires_at = excluded.expires_at, created_at = now()`,
      [runId, input.sha256, input.data.length, input.data, run.source_expires_at]
    );
    const ocr = { ...(run.ocr || {}), status: "CORRECTED", correction_sha256: input.sha256, corrections: input.corrections, version: storedVersion + 1 };
    await client.query("update market_review_runs set ocr = $2::jsonb, updated_at = now() where run_id = $1", [runId, JSON.stringify(ocr)]);
    await client.query(
      `insert into market_review_events (event_id, run_id, event_type, status, attempt, payload)
       values ($1,$2,'OCR_CORRECTED',$3,$4,$5::jsonb)`,
      [`${runId}:ocr-corrected:${storedVersion + 1}`, runId, run.status, run.attempt, JSON.stringify({ version: storedVersion + 1, sha256: input.sha256 })]
    );
    return lockedRun(client, runId);
  });
}

export async function cleanupMarketReviewSources(now = new Date()) {
  await ensureMarketReviewSchema();
  return withTransaction(async (client) => {
    const result = await client.query("select * from market_review_runs where source_deleted_at is null and (status = 'COMPLETED' or source_expires_at <= $1) for update", [now.toISOString()]);
    const eligible = result.rows.map((row) => rowToRun(row)).filter((run) => shouldDeleteMarketReviewSources(run, now.getTime()));
    for (const run of eligible) {
      await deleteSourcesForRun(client, run.run_id);
      await client.query("update market_review_runs set source_deleted_at = now(), updated_at = now() where run_id = $1", [run.run_id]);
      await client.query(
        `insert into market_review_events (event_id, run_id, event_type, status, attempt, payload)
         values ($1,$2,'SOURCE_CLEANED',$3,$4,'{}'::jsonb) on conflict (event_id) do nothing`,
        [`${run.run_id}:source-cleaned`, run.run_id, run.status, run.attempt]
      );
    }
    return { deletedRunCount: eligible.length, runIds: eligible.map((run) => run.run_id) };
  });
}
