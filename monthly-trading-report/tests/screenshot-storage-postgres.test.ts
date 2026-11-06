import assert from "node:assert/strict";
import { randomUUID } from "node:crypto";
import test from "node:test";
import { Client } from "pg";
import { readFile } from "node:fs/promises";
import { trade } from "./fixtures/trade-review-export";
import { copyImageToStorage } from "../lib/screenshot-storage";

const databaseUrl = process.env.BACKUP_TEST_DATABASE_URL;

test("PostgreSQL Storage cutover preserves live saves, history checkpoints, shared images and self-contained backups", {
  skip: databaseUrl ? false : "Requires isolated local PostgreSQL."
}, async () => {
  const url = new URL(databaseUrl!);
  assert.equal(url.hostname, "localhost");
  assert.match(url.pathname, /^\/trading_dashboard_backup_test(?:_[a-z0-9]+)?$/);
  const schema = `storage_test_${randomUUID().replaceAll("-", "")}`;
  const admin = new Client({ connectionString: url.toString(), ssl: false });
  await admin.connect();
  const previous = { ...process.env };
  const originalFetch = globalThis.fetch;
  const objects = new Map<string, Buffer>();
  let failReads = false;
  let failWrites = false;
  let pool: ReturnType<typeof import("../lib/store")["getPool"]> = null;
  globalThis.fetch = async (input, options) => {
    if (new URL(String(input)).pathname.includes("/bucket/")) return Response.json({ public: false });
    const key = new URL(String(input)).pathname.replace("/authenticated/", "/");
    if (options?.method === "POST") {
      if (failWrites) return new Response("synthetic failure", { status: 503 });
      if (objects.has(key)) return new Response("exists", { status: 409 });
      objects.set(key, Buffer.from(options.body as Uint8Array));
      return new Response("ok");
    }
    const bytes = objects.get(key);
    return bytes && !failReads ? new Response(new Uint8Array(bytes)) : new Response("missing", { status: 404 });
  };
  try {
    await admin.query(`create schema ${schema}`);
    url.searchParams.set("options", `-c search_path=${schema}`);
    process.env.DATABASE_URL = url.toString();
    process.env.SUPABASE_STORAGE_URL = "https://synthetic.supabase.co";
    process.env.SUPABASE_STORAGE_SERVICE_KEY = "test-only";
    const store = await import("../lib/store");
    pool = store.getPool();
    assert.ok(pool);
    // Rehearse the explicit migration against the original NOT NULL schema.
    await pool.query(`create table trade_screenshots (id text primary key, trade_id text not null,
      user_id text not null, file_name text not null default '', mime_type text not null,
      image_data bytea not null, created_at timestamptz not null default now());
      create table cam_journal_screenshots (id text primary key, entity_type text not null,
      entity_id text not null, file_name text not null default '', mime_type text not null,
      image_data bytea not null, created_at timestamptz not null default now())`);
    const migration = (await readFile("migrations/20261105-screenshot-storage.sql", "utf8")).replaceAll("public.", `${schema}.`);
    await pool.query(migration);
    await pool.query(migration); // safe to retry
    const original = (await store.upsertTrade(trade({ userId: "branden", screenshots: [] }))).trade;
    const bytes = Buffer.from([0, 1, 128, 255]);
    const legacy = await store.saveTradeScreenshot(original.id, "branden", "original.png", "image/png", bytes);
    // Copy existing bytes, retaining the DB copy until verification.
    const copied = await copyImageToStorage("trade_screenshots", legacy.id, "image/png", bytes);
    await pool.query("update trade_screenshots set storage_path=$2,image_sha256=$3,image_bytes=$4 where id=$1",
      [legacy.id, copied.storage_path, copied.image_sha256, copied.image_bytes]);
    assert.deepEqual((await store.getTradeScreenshot(legacy.id))?.imageData, bytes);
    await pool.query("update trade_screenshots set image_data=null where id=$1", [legacy.id]);
    assert.deepEqual((await store.getTradeScreenshot(legacy.id))?.imageData, bytes);
    process.env.SCREENSHOT_STORAGE_WRITES = "enabled";
    const shared = await store.saveCamJournalScreenshot("setup-strategy-example", "example-1", "example.png", "image/png", bytes);
    await store.saveCamJournalState({ trades: [{ id: "live-1" }] });
    await pool.query("insert into app_settings(key,value) values ('setup_checklist_templates',$1::jsonb)",
      [JSON.stringify({ versions: [{ shots: [shared.url] }] })]);

    // Existing old checkpoints survive both routine saves and the nightly job.
    await pool.query("insert into cam_journal_revisions(state,reason,created_at) values ('{}','recovery-pinned',now()-interval '200 days'), ('{}','migration-pinned',now()-interval '200 days'), ('{}','nightly',now()-interval '120 days')");
    await store.saveCamJournalState({ trades: [{ id: "live-2" }] });
    await store.saveCamJournalState({ trades: [{ id: "live-3" }] });
    assert.deepEqual((await store.getCamJournalState()).trades, [{ id: "live-3" }]);
    const saves = await pool.query("select state from cam_journal_revisions where reason='save'");
    assert.equal(saves.rows.length, 1);
    assert.deepEqual(saves.rows[0].state.trades, [{ id: "live-2" }]);
    assert.equal((await store.snapshotCamJournalState()).created, true);
    assert.equal((await store.snapshotCamJournalState()).created, false);
    const history = await pool.query("select reason,state from cam_journal_revisions");
    const nightlies = history.rows.filter(row => row.reason === "nightly");
    assert.equal(nightlies.length, 2);
    assert.ok(nightlies.some(row => row.state.trades?.[0]?.id === "live-3"));
    assert.ok(nightlies.some(row => Object.keys(row.state).length === 0));
    assert.equal(history.rows.filter(row => row.reason.endsWith("pinned")).length, 2);

    const download = await store.exportBrandenJournalBackup();
    const backup = await new Response(download.body).json();
    assert.equal(backup.version, 1);
    assert.deepEqual(Buffer.from(backup.tradeScreenshots[0].imageDataBase64, "base64"), bytes);
    assert.equal(backup.sharedScreenshots.length, 1);
    assert.deepEqual(Buffer.from(backup.sharedScreenshots[0].imageDataBase64, "base64"), bytes);
    await store.importBrandenJournalBackup(backup);
    assert.deepEqual((await store.getTradeScreenshot(legacy.id))?.imageData, bytes);
    assert.deepEqual((await store.getCamJournalScreenshot(shared.id))?.imageData, bytes);
    const persisted = await pool.query("select image_data,storage_path from trade_screenshots");
    assert.equal(persisted.rows[0].image_data, null);
    assert.ok(persisted.rows[0].storage_path);
    assert.deepEqual((await store.getCamJournalState()).trades, [{ id: "live-3" }]);

    // Failed uploads must not append broken references to the parent trade.
    const before = await pool.query("select screenshots from trade_logs where id=$1", [original.id]);
    failWrites = true;
    await assert.rejects(store.saveTradeScreenshot(original.id, "branden", "new.png", "image/png", Buffer.from("different")), /upload failed/);
    assert.deepEqual((await pool.query("select screenshots from trade_logs where id=$1", [original.id])).rows, before.rows);
    failWrites = false;
    failReads = true;
    const failedBackup = await store.exportBrandenJournalBackup();
    await assert.rejects(new Response(failedBackup.body).blob(), /Storage read failed/);
    failReads = false;
    const bad = structuredClone(backup);
    bad.sharedScreenshots[0].imageDataBase64 = Buffer.from("replacement").toString("base64");
    await assert.rejects(store.importBrandenJournalBackup(bad), /conflicts/);
    assert.deepEqual((await store.getTradeScreenshot(legacy.id))?.imageData, bytes);
    // Legacy files without the optional shared section remain importable.
    delete backup.sharedScreenshots;
    await store.importBrandenJournalBackup(backup);
    const idle = await admin.query("select count(*)::int as count from pg_stat_activity where datname=current_database() and state='idle in transaction'");
    assert.equal(idle.rows[0].count, 0);
  } finally {
    globalThis.fetch = originalFetch;
    if (pool) await pool.end();
    for (const key of ["DATABASE_URL", "SUPABASE_STORAGE_URL", "SUPABASE_STORAGE_SERVICE_KEY", "SCREENSHOT_STORAGE_WRITES"]) {
      if (previous[key] === undefined) delete process.env[key]; else process.env[key] = previous[key];
    }
    try { await admin.query(`drop schema if exists ${schema} cascade`); }
    finally { await admin.end(); }
  }
});
