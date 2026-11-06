import { mockR2 } from "./fixtures/mock-r2";
import assert from "node:assert/strict";
import { randomUUID, createHash } from "node:crypto";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Client, type QueryResult } from "pg";
import test from "node:test";
import { runCleanup } from "../scripts/supabase-cleanup";

const databaseUrl = process.env.BACKUP_TEST_DATABASE_URL;
test("operator refuses incomplete/corrupt copies, then prunes explicit history and releases verified bytes without deleting images", {
  skip: databaseUrl ? false : "Requires isolated local PostgreSQL."
}, async () => {
  const url = new URL(databaseUrl!);
  assert.equal(url.hostname, "localhost");
  assert.match(url.pathname, /^\/trading_dashboard_backup_test(?:_[a-z0-9]+)?$/);
  const name = `trading_dashboard_backup_test_${randomUUID().replaceAll("-", "").slice(0, 12)}`;
  const admin = new Client({ connectionString: url.toString(), ssl: false });
  await admin.connect();
  const previous = { ...process.env };
  const r2 = mockR2();
  const { objects } = r2;
  const directory = await mkdtemp(join(tmpdir(), "cleanup-test-"));
  let db: Client | undefined;
  try {
    await admin.query(`create database ${name}`);
    url.pathname = `/${name}`;
    process.env.DATABASE_URL = url.toString();
    process.env.CLEANUP_EXPECTED_HOST = "localhost";
    process.env.CLEANUP_EXPECTED_PROJECT_REF = "synthetic";
    db = new Client({ connectionString: url.toString(), ssl: false });
    await db.connect();
    await db.query(`create table trade_screenshots (id text primary key, mime_type text, image_data bytea not null);
      create table cam_journal_screenshots (id text primary key, mime_type text, image_data bytea not null);
      create table cam_journal_revisions (id bigserial primary key, state jsonb, reason text);
      create table app_settings (key text primary key, value jsonb);
      insert into cam_journal_revisions(state,reason) values ('{}','save'),('{}','save'),('{}','nightly'),('{}','recovery');
      insert into app_settings values ('cam_journal_state','{"trades":[{"id":"live"}]}');`);
    await db.query(await readFile("migrations/20261105-screenshot-storage.sql", "utf8"));
    for (const table of ["trade_screenshots", "cam_journal_screenshots"]) {
      await db.query(`insert into ${table}(id,mime_type,image_data) values ('image-1','image/png',$1)`, [Buffer.from(table)]);
    }
    const manifest = join(directory, "manifest.json");
    await assert.rejects(runCleanup("copy", manifest), /Mutations require/);
    process.env.CLEANUP_WRITES = "confirmed";
    const dump = join(directory, "synthetic-dump");
    const dumpBytes = Buffer.from("Synthetic backup evidence for operator gate test only");
    await writeFile(dump, dumpBytes);
    const attestation = join(directory, "backup.json");
    await writeFile(attestation, JSON.stringify({ host: "localhost", database: name, projectRef: "synthetic",
      dumpFile: dump, sha256: createHash("sha256").update(dumpBytes).digest("hex"),
      restoreVerifiedAt: new Date().toISOString(), secondCopyVerified: true }));
    process.env.CLEANUP_BACKUP_ATTESTATION = attestation;
    await runCleanup("copy", manifest);
    await runCleanup("copy", manifest); // resumable immutable copies
    assert.equal(objects.size, 2);
    await runCleanup("verify", manifest);
    const completeManifest = await readFile(manifest, "utf8");
    const incomplete = JSON.parse(completeManifest);
    incomplete.images.pop();
    await writeFile(manifest, JSON.stringify(incomplete));
    await assert.rejects(runCleanup("verify", manifest), /coverage changed/);
    await writeFile(manifest, completeManifest);
    await assert.rejects(runCleanup("release", manifest), /Pause writes/);
    process.env.CLEANUP_MAINTENANCE = "writes-and-jobs-paused";
    await assert.rejects(runCleanup("release", manifest), /CLEANUP_RELEASE_BATCH_SIZE/);
    process.env.CLEANUP_RELEASE_BATCH_SIZE = "1";
    const first = objects.entries().next().value!;
    objects.set(first[0], Buffer.from("corrupt"));
    await assert.rejects(runCleanup("release", manifest), /integrity/);
    assert.equal((await db.query("select count(*)::int as n from trade_screenshots where image_data is not null")).rows[0].n, 1);
    objects.set(first[0], first[1]);
    // The current-state check prevents pruning a stale candidate manifest.
    await db.query("update app_settings set value='{}' where key='cam_journal_state'");
    await assert.rejects(runCleanup("prune", manifest), /journal changed/);
    assert.equal((await db.query("select count(*)::int as n from cam_journal_revisions")).rows[0].n, 4);
    await runCleanup("copy", manifest);
    await runCleanup("prune", manifest);
    assert.deepEqual((await db.query("select reason from cam_journal_revisions order by id")).rows.map(row => row.reason), ["save", "nightly", "recovery"]);
    await runCleanup("release", manifest);
    assert.equal((await db.query("select count(*)::int as n from cam_journal_screenshots where image_data is not null")).rows[0].n, 1);
    await runCleanup("release", manifest); // resumes after the first batch
    await runCleanup("release", manifest); // safe retry after full release
    for (const table of ["trade_screenshots", "cam_journal_screenshots"]) {
      const result: QueryResult<{ image_data: Buffer | null; storage_path: string | null }> = await db.query(`select image_data, storage_path from ${table}`);
      assert.equal(result.rows.length, 1);
      assert.equal(result.rows[0].image_data, null);
      assert.ok(result.rows[0].storage_path);
    }
    await assert.rejects(runCleanup("compact", manifest), /Confirm Supabase/);
    process.env.CLEANUP_COMPACTION = "headroom-and-locks-verified";
    await runCleanup("compact", manifest);
    process.env.CLEANUP_EXPECTED_HOST = "wrong-host";
    await assert.rejects(runCleanup("audit"), /match the intended/);
  } finally {
    r2.restore();
    for (const key of ["DATABASE_URL", "CLEANUP_RELEASE_BATCH_SIZE",
      "CLEANUP_EXPECTED_HOST", "CLEANUP_EXPECTED_PROJECT_REF", "CLEANUP_WRITES", "CLEANUP_BACKUP_ATTESTATION", "CLEANUP_MAINTENANCE", "CLEANUP_COMPACTION"]) {
      if (previous[key] === undefined) delete process.env[key]; else process.env[key] = previous[key];
    }
    if (db) await db.end();
    try { await admin.query(`drop database if exists ${name}`); }
    finally { await admin.end(); await rm(directory, { recursive: true, force: true }); }
  }
});
