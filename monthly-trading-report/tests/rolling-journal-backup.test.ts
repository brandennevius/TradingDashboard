import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { mkdtemp, readFile, readdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";
import { Client } from "pg";
import { copyImageToStorage } from "../lib/screenshot-storage";
import { runRollingJournalBackup } from "../scripts/rolling-journal-backup";
import { mockR2 } from "./fixtures/mock-r2";

test("encrypted nightly backup restores PostgreSQL and keeps one copy of identical images", {
  skip: !process.env.BACKUP_TEST_DATABASE_URL
}, async () => {
  const database = new URL(process.env.BACKUP_TEST_DATABASE_URL!);
  assert.equal(database.hostname, "localhost");
  assert.match(database.pathname, /^\/trading_dashboard_backup_test(?:_[a-z0-9]+)?$/);
  const restore = new URL(database);
  restore.pathname += "_restored";
  const temp = await mkdtemp(path.join(tmpdir(), "journal-backup-test-"));
  const backup = path.join(temp, "archive");
  const previous = { ...process.env };
  const r2 = mockR2();
  const password = Buffer.from("disposable-encryption-key\n");
  const db = new Client({ connectionString: database.toString(), ssl: false });
  const run = (file: string, args: string[], input?: Buffer) => {
    const result = spawnSync(file, args, { input });
    assert.equal(result.status, 0, result.stderr?.toString());
  };
  try {
    Object.assign(process.env, {
      DATABASE_URL: database.toString(), CLEANUP_EXPECTED_HOST: "localhost",
      CLEANUP_EXPECTED_PROJECT_REF: "test", JOURNAL_BACKUP_ROOT: backup, GNUPGHOME: temp
    });
    await db.connect();
    await db.query(`create table public.trade_screenshots
      (id text primary key, storage_path text, image_sha256 text, image_bytes bigint);
      create table public.cam_journal_screenshots (like public.trade_screenshots including all)`);
    const bytes = Buffer.from("test screenshot bytes");
    const object = await copyImageToStorage("trade_screenshots", "test-image", "image/png", bytes);
    await db.query("insert into trade_screenshots values ($1,$2,$3,$4)", [
      "test-image", object.storage_path, object.image_sha256, object.image_bytes
    ]);
    await runRollingJournalBackup(password);
    r2.controls.failReads = true;
    await runRollingJournalBackup(password);
    const points = await readdir(path.join(backup, "points"));
    assert.equal(points.length, 2);
    assert.deepEqual(await readdir(path.join(backup, "objects")), [object.image_sha256 + ".gpg"]);
    run("createdb", ["--maintenance-db=" + database.toString(), restore.pathname.slice(1)]);
    const decrypt = (file: string, output: string) => run("gpg", [
      "--batch", "--pinentry-mode", "loopback", "--passphrase-fd", "0", "--output", output, "--decrypt", file
    ], password);
    const dump = path.join(temp, "restored.dump");
    decrypt(path.join(backup, "points", points[0], "database.dump.gpg"), dump);
    decrypt(path.join(backup, "objects", object.image_sha256 + ".gpg"), path.join(temp, "image"));
    run("pg_restore", ["--exit-on-error", "--dbname=" + restore.toString(), dump]);
    const restored = new Client({ connectionString: restore.toString(), ssl: false });
    await restored.connect();
    try {
      const { rows } = await restored.query("select id, image_sha256 from trade_screenshots");
      assert.deepEqual(rows, [{ id: "test-image", image_sha256: object.image_sha256 }]);
      assert.deepEqual(await readFile(path.join(temp, "image")), bytes);
    } finally {
      await restored.end();
    }
    const oldPoint = path.join(backup, "points", points[0]);
    const oldManifest = path.join(oldPoint, "manifest.json.gpg");
    const plaintext = path.join(temp, "manifest.json");
    decrypt(oldManifest, plaintext);
    const expired = JSON.parse(await readFile(plaintext, "utf8"));
    expired.at = "2000-01-01T00:00:00.000Z";
    await writeFile(plaintext, JSON.stringify(expired));
    await rm(oldManifest);
    run("gpg", ["--batch", "--pinentry-mode", "loopback", "--passphrase-fd", "0",
      "--symmetric", "--output", oldManifest, plaintext], password);
    process.env.JOURNAL_BACKUP_ENABLE_PRUNE = "true";
    await runRollingJournalBackup(password);
    assert.equal((await readdir(path.join(backup, "points"))).length, 2);
    assert.deepEqual(await readdir(path.join(backup, "objects")), [object.image_sha256 + ".gpg"]);
  } finally {
    await db.end();
    r2.restore();
    process.env = previous;
    await rm(temp, { recursive: true, force: true });
  }
});
