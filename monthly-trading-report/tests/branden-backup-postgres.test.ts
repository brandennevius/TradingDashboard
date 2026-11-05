import assert from "node:assert/strict";
import { randomUUID } from "node:crypto";
import test from "node:test";
import { Client } from "pg";
import { trade } from "./fixtures/trade-review-export";

const databaseUrl = process.env.BACKUP_TEST_DATABASE_URL;

test("PostgreSQL backup round-trips a consistent snapshot with screenshots and preserves the other owner", {
  skip: databaseUrl ? false : "Set BACKUP_TEST_DATABASE_URL to an isolated local test database."
}, async () => {
  const url = new URL(databaseUrl!);
  // Never run restore/cleanup against a deployed database or an arbitrary local database.
  assert.equal(url.hostname, "localhost", "Backup tests require a localhost database host.");
  assert.match(url.pathname, /^\/trading_dashboard_backup_test(?:_[a-z0-9]+)?$/);
  const schema = `backup_test_${randomUUID().replaceAll("-", "")}`;
  const admin = new Client({ connectionString: url.toString(), ssl: false });
  await admin.connect();
  const previousDatabase = process.env.DATABASE_URL;
  let pool: ReturnType<typeof import("../lib/store")["getPool"]> = null;
  try {
    await admin.query(`create schema ${schema}`);
    url.searchParams.set("options", `-c search_path=${schema}`);
    process.env.DATABASE_URL = url.toString();
    const store = await import("../lib/store");
    pool = store.getPool();
    assert.ok(pool);
    const original = (await store.upsertTrade(trade({
      userId: "branden", portfolioTag: "Synthetic", notes: "Before export: café ☃", stopPrice: 98, screenshots: []
    }))).trade;
    const other = (await store.upsertTrade(trade({
      userId: "cam", portfolioTag: "Other", notes: "Other owner must survive", screenshots: []
    }))).trade;
    await store.saveBrandenPortfolioSettings({ portfolios: ["Synthetic"], defaultPortfolio: "Synthetic" });
    const bytes = Buffer.from([0, 1, 2, 3, 128, 255]);
    const screenshot = await store.saveTradeScreenshot(original.id, "branden", "synthetic.png", "image/png", bytes);

    const download = await store.exportBrandenJournalBackup();
    // A real concurrent row update must not change the export's established snapshot.
    await pool.query("update trade_logs set notes = $1 where id = $2", ["Changed during export", original.id]);
    const backup = await new Response(download.body).json();
    assert.equal(backup.trades.length, 1);
    assert.equal(backup.trades[0].id, original.id);
    assert.equal(backup.trades[0].notes, "Before export: café ☃");
    assert.equal(backup.trades[0].stopPrice, 98);
    assert.equal(backup.tradeScreenshots.length, 1);
    assert.deepEqual(Buffer.from(backup.tradeScreenshots[0].imageDataBase64, "base64"), bytes);

    const restored = await store.importBrandenJournalBackup(backup);
    assert.equal(restored.trades, 1);
    assert.equal(restored.screenshots, 1);
    const all = await store.listTrades();
    assert.equal(all.find(row => row.id === original.id)?.notes, "Before export: café ☃");
    assert.equal(all.find(row => row.id === other.id)?.notes, "Other owner must survive");
    assert.deepEqual((await store.getTradeScreenshot(screenshot.id))?.imageData, bytes);

    const cancelled = await store.exportBrandenJournalBackup();
    await cancelled.body.cancel();
    const idle = await admin.query("select count(*)::int as count from pg_stat_activity where datname = current_database() and state = 'idle in transaction'");
    assert.equal(idle.rows[0].count, 0);
  } finally {
    if (pool) await pool.end();
    if (previousDatabase === undefined) delete process.env.DATABASE_URL;
    else process.env.DATABASE_URL = previousDatabase;
    try { await admin.query(`drop schema if exists ${schema} cascade`); }
    finally { await admin.end(); }
  }
});
