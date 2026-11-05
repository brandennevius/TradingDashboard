import assert from "node:assert/strict";
import test from "node:test";
import { createBackupDownload, createPostgresBackupDownload, type BackupQueryClient, type BrandenJournalBackup } from "../lib/branden-backup-stream";

const image = Buffer.from([0, 1, 2, 127, 128, 255]);
const report = { id: "report-1", userId: "branden", notes: "Résumé ☃" };
const trade = { id: "trade-1", userId: "branden", symbol: "XYZ", notes: 'Wait for "confirmation"', stopPrice: 98, executions: [{ id: "fill-1", shares: 10 }] };
const settings = { branden_portfolio_names: { portfolios: ["Main"], defaultPortfolio: "Main" } };

function database(options: { screenshots?: number; fail?: string } = {}) {
  const queries: Array<{ sql: string; values?: unknown[] }> = [];
  const fetched = { reports: false, trades: false, screenshots: 0 };
  let releases = 0;
  const client: BackupQueryClient = {
    async query(sql, values) {
      queries.push({ sql, values });
      if (options.fail && sql.includes(options.fail)) throw new Error("Synthetic database failure");
      if (sql.startsWith("select key")) return { rows: Object.entries(settings).map(([key, value]) => ({ key, value })) };
      if (sql === "fetch forward 100 from backup_reports") {
        const rows = fetched.reports ? [] : [report]; fetched.reports = true; return { rows };
      }
      if (sql === "fetch forward 100 from backup_trades") {
        const rows = fetched.trades ? [] : [trade]; fetched.trades = true; return { rows };
      }
      if (sql === "fetch forward 1 from backup_screenshots") {
        if (fetched.screenshots >= (options.screenshots ?? 2)) return { rows: [] };
        fetched.screenshots++;
        return { rows: [{ id: `shot-${fetched.screenshots}`, trade_id: trade.id, user_id: "branden", file_name: "chart.png", mime_type: "image/png", image_data: image, created_at: "2026-11-01T12:00:00Z" }] };
      }
      return { rows: [] };
    },
    release() { releases++; }
  };
  return { client, queries, fetched, releases: () => releases };
}

const mapping = {
  reportColumns: "id, user_id", tradeColumns: "id, user_id", settingKeys: ["branden_portfolio_names"],
  mapReport: (row: Record<string, unknown>) => row as unknown as BrandenJournalBackup["reports"][number],
  mapTrade: (row: Record<string, unknown>) => row as unknown as BrandenJournalBackup["trades"][number]
};

test("streamed PostgreSQL backup retains the version-1 format, records, Unicode, and exact image bytes", async () => {
  const db = database();
  const download = await createPostgresBackupDownload(db.client, mapping);
  const backup = await new Response(download.body).json();
  assert.deepEqual(backup, {
    format: "branden-journal-backup", version: 1, exportedAt: download.exportedAt, userId: "branden",
    reports: [report], trades: [trade], settings,
    tradeScreenshots: [1, 2].map(index => ({
      id: `shot-${index}`, tradeId: trade.id, userId: "branden", fileName: "chart.png", mimeType: "image/png",
      imageDataBase64: image.toString("base64"), createdAt: "2026-11-01T12:00:00.000Z"
    }))
  });
  assert.deepEqual(Buffer.from(backup.tradeScreenshots[0].imageDataBase64, "base64"), image);
  assert.equal(db.queries[0].sql, "begin isolation level repeatable read read only");
  const cursors = db.queries.filter(query => query.sql.startsWith("declare"));
  assert.equal(cursors.length, 3);
  for (const cursor of cursors) {
    assert.match(cursor.sql, /where user_id = \$1/);
    assert.deepEqual(cursor.values, ["branden"]);
  }
  assert.equal(db.queries.at(-1)?.sql, "commit");
  assert.equal(db.releases(), 1);
});

test("large screenshot collections are fetched one image at a time only as the consumer reads", async () => {
  const db = database({ screenshots: 10_000 });
  const download = await createPostgresBackupDownload(db.client, mapping);
  assert.equal(db.fetched.screenshots, 0);
  const reader = download.body.getReader();
  const decoder = new TextDecoder();
  while (true) {
    const chunk = await reader.read();
    assert.equal(chunk.done, false);
    if (decoder.decode(chunk.value).includes('"imageDataBase64"')) break;
  }
  assert.equal(db.fetched.screenshots, 1, "The remaining 9,999 images must not be fetched or encoded eagerly");
  assert.equal(db.releases(), 0);
  await reader.cancel();
  assert.equal(db.queries.at(-1)?.sql, "rollback");
  assert.equal(db.releases(), 1);
  assert.equal(db.queries.some(query => query.sql === "commit"), false);
});

test("cancellation before the first byte closes the read-only transaction and releases its connection", async () => {
  const db = database();
  const download = await createPostgresBackupDownload(db.client, mapping);
  await download.body.cancel();
  assert.equal(db.fetched.screenshots, 0);
  assert.equal(db.queries.at(-1)?.sql, "rollback");
  assert.equal(db.releases(), 1);
});

test("a mid-download database failure rejects body consumption and cleans up without a successful commit", async () => {
  const db = database({ fail: "fetch forward 1" });
  const download = await createPostgresBackupDownload(db.client, mapping);
  await assert.rejects(new Response(download.body).blob(), /Synthetic database failure/);
  assert.equal(db.queries.at(-1)?.sql, "rollback");
  assert.equal(db.releases(), 1);
  assert.equal(db.queries.some(query => query.sql === "commit"), false);
});

test("setup failures release the connection before an export response is returned", async () => {
  for (const fail of ["begin", "declare backup_trades"]) {
    const db = database({ fail });
    await assert.rejects(createPostgresBackupDownload(db.client, mapping), /Synthetic database failure/);
    assert.equal(db.releases(), 1);
    assert.equal(db.queries.some(query => query.sql === "rollback"), fail !== "begin");
  }
});

test("a failed final commit rejects the download, rolls back, and releases the connection", async () => {
  const db = database({ fail: "commit" });
  const download = await createPostgresBackupDownload(db.client, mapping);
  await assert.rejects(new Response(download.body).blob(), /Synthetic database failure/);
  assert.equal(db.queries.at(-1)?.sql, "rollback");
  assert.equal(db.releases(), 1);
});

test("local exports use the same downloadable format, including empty sections", async () => {
  const backup: BrandenJournalBackup = {
    format: "branden-journal-backup", version: 1, exportedAt: "2026-11-04T12:00:00Z", userId: "branden",
    reports: [], trades: [], settings: {}, tradeScreenshots: []
  };
  assert.deepEqual(await new Response(createBackupDownload(backup).body).json(), backup);
});
