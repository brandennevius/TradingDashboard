import { readStoredImage, sharedScreenshotIds, type ImageSource } from "./screenshot-storage";
import type { MonthlyReport, TradeLogEntry } from "./types";

export type BrandenJournalBackup = {
  format: "branden-journal-backup";
  version: 1;
  exportedAt: string;
  userId: "branden";
  reports: MonthlyReport[];
  trades: TradeLogEntry[];
  settings: Record<string, unknown>;
  sharedScreenshots?: Array<{
    id: string; entityType: string; entityId: string; fileName: string;
    mimeType: string; imageDataBase64: string; createdAt: string;
  }>;
  tradeScreenshots: Array<{
    id: string;
    tradeId: string;
    userId: string;
    fileName: string;
    mimeType: string;
    imageDataBase64: string;
    createdAt: string;
  }>;
};

type BackupSource = Omit<BrandenJournalBackup, "reports" | "trades" | "tradeScreenshots" | "sharedScreenshots"> & {
  sharedScreenshots?: Iterable<NonNullable<BrandenJournalBackup["sharedScreenshots"]>[number]> | AsyncIterable<NonNullable<BrandenJournalBackup["sharedScreenshots"]>[number]>;
  reports: Iterable<MonthlyReport> | AsyncIterable<MonthlyReport>;
  trades: Iterable<TradeLogEntry> | AsyncIterable<TradeLogEntry>;
  tradeScreenshots: Iterable<BrandenJournalBackup["tradeScreenshots"][number]> | AsyncIterable<BrandenJournalBackup["tradeScreenshots"][number]>;
};

export type BackupDownload = { exportedAt: string; body: ReadableStream<Uint8Array> };

async function* jsonArray(rows: Iterable<unknown> | AsyncIterable<unknown>) {
  yield "[";
  let separator = "";
  for await (const row of rows) {
    yield separator + JSON.stringify(row);
    separator = ",";
  }
  yield "]";
}

async function* jsonBackup(source: BackupSource) {
  const { format, version, exportedAt, userId } = source;
  yield JSON.stringify({ format, version, exportedAt, userId }).slice(0, -1);
  yield ',"reports":';
  yield* jsonArray(source.reports);
  yield ',"trades":';
  yield* jsonArray(source.trades);
  yield ',"settings":' + JSON.stringify(source.settings);
  yield ',"tradeScreenshots":';
  yield* jsonArray(source.tradeScreenshots);
  if (source.sharedScreenshots) {
    yield ',"sharedScreenshots":';
    yield* jsonArray(source.sharedScreenshots);
  }
  yield "}";
}

export function createBackupDownload(
  source: BackupSource,
  finish: (completed: boolean) => Promise<void> = async () => {}
): BackupDownload {
  const iterator = jsonBackup(source);
  const encoder = new TextEncoder();
  let completion: Promise<void> | undefined;
  const close = (completed: boolean) => completion ??= finish(completed);
  return {
    exportedAt: source.exportedAt,
    body: new ReadableStream<Uint8Array>({
      async pull(controller) {
        try {
          const chunk = await iterator.next();
          if (chunk.done) {
            await close(true);
            controller.close();
          } else {
            controller.enqueue(encoder.encode(chunk.value));
          }
        } catch (error) {
          // Preserve the original failure even if rollback also fails.
          await close(false).catch(() => {});
          controller.error(error);
        }
      },
      async cancel() {
        try { await iterator.return(undefined); }
        finally { await close(false); }
      }
    }, { highWaterMark: 0 })
  };
}

// A structural interface keeps the real PostgreSQL adapter testable without
// requiring production credentials. The SQL column lists come only from store.ts.
export type BackupQueryClient = {
  query(sql: string, values?: unknown[]): Promise<{ rows: Record<string, unknown>[] }>;
  release(): void;
};

export async function createPostgresBackupDownload(client: BackupQueryClient, options: {
  reportColumns: string;
  tradeColumns: string;
  settingKeys: readonly string[];
  mapReport(row: Record<string, unknown>): MonthlyReport;
  mapTrade(row: Record<string, unknown>): TradeLogEntry;
}): Promise<BackupDownload> {
  let transactionStarted = false;
  try {
    await client.query("begin isolation level repeatable read read only");
    transactionStarted = true;
    await client.query(
      `declare backup_reports no scroll cursor for select ${options.reportColumns} from monthly_reports where user_id = $1 order by month asc`,
      ["branden"]
    );
    await client.query(
      `declare backup_trades no scroll cursor for select ${options.tradeColumns} from trade_logs where user_id = $1 order by entry_date desc, created_at desc`,
      ["branden"]
    );
    await client.query(
      "declare backup_screenshots no scroll cursor for select id, trade_id, user_id, file_name, mime_type, image_data, storage_path, image_sha256, image_bytes, created_at from trade_screenshots where user_id = $1 order by created_at asc, id asc",
      ["branden"]
    );
    const settings = await client.query("select key, value from app_settings where key = any($1::text[])", [options.settingKeys]);

    const sharedIds = sharedScreenshotIds(Object.fromEntries(settings.rows.map(row => [String(row.key), row.value])));
    if (sharedIds.length) {
      await client.query(`declare backup_shared_screenshots no scroll cursor for
        select id, entity_type, entity_id, file_name, mime_type, image_data,
          storage_path, image_sha256, image_bytes, created_at
        from cam_journal_screenshots where id = any($1::text[]) order by id`, [sharedIds]);
    }

    async function* rows(cursor: string, batchSize: number) {
      while (true) {
        const batch = await client.query(`fetch forward ${batchSize} from ${cursor}`);
        if (!batch.rows.length) return;
        yield* batch.rows;
      }
    }
    async function* reports() {
      for await (const row of rows("backup_reports", 100)) yield options.mapReport(row);
    }
    async function* trades() {
      for await (const row of rows("backup_trades", 100)) yield options.mapTrade(row);
    }
    async function* screenshots() {
      // Images dominate backup size: fetch/encode one at a time, on demand.
      for await (const row of rows("backup_screenshots", 1)) {
        yield {
          id: String(row.id), tradeId: String(row.trade_id), userId: String(row.user_id),
          fileName: String(row.file_name || ""), mimeType: String(row.mime_type || "image/png"),
          imageDataBase64: (await readStoredImage(row as ImageSource)).toString("base64"),
          createdAt: new Date(String(row.created_at)).toISOString()
        };
      }
    }
    async function* sharedScreenshots() {
      let found = 0;
      for await (const row of rows("backup_shared_screenshots", 1)) {
        found++;
        yield {
          id: String(row.id), entityType: String(row.entity_type), entityId: String(row.entity_id),
          fileName: String(row.file_name || ""), mimeType: String(row.mime_type),
          imageDataBase64: (await readStoredImage(row as ImageSource)).toString("base64"),
          createdAt: new Date(String(row.created_at)).toISOString()
        };
      }
      if (found !== sharedIds.length) throw new Error("Backup is missing referenced shared screenshots.");
    }
    return createBackupDownload({
      format: "branden-journal-backup", version: 1, userId: "branden", exportedAt: new Date().toISOString(),
      ...(sharedIds.length ? { sharedScreenshots: sharedScreenshots() } : {}),
      reports: reports(), trades: trades(), tradeScreenshots: screenshots(),
      settings: Object.fromEntries(settings.rows.map(row => [String(row.key), row.value]))
    }, async completed => {
      try {
        await client.query(completed ? "commit" : "rollback");
      } catch (error) {
        if (completed) await client.query("rollback").catch(() => {});
        throw error;
      }
      finally { client.release(); }
    });
  } catch (error) {
    try { if (transactionStarted) await client.query("rollback"); }
    catch { /* Preserve the original setup error. */ }
    finally { client.release(); }
    throw error;
  }
}
