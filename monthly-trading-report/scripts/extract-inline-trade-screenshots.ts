import { Pool } from "pg";
import { decodeInlineImage, hasInlineImages, writeTradeWithInlineScreenshots } from "../lib/inline-trade-screenshots";

async function main() {
const apply = process.argv.includes("--apply");
if (process.argv.slice(2).some(arg => !["--apply", "--dry-run"].includes(arg))) throw new Error("Use --dry-run (default) or --apply.");
if (apply && process.argv.includes("--dry-run")) throw new Error("Choose one mode.");
const connectionString = process.env.DATABASE_URL;
if (!connectionString) throw new Error("DATABASE_URL is required.");
if (apply) {
  const target = new URL(connectionString);
  if (process.env.INLINE_MIGRATION_BACKUP_VERIFIED !== "yes" ||
    !process.env.CLEANUP_EXPECTED_HOST || target.hostname !== process.env.CLEANUP_EXPECTED_HOST ||
    !process.env.CLEANUP_EXPECTED_PROJECT_REF || !decodeURIComponent(target.username).endsWith(`.${process.env.CLEANUP_EXPECTED_PROJECT_REF}`) ||
    process.env.SCREENSHOT_STORAGE_WRITES !== "enabled") throw new Error("Apply requires a verified backup, exact expected database host/project, and enabled R2 writes.");
}
const db = new Pool({connectionString, max:1, connectionTimeoutMillis:10000,
  ssl: new URL(connectionString).hostname === "localhost" ? false : {rejectUnauthorized:true}});
try {
  const client = await db.connect();
  let rows;
  try {
    await client.query("begin isolation level repeatable read read only");
    await client.query("set local statement_timeout='60s'");
    rows = (await client.query("select id,user_id,screenshots from trade_logs where screenshots::text like '%data:%'")).rows;
    await client.query("rollback");
  } catch(error) {await client.query("rollback");throw error;}
  finally {client.release();}
  let trades=0,images=0,decodedOriginalBytes=0;
  for (const row of rows) {
    if (!Array.isArray(row.screenshots) || !hasInlineImages(row.screenshots)) continue;
    for (const value of row.screenshots) if (value.startsWith("data:")) { images++; decodedOriginalBytes+=decodeInlineImage(value).bytes.length; }
    if (apply) await writeTradeWithInlineScreenshots(db,row.id,row.user_id,row.screenshots,0,
      `update trade_logs set screenshots=$1::jsonb, updated_at=now()
       where id=$2 and user_id=$3 and screenshots=$4::jsonb returning id`,
      [null,row.id,row.user_id,JSON.stringify(row.screenshots)]);
    trades++;
  }
  console.log(JSON.stringify({mode:apply?"applied":"read-only dry-run",trades,images,decodedOriginalBytes}));
} finally {await db.end();}

}
main().catch(error => { console.error(error instanceof Error ? error.message : "Extraction failed."); process.exitCode=1; });
