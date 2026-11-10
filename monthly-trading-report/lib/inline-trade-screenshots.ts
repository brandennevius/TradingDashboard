import { createHash } from "node:crypto";
import type { Pool, PoolClient } from "pg";
import { copyImageToStorage } from "./screenshot-storage";

export function decodeInlineImage(value: string) {
  const match = value.match(/^data:(image\/[a-zA-Z0-9.+-]+);base64,([A-Za-z0-9+/]*={0,2})$/);
  if (!match) throw new Error("Invalid inline screenshot.");
  const bytes = Buffer.from(match[2], "base64");
  if (!bytes.length || bytes.toString("base64") !== match[2]) throw new Error("Invalid screenshot base64.");
  return { bytes, mimeType: match[1].toLowerCase() };
}

export const hasInlineImages = (screenshots: string[]) => screenshots.some(value => value.startsWith("data:"));

export async function prepareInlineScreenshots(tradeId: string, userId: string, screenshots: string[]) {
  const rows = [];
  const references = [];
  for (const [index, value] of screenshots.entries()) {
    if (!value.startsWith("data:")) { references.push(value); continue; }
    const { bytes, mimeType } = decodeInlineImage(value);
    // Stable across retries, distinct for repeated identical attachments by slot.
    const id = createHash("sha256").update(JSON.stringify([tradeId, userId, index, value])).digest("hex");
    const source = await copyImageToStorage("trade_screenshots", id, mimeType, bytes);
    rows.push({ id, tradeId, userId, mimeType, ...source });
    references.push(`/api/trades/${encodeURIComponent(tradeId)}/screenshots/${id}`);
  }
  return { rows, references };
}

export async function persistInlineScreenshots(client: PoolClient, prepared: Awaited<ReturnType<typeof prepareInlineScreenshots>>) {
  for (const row of prepared.rows) {
    await client.query(`insert into trade_screenshots
      (id, trade_id, user_id, file_name, mime_type, image_data, storage_path, image_sha256, image_bytes)
      values ($1,$2,$3,$4,$5,null,$6,$7,$8) on conflict (id) do nothing`,
    [row.id,row.tradeId,row.userId,"original-screenshot",row.mimeType,row.storage_path,row.image_sha256,row.image_bytes]);
    const check = await client.query(`select id from trade_screenshots where id=$1 and trade_id=$2 and user_id=$3
      and storage_path=$4 and image_sha256=$5 and image_bytes=$6 and mime_type=$7`,
    [row.id,row.tradeId,row.userId,row.storage_path,row.image_sha256,row.image_bytes,row.mimeType]);
    if (check.rowCount !== 1) throw new Error("Screenshot identity conflict; trade was not changed.");
  }
}

/** Original bytes are verified in R2 before any reference can commit. */
export async function writeTradeWithInlineScreenshots(
  db: Pool, tradeId: string, userId: string, screenshots: string[], screenshotIndex: number,
  sql: string, values: unknown[]
) {
  if (!hasInlineImages(screenshots) || process.env.SCREENSHOT_STORAGE_WRITES !== "enabled") return db.query(sql, values);
  const prepared = await prepareInlineScreenshots(tradeId,userId,screenshots);
  const nextValues = [...values]; nextValues[screenshotIndex] = JSON.stringify(prepared.references);
  const client = await db.connect();
  try {
    await client.query("begin");
    await client.query("set local lock_timeout='3s'");
    await client.query("set local statement_timeout='60s'");
    const result = await client.query(sql,nextValues);
    if (result.rowCount !== 1) throw new Error("Trade not found or changed concurrently.");
    await persistInlineScreenshots(client,prepared);
    await client.query("commit");
    return result;
  } catch (error) { await client.query("rollback"); throw error; }
  finally { client.release(); }
}
