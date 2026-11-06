/** Operator tool. No changes by default. See docs/supabase-cleanup.md. */
import fs from "node:fs/promises";
import { createReadStream } from "node:fs";
import { createHash } from "node:crypto";
import { pathToFileURL } from "node:url";
import { Pool } from "pg";
import { copyImageToStorage, imageHash, readStoredImage, screenshotStorageTarget, type ScreenshotTable, type ImageSource } from "../lib/screenshot-storage";

const tables: ScreenshotTable[] = ["trade_screenshots", "cam_journal_screenshots"];
type Entry = { table: ScreenshotTable; id: string; path: string; sha256: string; md5: string; bytes: number };
type Manifest = { version: 2; storage?: { account: string; bucket: string }; host: string; database: string; projectRef: string; createdAt: string; images: Entry[]; revisionIds: string[]; liveStateHash: string };

async function fileHash(path: string) {
  const hash = createHash("sha256");
  for await (const bytes of createReadStream(path)) hash.update(bytes);
  return hash.digest("hex");
}

export async function runCleanup(command = "audit", manifestFile?: string) {
  if (!["audit", "copy", "verify", "release", "prune", "compact"].includes(command)) throw new Error("Unknown cleanup command.");
  const connectionString = process.env.DATABASE_URL;
  if (!connectionString) throw new Error("DATABASE_URL is required.");
  const url = new URL(connectionString);
  if (!process.env.CLEANUP_EXPECTED_HOST || url.hostname !== process.env.CLEANUP_EXPECTED_HOST) {
    throw new Error("CLEANUP_EXPECTED_HOST must match the intended database host exactly.");
  }
  if (url.port === "6543") throw new Error("Use a direct or session PostgreSQL connection.");
  const projectRef = process.env.CLEANUP_EXPECTED_PROJECT_REF;
  if (!projectRef || (url.hostname !== "localhost" &&
      url.hostname !== `db.${projectRef}.supabase.co` && decodeURIComponent(url.username) !== `postgres.${projectRef}`)) {
    throw new Error("Postgres connection does not match CLEANUP_EXPECTED_PROJECT_REF.");
  }
  const storage = ["copy", "verify", "release"].includes(command) ? screenshotStorageTarget() : undefined;
  if (storage && (storage.account !== process.env.CLEANUP_EXPECTED_R2_ACCOUNT_ID ||
      storage.bucket !== process.env.CLEANUP_EXPECTED_R2_BUCKET)) {
    throw new Error("R2 account and bucket must match the expected migration destination.");
  }
  const mutating = !["audit", "verify"].includes(command);
  if (mutating) {
    if (process.env.CLEANUP_WRITES !== "confirmed") throw new Error("Mutations require CLEANUP_WRITES=confirmed.");
    // Attestation is operator evidence of a rehearsal, not a substitute for it.
    const attestation = JSON.parse(await fs.readFile(process.env.CLEANUP_BACKUP_ATTESTATION || "", "utf8"));
    if (attestation.projectRef !== projectRef || attestation.host !== url.hostname || attestation.database !== decodeURIComponent(url.pathname.slice(1)) ||
        !attestation.restoreVerifiedAt || !attestation.secondCopyVerified ||
        await fileHash(attestation.dumpFile) !== attestation.sha256) throw new Error("Verified backup attestation does not match this target/dump.");
    if (["release", "prune", "compact"].includes(command) && process.env.CLEANUP_MAINTENANCE !== "writes-and-jobs-paused") {
      throw new Error("Pause writes and jobs before destructive cleanup.");
    }
  }
  const db = new Pool({ connectionString, max: 2,
    // Use trusted roots; operators supply their provider CA through normal TLS configuration.
    ssl: url.hostname === "localhost" ? false : { rejectUnauthorized: true } });
  const target = { projectRef, host: url.hostname, database: decodeURIComponent(url.pathname.slice(1)) };
  try {
    await db.query("set search_path = public");
    if (command === "audit") {
      const result = await db.query(`select current_database() as database, pg_database_size(current_database())::text as database_bytes`);
      const sizes = await db.query(`select relname, pg_total_relation_size(relid)::text as bytes from pg_statio_user_tables order by pg_total_relation_size(relid) desc`);
      const reasons = await db.query("select reason, count(*)::text as rows from cam_journal_revisions group by reason");
      const settings = await db.query("select key, pg_column_size(value) as bytes from app_settings order by bytes desc");
      console.log(JSON.stringify({ ...target, database: result.rows, sizes: sizes.rows, revisions: reasons.rows, settings: settings.rows }, null, 2));
      return;
    }
    if (!manifestFile) throw new Error("A private manifest file path is required.");
    let manifest: Manifest;
    if (command === "copy") {
      // Can resume an existing manifest. Source hashes are rechecked on every pass.
      try { manifest = JSON.parse(await fs.readFile(manifestFile, "utf8")); }
      catch (error) {
        if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
        manifest = { version: 2, ...target, storage, createdAt: new Date().toISOString(), images: [], revisionIds: [], liveStateHash: "" };
      }
    } else manifest = JSON.parse(await fs.readFile(manifestFile, "utf8"));
    if (manifest.version !== 2 || manifest.projectRef !== projectRef || manifest.host !== target.host || manifest.database !== target.database) throw new Error("Manifest target mismatch.");
    if (storage && (manifest.storage?.account !== storage.account || manifest.storage?.bucket !== storage.bucket)) {
      throw new Error("Manifest R2 destination mismatch.");
    }
    const save = async () => {
      await fs.writeFile(`${manifestFile}.tmp`, JSON.stringify(manifest, null, 2), { mode: 0o600 });
      await fs.rename(`${manifestFile}.tmp`, manifestFile);
    };
    if (command === "copy") {
      for (const table of tables) {
        let after = "";
        for (;;) {
          // Fetch one at a time: image sizes can vary substantially.
          const result = await db.query(`select * from ${table} where id > $1 order by id limit 1`, [after]);
          const row = result.rows[0];
          if (!row) break;
          after = row.id;
          const bytes: Buffer = row.image_data ?? await readStoredImage(row as ImageSource);
          const stored = await copyImageToStorage(table, row.id, row.mime_type, bytes);
          const updated = await db.query(`update ${table} set storage_path=$2, image_sha256=$3, image_bytes=$4
            where id=$1 and image_data is not distinct from $5 and storage_path is not distinct from $6`,
          [row.id, stored.storage_path, stored.image_sha256, stored.image_bytes, row.image_data, row.storage_path]);
          if (updated.rowCount !== 1) throw new Error("Image changed during migration; rerun copy.");
          manifest.images = manifest.images.filter(e => e.table !== table || e.id !== row.id);
          manifest.images.push({ table, id: row.id, path: stored.storage_path, sha256: stored.image_sha256, md5: createHash("md5").update(bytes).digest("hex"), bytes: stored.image_bytes });
          await save();
          console.log(JSON.stringify({ table, id: row.id, copiedBytes: bytes.length }));
        }
      }
      // Candidate IDs are explicit and only routine saves. Preserve latest pre-save.
      const revisions = await db.query(`select id::text from cam_journal_revisions where reason='save'
        and id <> (select max(id) from cam_journal_revisions where reason='save') order by id`);
      manifest.revisionIds = revisions.rows.map(row => row.id);
      const live = await db.query("select md5(value::text) as hash from app_settings where key='cam_journal_state'");
      if (!live.rows[0]) throw new Error("Missing authoritative Cam state.");
      manifest.liveStateHash = live.rows[0].hash;
      await save();
      return;
    }
    if (command === "verify" || command === "release") {
      const batchLimit = Number(process.env.CLEANUP_RELEASE_BATCH_SIZE);
      if (command === "release" && (!Number.isSafeInteger(batchLimit) || batchLimit < 1)) {
        throw new Error("Set CLEANUP_RELEASE_BATCH_SIZE to the verified safe row count for this batch.");
      }
      let releasedRows = 0;
      const pending: Entry[] = [];
      // Prove exact, complete manifest coverage first. No clearing on partial success.
      for (const table of tables) {
        const count = await db.query(`select count(*)::int as count from ${table}`);
        const entries = manifest.images.filter(e => e.table === table);
        if (new Set(entries.map(e => e.id)).size !== entries.length || count.rows[0].count !== entries.length) throw new Error("Manifest coverage changed; rerun copy.");
      }
      for (const e of manifest.images) {
        if (!tables.includes(e.table)) throw new Error("Invalid manifest table.");
        const result = await db.query(`select * from ${e.table} where id=$1`, [e.id]);
        const row = result.rows[0];
        if (!row || row.storage_path !== e.path || row.image_sha256 !== e.sha256 || Number(row.image_bytes) !== e.bytes) throw new Error("Manifest identity mismatch.");
        if (row.image_data) {
          if (imageHash(row.image_data) !== e.sha256) throw new Error("Source hash changed.");
          pending.push(e);
        }
        const object = await readStoredImage({ ...row, image_data: null });
        if (createHash("md5").update(object).digest("hex") !== e.md5) throw new Error("Manifest MD5 mismatch.");
      }
      if (command === "release") {
        for (const e of pending.slice(0, batchLimit)) {
          const result = await db.query(`update ${e.table} set image_data=null
            where id=$1 and storage_path=$2 and image_sha256=$3 and image_bytes=$4
              and (image_data is null or image_data=$5)`, [e.id, e.path, e.sha256, e.bytes,
            await readStoredImage({ image_data: null, storage_path: e.path, image_sha256: e.sha256, image_bytes: e.bytes })]);
          if (result.rowCount !== 1) throw new Error("Image changed during release; stopped.");
          releasedRows++;
        }
      }
      const remainingRows = pending.length - releasedRows;
      console.log(JSON.stringify({ verifiedImages: manifest.images.length, releasedRows, remainingRows }));
    } else if (command === "prune") {
      const client = await db.connect();
      try {
        await client.query("begin");
        await client.query("select pg_advisory_xact_lock(hashtext('cam-journal-revisions'))");
        const state = await client.query("select md5(value::text) as hash from app_settings where key='cam_journal_state' for update");
        if (state.rows[0]?.hash !== manifest.liveStateHash) throw new Error("Live journal changed; refresh maintenance manifest.");
        const deleted = await client.query(`delete from cam_journal_revisions where id=any($1::bigint[]) and reason='save'
          and id <> (select max(id) from cam_journal_revisions where reason='save') returning id`, [manifest.revisionIds]);
        if (deleted.rowCount !== manifest.revisionIds.length) throw new Error("Revision candidates changed; transaction rolled back.");
        await client.query("commit");
        console.log(JSON.stringify({ prunedRevisions: deleted.rowCount }));
      } catch (error) { await client.query("rollback"); throw error; }
      finally { client.release(); }
    } else if (command === "compact") {
      if (process.env.CLEANUP_COMPACTION !== "headroom-and-locks-verified") throw new Error("Confirm Supabase-supported compaction, headroom and locks first.");
      // Same session: SET must apply to the VACUUM commands.
      const client = await db.connect();
      try {
        await client.query("set lock_timeout='5s'");
        for (const table of ["cam_journal_revisions", ...tables]) await client.query(`vacuum (full, analyze) public.${table}`);
      } finally { client.release(); }
    }
    const measured = await db.query("select pg_database_size(current_database())::text as database_bytes");
    console.log(JSON.stringify(measured.rows[0]));
  } finally { await db.end(); }
}

if (process.argv[1] && pathToFileURL(process.argv[1]).href === import.meta.url) runCleanup(process.argv[2], process.argv[3]).catch(error => {
  // Avoid logging database URLs, object credentials, SQL parameter bytes or driver details.
  console.error(error instanceof Error ? error.message.replace(/postgres(?:ql)?:\/\/\S+/gi, "[database URL redacted]") : "Cleanup failed.");
  process.exitCode = 1;
});
