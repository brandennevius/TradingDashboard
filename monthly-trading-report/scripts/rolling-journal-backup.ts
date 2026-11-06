import { spawn } from "node:child_process";
import { createHash, randomUUID } from "node:crypto";
import { createReadStream } from "node:fs";
import { access, mkdir, mkdtemp, readdir, readFile, rename, rm, writeFile } from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { Client } from "pg";
import { readStoredImage, screenshotBucketBytes, screenshotStorageTarget, type ImageSource } from "../lib/screenshot-storage";

type ImageRecord = ImageSource & {
  source_table: "trade_screenshots" | "cam_journal_screenshots";
  id: string;
  storage_path: string;
  image_sha256: string;
  image_bytes: string;
};

const env = process.env;
const hash = (bytes: Buffer) => createHash("sha256").update(bytes).digest("hex");
const exists = async (file: string) => access(file).then(() => true, error => {
  if (error.code === "ENOENT") return false;
  throw error;
});

async function command(executable: string, args: string[], passphrase?: Buffer) {
  await new Promise<void>((resolve, reject) => {
    const child = spawn(executable, args, { stdio: passphrase ? ["pipe", "ignore", "inherit"] : ["ignore", "ignore", "inherit"] });
    child.on("error", reject);
    child.on("close", code => code === 0 ? resolve() : reject(new Error(executable + " failed with status " + code)));
    child.stdin?.on("error", reject);
    child.stdin?.end(passphrase);
  });
}

async function readBackupImage(image: ImageRecord) {
  for (let attempt = 1; ; attempt++) {
    try {
      return await readStoredImage({ ...image, image_data: null });
    } catch (error) {
      const message = error instanceof Error ? error.message : "";
      if (attempt >= 5 || !/bad record mac|ECONNRESET|socket hang up/i.test(message)) throw error;
      await new Promise(resolve => setTimeout(resolve, attempt * 500));
    }
  }
}

async function fileHash(file: string) {
  const digest = createHash("sha256");
  for await (const bytes of createReadStream(file)) digest.update(bytes);
  return digest.digest("hex");
}

export async function runRollingJournalBackup(testPassphrase?: Buffer) {
  process.umask(0o077);
  const root = env.JOURNAL_BACKUP_ROOT;
  const database = env.DATABASE_URL;
  const retention = Number(env.JOURNAL_BACKUP_RETENTION_DAYS ?? 30);
  if (!root || !path.isAbsolute(root) || !database || !Number.isSafeInteger(retention) || retention < 1) {
    throw new Error("Set an absolute JOURNAL_BACKUP_ROOT, DATABASE_URL and a positive retention period.");
  }
  const url = new URL(database);
  const project = env.CLEANUP_EXPECTED_PROJECT_REF;
  const localTest = Boolean(testPassphrase) && url.hostname === "localhost" &&
    /^\/trading_dashboard_backup_test(?:_[a-z0-9]+)?$/.test(url.pathname);
  if (testPassphrase && !localTest) throw new Error("Test passphrases are only supported on disposable local databases.");
  if (!project || url.hostname !== env.CLEANUP_EXPECTED_HOST || url.port === "6543" ||
      (!localTest && url.hostname !== "db." + project + ".supabase.co" && decodeURIComponent(url.username) !== "postgres." + project)) {
    throw new Error("Backup database does not match the configured project and session endpoint.");
  }
  const storage = screenshotStorageTarget();
  if (storage.account !== env.CLEANUP_EXPECTED_R2_ACCOUNT_ID || storage.bucket !== env.CLEANUP_EXPECTED_R2_BUCKET) {
    throw new Error("Backup bucket does not match the configured source.");
  }
  if (!localTest && (url.password || url.search || !env.PGPASSFILE || !env.NODE_EXTRA_CA_CERTS)) {
    throw new Error("Use a private pgpass file and a verified database CA.");
  }
  const passphraseChunks: Buffer[] = [];
  let passphraseBytes = 0;
  for await (const chunk of testPassphrase ? [testPassphrase] : process.stdin) {
    passphraseBytes += chunk.length;
    if (passphraseBytes > 256) throw new Error("Backup passphrase input is too long.");
    passphraseChunks.push(chunk);
  }
  const passphrase = Buffer.concat(passphraseChunks);
  if (passphrase.length < 17 || passphrase.length > 256 || !passphrase.toString().endsWith("\n")) {
    throw new Error("Supply the backup passphrase on standard input.");
  }
  const gpg = env.JOURNAL_BACKUP_GPG ?? "gpg";
  const gpgArgs = ["--quiet", "--batch", "--pinentry-mode", "loopback", "--no-symkey-cache", "--passphrase-fd", "0"];
  const encrypt = (source: string, target: string) => command(gpg, [
    ...gpgArgs, "--symmetric", "--cipher-algo", "AES256", "--compress-algo", "none", "--output", target, source
  ], passphrase);
  const decrypt = (source: string, target: string) => command(gpg, [...gpgArgs, "--output", target, "--decrypt", source], passphrase);
  const objects = path.join(root, "objects");
  const points = path.join(root, "points");
  await mkdir(root, { recursive: true });
  const lock = path.join(root, ".lock");
  await mkdir(lock);
  try {
    await mkdir(objects, { recursive: true });
    await mkdir(points, { recursive: true });
    const temp = await mkdtemp(path.join(root, ".working-"));
    try {
      const start = new Date();
      const point = start.toISOString().replace(/[:.]/g, "-") + "-" + randomUUID();
      const dump = path.join(temp, "database.dump");
      const db = new Client({ connectionString: database, ssl: localTest ? false : { rejectUnauthorized: true }, connectionTimeoutMillis: 8000 });
      await db.connect();
      let images: ImageRecord[];
      try {
        await db.query("begin isolation level repeatable read read only");
        const snapshot = (await db.query("select pg_export_snapshot() as id")).rows[0].id;
        images = (await db.query<ImageRecord>(`
          select 'trade_screenshots' as source_table, id, storage_path, image_sha256, image_bytes
            from public.trade_screenshots where storage_path is not null
          union all
          select 'cam_journal_screenshots', id, storage_path, image_sha256, image_bytes
            from public.cam_journal_screenshots where storage_path is not null
          order by source_table,id`)).rows;
        const stats = (await db.query("select pg_database_size(current_database())::text as bytes")).rows[0];
        if (Number(stats.bytes) >= 350_000_000) console.warn("Supabase database is at or above 350 MB.");
        const pgDumpUrl = new URL(database);
        pgDumpUrl.search = localTest ? "sslmode=disable" :
          "sslmode=verify-full&sslrootcert=" + encodeURIComponent(env.NODE_EXTRA_CA_CERTS!);
        await command(env.JOURNAL_BACKUP_PG_DUMP ?? "pg_dump", [
          "--no-password", "--format=custom", "--snapshot=" + snapshot, "--file=" + dump, pgDumpUrl.toString()
        ]);
        await db.query("commit");
      } finally {
        await db.end();
      }
      const source = path.join(temp, "source");
      const checked = path.join(temp, "checked");
      const staged = path.join(temp, "encrypted");
      const repositoryIdentity = JSON.stringify({ project, storage });
      const marker = path.join(root, "repository.json.gpg");
      if (await exists(marker)) {
        await decrypt(marker, checked);
        if (await readFile(checked, "utf8") !== repositoryIdentity) throw new Error("Archive source does not match.");
        await rm(checked);
      } else {
        if ((await readdir(points)).length || (await readdir(objects)).length) {
          throw new Error("Existing archive has no identity record.");
        }
        await writeFile(source, repositoryIdentity);
        await encrypt(source, staged);
        await rename(staged, marker);
        await rm(source);
      }
      if (!localTest) {
        const bucketBytes = await screenshotBucketBytes();
        console.log(JSON.stringify({ bucketBytes }));
        if (bucketBytes >= 7_000_000_000) console.warn("R2 screenshot bucket is at or above 7 GB.");
      }
      let newObjects = 0;
      for (const image of images) {
        if (!/^[a-f0-9]{64}$/.test(image.image_sha256)) throw new Error("Invalid screenshot SHA-256.");
        const archive = path.join(objects, image.image_sha256 + ".gpg");
        if (await exists(archive)) continue;
        const bytes = await readBackupImage(image);
        await writeFile(source, bytes);
        await encrypt(source, staged);
        await decrypt(staged, checked);
        if (await fileHash(checked) !== image.image_sha256) throw new Error("Encrypted screenshot verification failed.");
        await rename(staged, archive);
        await rm(source);
        await rm(checked);
        newObjects++;
      }
      const manifest = { format: 1, at: start.toISOString(), project, storage, dumpSha256: await fileHash(dump), images };
      const manifestFile = path.join(temp, "manifest.json");
      await writeFile(manifestFile, JSON.stringify(manifest));
      const pointDir = path.join(temp, "point");
      await mkdir(pointDir);
      const encryptedDump = path.join(pointDir, "database.dump.gpg");
      await encrypt(dump, encryptedDump);
      await decrypt(encryptedDump, checked);
      if (await fileHash(checked) !== manifest.dumpSha256) throw new Error("Encrypted database verification failed.");
      await rm(checked);
      await encrypt(manifestFile, path.join(pointDir, "manifest.json.gpg"));
      await decrypt(path.join(pointDir, "manifest.json.gpg"), checked);
      if (hash(await readFile(checked)) !== hash(await readFile(manifestFile))) throw new Error("Encrypted manifest verification failed.");
      await rm(checked);
      await rename(pointDir, path.join(points, point));
      console.log(JSON.stringify({ point, images: images.length, newObjects, retentionDays: retention }));
      if (env.JOURNAL_BACKUP_ENABLE_PRUNE !== "true") return;
      const cutoff = start.getTime() - retention * 86400000;
      const keep = new Set<string>();
      const expired: string[] = [];
      for (const item of await readdir(points, { withFileTypes: true })) {
        if (!item.isDirectory()) throw new Error("Unexpected restore point entry.");
        const folder = path.join(points, item.name);
        await decrypt(path.join(folder, "manifest.json.gpg"), checked);
        const record = JSON.parse(await readFile(checked, "utf8")) as typeof manifest;
        await rm(checked);
        const timestamp = Date.parse(record.at);
        if (record.format !== 1 || record.project !== project || record.storage.account !== storage.account ||
            record.storage.bucket !== storage.bucket || !Number.isFinite(timestamp)) {
          throw new Error("Unrecognized backup manifest; retention stopped.");
        }
        if (timestamp < cutoff) expired.push(folder);
        else {
          for (const image of record.images) {
            if (!/^[a-f0-9]{64}$/.test(image.image_sha256)) throw new Error("Invalid retained screenshot hash.");
            keep.add(image.image_sha256 + ".gpg");
          }
        }
      }
      for (const name of keep) {
        if (!await exists(path.join(objects, name))) throw new Error("Missing retained screenshot; retention stopped.");
      }
      for (const folder of expired) await rm(folder, { recursive: true });
      for (const name of await readdir(objects)) {
        if (/^[a-f0-9]{64}\.gpg$/.test(name) && !keep.has(name)) await rm(path.join(objects, name));
      }
      console.log(JSON.stringify({ expiredPoints: expired.length, retainedObjects: keep.size }));
      if (!localTest && env.JOURNAL_BACKUP_ENABLE_LEGACY_PRUNE === "true") {
        const legacy = new Client({ connectionString: database, ssl: { rejectUnauthorized: true }, connectionTimeoutMillis: 8000 });
        await legacy.connect();
        try {
          const result = await legacy.query(`delete from public.cam_journal_revisions
            where reason = 'nightly' and created_at < now() - make_interval(days => $1)`, [retention]);
          console.log(JSON.stringify({ expiredCamNightlies: result.rowCount }));
        } finally {
          await legacy.end();
        }
      }
    } finally {
      await rm(temp, { recursive: true, force: true });
    }
  } finally {
    await rm(lock, { recursive: true, force: true });
  }
}

if (process.argv[1] && pathToFileURL(process.argv[1]).href === import.meta.url) runRollingJournalBackup().catch(error => {
  console.error(error instanceof Error ? error.message.replace(/postgres(?:ql)?:\/\/\S+/gi, "[database URL redacted]") : "Backup failed.");
  process.exitCode = 1;
});
