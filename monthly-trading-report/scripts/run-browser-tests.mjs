import { spawn } from "node:child_process";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { randomUUID } from "node:crypto";
import { createHash } from "node:crypto";
import { Client } from "pg";

// Never point the browser suite at a user's server, data directory, or database.
const directory = await mkdtemp(join(tmpdir(), "trading-dashboard-e2e-"));
const password = randomUUID();
const screenshotRoutes = process.argv.includes("--screenshot-routes");
let testDatabase;
let testSchema;
const env = { ...process.env, DATABASE_URL: "", LOCAL_DATA_DIR: directory,
  E2E_DATA_DIR: directory, E2E_PASSWORD: password, NODE_ENV: "development",
  APP_SECRET: randomUUID(), TRADER_USERS: `cam:${password},branden:${password}`,
  CODEX_JOURNAL_PASSWORD: "", ALPACA_API_KEY: "", ALPACA_API_SECRET: "",
  OPENAI_API_KEY: "", CRON_SECRET: "", R2_ACCOUNT_ID: "", R2_ACCESS_KEY_ID: "", R2_SECRET_ACCESS_KEY: "", R2_SCREENSHOT_BUCKET: "",
  SCREENSHOT_DIRECT_SERVING: "", SCREENSHOT_STORAGE_WRITES: "", NEXT_TELEMETRY_DISABLED: "1" };
for (const key of ["HOST", "PORT", "SECURE", "USERNAME", "PASSWORD"]) env[`SNAPSHOT_SMTP_${key}`] = "";
for (const key of ["FROM", "TO"]) env[`SNAPSHOT_EMAIL_${key}`] = "";
try {
  if (screenshotRoutes) {
    const target = new URL(process.env.BACKUP_TEST_DATABASE_URL || "http://invalid");
    if (target.hostname !== "localhost" || !/^\/trading_dashboard_backup_test(?:_[a-z0-9]+)?$/.test(target.pathname)) throw new Error("Screenshot route tests require isolated local PostgreSQL.");
    testDatabase = new Client({connectionString:target.toString(),ssl:false});
    await testDatabase.connect();
    testSchema = `image_routes_${randomUUID().replaceAll("-", "")}`;
    await testDatabase.query(`create schema ${testSchema}`);
    target.searchParams.set("options",`-c search_path=${testSchema}`);
    env.DATABASE_URL = target.toString(); env.E2E_SCREENSHOT_ROUTES = "1";
    env.SCREENSHOT_DIRECT_SERVING = "enabled";
    env.R2_ACCOUNT_ID = "0123456789abcdef0123456789abcdef";
    env.R2_SCREENSHOT_BUCKET = "trade-screenshots";
    env.R2_ACCESS_KEY_ID = "test-only"; env.R2_SECRET_ACCESS_KEY = "test-only";
    const png = Buffer.from("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAACXBIWXMAAAPoAAAD6AG1e1JrAAAADElEQVQImWMITSsHAAJGATN1EKEHAAAAAElFTkSuQmCC","base64");
    const hash = createHash("sha256").update(png).digest("hex");
    for (const [table,owners] of [["trade_screenshots","trade_id text,user_id text"],["cam_journal_screenshots","entity_type text,entity_id text"]]) {
      await testDatabase.query(`create table ${testSchema}.${table}(id text primary key,${owners},file_name text,mime_type text,image_data bytea,storage_path text,image_sha256 text,image_bytes bigint,created_at timestamptz default now())`);
    }
    await testDatabase.query(`insert into ${testSchema}.trade_screenshots values ('trade-shot','trade-1','branden','chart.png','image/png',$1,$2,$3,$4,now())`,[png,`trade_screenshots/trade-shot/${hash}`,hash,png.length]);
    for (const [id,type] of [["shared-shot","setup"],["watch-shot","watchlist-item"]]) await testDatabase.query(`insert into ${testSchema}.cam_journal_screenshots values ($1,$2,'entity','chart\u202f.png','image/png',$3,$4,$5,$6,now())`,[id,type,png,`cam_journal_screenshots/${id}/${hash}`,hash,png.length]);
  }
  const args=process.argv.slice(2).filter(arg=>arg!=="--screenshot-routes");
  const child = spawn(process.execPath, ["node_modules/@playwright/test/cli.js", "test", ...args],
    { stdio: "inherit", env });
  const stop = () => child.kill("SIGTERM");
  process.on("SIGINT", stop);
  process.on("SIGTERM", stop);
  try {
    process.exitCode = await new Promise((resolve, reject) => {
      child.on("error", reject);
      child.on("exit", code => resolve(code ?? 1));
    });
  } finally {
    process.off("SIGINT", stop);
    process.off("SIGTERM", stop);
  }
} finally {
  if (testDatabase) {
    try { if(testSchema) await testDatabase.query(`drop schema ${testSchema} cascade`); }
    finally { await testDatabase.end(); }
  }
  await rm(directory, { recursive: true, force: true });
}
