import { spawn } from "node:child_process";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { randomUUID } from "node:crypto";

// Never point the browser suite at a user's server, data directory, or database.
const directory = await mkdtemp(join(tmpdir(), "trading-dashboard-e2e-"));
const password = randomUUID();
const env = { ...process.env, DATABASE_URL: "", LOCAL_DATA_DIR: directory,
  E2E_DATA_DIR: directory, E2E_PASSWORD: password, NODE_ENV: "development",
  APP_SECRET: randomUUID(), TRADER_USERS: `cam:${password},branden:${password}`,
  CODEX_JOURNAL_PASSWORD: "", ALPACA_API_KEY: "", ALPACA_API_SECRET: "",
  OPENAI_API_KEY: "", CRON_SECRET: "", NEXT_TELEMETRY_DISABLED: "1" };
for (const key of ["HOST", "PORT", "SECURE", "USERNAME", "PASSWORD"]) env[`SNAPSHOT_SMTP_${key}`] = "";
for (const key of ["FROM", "TO"]) env[`SNAPSHOT_EMAIL_${key}`] = "";
try {
  const child = spawn(process.execPath, ["node_modules/@playwright/test/cli.js", "test", ...process.argv.slice(2)],
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
  await rm(directory, { recursive: true, force: true });
}
