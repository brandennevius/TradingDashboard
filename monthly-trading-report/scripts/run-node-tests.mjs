import { spawn } from "node:child_process";
import { readdir } from "node:fs/promises";

// Existing storage tests choose their own temporary cwd. A caller's development
// LOCAL_DATA_DIR must not redirect those writes into a real journal directory.
const files = process.argv.length > 2 ? process.argv.slice(2)
  : (await readdir("tests")).filter(name => /\.test\.(ts|mjs|js)$/.test(name)).sort().map(name => `tests/${name}`);
const env = { ...process.env, NODE_ENV: "test", DATABASE_URL: "", LOCAL_DATA_DIR: "",
  ALPACA_API_KEY: "", ALPACA_API_SECRET: "", OPENAI_API_KEY: "", CRON_SECRET: "" };
for (const key of ["HOST", "PORT", "SECURE", "USERNAME", "PASSWORD"]) env[`SNAPSHOT_SMTP_${key}`] = "";
for (const key of ["FROM", "TO"]) env[`SNAPSHOT_EMAIL_${key}`] = "";
const child = spawn(process.execPath, ["--import", "tsx", "--test", ...files], { stdio: "inherit", env });
process.exitCode = await new Promise((resolve, reject) => {
  child.on("error", reject);
  child.on("exit", code => resolve(code ?? 1));
});
