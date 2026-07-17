import { generateDailyPortfolioSnapshot } from "../lib/daily-portfolio-snapshot-server";
import { sendDailyPortfolioSnapshotEmail } from "../lib/snapshot-email";

function argument(name: string) {
  const index = process.argv.indexOf(name);
  return index >= 0 ? process.argv[index + 1] : undefined;
}

async function main() {
  const session = argument("--session");
  if (!session) throw new Error("Usage: npm run snapshot:daily -- --session YYYY-MM-DD [--account NAME] [--send-email]");
  const result = await generateDailyPortfolioSnapshot({ session, accountName: argument("--account") });
  const email = process.argv.includes("--send-email")
    ? await sendDailyPortfolioSnapshotEmail({ snapshot: result.snapshot, markdown: result.markdown, baseName: result.baseName })
    : { status: "not_requested" as const };
  process.stdout.write(`${JSON.stringify({
    status: result.snapshot.snapshot_status,
    jsonPath: result.jsonPath,
    markdownPath: result.markdownPath,
    openPositions: result.snapshot.open_positions.length,
    closedTrades: result.snapshot.trades_closed_during_session.length,
    criticalWarnings: result.snapshot.critical_warning_count,
    email
  }, null, 2)}\n`);
}

main().catch((error) => {
  process.stderr.write(`${error instanceof Error ? error.message : "Snapshot generation failed."}\n`);
  process.exitCode = 1;
});
