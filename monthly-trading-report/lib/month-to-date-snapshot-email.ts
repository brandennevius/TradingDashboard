import { snapshotEmailConfiguration, type SnapshotEmailTransport } from "./snapshot-email";
import { aggregateMtdDiagnostics, type buildMonthToDateSnapshot } from "./month-to-date-snapshot";

type Snapshot = ReturnType<typeof buildMonthToDateSnapshot>;
type Environment = Record<string, string | undefined>;

export async function sendMonthToDateSnapshotEmail(input: {
  snapshot: Snapshot;
  markdown: string;
  baseName: string;
  zip?: Buffer;
  transport?: SnapshotEmailTransport;
  environment?: Environment;
}) {
  if (input.snapshot.status === "BLOCKED") return { status: "not_sent" as const, reason: "Blocked snapshots are not emailed." };
  const configuration = snapshotEmailConfiguration(input.environment);
  if (!configuration.configured) return { status: "disabled" as const, reason: "SMTP environment variables are incomplete." };
  const { values } = configuration;
  const transport = input.transport || (await import("nodemailer")).default.createTransport({
    host: values.host,
    port: values.port,
    secure: values.secure,
    auth: { user: values.username, pass: values.password }
  });
  const attachments: Array<{ filename: string; content: string | Buffer; contentType: string }> = [
    { filename: `${input.baseName}.json`, content: `${JSON.stringify(input.snapshot, null, 2)}\n`, contentType: "application/json" },
    { filename: `${input.baseName}.md`, content: input.markdown, contentType: "text/markdown" }
  ];
  if (input.zip) attachments.push({ filename: `${input.baseName}.zip`, content: input.zip, contentType: "application/zip" });
  const diagnosticSummary = aggregateMtdDiagnostics(input.snapshot.diagnostics);
  const body = [
    `Portfolio: ${input.snapshot.portfolio.portfolio_name}`,
    `Period: ${input.snapshot.period.month} through ${input.snapshot.period.asOfDate}`,
    `Status: ${input.snapshot.status}`,
    `Current equity: ${input.snapshot.account_summary.current_equity ?? "unavailable"}`,
    `Drawdown cushion: ${input.snapshot.account_summary.remaining_drawdown_cushion ?? "unavailable"}`,
    `Realized MTD P&L: ${input.snapshot.account_summary.realized_mtd_pnl}`,
    `Current planned downside risk: ${input.snapshot.risk_summary.current_planned_downside_risk ?? "unavailable"}`,
    `Included trades: ${input.snapshot.performance_summary.total_included_trades}`,
    `Warning count: ${input.snapshot.diagnostics.length}`,
    `Warnings by code: ${diagnosticSummary.map((item) => `${item.code}=${item.count}`).join(", ") || "none"}`,
    `Attachments: ${attachments.map((item) => item.filename).join(", ")}`
  ].join("\n");
  await transport.sendMail({
    from: values.from,
    to: values.to,
    subject: `Trading Dashboard MTD Snapshot — ${input.snapshot.period.month} through ${input.snapshot.period.asOfDate}`,
    text: body,
    attachments
  });
  return { status: "sent" as const };
}
