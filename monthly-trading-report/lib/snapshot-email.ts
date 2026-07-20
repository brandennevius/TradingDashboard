import type { buildDailyPortfolioSnapshot } from "./daily-portfolio-snapshot";

export type SnapshotEmailTransport = {
  sendMail(message: { from: string; to: string; subject: string; text: string; attachments: Array<{ filename: string; content: string | Buffer; contentType: string }> }): Promise<unknown>;
};

type Snapshot = ReturnType<typeof buildDailyPortfolioSnapshot>;

type SnapshotEmailEnvironment = Record<string, string | undefined>;

export function snapshotEmailConfiguration(environment: SnapshotEmailEnvironment = process.env) {
  const values = {
    host: environment.SNAPSHOT_SMTP_HOST || "",
    port: Number(environment.SNAPSHOT_SMTP_PORT || 0),
    secure: environment.SNAPSHOT_SMTP_SECURE === "true",
    username: environment.SNAPSHOT_SMTP_USERNAME || "",
    password: environment.SNAPSHOT_SMTP_PASSWORD || "",
    from: environment.SNAPSHOT_EMAIL_FROM || "",
    to: environment.SNAPSHOT_EMAIL_TO || ""
  };
  const configured = Boolean(values.host && values.port && values.username && values.password && values.from && values.to);
  return { configured, values };
}

export async function sendDailyPortfolioSnapshotEmail(input: {
  snapshot: Snapshot;
  markdown: string;
  baseName: string;
  transport?: SnapshotEmailTransport;
  environment?: SnapshotEmailEnvironment;
}) {
  const configuration = snapshotEmailConfiguration(input.environment);
  if (!configuration.configured) return { status: "disabled" as const, reason: "SMTP environment variables are incomplete." };
  const { values } = configuration;
  const transport = input.transport || (await import("nodemailer")).default.createTransport({
    host: values.host,
    port: values.port,
    secure: values.secure,
    auth: { user: values.username, pass: values.password }
  });
  const emailSummary = [
    `Requested session: ${input.snapshot.metadata.requested_session}`,
    `Broker import complete: ${input.snapshot.metadata.broker_import_complete}`,
    `Portfolio data timestamp: ${input.snapshot.metadata.portfolio_data_as_of || "unavailable"}`,
    `Position count: ${input.snapshot.open_positions.length}`,
    `Closed-trade count: ${input.snapshot.trades_closed_during_session.length}`,
    `Critical warning count: ${input.snapshot.critical_warning_count}`,
    "",
    input.markdown
  ].join("\n");
  await transport.sendMail({
    from: values.from,
    to: values.to,
    subject: `Trading Dashboard Snapshot — ${input.snapshot.metadata.requested_session}`,
    text: emailSummary,
    attachments: [
      { filename: `${input.baseName}.json`, content: `${JSON.stringify(input.snapshot, null, 2)}\n`, contentType: "application/json" },
      { filename: `${input.baseName}.md`, content: input.markdown, contentType: "text/markdown" }
    ]
  });
  return { status: "sent" as const };
}
