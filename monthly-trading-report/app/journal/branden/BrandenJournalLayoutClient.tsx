"use client";

import { useEffect, useRef, useState } from "react";
import { usePathname } from "next/navigation";
import BrandenSidebar from "@/app/components/BrandenSidebar";
import { buildDailySnapshotRequestBody } from "@/lib/daily-portfolio-snapshot-request";
import type { TraderUser } from "@/lib/types";

type PortfolioSettingsResponse = {
  defaultPortfolio?: string;
};

type SnapshotValidationDiagnostic = {
  requestedSession: string;
  portfolio: string;
  latestBrokerImportTimestamp: string | null;
  latestStatementCoverageDate: string | null;
  totalImportedTradeCount: number;
  needsReviewCount: number;
  missingExecutionsCount: number;
  validationCodes: string[];
  samples: Record<string, Array<{ ticker: string; tradeId: string }> | undefined>;
  needsReviewRows: Array<{
    ticker: string;
    tradeId: string;
    entryDate: string;
    exitDate: string | null;
    status: "OPEN" | "CLOSED";
    affectsRequestedSnapshot: boolean;
    blockingReason: string | null;
  }>;
};

type SnapshotSessionDiagnostic = {
  selectedSession: string;
  submittedSession: string;
  currentNewYorkDateTime: string;
  latestCompletedSession: string;
  regularSessionCompletionTime: string;
  validationCodes: ["SNAPSHOT_SESSION_NOT_COMPLETE"];
};

function formatSnapshotValidationError(data: { error?: string; codes?: string[]; diagnostic?: SnapshotValidationDiagnostic | SnapshotSessionDiagnostic }) {
  const diagnostic = data.diagnostic;
  if (!diagnostic) return data.error || "Could not generate daily snapshot.";
  if ("selectedSession" in diagnostic) {
    return [
      "Snapshot not generated.",
      "",
      `Selected session: ${diagnostic.selectedSession}`,
      `Date submitted: ${diagnostic.submittedSession}`,
      `Current New York time: ${diagnostic.currentNewYorkDateTime}`,
      `Latest completed session: ${diagnostic.latestCompletedSession}`,
      `Regular-session completion time: ${diagnostic.regularSessionCompletionTime}`,
      "",
      "Blocking validation:",
      "SNAPSHOT_SESSION_NOT_COMPLETE",
      "",
      "The selected market session has not completed. Generate the snapshot after the broker import is complete following today’s market close."
    ].join("\n");
  }
  const lines = [
    "Snapshot not generated.",
    "",
    `Portfolio: ${diagnostic.portfolio}`,
    `Requested session: ${diagnostic.requestedSession}`,
    "",
    "Blocking validation:",
    ...diagnostic.validationCodes
  ];
  for (const code of diagnostic.validationCodes) {
    const samples = diagnostic.samples[code] || [];
    if (code === "BROKER_IMPORT_NEEDS_REVIEW") lines.push("", `${diagnostic.needsReviewCount} imported rows still require review:`);
    if (code === "BROKER_IMPORT_MISSING_EXECUTIONS") lines.push("", `${diagnostic.missingExecutionsCount} imported trades have no execution records:`);
    if (samples.length) lines.push(...samples.map((sample) => `- ${sample.ticker} (${sample.tradeId})`));
  }
  if (diagnostic.needsReviewRows.length) {
    lines.push("", "Needs review row diagnostics:");
    lines.push(...diagnostic.needsReviewRows.map((row) =>
      `- ${row.ticker} (${row.tradeId}): ${row.status}, entry ${row.entryDate}, exit ${row.exitDate || "open"}; ${row.affectsRequestedSnapshot ? `affects snapshot — ${row.blockingReason}` : "unrelated to requested snapshot"}`
    ));
  }
  if (diagnostic.latestBrokerImportTimestamp) lines.push("", `Latest broker import: ${diagnostic.latestBrokerImportTimestamp}`);
  if (diagnostic.latestStatementCoverageDate) lines.push(`Statement coverage date: ${diagnostic.latestStatementCoverageDate}`);
  return lines.join("\n");
}

export default function BrandenJournalLayoutClient({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const [user, setUser] = useState<TraderUser | null>(null);
  const [defaultPortfolio, setDefaultPortfolio] = useState("");
  const [snapshotSession, setSnapshotSession] = useState(() => new Intl.DateTimeFormat("en-CA", { timeZone: "America/New_York" }).format(new Date()));
  const [lastSubmittedSnapshotSession, setLastSubmittedSnapshotSession] = useState("");
  const [latestCompletedSnapshotSession, setLatestCompletedSnapshotSession] = useState("");
  const [mtdMonth, setMtdMonth] = useState(() => new Intl.DateTimeFormat("en-CA", { timeZone: "America/New_York", year: "numeric", month: "2-digit" }).format(new Date()));
  const [mtdAsOfDate, setMtdAsOfDate] = useState(() => new Intl.DateTimeFormat("en-CA", { timeZone: "America/New_York" }).format(new Date()));
  const [mtdStatus, setMtdStatus] = useState("Not generated");
  const [mtdWarningCount, setMtdWarningCount] = useState<number | null>(null);
  const [mtdFilenames, setMtdFilenames] = useState<string[]>([]);
  const [lastMtdDownload, setLastMtdDownload] = useState<{ snapshot: unknown; markdown: string; filenames: { json: string; markdown: string; zip: string } } | null>(null);
  const [mtdEmailConfigured, setMtdEmailConfigured] = useState(false);
  const [isImporting, setIsImporting] = useState(false);
  const [isGeneratingSnapshot, setIsGeneratingSnapshot] = useState(false);
  const [isGeneratingMtdSnapshot, setIsGeneratingMtdSnapshot] = useState(false);
  const cfImportInputRef = useRef<HTMLInputElement | null>(null);
  const pendingImportPortfolioRef = useRef("");

  useEffect(() => {
    let cancelled = false;

    async function loadSidebarContext() {
      const [sessionResponse, portfolioResponse, mtdConfigResponse] = await Promise.all([
        fetch("/api/session", { cache: "no-store" }),
        fetch("/api/settings/branden-portfolios", { cache: "no-store" }),
        fetch("/api/journal/branden/mtd-snapshot", { cache: "no-store" })
      ]);
      const sessionData = await sessionResponse.json().catch(() => ({}));
      const portfolioData = (await portfolioResponse.json().catch(() => ({}))) as PortfolioSettingsResponse;
      const mtdConfigData = await mtdConfigResponse.json().catch(() => ({}));

      if (cancelled) return;

      setUser(sessionResponse.ok ? sessionData.user || null : null);
      setDefaultPortfolio(portfolioResponse.ok ? String(portfolioData.defaultPortfolio || "") : "");
      setMtdEmailConfigured(Boolean(mtdConfigResponse.ok && mtdConfigData.emailConfigured));
    }

    loadSidebarContext().catch(() => {
      if (!cancelled) {
        setUser(null);
        setDefaultPortfolio("");
      }
    });

    return () => {
      cancelled = true;
    };
  }, []);

  const canEditBrandenJournal = user?.id === "branden" && !user.readOnly;
  const canGenerateSnapshot = user?.id === "branden";

  function downloadBlob(filename: string, blob: Blob) {
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    link.click();
    URL.revokeObjectURL(url);
  }

  async function generateSnapshot(sendEmail: boolean) {
    if (!canGenerateSnapshot) return;
    const selectedSession = window.prompt("Snapshot session (YYYY-MM-DD)", snapshotSession.trim());
    if (selectedSession === null) return;
    const session = selectedSession.trim();
    if (!session) return;
    setSnapshotSession(session);
    const accountName = window.prompt("Portfolio", defaultPortfolio.trim())?.trim();
    if (!accountName) return;
    setIsGeneratingSnapshot(true);
    setLastSubmittedSnapshotSession(session);
    try {
      const response = await fetch("/api/journal/branden/daily-snapshot", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(buildDailySnapshotRequestBody(session, accountName, sendEmail))
      });
      const data = await response.json().catch(() => ({}));
      const reportedLatestSession = data.datePath?.latestCompletedSession || data.diagnostic?.latestCompletedSession || "";
      if (reportedLatestSession) setLatestCompletedSnapshotSession(reportedLatestSession);
      if (!response.ok) throw new Error(formatSnapshotValidationError(data));
      if (sendEmail && data.email?.status !== "sent") {
        throw new Error(data.email?.reason || "The snapshot was generated, but email delivery is not configured.");
      }
      if (!sendEmail) {
        const JSZip = (await import("jszip")).default;
        const archive = new JSZip();
        archive.file(data.filenames.json, `${JSON.stringify(data.snapshot, null, 2)}\n`);
        archive.file(data.filenames.markdown, data.markdown);
        const bundle = await archive.generateAsync({ type: "blob" });
        downloadBlob(data.filenames.json.replace(/\.json$/i, ".zip"), bundle);
      }
      const emailMessage = sendEmail ? " Email sent." : " Snapshot downloaded.";
      const unrelatedReviewRows = (data.brokerDiagnostic?.needsReviewRows || []).filter((row: SnapshotValidationDiagnostic["needsReviewRows"][number]) => !row.affectsRequestedSnapshot);
      const reviewMessage = unrelatedReviewRows.length
        ? ` ${unrelatedReviewRows.length} unrelated broker-import row${unrelatedReviewRows.length === 1 ? "" : "s"} still need review:\n${unrelatedReviewRows.map((row: SnapshotValidationDiagnostic["needsReviewRows"][number]) => `- ${row.ticker} (${row.tradeId}): ${row.status}, entry ${row.entryDate}, exit ${row.exitDate || "open"}; unrelated to this snapshot`).join("\n")}`
        : "";
      window.alert(`Daily snapshot generated with status ${data.snapshot.snapshot_status}.${reviewMessage}${emailMessage}`);
    } catch (error) {
      window.alert(error instanceof Error ? error.message : "Could not generate daily snapshot.");
    } finally {
      setIsGeneratingSnapshot(false);
    }
  }

  async function generateMtdSnapshot(sendEmail: boolean) {
    if (!canGenerateSnapshot || isGeneratingMtdSnapshot) return;
    if (!/^\d{4}-\d{2}$/.test(mtdMonth) || !/^\d{4}-\d{2}-\d{2}$/.test(mtdAsOfDate)) {
      window.alert("Choose a valid month and as-of date.");
      return;
    }
    if (!mtdAsOfDate.startsWith(`${mtdMonth}-`)) {
      window.alert("The as-of date must fall within the selected month.");
      return;
    }
    const portfolioName = defaultPortfolio.trim();
    if (!portfolioName) {
      window.alert("Select a default portfolio in Portfolio settings first.");
      return;
    }
    setIsGeneratingMtdSnapshot(true);
    setMtdStatus("Generating");
    setMtdWarningCount(null);
    setMtdFilenames([]);
    setLastMtdDownload(null);
    try {
      const response = await fetch("/api/journal/branden/mtd-snapshot", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ month: mtdMonth, asOfDate: mtdAsOfDate, portfolioName, sendEmail })
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(`${data.code ? `${data.code}: ` : ""}${data.error || "Could not generate the MTD snapshot."}`);
      if (sendEmail && data.email?.status !== "sent") throw new Error(data.email?.reason || "The MTD snapshot was generated, but email delivery is not configured.");
      const filenames = [data.filenames?.json, data.filenames?.markdown].filter(Boolean);
      setMtdStatus(data.snapshot.status);
      setMtdWarningCount(data.snapshot.diagnostics?.length || 0);
      setMtdFilenames(filenames);
      setLastMtdDownload({ snapshot: data.snapshot, markdown: data.markdown, filenames: data.filenames });
      if (!sendEmail) {
        await downloadMtdBundle({ snapshot: data.snapshot, markdown: data.markdown, filenames: data.filenames });
      }
      window.alert(`MTD snapshot generated with status ${data.snapshot.status}.${sendEmail ? " Email sent." : " ZIP downloaded."}`);
    } catch (error) {
      setMtdStatus("BLOCKED");
      window.alert(error instanceof Error ? error.message : "Could not generate the MTD snapshot.");
    } finally {
      setIsGeneratingMtdSnapshot(false);
    }
  }

  async function downloadMtdBundle(payload: { snapshot: unknown; markdown: string; filenames: { json: string; markdown: string; zip: string } }) {
    const JSZip = (await import("jszip")).default;
    const archive = new JSZip();
    archive.file(payload.filenames.json, `${JSON.stringify(payload.snapshot, null, 2)}\n`);
    archive.file(payload.filenames.markdown, payload.markdown);
    const bundle = await archive.generateAsync({ type: "blob" });
    downloadBlob(payload.filenames.zip, bundle);
  }

  async function choosePortfolioForImport() {
    const targetPortfolio = window.prompt("Portfolio for broker statement import", defaultPortfolio.trim());
    if (targetPortfolio === null) return "";
    const normalized = targetPortfolio.trim();
    if (!normalized) {
      window.alert("Choose a portfolio before importing a broker statement.");
      return "";
    }
    pendingImportPortfolioRef.current = normalized;
    return normalized;
  }

  async function importBrokerStatement(files: FileList | null) {
    if (!files?.length || !canEditBrandenJournal) return;
    const targetPortfolio = pendingImportPortfolioRef.current.trim() || defaultPortfolio.trim();
    if (!targetPortfolio) {
      window.alert("Choose a portfolio before importing a broker statement.");
      return;
    }

    setIsImporting(true);
    const formData = new FormData();
    formData.append("file", files[0]);
    formData.append("portfolioTag", targetPortfolio);

    try {
      const response = await fetch("/api/import/cf-statement", { method: "POST", body: formData });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        window.alert(data.error || "Could not import broker statement.");
        return;
      }

      window.alert(`Broker statement imported. Open: ${data.openTrades}. Closed: ${data.closedTrades}. Updated: ${data.updated}.`);
      window.location.reload();
    } finally {
      pendingImportPortfolioRef.current = "";
      setIsImporting(false);
    }
  }

  const accountActions = [
    ...(canGenerateSnapshot ? [
      {
        key: "generate-mtd-snapshot",
        label: isGeneratingMtdSnapshot ? "Generating MTD snapshot..." : "Generate MTD Snapshot",
        icon: "M",
        disabled: isGeneratingMtdSnapshot,
        onClick: () => generateMtdSnapshot(false)
      },
      {
        key: "generate-send-mtd-snapshot",
        label: "Generate and Send MTD Snapshot",
        icon: "E",
        disabled: isGeneratingMtdSnapshot || !mtdEmailConfigured,
        onClick: () => generateMtdSnapshot(true)
      },
      {
        key: "generate-send-daily-snapshot",
        label: isGeneratingSnapshot ? "Generating snapshot..." : "Generate and Send Daily Snapshot",
        icon: "S",
        disabled: isGeneratingSnapshot,
        onClick: () => generateSnapshot(true)
      },
      {
        key: "download-daily-snapshot",
        label: "Download Snapshot Only",
        icon: "D",
        disabled: isGeneratingSnapshot,
        onClick: () => generateSnapshot(false)
      }
    ] : []),
    ...(canEditBrandenJournal ? [
        {
          key: "import-broker-statement",
          label: isImporting ? "Importing statement..." : "Import broker statement",
          icon: "I",
          disabled: isImporting,
          onClick: async () => {
            const targetPortfolio = await choosePortfolioForImport();
            if (targetPortfolio) cfImportInputRef.current?.click();
          }
        }
      ] : [])
  ];

  return (
    <main className="trade-log-shell branden-journal-shell branden-route-shell sidebar-expanded">
      <input
        ref={cfImportInputRef}
        className="trade-file-input"
        type="file"
        accept="application/pdf,.pdf"
        disabled={isImporting || !canEditBrandenJournal}
        onChange={(event) => {
          importBrokerStatement(event.target.files);
          event.currentTarget.value = "";
        }}
      />
      <BrandenSidebar activeHref={pathname || "/journal/branden/dashboard"} accountActions={accountActions} />
      {canGenerateSnapshot ? (
        <section
          aria-label="Month-to-date snapshot controls"
          style={{ position: "fixed", right: 18, bottom: 168, zIndex: 40, width: 300, padding: 12, border: "1px solid rgba(74,222,128,.35)", borderRadius: 10, background: "rgba(15,23,42,.96)", color: "#e2e8f0", boxShadow: "0 12px 32px rgba(0,0,0,.3)" }}
        >
          <strong style={{ display: "block", fontSize: 12, marginBottom: 8 }}>Month-to-Date Snapshot</strong>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
            <label style={{ display: "grid", gap: 4, fontSize: 11 }}>
              Month
              <input type="month" value={mtdMonth} onChange={(event) => setMtdMonth(event.currentTarget.value)} style={{ width: "100%", padding: "6px 7px", borderRadius: 6 }} />
            </label>
            <label style={{ display: "grid", gap: 4, fontSize: 11 }}>
              As-of date
              <input type="date" value={mtdAsOfDate} onChange={(event) => setMtdAsOfDate(event.currentTarget.value)} style={{ width: "100%", padding: "6px 7px", borderRadius: 6 }} />
            </label>
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6, marginTop: 8 }}>
            <button type="button" disabled={isGeneratingMtdSnapshot} onClick={() => generateMtdSnapshot(false)} style={{ padding: "7px", borderRadius: 6 }}>Generate</button>
            <button type="button" disabled={isGeneratingMtdSnapshot || !mtdEmailConfigured} onClick={() => generateMtdSnapshot(true)} style={{ padding: "7px", borderRadius: 6 }}>{mtdEmailConfigured ? "Generate + Send" : "Email unavailable"}</button>
          </div>
          <div style={{ display: "grid", gap: 2, marginTop: 8, fontSize: 10, color: "#94a3b8" }}>
            <span>Portfolio: {defaultPortfolio || "not selected"}</span>
            <span>Status: {mtdStatus}{mtdWarningCount === null ? "" : ` · ${mtdWarningCount} warnings`}</span>
            {mtdFilenames.map((filename) => <span key={filename}>{filename}</span>)}
          </div>
          {lastMtdDownload ? <button type="button" onClick={() => downloadMtdBundle(lastMtdDownload)} style={{ width: "100%", marginTop: 7, padding: "6px", borderRadius: 6 }}>Download generated ZIP</button> : null}
        </section>
      ) : null}
      {canGenerateSnapshot ? (
        <section
          aria-label="Daily snapshot date selection"
          style={{ position: "fixed", right: 18, bottom: 18, zIndex: 40, width: 270, padding: 12, border: "1px solid rgba(148,163,184,.35)", borderRadius: 10, background: "rgba(15,23,42,.96)", color: "#e2e8f0", boxShadow: "0 12px 32px rgba(0,0,0,.3)" }}
        >
          <label style={{ display: "grid", gap: 6, fontSize: 12 }}>
            <strong>Daily snapshot date</strong>
            <input
              type="date"
              value={snapshotSession}
              onChange={(event) => setSnapshotSession(event.currentTarget.value)}
              style={{ width: "100%", padding: "7px 8px", borderRadius: 6 }}
            />
          </label>
          <div style={{ display: "grid", gap: 3, marginTop: 8, fontSize: 11, color: "#94a3b8" }}>
            <span>Selected date: {snapshotSession || "—"}</span>
            <span>Date submitted: {lastSubmittedSnapshotSession || "—"}</span>
            <span>Latest completed session: {latestCompletedSnapshotSession || "shown after submission"}</span>
          </div>
        </section>
      ) : null}
      {children}
    </main>
  );
}
