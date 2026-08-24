"use client";

import { useEffect, useRef, useState } from "react";
import { usePathname } from "next/navigation";
import BrandenSidebar from "@/app/components/BrandenSidebar";
import { BrandenSnapshotActionsProvider } from "./BrandenSnapshotActionsContext";
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

type MtdBlockingDiagnostic = {
  code?: string;
  symbol?: string;
  field?: string;
  message?: string;
};

function formatMtdSnapshotValidationError(data: {
  error?: string;
  code?: string;
  diagnostic?: {
    portfolio?: string;
    requested_as_of_date?: string;
    effective_broker_coverage_date?: string | null;
    coverage_gap_days?: number | null;
    snapshot_status?: string;
    weekly_focus?: { status?: string; summary?: string | null };
    validationErrors?: string[];
    blockingDiagnostics?: MtdBlockingDiagnostic[];
  };
}) {
  const lines = [data.code ? `${data.code}: ${data.error || "MTD snapshot validation failed."}` : data.error || "Could not generate the MTD snapshot."];
  const diagnostic = data.diagnostic;
  if (diagnostic?.portfolio) lines.push("", `Portfolio: ${diagnostic.portfolio}`);
  if (diagnostic?.snapshot_status) lines.push(`Status: ${diagnostic.snapshot_status}`);
  if (diagnostic?.requested_as_of_date) lines.push(`Requested as-of date: ${diagnostic.requested_as_of_date}`);
  if (diagnostic?.effective_broker_coverage_date !== undefined) lines.push(`Effective broker coverage date: ${diagnostic.effective_broker_coverage_date || "unavailable"}`);
  if (diagnostic?.coverage_gap_days !== undefined && diagnostic.coverage_gap_days !== null) lines.push(`Coverage gap: ${diagnostic.coverage_gap_days} day${diagnostic.coverage_gap_days === 1 ? "" : "s"}`);
  if (diagnostic?.weekly_focus?.status) lines.push(`Weekly focus: ${diagnostic.weekly_focus.status}${diagnostic.weekly_focus.summary ? ` — ${diagnostic.weekly_focus.summary}` : ""}`);
  if (diagnostic?.blockingDiagnostics?.length) {
    lines.push("", "Blocking data issues:");
    diagnostic.blockingDiagnostics.forEach((item) => {
      const context = [item.symbol, item.field].filter(Boolean).join(" · ");
      lines.push(`- ${item.code || "VALIDATION_ERROR"}${context ? ` (${context})` : ""}: ${item.message || "Required data are unavailable."}`);
    });
  }
  if (diagnostic?.validationErrors?.length) {
    lines.push("", "Schema validation:", ...diagnostic.validationErrors.map((item) => `- ${item}`));
  }
  return lines.join("\n");
}

export default function BrandenJournalLayoutClient({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const [user, setUser] = useState<TraderUser | null>(null);
  const [defaultPortfolio, setDefaultPortfolio] = useState("");
  const [snapshotSession, setSnapshotSession] = useState(() => new Intl.DateTimeFormat("en-CA", { timeZone: "America/New_York" }).format(new Date()));
  const [mtdMonth, setMtdMonth] = useState(() => new Intl.DateTimeFormat("en-CA", { timeZone: "America/New_York", year: "numeric", month: "2-digit" }).format(new Date()));
  const [mtdAsOfDate, setMtdAsOfDate] = useState(() => new Intl.DateTimeFormat("en-CA", { timeZone: "America/New_York" }).format(new Date()));
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
    try {
      const response = await fetch("/api/journal/branden/daily-snapshot", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(buildDailySnapshotRequestBody(session, accountName, sendEmail))
      });
      const data = await response.json().catch(() => ({}));
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
    const selectedMonth = window.prompt("MTD snapshot month (YYYY-MM)", mtdMonth)?.trim();
    if (!selectedMonth) return;
    const selectedAsOfDate = window.prompt("MTD snapshot as-of date (YYYY-MM-DD)", mtdAsOfDate)?.trim();
    if (!selectedAsOfDate) return;
    if (!/^\d{4}-\d{2}$/.test(selectedMonth) || !/^\d{4}-\d{2}-\d{2}$/.test(selectedAsOfDate)) {
      window.alert("Choose a valid month and as-of date.");
      return;
    }
    if (!selectedAsOfDate.startsWith(`${selectedMonth}-`)) {
      window.alert("The as-of date must fall within the selected month.");
      return;
    }
    setMtdMonth(selectedMonth);
    setMtdAsOfDate(selectedAsOfDate);
    const portfolioName = defaultPortfolio.trim();
    if (!portfolioName) {
      window.alert("Select a default portfolio in Portfolio settings first.");
      return;
    }
    setIsGeneratingMtdSnapshot(true);
    try {
      const response = await fetch("/api/journal/branden/mtd-snapshot", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ month: selectedMonth, asOfDate: selectedAsOfDate, portfolioName, sendEmail })
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(formatMtdSnapshotValidationError(data));
      if (sendEmail && data.email?.status !== "sent") throw new Error(data.email?.reason || "The MTD snapshot was generated, but email delivery is not configured.");
      if (!sendEmail) {
        await downloadMtdBundle({ snapshot: data.snapshot, markdown: data.markdown, filenames: data.filenames });
      }
      window.alert(`MTD snapshot generated with status ${data.snapshot.status}.${sendEmail ? " Email sent." : " ZIP downloaded."}`);
    } catch (error) {
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
      <BrandenSnapshotActionsProvider
        value={{
          canGenerateSnapshot,
          isGeneratingDailySnapshot: isGeneratingSnapshot,
          isGeneratingMtdSnapshot,
          generateDailySnapshot: () => generateSnapshot(false),
          generateMtdSnapshot: () => generateMtdSnapshot(false),
          generateAndSendDailySnapshot: () => generateSnapshot(true),
          generateAndSendMtdSnapshot: () => generateMtdSnapshot(true),
          isMtdEmailConfigured: mtdEmailConfigured
        }}
      >
        {children}
      </BrandenSnapshotActionsProvider>
    </main>
  );
}
