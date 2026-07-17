"use client";

import { useEffect, useRef, useState } from "react";
import { usePathname } from "next/navigation";
import BrandenSidebar from "@/app/components/BrandenSidebar";
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

function formatSnapshotValidationError(data: { error?: string; codes?: string[]; diagnostic?: SnapshotValidationDiagnostic }) {
  const diagnostic = data.diagnostic;
  if (!diagnostic) return data.error || "Could not generate daily snapshot.";
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
  if (diagnostic.latestBrokerImportTimestamp) lines.push("", `Latest broker import: ${diagnostic.latestBrokerImportTimestamp}`);
  if (diagnostic.latestStatementCoverageDate) lines.push(`Statement coverage date: ${diagnostic.latestStatementCoverageDate}`);
  return lines.join("\n");
}

export default function BrandenJournalLayoutClient({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const [user, setUser] = useState<TraderUser | null>(null);
  const [defaultPortfolio, setDefaultPortfolio] = useState("");
  const [isImporting, setIsImporting] = useState(false);
  const [isGeneratingSnapshot, setIsGeneratingSnapshot] = useState(false);
  const cfImportInputRef = useRef<HTMLInputElement | null>(null);
  const pendingImportPortfolioRef = useRef("");

  useEffect(() => {
    let cancelled = false;

    async function loadSidebarContext() {
      const [sessionResponse, portfolioResponse] = await Promise.all([
        fetch("/api/session", { cache: "no-store" }),
        fetch("/api/settings/branden-portfolios", { cache: "no-store" })
      ]);
      const sessionData = await sessionResponse.json().catch(() => ({}));
      const portfolioData = (await portfolioResponse.json().catch(() => ({}))) as PortfolioSettingsResponse;

      if (cancelled) return;

      setUser(sessionResponse.ok ? sessionData.user || null : null);
      setDefaultPortfolio(portfolioResponse.ok ? String(portfolioData.defaultPortfolio || "") : "");
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
    const today = new Intl.DateTimeFormat("en-CA", { timeZone: "America/New_York" }).format(new Date());
    const session = window.prompt("Completed U.S. market session (YYYY-MM-DD)", today)?.trim();
    if (!session) return;
    const accountName = window.prompt("Portfolio", defaultPortfolio.trim())?.trim();
    if (!accountName) return;
    setIsGeneratingSnapshot(true);
    try {
      const response = await fetch("/api/journal/branden/daily-snapshot", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session, accountName, sendEmail })
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(formatSnapshotValidationError(data));
      const JSZip = (await import("jszip")).default;
      const archive = new JSZip();
      archive.file(data.filenames.json, `${JSON.stringify(data.snapshot, null, 2)}\n`);
      archive.file(data.filenames.markdown, data.markdown);
      const bundle = await archive.generateAsync({ type: "blob" });
      downloadBlob(data.filenames.json.replace(/\.json$/i, ".zip"), bundle);
      const emailMessage = sendEmail ? ` Email: ${data.email?.status || "unknown"}.` : "";
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
        key: "generate-daily-snapshot",
        label: isGeneratingSnapshot ? "Generating snapshot..." : "Generate Daily Snapshot",
        icon: "S",
        disabled: isGeneratingSnapshot,
        onClick: () => generateSnapshot(false)
      },
      {
        key: "generate-and-send-daily-snapshot",
        label: isGeneratingSnapshot ? "Generating snapshot..." : "Generate and Send Daily Snapshot",
        icon: "E",
        disabled: isGeneratingSnapshot,
        onClick: () => generateSnapshot(true)
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
      {children}
    </main>
  );
}
