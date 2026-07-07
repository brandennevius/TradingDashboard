"use client";

import { useEffect, useRef, useState } from "react";
import { usePathname } from "next/navigation";
import BrandenSidebar from "@/app/components/BrandenSidebar";
import type { TraderUser } from "@/lib/types";

type PortfolioSettingsResponse = {
  defaultPortfolio?: string;
};

export default function BrandenJournalLayoutClient({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const [user, setUser] = useState<TraderUser | null>(null);
  const [defaultPortfolio, setDefaultPortfolio] = useState("");
  const [isImporting, setIsImporting] = useState(false);
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

  const accountActions = canEditBrandenJournal
    ? [
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
      ]
    : [];

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
