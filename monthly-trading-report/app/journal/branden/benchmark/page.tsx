"use client";

import Link from "next/link";
import { useEffect, useMemo, useRef, useState } from "react";
import TradeBenchmarkAnalyzer from "@/app/components/TradeBenchmarkAnalyzer";
import type { TradeLogEntry, TraderUser } from "@/lib/types";

type PortfolioSettingsResponse = {
  portfolios?: string[];
  defaultPortfolio?: string;
};

export default function BrandenBenchmarkPage() {
  const [user, setUser] = useState<TraderUser | null>(null);
  const [trades, setTrades] = useState<TradeLogEntry[]>([]);
  const [defaultPortfolio, setDefaultPortfolio] = useState("");
  const [activePortfolio, setActivePortfolio] = useState("");
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState("");
  const [status, setStatus] = useState("");
  const [isImporting, setIsImporting] = useState(false);
  const cfImportInputRef = useRef<HTMLInputElement | null>(null);
  const pendingImportPortfolioRef = useRef("");

  useEffect(() => {
    let cancelled = false;

    async function loadPageData() {
      setIsLoading(true);
      setError("");

      const benchmarkResponse = await fetch("/api/journal/branden/benchmark", { cache: "no-store" });
      const benchmarkData = await benchmarkResponse.json().catch(() => ({}));

      if (cancelled) return;

      if (!benchmarkResponse.ok || !benchmarkData.user) {
        setError(benchmarkData.error || "Sign in to view Benchmark.");
        setIsLoading(false);
        return;
      }

      setUser(benchmarkData.user || null);
      setTrades(Array.isArray(benchmarkData.trades) ? benchmarkData.trades : []);
      setDefaultPortfolio(String(benchmarkData.defaultPortfolio || ""));
      setActivePortfolio(String(benchmarkData.defaultPortfolio || ""));
      setIsLoading(false);
    }

    loadPageData().catch((loadError) => {
      if (!cancelled) {
        setError(loadError instanceof Error ? loadError.message : "Could not load benchmark.");
        setIsLoading(false);
      }
    });

    return () => {
      cancelled = true;
    };
  }, []);

  const canEditBrandenJournal = user?.id === "branden" && !user.readOnly;
  const brandenTrades = useMemo(() => trades.filter((trade) => trade.userId === "branden" && !trade.hidden), [trades]);
  const sidebarActions = canEditBrandenJournal
    ? [
        {
          key: "import-broker-statement",
          label: "Import broker statement",
          icon: "I",
          disabled: isImporting,
          onClick: async () => {
            const targetPortfolio = await choosePortfolioForImport();
            if (targetPortfolio) cfImportInputRef.current?.click();
          }
        }
      ]
    : [];

  async function choosePortfolioForImport() {
    const defaultValue = activePortfolio.trim() || defaultPortfolio.trim();
    const targetPortfolio = window.prompt("Portfolio for broker statement import", defaultValue);
    if (targetPortfolio === null) return "";
    const normalized = targetPortfolio.trim();
    if (!normalized) {
      setStatus("Choose a portfolio before importing a broker statement.");
      return "";
    }
    pendingImportPortfolioRef.current = normalized;
    return normalized;
  }

  async function importBrokerStatement(files: FileList | null) {
    if (!files?.length || !canEditBrandenJournal) return;
    const targetPortfolio = pendingImportPortfolioRef.current.trim() || activePortfolio.trim();
    if (!targetPortfolio) {
      setStatus("Choose a portfolio before importing a broker statement.");
      return;
    }

    setIsImporting(true);
    setStatus("Importing broker statement...");
    const formData = new FormData();
    formData.append("file", files[0]);
    formData.append("portfolioTag", targetPortfolio);

    try {
      const response = await fetch("/api/import/cf-statement", { method: "POST", body: formData });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        setStatus(data.error || "Could not import broker statement.");
        return;
      }
      setStatus(`Broker statement imported. Open: ${data.openTrades}. Closed: ${data.closedTrades}. Updated: ${data.updated}.`);
      const tradesResponse = await fetch("/api/journal/branden/benchmark", { cache: "no-store" });
      const tradesData = await tradesResponse.json().catch(() => ({}));
      if (tradesResponse.ok) setTrades(Array.isArray(tradesData.trades) ? tradesData.trades : []);
      setActivePortfolio(targetPortfolio);
    } finally {
      pendingImportPortfolioRef.current = "";
      setIsImporting(false);
    }
  }

  return (
    <div className="branden-journal-content">
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

        <header className="branden-route-header">
          <div>
            <p className="eyebrow">Branden journal</p>
            <h1>Benchmark</h1>
            <span>Closed trades compared with SPY</span>
          </div>
          <nav className="branden-route-nav" aria-label="Branden journal pages">
            <Link href="/journal/branden/dashboard">Dashboard</Link>
            <Link href="/journal/branden/trade-log">Trade Log</Link>
            <Link className="active" href="/journal/branden/benchmark">Benchmark</Link>
          </nav>
        </header>

        {status ? <p className="status trade-log-status">{status}</p> : null}
        {error ? <p className="status error">{error}</p> : null}
        {isLoading ? <p className="status">Loading benchmark...</p> : null}
        {!isLoading && !error ? <TradeBenchmarkAnalyzer trades={brandenTrades} activePortfolio={activePortfolio} /> : null}
      </div>
  );
}
