"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import OpenHeatDashboard from "@/app/components/OpenHeatDashboard";
import type { TradeLogEntry, TraderUser } from "@/lib/types";

type PortfolioSettingsResponse = {
  portfolios?: string[];
  defaultPortfolio?: string;
  portfolioMeta?: Record<string, PortfolioMeta>;
};

type PortfolioMeta = {
  currentEquity?: number;
  statementEquity?: number;
  floatingPnl?: number;
  equitySource?: string;
  equityUpdatedAt?: string;
  equityStatementDate?: string;
};

export default function BrandenOpenPositionsPage() {
  const [user, setUser] = useState<TraderUser | null>(null);
  const [trades, setTrades] = useState<TradeLogEntry[]>([]);
  const [defaultPortfolio, setDefaultPortfolio] = useState("");
  const [portfolioMeta, setPortfolioMeta] = useState<Record<string, PortfolioMeta>>({});
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

      const tradesResponse = await fetch("/api/public/branden-open-positions", { cache: "no-store" });
      const publicData = (await tradesResponse.json().catch(() => ({}))) as PortfolioSettingsResponse & {
        error?: string;
        trades?: unknown;
        user?: TraderUser | null;
      };
      const tradesData = publicData;
      const portfoliosData = publicData;

      if (cancelled) {
        return;
      }

      if (!tradesResponse.ok) {
        setError(tradesData.error || "Could not load trades.");
        setIsLoading(false);
        return;
      }

      const nextDefaultPortfolio = String(portfoliosData.defaultPortfolio || "");
      setUser(publicData.user || null);
      setTrades(Array.isArray(tradesData.trades) ? tradesData.trades : []);
      setDefaultPortfolio(nextDefaultPortfolio);
      setPortfolioMeta(portfoliosData.portfolioMeta || {});
      setActivePortfolio(nextDefaultPortfolio);
      setIsLoading(false);
    }

    loadPageData().catch((loadError) => {
      if (!cancelled) {
        setError(loadError instanceof Error ? loadError.message : "Could not load open positions.");
        setIsLoading(false);
      }
    });

    return () => {
      cancelled = true;
    };
  }, []);

  const brandenTrades = useMemo(
    () => trades.filter((trade) => trade.userId === "branden" && !trade.hidden),
    [trades]
  );
  const openPositionCount = useMemo(
    () =>
      brandenTrades.filter((trade) => trade.status === "OPEN" && (!activePortfolio || trade.portfolioTag === activePortfolio)).length,
    [activePortfolio, brandenTrades]
  );
  const openTradeDetail = (tradeId: string) => {
    window.location.href = `/journal/branden/dashboard?tradeId=${encodeURIComponent(tradeId)}`;
  };
  const canEditBrandenJournal = user?.id === "branden" && !user.readOnly;
  const sidebarActions = canEditBrandenJournal
    ? [
        {
          key: "import-broker-statement",
          label: "Import broker statement",
          icon: "I",
          disabled: isImporting,
          onClick: async () => {
            const targetPortfolio = await choosePortfolioForImport();
            if (targetPortfolio) {
              cfImportInputRef.current?.click();
            }
          }
        }
      ]
    : [];

  async function choosePortfolioForImport() {
    const defaultValue = activePortfolio.trim() || defaultPortfolio.trim();
    const targetPortfolio = window.prompt("Portfolio for broker statement import", defaultValue);

    if (targetPortfolio === null) {
      return "";
    }

    const normalized = targetPortfolio.trim();

    if (!normalized) {
      setStatus("Choose a portfolio before importing a broker statement.");
      return "";
    }

    pendingImportPortfolioRef.current = normalized;
    return normalized;
  }

  async function importBrokerStatement(files: FileList | null) {
    if (!files?.length || !canEditBrandenJournal) {
      return;
    }

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
      const response = await fetch("/api/import/cf-statement", {
        method: "POST",
        body: formData
      });
      const data = await response.json().catch(() => ({}));

      if (!response.ok) {
        setStatus(data.error || "Could not import broker statement.");
        return;
      }

      setStatus(
        `Broker statement imported. Open: ${data.openTrades}. Closed: ${data.closedTrades}. Updated: ${data.updated}.`
      );

      const tradesResponse = await fetch("/api/public/branden-open-positions", { cache: "no-store" });
      const portfoliosData = (await tradesResponse.json().catch(() => ({}))) as PortfolioSettingsResponse & {
        trades?: unknown;
      };

      if (tradesResponse.ok) {
        setTrades(Array.isArray(portfoliosData.trades) ? portfoliosData.trades : []);
        setDefaultPortfolio(String(portfoliosData.defaultPortfolio || ""));
        setPortfolioMeta(portfoliosData.portfolioMeta || {});
      }

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
            <h1>Open Positions</h1>
            <span>{openPositionCount} open {openPositionCount === 1 ? "position" : "positions"}</span>
          </div>
        </header>

        {status ? <p className="status trade-log-status">{status}</p> : null}
        {error ? <p className="status error">{error}</p> : null}
        {isLoading ? <p className="status">Loading open positions...</p> : null}

        {!isLoading && !error ? (
          <div className="branden-open-positions-stack">
            <OpenHeatDashboard
              trades={brandenTrades}
              activePortfolio={activePortfolio}
              onSelectTrade={user ? openTradeDetail : undefined}
              portfolioMeta={portfolioMeta}
            />
          </div>
        ) : null}
      </div>
  );
}
