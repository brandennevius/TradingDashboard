"use client";

import { useEffect, useMemo, useState } from "react";
import OpenHeatDashboard from "@/app/components/OpenHeatDashboard";
import type { BrokerPortfolioSnapshot } from "@/lib/broker-portfolio-snapshot";
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
  const [brokerPortfolioSnapshots, setBrokerPortfolioSnapshots] = useState<BrokerPortfolioSnapshot[]>([]);
  const [activePortfolio, setActivePortfolio] = useState("");
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;

    async function loadPageData() {
      setIsLoading(true);
      setError("");

      const [tradesResponse, snapshotsResponse] = await Promise.all([
        fetch("/api/public/branden-open-positions", { cache: "no-store" }),
        fetch("/api/journal/branden/daily-review", { cache: "no-store" })
      ]);
      const publicData = (await tradesResponse.json().catch(() => ({}))) as PortfolioSettingsResponse & {
        error?: string;
        trades?: unknown;
        user?: TraderUser | null;
      };
      const snapshotsData = (await snapshotsResponse.json().catch(() => ({}))) as {
        brokerPortfolioSnapshots?: BrokerPortfolioSnapshot[];
        trades?: TradeLogEntry[];
        user?: TraderUser | null;
        defaultPortfolio?: string;
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

      const nextDefaultPortfolio = String(
        (snapshotsResponse.ok ? snapshotsData.defaultPortfolio : "") || portfoliosData.defaultPortfolio || ""
      );
      setUser((snapshotsResponse.ok ? snapshotsData.user : null) || publicData.user || null);
      setTrades(
        snapshotsResponse.ok && Array.isArray(snapshotsData.trades)
          ? snapshotsData.trades
          : Array.isArray(tradesData.trades)
            ? tradesData.trades
            : []
      );
      setDefaultPortfolio(nextDefaultPortfolio);
      setPortfolioMeta(portfoliosData.portfolioMeta || {});
      setBrokerPortfolioSnapshots(
        snapshotsResponse.ok && Array.isArray(snapshotsData.brokerPortfolioSnapshots)
          ? snapshotsData.brokerPortfolioSnapshots
          : []
      );
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
  return (
    <div className="branden-journal-content">
        <header className="branden-route-header">
          <div>
            <p className="eyebrow">Branden journal</p>
            <h1>Open Positions</h1>
            <span>{openPositionCount} open {openPositionCount === 1 ? "position" : "positions"}</span>
          </div>
        </header>

        {error ? <p className="status error">{error}</p> : null}
        {isLoading ? <p className="status">Loading open positions...</p> : null}

        {!isLoading && !error ? (
          <div className="branden-open-positions-stack">
            <OpenHeatDashboard
              trades={brandenTrades}
              activePortfolio={activePortfolio}
              onSelectTrade={user ? openTradeDetail : undefined}
              portfolioMeta={portfolioMeta}
              brokerPortfolioSnapshots={brokerPortfolioSnapshots}
            />
          </div>
        ) : null}
      </div>
  );
}
