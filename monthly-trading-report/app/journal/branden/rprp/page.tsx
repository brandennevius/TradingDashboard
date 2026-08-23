"use client";

import { useEffect, useMemo, useState } from "react";
import RprpSizer from "@/app/components/RprpSizer";
import type { TradeLogEntry, TraderUser } from "@/lib/types";

type PortfolioMeta = {
  currentEquity?: number;
  statementEquity?: number;
  floatingPnl?: number;
  equitySource?: string;
  equityUpdatedAt?: string;
  equityStatementDate?: string;
};

type PortfolioSettingsResponse = {
  portfolios?: string[];
  defaultPortfolio?: string;
  portfolioMeta?: Record<string, PortfolioMeta>;
};

export default function BrandenRprpPage() {
  const [user, setUser] = useState<TraderUser | null>(null);
  const [trades, setTrades] = useState<TradeLogEntry[]>([]);
  const [portfolioMeta, setPortfolioMeta] = useState<Record<string, PortfolioMeta>>({});
  const [activePortfolio, setActivePortfolio] = useState("");
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;

    async function loadPageData() {
      setIsLoading(true);
      setError("");

      const response = await fetch("/api/public/branden-open-positions", { cache: "no-store" });
      const data = (await response.json().catch(() => ({}))) as PortfolioSettingsResponse & {
        error?: string;
        trades?: unknown;
        user?: TraderUser | null;
      };

      if (cancelled) return;

      if (!response.ok) {
        setError(data.error || "Could not load open positions.");
        setIsLoading(false);
        return;
      }

      setUser(data.user || null);
      setTrades(Array.isArray(data.trades) ? data.trades : []);
      setPortfolioMeta(data.portfolioMeta || {});
      setActivePortfolio(String(data.defaultPortfolio || ""));
      setIsLoading(false);
    }

    loadPageData().catch((loadError) => {
      if (!cancelled) {
        setError(loadError instanceof Error ? loadError.message : "Could not load RPRP.");
        setIsLoading(false);
      }
    });

    return () => {
      cancelled = true;
    };
  }, []);

  const brandenTrades = useMemo(() => trades.filter((trade) => trade.userId === "branden" && !trade.hidden), [trades]);

  return (
    <div className="branden-journal-content">
        {error ? <p className="status error">{error}</p> : null}
        {isLoading ? <p className="status">Loading RPRP portfolio data...</p> : null}
        {!isLoading && !error ? <RprpSizer trades={brandenTrades} activePortfolio={activePortfolio} portfolioMeta={portfolioMeta} /> : null}
      </div>
  );
}
