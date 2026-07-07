"use client";

import { useEffect, useMemo, useState } from "react";
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

function money(value: number) {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0
  }).format(Number.isFinite(value) ? value : 0);
}

function sortedUnique(values: string[]) {
  return Array.from(new Set(values.map((value) => value.trim()).filter(Boolean))).sort((a, b) => a.localeCompare(b));
}

export default function BrandenPortfoliosPage() {
  const [user, setUser] = useState<TraderUser | null>(null);
  const [trades, setTrades] = useState<TradeLogEntry[]>([]);
  const [savedPortfolios, setSavedPortfolios] = useState<string[]>([]);
  const [defaultPortfolio, setDefaultPortfolio] = useState("");
  const [activePortfolio, setActivePortfolio] = useState("");
  const [newPortfolio, setNewPortfolio] = useState("");
  const [status, setStatus] = useState("");
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(true);

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
        setError(data.error || "Could not load portfolios.");
        setIsLoading(false);
        return;
      }

      setUser(data.user || null);
      setTrades(Array.isArray(data.trades) ? data.trades : []);
      setSavedPortfolios(Array.isArray(data.portfolios) ? data.portfolios : []);
      setDefaultPortfolio(String(data.defaultPortfolio || ""));
      setActivePortfolio(String(data.defaultPortfolio || ""));
      setIsLoading(false);
    }

    loadPageData().catch((loadError) => {
      if (!cancelled) {
        setError(loadError instanceof Error ? loadError.message : "Could not load portfolios.");
        setIsLoading(false);
      }
    });

    return () => {
      cancelled = true;
    };
  }, []);

  const brandenTrades = useMemo(() => trades.filter((trade) => trade.userId === "branden" && !trade.hidden), [trades]);
  const portfolioOptions = useMemo(
    () => sortedUnique([...savedPortfolios, ...brandenTrades.map((trade) => trade.portfolioTag).filter(Boolean)]),
    [brandenTrades, savedPortfolios]
  );
  const canEdit = user?.id === "branden" && !user.readOnly;

  async function savePortfolios(nextPortfolios: string[], nextDefaultPortfolio = defaultPortfolio) {
    const normalizedPortfolios = sortedUnique(nextPortfolios);
    const normalizedDefault = nextDefaultPortfolio && normalizedPortfolios.includes(nextDefaultPortfolio) ? nextDefaultPortfolio : "";
    const response = await fetch("/api/settings/branden-portfolios", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ portfolios: normalizedPortfolios, defaultPortfolio: normalizedDefault })
    });
    const data = await response.json().catch(() => ({}));

    if (!response.ok) {
      setStatus(data.error || "Could not save portfolios.");
      return null;
    }

    setSavedPortfolios(data.portfolios || []);
    setDefaultPortfolio(data.defaultPortfolio || "");
    return data.portfolios || [];
  }

  async function createPortfolio() {
    const name = newPortfolio.trim();
    if (!name) {
      setStatus("Enter a portfolio name.");
      return;
    }
    const saved = await savePortfolios([...portfolioOptions, name], defaultPortfolio);
    if (!saved) return;
    setNewPortfolio("");
    setActivePortfolio(name);
    setStatus(`${name} portfolio saved.`);
  }

  async function saveDefault(nextDefaultPortfolio: string) {
    const saved = await savePortfolios(portfolioOptions, nextDefaultPortfolio);
    if (!saved) return;
    setDefaultPortfolio(nextDefaultPortfolio);
    setActivePortfolio(nextDefaultPortfolio);
    setStatus(nextDefaultPortfolio ? `Default portfolio set to ${nextDefaultPortfolio}.` : "Default portfolio set to all portfolios.");
  }

  return (
    <div className="branden-journal-content">
        <header className="branden-route-header">
          <div>
            <p className="eyebrow">Branden journal</p>
            <h1>Portfolios</h1>
            <span>Manage saved portfolio views and default portfolio.</span>
          </div>
        </header>

        {status ? <p className="status trade-log-status">{status}</p> : null}
        {error ? <p className="status error">{error}</p> : null}
        {isLoading ? <p className="status">Loading portfolios...</p> : null}

        {!isLoading && !error ? (
          <section className="column-settings-panel">
            {!canEdit && user ? <p className="muted">Read-only access. Portfolio settings cannot be changed.</p> : null}
            <div className="branden-settings-grid">
              <article className="branden-settings-card">
                <span className="eyebrow">Create portfolio</span>
                <h4>New portfolio view</h4>
                <div className="trade-portfolio-builder">
                  <label>
                    Portfolio name
                    <input value={newPortfolio} onChange={(event) => setNewPortfolio(event.target.value)} placeholder="Long Term, IRA, Cash" disabled={!canEdit} />
                  </label>
                  <button className="trade-muted-button" type="button" onClick={createPortfolio} disabled={!canEdit}>
                    Save Portfolio
                  </button>
                </div>
              </article>
              <article className="branden-settings-card">
                <span className="eyebrow">Active view</span>
                <h4>Current portfolio filter</h4>
                <label>
                  Portfolio view
                  <select value={activePortfolio} onChange={(event) => setActivePortfolio(event.target.value)}>
                    <option value="">All portfolios</option>
                    {portfolioOptions.map((portfolio) => (
                      <option key={portfolio} value={portfolio}>{portfolio}</option>
                    ))}
                  </select>
                </label>
              </article>
              <article className="branden-settings-card">
                <span className="eyebrow">Default view</span>
                <h4>Default portfolio</h4>
                <label>
                  Default portfolio
                  <select value={defaultPortfolio} onChange={(event) => saveDefault(event.target.value)} disabled={!canEdit}>
                    <option value="">All portfolios</option>
                    {portfolioOptions.map((portfolio) => (
                      <option key={portfolio} value={portfolio}>{portfolio}</option>
                    ))}
                  </select>
                </label>
              </article>
            </div>
            <div className="portfolio-card-grid">
              {portfolioOptions.length ? (
                portfolioOptions.map((portfolio) => {
                  const portfolioTrades = brandenTrades.filter((trade) => trade.portfolioTag === portfolio);
                  const portfolioPnl = portfolioTrades.reduce((total, trade) => total + trade.pnl, 0);

                  return (
                    <article className="portfolio-card" key={portfolio}>
                      <div>
                        <span className="eyebrow">Saved portfolio</span>
                        <h4>{portfolio}</h4>
                        {defaultPortfolio === portfolio ? <p className="portfolio-default-pill">Default</p> : null}
                      </div>
                      <dl>
                        <div><dt>Trades</dt><dd>{portfolioTrades.length}</dd></div>
                        <div><dt>Net P&amp;L</dt><dd className={portfolioPnl >= 0 ? "trade-positive" : "trade-negative"}>{money(portfolioPnl)}</dd></div>
                      </dl>
                      <button className="trade-muted-button" type="button" onClick={() => setActivePortfolio(portfolio)}>View Portfolio</button>
                      <button className="trade-muted-button" type="button" onClick={() => saveDefault(portfolio)} disabled={!canEdit}>Set Default</button>
                    </article>
                  );
                })
              ) : (
                <p className="muted">No saved portfolios yet.</p>
              )}
            </div>
          </section>
        ) : null}
      </div>
  );
}
