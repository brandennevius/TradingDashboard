"use client";

import { ChangeEvent, useEffect, useMemo, useRef, useState } from "react";
import { useBrandenSnapshotActions } from "../BrandenSnapshotActionsContext";
import type { TradeLogEntry, TraderUser } from "@/lib/types";

type BrandenTradeColumnKey =
  | "status"
  | "side"
  | "symbol"
  | "setup"
  | "portfolio"
  | "openDate"
  | "entry"
  | "size"
  | "closeDate"
  | "exit"
  | "stop"
  | "commission"
  | "usedMargin"
  | "takeProfit"
  | "risk"
  | "cost"
  | "netReturn"
  | "r"
  | "mistake"
  | "custom"
  | "grade"
  | "review";

type BrandenColumnPreference = {
  key: BrandenTradeColumnKey;
  visible: boolean;
};

type PortfolioSettingsResponse = {
  portfolios?: string[];
  defaultPortfolio?: string;
};

const defaultBrandenColumns: BrandenColumnPreference[] = [
  { key: "status", visible: true },
  { key: "side", visible: true },
  { key: "symbol", visible: true },
  { key: "setup", visible: true },
  { key: "portfolio", visible: true },
  { key: "openDate", visible: true },
  { key: "entry", visible: true },
  { key: "size", visible: true },
  { key: "closeDate", visible: true },
  { key: "exit", visible: true },
  { key: "stop", visible: false },
  { key: "commission", visible: false },
  { key: "usedMargin", visible: false },
  { key: "takeProfit", visible: false },
  { key: "risk", visible: false },
  { key: "cost", visible: true },
  { key: "netReturn", visible: true },
  { key: "r", visible: true },
  { key: "mistake", visible: false },
  { key: "custom", visible: true },
  { key: "grade", visible: true },
  { key: "review", visible: true }
];

function currentDate() {
  const now = new Date();
  return `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, "0")}-${String(now.getDate()).padStart(2, "0")}`;
}

function sortedUnique(values: string[]) {
  return Array.from(new Set(values.map((value) => value.trim()).filter(Boolean))).sort((a, b) => a.localeCompare(b));
}

function normalizeColumns(value: unknown): BrandenColumnPreference[] {
  const seen = new Set<string>();
  const normalized: BrandenColumnPreference[] = [];

  if (Array.isArray(value)) {
    value.forEach((item) => {
      if (!item || typeof item !== "object") return;
      const raw = item as Record<string, unknown>;
      const key = String(raw.key || "") as BrandenTradeColumnKey;
      if (!defaultBrandenColumns.some((column) => column.key === key) || seen.has(key)) return;
      seen.add(key);
      normalized.push({ key, visible: "visible" in raw ? Boolean(raw.visible) : true });
    });
  }

  defaultBrandenColumns.forEach((column) => {
    if (!seen.has(column.key)) normalized.push(column);
  });

  return normalized;
}

function columnLabel(key: BrandenTradeColumnKey) {
  const labels: Record<BrandenTradeColumnKey, string> = {
    status: "Status",
    side: "Side",
    symbol: "Symbol",
    setup: "Setup",
    portfolio: "Portfolio",
    openDate: "Open Date",
    entry: "Entry",
    size: "Size",
    closeDate: "Close Date",
    exit: "Exit",
    stop: "Stop",
    commission: "Commission",
    usedMargin: "Used Margin",
    takeProfit: "Take Profit",
    risk: "Risk",
    cost: "Cost",
    netReturn: "Net Return",
    r: "R",
    mistake: "Mistake",
    custom: "Custom Tags",
    grade: "Grade",
    review: "Review"
  };
  return labels[key] || key;
}

function money(value: number) {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0
  }).format(Number.isFinite(value) ? value : 0);
}

export default function BrandenSettingsPage() {
  const {
    canGenerateSnapshot,
    generateDailySnapshot,
    generateAndSendDailySnapshot,
    generateMtdSnapshot,
    generateAndSendMtdSnapshot,
    isGeneratingDailySnapshot,
    isGeneratingMtdSnapshot,
    isMtdEmailConfigured
  } = useBrandenSnapshotActions();
  const backupInputRef = useRef<HTMLInputElement | null>(null);
  const [user, setUser] = useState<TraderUser | null>(null);
  const [portfolios, setPortfolios] = useState<string[]>([]);
  const [defaultPortfolio, setDefaultPortfolio] = useState("");
  const [activePortfolio, setActivePortfolio] = useState("");
  const [newPortfolio, setNewPortfolio] = useState("");
  const [trades, setTrades] = useState<TradeLogEntry[]>([]);
  const [selectedHiddenTradeIds, setSelectedHiddenTradeIds] = useState<string[]>([]);
  const [preferences, setPreferences] = useState<Record<string, BrandenColumnPreference[]>>({});
  const [status, setStatus] = useState("");
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(true);

  const canEdit = user?.id === "branden" && !user.readOnly;
  const columnScope = activePortfolio || "__all__";
  const activeColumns = useMemo(
    () => normalizeColumns(preferences[columnScope] || preferences.__all__ || defaultBrandenColumns),
    [columnScope, preferences]
  );
  const hiddenTrades = useMemo(
    () =>
      trades
        .filter((trade) => trade.userId === "branden" && trade.hidden)
        .sort((a, b) => (b.entryDate || "").localeCompare(a.entryDate || "") || a.symbol.localeCompare(b.symbol)),
    [trades]
  );
  const selectedHiddenTrades = useMemo(
    () => hiddenTrades.filter((trade) => selectedHiddenTradeIds.includes(trade.id)),
    [hiddenTrades, selectedHiddenTradeIds]
  );
  const allHiddenSelected = hiddenTrades.length > 0 && hiddenTrades.every((trade) => selectedHiddenTradeIds.includes(trade.id));

  useEffect(() => {
    let cancelled = false;

    async function loadSettings() {
      setIsLoading(true);
      setError("");

      const [sessionResponse, portfolioResponse, columnResponse, tradeResponse] = await Promise.all([
        fetch("/api/session", { cache: "no-store" }),
        fetch("/api/settings/branden-portfolios", { cache: "no-store" }),
        fetch("/api/settings/branden-columns", { cache: "no-store" }),
        fetch("/api/trades", { cache: "no-store" })
      ]);

      const sessionData = await sessionResponse.json().catch(() => ({}));
      const portfolioData = (await portfolioResponse.json().catch(() => ({}))) as PortfolioSettingsResponse & { error?: string };
      const columnData = await columnResponse.json().catch(() => ({}));
      const tradeData = await tradeResponse.json().catch(() => ({}));

      if (cancelled) return;

      if (!sessionResponse.ok || !portfolioResponse.ok || !columnResponse.ok || !tradeResponse.ok) {
        setError(sessionData.error || portfolioData.error || columnData.error || tradeData.error || "Could not load settings.");
        setIsLoading(false);
        return;
      }

      setUser(sessionData.user || null);
      setPortfolios(Array.isArray(portfolioData.portfolios) ? portfolioData.portfolios : []);
      setDefaultPortfolio(String(portfolioData.defaultPortfolio || ""));
      setActivePortfolio(String(portfolioData.defaultPortfolio || ""));
      setTrades(Array.isArray(tradeData.trades) ? tradeData.trades : []);
      setPreferences(columnData.preferences && typeof columnData.preferences === "object" ? columnData.preferences : {});
      setIsLoading(false);
    }

    loadSettings().catch((loadError) => {
      if (!cancelled) {
        setError(loadError instanceof Error ? loadError.message : "Could not load settings.");
        setIsLoading(false);
      }
    });

    return () => {
      cancelled = true;
    };
  }, []);

  async function saveColumns(nextPreferences: Record<string, BrandenColumnPreference[]>) {
    const response = await fetch("/api/settings/branden-columns", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ preferences: nextPreferences })
    });
    const data = await response.json().catch(() => ({}));

    if (!response.ok) {
      setStatus(data.error || "Could not save column settings.");
      return;
    }

    setPreferences(data.preferences && typeof data.preferences === "object" ? data.preferences : nextPreferences);
    setStatus("Column settings saved.");
  }

  function toggleColumn(key: BrandenTradeColumnKey) {
    const nextColumns = activeColumns.map((column) => (column.key === key ? { ...column, visible: !column.visible } : column));
    void saveColumns({ ...preferences, [columnScope]: nextColumns });
  }

  function toggleHiddenTradeSelection(tradeId: string) {
    setSelectedHiddenTradeIds((current) =>
      current.includes(tradeId) ? current.filter((selectedId) => selectedId !== tradeId) : [...current, tradeId]
    );
  }

  function toggleAllHiddenTrades() {
    setSelectedHiddenTradeIds(allHiddenSelected ? [] : hiddenTrades.map((trade) => trade.id));
  }

  async function unhideSelectedTrades() {
    if (!canEdit) {
      setStatus("Read-only access. Hidden trades cannot be restored.");
      return;
    }

    if (!selectedHiddenTrades.length) {
      setStatus("Select hidden trades to restore.");
      return;
    }

    setStatus(`Restoring ${selectedHiddenTrades.length} hidden ${selectedHiddenTrades.length === 1 ? "trade" : "trades"}...`);
    setError("");

    try {
      const updatedTrades: TradeLogEntry[] = [];
      for (const trade of selectedHiddenTrades) {
        const customTags = trade.customTags.filter((tag) => tag !== "Manually hidden");
        const response = await fetch(`/api/trades/${encodeURIComponent(trade.id)}`, {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ hidden: false, customTags })
        });
        const data = await response.json().catch(() => ({}));
        if (!response.ok) {
          throw new Error(data.error || `Could not restore ${trade.symbol}.`);
        }
        if (data.trade) updatedTrades.push(data.trade);
      }

      setTrades((current) =>
        current.map((trade) => updatedTrades.find((updated) => updated.id === trade.id) || trade)
      );
      setSelectedHiddenTradeIds([]);
      setStatus(`${updatedTrades.length} hidden ${updatedTrades.length === 1 ? "trade" : "trades"} restored.`);
    } catch (unhideError) {
      setError(unhideError instanceof Error ? unhideError.message : "Could not restore hidden trades.");
    }
  }

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
      return;
    }

    setPortfolios(Array.isArray(data.portfolios) ? data.portfolios : normalizedPortfolios);
    setDefaultPortfolio(String(data.defaultPortfolio || ""));
    setStatus("Portfolio settings saved.");
  }

  function createPortfolio() {
    const name = newPortfolio.trim();
    if (!name) {
      setStatus("Enter a portfolio name.");
      return;
    }
    void savePortfolios([...portfolios, name]).then(() => {
      setActivePortfolio(name);
      setNewPortfolio("");
    });
  }

  async function exportBackup() {
    setStatus("Preparing full journal backup...");
    const response = await fetch("/api/settings/branden-backup", { cache: "no-store" });
    if (!response.ok) {
      const data = await response.json().catch(() => ({}));
      setStatus(data.error || "Could not export the journal backup.");
      return;
    }
    const blob = await response.blob();
    const disposition = response.headers.get("Content-Disposition") || "";
    const fileName = disposition.match(/filename="([^"]+)"/)?.[1] || `branden-journal-backup-${currentDate()}.json`;
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = fileName;
    link.click();
    URL.revokeObjectURL(url);
    setStatus("Full journal backup exported.");
  }

  async function importBackup(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    event.currentTarget.value = "";
    if (!file) return;

    const confirmed = window.confirm(
      "Restore this full Branden journal backup?\n\nThis replaces the current Branden trades, reports, settings, watchlists, market-cycle history, and trade screenshots with the contents of the backup file."
    );
    if (!confirmed) return;

    setStatus("Validating and restoring journal backup...");
    try {
      const backup = JSON.parse(await file.text());
      const response = await fetch("/api/settings/branden-backup", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(backup)
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        setStatus(data.error || "Could not import the journal backup.");
        return;
      }
      setStatus(
        `Backup restored: ${data.restored?.trades || 0} trades, ${data.restored?.reports || 0} reports, ${data.restored?.screenshots || 0} screenshots. Reloading...`
      );
      window.setTimeout(() => window.location.reload(), 800);
    } catch (importError) {
      setStatus(importError instanceof Error ? importError.message : "The selected backup file is invalid.");
    }
  }

  return (
    <div className="branden-journal-content">
        <header className="branden-route-header">
          <div>
            <p className="eyebrow">Branden journal</p>
            <h1>Settings</h1>
            <span>Backups, portfolio defaults, and trade-log column preferences.</span>
          </div>
        </header>

        {status ? <p className="status trade-log-status">{status}</p> : null}
        {error ? <p className="status error">{error}</p> : null}
        {isLoading ? <p className="status">Loading settings...</p> : null}

        <input ref={backupInputRef} className="trade-file-input" type="file" accept="application/json,.json" onChange={importBackup} />

        {!isLoading && !error ? (
          <section className="column-settings-panel">
            {!canEdit && user ? <p className="muted">Read-only access. Settings cannot be changed.</p> : null}
            <div className="branden-settings-grid">
              <article className="branden-settings-card">
                <span className="eyebrow">Reports</span>
                <h4>Snapshot reports</h4>
                <p>Generate and email Daily or Month-to-Date snapshots from the selected portfolio. Download-only actions remain available here for validation.</p>
                <div className="branden-settings-actions">
                  <button className="trade-primary-button" type="button" disabled={!canGenerateSnapshot || isGeneratingDailySnapshot} onClick={generateAndSendDailySnapshot}>
                    {isGeneratingDailySnapshot ? "Generating Daily Snapshot..." : "Generate and Send Daily Snapshot"}
                  </button>
                  <button className="trade-primary-button" type="button" disabled={!canGenerateSnapshot || isGeneratingMtdSnapshot || !isMtdEmailConfigured} onClick={generateAndSendMtdSnapshot}>
                    {isGeneratingMtdSnapshot ? "Generating MTD Snapshot..." : "Generate and Send MTD Snapshot"}
                  </button>
                  <button className="trade-muted-button" type="button" disabled={!canGenerateSnapshot || isGeneratingDailySnapshot} onClick={generateDailySnapshot}>
                    Download Daily Snapshot Only
                  </button>
                  <button className="trade-muted-button" type="button" disabled={!canGenerateSnapshot || isGeneratingMtdSnapshot} onClick={generateMtdSnapshot}>
                    Download MTD Snapshot Only
                  </button>
                </div>
                {!isMtdEmailConfigured ? <p className="muted">MTD email delivery is disabled until SMTP settings are configured.</p> : null}
              </article>
              <article className="branden-settings-card">
                <span className="eyebrow">Backup</span>
                <h4>Full journal backup</h4>
                <p>Export or restore trades, executions, screenshots, setup builder data, watchlists, portfolios, and settings.</p>
                <div className="branden-settings-actions">
                  <button className="trade-muted-button" type="button" onClick={exportBackup}>
                    Export Full Backup
                  </button>
                  <button className="trade-danger-button" type="button" onClick={() => backupInputRef.current?.click()} disabled={!canEdit}>
                    Import Full Backup
                  </button>
                </div>
              </article>
            </div>

            <div className="column-settings-head">
              <div>
                <p className="eyebrow">Trade Log Columns</p>
                <h3>{activePortfolio ? `${activePortfolio} columns` : "Default columns"}</h3>
              </div>
              <label>
                Column scope
                <select value={activePortfolio} onChange={(event) => setActivePortfolio(event.target.value)}>
                  <option value="">Default / all portfolios</option>
                  {portfolios.map((portfolio) => (
                    <option key={portfolio} value={portfolio}>
                      {portfolio}
                    </option>
                  ))}
                </select>
              </label>
            </div>

            <div className="column-settings-list">
              {activeColumns.map((column) => (
                <div className="column-settings-row" key={column.key}>
                  <label>
                    <input type="checkbox" checked={column.visible} onChange={() => toggleColumn(column.key)} disabled={!canEdit} />
                    {columnLabel(column.key)}
                  </label>
                </div>
              ))}
            </div>

            <section className="hidden-trades-panel">
              <div className="hidden-trades-head">
                <div>
                  <p className="eyebrow">Hidden trades</p>
                  <h4>Excluded from reports</h4>
                  <p>{hiddenTrades.length} hidden {hiddenTrades.length === 1 ? "trade" : "trades"}</p>
                </div>
                <div className="hidden-trades-actions">
                  <button className="trade-muted-button" type="button" onClick={unhideSelectedTrades} disabled={!canEdit || !selectedHiddenTrades.length}>
                    Restore selected
                  </button>
                </div>
              </div>
              <div className="hidden-trades-table-wrap">
                <table className="hidden-trades-table">
                  <thead>
                    <tr>
                      <th>
                        <input
                          type="checkbox"
                          checked={allHiddenSelected}
                          disabled={!hiddenTrades.length}
                          aria-label={allHiddenSelected ? "Clear hidden trade selection" : "Select all hidden trades"}
                          onChange={toggleAllHiddenTrades}
                        />
                      </th>
                      <th>Symbol</th>
                      <th>Status</th>
                      <th>Side</th>
                      <th>Open Date</th>
                      <th>Close Date</th>
                      <th>Net Return</th>
                      <th>Portfolio</th>
                      <th>Tags</th>
                    </tr>
                  </thead>
                  <tbody>
                    {hiddenTrades.map((trade) => (
                      <tr key={trade.id}>
                        <td>
                          <input
                            type="checkbox"
                            checked={selectedHiddenTradeIds.includes(trade.id)}
                            aria-label={`Select hidden ${trade.symbol} trade from ${trade.entryDate}`}
                            onChange={() => toggleHiddenTradeSelection(trade.id)}
                          />
                        </td>
                        <td>#{trade.symbol}</td>
                        <td>{trade.status}</td>
                        <td>{trade.side}</td>
                        <td>{trade.entryDate || "-"}</td>
                        <td>{trade.exitDate || "-"}</td>
                        <td>{money(trade.pnl)}</td>
                        <td>{trade.portfolioTag || "-"}</td>
                        <td>{trade.customTags.length ? trade.customTags.join(", ") : "-"}</td>
                      </tr>
                    ))}
                    {!hiddenTrades.length ? (
                      <tr>
                        <td colSpan={9}>No hidden trades.</td>
                      </tr>
                    ) : null}
                  </tbody>
                </table>
              </div>
            </section>
          </section>
        ) : null}
      </div>
  );
}
