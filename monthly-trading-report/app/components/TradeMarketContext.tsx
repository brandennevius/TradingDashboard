"use client";

import React, { useEffect, useState } from "react";
import type { MarketSnapshot, TradeMarketContext as Context } from "@/lib/trade-market-context";

function reading(snapshot: MarketSnapshot | null, label: string) {
  if (!snapshot) return "Unavailable";
  const gauge = snapshot.gauge;
  if (label === "Overall") return gauge.overall_state;
  if (label === "Leadership") return `${gauge.leadership.above21}/${gauge.leadership.total} above 21 EMA`;
  if (label === "Extension") {
    const states = gauge.index_regimes.map((r) => r.extension);
    return states.includes("Extended") ? "Extended" : states.includes("Caution") ? "Caution" : "Normal";
  }
  return gauge.components.find((c) => c.label === label)?.state || "Unavailable";
}

export default function TradeMarketContext({ tradeId, updatedAt }: { tradeId: string; updatedAt: string }) {
  const [context, setContext] = useState<Context | null>(null);
  const [error, setError] = useState("");
  const [warning, setWarning] = useState("");
  useEffect(() => {
    const controller = new AbortController();
    setContext(null); setError(""); setWarning("");
    fetch(`/api/journal/branden/market-context?tradeId=${encodeURIComponent(tradeId)}`, { cache: "no-store", signal: controller.signal })
      .then(async (r) => { const data = await r.json(); if (!r.ok) throw new Error(data.error || "Market history unavailable."); return data; })
      .then((data) => { setContext(data.context); setWarning(data.warning || ""); })
      .catch((e) => { if (!controller.signal.aborted) setError(e.message); });
    return () => controller.abort();
  }, [tradeId, updatedAt]);
  return <article className="trade-detail-section market-context-panel" aria-label="Market context">
    <h3>Market Context</h3>
    {error ? <p role="status">{error}</p> : !context ? <p role="status">Loading market history…</p> : <MarketContextComparison key={tradeId} context={context} warning={warning} />}
  </article>;
}

export function MarketContextComparison({ context, warning = "" }: { context: Context; warning?: string }) {
  const [expanded, setExpanded] = useState(false);
  return <>
      <div className="market-context-table-wrap"><table className="market-context-table">
        <thead><tr><th scope="col">Condition</th><th scope="col">At entry<small>{context.entrySession || "Unavailable"} close · {context.entry?.source || "Unavailable"}</small></th>
          <th scope="col">{context.endpointLabel}<small>{context.endpointSession || "Unavailable"} close · {context.endpoint?.source || "Unavailable"}</small></th></tr></thead>
        <tbody>{["Overall", "Short term", "Medium term", "Long term", "Leadership", "Extension"].map((label) => <tr key={label}>
          <th scope="row">{label}</th><td>{reading(context.entry, label)}</td><td>{reading(context.endpoint, label)}</td>
        </tr>)}</tbody>
      </table></div>
      <p className="market-context-note">Completed session closes only. Unzoned execution times use the preceding session to avoid future information.</p>
      <p>{context.changedDuringTrade ? context.entryState === context.endpointState ? "The market changed during this trade, then returned to the entry state." : "The market changed during this trade." : context.history.length ? "No overall state change in the available history." : "Market history is not available for this period."}
        {context.missingSessions ? ` ${context.missingSessions} session(s) unavailable; history is incomplete.` : ""}</p>
      {warning ? <p className="market-context-note" role="status">{warning}</p> : null}
      <details open={expanded} onToggle={(event) => setExpanded(event.currentTarget.open)}>
        <summary>View market history ({context.history.length} saved sessions)</summary>
        {expanded ? <div className="market-context-history"><table className="market-context-table">
          <thead><tr><th scope="col">Session close</th><th scope="col">Overall</th><th scope="col">Record</th></tr></thead>
          <tbody>{context.history.map((s) => <tr key={`${s.version}-${s.session}`}><td>{s.session}</td><td>{s.state}</td><td>{s.source}</td></tr>)}</tbody>
        </table></div> : null}
      </details>
  </>;
}
