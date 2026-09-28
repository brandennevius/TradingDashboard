import test from "node:test";
import assert from "node:assert/strict";
import { buildMergedTrade, mergeTradeRecords, reconcileTradeMerges, undoTradeMerge } from "../lib/trade-merge";
import { parseCfStatementText, buildCfTradesFromExecutionHistory } from "../lib/cf-statement";
import { applyManualFieldsToCfStatementTrade } from "../lib/cf-import-reconciliation";
import type { TradeLogEntry } from "../lib/types";

function copper(): TradeLogEntry[] {
  const pnls = [46.21, 76.58, 59.57, 8.76, 10.67, 10.59, -16.30];
  return parseCfStatementText(pnls.map((pnl, i) => `1834251:${i} 28/09/2026 12:0${i}:00.000 Buy ${i === 6 ? "1.00" : "0.00"} COPPER 6.57 ${1000 + i} ${pnl.toFixed(2)} —`).join("\n"), "branden", "CF_Statement").trades.map((t, i) => ({ ...t, id: `copper-${i}`, executions: t.executions || [], groupId: "", groupRole: "none", hidden: false, createdAt: "2026-09-28T12:00:00Z", updatedAt: "2026-09-28T12:00:00Z" }));
}

test("seven copper settlements become one winning trade and undo restores exact sources", () => {
  const original = copper();
  const state = mergeTradeRecords(original, original.map(t => t.id), "branden", "merged");
  const visible = state.filter(t => !t.hidden);
  assert.equal(visible.length, 1);
  assert.equal(visible[0].pnl, 196.08);
  assert.equal(visible[0].status, "WIN");
  assert.equal(visible[0].executions.length, 7);
  assert.equal(visible[0].shares, 0);
  assert.equal(visible[0].avgEntry, 0);
  assert.equal(visible[0].returnPercent, 0);
  assert.deepEqual(undoTradeMerge(state, "merged", "branden"), original);
});

test("CF rebuild preserves merged membership by broker executions even when row IDs change", () => {
  const original = copper();
  const state = mergeTradeRecords(original, original.map(t => t.id), "branden", "merged");
  const parent = state.find(t => t.id === "merged")!;
  parent.notes = "Reviewed copper trade";
  const history = original.flatMap(t => t.executions.map(e => ({ ...e, source: t.symbol })));
  const rebuilt = buildCfTradesFromExecutionHistory(history, [], [], "branden", "CF_Statement").map((t, i) => ({ ...original[i], ...applyManualFieldsToCfStatementTrade(t, state), id: `rebuilt-${i}`, executions: t.executions || [], hidden: false, groupId: "", groupRole: "none" as const }));
  const once = reconcileTradeMerges([...rebuilt, parent]);
  const twice = reconcileTradeMerges(once);
  assert.equal(twice.filter(t => !t.hidden).length, 1);
  assert.equal(twice.find(t => t.id === "merged")?.pnl, 196.08);
  assert.equal(twice.find(t => t.id === "merged")?.notes, "Reviewed copper trade");
  assert.equal(twice.filter(t => t.groupRole === "child").length, 7);
  assert.equal(undoTradeMerge(twice, "merged", "branden").filter(t => !t.hidden).length, 7);
});

test("selection validation rejects foreign, hidden, incompatible, duplicate and nested merges", () => {
  const rows = copper();
  assert.throws(() => mergeTradeRecords(rows, [rows[0].id], "branden", "x"));
  assert.throws(() => mergeTradeRecords(rows, [rows[0].id, rows[0].id], "branden", "x"));
  assert.throws(() => mergeTradeRecords(rows, [rows[0].id, rows[1].id], "other-user", "x"));
  for (const change of [{ symbol: "OTHER" }, { side: "LONG" }, { portfolioTag: "Other" }, { hidden: true }, { groupRole: "child" }]) {
    const changed = rows.map((t, i) => i === 1 ? { ...t, ...change } as TradeLogEntry : t);
    assert.throws(() => mergeTradeRecords(changed, [rows[0].id, rows[1].id], "branden", "x"));
  }
});

test("known quantities retain weighted execution prices and review evidence", () => {
  const rows = copper().slice(0, 2).map((t, i) => ({ ...t, pnl: 10, shares: 1, customTags: [], notes: `note-${i}`, screenshots: [`shot-${i}`], executions: [
    { ...t.executions[0], id: `entry-${i}`, sourceKey: `entry-${i}`, type: "ENTRY" as const, shares: 1, price: 10 + i, pnl: 0, time: "09:00:00" },
    { ...t.executions[0], shares: 1, price: 20 + i, pnl: 10 }
  ] }));
  const merged = buildMergedTrade(rows, "merged");
  assert.equal(merged.avgEntry, 10.5);
  assert.equal(merged.exitPrice, 20.5);
  assert.equal(merged.shares, 2);
  assert.equal(merged.pnl, 20);
  assert.equal(merged.screenshots.length, 2);
  assert(merged.notes.includes("note-0") && merged.notes.includes("note-1"));
});

test("incomplete statement replacement cannot silently drop merged settlements", () => {
  const rows = copper();
  const state = mergeTradeRecords(rows, rows.map(t => t.id), "branden", "merged");
  assert.throws(() => reconcileTradeMerges(state.filter(t => t.id !== rows[0].id)), /omits executions/);
});

test("a rebuilt cycle overlapping two manual groups is rejected instead of double-counted", () => {
  const rows = copper();
  let state = mergeTradeRecords(rows, rows.slice(0, 2).map(t => t.id), "branden", "merge-one");
  state = mergeTradeRecords(state, rows.slice(2).map(t => t.id), "branden", "merge-two");
  const combinedRaw = { ...rows[0], executions: rows.flatMap(t => t.executions) };
  assert.throws(() => reconcileTradeMerges([...state.filter(t => t.groupRole === "parent"), combinedRaw]), /multiple merged/);
});
