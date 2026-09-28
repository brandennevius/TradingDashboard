import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { parseCfStatementText } from "../lib/cf-statement";

test("stored merge survives CF replacement, blocks child edits, and can be undone", async () => {
  const cwd = process.cwd();
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), "trade-merge-store-"));
  const oldDb = process.env.DATABASE_URL;
  delete process.env.DATABASE_URL;
  process.chdir(dir);
  try {
    const store = await import("../lib/store");
    const parsed = parseCfStatementText([
      "1834251:1 28/09/2026 12:00:00.000 Buy 0.00 COPPER 6.57 1001 46.21 —",
      "1834251:2 28/09/2026 12:01:00.000 Buy 1.00 COPPER 6.59 1002 -16.30 —"
    ].join("\n"), "branden", "CF_Statement");
    await store.replaceCfStatementTrades("branden", "CF_Statement", parsed.trades);
    const original = await store.listTrades();
    let visible = await store.changeTradeMerge("branden", original.map(t => t.id));
    assert.equal(visible.length, 1);
    const parentId = visible[0].id;
    assert.equal(visible[0].pnl, 29.91);
    await assert.rejects(store.setTradeHidden(original[0].id, "branden", false), /Undo the merge/);
    await assert.rejects(store.deleteTrade(parentId, "branden"), /Undo the merge/);
    await store.updateTrade(parentId, "branden", { ...visible[0], notes: "Keep this review" });
    await store.replaceCfStatementTrades("branden", "CF_Statement", parsed.trades);
    visible = await store.listBrandenVisibleTrades();
    assert.equal(visible.length, 1);
    assert.equal(visible[0].pnl, 29.91);
    assert.equal(visible[0].notes, "Keep this review");
    assert.equal(visible[0].groupRole, "parent");
    const restored = await store.changeTradeMerge("branden", parentId);
    assert.equal(restored.length, 2);
    assert.equal(restored.every(t => t.groupRole === "none"), true);
    assert.equal(Math.round(restored.reduce((sum, t) => sum + t.pnl, 0) * 100), 2991);
  } finally {
    process.chdir(cwd);
    if (oldDb !== undefined) process.env.DATABASE_URL = oldDb;
    await fs.rm(dir, { recursive: true, force: true });
  }
});
