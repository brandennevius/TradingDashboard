import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";

test("portfolio deletion removes hidden and closed records only for the exact owner and name", async () => {
  const cwd = process.cwd();
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), "portfolio-delete-"));
  process.chdir(dir);
  try {
    const store = await import("../lib/store");
    await fs.mkdir("data");
    const row = (id: string, portfolioTag: string, extra = {}) => ({
      id, portfolioTag, userId: "branden", status: "OPEN", groupId: "", groupRole: "none",
      reviewSections: { setup: "", entry: "", exit: "", didRight: "", didWrong: "", general: "" }, ...extra
    });
    const target = "Campus Fund - Main";
    const preserved = [row("current", "CF_Statement"), row("similar", `${target} 2`), row("cam", target, { userId: "cam" })];
    const original = [row("open", target), row("closed", target, { status: "WIN" }), row("hidden", target, { hidden: true }), ...preserved];
    await fs.writeFile("data/trade-logs.json", JSON.stringify(original));
    const keepSnapshot = { userId: "branden", portfolioTag: "CF_Statement", coverageDate: "2026-10-30", sourceHash: "keep", importedAt: "2026-10-31T00:00:00.000Z" };
    await fs.writeFile("data/app-settings.json", JSON.stringify({
      unrelated: { keep: true },
      [`ibkr_flex:branden:${target}`]: { trades: ["old execution"] },
      "ibkr_flex:branden:IBKR Personal": { trades: ["keep execution"] },
      brandenPortfolioNames: { portfolios: [target, "CF_Statement"], defaultPortfolio: target, portfolioMeta: { [target]: { currentEquity: 100 }, CF_Statement: { currentEquity: 200 } } },
      brandenColumnPreferences: { [target]: ["old"], CF_Statement: ["keep"] },
      brandenBrokerPortfolioSnapshots: [{ ...keepSnapshot, portfolioTag: target }, keepSnapshot]
    }));
    const result = await store.deleteBrandenPortfolio(target);
    assert.equal(result.deletedTrades, 3);
    assert.deepEqual(result.portfolios, ["CF_Statement"]);
    assert.equal(result.defaultPortfolio, "");
    const raw = JSON.parse(await fs.readFile("data/trade-logs.json", "utf8"));
    assert.deepEqual(raw, preserved);
    const settings = JSON.parse(await fs.readFile("data/app-settings.json", "utf8"));
    assert.deepEqual(settings.unrelated, { keep: true });
    assert.deepEqual(settings.brandenColumnPreferences, { CF_Statement: ["keep"] });
    assert.equal(settings.brandenBrokerPortfolioSnapshots.length, 1);
    assert.equal(settings.brandenBrokerPortfolioSnapshots[0].sourceHash, "keep");
    assert.equal(settings[`ibkr_flex:branden:${target}`], undefined);
    assert.deepEqual(settings["ibkr_flex:branden:IBKR Personal"], { trades: ["keep execution"] });
    assert.equal(settings.brandenPortfolioNames.portfolioMeta[target], undefined);
    assert.equal(settings.brandenPortfolioNames.portfolioMeta.CF_Statement.currentEquity, 200);

    // Rejected requests leave both storage files byte-for-byte unchanged.
    const beforeTrades = await fs.readFile("data/trade-logs.json", "utf8");
    const beforeSettings = await fs.readFile("data/app-settings.json", "utf8");
    await assert.rejects(store.deleteBrandenPortfolio(""), /named portfolio/);
    await assert.rejects(store.deleteBrandenPortfolio("Missing"), /not found/);
    assert.equal(await fs.readFile("data/trade-logs.json", "utf8"), beforeTrades);
    assert.equal(await fs.readFile("data/app-settings.json", "utf8"), beforeSettings);

    await fs.writeFile("data/trade-logs.json", JSON.stringify([row("merged", "CF_Statement", { groupId: "parent", groupRole: "child" })]));
    await assert.rejects(store.deleteBrandenPortfolio("CF_Statement"), /Undo.*merges/);
    assert.equal(await fs.readFile("data/app-settings.json", "utf8"), beforeSettings);
    assert.equal((await store.listTrades()).length, 1);

    // An empty saved portfolio can be deleted; another default is retained.
    await store.saveBrandenPortfolioSettings({ portfolios: ["Empty", "CF_Statement"], defaultPortfolio: "CF_Statement" });
    const empty = await store.deleteBrandenPortfolio("Empty");
    assert.equal(empty.deletedTrades, 0);
    assert.equal(empty.defaultPortfolio, "CF_Statement");
  } finally {
    process.chdir(cwd);
    await fs.rm(dir, { recursive: true, force: true });
  }
});
