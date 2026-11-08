import { test, expect } from "@playwright/test";
import { readFile } from "node:fs/promises";
import { join } from "node:path";
import {
  analyzerTemplate,
  analyzerTrade,
  analyzerTrades,
  analyzerAnswers
} from "../tests/fixtures/strategy-analyzer";

test("saved criteria drive comparisons, combined subsets and trade evidence without rewriting the journal", async ({
  page
}, testInfo) => {
  test.setTimeout(120_000); // Includes cold Next.js compilation and the complete scope/filter walkthrough.
  expect(
    (
      await page.request.post("/api/login", {
        data: { userId: "branden", password: process.env.E2E_PASSWORD }
      })
    ).status()
  ).toBe(200);
  expect(
    (
      await page.request.put("/api/settings/branden-portfolios", {
        data: {
          portfolios: ["Analyzer Test", "Other Analyzer"],
          defaultPortfolio: "Analyzer Test"
        }
      })
    ).status()
  ).toBe(200);
  const savedTemplate = await page.request.put(
    "/api/settings/setup-checklists",
    { data: { setupChecklists: [analyzerTemplate] } }
  );
  expect(savedTemplate.status()).toBe(200);
  // Create an actual archived version through the existing setup-versioning API.
  const first = (await savedTemplate.json()).setupChecklists.find(
    (item: { setupName: string }) =>
      item.setupName === analyzerTemplate.setupName
  );
  const v2 = structuredClone(first);
  v2.groups[0].criteria = v2.groups[0].criteria.filter(
    (item: { id: string }) => item.id !== "trend"
  );
  v2.criteria = v2.groups[0].criteria;
  const versioned = await page.request.put("/api/settings/setup-checklists", {
    data: { setupChecklists: [v2] }
  });
  expect(versioned.status()).toBe(200);
  const latest = (await versioned.json()).setupChecklists.find(
    (item: { archived: boolean }) => !item.archived
  );
  const trades = [
    ...analyzerTrades(),
    analyzerTrade("analyzer-open", 0, analyzerAnswers(true, true, 4), {
      status: "OPEN"
    }),
    analyzerTrade("analyzer-other", 999, analyzerAnswers(true, true, 4), {
      portfolioTag: "Other Analyzer"
    }),
    analyzerTrade(
      "analyzer-v2",
      -500,
      analyzerAnswers(true, true, 4).filter((item) => item.id !== "trend"),
      { setupTags: [latest.setupName] }
    )
  ];
  for (const trade of trades) {
    const saved = await page.request.post("/api/trades", { data: trade });
    expect(saved.status()).toBe(200);
  }
  const diskPath = join(process.env.E2E_DATA_DIR!, "trade-logs.json");
  const diskBefore = await readFile(diskPath, "utf8");
  const consoleErrors: string[] = [];
  page.on("pageerror", (error) => consoleErrors.push(error.message));
  await page.goto("/journal/branden/strategy-analyzer");
  await expect(
    page.getByRole("heading", { name: "Strategy Analyzer", exact: true })
  ).toBeVisible();
  await expect(
    page
      .getByRole("navigation", { name: "Branden journal navigation" })
      .getByRole("link", { name: "Strategy Analyzer", exact: true })
  ).toHaveAttribute("aria-current", "page");
  await page
    .getByLabel("Setup / version")
    .selectOption(analyzerTemplate.setupName);
  const subset = page.getByRole("table", {
    name: "Subset performance",
    exact: true
  });
  const all = subset
    .getByRole("row")
    .filter({
      has: page.getByRole("rowheader", {
        name: "All scoped trades",
        exact: true
      })
    });
  const matching = subset
    .getByRole("row")
    .filter({
      has: page.getByRole("rowheader", {
        name: "Matching criteria",
        exact: true
      })
    });
  await expect(all.getByRole("cell").first()).toHaveText("6");
  await expect(all).toContainText("50.0%");
  await expect(all).toContainText("$200.00");
  const comparisons = page.getByRole("table", {
    name: "Criterion comparisons",
    exact: true
  });
  const trend = comparisons
    .getByRole("row")
    .filter({ hasText: "Trade with trend" });
  await expect(trend.getByRole("cell").nth(1)).toContainText("0.0%");
  await expect(trend.getByRole("cell").nth(2)).toContainText("66.7%");
  await expect(trend.getByRole("cell").nth(3)).toContainText("-66.7 pp");
  await expect(trend.getByRole("cell").nth(4)).toHaveText("1");
  const volumeFilter = page.getByLabel("Require Entry: Volume confirmed", {
    exact: true
  });
  const trendFilter = page.getByLabel("Require Entry: Trade with trend", {
    exact: true
  });
  await volumeFilter.selectOption("met");
  await trendFilter.selectOption("not-met");
  await expect(matching.getByRole("cell").first()).toHaveText("2");
  await expect(matching).toContainText("100.0%");
  await expect(matching).toContainText("$300.00");
  await expect(matching).toContainText("+1.50R");
  await expect(trend.getByRole("cell").nth(1)).toContainText("1 trade");
  const details = page.getByRole("table", {
    name: "Matching trades",
    exact: true
  });
  await expect(details.getByRole("row")).toHaveCount(3);
  await expect(details).toContainText("ALPHA");
  await expect(details).toContainText("BETA");
  await expect(details).toContainText("Incomplete");
  await expect(
    details.getByRole("link", { name: "Review ALPHA (opens in a new tab)" })
  ).toHaveAttribute("href", /dashboard\?tradeId=/);
  await expect(details.getByRole("link").first()).toHaveAttribute(
    "target",
    "_blank"
  );
  await page.screenshot({
    path: testInfo.outputPath("strategy-analyzer-desktop.png"),
    fullPage: true
  });

  // Detail-only outcome filters must not change the performance denominator.
  await page.getByLabel("Show outcome").selectOption("loss");
  await expect(
    page.getByText("No matching trades to show.", { exact: false })
  ).toBeVisible();
  await expect(matching.getByRole("cell").first()).toHaveText("2");
  await page.getByLabel("Show outcome").selectOption("");
  await page.getByLabel("Find symbol").fill("beta");
  await expect(details.getByRole("row")).toHaveCount(2);
  await page.getByLabel("Find symbol").fill("");
  await page.getByRole("button", { name: "Clear criteria filters" }).click();
  const pointFilter = page.getByLabel("Require Entry: Valuation (max 4)", {
    exact: true
  });
  await pointFilter.selectOption("partial");
  await expect(details.getByRole("row")).toHaveCount(2);
  await expect(details).toContainText("BETA");
  await expect(details).toContainText("2/4");
  await page.getByRole("button", { name: "Clear criteria filters" }).click();
  await trendFilter.selectOption("unknown");
  await expect(details).toContainText("UNKNOWN");
  await expect(matching.getByRole("cell").last()).toHaveText("0/1");
  await page.getByRole("button", { name: "Clear criteria filters" }).click();
  await page
    .getByRole("combobox", { name: "Portfolio", exact: true })
    .selectOption("Other Analyzer");
  await expect(all.getByRole("cell").first()).toHaveText("1");
  await expect(all).toContainText("$999.00");
  await expect(page.getByLabel("Viewing portfolio")).toHaveValue(
    "Other Analyzer"
  );
  await page.getByLabel("Viewing portfolio").selectOption("Analyzer Test");
  await expect(all.getByRole("cell").first()).toHaveText("6");
  await page.getByLabel("Setup / version").selectOption(latest.setupName);
  await expect(
    comparisons.getByRole("row").filter({ hasText: "Trade with trend" })
  ).toHaveCount(0);
  await expect(all.getByRole("cell").first()).toHaveText("1");
  await expect(all).toContainText("-$500.00");
  await page
    .getByLabel("Setup / version")
    .selectOption(analyzerTemplate.setupName);
  await page.getByLabel("Entry from", { exact: true }).fill("2026-10-01");
  await page.getByLabel("Entry through", { exact: true }).fill("2026-09-01");
  await expect(page.getByRole("main").getByRole("alert")).toHaveText(
    "Entry from must be on or before Entry through."
  );
  await page.getByLabel("Entry from", { exact: true }).fill("");
  await page.getByLabel("Entry through", { exact: true }).fill("");
  await page.setViewportSize({ width: 390, height: 844 });
  await expect(
    page.getByRole("heading", { name: "Strategy Analyzer", exact: true })
  ).toBeVisible();
  await expect(page.getByLabel("Setup / version")).toBeVisible();
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= window.innerWidth + 1
    )
  ).toBe(true);
  await page.screenshot({
    path: testInfo.outputPath("strategy-analyzer-mobile.png"),
    fullPage: true
  });
  expect(await readFile(diskPath, "utf8")).toBe(diskBefore);
  expect(consoleErrors).toEqual([]);
});

test("load failures explain the problem and retry recovers; empty data explains how to begin", async ({
  page
}) => {
  expect(
    (
      await page.request.post("/api/login", {
        data: { userId: "branden", password: process.env.E2E_PASSWORD }
      })
    ).status()
  ).toBe(200);
  await page.route("**/api/journal/branden/trade-log", (route) =>
    route.fulfill({ status: 500, json: { error: "Synthetic load failure" } })
  );
  await page.goto("/journal/branden/strategy-analyzer");
  await expect(page.getByRole("main").getByRole("alert")).toContainText(
    "Synthetic load failure"
  );
  await page.route("**/api/journal/branden/trade-log", (route) =>
    route.fulfill({
      json: {
        user: { id: "branden" },
        trades: [],
        setupChecklists: [],
        portfolios: []
      }
    })
  );
  await page.getByRole("button", { name: "Try again", exact: true }).click();
  await expect(
    page.getByRole("heading", {
      name: "Start with a setup and recorded trades"
    })
  ).toBeVisible();
});
