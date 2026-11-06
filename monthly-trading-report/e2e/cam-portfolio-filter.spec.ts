import { expect, test } from "@playwright/test";

test("Cam can filter historical and unassigned portfolios in the journal", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", error => errors.push(error.message));
  const login = await page.request.post("/api/login", {
    data: { userId: "cam", password: process.env.E2E_PASSWORD }
  });
  expect(login.ok()).toBe(true);
  const fixture = [
    { ticker: "MAIN", portfolioTag: " main " },
    { ticker: "HIST", portfolioTag: "Retired account" },
    { ticker: "NONE", portfolioTag: "  " },
    { ticker: "MISS" }
  ];
  const saved = await page.request.put("/api/cam-journal", { data: { state: {
    trades: fixture.map((trade, index) => ({ id: String(index + 1), date: "2026-01-01",
      status: "Closed", direction: "Long", ...trade })),
    portfolios: [{ id: "main", name: "Main" }]
  } } });
  expect(saved.ok()).toBe(true);
  await page.goto("/cam-journal/index.html");
  await expect(page.locator("main")).toHaveAttribute("aria-busy", "false");

  const picker = (id: string) => page.locator(`.multi-check[data-select-id="${id}"]`);
  const report = picker("reportPortfolio");
  const count = page.locator("#reportTradeCount");
  await expect(count).toHaveText("4");
  await report.getByRole("button").click();
  const main = report.getByRole("checkbox", { name: /^main$/i });
  await expect(main).toHaveCount(1);
  await main.check();
  await expect(count).toHaveText("1");
  await expect(report.getByRole("button")).toHaveText("Main");
  await main.uncheck();
  await expect(count).toHaveText("4");
  await report.getByRole("checkbox", { name: "No portfolio", exact: true }).check();
  await expect(count).toHaveText("2");

  await page.getByRole("button", { name: "Trade Log", exact: true }).click();
  const log = picker("logReportPortfolio");
  await expect(log.getByRole("button")).toHaveText("No portfolio");
  const trades = page.locator("#tradeList tr");
  await expect(trades).toHaveCount(2);
  await expect(page.locator("#tradeList")).toContainText("MISS");
  await log.getByRole("button").click();
  await log.getByRole("checkbox", { name: "Retired account", exact: true }).check();
  await expect(trades).toHaveCount(3);
  await log.getByRole("checkbox", { name: "No portfolio", exact: true }).uncheck();
  await expect(trades).toHaveCount(1);
  await expect(log.getByRole("button")).toHaveText("Retired account");

  await page.getByRole("button", { name: "Playbook", exact: true }).click();
  const playbook = picker("playbookPortfolio");
  await playbook.getByRole("button").click();
  await playbook.getByRole("checkbox", { name: "No portfolio", exact: true }).check();
  await expect(page.locator("#playbookPageSummary")).toContainText("of 2");
  expect(errors).toEqual([]);
});
