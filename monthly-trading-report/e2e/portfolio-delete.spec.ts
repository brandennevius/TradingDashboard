import { test, expect } from "@playwright/test";

test("portfolio delete confirms scope, downloads a backup, persists, and rejects non-owner access", async ({ page, playwright }) => {
  const endpoint = "/api/settings/branden-portfolios";
  const anonymous = await playwright.request.newContext({ baseURL: "http://127.0.0.1:3100" });
  try {
    expect((await anonymous.delete(endpoint, { data: { portfolio: "Legacy" } })).status()).toBe(401);
    expect((await anonymous.post("/api/login", { data: { userId: "cam", password: process.env.E2E_PASSWORD } })).status()).toBe(200);
    expect((await anonymous.delete(endpoint, { data: { portfolio: "Legacy" } })).status()).toBe(403);
  } finally { await anonymous.dispose(); }
  expect((await page.request.post("/api/login", { data: { userId: "branden", password: process.env.E2E_PASSWORD } })).status()).toBe(200);
  expect((await page.request.put(endpoint, { data: { portfolios: ["Legacy", "Keep"], defaultPortfolio: "Legacy" } })).status()).toBe(200);
  expect((await page.request.delete(endpoint, { data: { portfolio: "" } })).status()).toBe(400);
  expect((await page.request.delete(endpoint, { data: { portfolio: "Unknown" } })).status()).toBe(404);

  // A real stored trade must disappear from the unfiltered open-position API too.
  const created = await page.request.post("/api/trades", { data: {
    symbol: "LEGACY", entryDate: "2026-06-10", portfolioTag: "Legacy", side: "LONG", status: "OPEN", risk: 250
  } });
  expect(created.ok()).toBe(true);
  await page.goto("/journal/branden/portfolios");
  const remove = page.getByRole("button", { name: "Delete portfolio Legacy", exact: true });
  await expect(remove).toBeVisible();
  page.once("dialog", dialog => dialog.dismiss());
  await remove.click();
  await expect(remove).toBeVisible();
  expect((await (await page.request.get(endpoint)).json()).portfolios).toContain("Legacy");

  // A failed backup must never submit a deletion.
  await page.route("**/api/settings/branden-backup", route => route.fulfill({ status: 500, body: "{}" }));
  page.once("dialog", dialog => dialog.accept());
  await remove.click();
  await expect(page.getByText("Could not download a journal backup. Portfolio was not deleted.")).toBeVisible();
  expect((await (await page.request.get(endpoint)).json()).portfolios).toContain("Legacy");
  await page.unroute("**/api/settings/branden-backup");

  page.once("dialog", dialog => dialog.accept());
  const download = page.waitForEvent("download");
  const deleted = page.waitForResponse(response => response.url().endsWith(endpoint) && response.request().method() === "DELETE");
  await remove.click();
  expect((await download).suggestedFilename()).toMatch(/^branden-journal-backup-/);
  expect((await deleted).status()).toBe(200);
  await expect(remove).toHaveCount(0);
  await expect(page.getByRole("button", { name: "Delete portfolio Keep", exact: true })).toBeVisible();
  const settings = await (await page.request.get(endpoint)).json();
  expect(settings.portfolios).toEqual(["Keep"]);
  expect(settings.defaultPortfolio).toBe("");
  const open = await (await page.request.get("/api/public/branden-open-positions")).json();
  expect(open.trades.filter((trade: { portfolioTag: string }) => trade.portfolioTag === "Legacy")).toEqual([]);
  await page.reload();
  await expect(remove).toHaveCount(0);
  await expect(page.getByLabel("Viewing portfolio")).toHaveValue("");
});
