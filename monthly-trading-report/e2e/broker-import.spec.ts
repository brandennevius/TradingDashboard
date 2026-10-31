import { test, expect, type Page, type APIRequestContext } from "@playwright/test";
import { readFile } from "node:fs/promises";
import path from "node:path";

const password = process.env.E2E_PASSWORD!;
const fixture = (name: string) => path.join(process.cwd(), "e2e", "fixtures", name);

async function login(request: APIRequestContext, userId = "cam") {
  const response = await request.post("/api/login", { data: { userId, password } });
  expect(response.status()).toBe(200);
}
async function savedState(request: APIRequestContext) {
  const response = await request.get("/api/cam-journal");
  expect(response.status()).toBe(200);
  return (await response.json()).state;
}
async function openJournal(page: Page) {
  // A real GET response alone precedes boot/render completion; wait for the UI too.
  const loaded = page.waitForResponse(response => response.url().endsWith("/api/cam-journal") && response.request().method() === "GET");
  await page.goto("/cam-journal/index.html");
  expect((await loaded).status()).toBe(200);
  await expect(page.locator("main")).toHaveAttribute("aria-busy", "false");
  await expect(page.locator("#reportTradeCount")).toHaveText(/\d+/);
  await expect(page.locator("aside").getByLabel("Import Broker CSV", { exact: true })).toBeEnabled();
}
async function upload(page: Page, name: string) {
  await page.locator("aside").getByLabel("Import Broker CSV", { exact: true }).setInputFiles(fixture(name));
}
async function waitForSave(page: Page, action: () => Promise<unknown>) {
  const saved = page.waitForResponse(response => response.url().endsWith("/api/cam-journal")
    && response.request().method() === "PUT");
  await action();
  expect((await saved).status()).toBe(200);
}

test.beforeEach(async ({ page }) => {
  // Cam is an existing identity; credentials, server, storage and browser are per run/test.
  await login(page.request);
  const reset = await page.request.put("/api/cam-journal", { data: { state: {
    trades: [], setups: [], portfolios: [], tags: { secondary: [], mistakes: [] },
    monthlyReviews: {}, watchlistItems: []
  } } });
  expect(reset.status()).toBe(200);
  await openJournal(page);
});

test("synthetic import persists, reimport preserves executions and manual review", async ({ page, browser }) => {
  await waitForSave(page, () => upload(page, "synthetic-schwab.csv"));
  await expect(page.getByRole("status")).toContainText("New: 1");
  await expect(page.getByRole("status")).toContainText("File P/L: $28.00");
  await page.getByRole("button", { name: "Trade Log", exact: true }).click();
  const row = page.locator("#tradeList tr").filter({ hasText: "XYZ" });
  await expect(row).toHaveCount(1);
  await expect(row).toContainText("$28.00");
  await row.getByText("XYZ", { exact: true }).click();
  const dialog = page.getByRole("dialog", { name: /XYZ/ });
  await dialog.getByLabel("Stop", { exact: true }).fill("98");
  await dialog.getByLabel("Trade Notes", { exact: true }).fill("Synthetic review: wait for confirmation.");
  await waitForSave(page, () => dialog.getByRole("button", { name: "Save Changes", exact: true }).first().click());
  const before = await savedState(page.request);
  expect(before.trades).toHaveLength(1);
  const trade = before.trades[0];
  // Independent arithmetic: 10*(103-100) - 1 - 1 = $28. No parser-derived oracle.
  expect(trade).toMatchObject({ ticker: "XYZ", size: 10, direction: "Long", status: "Closed",
    entry: 100, exit: 103, pl: 28, stop: "98", notes: "Synthetic review: wait for confirmation." });
  expect(trade.rawExecutions).toHaveLength(2);
  expect(new Set(trade.executionKeys).size).toBe(2);
  expect(trade.rawExecutions.map((e: { signedQty: number }) => e.signedQty)).toEqual([10, -10]);

  await page.reload();
  await expect(page.locator("#reportTradeCount")).toHaveText("1");
  await waitForSave(page, () => upload(page, "synthetic-schwab.csv"));
  await expect(page.getByRole("status")).toContainText("New: 0. Updated: 0. Duplicates ignored: 1");
  expect((await savedState(page.request)).trades).toEqual(before.trades);
  const disk = JSON.parse(await readFile(path.join(process.env.E2E_DATA_DIR!, "app-settings.json"), "utf8"));
  expect(disk.camJournalState.trades).toEqual(before.trades);

  // Fresh browser has no localStorage cache: recovery must come from the real server.
  const fresh = await browser.newContext({ baseURL: "http://127.0.0.1:3100" });
  try {
    await login(fresh.request);
    const freshPage = await fresh.newPage();
    await openJournal(freshPage);
    await expect(freshPage.locator("#reportTradeCount")).toHaveText("1");
    await freshPage.getByRole("button", { name: "Trade Log", exact: true }).click();
    await freshPage.getByRole("cell", { name: "XYZ", exact: true }).click();
    await expect(freshPage.getByLabel("Trade Notes", { exact: true })).toHaveValue(trade.notes);
    await expect.poll(async () => Number(await freshPage.getByLabel("Stop", { exact: true }).inputValue())).toBe(98);
  } finally { await fresh.close(); }
});

test("unrecognized CSV explains supported input and leaves saved records unchanged", async ({ page }) => {
  await waitForSave(page, () => upload(page, "synthetic-schwab.csv"));
  const before = await savedState(page.request);
  const file = path.join(process.env.E2E_DATA_DIR!, "app-settings.json");
  const diskBefore = await readFile(file, "utf8");
  await upload(page, "invalid.csv");
  await expect(page.getByRole("status")).toContainText("No broker trades found.");
  await expect(page.getByRole("status")).toContainText("Thinkorswim/Schwab account statement");
  expect(await savedState(page.request)).toEqual(before);
  expect(await readFile(file, "utf8")).toBe(diskBefore);
  await page.reload();
  await expect(page.locator("#reportTradeCount")).toHaveText("1");
});

test("journal API rejects anonymous access and another trader's writes", async ({ page, playwright }) => {
  const before = await savedState(page.request);
  const anonymous = await playwright.request.newContext({ baseURL: "http://127.0.0.1:3100" });
  try {
    expect((await anonymous.get("/api/cam-journal")).status()).toBe(401);
    expect((await anonymous.put("/api/cam-journal", { data: { state: { trades: [] } } })).status()).toBe(401);
    await login(anonymous, "branden");
    expect((await anonymous.put("/api/cam-journal", { data: { state: { trades: [] } } })).status()).toBe(403);
    expect(await savedState(page.request)).toEqual(before);
  } finally { await anonymous.dispose(); }
});
