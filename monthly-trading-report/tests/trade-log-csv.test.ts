import assert from "node:assert/strict";
import test from "node:test";
import { buildTradeLogCsv, tradeLogCsvFilename } from "../lib/trade-log-csv";
import type { TradeLogEntry } from "../lib/types";

function trade(overrides: Partial<TradeLogEntry> = {}): TradeLogEntry {
  return {
    id: "trade-1", userId: "branden", importSource: "CF statement", importRowKey: "source-1",
    symbol: "FRO", side: "LONG", status: "WIN", entryDate: "2026-09-01", exitDate: "2026-09-03",
    openTime: "09:31", closeTime: "15:40", avgEntry: 44.38, exitPrice: 46, stopPrice: 41,
    takeProfitPrice: 50.28, shares: 55, commission: 2.5, usedMargin: 1_200, risk: 185, pnl: 89.1,
    rMultiple: 0.4816, returnPercent: 3.65, daysInTrade: 2, setupTags: ["CANSLIM", "Breakout"],
    mistakeTags: ["Late entry"], customTags: ["Reviewed"], manualGrade: "B", portfolioTag: "CF Statement",
    emotion: "Calm", tradeQuality: "Good", checklistItems: [], notes: "Legacy notes",
    reviewSections: {
      setup: "Flat base", entry: "Bought the trigger", exit: "Sold into strength",
      didRight: "Sized from the stop", didWrong: "Added late", general: "Follow the plan next time"
    },
    screenshots: ["/api/trades/trade-1/screenshots/image-1"], chartLinks: ["https://charts.example/FRO"],
    executions: [{ id: "fill-1", type: "ENTRY", date: "2026-09-01", time: "09:31", side: "LONG", shares: 55,
      price: 44.38, pnl: 0, commission: 1.25, source: "CF statement", sourceKey: "fill-key-1" }],
    hidden: false, groupId: "", groupRole: "none", createdAt: "2026-09-01T00:00:00Z", updatedAt: "2026-09-03T00:00:00Z",
    ...overrides
  };
}

const context = {
  startDate: "2026-09-01", endDate: "2026-09-07", portfolio: "CF Statement", baseUrl: "https://monthly-trading-report.vercel.app"
};

test("exports lifecycle, selected-period, review, screenshot, excursion, and execution evidence", () => {
  const lifecycle = trade();
  const periodTrade = trade({ pnl: 40, rMultiple: 0.2162, status: "BREAKEVEN" });
  const csv = buildTradeLogCsv([{
    trade: lifecycle,
    periodTrade,
    grade: "B",
    reviewStatus: "Complete",
    excursion: {
      status: "AVAILABLE", reason: "", algorithmVersion: "v1", inputHash: "hash", provider: "FMP",
      providerSymbol: "FRO", instrumentMode: "EXACT", instrumentLabel: "Listed security", interval: "5min",
      asOf: "2026-09-03", isOpen: false, mfeDollars: 120, maeDollars: -35, mfeR: 0.65, maeR: -0.19,
      mfeTimestamp: "2026-09-02 10:00", maeTimestamp: "2026-09-01 10:00", priceScale: 1, barsEvaluated: 50
    }
  }], context);

  assert.match(csv, /"TRADE","trade-1"/);
  assert.match(csv, /"89.1","0.4816","2026-09-01","2026-09-07","40","0.2162"/);
  assert.match(csv, /"Flat base","Bought the trigger","Sold into strength","Sized from the stop","Added late","Follow the plan next time"/);
  assert.match(csv, /https:\/\/monthly-trading-report\.vercel\.app\/api\/trades\/trade-1\/screenshots\/image-1/);
  assert.match(csv, /"AVAILABLE","120","0.65","-35","-0.19"/);
  assert.match(csv, /"EXECUTION","trade-1","fill-1"/);
  assert.match(csv, /"ENTRY","2026-09-01","09:31","LONG","55","44.38","0","1.25","CF statement","fill-key-1"/);
});

test("formula-like review text is neutralized for spreadsheet safety", () => {
  const unsafe = trade({ reviewSections: { setup: "=HYPERLINK(\"bad\")", entry: "", exit: "", didRight: "", didWrong: "", general: "" } });
  const csv = buildTradeLogCsv([{ trade: unsafe, periodTrade: unsafe, grade: "B", reviewStatus: "Needs Review" }], context);
  assert.match(csv, /"'=HYPERLINK\(""bad""\)"/);
});

test("filename identifies portfolio and selected date range", () => {
  assert.equal(
    tradeLogCsvFilename(context),
    "branden-trade-log-CF-Statement-2026-09-01-to-2026-09-07.csv"
  );
});
