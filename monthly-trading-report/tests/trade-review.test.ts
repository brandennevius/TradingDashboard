import assert from "node:assert/strict";
import test from "node:test";
import { hasCompletedTradeReview, requiredTradeReviewSections, tradeNeedsReview } from "../lib/trade-review";
import type { SetupChecklistTemplate, TradeLogEntry } from "../lib/types";

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


test("all five review fields are required; general review is optional", () => {
  const review = { ...trade().reviewSections!, general: "" };
  assert.equal(hasCompletedTradeReview(review), true);
  for (const key of requiredTradeReviewSections) {
    for (const missing of ["", " \n "]) {
      assert.equal(hasCompletedTradeReview({ ...review, [key]: missing, general: "Extra notes" }), false, key);
    }
  }
  assert.equal(hasCompletedTradeReview(undefined), false);
});

test("complete trade can have a manual grade without checklist items or general review", () => {
  assert.equal(tradeNeedsReview(trade({ reviewSections: { ...trade().reviewSections!, general: "" } }), []), false);
});

test("each missing requirement keeps the trade in Needs Review", () => {
  const cases: Partial<TradeLogEntry>[] = [
    { risk: 0 }, { risk: NaN },
    { manualGrade: "" }, { manualGrade: "  " }, { manualGrade: "Unscored" },
    { setupTags: [] }, { setupTags: ["  "] },
    { screenshots: [] }, { screenshots: ["  "] },
    { reviewSections: undefined, notes: "Legacy review" },
    ...requiredTradeReviewSections.map((key) => ({ reviewSections: { ...trade().reviewSections!, [key]: "" } }))
  ];
  for (const override of cases) {
    assert.equal(tradeNeedsReview(trade(override), []), true, JSON.stringify(override));
  }
});

test("chart links alone cannot replace a screenshot", () => {
  assert.equal(tradeNeedsReview(trade({ screenshots: [], chartLinks: ["https://charts.example/FRO"] }), []), true);
  assert.equal(tradeNeedsReview(trade({ chartLinks: [] }), []), false);
});

test("a calculated grade qualifies, but checklist points without a grade do not", () => {
  const template: SetupChecklistTemplate = {
    id: "setup-1", setupName: "CANSLIM", description: "", criteria: [], groups: [],
    gradeBands: [{ id: "c", label: "C", minScore: 0, maxScore: null }]
  };
  const scored = trade({ manualGrade: "", checklistItems: [{ id: "1", criteria: "Breakout", points: 10, met: false }] });
  assert.equal(tradeNeedsReview(scored, [template]), false);
  assert.equal(tradeNeedsReview(scored, []), true);
  assert.equal(tradeNeedsReview(scored, [{ ...template, gradeBands: [] }]), true);
});
