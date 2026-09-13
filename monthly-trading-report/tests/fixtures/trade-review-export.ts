import type { TradeLogEntry, SetupChecklistTemplate } from "../../lib/types";
import type { AiReview, ReviewEvidence } from "../../lib/trade-review-export";
import { unavailableTradeExcursion } from "../../lib/trade-excursion";
export function trade(overrides: Partial<TradeLogEntry> = {}): TradeLogEntry {
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

export const templates: SetupChecklistTemplate[] = [{ id: "s1", setupName: "CANSLIM", description: "", criteria: [], groups: [],
  gradeBands: [{ id: "A", label: "A", minScore: 8, maxScore: null }, { id: "C", label: "C", minScore: 0, maxScore: 7 }],
  knowledgeSources: [{ id: "source", title: "Breakout rules", sourceType: "notes", url: "", content: "Enter near the pivot; avoid chasing extended breakouts.", active: true, createdAt: "", updatedAt: "" }],
  strategyExamples: [{ id: "ex1", symbol: "EXAMPLE", setupType: "CANSLIM", quality: "failed", outcome: "Failed breakout", source: "Journal", sourceUrl: "", notes: "Entry was extended.", screenshots: [], active: true, createdAt: "", updatedAt: "" }]
}];
export function evidence(): ReviewEvidence { return { images: {}, excursions: { "trade-1": unavailableTradeExcursion(trade(), "No bars available", { asOf: "2026-09-03" }) } }; }
export function review(): AiReview { return {
  overallTakeaway: "The selected trade shows a late add after the initial entry. The immediate focus is to define the add trigger before entering.",
  keyThemes: ["Risk was defined before entry, but the add was late."], improved: ["Initial risk was documented. No prior period was supplied to establish improvement."],
  needsWork: ["The late add needs a written trigger."], bottomLine: "Keep the risk planning and make the add decision explicit.",
  exposureAnalysis: { summary: "The sample contains one energy-shipping equity, so it does not establish concentrated or correlated exposure.", groups: [{
    label: "Energy shipping (inferred)", type: "industry", symbols: ["FRO"], evidenceTradeIds: ["trade-1"],
    performance: "One winning trade, +0.48R.", correlation: "A single observation cannot show shared movement.",
    takeaway: "Track this exposure only if additional related names are added.", confidence: "medium"
  }] },
  workOn: { primaryFocus: "Define the add trigger before entering. This is the clearest execution gap in this sample.", priorities: [{
    scope: "single_observation", issue: "Late add", evidenceTradeIds: ["trade-1"], evidence: "The trader recorded Added late in What I did wrong. This is one observed event, not an established recurring pattern.",
    outcomeImpact: "An extended add can worsen average entry and increase exposure away from the planned trigger. The supplied evidence does not establish a recoverable dollar amount.",
    rule: "Write the add trigger and maximum total risk before the initial entry. Add only when both conditions remain valid.",
    measure: "For the next five trades, record whether an add occurred and whether it met the written trigger. Target full adherence, then review the exceptions.", confidence: "medium"
  }] },
  tradeReviews: { "trade-1": { mainLesson: "Define adds before entry." } }
}; }
