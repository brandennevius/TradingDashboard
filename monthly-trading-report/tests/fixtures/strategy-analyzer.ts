import { trade } from "./trade-review-export";
import type {
  SetupChecklistTemplate,
  TradeChecklistItem,
  TradeLogEntry
} from "../../lib/types";

export const analyzerTemplate: SetupChecklistTemplate = {
  id: "analyzer-breakout",
  setupName: "Analyzer Breakout",
  description: "Synthetic strategy analysis example",
  familyId: "analyzer-breakout",
  familyName: "Analyzer Breakout",
  version: 1,
  gradeBands: [
    { id: "a", label: "A", minScore: 6, maxScore: null },
    { id: "incomplete", label: "Incomplete", minScore: 0, maxScore: 5 }
  ],
  groups: [
    {
      id: "entry",
      name: "Entry",
      criteria: [
        {
          id: "volume",
          criteria: "Volume confirmed",
          points: 1,
          inputType: "boolean"
        },
        {
          id: "trend",
          criteria: "Trade with trend",
          points: 1,
          inputType: "boolean"
        },
        {
          id: "valuation",
          criteria: "Valuation",
          points: 4,
          inputType: "points"
        }
      ]
    }
  ],
  criteria: []
};
analyzerTemplate.criteria = analyzerTemplate.groups[0].criteria;

export function analyzerAnswers(
  volume: boolean,
  trend: boolean,
  score: number
): TradeChecklistItem[] {
  return analyzerTemplate.criteria.map((item) => ({
    ...item,
    groupName: "Entry",
    met:
      item.id === "volume" ? volume : item.id === "trend" ? trend : score > 0,
    score: item.id === "valuation" ? score : undefined
  }));
}

export function analyzerTrade(
  id: string,
  pnl: number,
  checklistItems: TradeChecklistItem[],
  overrides: Partial<TradeLogEntry> = {}
) {
  return trade({
    id,
    symbol: id.toUpperCase(),
    importSource: "synthetic-analyzer",
    importRowKey: id,
    status: pnl > 0 ? "WIN" : pnl < 0 ? "LOSS" : "BREAKEVEN",
    pnl,
    risk: 100,
    commission: 0,
    rMultiple: pnl / 100,
    manualGrade: "",
    setupTags: [analyzerTemplate.setupName],
    portfolioTag: "Analyzer Test",
    checklistItems,
    screenshots: [],
    executions: [],
    ...overrides
  });
}

export function analyzerTrades() {
  return [
    analyzerTrade("alpha", 200, analyzerAnswers(true, false, 4)),
    analyzerTrade("beta", 100, analyzerAnswers(true, false, 2)),
    analyzerTrade("gamma", -100, analyzerAnswers(true, true, 4)),
    analyzerTrade("delta", -50, analyzerAnswers(false, true, 0)),
    analyzerTrade("epsilon", 0, analyzerAnswers(false, false, 0)),
    analyzerTrade("unknown", 50, [], { risk: 0, rMultiple: 0 })
  ];
}
