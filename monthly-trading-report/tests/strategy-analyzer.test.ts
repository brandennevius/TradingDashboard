import assert from "node:assert/strict";
import test from "node:test";
import {
  analyzeStrategy,
  criterionAnswers,
  strategyCriteria,
  strategySetupOptions,
  type StrategyScope
} from "../lib/strategy-analyzer";
import {
  analyzerAnswers,
  analyzerTemplate,
  analyzerTrade,
  analyzerTrades
} from "./fixtures/strategy-analyzer";

const scope: StrategyScope = {
  setup: analyzerTemplate.setupName,
  portfolio: "",
  start: "",
  end: "",
  side: ""
};

test("criterion cohorts reveal winners missing a rule without excluding incomplete or unscored trades", () => {
  const result = analyzeStrategy(
    analyzerTrades(),
    [analyzerTemplate],
    scope,
    {}
  );
  assert.deepEqual(result.summary.baseline, {
    count: 6,
    wins: 3,
    losses: 2,
    breakeven: 1,
    winRate: 0.5,
    netPnl: 200,
    averagePnl: 200 / 6,
    averageR: 0.3,
    rCount: 5
  });
  const trend = result.comparisons.find(
    (row) => row.criterion.label === "Trade with trend"
  )!;
  assert.deepEqual(trend.met, {
    count: 2,
    wins: 0,
    losses: 2,
    breakeven: 0,
    winRate: 0,
    netPnl: -150,
    averagePnl: -75,
    averageR: -0.75,
    rCount: 2
  });
  assert.equal(trend.notMet.count, 3);
  assert.equal(trend.notMet.wins, 2);
  assert.equal(trend.notMet.winRate, 2 / 3);
  assert.equal(trend.notMet.averageR, 1);
  assert.equal(trend.winRateDifference, -2 / 3);
  assert.equal(trend.unknown, 1);
});

test("combined conditions use AND and each comparison holds other conditions fixed", () => {
  const initial = analyzeStrategy(
    analyzerTrades(),
    [analyzerTemplate],
    scope,
    {}
  );
  const [volume, trend] = initial.criteria;
  const result = analyzeStrategy(analyzerTrades(), [analyzerTemplate], scope, {
    [volume.key]: "met",
    [trend.key]: "not-met"
  });
  assert.deepEqual(
    result.matching.map((trade) => trade.symbol),
    ["ALPHA", "BETA"]
  );
  assert.equal(result.summary.matching.netPnl, 300);
  assert.equal(result.summary.matching.winRate, 1);
  assert.equal(result.summary.matching.averageR, 1.5);
  assert.equal(result.summary.others.count, 4);
  // The trend row still shows GAMMA, despite requiring trend to be NOT met above.
  const comparison = result.comparisons.find(
    (row) => row.criterion.key === trend.key
  )!;
  assert.equal(comparison.met.count, 1);
  assert.equal(comparison.met.netPnl, -100);
  assert.equal(comparison.notMet.count, 2);
  assert.equal(comparison.unknown, 0);
  assert.equal(result.comparisons[0].notMet.count, 1); // EPSILON, holding trend not met
});

test("absent, conflicting, and changed historical answers are not treated as unchecked", () => {
  const original = analyzerTrades();
  const changed = analyzerTrade("changed", 10, [
    { ...analyzerAnswers(true, false, 4)[1], criteria: "Reworded trend rule" }
  ]);
  const conflict = analyzerTrade("conflict", 10, [
    analyzerAnswers(true, false, 4)[1],
    analyzerAnswers(true, true, 4)[1]
  ]);
  const trades = [...original, changed, conflict];
  const before = JSON.stringify({ trades, analyzerTemplate });
  const criteria = strategyCriteria(trades, [analyzerTemplate], scope.setup);
  const trend = criteria.find((item) => item.label === "Trade with trend")!;
  assert.equal(criterionAnswers(conflict).get(trend.key)?.state, "unknown");
  assert.equal(criteria.filter((item) => /trend/i.test(item.label)).length, 2);
  const result = analyzeStrategy(trades, [analyzerTemplate], scope, {
    [trend.key]: "unknown"
  });
  assert.deepEqual(
    result.matching.map((trade) => trade.symbol),
    ["UNKNOWN", "CHANGED", "CONFLICT"]
  );
  const unchecked = analyzeStrategy(trades, [analyzerTemplate], scope, {
    [trend.key]: "not-met"
  });
  assert.deepEqual(
    unchecked.matching.map((trade) => trade.symbol),
    ["ALPHA", "BETA", "EPSILON"]
  );
  assert.equal(JSON.stringify({ trades, analyzerTemplate }), before);
});

test("point criteria distinguish full, partial, zero and invalid or unrecorded scores", () => {
  const trades = analyzerTrades();
  const criteria = strategyCriteria(trades, [analyzerTemplate], scope.setup);
  const points = criteria.find((item) => item.inputType === "points")!;
  const analyze = (filter: "met" | "not-met" | "partial" | "unknown") =>
    analyzeStrategy(trades, [analyzerTemplate], scope, {
      [points.key]: filter
    });
  assert.deepEqual(
    analyze("met").matching.map((trade) => trade.symbol),
    ["ALPHA", "GAMMA"]
  );
  assert.deepEqual(
    analyze("partial").matching.map((trade) => trade.symbol),
    ["BETA"]
  );
  assert.deepEqual(
    analyze("not-met").matching.map((trade) => trade.symbol),
    ["BETA", "DELTA", "EPSILON"]
  );
  assert.deepEqual(
    analyze("unknown").matching.map((trade) => trade.symbol),
    ["UNKNOWN"]
  );
  for (const score of [undefined, NaN, -1, 5]) {
    const saved = analyzerTrade("invalid", 20, [
      { ...analyzerAnswers(true, true, 4)[2], score }
    ]);
    assert.equal(criterionAnswers(saved).get(points.key)?.state, "unknown");
  }
  const weightedBoolean = analyzerTrade("weighted", 20, [
    { ...analyzerAnswers(true, false, 4)[0], points: 4 }
  ]);
  assert.equal([...criterionAnswers(weightedBoolean).values()][0].state, "met");
});

test("scope isolates setup versions, primary tags, portfolio, direction, dates, owner and visible closed parents", () => {
  const original = analyzerTrades()[0];
  const variations = [
    { id: "open", status: "OPEN" as const },
    { id: "hidden", hidden: true },
    { id: "child", groupRole: "child" as const },
    { id: "cam", userId: "cam" },
    { id: "version", setupTags: ["Analyzer Breakout v2"] },
    { id: "secondary", setupTags: ["Other", analyzerTemplate.setupName] },
    { id: "other-portfolio", portfolioTag: "Other" },
    { id: "short", side: "SHORT" as const },
    { id: "before", entryDate: "2026-08-31" },
    { id: "after", entryDate: "2026-09-02" }
  ];
  const trades = [
    original,
    ...variations.map((change) => ({ ...original, ...change }))
  ];
  const result = analyzeStrategy(
    trades,
    [analyzerTemplate],
    {
      ...scope,
      portfolio: "Analyzer Test",
      side: "LONG",
      start: "2026-09-01",
      end: "2026-09-01"
    },
    {}
  );
  assert.deepEqual(
    result.baseline.map((trade) => trade.id),
    ["alpha"]
  );
  assert.equal(result.coverage.open, 1);
  assert.deepEqual(
    strategySetupOptions(trades, [{ ...analyzerTemplate, archived: true }]).map(
      (option) => [option.name, option.archived]
    ),
    [
      ["Analyzer Breakout v2", false],
      ["Other", false],
      ["Analyzer Breakout", true]
    ]
  );
  assert.equal(
    analyzeStrategy(
      trades,
      [analyzerTemplate],
      { ...scope, setup: "Analyzer Breakout v2" },
      {}
    ).baseline.length,
    1
  );
});

test("empty groups use null rates and riskless trades never turn into zero-R observations", () => {
  const result = analyzeStrategy(
    [
      analyzerTrade("riskless", 50, analyzerAnswers(true, false, 4), {
        risk: 0,
        rMultiple: 0
      })
    ],
    [analyzerTemplate],
    scope,
    {}
  );
  assert.equal(result.summary.baseline.winRate, 1);
  assert.equal(result.summary.baseline.averageR, null);
  assert.equal(result.summary.baseline.rCount, 0);
  assert.equal(result.comparisons[0].notMet.winRate, null);
  assert.equal(result.comparisons[0].winRateDifference, null);
  const empty = analyzeStrategy([], [analyzerTemplate], scope, {});
  assert.equal(empty.summary.matching.winRate, null);
  assert.equal(empty.summary.matching.averagePnl, null);
  assert.equal(empty.summary.matching.averageR, null);
  assert.equal(empty.criteria.length, 3);
});
