import type {
  SetupChecklistTemplate,
  TradeChecklistItem,
  TradeLogEntry
} from "./types";

export type CriterionState = "met" | "not-met" | "partial" | "unknown";
export type CriterionFilter = "any" | CriterionState;
export type CriterionFilters = Record<string, CriterionFilter>;
export type StrategyScope = {
  setup: string;
  portfolio: string;
  start: string;
  end: string;
  side: string;
};
export type AnalyzerCriterion = {
  key: string;
  label: string;
  group: string;
  inputType: "boolean" | "points";
  points: number;
};
export type StrategySummary = {
  count: number;
  wins: number;
  losses: number;
  breakeven: number;
  winRate: number | null;
  netPnl: number;
  averagePnl: number | null;
  averageR: number | null;
  rCount: number;
};

const normalize = (value: string) =>
  value.trim().replace(/\s+/g, " ").toLowerCase();
export const primaryStrategy = (trade: TradeLogEntry) =>
  trade.setupTags[0]?.trim() || "";

function criterionDefinition(
  item: Pick<
    TradeChecklistItem,
    "criteria" | "points" | "inputType" | "groupName"
  >
): AnalyzerCriterion | null {
  if (
    !item.criteria.trim() ||
    !Number.isFinite(item.points) ||
    item.points <= 0
  )
    return null;
  const group = item.groupName?.trim() || "Checklist";
  const inputType = item.inputType === "points" ? "points" : "boolean";
  // Saved definitions, rather than mutable template IDs, identify comparable answers.
  // Rewording, regrouping, changing type, or changing maximum points stays separate.
  return {
    key: JSON.stringify([
      normalize(group),
      normalize(item.criteria),
      inputType,
      item.points
    ]),
    label: item.criteria.trim(),
    group,
    inputType,
    points: item.points
  };
}

export function strategyCriteria(
  trades: TradeLogEntry[],
  templates: SetupChecklistTemplate[],
  setup: string
): AnalyzerCriterion[] {
  const definitions = new Map<string, AnalyzerCriterion>();
  const add = (item: Parameters<typeof criterionDefinition>[0]) => {
    const definition = criterionDefinition(item);
    if (definition) definitions.set(definition.key, definition);
  };
  const template = templates.find(
    (item) => normalize(item.setupName) === normalize(setup)
  );
  if (template) {
    const groups = template.groups.length
      ? template.groups
      : [{ name: "Checklist", criteria: template.criteria }];
    for (const group of groups)
      for (const criterion of group.criteria)
        add({ ...criterion, groupName: group.name });
  }
  for (const trade of trades) {
    if (
      trade.userId === "branden" &&
      !trade.hidden &&
      trade.groupRole !== "child" &&
      normalize(primaryStrategy(trade)) === normalize(setup)
    ) {
      for (const item of trade.checklistItems) add(item);
    }
  }
  return [...definitions.values()];
}

function savedAnswer(item: TradeChecklistItem): CriterionState {
  if (item.inputType !== "points")
    return typeof item.met === "boolean"
      ? item.met
        ? "met"
        : "not-met"
      : "unknown";
  if (
    typeof item.score !== "number" ||
    !Number.isFinite(item.score) ||
    item.score < 0 ||
    item.score > item.points
  )
    return "unknown";
  return item.score === item.points
    ? "met"
    : item.score === 0
      ? "not-met"
      : "partial";
}

export function criterionAnswers(
  trade: TradeLogEntry
): Map<string, { state: CriterionState; score?: number }> {
  const answers = new Map<string, { state: CriterionState; score?: number }>();
  for (const item of trade.checklistItems) {
    const definition = criterionDefinition(item);
    if (!definition) continue;
    const answer = {
      state: savedAnswer(item),
      score: item.inputType === "points" ? item.score : undefined
    };
    const previous = answers.get(definition.key);
    // Conflicting duplicate rows must not silently pick a convenient answer.
    answers.set(
      definition.key,
      previous &&
        (previous.state !== answer.state || previous.score !== answer.score)
        ? { state: "unknown" }
        : answer
    );
  }
  return answers;
}

export function strategyTradeR(trade: TradeLogEntry): number | null {
  const r =
    trade.risk > 0 && Number.isFinite(trade.risk) && Number.isFinite(trade.pnl)
      ? trade.pnl / trade.risk
      : NaN;
  return Number.isFinite(r) ? r : null;
}

export function summarizeStrategyTrades(
  trades: TradeLogEntry[]
): StrategySummary {
  const count = trades.length;
  const wins = trades.filter((trade) => trade.pnl > 0).length;
  const losses = trades.filter((trade) => trade.pnl < 0).length;
  const netPnl = trades.reduce((sum, trade) => sum + trade.pnl, 0);
  const multiples = trades
    .map(strategyTradeR)
    .filter((r): r is number => r !== null);
  return {
    count,
    wins,
    losses,
    breakeven: count - wins - losses,
    winRate: count ? wins / count : null,
    netPnl,
    averagePnl: count ? netPnl / count : null,
    rCount: multiples.length,
    averageR: multiples.length
      ? multiples.reduce((sum, r) => sum + r, 0) / multiples.length
      : null
  };
}

export function strategySetupOptions(
  trades: TradeLogEntry[],
  templates: SetupChecklistTemplate[]
) {
  const options = new Map<
    string,
    { name: string; archived: boolean; version: number; count: number }
  >();
  for (const template of templates)
    options.set(normalize(template.setupName), {
      name: template.setupName,
      archived: Boolean(template.archived),
      version: template.version || 1,
      count: 0
    });
  for (const trade of trades) {
    const name = primaryStrategy(trade);
    if (
      !name ||
      trade.userId !== "branden" ||
      trade.hidden ||
      trade.groupRole === "child"
    )
      continue;
    const key = normalize(name);
    const option = options.get(key) || {
      name,
      archived: false,
      version: 0,
      count: 0
    };
    if (
      ["WIN", "LOSS", "BREAKEVEN"].includes(trade.status) &&
      Number.isFinite(trade.pnl)
    )
      option.count++;
    options.set(key, option);
  }
  return [...options.values()].sort(
    (a, b) =>
      Number(a.archived) - Number(b.archived) || a.name.localeCompare(b.name)
  );
}

export function analyzeStrategy(
  trades: TradeLogEntry[],
  templates: SetupChecklistTemplate[],
  scope: StrategyScope,
  filters: CriterionFilters
) {
  const criteria = strategyCriteria(trades, templates, scope.setup);
  const scoped = trades.filter(
    (trade) =>
      Boolean(scope.setup) &&
      trade.userId === "branden" &&
      !trade.hidden &&
      trade.groupRole !== "child" &&
      normalize(primaryStrategy(trade)) === normalize(scope.setup) &&
      (!scope.portfolio || trade.portfolioTag === scope.portfolio) &&
      (!scope.side || trade.side === scope.side) &&
      (!scope.start || trade.entryDate >= scope.start) &&
      (!scope.end || trade.entryDate <= scope.end)
  );
  const baseline = scoped.filter(
    (trade) =>
      ["WIN", "LOSS", "BREAKEVEN"].includes(trade.status) &&
      Number.isFinite(trade.pnl)
  );
  const answers = new Map(
    baseline.map((trade) => [trade.id, criterionAnswers(trade)])
  );
  const stateFor = (trade: TradeLogEntry, key: string) =>
    answers.get(trade.id)?.get(key)?.state || "unknown";
  const active = criteria.filter(
    (criterion) => filters[criterion.key] && filters[criterion.key] !== "any"
  );
  function matches(trade: TradeLogEntry, except?: string) {
    return active.every((criterion) => {
      if (criterion.key === except) return true;
      const state = stateFor(trade, criterion.key);
      return filters[criterion.key] === "not-met"
        ? state === "not-met" || state === "partial"
        : state === filters[criterion.key];
    });
  }
  const matching = baseline.filter((trade) => matches(trade));
  const others = baseline.filter((trade) => !matches(trade));
  return {
    criteria,
    baseline,
    matching,
    others,
    answers,
    activeFilters: active.length,
    summary: {
      baseline: summarizeStrategyTrades(baseline),
      matching: summarizeStrategyTrades(matching),
      others: summarizeStrategyTrades(others)
    },
    coverage: {
      open: scoped.filter((trade) => trade.status === "OPEN").length,
      invalidPnl: scoped.filter(
        (trade) => trade.status !== "OPEN" && !Number.isFinite(trade.pnl)
      ).length,
      noChecklist: baseline.filter((trade) => !trade.checklistItems.length)
        .length
    },
    comparisons: criteria.map((criterion) => {
      // Hold every OTHER selected condition fixed so this criterion can still be compared fairly.
      const context = baseline.filter((trade) => matches(trade, criterion.key));
      const met = summarizeStrategyTrades(
        context.filter((trade) => stateFor(trade, criterion.key) === "met")
      );
      const notMet = summarizeStrategyTrades(
        context.filter((trade) =>
          ["not-met", "partial"].includes(stateFor(trade, criterion.key))
        )
      );
      const unknown = context.filter(
        (trade) => stateFor(trade, criterion.key) === "unknown"
      ).length;
      return {
        criterion,
        met,
        notMet,
        unknown,
        winRateDifference:
          met.winRate !== null && notMet.winRate !== null
            ? met.winRate - notMet.winRate
            : null,
        averageRDifference:
          met.averageR !== null && notMet.averageR !== null
            ? met.averageR - notMet.averageR
            : null
      };
    })
  };
}
