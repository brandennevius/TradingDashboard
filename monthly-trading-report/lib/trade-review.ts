import type { SetupChecklistTemplate, TradeLogEntry, TradeReviewSections } from "./types";

export const emptyTradeReviewSections: TradeReviewSections = {
  setup: "",
  entry: "",
  exit: "",
  didRight: "",
  didWrong: "",
  general: ""
};

export function normalizeTradeReviewSections(value: unknown): TradeReviewSections {
  const source = value && typeof value === "object" ? value as Partial<Record<keyof TradeReviewSections, unknown>> : {};
  return {
    setup: String(source.setup || ""),
    entry: String(source.entry || ""),
    exit: String(source.exit || ""),
    didRight: String(source.didRight || ""),
    didWrong: String(source.didWrong || ""),
    general: String(source.general || "")
  };
}

export function hasTradeReviewContent(value: TradeReviewSections) {
  return Object.values(value).some((section) => section.trim().length > 0);
}

export const requiredTradeReviewSections = ["setup", "entry", "exit", "didRight", "didWrong"] as const;

export function hasCompletedTradeReview(value: unknown) {
  const sections = normalizeTradeReviewSections(value);
  return requiredTradeReviewSections.every((key) => sections[key].trim().length > 0);
}

const legacyLabels: Array<{ label: string; key: keyof TradeReviewSections }> = [
  { label: "What did I do right", key: "didRight" },
  { label: "What did I do wrong", key: "didWrong" },
  { label: "Exit strategy", key: "exit" },
  { label: "General review", key: "general" },
  { label: "Setup", key: "setup" },
  { label: "Entry", key: "entry" },
  { label: "Exit", key: "exit" }
];

export function reviewSectionsFromLegacyNotes(notes: string) {
  const text = String(notes || "");
  const matches = legacyLabels
    .flatMap(({ label, key }) => {
      const expression = new RegExp(`(?:^|\\n)\\s*${label.replace(/[.*+?^${}()|[\\]\\]/g, "\\$&")}\\s*:\\s*`, "gi");
      return Array.from(text.matchAll(expression)).map((match) => ({ key, start: match.index! + match[0].length, labelStart: match.index! }));
    })
    .sort((a, b) => a.labelStart - b.labelStart);

  if (!matches.length) return normalizeTradeReviewSections(undefined);
  const sections = normalizeTradeReviewSections(undefined);
  matches.forEach((match, index) => {
    const value = text.slice(match.start, matches[index + 1]?.labelStart ?? text.length).trim();
    if (value && !sections[match.key]) sections[match.key] = value;
  });
  return sections;
}

export function resolvedTradeReviewSections(value: unknown, legacyNotes = "") {
  const sections = normalizeTradeReviewSections(value);
  return hasTradeReviewContent(sections) ? sections : reviewSectionsFromLegacyNotes(legacyNotes);
}

function setupTemplateFor(setupName: string, templates: SetupChecklistTemplate[]) {
  return templates.find((template) => template.setupName.trim().toLowerCase() === setupName.trim().toLowerCase());
}

export function tradeChecklistScore(trade: TradeLogEntry, templates: SetupChecklistTemplate[]) {
  const template = setupTemplateFor(trade.setupTags[0] || "", templates);
  const items = trade.checklistItems || [];
  const total = items.reduce((sum, item) => sum + Number(item.points || 0), 0);
  const earned = items.reduce((sum, item) => {
    const points = Number(item.points || 0);
    if ((item.inputType || "boolean") === "points") {
      return sum + Math.max(0, Math.min(points, Number(item.score || 0)));
    }

    return sum + (item.met ? points : 0);
  }, 0);
  const manualGrade = trade.manualGrade?.trim();

  if (!template?.gradeBands?.length || !total) {
    return { earned, total, grade: manualGrade || "Unscored" };
  }

  if (manualGrade) {
    return { earned, total, grade: manualGrade };
  }

  const grade = [...template.gradeBands]
    .sort((a, b) => b.minScore - a.minScore)
    .find((band) => earned >= band.minScore && (band.maxScore === null || earned <= band.maxScore));

  return { earned, total, grade: grade?.label || "Unscored" };
}

export function tradeNeedsReview(trade: TradeLogEntry, templates: SetupChecklistTemplate[]) {
  const grade = tradeChecklistScore(trade, templates).grade.trim();
  return (
    !Number.isFinite(trade.risk) || !trade.risk ||
    !grade || grade.toLowerCase() === "unscored" ||
    !trade.setupTags.some((setup) => setup.trim().length > 0) ||
    !hasCompletedTradeReview(trade.reviewSections) ||
    !trade.screenshots.some((screenshot) => screenshot.trim().length > 0)
  );
}
