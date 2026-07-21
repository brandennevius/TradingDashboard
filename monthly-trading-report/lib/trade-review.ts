import type { TradeReviewSections } from "./types";

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

export const minimumCompletedTradeReviewSections = 3;

export function completedTradeReviewSectionCount(value: unknown) {
  return Object.values(normalizeTradeReviewSections(value))
    .filter((section) => section.trim().length > 0)
    .length;
}

export function hasCompletedTradeReview(value: unknown, legacyNotes = "") {
  const completedSections = completedTradeReviewSectionCount(value);
  if (completedSections >= minimumCompletedTradeReviewSections) return true;

  // Preserve completion for older trades that only have the legacy free-form notes field.
  return completedSections === 0 && String(legacyNotes || "").trim().length > 0;
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
