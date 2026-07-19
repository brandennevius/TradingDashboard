export type WeeklyFocusStatus = "AVAILABLE" | "NOT_SET" | "CLEARED";

export type WeeklyFocus = {
  status: WeeklyFocusStatus;
  week_start: string | null;
  updated_at: string | null;
  source: "USER_DEFINED_WEEKLY_REVIEW" | null;
  summary: string | null;
  focus_items: string[];
};

export const weeklyFocusNotSet: WeeklyFocus = {
  status: "NOT_SET",
  week_start: null,
  updated_at: null,
  source: null,
  summary: null,
  focus_items: []
};

function cleanItems(value: unknown) {
  return Array.isArray(value) ? value.map(String).map((item) => item.trim()).filter(Boolean) : [];
}

export function normalizeWeeklyFocus(value: unknown): WeeklyFocus {
  if (!value || typeof value !== "object") return { ...weeklyFocusNotSet };
  const raw = value as Record<string, unknown>;
  const summary = String(raw.summary || "").trim();
  const focusItems = cleanItems(raw.focus_items);
  const status = raw.status === "CLEARED" ? "CLEARED" : summary || focusItems.length ? "AVAILABLE" : "NOT_SET";
  if (status === "NOT_SET") return { ...weeklyFocusNotSet };
  return {
    status,
    week_start: /^\d{4}-\d{2}-\d{2}$/.test(String(raw.week_start || "")) ? String(raw.week_start) : null,
    updated_at: String(raw.updated_at || "") || null,
    source: "USER_DEFINED_WEEKLY_REVIEW",
    summary: status === "AVAILABLE" ? summary || null : null,
    focus_items: status === "AVAILABLE" ? focusItems : []
  };
}

function newYorkDateParts(date: Date) {
  const parts = new Intl.DateTimeFormat("en-US", {
    timeZone: "America/New_York", year: "numeric", month: "2-digit", day: "2-digit", weekday: "short"
  }).formatToParts(date);
  const values = Object.fromEntries(parts.map((part) => [part.type, part.value]));
  return { date: `${values.year}-${values.month}-${values.day}`, weekday: values.weekday };
}

export function weeklyFocusWeekStart(date: Date) {
  const local = newYorkDateParts(date);
  const weekday = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"].indexOf(local.weekday);
  const delta = weekday === 0 ? 1 : weekday === 6 ? 2 : 1 - weekday;
  const calendar = new Date(`${local.date}T12:00:00Z`);
  calendar.setUTCDate(calendar.getUTCDate() + delta);
  return calendar.toISOString().slice(0, 10);
}

export function weeklyFocusUpdatedAt(date: Date) {
  const parts = new Intl.DateTimeFormat("en-US", {
    timeZone: "America/New_York", year: "numeric", month: "2-digit", day: "2-digit",
    hour: "2-digit", minute: "2-digit", second: "2-digit", hourCycle: "h23", timeZoneName: "longOffset"
  }).formatToParts(date);
  const values = Object.fromEntries(parts.map((part) => [part.type, part.value]));
  return `${values.year}-${values.month}-${values.day}T${values.hour}:${values.minute}:${values.second}${String(values.timeZoneName || "GMT-05:00").replace("GMT", "")}`;
}

export function createUserDefinedWeeklyFocus(input: { summary?: unknown; focusItems?: unknown }, now = new Date()): WeeklyFocus {
  const summary = String(input.summary || "").trim();
  const focusItems = cleanItems(input.focusItems);
  const common = {
    week_start: weeklyFocusWeekStart(now),
    updated_at: weeklyFocusUpdatedAt(now),
    source: "USER_DEFINED_WEEKLY_REVIEW" as const
  };
  if (!summary && !focusItems.length) {
    return { status: "CLEARED", ...common, summary: null, focus_items: [] };
  }
  return { status: "AVAILABLE", ...common, summary: summary || null, focus_items: focusItems };
}

export function weeklyFocusFromSnapshot(snapshot: unknown) {
  const raw = snapshot && typeof snapshot === "object" ? (snapshot as Record<string, unknown>).weekly_focus : undefined;
  return normalizeWeeklyFocus(raw);
}
