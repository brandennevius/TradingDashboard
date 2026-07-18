import { NextResponse } from "next/server";
import { getSessionUser } from "@/lib/auth";
import { listTrades, upsertTrade } from "@/lib/store";
import { normalizeTradeReviewSections } from "@/lib/trade-review";
import type { TradeChecklistItem, TradeExecution, TradeLogInput, TradeSide, TradeStatus } from "@/lib/types";

function numberValue(value: unknown) {
  const number = Number(value);
  return Number.isFinite(number) ? number : 0;
}

function stringArray(value: unknown) {
  if (!Array.isArray(value)) {
    return [];
  }

  return value.map(String).map((item) => item.trim()).filter(Boolean);
}

function checklistItems(value: unknown): TradeChecklistItem[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value
    .map((item, index) => {
      if (!item || typeof item !== "object") {
        return null;
      }

      const rawItem = item as Record<string, unknown>;
      const criteria = String(rawItem.criteria || "").trim();
      const points = numberValue(rawItem.points);

      if (!criteria || points <= 0) {
        return null;
      }

      return {
        id: String(rawItem.id || `criteria-${index}-${Date.now()}`),
        criteria,
        points,
        met: Boolean(rawItem.met)
      };
    })
    .filter(Boolean) as TradeChecklistItem[];
}

function tradeExecutions(value: unknown): TradeExecution[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value
    .map((item, index) => {
      if (!item || typeof item !== "object") {
        return null;
      }

      const rawItem = item as Record<string, unknown>;
      return {
        id: String(rawItem.id || rawItem.sourceKey || `execution-${index}`),
        type: rawItem.type === "EXIT" ? "EXIT" : "ENTRY",
        date: String(rawItem.date || ""),
        time: String(rawItem.time || ""),
        side: tradeSide(rawItem.side),
        shares: numberValue(rawItem.shares),
        price: numberValue(rawItem.price),
        pnl: numberValue(rawItem.pnl),
        commission: numberValue(rawItem.commission),
        source: String(rawItem.source || ""),
        sourceKey: String(rawItem.sourceKey || "")
      };
    })
    .filter(Boolean) as TradeExecution[];
}

function tradeSide(value: unknown): TradeSide {
  return value === "SHORT" ? "SHORT" : "LONG";
}

function tradeStatus(value: unknown): TradeStatus {
  return value === "WIN" || value === "LOSS" || value === "BREAKEVEN" ? value : "OPEN";
}

export async function GET() {
  const user = await getSessionUser();

  if (!user) {
    return NextResponse.json({ error: "Unauthorized." }, { status: 401 });
  }

  try {
    const trades = await listTrades();
    return NextResponse.json({ trades });
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Could not load trades." },
      { status: 500 }
    );
  }
}

export async function POST(request: Request) {
  const user = await getSessionUser();

  if (!user) {
    return NextResponse.json({ error: "Unauthorized." }, { status: 401 });
  }

  if (user.readOnly) {
    return NextResponse.json({ error: "This account is read-only." }, { status: 403 });
  }

  const body = await request.json();
  const symbol = String(body.symbol || "").trim().toUpperCase();
  const entryDate = String(body.entryDate || "");

  if (!symbol) {
    return NextResponse.json({ error: "Symbol is required." }, { status: 400 });
  }

  if (!/^\d{4}-\d{2}-\d{2}$/.test(entryDate)) {
    return NextResponse.json({ error: "Entry date is required." }, { status: 400 });
  }

  const input: TradeLogInput = {
    userId: user.id,
    importSource: String(body.importSource || "").trim(),
    importRowKey: String(body.importRowKey || "").trim(),
    symbol,
    side: tradeSide(body.side),
    status: tradeStatus(body.status),
    entryDate,
    exitDate: String(body.exitDate || ""),
    openTime: String(body.openTime || ""),
    closeTime: String(body.closeTime || ""),
    avgEntry: numberValue(body.avgEntry),
    exitPrice: numberValue(body.exitPrice),
    stopPrice: numberValue(body.stopPrice),
    takeProfitPrice: numberValue(body.takeProfitPrice),
    shares: numberValue(body.shares),
    commission: numberValue(body.commission),
    usedMargin: numberValue(body.usedMargin),
    risk: numberValue(body.risk),
    pnl: numberValue(body.pnl),
    rMultiple: numberValue(body.rMultiple),
    returnPercent: numberValue(body.returnPercent),
    daysInTrade: numberValue(body.daysInTrade),
    setupTags: stringArray(body.setupTags),
    mistakeTags: stringArray(body.mistakeTags),
    customTags: stringArray(body.customTags),
    manualGrade: String(body.manualGrade || "").trim(),
    portfolioTag: String(body.portfolioTag || "").trim(),
    emotion: String(body.emotion || "").trim(),
    tradeQuality: String(body.tradeQuality || "").trim(),
    checklistItems: checklistItems(body.checklistItems),
    notes: String(body.notes || ""),
    reviewSections: normalizeTradeReviewSections(body.reviewSections),
    screenshots: stringArray(body.screenshots),
    chartLinks: stringArray(body.chartLinks),
    executions: tradeExecutions(body.executions),
    groupId: "",
    groupRole: "none"
  };

  try {
    const result = await upsertTrade(input);
    return NextResponse.json(result);
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Could not save trade." },
      { status: 500 }
    );
  }
}
