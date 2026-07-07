import { NextResponse } from "next/server";
import { getSessionUser } from "@/lib/auth";
import { listReports, upsertReport } from "@/lib/store";
import type { MonthlyReportInput } from "@/lib/types";

function numberValue(value: unknown) {
  const number = Number(value);
  return Number.isFinite(number) ? number : 0;
}

export async function GET() {
  const user = await getSessionUser();

  if (!user) {
    return NextResponse.json({ error: "Unauthorized." }, { status: 401 });
  }

  try {
    const reports = await listReports();
    return NextResponse.json({ reports });
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Could not load reports." },
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
  const month = String(body.month || "");

  if (!/^\d{4}-\d{2}$/.test(month)) {
    return NextResponse.json({ error: "Month must use YYYY-MM format." }, { status: 400 });
  }

  const input: MonthlyReportInput = {
    userId: user.id,
    month,
    accountSize: numberValue(body.accountSize),
    totalReturn: numberValue(body.totalReturn),
    percentReturn: numberValue(body.percentReturn),
    netPnl: numberValue(body.netPnl),
    totalPayouts: numberValue(body.totalPayouts),
    totalTrades: numberValue(body.totalTrades),
    winRate: numberValue(body.winRate),
    avgR: numberValue(body.avgR),
    totalR: numberValue(body.totalR),
    avgWinR: numberValue(body.avgWinR),
    avgLossR: numberValue(body.avgLossR),
    avgWin: numberValue(body.avgWin),
    avgLoss: numberValue(body.avgLoss),
    avgRisk: numberValue(body.avgRisk),
    currentRiskPercent: numberValue(body.currentRiskPercent),
    expectedValueR: numberValue(body.expectedValueR),
    sharpeRatio: numberValue(body.sharpeRatio),
    avgTradeLength: numberValue(body.avgTradeLength),
    avgSwingLength: numberValue(body.avgSwingLength),
    longestWinStreak: numberValue(body.longestWinStreak),
    longestLossStreak: numberValue(body.longestLossStreak),
    notes: String(body.notes || "")
  };

  try {
    const report = await upsertReport(input);
    return NextResponse.json({ report });
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Could not save report." },
      { status: 500 }
    );
  }
}
