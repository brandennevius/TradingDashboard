import { Packer } from "docx";
import { NextResponse } from "next/server";
import { getSessionUser } from "@/lib/auth";
import { getSetupChecklistTemplates, listBrandenVisibleTrades } from "@/lib/store";
import { tradeReviewMissingFields } from "@/lib/trade-review";
import { buildDocument, generateAiReview, safeFilePart, sortedTradesByRequest, tradeForRange } from "@/lib/trade-review-export";
import { collectReviewEvidence } from "@/lib/trade-review-evidence";

export const maxDuration = 300;

export async function POST(request: Request) {
  const user = await getSessionUser();
  if (!user || (user.journalOwnerId || user.id) !== "branden") return NextResponse.json({ error: "Unauthorized." }, { status: 401 });
  const body = await request.json().catch(() => ({}));
  const tradeIds = Array.isArray(body.tradeIds) ? Array.from(new Set<string>(body.tradeIds.map(String).filter(Boolean))) : [];
  const startDate = String(body.startDate || "");
  const endDate = String(body.endDate || "");
  if (!tradeIds.length) return NextResponse.json({ error: "No filtered trades were selected for export." }, { status: 400 });
  if ([startDate, endDate].some((date) => date && !/^\d{4}-\d{2}-\d{2}$/.test(date)) || (startDate && endDate && startDate > endDate)) {
    return NextResponse.json({ error: "Choose a valid date range." }, { status: 400 });
  }
  try {
    const [allTrades, templates] = await Promise.all([listBrandenVisibleTrades(), getSetupChecklistTemplates()]);
    const lifecycleTrades = sortedTradesByRequest(allTrades, tradeIds);
    if (lifecycleTrades.length !== tradeIds.length) return NextResponse.json({ error: "Some trades are no longer available. Refresh the trade log before exporting." }, { status: 409 });
    const incomplete = lifecycleTrades.map((trade) => ({ id: trade.id, symbol: trade.symbol, entryDate: trade.entryDate, missing: tradeReviewMissingFields(trade, templates) })).filter((trade) => trade.missing.length);
    if (incomplete.length) return NextResponse.json({
      error: `Complete the required review fields before exporting:\n${incomplete.map((trade) => `${trade.symbol} (${trade.entryDate}): ${trade.missing.join(", ")}`).join("\n")}`,
      incompleteTrades: incomplete
    }, { status: 422 });
    if (!process.env.OPENAI_API_KEY) return NextResponse.json({ error: "The review service has no OpenAI API key configured." }, { status: 503 });
    // Calculate excursion from the full lifecycle, never from period-adjusted P&L or exit dates.
    const signal = AbortSignal.any([request.signal, AbortSignal.timeout(275_000)]);
    const evidence = await collectReviewEvidence(lifecycleTrades, templates, signal);
    const trades = lifecycleTrades.map((trade) => tradeForRange(trade, startDate, endDate));
    const review = await generateAiReview(trades, templates, evidence, startDate, endDate, signal);
    const document = await buildDocument(trades, templates, startDate, endDate, evidence, review);
    const buffer = await Packer.toBuffer(document);
    const filename = `branden-trade-review-${safeFilePart(startDate || "all")}-to-${safeFilePart(endDate || "today")}.docx`;
    return new NextResponse(new Uint8Array(buffer), { headers: {
      "Content-Type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
      "Content-Disposition": `attachment; filename="${filename}"`, "Cache-Control": "no-store"
    } });
  } catch (error) {
    return NextResponse.json({ error: error instanceof Error ? error.message : "Could not generate the trade review." }, { status: 502 });
  }
}
