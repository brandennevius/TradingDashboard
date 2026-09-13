import { Packer } from "docx";
import { NextResponse } from "next/server";
import { getSessionUser } from "@/lib/auth";
import { getSetupChecklistTemplates, listBrandenVisibleTrades } from "@/lib/store";
import { tradeReviewMissingFields } from "@/lib/trade-review";
import { buildDocument, completedAiReview, pendingAiReviewId, retrieveAiReview, safeFilePart, sortedTradesByRequest, startAiReview, tradeForRange } from "@/lib/trade-review-export";
import { collectReviewEvidence } from "@/lib/trade-review-evidence";

export const maxDuration = 300;

function logReviewExport(event: string, detail: Record<string, unknown>) {
  console.log(JSON.stringify({ scope: "ai-review-export", event, ...detail }));
}

function exportErrorMessage(error: unknown) {
  if (error instanceof Error && (error.name === "TimeoutError" || error.name === "AbortError")) {
    return "This export stage took too long. Retry the export; the AI review will continue in the background between status checks.";
  }
  return error instanceof Error ? error.message : "Could not generate the trade review.";
}

export async function POST(request: Request) {
  const startedAt = Date.now();
  const user = await getSessionUser();
  if (!user || (user.journalOwnerId || user.id) !== "branden") return NextResponse.json({ error: "Unauthorized." }, { status: 401 });
  const body = await request.json().catch(() => ({}));
  const tradeIds = Array.isArray(body.tradeIds) ? Array.from(new Set<string>(body.tradeIds.map(String).filter(Boolean))) : [];
  const startDate = String(body.startDate || "");
  const endDate = String(body.endDate || "");
  const reviewJobId = String(body.reviewJobId || "");
  if (!tradeIds.length) return NextResponse.json({ error: "No filtered trades were selected for export." }, { status: 400 });
  if ([startDate, endDate].some((date) => date && !/^\d{4}-\d{2}-\d{2}$/.test(date)) || (startDate && endDate && startDate > endDate)) {
    return NextResponse.json({ error: "Choose a valid date range." }, { status: 400 });
  }
  try {
    logReviewExport("start", { phase: reviewJobId ? "poll" : "submit", tradeCount: tradeIds.length });
    const signal = AbortSignal.any([request.signal, AbortSignal.timeout(240_000)]);
    const aiResponse = reviewJobId ? await retrieveAiReview(reviewJobId, signal) : null;
    const pendingId = aiResponse ? pendingAiReviewId(aiResponse) : "";
    if (pendingId) {
      logReviewExport("pending", { phase: "poll", ms: Date.now() - startedAt });
      return NextResponse.json({ reviewJobId: pendingId, status: aiResponse?.status }, { status: 202 });
    }

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
    const evidence = await collectReviewEvidence(lifecycleTrades, templates, signal);
    const imageOccurrences = Object.values(evidence.images).reduce((total, images) => total + images.length, 0);
    const uniqueImages = new Set(Object.values(evidence.images).flat().map((image) => image.dataUrl)).size;
    logReviewExport("evidence-ready", { phase: reviewJobId ? "finalize" : "submit", ms: Date.now() - startedAt, imageOccurrences, uniqueImages });
    const trades = lifecycleTrades.map((trade) => tradeForRange(trade, startDate, endDate));
    let response = aiResponse;
    let priceCheck: { inputTokens: number; maximumCostUsd: number; model: string } | null = null;
    if (!response) {
      const started = await startAiReview(trades, templates, evidence, startDate, endDate, signal);
      response = started.response;
      priceCheck = { inputTokens: started.inputTokens, maximumCostUsd: started.maximumCostUsd, model: started.model };
      logReviewExport("price-checked", { inputTokens: started.inputTokens, maximumCostUsd: Number(started.maximumCostUsd.toFixed(4)) });
    }
    const responsePendingId = pendingAiReviewId(response);
    if (responsePendingId) {
      logReviewExport("submitted", { ms: Date.now() - startedAt });
      return NextResponse.json({ reviewJobId: responsePendingId, status: response.status, ...(priceCheck || {}) }, { status: 202 });
    }
    const review = completedAiReview(response, trades, templates, evidence);
    const document = await buildDocument(trades, templates, startDate, endDate, evidence, review);
    const buffer = await Packer.toBuffer(document);
    const filename = `branden-trade-review-${safeFilePart(startDate || "all")}-to-${safeFilePart(endDate || "today")}.docx`;
    logReviewExport("complete", { phase: reviewJobId ? "finalize" : "submit", ms: Date.now() - startedAt, bytes: buffer.length });
    return new NextResponse(new Uint8Array(buffer), { headers: {
      "Content-Type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
      "Content-Disposition": `attachment; filename="${filename}"`, "Cache-Control": "no-store"
    } });
  } catch (error) {
    const message = exportErrorMessage(error);
    logReviewExport("failed", { phase: reviewJobId ? "poll" : "submit", ms: Date.now() - startedAt, error: message });
    return NextResponse.json({ error: message }, { status: 502 });
  }
}
