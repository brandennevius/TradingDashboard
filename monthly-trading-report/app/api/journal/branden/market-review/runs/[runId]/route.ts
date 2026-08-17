import { NextResponse } from "next/server";
import { requireMarketReviewDashboardUser, marketReviewErrorResponse } from "@/lib/market-review-api";
import { MarketReviewValidationError } from "@/lib/market-review-contract";
import { getMarketReviewRun } from "@/lib/market-review-store";

export async function GET(_request: Request, context: { params: Promise<{ runId: string }> }) {
  try {
    await requireMarketReviewDashboardUser(false);
    const { runId } = await context.params;
    const run = await getMarketReviewRun(runId);
    if (!run) throw new MarketReviewValidationError("REVIEW_RUN_NOT_FOUND", "The requested market review run does not exist.");
    return NextResponse.json({ run }, { headers: { "Cache-Control": "no-store" } });
  } catch (error) {
    return marketReviewErrorResponse(error);
  }
}
