import { NextResponse } from "next/server";
import { requireMarketReviewDashboardUser, marketReviewErrorResponse } from "@/lib/market-review-api";
import { resolveMarketReviewBaseUrl, retryAndDispatchMarketReview } from "@/lib/market-review-service";

export async function POST(request: Request, context: { params: Promise<{ runId: string }> }) {
  try {
    await requireMarketReviewDashboardUser(true);
    const { runId } = await context.params;
    const run = await retryAndDispatchMarketReview(runId, resolveMarketReviewBaseUrl(request.url));
    return NextResponse.json({ run }, { headers: { "Cache-Control": "no-store" } });
  } catch (error) {
    return marketReviewErrorResponse(error);
  }
}
