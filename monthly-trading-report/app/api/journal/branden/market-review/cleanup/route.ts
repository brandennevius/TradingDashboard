import { NextResponse } from "next/server";
import { requireMarketReviewDashboardUser, marketReviewErrorResponse } from "@/lib/market-review-api";
import { cleanupMarketReviewSources } from "@/lib/market-review-store";

export async function POST() {
  try {
    await requireMarketReviewDashboardUser(true);
    return NextResponse.json(await cleanupMarketReviewSources(), { headers: { "Cache-Control": "no-store" } });
  } catch (error) {
    return marketReviewErrorResponse(error);
  }
}
