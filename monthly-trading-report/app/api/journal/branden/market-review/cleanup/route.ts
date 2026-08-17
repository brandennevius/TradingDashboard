import { NextResponse } from "next/server";
import { requireMarketReviewDashboardUser, marketReviewErrorResponse } from "@/lib/market-review-api";
import { cleanupExpiredMarketReviewSources } from "@/lib/market-review-service";

export async function POST() {
  try {
    await requireMarketReviewDashboardUser(true);
    return NextResponse.json(await cleanupExpiredMarketReviewSources(), { headers: { "Cache-Control": "no-store" } });
  } catch (error) {
    return marketReviewErrorResponse(error);
  }
}
