import { NextResponse } from "next/server";
import { requireMarketReviewDashboardUser, marketReviewErrorResponse } from "@/lib/market-review-api";
import { createAndDispatchMarketReview, resolveMarketReviewBaseUrl } from "@/lib/market-review-service";
import { listMarketReviewRuns } from "@/lib/market-review-store";
import { requireMarketReviewCreateJson, validateMarketReviewBlobReference } from "@/lib/market-review-upload";

export const runtime = "nodejs";
export const maxDuration = 60;

export async function GET() {
  try {
    await requireMarketReviewDashboardUser(false);
    return NextResponse.json({ runs: await listMarketReviewRuns() }, { headers: { "Cache-Control": "no-store" } });
  } catch (error) {
    return marketReviewErrorResponse(error);
  }
}

export async function POST(request: Request) {
  try {
    await requireMarketReviewDashboardUser(true);
    requireMarketReviewCreateJson(request.headers.get("content-type"));
    const body = await request.json().catch(() => null) as { session_date?: unknown; marketsurge_pdf?: unknown } | null;
    const session = String(body?.session_date || "").trim();
    const pdfBlob = validateMarketReviewBlobReference(body?.marketsurge_pdf, session);
    const run = await createAndDispatchMarketReview({
      session,
      pdfBlob,
      baseUrl: resolveMarketReviewBaseUrl(request.url)
    });
    return NextResponse.json({ run }, { status: 201, headers: { "Cache-Control": "no-store" } });
  } catch (error) {
    return marketReviewErrorResponse(error);
  }
}
