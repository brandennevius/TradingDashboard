import { NextResponse } from "next/server";
import { requireMarketReviewDashboardUser, marketReviewErrorResponse } from "@/lib/market-review-api";
import { createAndDispatchMarketReview, resolveMarketReviewBaseUrl } from "@/lib/market-review-service";
import { listMarketReviewRuns } from "@/lib/market-review-store";

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
    const formData = await request.formData();
    const session = String(formData.get("session_date") || "").trim();
    const file = formData.get("marketsurge_pdf");
    if (!(file instanceof File)) {
      return NextResponse.json({ error: "A MarketSurge screenshot PDF is required.", code: "MARKETSURGE_PDF_REQUIRED" }, { status: 400 });
    }
    const run = await createAndDispatchMarketReview({
      session,
      pdfFilename: file.name,
      pdfMimeType: file.type,
      pdfData: Buffer.from(await file.arrayBuffer()),
      baseUrl: resolveMarketReviewBaseUrl(request.url)
    });
    return NextResponse.json({ run }, { status: 201, headers: { "Cache-Control": "no-store" } });
  } catch (error) {
    return marketReviewErrorResponse(error);
  }
}
