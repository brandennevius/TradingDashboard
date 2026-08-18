import { NextResponse } from "next/server";
import { bearerToken, marketReviewErrorResponse } from "@/lib/market-review-api";
import { MarketReviewValidationError } from "@/lib/market-review-contract";
import { getMarketReviewWorkerInput, resolveMarketReviewBaseUrl } from "@/lib/market-review-service";

export const runtime = "nodejs";

export async function GET(request: Request, context: { params: Promise<{ runId: string }> }) {
  try {
    const token = bearerToken(request);
    if (!token) throw new MarketReviewValidationError("WORKER_AUTH_REQUIRED", "DASHBOARD_WORKER_SECRET must be presented as a bearer token.");
    const { runId } = await context.params;
    const attempt = Number(request.headers.get("x-review-attempt"));
    const input = await getMarketReviewWorkerInput(runId, token, resolveMarketReviewBaseUrl(request.url), {
      attempt,
      source_hashes: {
        marketsurge_pdf_sha256: request.headers.get("x-marketsurge-pdf-sha256") || "",
        snapshot_json_sha256: request.headers.get("x-snapshot-json-sha256") || "",
        snapshot_markdown_sha256: request.headers.get("x-snapshot-markdown-sha256") || "",
        market_gauge_json_sha256: request.headers.get("x-market-gauge-json-sha256") || ""
      }
    });
    return NextResponse.json(input, { headers: { "Cache-Control": "private, no-store" } });
  } catch (error) {
    return marketReviewErrorResponse(error);
  }
}
