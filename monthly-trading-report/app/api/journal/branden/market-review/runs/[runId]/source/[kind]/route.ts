import { marketReviewErrorResponse } from "@/lib/market-review-api";
import { MarketReviewValidationError, type MarketReviewSourceKind } from "@/lib/market-review-contract";
import { authorizeMarketReviewSource } from "@/lib/market-review-service";

export const runtime = "nodejs";

const sourceKinds = new Set<MarketReviewSourceKind>(["marketsurge_pdf", "snapshot_json", "snapshot_markdown", "ocr_corrections_json"]);

export async function GET(request: Request, context: { params: Promise<{ runId: string; kind: string }> }) {
  try {
    const { runId, kind: rawKind } = await context.params;
    const kind = rawKind as MarketReviewSourceKind;
    if (!sourceKinds.has(kind)) throw new MarketReviewValidationError("SOURCE_KIND_INVALID", "The requested review source kind is invalid.");
    const token = new URL(request.url).searchParams.get("token") || "";
    if (!token) throw new MarketReviewValidationError("TOKEN_REQUIRED", "A signed source token is required.");
    const source = await authorizeMarketReviewSource(runId, kind, token);
    return new Response(new Uint8Array(source.data), {
      headers: {
        "Content-Type": source.mediaType,
        "Content-Length": String(source.sizeBytes),
        "Content-Disposition": `attachment; filename="${source.filename.replace(/["\\\r\n]/g, "_")}"`,
        "Cache-Control": "private, no-store",
        "X-Content-SHA256": source.sha256,
        "X-Content-Type-Options": "nosniff"
      }
    });
  } catch (error) {
    return marketReviewErrorResponse(error);
  }
}
