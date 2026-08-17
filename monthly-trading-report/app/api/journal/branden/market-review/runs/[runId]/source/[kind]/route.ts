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
    let body: BodyInit;
    if (source.data) {
      body = new Uint8Array(source.data);
    } else if (source.storageUrl && source.storagePathname) {
      const blob = await get(source.storageUrl, { access: "private", useCache: false });
      if (!blob || blob.statusCode !== 200 || !blob.stream) {
        throw new MarketReviewValidationError("SOURCE_NOT_FOUND", "The requested exact source does not exist or has been deleted.");
      }
      if (
        blob.blob.url !== source.storageUrl ||
        blob.blob.pathname !== source.storagePathname ||
        blob.blob.size !== source.sizeBytes ||
        blob.blob.contentType?.toLowerCase() !== source.mediaType.toLowerCase()
      ) {
        throw new MarketReviewValidationError("SOURCE_CORRELATION_MISMATCH", "Private Blob metadata does not match the frozen review source.");
      }
      body = blob.stream;
    } else {
      throw new MarketReviewValidationError("SOURCE_NOT_FOUND", "The requested exact source has no stored content.");
    }
    return new Response(body, {
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
import { get } from "@vercel/blob";
