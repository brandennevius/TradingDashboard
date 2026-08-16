import { requireMarketReviewDashboardUser, marketReviewErrorResponse } from "@/lib/market-review-api";
import { MarketReviewValidationError, type MarketReviewArtifactKind } from "@/lib/market-review-contract";
import { getMarketReviewArtifact } from "@/lib/market-review-store";

const artifactKinds = new Set<MarketReviewArtifactKind>(["pdf", "markdown", "json"]);

export async function GET(_request: Request, context: { params: Promise<{ runId: string; kind: string }> }) {
  try {
    await requireMarketReviewDashboardUser(false);
    const { runId, kind: rawKind } = await context.params;
    const kind = rawKind as MarketReviewArtifactKind;
    if (!artifactKinds.has(kind)) throw new MarketReviewValidationError("RESULT_ARTIFACT_KIND_INVALID", "The requested result artifact kind is invalid.");
    const artifact = await getMarketReviewArtifact(runId, kind);
    if (!artifact) throw new MarketReviewValidationError("RESULT_ARTIFACT_NOT_FOUND", "The requested result artifact does not exist.");
    return new Response(new Uint8Array(artifact.data), {
      headers: {
        "Content-Type": artifact.mediaType,
        "Content-Length": String(artifact.sizeBytes),
        "Content-Disposition": `attachment; filename="${artifact.filename.replace(/["\\\r\n]/g, "_")}"`,
        "Cache-Control": "private, no-store",
        "X-Content-SHA256": artifact.sha256,
        "X-Content-Type-Options": "nosniff"
      }
    });
  } catch (error) {
    return marketReviewErrorResponse(error);
  }
}
