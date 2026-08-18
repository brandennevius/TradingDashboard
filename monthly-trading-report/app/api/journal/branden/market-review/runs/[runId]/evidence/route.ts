import { get } from "@vercel/blob";
import { requireMarketReviewDashboardUser, marketReviewErrorResponse } from "@/lib/market-review-api";
import { MarketReviewValidationError } from "@/lib/market-review-contract";
import { buildMarketReviewDownloadResponse, marketReviewSourceMediaType } from "@/lib/market-review-download";
import { getMarketReviewRun, getMarketReviewSource } from "@/lib/market-review-store";

export const runtime = "nodejs";

export async function GET(_request: Request, context: { params: Promise<{ runId: string }> }) {
  try {
    await requireMarketReviewDashboardUser(false);
    const { runId } = await context.params;
    const run = await getMarketReviewRun(runId);
    if (!run) throw new MarketReviewValidationError("RUN_NOT_FOUND", "The requested market-review run does not exist.");
    if (run.source_deleted_at) throw new MarketReviewValidationError("SOURCE_NOT_FOUND", "The frozen MarketSurge evidence has been deleted under the source-retention policy.");

    const source = await getMarketReviewSource(runId, "marketsurge_pdf");
    if (!source || source.sha256 !== run.source_hashes.marketsurge_pdf_sha256) {
      throw new MarketReviewValidationError("SOURCE_CORRELATION_MISMATCH", "Frozen MarketSurge evidence does not match the selected review run.");
    }

    let body: BodyInit;
    if (source.data) {
      body = new Uint8Array(source.data);
    } else if (source.storageUrl && source.storagePathname) {
      const blob = await get(source.storageUrl, { access: "private", useCache: false });
      if (!blob || blob.statusCode !== 200 || !blob.stream) {
        throw new MarketReviewValidationError("SOURCE_NOT_FOUND", "The frozen MarketSurge evidence is unavailable.");
      }
      if (
        blob.blob.url !== source.storageUrl
        || blob.blob.pathname !== source.storagePathname
        || blob.blob.size !== source.sizeBytes
        || blob.blob.contentType?.toLowerCase() !== source.mediaType.toLowerCase()
      ) {
        throw new MarketReviewValidationError("SOURCE_CORRELATION_MISMATCH", "Private Blob metadata does not match the frozen MarketSurge evidence.");
      }
      body = blob.stream;
    } else {
      throw new MarketReviewValidationError("SOURCE_NOT_FOUND", "The frozen MarketSurge evidence has no stored content.");
    }

    return buildMarketReviewDownloadResponse(body, {
      filename: source.filename,
      contentType: marketReviewSourceMediaType("marketsurge_pdf", source.mediaType),
      sizeBytes: source.sizeBytes,
      sha256: source.sha256,
      disposition: "inline"
    });
  } catch (error) {
    return marketReviewErrorResponse(error);
  }
}
