import { del } from "@vercel/blob";
import { handleUpload, type HandleUploadBody } from "@vercel/blob/client";
import { NextResponse } from "next/server";
import { requireMarketReviewDashboardUser, marketReviewErrorResponse } from "@/lib/market-review-api";
import { MARKET_REVIEW_MAX_SOURCE_PDF_BYTES, MarketReviewValidationError } from "@/lib/market-review-contract";
import {
  MARKET_REVIEW_UPLOAD_TOKEN_TTL_MS,
  marketReviewBlobPathname,
  parseMarketReviewUploadClientPayload,
  validateMarketReviewBlobReference
} from "@/lib/market-review-upload";

export const runtime = "nodejs";

export async function POST(request: Request) {
  try {
    const body = await request.json() as HandleUploadBody;
    const response = await handleUpload({
      request,
      body,
      onBeforeGenerateToken: async (pathname, clientPayload) => {
        await requireMarketReviewDashboardUser(true);
        const descriptor = parseMarketReviewUploadClientPayload(clientPayload);
        if (pathname !== marketReviewBlobPathname(descriptor)) {
          throw new MarketReviewValidationError("MARKETSURGE_BLOB_PATH_MISMATCH", "The requested private Blob path does not match the authenticated upload metadata.");
        }
        return {
          allowedContentTypes: ["application/pdf"],
          maximumSizeInBytes: MARKET_REVIEW_MAX_SOURCE_PDF_BYTES,
          validUntil: Date.now() + MARKET_REVIEW_UPLOAD_TOKEN_TTL_MS,
          addRandomSuffix: false,
          allowOverwrite: false,
          cacheControlMaxAge: 60,
          tokenPayload: JSON.stringify(descriptor)
        };
      },
      onUploadCompleted: async ({ blob, tokenPayload }) => {
        const descriptor = parseMarketReviewUploadClientPayload(tokenPayload || null);
        if (blob.pathname !== marketReviewBlobPathname(descriptor) || blob.contentType?.toLowerCase() !== "application/pdf") {
          throw new MarketReviewValidationError("MARKETSURGE_BLOB_PATH_MISMATCH", "Completed private Blob upload does not match its authenticated token metadata.");
        }
      }
    });
    return NextResponse.json(response, { headers: { "Cache-Control": "no-store" } });
  } catch (error) {
    return marketReviewErrorResponse(error);
  }
}

export async function DELETE(request: Request) {
  try {
    await requireMarketReviewDashboardUser(true);
    const body = await request.json().catch(() => null) as { marketsurge_pdf?: unknown } | null;
    const reference = validateMarketReviewBlobReference(body?.marketsurge_pdf);
    await del(reference.blob_url);
    return new NextResponse(null, { status: 204, headers: { "Cache-Control": "no-store" } });
  } catch (error) {
    return marketReviewErrorResponse(error);
  }
}
