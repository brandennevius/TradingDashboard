import { NextResponse } from "next/server";
import { requireMarketReviewDashboardUser, marketReviewErrorResponse } from "@/lib/market-review-api";
import { MarketReviewValidationError, sha256 } from "@/lib/market-review-contract";
import { saveMarketReviewOcrCorrections } from "@/lib/market-review-store";

export async function POST(request: Request, context: { params: Promise<{ runId: string }> }) {
  try {
    await requireMarketReviewDashboardUser(true);
    const { runId } = await context.params;
    const body = await request.json().catch(() => null) as { expected_version?: unknown; corrections?: unknown } | null;
    if (!body || !Number.isInteger(body.expected_version) || !Array.isArray(body.corrections)) {
      throw new MarketReviewValidationError("OCR_CORRECTIONS_INVALID", "expected_version and a corrections array are required.");
    }
    if (body.corrections.length > 500) throw new MarketReviewValidationError("OCR_CORRECTIONS_TOO_LARGE", "At most 500 OCR corrections may be submitted.");
    const data = Buffer.from(`${JSON.stringify({ expected_version: body.expected_version, corrections: body.corrections }, null, 2)}\n`, "utf8");
    if (data.length > 512 * 1024) throw new MarketReviewValidationError("OCR_CORRECTIONS_TOO_LARGE", "OCR corrections may not exceed 512 KB.");
    const run = await saveMarketReviewOcrCorrections(runId, {
      expectedVersion: Number(body.expected_version),
      corrections: body.corrections,
      data,
      sha256: sha256(data)
    });
    return NextResponse.json({ run }, { headers: { "Cache-Control": "no-store" } });
  } catch (error) {
    return marketReviewErrorResponse(error);
  }
}
