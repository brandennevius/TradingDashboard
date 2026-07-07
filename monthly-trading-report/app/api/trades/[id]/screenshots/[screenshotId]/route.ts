import { NextResponse } from "next/server";
import { getSessionUser } from "@/lib/auth";
import { getTradeScreenshot } from "@/lib/store";

export async function GET(
  _request: Request,
  context: { params: Promise<{ id: string; screenshotId: string }> }
) {
  const user = await getSessionUser();

  if (!user) {
    return NextResponse.json({ error: "Unauthorized." }, { status: 401 });
  }

  const { id, screenshotId } = await context.params;
  const screenshot = await getTradeScreenshot(screenshotId);
  const allowedOwnerId = user.journalOwnerId || user.id;

  if (!screenshot || screenshot.tradeId !== id || screenshot.userId !== allowedOwnerId) {
    return NextResponse.json({ error: "Screenshot not found." }, { status: 404 });
  }

  const imageBody = new Uint8Array(screenshot.imageData);
  const safeFileName =
    screenshot.fileName
      .normalize("NFKD")
      .replace(/[^\x20-\x7E]/g, "_")
      .replace(/["\\\r\n]/g, "_") || "trade-screenshot";

  return new NextResponse(imageBody, {
    headers: {
      "Content-Type": screenshot.mimeType,
      "Content-Disposition": `inline; filename="${safeFileName}"`,
      "Cache-Control": "public, max-age=31536000, s-maxage=31536000, immutable"
    }
  });
}
