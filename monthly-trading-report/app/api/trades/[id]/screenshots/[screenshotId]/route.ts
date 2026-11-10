import { privateScreenshotRedirect, shouldRedirectScreenshot } from "@/lib/screenshot-storage";
import { getTradeScreenshotMetadata } from "@/lib/store";
import { NextResponse } from "next/server";
import { getSessionUser } from "@/lib/auth";
import { getTradeScreenshot } from "@/lib/store";

export async function GET(
  request: Request,
  context: { params: Promise<{ id: string; screenshotId: string }> }
) {
  const user = await getSessionUser();

  if (!user) {
    return NextResponse.json({ error: "Unauthorized." }, { status: 401 });
  }

  const { id, screenshotId } = await context.params;
  if (shouldRedirectScreenshot(request)) {
    const metadata = await getTradeScreenshotMetadata(screenshotId);
    if (!metadata || metadata.trade_id !== id || metadata.user_id !== (user.journalOwnerId || user.id)) return NextResponse.json({error:"Screenshot not found."},{status:404});
    const redirect = await privateScreenshotRedirect(metadata);
    if (redirect) return redirect;
  }
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
      "Cache-Control": "private, no-store"
    }
  });
}
