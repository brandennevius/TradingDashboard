import { NextResponse } from "next/server";
import { getSessionUser } from "@/lib/auth";
import { getCamJournalScreenshot } from "@/lib/store";

export async function GET(
  _request: Request,
  context: { params: Promise<{ screenshotId: string }> }
) {
  const user = await getSessionUser();
  if (!user) return NextResponse.json({ error: "Unauthorized." }, { status: 401 });

  const { screenshotId } = await context.params;
  const screenshot = await getCamJournalScreenshot(screenshotId);
  if (!screenshot || screenshot.entityType !== "watchlist-item") {
    return NextResponse.json({ error: "Watchlist screenshot not found." }, { status: 404 });
  }

  return new NextResponse(new Uint8Array(screenshot.imageData), {
    headers: {
      "Content-Type": screenshot.mimeType,
      "Content-Disposition": `inline; filename="${screenshot.fileName.replace(/["\\\r\n]/g, "_") || "watchlist-screenshot"}"`,
      "Cache-Control": "private, max-age=86400",
      "X-Content-Type-Options": "nosniff"
    }
  });
}
