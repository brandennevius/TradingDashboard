import { NextResponse } from "next/server";
import { getSessionUser } from "@/lib/auth";
import { deleteCamJournalScreenshot, getCamJournalScreenshot } from "@/lib/store";

export async function GET(
  _request: Request,
  context: { params: Promise<{ screenshotId: string }> }
) {
  const user = await getSessionUser();
  if (!user) return NextResponse.json({ error: "Unauthorized." }, { status: 401 });

  const { screenshotId } = await context.params;
  const screenshot = await getCamJournalScreenshot(screenshotId);
  if (!screenshot) return NextResponse.json({ error: "Screenshot not found." }, { status: 404 });

  return new NextResponse(new Uint8Array(screenshot.imageData), {
    headers: {
      "Content-Type": screenshot.mimeType,
      "Content-Disposition": `inline; filename="${screenshot.fileName.replace(/["\\\r\n]/g, "_") || "screenshot"}"`,
      "Cache-Control": "public, max-age=31536000, s-maxage=31536000, immutable"
    }
  });
}

export async function DELETE(
  _request: Request,
  context: { params: Promise<{ screenshotId: string }> }
) {
  const user = await getSessionUser();
  if (!user) return NextResponse.json({ error: "Unauthorized." }, { status: 401 });
  if (user.id !== "cam") return NextResponse.json({ error: "Cam journal is read-only for this user." }, { status: 403 });

  const { screenshotId } = await context.params;
  await deleteCamJournalScreenshot(screenshotId);
  return NextResponse.json({ ok: true });
}
