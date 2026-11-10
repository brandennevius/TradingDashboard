import { privateScreenshotRedirect, shouldRedirectScreenshot, screenshotContentDisposition } from "@/lib/screenshot-storage";
import { getCamJournalScreenshotMetadata } from "@/lib/store";
import { NextResponse } from "next/server";
import { getSessionUser } from "@/lib/auth";
import { deleteCamJournalScreenshot, getCamJournalScreenshot } from "@/lib/store";

export async function GET(
  request: Request,
  context: { params: Promise<{ screenshotId: string }> }
) {
  const user = await getSessionUser();
  if (!user) return NextResponse.json({ error: "Unauthorized." }, { status: 401 });

  const { screenshotId } = await context.params;
  if (shouldRedirectScreenshot(request)) {
    const metadata = await getCamJournalScreenshotMetadata(screenshotId);
    if (!metadata) return NextResponse.json({error:"Screenshot not found."},{status:404});
    const redirect = await privateScreenshotRedirect(metadata);
    if (redirect) return redirect;
  }
  const screenshot = await getCamJournalScreenshot(screenshotId);
  if (!screenshot) return NextResponse.json({ error: "Screenshot not found." }, { status: 404 });

  return new NextResponse(new Uint8Array(screenshot.imageData), {
    headers: {
      "Content-Type": screenshot.mimeType,
      "Content-Disposition": screenshotContentDisposition(screenshot.fileName),
      "Cache-Control": "private, no-store"
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
