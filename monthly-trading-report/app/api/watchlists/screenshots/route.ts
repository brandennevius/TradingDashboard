import { NextResponse } from "next/server";
import { getSessionUser } from "@/lib/auth";
import { saveCamJournalScreenshot } from "@/lib/store";

const MAX_SCREENSHOT_BYTES = 3_500_000;

export async function POST(request: Request) {
  const user = await getSessionUser();
  if (!user) return NextResponse.json({ error: "Unauthorized." }, { status: 401 });
  if (user.readOnly) return NextResponse.json({ error: "This account is read-only." }, { status: 403 });

  const formData = await request.formData();
  const file = formData.get("file");
  const itemId = String(formData.get("itemId") || "");

  if (!(file instanceof File) || !file.type.startsWith("image/")) {
    return NextResponse.json({ error: "An image file is required." }, { status: 400 });
  }
  if (!itemId) return NextResponse.json({ error: "A watchlist item is required." }, { status: 400 });
  if (file.size > MAX_SCREENSHOT_BYTES) {
    return NextResponse.json({ error: "Screenshot must be smaller than 3.5 MB." }, { status: 413 });
  }

  try {
    const saved = await saveCamJournalScreenshot("watchlist-item", itemId, file.name, file.type, Buffer.from(await file.arrayBuffer()));
    return NextResponse.json({
      ...saved,
      url: `/api/watchlists/screenshots/${encodeURIComponent(saved.id)}`
    });
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Could not save screenshot." },
      { status: 500 }
    );
  }
}
