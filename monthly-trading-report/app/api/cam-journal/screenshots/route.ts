import { NextResponse } from "next/server";
import { getSessionUser } from "@/lib/auth";
import { saveCamJournalScreenshot } from "@/lib/store";

const MAX_SCREENSHOT_BYTES = 3_500_000;

export async function POST(request: Request) {
  const user = await getSessionUser();
  if (!user) return NextResponse.json({ error: "Unauthorized." }, { status: 401 });
  if (user.id !== "cam") return NextResponse.json({ error: "Cam journal is read-only for this user." }, { status: 403 });

  const formData = await request.formData();
  const file = formData.get("file");
  const entityType = String(formData.get("entityType") || "trade");
  const entityId = String(formData.get("entityId") || "");

  if (!(file instanceof File) || !file.type.startsWith("image/")) {
    return NextResponse.json({ error: "An image file is required." }, { status: 400 });
  }
  if (!entityId) return NextResponse.json({ error: "Screenshot owner is required." }, { status: 400 });
  if (file.size > MAX_SCREENSHOT_BYTES) {
    return NextResponse.json({ error: "Screenshot must be smaller than 3.5 MB." }, { status: 413 });
  }

  try {
    return NextResponse.json(
      await saveCamJournalScreenshot(entityType, entityId, file.name, file.type, Buffer.from(await file.arrayBuffer()))
    );
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Could not save screenshot." },
      { status: 500 }
    );
  }
}
