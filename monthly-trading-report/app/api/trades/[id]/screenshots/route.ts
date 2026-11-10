import { NextResponse } from "next/server";
import { getSessionUser } from "@/lib/auth";
import { saveTradeScreenshot } from "@/lib/store";

const MAX_SCREENSHOT_BYTES = 3_500_000;

export async function POST(request: Request, context: { params: Promise<{ id: string }> }) {
  const user = await getSessionUser();

  if (!user) {
    return NextResponse.json({ error: "Unauthorized." }, { status: 401 });
  }

  if (user.readOnly) {
    return NextResponse.json({ error: "This account is read-only." }, { status: 403 });
  }

  const { id } = await context.params;
  const formData = await request.formData();
  const file = formData.get("file");

  if (!(file instanceof File) || !file.type.startsWith("image/")) {
    return NextResponse.json({ error: "An image file is required." }, { status: 400 });
  }

  if (file.size > MAX_SCREENSHOT_BYTES) {
    return NextResponse.json({ error: "Screenshot must be smaller than 3.5 MB." }, { status: 413 });
  }

  try {
    const uploadKey = request.headers.get("idempotency-key") || undefined;
    if (uploadKey && !/^[a-f0-9-]{36}$/i.test(uploadKey)) return NextResponse.json({error:"Invalid upload key."},{status:400});
    const saved = await saveTradeScreenshot(id, user.id, file.name, file.type, Buffer.from(await file.arrayBuffer()), uploadKey);
    return NextResponse.json(saved);
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Could not save screenshot." },
      { status: 500 }
    );
  }
}
