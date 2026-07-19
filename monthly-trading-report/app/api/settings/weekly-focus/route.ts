import { NextResponse } from "next/server";
import { getSessionUser } from "@/lib/auth";
import { getWeeklyProcessFocus, saveWeeklyProcessFocus } from "@/lib/store";

export async function GET() {
  const user = await getSessionUser();
  if (!user) return NextResponse.json({ error: "Unauthorized." }, { status: 401 });
  try {
    const ownerId = user.journalOwnerId || user.id;
    return NextResponse.json({ focus: await getWeeklyProcessFocus(ownerId) });
  } catch (error) {
    return NextResponse.json({ error: error instanceof Error ? error.message : "Could not load weekly focus." }, { status: 500 });
  }
}

export async function PUT(request: Request) {
  const user = await getSessionUser();
  if (!user) return NextResponse.json({ error: "Unauthorized." }, { status: 401 });
  if (user.readOnly) return NextResponse.json({ error: "This account is read-only." }, { status: 403 });
  try {
    const body = await request.json().catch(() => ({}));
    const ownerId = user.journalOwnerId || user.id;
    const focus = await saveWeeklyProcessFocus(ownerId, {
      summary: body.summary,
      focusItems: body.focusItems
    });
    return NextResponse.json({ focus });
  } catch (error) {
    return NextResponse.json({ error: error instanceof Error ? error.message : "Could not save weekly focus." }, { status: 500 });
  }
}
