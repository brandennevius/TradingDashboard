import { NextResponse } from "next/server";
import { getSessionUser } from "@/lib/auth";
import { getMarketCycleEntries, saveMarketCycleEntry } from "@/lib/store";

function numberValue(value: unknown) {
  const number = Number(value);
  return Number.isFinite(number) ? number : 0;
}

export async function GET() {
  const user = await getSessionUser();

  if (!user) {
    return NextResponse.json({ error: "Unauthorized." }, { status: 401 });
  }

  try {
    const entries = await getMarketCycleEntries(user.journalOwnerId || user.id);
    return NextResponse.json({ user, entries });
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Could not load market cycle entries." },
      { status: 500 }
    );
  }
}

export async function PUT(request: Request) {
  const user = await getSessionUser();

  if (!user) {
    return NextResponse.json({ error: "Unauthorized." }, { status: 401 });
  }

  if (user.readOnly) {
    return NextResponse.json({ error: "This account is read-only." }, { status: 403 });
  }

  const body = await request.json();
  const date = String(body.date || "");

  if (!/^\d{4}-\d{2}-\d{2}$/.test(date)) {
    return NextResponse.json({ error: "A valid date is required." }, { status: 400 });
  }

  try {
    const entries = await saveMarketCycleEntry({
      userId: user.id,
      date,
      trendDay: numberValue(body.trendDay),
      phase: String(body.phase || ""),
      notes: String(body.notes || "")
    });
    return NextResponse.json({ entries });
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Could not save market cycle entry." },
      { status: 500 }
    );
  }
}
