import { NextResponse } from "next/server";
import { getSessionUser } from "@/lib/auth";
import { changeTradeMerge } from "@/lib/store";

export async function POST(request: Request) {
  const user = await getSessionUser();
  if (!user) return NextResponse.json({ error: "Unauthorized." }, { status: 401 });
  if (user.readOnly) return NextResponse.json({ error: "This account is read-only." }, { status: 403 });
  const body = await request.json().catch(() => null);
  if (!body || !Array.isArray(body.tradeIds) || body.tradeIds.length < 2 || body.tradeIds.length > 100 || body.tradeIds.some((id: unknown) => typeof id !== "string" || !id)) {
    return NextResponse.json({ error: "Select between 2 and 100 trades." }, { status: 400 });
  }
  try {
    return NextResponse.json({ trades: await changeTradeMerge(user.id, body.tradeIds) });
  } catch (error) {
    return NextResponse.json({ error: error instanceof Error ? error.message : "Could not merge trades." }, { status: 400 });
  }
}

export async function DELETE(request: Request) {
  const user = await getSessionUser();
  if (!user) return NextResponse.json({ error: "Unauthorized." }, { status: 401 });
  if (user.readOnly) return NextResponse.json({ error: "This account is read-only." }, { status: 403 });
  const body = await request.json().catch(() => null);
  if (!body || typeof body.tradeId !== "string" || !body.tradeId) return NextResponse.json({ error: "Select one merged trade." }, { status: 400 });
  try {
    return NextResponse.json({ trades: await changeTradeMerge(user.id, body.tradeId) });
  } catch (error) {
    return NextResponse.json({ error: error instanceof Error ? error.message : "Could not undo merge." }, { status: 400 });
  }
}
