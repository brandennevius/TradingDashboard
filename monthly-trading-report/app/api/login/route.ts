import { NextResponse } from "next/server";
import { authenticate, setSession } from "@/lib/auth";

export async function POST(request: Request) {
  const body = (await request.json()) as { userId?: string; password?: string };
  const user = authenticate(body.userId || "", body.password || "");

  if (!user) {
    return NextResponse.json({ error: "Invalid login." }, { status: 401 });
  }

  await setSession(user);
  return NextResponse.json({ user });
}
