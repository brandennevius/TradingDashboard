import { NextResponse } from "next/server";
import { getSessionUser } from "@/lib/auth";
import { getCamJournalState, saveCamJournalState } from "@/lib/store";

export async function GET() {
  const user = await getSessionUser();

  if (!user) {
    return NextResponse.json({ error: "Unauthorized." }, { status: 401 });
  }

  try {
    const state = await getCamJournalState();
    return NextResponse.json({ state, readOnly: user.id !== "cam", user });
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Could not load Cam journal." },
      { status: 500 }
    );
  }
}

export async function PUT(request: Request) {
  const user = await getSessionUser();

  if (!user) {
    return NextResponse.json({ error: "Unauthorized." }, { status: 401 });
  }

  if (user.id !== "cam") {
    return NextResponse.json({ error: "Cam journal is read-only for this user." }, { status: 403 });
  }

  const body = await request.json();
  const state = body && typeof body.state === "object" ? body.state : {};
  const serializedState = JSON.stringify(state);

  if (serializedState.includes("data:image/")) {
    return NextResponse.json(
      { error: "Embedded screenshots are not allowed. Upload screenshots separately before saving the journal." },
      { status: 413 }
    );
  }

  try {
    const saved = await saveCamJournalState(state);
    return NextResponse.json({ state: saved, readOnly: false, user });
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Could not save Cam journal." },
      { status: 500 }
    );
  }
}
