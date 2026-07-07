import { NextResponse } from "next/server";
import { getSessionUser } from "@/lib/auth";
import { getBrandenColumnPreferences, saveBrandenColumnPreferences } from "@/lib/store";

export async function GET() {
  const user = await getSessionUser();

  if (!user) {
    return NextResponse.json({ error: "Unauthorized." }, { status: 401 });
  }

  try {
    const preferences = await getBrandenColumnPreferences();
    return NextResponse.json({ preferences });
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Could not load Branden columns." },
      { status: 500 }
    );
  }
}

export async function PUT(request: Request) {
  const user = await getSessionUser();

  if (!user) {
    return NextResponse.json({ error: "Unauthorized." }, { status: 401 });
  }

  if (user.id !== "branden") {
    return NextResponse.json({ error: "Only Branden can update these columns." }, { status: 403 });
  }

  const body = await request.json();
  const preferences = body.preferences && typeof body.preferences === "object" ? body.preferences : {};

  try {
    const saved = await saveBrandenColumnPreferences(preferences);
    return NextResponse.json({ preferences: saved });
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Could not save Branden columns." },
      { status: 500 }
    );
  }
}
