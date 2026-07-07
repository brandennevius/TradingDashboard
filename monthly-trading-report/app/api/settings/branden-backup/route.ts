import { NextResponse } from "next/server";
import { getSessionUser } from "@/lib/auth";
import { exportBrandenJournalBackup, importBrandenJournalBackup } from "@/lib/store";

export async function GET() {
  const user = await getSessionUser();
  if (!user) return NextResponse.json({ error: "Unauthorized." }, { status: 401 });
  if (user.id !== "branden" || user.readOnly) {
    return NextResponse.json({ error: "Only Branden can export this journal." }, { status: 403 });
  }

  try {
    const backup = await exportBrandenJournalBackup();
    const date = backup.exportedAt.slice(0, 10);
    return new NextResponse(JSON.stringify(backup), {
      headers: {
        "Content-Type": "application/json; charset=utf-8",
        "Content-Disposition": `attachment; filename="branden-journal-backup-${date}.json"`,
        "Cache-Control": "no-store"
      }
    });
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Could not export the journal backup." },
      { status: 500 }
    );
  }
}

export async function POST(request: Request) {
  const user = await getSessionUser();
  if (!user) return NextResponse.json({ error: "Unauthorized." }, { status: 401 });
  if (user.id !== "branden" || user.readOnly) {
    return NextResponse.json({ error: "Only Branden can import this journal." }, { status: 403 });
  }

  try {
    const backup = await request.json();
    const restored = await importBrandenJournalBackup(backup);
    return NextResponse.json({ restored });
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Could not import the journal backup." },
      { status: 400 }
    );
  }
}
