import { NextResponse } from "next/server";
import { getSessionUser } from "@/lib/auth";
import { deleteReport } from "@/lib/store";

export async function DELETE(_request: Request, context: { params: Promise<{ id: string }> }) {
  const user = await getSessionUser();

  if (!user) {
    return NextResponse.json({ error: "Unauthorized." }, { status: 401 });
  }

  if (user.readOnly) {
    return NextResponse.json({ error: "This account is read-only." }, { status: 403 });
  }

  const { id } = await context.params;
  try {
    await deleteReport(id, user.id);
    return NextResponse.json({ ok: true });
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Could not delete report." },
      { status: 500 }
    );
  }
}
