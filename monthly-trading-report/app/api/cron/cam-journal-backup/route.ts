import { NextResponse } from "next/server";
import { snapshotCamJournalState } from "@/lib/store";

export async function GET(request: Request) {
  const authorization = request.headers.get("authorization");
  if (!process.env.CRON_SECRET || authorization !== `Bearer ${process.env.CRON_SECRET}`) {
    return NextResponse.json({ error: "Unauthorized." }, { status: 401 });
  }

  try {
    return NextResponse.json(await snapshotCamJournalState("nightly"));
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Could not create backup." },
      { status: 500 }
    );
  }
}
