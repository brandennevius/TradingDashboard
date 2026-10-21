import { NextResponse } from "next/server";
import { refreshMarketArchive } from "@/lib/market-context-service";
export const maxDuration = 60;
export async function GET(request: Request) {
  if (!process.env.CRON_SECRET || request.headers.get("authorization") !== `Bearer ${process.env.CRON_SECRET}`) return NextResponse.json({ error: "Unauthorized." }, { status: 401 });
  try { return NextResponse.json(await refreshMarketArchive()); }
  catch { return NextResponse.json({ error: "Could not archive completed market sessions." }, { status: 502 }); }
}
