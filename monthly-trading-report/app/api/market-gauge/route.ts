import { NextResponse } from "next/server";
import { calculateGauge, loadGaugeCandles } from "@/lib/market-gauge";
export async function GET(request: Request) {
  const requested = new URL(request.url).searchParams.get("session_date") || "";
  if (requested && !/^\d{4}-\d{2}-\d{2}$/.test(requested)) return NextResponse.json({ error: "session_date must use YYYY-MM-DD" }, { status: 400 });
  try {
    const data = await loadGaugeCandles();
    const session = requested || data.SPY?.at(-1)?.time || "";
    const gauge = calculateGauge(data, session);
    if (!gauge) return NextResponse.json({ error: "Exact-session index evidence is incomplete.", code: "MARKET_GAUGE_SESSION_EVIDENCE_INCOMPLETE" }, { status: 422 });
    return NextResponse.json(gauge, { headers: { "Cache-Control": requested ? "private, no-store" : "public, s-maxage=900, stale-while-revalidate=3600" } });
  } catch { return NextResponse.json({ error: "Could not load market gauge." }, { status: 502 }); }
}
