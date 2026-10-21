import { after, NextResponse } from "next/server";
import { getSessionUser } from "@/lib/auth";
import { listBrandenVisibleTrades } from "@/lib/store";
import { readMarketArchive } from "@/lib/market-context-store";
import { refreshMarketArchive } from "@/lib/market-context-service";
import { buildTradeMarketContext, summarizeMarketContext } from "@/lib/trade-market-context";

export const maxDuration = 60;
export async function GET(request: Request) {
  const user = await getSessionUser();
  if (!user || (user.id !== "branden" && user.journalOwnerId !== "branden")) return NextResponse.json({ error: "Unauthorized." }, { status: 401 });
  try {
    let archive = await readMarketArchive();
    let warning = "";
    if (!archive.refreshedAt) {
      try { await refreshMarketArchive(); archive = await readMarketArchive(); }
      catch { warning = "Market history is unavailable. Trade records are unaffected."; }
    } else if (Date.now() - Date.parse(archive.refreshedAt) > 60 * 60 * 1000) {
      after(async () => { try { await refreshMarketArchive(); } catch (e) { console.error("Market context refresh failed", e instanceof Error ? e.message : "unknown"); } });
      warning = "Showing saved market history; a refresh is running. Dates identify the available sessions.";
    }
    const trades = await listBrandenVisibleTrades();
    const id = new URL(request.url).searchParams.get("tradeId");
    if (id) {
      const trade = trades.find((t) => t.id === id);
      if (!trade) return NextResponse.json({ error: "Trade not found." }, { status: 404 });
      return NextResponse.json({ context: buildTradeMarketContext(trade, archive), warning }, { headers: { "Cache-Control": "private, no-store" } });
    }
    return NextResponse.json({ contexts: Object.fromEntries(trades.map((t) => [t.id, summarizeMarketContext(buildTradeMarketContext(t, archive))])),
      refreshedAt: archive.refreshedAt, warning }, { headers: { "Cache-Control": "private, no-store" } });
  } catch { return NextResponse.json({ error: "Could not load market history." }, { status: 503 }); }
}

export async function POST() {
  const user = await getSessionUser();
  if (!user || user.id !== "branden" || user.readOnly) return NextResponse.json({ error: "Unauthorized." }, { status: 403 });
  try { return NextResponse.json(await refreshMarketArchive()); }
  catch { return NextResponse.json({ error: "Market history refresh failed. Existing history is retained." }, { status: 502 }); }
}
