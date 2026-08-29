import { NextResponse } from "next/server";
import { getMarketCandlesWithProvider, marketTimeframe } from "@/lib/market-data";

export async function GET(request: Request, { params }: { params: Promise<{ symbol: string }> }) {
  const { symbol: rawSymbol } = await params;
  const timeframe = marketTimeframe(new URL(request.url).searchParams.get("timeframe"));
  const symbol = decodeURIComponent(rawSymbol || "");

  if (!symbol.trim()) {
    return NextResponse.json({ error: "A symbol is required." }, { status: 400 });
  }

  const result = await getMarketCandlesWithProvider(symbol, timeframe);

  if (!result.candles.length) {
    return NextResponse.json({ error: `Could not load ${result.symbol || symbol} candles.`, ...result }, { status: 502 });
  }

  return NextResponse.json(result);
}
