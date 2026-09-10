import { NextResponse } from "next/server";
import { getSessionUser } from "@/lib/auth";
import { fetchTradeIntradayBars } from "@/lib/trade-excursion-market-data";
import { calculateTradeExcursion, excursionInstrument, tradeExcursionInputHash, unavailableTradeExcursion } from "@/lib/trade-excursion";
import { getCachedTradeExcursion, getTradeForUser, saveCachedTradeExcursion } from "@/lib/store";

export const maxDuration = 60;

function newYorkDate() {
  return new Intl.DateTimeFormat("en-CA", {
    timeZone: "America/New_York", year: "numeric", month: "2-digit", day: "2-digit"
  }).format(new Date());
}

export async function GET(_request: Request, context: { params: Promise<{ id: string }> }) {
  const user = await getSessionUser();
  if (!user) return NextResponse.json({ error: "Unauthorized." }, { status: 401 });
  const { id } = await context.params;

  try {
    const trade = await getTradeForUser(id, user.id);
    if (!trade) return NextResponse.json({ error: "Trade not found." }, { status: 404 });
    const asOf = trade.status === "OPEN" ? newYorkDate() : trade.exitDate;
    const inputHash = tradeExcursionInputHash(trade, asOf);
    const cached = await getCachedTradeExcursion(trade.id, user.id, inputHash);
    if (cached) return NextResponse.json({ excursion: cached, cached: true });
    const instrument = excursionInstrument(trade.symbol);
    if (!instrument.providerSymbol) {
      return NextResponse.json({ excursion: unavailableTradeExcursion(trade, "No compatible market-data symbol is configured.", { asOf }) });
    }
    const marketData = await fetchTradeIntradayBars(instrument, trade.entryDate, asOf);
    const excursion = marketData.error
      ? unavailableTradeExcursion(trade, marketData.error, { provider: marketData.provider, interval: marketData.interval, asOf })
      : calculateTradeExcursion(trade, marketData.bars, {
          provider: marketData.provider, interval: marketData.interval, asOf, instrument
        });
    await saveCachedTradeExcursion(trade.id, user.id, excursion);
    return NextResponse.json({ excursion });
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Could not calculate trade excursion." },
      { status: 500 }
    );
  }
}
