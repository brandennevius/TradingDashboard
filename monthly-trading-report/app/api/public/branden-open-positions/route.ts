import { NextResponse } from "next/server";
import { createApiTimer } from "@/lib/apiTiming";
import { getSessionUser } from "@/lib/auth";
import { getBrandenPortfolioSettings, listBrandenOpenPositionTrades } from "@/lib/store";

export async function GET() {
  const logTiming = createApiTimer("/api/public/branden-open-positions");
  try {
    const [user, trades, portfolioSettings] = await Promise.all([
      getSessionUser(),
      listBrandenOpenPositionTrades(),
      getBrandenPortfolioSettings()
    ]);
    const publicTrades = trades
      .map((trade) => ({
        id: trade.id,
        userId: trade.userId,
        symbol: trade.symbol,
        side: trade.side,
        status: trade.status,
        entryDate: trade.entryDate,
        avgEntry: trade.avgEntry,
        stopPrice: trade.stopPrice,
        shares: trade.shares,
        risk: trade.risk,
        pnl: trade.pnl,
        rMultiple: trade.rMultiple,
        portfolioTag: trade.portfolioTag,
        hidden: false
      }));
    logTiming(200, { trades: publicTrades.length, portfolios: portfolioSettings.portfolios.length, user: user?.id || null });

    return NextResponse.json({
      user,
      trades: publicTrades,
      portfolios: portfolioSettings.portfolios,
      defaultPortfolio: portfolioSettings.defaultPortfolio,
      portfolioMeta: portfolioSettings.portfolioMeta || {}
    });
  } catch (error) {
    logTiming(500, { error: error instanceof Error ? error.message : "unknown" });
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Could not load open positions." },
      { status: 500 }
    );
  }
}
