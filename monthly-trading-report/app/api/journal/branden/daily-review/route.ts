import { NextResponse } from "next/server";
import { createApiTimer } from "@/lib/apiTiming";
import { getSessionUser } from "@/lib/auth";
import { getBrandenPortfolioSettings, listBrandenDailyReviewTrades } from "@/lib/store";

export async function GET() {
  const logTiming = createApiTimer("/api/journal/branden/daily-review");
  const user = await getSessionUser();

  if (!user) {
    logTiming(401);
    return NextResponse.json({ error: "Unauthorized." }, { status: 401 });
  }

  try {
    const [trades, portfolioSettings] = await Promise.all([listBrandenDailyReviewTrades(), getBrandenPortfolioSettings()]);
    logTiming(200, { trades: trades.length, portfolios: portfolioSettings.portfolios.length });

    return NextResponse.json({
      user,
      trades,
      portfolios: portfolioSettings.portfolios,
      defaultPortfolio: portfolioSettings.defaultPortfolio,
      portfolioMeta: portfolioSettings.portfolioMeta || {}
    });
  } catch (error) {
    logTiming(500, { error: error instanceof Error ? error.message : "unknown" });
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Could not load daily review." },
      { status: 500 }
    );
  }
}
