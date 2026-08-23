import { NextResponse } from "next/server";
import { createApiTimer } from "@/lib/apiTiming";
import { getSessionUser } from "@/lib/auth";
import { getBrandenPortfolioSettings, listBrandenDailyReviewTrades, listBrokerPortfolioSnapshots } from "@/lib/store";

export async function GET() {
  const logTiming = createApiTimer("/api/journal/branden/daily-review");
  const user = await getSessionUser();

  if (!user) {
    logTiming(401);
    return NextResponse.json({ error: "Unauthorized." }, { status: 401 });
  }

  try {
    const [trades, portfolioSettings, brokerPortfolioSnapshots] = await Promise.all([
      listBrandenDailyReviewTrades(),
      getBrandenPortfolioSettings(),
      listBrokerPortfolioSnapshots(user.id)
    ]);
    logTiming(200, { trades: trades.length, portfolios: portfolioSettings.portfolios.length });

    return NextResponse.json({
      user,
      trades,
      portfolios: portfolioSettings.portfolios,
      defaultPortfolio: portfolioSettings.defaultPortfolio,
      brokerPortfolioSnapshots
    });
  } catch (error) {
    logTiming(500, { error: error instanceof Error ? error.message : "unknown" });
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Could not load daily review." },
      { status: 500 }
    );
  }
}
