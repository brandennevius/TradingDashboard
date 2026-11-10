import { privateJsonResponse } from "@/lib/private-json-response";
import { NextResponse } from "next/server";
import { createApiTimer } from "@/lib/apiTiming";
import { getSessionUser } from "@/lib/auth";
import {
  getBrandenColumnPreferences,
  getBrandenPortfolioSettings,
  getSetupChecklistTemplates,
  listBrandenVisibleTrades
} from "@/lib/store";

export async function GET(request: Request) {
  const logTiming = createApiTimer("/api/journal/branden/trade-log");
  const user = await getSessionUser();

  if (!user) {
    logTiming(401);
    return NextResponse.json({ error: "Unauthorized." }, { status: 401 });
  }

  try {
    const [trades, portfolioSettings, setupChecklists, preferences] = await Promise.all([
      listBrandenVisibleTrades(),
      getBrandenPortfolioSettings(),
      getSetupChecklistTemplates(),
      getBrandenColumnPreferences()
    ]);
    logTiming(200, {
      trades: trades.length,
      portfolios: portfolioSettings.portfolios.length,
      setupChecklists: setupChecklists.length,
      preferences: Array.isArray(preferences) ? preferences.length : 0
    });

    return privateJsonResponse(request, {
      user,
      trades,
      portfolios: portfolioSettings.portfolios,
      defaultPortfolio: portfolioSettings.defaultPortfolio,
      portfolioMeta: portfolioSettings.portfolioMeta || {},
      setupChecklists,
      preferences
    });
  } catch (error) {
    logTiming(500, { error: error instanceof Error ? error.message : "unknown" });
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Could not load trade log." },
      { status: 500 }
    );
  }
}
