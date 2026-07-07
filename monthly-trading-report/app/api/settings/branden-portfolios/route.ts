import { NextResponse } from "next/server";
import { getSessionUser } from "@/lib/auth";
import { getBrandenPortfolioSettings, saveBrandenPortfolioSettings } from "@/lib/store";

function normalizeNames(value: unknown) {
  if (!Array.isArray(value)) {
    return [];
  }

  return Array.from(new Set(value.map((name) => String(name || "").trim()).filter(Boolean))).sort((a, b) =>
    a.localeCompare(b)
  );
}

export async function GET() {
  const user = await getSessionUser();

  if (!user) {
    return NextResponse.json({ error: "Unauthorized." }, { status: 401 });
  }

  try {
    const settings = await getBrandenPortfolioSettings();
    return NextResponse.json(settings);
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Could not load portfolios." },
      { status: 500 }
    );
  }
}

export async function PUT(request: Request) {
  const user = await getSessionUser();

  if (!user) {
    return NextResponse.json({ error: "Unauthorized." }, { status: 401 });
  }

  if (user.id !== "branden") {
    return NextResponse.json({ error: "Only Branden can update these portfolios." }, { status: 403 });
  }

  const body = await request.json();
  const portfolios = normalizeNames(body.portfolios);
  const defaultPortfolio = String(body.defaultPortfolio || "").trim();

  try {
    const current = await getBrandenPortfolioSettings();
    const saved = await saveBrandenPortfolioSettings({ portfolios, defaultPortfolio, portfolioMeta: current.portfolioMeta });
    return NextResponse.json(saved);
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Could not save portfolios." },
      { status: 500 }
    );
  }
}
