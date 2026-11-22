import { NextResponse } from "next/server";
import { getSessionUser } from "@/lib/auth";
import { listBrokerPortfolioEquityHistory } from "@/lib/store";
import { privateJsonResponse } from "@/lib/private-json-response";

export async function GET(request: Request) {
  if (!await getSessionUser()) return NextResponse.json({ error: "Unauthorized." }, { status: 401 });
  try {
    return privateJsonResponse(request, { equityHistory: await listBrokerPortfolioEquityHistory("branden") });
  } catch {
    return NextResponse.json({ error: "Could not load account equity history." }, { status: 500 });
  }
}
