import { NextResponse } from "next/server";
import { getSessionUser } from "./auth";
import { MarketReviewValidationError } from "./market-review-contract";

export async function requireMarketReviewDashboardUser(write = false) {
  const user = await getSessionUser();
  if (!user) throw new MarketReviewValidationError("UNAUTHORIZED", "Sign in to access Campus Fund market reviews.");
  const canAccess = user.id === "branden" || user.journalOwnerId === "branden";
  if (!canAccess) throw new MarketReviewValidationError("FORBIDDEN", "This account cannot access Campus Fund market reviews.");
  if (write && (user.id !== "branden" || user.readOnly)) {
    throw new MarketReviewValidationError("READ_ONLY", "This account cannot start or change a market review.");
  }
  return user;
}

export function bearerToken(request: Request) {
  const header = request.headers.get("authorization") || "";
  return header.startsWith("Bearer ") ? header.slice(7).trim() : "";
}

export function marketReviewErrorResponse(error: unknown) {
  if (error instanceof MarketReviewValidationError) {
    const status = error.code === "UNAUTHORIZED" || error.code.startsWith("TOKEN_") || error.code.startsWith("WORKER_AUTH") ? 401
      : error.code === "FORBIDDEN" || error.code === "READ_ONLY" ? 403
        : error.code.includes("NOT_FOUND") ? 404
          : error.code.includes("CONFLICT") ? 409
            : 422;
    return NextResponse.json({ error: error.message, code: error.code, details: error.details }, { status });
  }
  return NextResponse.json(
    { error: error instanceof Error ? error.message : "Campus Fund market review request failed.", code: "MARKET_REVIEW_INTERNAL_ERROR" },
    { status: 500 }
  );
}
