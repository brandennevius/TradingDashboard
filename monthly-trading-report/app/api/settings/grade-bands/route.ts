import { NextResponse } from "next/server";
import { getSessionUser } from "@/lib/auth";
import { getChecklistGradeBands, saveChecklistGradeBands } from "@/lib/store";
import type { ChecklistGradeBand } from "@/lib/types";

function normalizeBands(value: unknown): ChecklistGradeBand[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value
    .map((band, index) => {
      if (!band || typeof band !== "object") {
        return null;
      }

      const rawBand = band as Record<string, unknown>;
      const label = String(rawBand.label || "").trim();
      const minScore = Number(rawBand.minScore);
      const maxScore = rawBand.maxScore === null || rawBand.maxScore === "" ? null : Number(rawBand.maxScore);

      if (!label || !Number.isFinite(minScore) || (maxScore !== null && !Number.isFinite(maxScore))) {
        return null;
      }

      return {
        id: String(rawBand.id || `grade-${index}-${Date.now()}`),
        label,
        minScore,
        maxScore
      };
    })
    .filter(Boolean) as ChecklistGradeBand[];
}

export async function GET() {
  const user = await getSessionUser();

  if (!user) {
    return NextResponse.json({ error: "Unauthorized." }, { status: 401 });
  }

  try {
    const gradeBands = await getChecklistGradeBands();
    return NextResponse.json({ gradeBands });
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Could not load grade bands." },
      { status: 500 }
    );
  }
}

export async function PUT(request: Request) {
  const user = await getSessionUser();

  if (!user) {
    return NextResponse.json({ error: "Unauthorized." }, { status: 401 });
  }

  if (user.readOnly) {
    return NextResponse.json({ error: "This account is read-only." }, { status: 403 });
  }

  const body = await request.json();
  const gradeBands = normalizeBands(body.gradeBands);

  if (!gradeBands.length) {
    return NextResponse.json({ error: "At least one grade band is required." }, { status: 400 });
  }

  if (gradeBands.some((band) => band.maxScore !== null && band.maxScore < band.minScore)) {
    return NextResponse.json({ error: "A max score cannot be lower than its min score." }, { status: 400 });
  }

  try {
    const savedBands = await saveChecklistGradeBands(gradeBands);
    return NextResponse.json({ gradeBands: savedBands });
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Could not save grade bands." },
      { status: 500 }
    );
  }
}
