import { NextResponse } from "next/server";
import { getSessionUser } from "@/lib/auth";
import { sendMonthToDateSnapshotEmail } from "@/lib/month-to-date-snapshot-email";
import { generateMonthToDateSnapshot, MonthToDateSnapshotValidationError } from "@/lib/month-to-date-snapshot-server";
import { snapshotEmailConfiguration } from "@/lib/snapshot-email";

export const runtime = "nodejs";

export async function GET() {
  const user = await getSessionUser();
  if (!user || user.id !== "branden") return NextResponse.json({ error: "Unauthorized." }, { status: 401 });
  return NextResponse.json({ emailConfigured: snapshotEmailConfiguration().configured });
}

export async function POST(request: Request) {
  const user = await getSessionUser();
  if (!user || user.id !== "branden") return NextResponse.json({ error: "Unauthorized." }, { status: 401 });
  try {
    const body = await request.json().catch(() => ({}));
    const result = await generateMonthToDateSnapshot({
      month: String(body.month || "") || undefined,
      asOfDate: String(body.asOfDate || "") || undefined,
      asOfTimestamp: String(body.asOfTimestamp || "") || undefined,
      portfolioName: String(body.portfolioName || "") || undefined,
      writeExports: false
    });
    const email = body.sendEmail
      ? await sendMonthToDateSnapshotEmail({ snapshot: result.snapshot, markdown: result.markdown, baseName: result.baseName })
      : { status: "not_requested" as const };
    return NextResponse.json({
      snapshot: result.snapshot,
      markdown: result.markdown,
      filenames: { json: `${result.baseName}.json`, markdown: `${result.baseName}.md`, zip: `${result.baseName}.zip` },
      email
    });
  } catch (error) {
    if (error instanceof MonthToDateSnapshotValidationError) {
      return NextResponse.json({ error: error.message, status: "BLOCKED", code: error.code, diagnostic: error.diagnostic }, { status: 422 });
    }
    return NextResponse.json({ error: error instanceof Error ? error.message : "Could not generate the MTD snapshot." }, { status: 500 });
  }
}
