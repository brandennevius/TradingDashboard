import { NextResponse } from "next/server";
import { getSessionUser } from "@/lib/auth";
import { generateDailyPortfolioSnapshot, SnapshotValidationError } from "@/lib/daily-portfolio-snapshot-server";
import { sendDailyPortfolioSnapshotEmail } from "@/lib/snapshot-email";

export const runtime = "nodejs";

export async function POST(request: Request) {
  const user = await getSessionUser();
  if (!user || user.id !== "branden") return NextResponse.json({ error: "Unauthorized." }, { status: 401 });
  try {
    const body = await request.json().catch(() => ({}));
    const result = await generateDailyPortfolioSnapshot({
      session: String(body.session || ""),
      accountName: String(body.accountName || "") || undefined,
      // Vercel functions run from a read-only /var/task filesystem. The browser
      // receives both payloads directly and performs the downloads client-side.
      writeExports: false
    });
    const email = body.sendEmail
      ? await sendDailyPortfolioSnapshotEmail({ snapshot: result.snapshot, markdown: result.markdown, baseName: result.baseName })
      : { status: "not_requested" as const };
    return NextResponse.json({
      snapshot: result.snapshot,
      markdown: result.markdown,
      filenames: { json: `${result.baseName}.json`, markdown: `${result.baseName}.md` },
      email
    });
  } catch (error) {
    if (error instanceof SnapshotValidationError) {
      return NextResponse.json({ error: error.message, code: error.code }, { status: 422 });
    }
    return NextResponse.json({ error: error instanceof Error ? error.message : "Database query failed while generating the daily snapshot." }, { status: 500 });
  }
}
