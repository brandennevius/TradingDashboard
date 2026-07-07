import crypto from "crypto";
import { NextResponse } from "next/server";
import { getSessionUser } from "@/lib/auth";
import { createFeedbackTicket, listFeedbackTickets } from "@/lib/store";
import type { FeedbackTicketKind } from "@/lib/types";

function stringValue(value: unknown) {
  return String(value || "").trim();
}

function stringArray(value: unknown) {
  if (!Array.isArray(value)) {
    return [];
  }

  return value.map(String).map((item) => item.trim()).filter(Boolean);
}

function ticketKind(value: unknown): FeedbackTicketKind {
  return String(value || "").toUpperCase() === "FEATURE" ? "FEATURE" : "BUG";
}

export async function GET() {
  const user = await getSessionUser();

  if (!user) {
    return NextResponse.json({ error: "Unauthorized." }, { status: 401 });
  }

  try {
    const tickets = await listFeedbackTickets();
    const visibleTickets = user.id === "branden" ? tickets : tickets.filter((ticket) => ticket.submittedBy === user.id);
    return NextResponse.json({ tickets: visibleTickets });
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Could not load tickets." },
      { status: 500 }
    );
  }
}

export async function POST(request: Request) {
  const user = await getSessionUser();

  if (!user) {
    return NextResponse.json({ error: "Unauthorized." }, { status: 401 });
  }

  if (user.readOnly) {
    return NextResponse.json({ error: "This account is read-only." }, { status: 403 });
  }

  const body = await request.json();
  const kind = ticketKind(body.kind);
  const title = stringValue(body.title);
  const summary = stringValue(body.summary);

  if (!title || !summary) {
    return NextResponse.json({ error: "Title and summary are required." }, { status: 400 });
  }

  try {
    const ticket = await createFeedbackTicket({
      kind,
      status: "OPEN",
      title,
      summary,
      details: stringValue(body.details),
      expectedBehavior: stringValue(body.expectedBehavior),
      reproductionSteps: stringValue(body.reproductionSteps),
      businessValue: stringValue(body.businessValue),
      screenshots: stringArray(body.screenshots),
      submittedBy: user.id,
      source: stringValue(body.source) || "cam-journal",
      messages: [
        {
          id: crypto.randomUUID(),
          author: user.id === "branden" ? "ADMIN" : "CAM",
          body: [
            summary,
            stringValue(body.details),
            kind === "BUG" && stringValue(body.expectedBehavior)
              ? `Expected: ${stringValue(body.expectedBehavior)}`
              : "",
            kind === "BUG" && stringValue(body.reproductionSteps)
              ? `Reproduction: ${stringValue(body.reproductionSteps)}`
              : "",
            kind === "FEATURE" && stringValue(body.businessValue)
              ? `Why it matters: ${stringValue(body.businessValue)}`
              : ""
          ]
            .filter(Boolean)
            .join("\n\n"),
          screenshots: stringArray(body.screenshots),
          createdAt: new Date().toISOString()
        }
      ],
      resolutionNotes: ""
    });

    return NextResponse.json({ ticket });
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Could not submit ticket." },
      { status: 500 }
    );
  }
}
