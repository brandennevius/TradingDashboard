import { NextResponse } from "next/server";
import { getSessionUser } from "@/lib/auth";
import { appendFeedbackTicketMessage, listFeedbackTickets, updateFeedbackTicket } from "@/lib/store";

function statusValue(value: unknown) {
  const status = String(value || "").toUpperCase();
  return status === "IN_PROGRESS" || status === "COMPLETED" ? status : "OPEN";
}

export async function PUT(request: Request, context: { params: Promise<{ id: string }> }) {
  const user = await getSessionUser();

  if (!user) {
    return NextResponse.json({ error: "Unauthorized." }, { status: 401 });
  }

  if (user.id !== "branden") {
    return NextResponse.json({ error: "Admin only." }, { status: 403 });
  }

  const { id } = await context.params;
  const body = await request.json();

  try {
    const tickets = await updateFeedbackTicket(id, {
      status: body.status ? statusValue(body.status) : undefined,
      resolutionNotes: body.resolutionNotes === undefined ? undefined : String(body.resolutionNotes || "")
    });
    const ticket = tickets.find((item) => item.id === id);

    if (!ticket) {
      return NextResponse.json({ error: "Ticket not found." }, { status: 404 });
    }

    return NextResponse.json({ ticket });
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Could not update ticket." },
      { status: 500 }
    );
  }
}

export async function GET(_request: Request, context: { params: Promise<{ id: string }> }) {
  const user = await getSessionUser();

  if (!user) {
    return NextResponse.json({ error: "Unauthorized." }, { status: 401 });
  }

  const { id } = await context.params;

  try {
    const tickets = await listFeedbackTickets();
    const ticket = tickets.find((item) => item.id === id && (user.id === "branden" || item.submittedBy === user.id));

    if (!ticket) {
      return NextResponse.json({ error: "Ticket not found." }, { status: 404 });
    }

    return NextResponse.json({ ticket });
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Could not load ticket." },
      { status: 500 }
    );
  }
}

export async function POST(request: Request, context: { params: Promise<{ id: string }> }) {
  const user = await getSessionUser();

  if (!user) {
    return NextResponse.json({ error: "Unauthorized." }, { status: 401 });
  }

  if (user.readOnly) {
    return NextResponse.json({ error: "This account is read-only." }, { status: 403 });
  }

  const { id } = await context.params;

  try {
    const tickets = await listFeedbackTickets();
    const ticket = tickets.find((item) => item.id === id);

    if (!ticket) {
      return NextResponse.json({ error: "Ticket not found." }, { status: 404 });
    }

    if (user.id !== "branden" && ticket.submittedBy !== user.id) {
      return NextResponse.json({ error: "Forbidden." }, { status: 403 });
    }

    const body = await request.json();
    const messageBody = String(body.body || "").trim();
    const screenshots = Array.isArray(body.screenshots) ? body.screenshots.map(String).filter(Boolean) : [];

    if (!messageBody) {
      return NextResponse.json({ error: "Message body is required." }, { status: 400 });
    }

    const updatedTickets = await appendFeedbackTicketMessage(id, {
      author: user.id === "branden" ? "ADMIN" : "CAM",
      body: messageBody,
      screenshots
    });
    const updatedTicket = updatedTickets.find((item) => item.id === id);

    return NextResponse.json({ ticket: updatedTicket });
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Could not add message." },
      { status: 500 }
    );
  }
}
