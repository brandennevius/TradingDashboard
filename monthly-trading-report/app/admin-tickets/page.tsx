"use client";

import { useEffect, useMemo, useState } from "react";
import type { FeedbackTicket } from "@/lib/types";

type TicketStatusFilter = "ALL" | "OPEN" | "IN_PROGRESS" | "COMPLETED";

export default function AdminTicketsPage() {
  const [tickets, setTickets] = useState<FeedbackTicket[]>([]);
  const [status, setStatus] = useState("");
  const [filter, setFilter] = useState<TicketStatusFilter>("ALL");
  const [selectedId, setSelectedId] = useState("");
  const [notesById, setNotesById] = useState<Record<string, string>>({});
  const [replyById, setReplyById] = useState<Record<string, string>>({});

  async function loadTickets(nextSelectedId?: string) {
    const response = await fetch("/api/tickets");
    const data = await response.json();

    if (!response.ok) {
      setStatus(data.error || "Could not load tickets.");
      setTickets([]);
      return;
    }

    const loadedTickets = data.tickets || [];
    setStatus("");
    setTickets(loadedTickets);
    setNotesById(
      Object.fromEntries(loadedTickets.map((ticket: FeedbackTicket) => [ticket.id, ticket.resolutionNotes || ""]))
    );
    setSelectedId((current) => {
      const candidate = nextSelectedId || current;
      if (candidate && loadedTickets.some((ticket: FeedbackTicket) => ticket.id === candidate)) {
        return candidate;
      }
      return loadedTickets[0]?.id || "";
    });
  }

  async function updateTicket(id: string, nextStatus: FeedbackTicket["status"]) {
    setStatus(`Updating ${id}...`);
    const response = await fetch(`/api/tickets/${id}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        status: nextStatus,
        resolutionNotes: notesById[id] || ""
      })
    });
    const data = await response.json();

    if (!response.ok) {
      setStatus(data.error || "Could not update ticket.");
      return;
    }

    setStatus(`${data.ticket.title} updated.`);
    await loadTickets(id);
  }

  async function sendReply(id: string) {
    const body = (replyById[id] || "").trim();

    if (!body) {
      setStatus("Reply is empty.");
      return;
    }

    setStatus(`Replying to ${id}...`);
    const response = await fetch(`/api/tickets/${id}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ body })
    });
    const data = await response.json();

    if (!response.ok) {
      setStatus(data.error || "Could not send reply.");
      return;
    }

    setReplyById((current) => ({ ...current, [id]: "" }));
    setStatus("Reply sent.");
    await loadTickets(id);
  }

  useEffect(() => {
    loadTickets();
  }, []);

  const visibleTickets = useMemo(
    () => (filter === "ALL" ? tickets : tickets.filter((ticket) => ticket.status === filter)),
    [filter, tickets]
  );

  const selectedTicket =
    visibleTickets.find((ticket) => ticket.id === selectedId) ||
    tickets.find((ticket) => ticket.id === selectedId) ||
    visibleTickets[0] ||
    null;

  return (
    <main className="admin-tickets-page">
      <header className="topbar">
        <div>
          <p className="eyebrow">Hidden admin</p>
          <h1>Feedback Tickets</h1>
        </div>
        <div className="top-actions">
          <a className="ghost-button" href="/">
            Back to app
          </a>
          <button className="ghost-button" type="button" onClick={() => loadTickets(selectedTicket?.id)}>
            Refresh
          </button>
        </div>
      </header>

      <section className="admin-ticket-toolbar">
        <label>
          Status
          <select value={filter} onChange={(event) => setFilter(event.target.value as TicketStatusFilter)}>
            <option value="ALL">All</option>
            <option value="OPEN">Open</option>
            <option value="IN_PROGRESS">In Progress</option>
            <option value="COMPLETED">Completed</option>
          </select>
        </label>
        {status ? <span className="status">{status}</span> : null}
      </section>

      <section className="admin-ticket-layout">
        <aside className="admin-ticket-sidebar">
          {visibleTickets.map((ticket) => (
            <button
              className={`admin-ticket-list-item ${selectedTicket?.id === ticket.id ? "active" : ""}`}
              key={ticket.id}
              type="button"
              onClick={() => setSelectedId(ticket.id)}
            >
              <div className="admin-ticket-list-top">
                <strong>{ticket.title}</strong>
                <span className={`ticket-status-pill ${ticket.status.toLowerCase()}`}>{ticket.status.replace("_", " ")}</span>
              </div>
              <p>{ticket.summary}</p>
              <span className="small">
                {ticket.kind} · {ticket.submittedBy} · {new Date(ticket.updatedAt).toLocaleString("en-US")}
              </span>
            </button>
          ))}
          {!visibleTickets.length ? <p className="empty-state">No tickets in this view.</p> : null}
        </aside>

        <section className="admin-ticket-thread-panel">
          {!selectedTicket ? (
            <p className="empty-state">Choose a ticket to open the thread.</p>
          ) : (
            <article className="admin-ticket-card">
              <div className="admin-ticket-head">
                <div>
                  <p className="eyebrow">
                    {selectedTicket.kind} · {selectedTicket.submittedBy} · {new Date(selectedTicket.createdAt).toLocaleString("en-US")}
                  </p>
                  <h2>{selectedTicket.title}</h2>
                </div>
                <span className={`ticket-status-pill ${selectedTicket.status.toLowerCase()}`}>
                  {selectedTicket.status.replace("_", " ")}
                </span>
              </div>

              <p className="admin-ticket-summary">{selectedTicket.summary}</p>

              <div className="admin-ticket-meta-grid">
                {selectedTicket.details ? (
                  <div>
                    <h3>Details</h3>
                    <p>{selectedTicket.details}</p>
                  </div>
                ) : null}
                {selectedTicket.expectedBehavior ? (
                  <div>
                    <h3>Expected</h3>
                    <p>{selectedTicket.expectedBehavior}</p>
                  </div>
                ) : null}
                {selectedTicket.reproductionSteps ? (
                  <div>
                    <h3>Reproduction</h3>
                    <p>{selectedTicket.reproductionSteps}</p>
                  </div>
                ) : null}
                {selectedTicket.businessValue ? (
                  <div>
                    <h3>Why it matters</h3>
                    <p>{selectedTicket.businessValue}</p>
                  </div>
                ) : null}
              </div>

              <section className="admin-ticket-thread">
                {selectedTicket.messages.map((message) => (
                  <div
                    className={`ticket-message ${message.author === "ADMIN" ? "admin" : "cam"}`}
                    key={message.id}
                  >
                    <div className="ticket-message-head">
                      <strong>{message.author === "ADMIN" ? "Codex / Admin" : "Cam"}</strong>
                      <span>{new Date(message.createdAt).toLocaleString("en-US")}</span>
                    </div>
                    <p>{message.body}</p>
                    {message.screenshots.length ? (
                      <div className="admin-ticket-shots">
                        {message.screenshots.map((shot, index) => (
                          <img key={`${message.id}-${index}`} src={shot} alt={`Attachment ${index + 1}`} />
                        ))}
                      </div>
                    ) : null}
                  </div>
                ))}
              </section>

              <label className="admin-ticket-notes">
                Resolution notes
                <textarea
                  value={notesById[selectedTicket.id] || ""}
                  onChange={(event) =>
                    setNotesById((current) => ({ ...current, [selectedTicket.id]: event.target.value }))
                  }
                  placeholder="Implementation notes, commit reference, rollout details."
                />
              </label>

              <label className="admin-ticket-notes">
                Reply in thread
                <textarea
                  value={replyById[selectedTicket.id] || ""}
                  onChange={(event) =>
                    setReplyById((current) => ({ ...current, [selectedTicket.id]: event.target.value }))
                  }
                  placeholder="Reply back to Cam here."
                />
              </label>

              <div className="admin-ticket-actions">
                <button type="button" onClick={() => sendReply(selectedTicket.id)}>
                  Send Reply
                </button>
                <button type="button" onClick={() => updateTicket(selectedTicket.id, "OPEN")}>
                  Reopen
                </button>
                <button className="ghost-button" type="button" onClick={() => updateTicket(selectedTicket.id, "IN_PROGRESS")}>
                  Mark In Progress
                </button>
                <button type="button" onClick={() => updateTicket(selectedTicket.id, "COMPLETED")}>
                  Mark Completed
                </button>
              </div>
            </article>
          )}
        </section>
      </section>
    </main>
  );
}
