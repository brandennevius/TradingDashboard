export function buildDailySnapshotRequestBody(session: string, accountName: string, sendEmail: boolean) {
  return { session: session.trim(), accountName: accountName.trim(), sendEmail };
}

export function snapshotSessionFromRequestBody(body: { session?: unknown }, fallback = "") {
  const submitted = typeof body.session === "string" ? body.session.trim() : "";
  return submitted || fallback;
}
