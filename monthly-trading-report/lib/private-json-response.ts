import { createHash } from "node:crypto";

/** Revalidate in a private browser cache; authentication must precede this call. */
export function privateJsonResponse(request: Request, payload: unknown) {
  const body = JSON.stringify(payload);
  const tag = `W/"${createHash("sha256").update(body).digest("hex")}"`;
  const headers = new Headers({
    "Content-Type": "application/json; charset=utf-8",
    "Cache-Control": "private, no-cache",
    "Vary": "Cookie",
    "ETag": tag
  });
  const supplied = request.headers.get("if-none-match");
  const matches = supplied?.split(",").some(value => {
    const candidate = value.trim();
    return candidate === "*" || candidate.replace(/^W\//, "") === tag.replace(/^W\//, "");
  });
  return new Response(matches ? null : body, { status: matches ? 304 : 200, headers });
}
