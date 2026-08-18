import type { MarketReviewArtifactKind } from "./market-review-contract";

export function marketReviewInlineArtifactUrl(downloadUrl: string) {
  const hashIndex = downloadUrl.indexOf("#");
  const path = hashIndex >= 0 ? downloadUrl.slice(0, hashIndex) : downloadUrl;
  const hash = hashIndex >= 0 ? downloadUrl.slice(hashIndex) : "";
  const separator = path.includes("?") ? "&" : "?";
  return `${path}${separator}disposition=inline${hash}`;
}

export function marketReviewArtifactDisposition(request: Request, kind: MarketReviewArtifactKind) {
  if (kind !== "pdf") return "attachment" as const;
  return new URL(request.url).searchParams.get("disposition") === "inline" ? "inline" as const : "attachment" as const;
}
