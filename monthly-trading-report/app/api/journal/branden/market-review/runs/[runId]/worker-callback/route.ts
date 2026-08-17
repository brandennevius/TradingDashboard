import { NextResponse } from "next/server";
import { bearerToken, marketReviewErrorResponse } from "@/lib/market-review-api";
import {
  MARKET_REVIEW_CALLBACK_SCHEMA_VERSION,
  MarketReviewValidationError,
  type MarketReviewArtifactInput,
  type MarketReviewCallbackPayload
} from "@/lib/market-review-contract";
import { acceptMarketReviewCallback } from "@/lib/market-review-service";

export const runtime = "nodejs";
export const maxDuration = 60;

type ArtifactMetadata = Omit<MarketReviewArtifactInput, "content_base64">;

function artifactFile(formData: FormData, field: "pdf" | "markdown" | "packet") {
  const value = formData.get(field);
  if (!(value instanceof File)) throw new MarketReviewValidationError("RESULT_ARTIFACT_FILE_MISSING", `Multipart field ${field} is required.`);
  return value;
}

async function completedPayload(request: Request) {
  const formData = await request.formData();
  const metadataText = String(formData.get("metadata") || "");
  if (!metadataText) throw new MarketReviewValidationError("CALLBACK_METADATA_REQUIRED", "Multipart field metadata is required.");
  let metadata: Omit<MarketReviewCallbackPayload, "artifacts"> & { artifacts?: ArtifactMetadata[] };
  try {
    metadata = JSON.parse(metadataText) as typeof metadata;
  } catch {
    throw new MarketReviewValidationError("CALLBACK_METADATA_INVALID", "Multipart metadata must be valid JSON.");
  }
  if (metadata.event_type !== "RESULTS_REGISTERED") {
    throw new MarketReviewValidationError("CALLBACK_EVENT_INVALID", "Multipart callbacks are reserved for RESULTS_REGISTERED.");
  }
  const files = new Map([
    ["pdf", artifactFile(formData, "pdf")],
    ["markdown", artifactFile(formData, "markdown")],
    ["json", artifactFile(formData, "packet")]
  ]);
  const artifactMetadata = metadata.artifacts || [];
  const artifacts: MarketReviewArtifactInput[] = [];
  for (const kind of ["pdf", "markdown", "json"] as const) {
    const expected = artifactMetadata.find((artifact) => artifact.kind === kind);
    if (!expected) throw new MarketReviewValidationError("RESULT_ARTIFACT_METADATA_INVALID", `Metadata for ${kind} is required.`);
    const file = files.get(kind)!;
    artifacts.push({
      ...expected,
      filename: expected.filename || file.name,
      media_type: expected.media_type || file.type,
      content_base64: Buffer.from(await file.arrayBuffer()).toString("base64")
    });
  }
  return { ...metadata, schema_version: MARKET_REVIEW_CALLBACK_SCHEMA_VERSION, artifacts } as MarketReviewCallbackPayload;
}

async function callbackPayload(request: Request) {
  const contentType = request.headers.get("content-type") || "";
  if (contentType.toLowerCase().startsWith("multipart/form-data")) return completedPayload(request);
  const payload = await request.json().catch(() => null) as MarketReviewCallbackPayload | null;
  if (!payload) throw new MarketReviewValidationError("CALLBACK_JSON_INVALID", "The callback body must be valid JSON or multipart form data.");
  if (payload.event_type === "RESULTS_REGISTERED") {
    throw new MarketReviewValidationError("RESULTS_MULTIPART_REQUIRED", "RESULTS_REGISTERED must use multipart form data.");
  }
  return payload;
}

export async function POST(request: Request, context: { params: Promise<{ runId: string }> }) {
  try {
    const token = bearerToken(request);
    if (!token) throw new MarketReviewValidationError("TOKEN_REQUIRED", "A worker callback bearer token is required.");
    const { runId } = await context.params;
    const payload = await callbackPayload(request);
    const result = await acceptMarketReviewCallback(runId, token, payload);
    return NextResponse.json({ run: result.run, duplicate: result.duplicate }, { headers: { "Cache-Control": "no-store" } });
  } catch (error) {
    return marketReviewErrorResponse(error);
  }
}
