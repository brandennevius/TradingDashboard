import { GetObjectCommand, paginateListObjectsV2, PutObjectCommand, S3Client } from "@aws-sdk/client-s3";
import sharp from "sharp";
import { createHash } from "node:crypto";

export type ScreenshotTable = "trade_screenshots" | "cam_journal_screenshots";
export type ImageSource = {
  image_data: Buffer | null;
  storage_path?: string | null;
  image_sha256?: string | null;
  image_bytes?: number | string | null;
};

export const imageHash = (bytes: Buffer) => createHash("sha256").update(bytes).digest("hex");

export const MAX_SCREENSHOT_INPUT_PIXELS = 40_000_000;

export class ScreenshotPixelLimitError extends Error {
  readonly status = 413;

  constructor() {
    super(`Screenshot exceeds the limit of ${MAX_SCREENSHOT_INPUT_PIXELS / 1_000_000} million pixels. Crop the image or reduce its width and height, then upload again.`);
    this.name = "ScreenshotPixelLimitError";
  }
}

export function screenshotStorageTarget() {
  const account = process.env.R2_ACCOUNT_ID;
  const bucket = process.env.R2_SCREENSHOT_BUCKET;
  if (!account || !/^[a-f0-9]{32}$/.test(account) ||
      !bucket || !/^[a-z0-9][a-z0-9-]{1,61}[a-z0-9]$/.test(bucket)) {
    throw new Error("Private screenshot R2 storage is not configured correctly.");
  }
  return { account, bucket };
}

let cached: { account: string; accessKeyId: string; secretAccessKey: string; client: S3Client } | undefined;
function storage() {
  const { account, bucket } = screenshotStorageTarget();
  const accessKeyId = process.env.R2_ACCESS_KEY_ID;
  const secretAccessKey = process.env.R2_SECRET_ACCESS_KEY;
  if (!accessKeyId || !secretAccessKey) throw new Error("Private screenshot R2 credentials are missing.");
  if (cached?.account !== account || cached.accessKeyId !== accessKeyId || cached.secretAccessKey !== secretAccessKey) {
    cached?.client.destroy();
    cached = { account, accessKeyId, secretAccessKey, client: new S3Client({
      endpoint: `https://${account}.r2.cloudflarestorage.com`,
      region: "auto",
      forcePathStyle: true,
      credentials: { accessKeyId, secretAccessKey },
      requestChecksumCalculation: "WHEN_REQUIRED",
      responseChecksumValidation: "WHEN_REQUIRED",
    }) };
  }
  return { client: cached.client, bucket };
}

export async function screenshotBucketBytes() {
  const { client, bucket } = storage();
  let bytes = 0;
  for await (const page of paginateListObjectsV2({ client }, { Bucket: bucket })) {
    for (const object of page.Contents ?? []) bytes += object.Size ?? 0;
  }
  return bytes;
}

function validatePath(path: string) {
  if (!/^(trade_screenshots|cam_journal_screenshots)\/[A-Za-z0-9_-]+\/[a-f0-9]{64}$/.test(path)) {
    throw new Error("Invalid screenshot Storage path.");
  }
}

export async function encodeNewScreenshot(fileName: string, mimeType: string, bytes: Buffer) {
  if (process.env.SCREENSHOT_STORAGE_WRITES !== "enabled") return { fileName, mimeType, bytes };
  try {
    const jpeg = await sharp(bytes, { limitInputPixels: MAX_SCREENSHOT_INPUT_PIXELS })
      .rotate()
      .resize(2560, 2560, { fit: "inside", withoutEnlargement: true })
      .flatten({ background: "#ffffff" })
      .jpeg({ quality: 85, chromaSubsampling: "4:4:4", mozjpeg: true })
      .toBuffer();
    return { fileName: fileName.replace(/\.[^.]+$/, "") + ".jpg", mimeType: "image/jpeg", bytes: jpeg };
  } catch (error) {
    if (error instanceof Error && /input image exceeds pixel limit/i.test(error.message)) {
      throw new ScreenshotPixelLimitError();
    }
    throw error;
  }
}

export async function readStoredImage(source: ImageSource): Promise<Buffer> {
  if (source.storage_path) {
    // Fail closed on corrupt/missing objects. A verified bytea copy can still serve
    // during the migration; Storage-only rows must never silently return no image.
    try {
      validatePath(source.storage_path);
      const { client, bucket } = storage();
      const response = await client.send(new GetObjectCommand({ Bucket: bucket, Key: source.storage_path }), {
        abortSignal: AbortSignal.timeout(30_000),
      });
      if (!response.Body) throw new Error("Screenshot R2 object has no body.");
      const bytes = Buffer.from(await response.Body.transformToByteArray());
      if (!source.image_sha256 || imageHash(bytes) !== source.image_sha256 || bytes.length !== Number(source.image_bytes)) {
        throw new Error("Screenshot Storage integrity check failed.");
      }
      return bytes;
    } catch (error) {
      if (source.image_data && imageHash(source.image_data) === source.image_sha256) return source.image_data;
      throw error;
    }
  }
  if (!source.image_data) throw new Error("Screenshot has no backing image.");
  return source.image_data;
}

/** Immutable upload plus download verification. Safe to retry after a DB failure. */
export async function copyImageToStorage(table: ScreenshotTable, id: string, mimeType: string, bytes: Buffer) {
  const hash = imageHash(bytes);
  const path = `${table}/${id}/${hash}`;
  validatePath(path);
  const { client, bucket } = storage();
  try {
    await client.send(new PutObjectCommand({
      Bucket: bucket, Key: path, Body: bytes, ContentType: mimeType || "application/octet-stream",
      IfNoneMatch: "*",
    }), { abortSignal: AbortSignal.timeout(30_000) });
  } catch (error) {
    if ((error as { $metadata?: { httpStatusCode?: number } }).$metadata?.httpStatusCode !== 412) throw error;
  }
  await readStoredImage({ image_data: null, storage_path: path, image_sha256: hash, image_bytes: bytes.length });
  return { storage_path: path, image_sha256: hash, image_bytes: bytes.length };
}

export async function prepareImage(table: ScreenshotTable, id: string, mimeType: string, bytes: Buffer): Promise<ImageSource> {
  if (process.env.SCREENSHOT_STORAGE_WRITES !== "enabled") return { image_data: bytes };
  return { image_data: null, ...await copyImageToStorage(table, id, mimeType, bytes) };
}

export function sharedScreenshotIds(settings: unknown): string[] {
  const ids = new Set<string>();
  function visit(value: unknown) {
    if (typeof value === "string") {
      const match = value.match(/^\/api\/(?:cam-journal|watchlists)\/screenshots\/([A-Za-z0-9_-]+)$/);
      if (match) ids.add(match[1]);
    } else if (Array.isArray(value)) value.forEach(visit);
    else if (value && typeof value === "object") Object.values(value).forEach(visit);
  }
  visit(settings);
  return [...ids].sort();
}
