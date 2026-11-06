import assert from "node:assert/strict";
import { GetObjectCommand, PutObjectCommand, S3Client } from "@aws-sdk/client-s3";

const config = {
  R2_ACCOUNT_ID: "0123456789abcdef0123456789abcdef",
  R2_SCREENSHOT_BUCKET: "trade-screenshots",
  R2_ACCESS_KEY_ID: "test-only",
  R2_SECRET_ACCESS_KEY: "test-only",
  CLEANUP_EXPECTED_R2_ACCOUNT_ID: "0123456789abcdef0123456789abcdef",
  CLEANUP_EXPECTED_R2_BUCKET: "trade-screenshots",
};

export function mockR2() {
  const previous = Object.fromEntries(Object.keys(config).map(key => [key, process.env[key]]));
  Object.assign(process.env, config);
  const send = S3Client.prototype.send;
  const objects = new Map<string, Buffer>();
  const controls = { failReads: false, failWrites: false, corrupt: false };
  const failure = (message: string, httpStatusCode: number) =>
    Object.assign(new Error(message), { $metadata: { httpStatusCode } });
  S3Client.prototype.send = (async (command: PutObjectCommand | GetObjectCommand) => {
    assert.equal(command.input.Bucket, config.R2_SCREENSHOT_BUCKET);
    const key = command.input.Key!;
    if (command instanceof PutObjectCommand) {
      assert.equal(command.input.IfNoneMatch, "*");
      if (controls.failWrites) throw failure("R2 upload failed", 503);
      if (objects.has(key)) throw failure("Object exists", 412);
      objects.set(key, Buffer.from(command.input.Body as Uint8Array));
      return {};
    }
    assert.ok(command instanceof GetObjectCommand);
    const bytes = objects.get(key);
    if (!bytes || controls.failReads) throw failure("R2 read failed", 404);
    return { Body: { transformToByteArray: async () => controls.corrupt ? Buffer.from("corrupt") : bytes } };
  }) as typeof send;
  return { objects, controls, restore() {
    S3Client.prototype.send = send;
    for (const [key, value] of Object.entries(previous)) {
      if (value === undefined) delete process.env[key];
      else process.env[key] = value;
    }
  } };
}
