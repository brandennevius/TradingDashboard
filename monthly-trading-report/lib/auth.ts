import crypto from "crypto";
import { cookies } from "next/headers";
import type { TraderUser } from "./types";

const COOKIE_NAME = "trading_report_session";

function getSecret() {
  return process.env.APP_SECRET || "local-dev-secret-change-before-deploying";
}

function configuredUsers(): Array<TraderUser & { password: string }> {
  const raw = process.env.TRADER_USERS || (process.env.NODE_ENV === "production" ? "" : "branden:password,cam:password");

  const users: Array<TraderUser & { password: string }> = raw
    .split(",")
    .map((pair) => pair.trim())
    .filter(Boolean)
    .map((pair) => {
      const [id, password] = pair.split(":");
      const normalizedId = id.trim().toLowerCase();

      return {
        id: normalizedId,
        name: normalizedId.charAt(0).toUpperCase() + normalizedId.slice(1),
        password: password?.trim() || ""
      };
    });

  if (!users.some((user) => user.id === "tim")) {
    users.push({
      id: "tim",
      name: "Tim",
      password: "tim",
      readOnly: true,
      journalOwnerId: "branden"
    });
  }

  return users;
}

function sign(value: string) {
  return crypto.createHmac("sha256", getSecret()).update(value).digest("hex");
}

export function getUsers() {
  return configuredUsers().map(({ password: _password, ...user }) => user);
}

export function authenticate(userId: string, password: string) {
  const normalizedId = userId.trim().toLowerCase();
  const user = configuredUsers().find((candidate) => candidate.id === normalizedId);

  if (!user || user.password !== password) {
    return null;
  }

  return { id: user.id, name: user.name, readOnly: user.readOnly, journalOwnerId: user.journalOwnerId };
}

export async function setSession(user: TraderUser) {
  const value = Buffer.from(JSON.stringify(user)).toString("base64url");
  const cookieStore = await cookies();

  cookieStore.set(COOKIE_NAME, `${value}.${sign(value)}`, {
    httpOnly: true,
    sameSite: "lax",
    secure: process.env.NODE_ENV === "production",
    path: "/",
    maxAge: 60 * 60 * 24 * 60
  });
}

export async function clearSession() {
  const cookieStore = await cookies();
  cookieStore.delete(COOKIE_NAME);
}

export async function getSessionUser() {
  const cookieStore = await cookies();
  const token = cookieStore.get(COOKIE_NAME)?.value;

  if (!token) {
    return null;
  }

  const [value, signature] = token.split(".");

  if (!value || !signature || sign(value) !== signature) {
    return null;
  }

  try {
    const parsed = JSON.parse(Buffer.from(value, "base64url").toString("utf8")) as TraderUser;
    const user = getUsers().find((candidate) => candidate.id === parsed.id);
    return user || null;
  } catch {
    return null;
  }
}
