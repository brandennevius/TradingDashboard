import assert from "node:assert/strict";
import { randomUUID } from "node:crypto";
import test from "node:test";
import { authenticate, getUsers } from "../lib/auth";

function withUsers(mentorPassword: string | undefined, run: () => void) {
  const originalUsers = process.env.TRADER_USERS;
  const originalMentorPassword = process.env.CODEX_JOURNAL_PASSWORD;
  const traderPassword = randomUUID();
  process.env.TRADER_USERS = `branden:${traderPassword},cam:${traderPassword}`;
  if (mentorPassword === undefined) delete process.env.CODEX_JOURNAL_PASSWORD;
  else process.env.CODEX_JOURNAL_PASSWORD = mentorPassword;
  try {
    run();
    assert.equal(authenticate("branden", traderPassword)?.id, "branden");
    assert.equal(authenticate("cam", traderPassword)?.id, "cam");
    assert.equal(authenticate("branden", traderPassword)?.readOnly, undefined);
  } finally {
    if (originalUsers === undefined) delete process.env.TRADER_USERS;
    else process.env.TRADER_USERS = originalUsers;
    if (originalMentorPassword === undefined) delete process.env.CODEX_JOURNAL_PASSWORD;
    else process.env.CODEX_JOURNAL_PASSWORD = originalMentorPassword;
  }
}

test("mentor login is absent unless explicitly configured", () => {
  for (const value of [undefined, ""]) {
    withUsers(value, () => {
      assert.equal(getUsers().some((user) => user.id === "codex"), false);
      assert.equal(authenticate("codex", ""), null);
    });
  }
});

test("mentor authenticates as a read-only delegate without exposing the credential", () => {
  const credential = randomUUID();
  withUsers(credential, () => {
    assert.deepEqual(authenticate(" CODEX ", credential), {
      id: "codex", name: "Codex", readOnly: true, journalOwnerId: "branden"
    });
    assert.equal(authenticate("codex", randomUUID()), null);
    assert.equal(JSON.stringify(getUsers()).includes(credential), false);
    assert.equal(JSON.stringify(authenticate("codex", credential)).includes(credential), false);
  });
});

test("dedicated mentor settings cannot inherit write permissions from a duplicate identity", () => {
  const credential = randomUUID();
  withUsers(credential, () => {
    process.env.TRADER_USERS += `,codex:${randomUUID()}`;
    const matches = getUsers().filter((user) => user.id === "codex");
    assert.equal(matches.length, 1);
    assert.equal(matches[0].readOnly, true);
    assert.equal(authenticate("codex", credential)?.journalOwnerId, "branden");
  });
});

test("removing the dedicated secret revokes mentor authentication", () => {
  const credential = randomUUID();
  withUsers(credential, () => {
    assert.ok(authenticate("codex", credential));
    delete process.env.CODEX_JOURNAL_PASSWORD;
    assert.equal(authenticate("codex", credential), null);
    assert.equal(getUsers().some((user) => user.id === "codex"), false);
  });
});
