import assert from "node:assert/strict";
import test from "node:test";
import { versionSetupTemplates } from "../lib/setup-versioning";
import { tradeChecklistScore } from "../lib/trade-review";
import { trade } from "./fixtures/trade-review-export";
import type { SetupChecklistTemplate } from "../lib/types";

function setup(): SetupChecklistTemplate {
  const criteria = [
    { id: "a", criteria: "Breakout", points: 2, inputType: "boolean" as const },
    { id: "b", criteria: "Volume", points: 3, inputType: "boolean" as const }
  ];
  return { id: "setup", setupName: "Breakout", description: "", criteria,
    groups: [{ id: "group", name: "Entry", criteria }],
    gradeBands: [{ id: "a", label: "A", minScore: 5, maxScore: null }, { id: "b", label: "B", minScore: 0, maxScore: 4 }] };
}

test("removing criteria creates v2 while historical trades retain their score and grade", () => {
  const original = setup();
  const updated = structuredClone(original);
  updated.groups[0].criteria.pop();
  updated.criteria = updated.groups[0].criteria;
  updated.gradeBands = [{ id: "a", label: "A", minScore: 2, maxScore: null }];
  const historical = { ...trade(), setupTags: [original.setupName], manualGrade: "", checklistItems: original.criteria.map((item) => ({ ...item, met: true })) };
  const before = tradeChecklistScore(historical, [original]);
  const versions = versionSetupTemplates([original], [updated]);
  assert.equal(versions.length, 2);
  assert.deepEqual(tradeChecklistScore(historical, versions), before);
  assert.equal(before.earned, 5);
  assert.equal(versions[0].archived, true);
  assert.deepEqual(versions[0].criteria, original.criteria);
  const latest = versions.find((item) => !item.archived)!;
  assert.equal(latest.setupName, "Breakout v2");
  assert.equal(latest.version, 2);
  assert.equal(latest.criteria.length, 1);
  assert.equal(tradeChecklistScore({ ...historical, setupTags: [latest.setupName], checklistItems: latest.criteria.map((item) => ({ ...item, met: true })) }, versions).earned, 2);
});

test("no-op and knowledge saves do not version; further grading edits create v3", () => {
  const first = versionSetupTemplates([], [setup()]);
  assert.deepEqual(versionSetupTemplates(first, first), first);
  const second = versionSetupTemplates(first, [{ ...first[0], gradeBands: [{ id: "new", label: "A", minScore: 1, maxScore: null }] }]);
  const latest = second.find((item) => !item.archived)!;
  const knowledge = versionSetupTemplates(second, [{ ...latest, description: "More guidance" }]);
  assert.equal(knowledge.length, 2);
  const third = versionSetupTemplates(knowledge, [{ ...knowledge[1], gradeBands: [] }]);
  assert.equal(third[2].setupName, "Breakout v3");
  assert.equal(third.filter((item) => !item.archived).length, 1);
});

test("renaming and removal preserve history; stale edits and reused names are rejected", () => {
  const original = setup();
  const renamed = versionSetupTemplates([original], [{ ...original, setupName: "Momentum" }]);
  assert.equal(renamed[0].setupName, "Breakout");
  assert.equal(renamed[1].setupName, "Momentum v2");
  assert.throws(() => versionSetupTemplates(renamed, [original]), /archived/);
  assert.throws(() => versionSetupTemplates(renamed, [{ ...original, id: "new" }]), /already in use/);
  const removed = versionSetupTemplates(renamed, []);
  assert(removed.every((item) => item.archived));
  assert.deepEqual(removed[0].gradeBands, original.gradeBands);
});

test("local save/load preserves versions and never rewrites trade records", async () => {
  const { mkdtemp, readFile, writeFile, mkdir, rm } = await import("node:fs/promises");
  const { tmpdir } = await import("node:os");
  const { join } = await import("node:path");
  const { execFileSync } = await import("node:child_process");
  const directory = await mkdtemp(join(tmpdir(), "setup-versioning-"));
  const loader = join(process.cwd(), "node_modules/tsx/dist/loader.mjs");
  const storePath = join(process.cwd(), "lib/store.ts");
  try {
    await mkdir(join(directory, "data"));
    const tradeData = JSON.stringify([trade()]);
    await writeFile(join(directory, "data/trade-logs.json"), tradeData);
    const source = `
      const assert = require('node:assert/strict');
      const {saveSetupChecklistTemplates, getSetupChecklistTemplates} = require(${JSON.stringify(storePath)});
      (async () => {
        const original = ${JSON.stringify(setup())};
        const first = await saveSetupChecklistTemplates([original]);
        assert.equal(first[0].version, 1);
        const draft = structuredClone(first[0]);
        draft.groups[0].criteria.pop();
        draft.criteria = draft.groups[0].criteria;
        await saveSetupChecklistTemplates([draft]);
        const loaded = await getSetupChecklistTemplates();
        assert.equal(loaded.length, 2);
        assert.equal(loaded[0].archived, true);
        assert.equal(loaded[0].criteria.length, 2);
        assert.equal(loaded[1].setupName, 'Breakout v2');
        const unchanged = await saveSetupChecklistTemplates(loaded.filter(t => !t.archived));
        assert.equal(unchanged.length, 2);
      })().catch(e => {console.error(e); process.exitCode = 1;});
    `;
    execFileSync(process.execPath, ["--import", loader, "-e", source], {
      cwd: directory,
      env: { ...process.env, DATABASE_URL: "", NODE_ENV: "test" },
      stdio: "pipe"
    });
    assert.equal(await readFile(join(directory, "data/trade-logs.json"), "utf8"), tradeData);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});
