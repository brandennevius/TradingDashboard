import { NextResponse } from "next/server";
import { getSessionUser } from "@/lib/auth";
import { getSetupChecklistTemplates, saveSetupChecklistTemplates } from "@/lib/store";
import type { ChecklistGradeBand, ChecklistInputType, SetupChecklistGroup, SetupChecklistTemplate, SetupTemplateCriterion } from "@/lib/types";

function numberValue(value: unknown) {
  const number = Number(value);
  return Number.isFinite(number) ? number : 0;
}

function normalizeInputType(value: unknown): ChecklistInputType {
  return String(value || "").toLowerCase() === "points" ? "points" : "boolean";
}

function normalizeCriteria(value: unknown): SetupTemplateCriterion[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value
    .map((item, index) => {
      if (!item || typeof item !== "object") {
        return null;
      }

      const rawItem = item as Record<string, unknown>;
      const criteria = String(rawItem.criteria || "").trim();
      const points = numberValue(rawItem.points);

      if (!criteria || points <= 0) {
        return null;
      }

      return {
        id: String(rawItem.id || `criteria-${index}-${Date.now()}`),
        criteria,
        points,
        inputType: normalizeInputType(rawItem.inputType),
        importTagKey: String(rawItem.importTagKey || "").trim(),
        importTagValue: String(rawItem.importTagValue || "").trim()
      };
    })
    .filter(Boolean) as SetupTemplateCriterion[];
}

function normalizeGroups(value: unknown, legacyCriteria: unknown): SetupChecklistGroup[] {
  if (Array.isArray(value) && value.length) {
    return value
      .map((group, index) => {
        if (!group || typeof group !== "object") {
          return null;
        }

        const rawGroup = group as Record<string, unknown>;
        const criteria = normalizeCriteria(rawGroup.criteria);
        const name = String(rawGroup.name || "").trim();

        if (!name && !criteria.length) {
          return null;
        }

        return {
          id: String(rawGroup.id || `group-${index}-${Date.now()}`),
          name: name || `Group ${index + 1}`,
          criteria
        } satisfies SetupChecklistGroup;
      })
      .filter(Boolean) as SetupChecklistGroup[];
  }

  const criteria = normalizeCriteria(legacyCriteria);
  return criteria.length
    ? [
        {
          id: `group-default-${Date.now()}`,
          name: "Checklist",
          criteria
        }
      ]
    : [];
}

const defaultGradeBands: ChecklistGradeBand[] = [
  { id: "grade-a-plus", label: "A+", minScore: 10, maxScore: null },
  { id: "grade-a", label: "A", minScore: 8, maxScore: 9 },
  { id: "grade-b-plus", label: "B+", minScore: 7, maxScore: 7 },
  { id: "grade-b", label: "B", minScore: 6, maxScore: 6 },
  { id: "grade-c", label: "C", minScore: 0, maxScore: 5 }
];

function normalizeGradeBands(value: unknown): ChecklistGradeBand[] {
  if (!Array.isArray(value)) {
    return defaultGradeBands;
  }

  const bands = value
    .map((band, index) => {
      if (!band || typeof band !== "object") {
        return null;
      }

      const rawBand = band as Record<string, unknown>;
      const label = String(rawBand.label || "").trim();
      const minScore = numberValue(rawBand.minScore);
      const maxScore = rawBand.maxScore === null || rawBand.maxScore === "" ? null : numberValue(rawBand.maxScore);

      if (!label) {
        return null;
      }

      return {
        id: String(rawBand.id || `grade-${index}-${Date.now()}`),
        label,
        minScore,
        maxScore
      };
    })
    .filter(Boolean) as ChecklistGradeBand[];

  return bands.length ? bands.sort((a, b) => b.minScore - a.minScore) : defaultGradeBands;
}

function normalizeTemplates(value: unknown): SetupChecklistTemplate[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value
    .map((template, index) => {
      if (!template || typeof template !== "object") {
        return null;
      }

      const rawTemplate = template as Record<string, unknown>;
      const setupName = String(rawTemplate.setupName || "").trim();

      if (!setupName) {
        return null;
      }

      const groups = normalizeGroups(rawTemplate.groups, rawTemplate.criteria);

      return {
        id: String(rawTemplate.id || `setup-template-${index}-${Date.now()}`),
        setupName,
        description: String(rawTemplate.description || "").trim(),
        knowledgeSources: normalizeKnowledgeSources(rawTemplate.knowledgeSources),
        strategyExamples: normalizeStrategyExamples(rawTemplate.strategyExamples),
        gradeBands: normalizeGradeBands(rawTemplate.gradeBands),
        criteria: groups.flatMap((group) => group.criteria),
        groups
      };
    })
    .filter(Boolean) as SetupChecklistTemplate[];
}

function normalizeKnowledgeSources(value: unknown) {
  if (!Array.isArray(value)) {
    return [];
  }

  return value
    .map((source, index) => {
      if (!source || typeof source !== "object") {
        return null;
      }

      const rawSource = source as Record<string, unknown>;
      const title = String(rawSource.title || "").trim();
      const content = String(rawSource.content || "").trim();
      const url = String(rawSource.url || "").trim();
      const sourceType = String(rawSource.sourceType || "notes").toLowerCase();

      if (!title && !content && !url) {
        return null;
      }

      return {
        id: String(rawSource.id || `setup-knowledge-${index}-${Date.now()}`),
        title: title || "Strategy knowledge",
        sourceType: sourceType === "resource" || sourceType === "document" ? sourceType : "notes",
        url,
        content,
        chunks: normalizeKnowledgeChunks(rawSource.chunks),
        active: rawSource.active === false ? false : true,
        createdAt: String(rawSource.createdAt || new Date().toISOString()),
        updatedAt: new Date().toISOString()
      };
    })
    .filter(Boolean);
}

function normalizeKnowledgeChunks(value: unknown) {
  if (!Array.isArray(value)) {
    return [];
  }

  return value
    .map((chunk, index) => {
      if (!chunk || typeof chunk !== "object") {
        return null;
      }

      const rawChunk = chunk as Record<string, unknown>;
      const content = String(rawChunk.content || "").trim();

      if (!content) {
        return null;
      }

      return {
        id: String(rawChunk.id || `setup-knowledge-chunk-${index}-${Date.now()}`),
        title: String(rawChunk.title || `Section ${index + 1}`).trim() || `Section ${index + 1}`,
        content,
        order: Number.isFinite(Number(rawChunk.order)) ? Number(rawChunk.order) : index
      };
    })
    .filter((chunk): chunk is { id: string; title: string; content: string; order: number } => Boolean(chunk))
    .sort((a, b) => a.order - b.order);
}

function normalizeStrategyExamples(value: unknown) {
  if (!Array.isArray(value)) {
    return [];
  }

  return value
    .map((example, index) => {
      if (!example || typeof example !== "object") {
        return null;
      }

      const rawExample = example as Record<string, unknown>;
      const symbol = String(rawExample.symbol || "").trim().toUpperCase();
      const setupType = String(rawExample.setupType || "").trim();
      const notes = String(rawExample.notes || "").trim();
      const sourceUrl = String(rawExample.sourceUrl || "").trim();
      const source = String(rawExample.source || "").trim();
      const screenshots = Array.isArray(rawExample.screenshots)
        ? rawExample.screenshots.map((screenshot) => String(screenshot || "").trim()).filter(Boolean)
        : [];
      const quality = String(rawExample.quality || "good").toLowerCase();

      if (!symbol && !setupType && !notes && !sourceUrl && !screenshots.length) {
        return null;
      }

      return {
        id: String(rawExample.id || `setup-example-${index}-${Date.now()}`),
        symbol,
        setupType,
        quality: quality === "ideal" || quality === "failed" || quality === "bad" || quality === "cautionary" ? quality : "good",
        outcome: String(rawExample.outcome || "").trim(),
        source,
        sourceUrl,
        notes,
        screenshots,
        active: rawExample.active === false ? false : true,
        createdAt: String(rawExample.createdAt || new Date().toISOString()),
        updatedAt: new Date().toISOString()
      };
    })
    .filter(Boolean);
}

export async function GET() {
  const user = await getSessionUser();

  if (!user) {
    return NextResponse.json({ error: "Unauthorized." }, { status: 401 });
  }

  try {
    const setupChecklists = await getSetupChecklistTemplates();
    return NextResponse.json({ user, setupChecklists });
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Could not load setup checklists." },
      { status: 500 }
    );
  }
}

export async function PUT(request: Request) {
  const user = await getSessionUser();

  if (!user) {
    return NextResponse.json({ error: "Unauthorized." }, { status: 401 });
  }

  if (user.readOnly) {
    return NextResponse.json({ error: "This account is read-only." }, { status: 403 });
  }

  const body = await request.json();
  const setupChecklists = normalizeTemplates(body.setupChecklists);

  if (!setupChecklists.length) {
    return NextResponse.json(
      { error: "Add at least one setup with a name and valid criteria before saving." },
      { status: 400 }
    );
  }

  if (setupChecklists.some((template) => !template.groups.length || !template.criteria.length)) {
    return NextResponse.json(
      { error: "Every setup checklist needs at least one valid group with one valid criteria row." },
      { status: 400 }
    );
  }

  try {
    const savedSetupChecklists = await saveSetupChecklistTemplates(setupChecklists);
    return NextResponse.json({ setupChecklists: savedSetupChecklists });
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Could not save setup checklists." },
      { status: 500 }
    );
  }
}
