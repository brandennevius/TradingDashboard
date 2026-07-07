"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import type { DragEvent } from "react";
import type {
  ChecklistGradeBand,
  ChecklistInputType,
  SetupChecklistGroup,
  SetupChecklistTemplate,
  SetupStrategyExampleQuality,
  SetupTemplateCriterion,
  TraderUser
} from "@/lib/types";

type DraggedCriterion = {
  templateId: string;
  groupId: string;
  criteriaId: string;
};

type SetupKnowledgeSourceDraft = NonNullable<SetupChecklistTemplate["knowledgeSources"]>[number];
type SetupStrategyExampleDraft = NonNullable<SetupChecklistTemplate["strategyExamples"]>[number];

const defaultChecklistGradeBands: ChecklistGradeBand[] = [
  { id: "grade-a-plus", label: "A+", minScore: 10, maxScore: null },
  { id: "grade-a", label: "A", minScore: 8, maxScore: 9 },
  { id: "grade-b-plus", label: "B+", minScore: 7, maxScore: 7 },
  { id: "grade-b", label: "B", minScore: 6, maxScore: 6 },
  { id: "grade-c", label: "C", minScore: 0, maxScore: 5 }
];

function numberValue(value: unknown) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : 0;
}

function uniqueId(prefix: string) {
  return `${prefix}-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function compactWhitespace(value: string) {
  return value.replace(/\s+/g, " ").trim();
}

function wordCount(value: string) {
  return compactWhitespace(value).split(" ").filter(Boolean).length;
}

function sourceTypeLabel(sourceType: SetupKnowledgeSourceDraft["sourceType"]) {
  if (sourceType === "document") return "Document";
  if (sourceType === "resource") return "Resource";
  return "Notes";
}

function shortDate(value: string) {
  if (!value) return "Not saved";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? "Not saved" : date.toLocaleDateString();
}

function sourcePreview(source: SetupKnowledgeSourceDraft) {
  const text = compactWhitespace(source.content || source.url || "");
  if (!text) return "No context added yet.";
  return text.length > 220 ? `${text.slice(0, 220)}...` : text;
}

function examplePreview(example: SetupStrategyExampleDraft) {
  const parts = [
    example.setupType,
    example.outcome,
    compactWhitespace(example.notes || "")
  ].filter(Boolean);
  const text = parts.join(" · ");
  if (!text) return "No example notes added yet.";
  return text.length > 220 ? `${text.slice(0, 220)}...` : text;
}

function exampleQualityLabel(quality: SetupStrategyExampleQuality) {
  const labels: Record<SetupStrategyExampleQuality, string> = {
    ideal: "Ideal",
    good: "Good",
    failed: "Failed",
    bad: "Bad",
    cautionary: "Cautionary"
  };
  return labels[quality] || "Good";
}

function chunkKnowledgeContent(source: SetupKnowledgeSourceDraft) {
  const paragraphs = (source.content || "")
    .split(/\n{2,}/)
    .map((paragraph) => paragraph.trim())
    .filter(Boolean);
  const chunks: NonNullable<SetupKnowledgeSourceDraft["chunks"]> = [];
  let current = "";

  paragraphs.forEach((paragraph) => {
    const next = current ? `${current}\n\n${paragraph}` : paragraph;
    if (next.length > 2200 && current) {
      chunks.push({
        id: uniqueId("setup-knowledge-chunk"),
        title: `${source.title || "Strategy knowledge"} ${chunks.length + 1}`,
        content: current,
        order: chunks.length
      });
      current = paragraph;
      return;
    }
    current = next;
  });

  if (current.trim()) {
    chunks.push({
      id: uniqueId("setup-knowledge-chunk"),
      title: `${source.title || "Strategy knowledge"} ${chunks.length + 1}`,
      content: current,
      order: chunks.length
    });
  }

  return chunks;
}

function readImageFile(file: File) {
  return new Promise<string>((resolve) => {
    const reader = new FileReader();
    reader.onload = () => {
      const originalDataUrl = String(reader.result || "");
      if (!originalDataUrl) {
        resolve("");
        return;
      }

      const image = new Image();
      image.onload = () => {
        const maxDimension = 1600;
        const scale = Math.min(1, maxDimension / Math.max(image.width, image.height));
        const width = Math.max(1, Math.round(image.width * scale));
        const height = Math.max(1, Math.round(image.height * scale));
        const canvas = document.createElement("canvas");
        canvas.width = width;
        canvas.height = height;
        const context = canvas.getContext("2d");
        if (!context) {
          resolve(originalDataUrl);
          return;
        }
        context.fillStyle = "#ffffff";
        context.fillRect(0, 0, width, height);
        context.drawImage(image, 0, 0, width, height);
        resolve(canvas.toDataURL("image/jpeg", 0.76));
      };
      image.onerror = () => resolve(originalDataUrl);
      image.src = originalDataUrl;
    };
    reader.onerror = () => resolve("");
    reader.readAsDataURL(file);
  });
}

function readTextFile(file: File) {
  return new Promise<string>((resolve) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result || ""));
    reader.onerror = () => resolve("");
    reader.readAsText(file);
  });
}

async function uploadStrategyExampleScreenshot(exampleId: string, dataUrl: string, fileName: string) {
  const response = await fetch(dataUrl);
  const blob = await response.blob();
  const formData = new FormData();
  formData.set("file", new File([blob], fileName, { type: blob.type || "image/jpeg" }));
  formData.set("exampleId", exampleId);
  const uploadResponse = await fetch("/api/settings/setup-checklists/screenshots", {
    method: "POST",
    body: formData
  });
  const data = await uploadResponse.json().catch(() => ({}));
  if (!uploadResponse.ok) {
    throw new Error(data.error || "Could not upload strategy example screenshot.");
  }
  return String(data.url || "");
}

function normalizeFileName(value: string) {
  return value.split(/[\\/]/).pop()?.trim().toLowerCase() || "";
}

function parseCsv(text: string) {
  const rows: string[][] = [];
  let row: string[] = [];
  let cell = "";
  let quoted = false;

  for (let index = 0; index < text.length; index += 1) {
    const char = text[index];
    const next = text[index + 1];

    if (char === '"' && quoted && next === '"') {
      cell += '"';
      index += 1;
      continue;
    }

    if (char === '"') {
      quoted = !quoted;
      continue;
    }

    if (char === "," && !quoted) {
      row.push(cell);
      cell = "";
      continue;
    }

    if ((char === "\n" || char === "\r") && !quoted) {
      if (char === "\r" && next === "\n") index += 1;
      row.push(cell);
      if (row.some((value) => value.trim())) rows.push(row);
      row = [];
      cell = "";
      continue;
    }

    cell += char;
  }

  row.push(cell);
  if (row.some((value) => value.trim())) rows.push(row);
  return rows;
}

function csvRecords(text: string) {
  const rows = parseCsv(text);
  const headers = rows[0]?.map((header) => header.trim()) || [];
  return rows.slice(1).map((row) =>
    Object.fromEntries(headers.map((header, index) => [header, String(row[index] || "").trim()]))
  );
}

function newSetupTemplateCriterion(inputType: ChecklistInputType = "boolean"): SetupTemplateCriterion {
  return {
    id: uniqueId("criteria"),
    criteria: "",
    points: 1,
    inputType,
    importTagKey: "",
    importTagValue: ""
  };
}

function newSetupTemplateGroup(name = "New Group"): SetupChecklistGroup {
  return {
    id: uniqueId("group"),
    name,
    criteria: [newSetupTemplateCriterion()]
  };
}

function newSetupTemplate(): SetupChecklistTemplate {
  return {
    id: uniqueId("setup-template"),
    setupName: "",
    description: "",
    knowledgeSources: [],
    strategyExamples: [],
    gradeBands: defaultChecklistGradeBands,
    criteria: [newSetupTemplateCriterion()],
    groups: [newSetupTemplateGroup("Checklist")]
  };
}

function otcPresetTemplate(): SetupChecklistTemplate {
  const make = (criteria: string, points: number, importTagKey: string, importTagValue: string): SetupTemplateCriterion => ({
    id: uniqueId("criteria"),
    criteria,
    points,
    inputType: "boolean",
    importTagKey,
    importTagValue
  });
  const groups = [
    {
      id: uniqueId("group-tech"),
      name: "Technicals",
      criteria: [
        make("Breakout setup confirmed", 2, "Breakout", "Yes"),
        make("Primary trend aligned", 2, "Trend", "Trend"),
        make("Fresh setup / not too extended", 1, "Freshness", "Yes")
      ]
    },
    {
      id: uniqueId("group-fund"),
      name: "Fundamentals",
      criteria: [
        make("Coverage in place", 1, "Coverage", "Yes"),
        make("COT supportive", 1, "COT", "Yes"),
        make("Valuation acceptable", 1, "Valuation", "Yes"),
        make("Seasonality supportive", 1, "Seasonality", "Yes"),
        make("Earnings catalyst > 25%", 2, "Earnings", "Yes")
      ]
    }
  ] satisfies SetupChecklistGroup[];

  return {
    id: uniqueId("setup-template"),
    setupName: "OTC",
    description: "Excel-backed OTC strategy. Imported tag fields can auto-check these rows from the trade sheet.",
    knowledgeSources: [
      {
        id: uniqueId("setup-knowledge"),
        title: "OTC strategy notes",
        sourceType: "notes",
        url: "",
        content: "Paste Branden's OTC trade plan notes here so AI reviews can judge OTC trades against the full strategy context.",
        active: true,
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString()
      }
    ],
    gradeBands: defaultChecklistGradeBands,
    criteria: groups.flatMap((group) => group.criteria),
    groups
  };
}

function canslimPresetTemplate(): SetupChecklistTemplate {
  const make = (criteria: string, points: number, inputType: ChecklistInputType = "boolean"): SetupTemplateCriterion => ({
    id: uniqueId("criteria"),
    criteria,
    points,
    inputType,
    importTagKey: "",
    importTagValue: ""
  });
  const groups = [
    {
      id: uniqueId("group-fund"),
      name: "Fundamentals",
      criteria: [
        make("Current EPS growth >= 25%", 2),
        make("Sales growth >= 20-25%", 2),
        make("Annual EPS growth strong", 1),
        make("Institutional sponsorship improving", 1)
      ]
    },
    {
      id: uniqueId("group-lead"),
      name: "Leadership",
      criteria: [
        make("Leader in leading industry group", 2),
        make("Relative strength near highs", 2),
        make("Price within range of 52-week highs", 1)
      ]
    },
    {
      id: uniqueId("group-tech"),
      name: "Technicals",
      criteria: [
        make("Proper base formed", 2, "points"),
        make("Valid pivot / buy point", 2, "points"),
        make("Breakout volume strong", 2, "points"),
        make("Not extended from pivot", 1)
      ]
    },
    {
      id: uniqueId("group-market"),
      name: "Market",
      criteria: [make("General market in confirmed uptrend", 2)]
    },
    {
      id: uniqueId("group-manage"),
      name: "Trade Management",
      criteria: [make("Stop moved to breakeven at 1:1", 1), make("Exit plan followed", 2, "points")]
    }
  ] satisfies SetupChecklistGroup[];

  return {
    id: uniqueId("setup-template"),
    setupName: "CANSLIM",
    description: "Website-only growth breakout checklist. Use this for CANSLIM / Minervini / Roppel-style leadership setups.",
    knowledgeSources: [
      {
        id: uniqueId("setup-knowledge"),
        title: "CANSLIM strategy notes",
        sourceType: "notes",
        url: "",
        content: "Add your own CANSLIM operating rules here: earnings/sales requirements, leadership requirements, base/pivot rules, market condition rules, buy discipline, sell rules, and risk management notes.",
        active: true,
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString()
      }
    ],
    gradeBands: defaultChecklistGradeBands,
    criteria: groups.flatMap((group) => group.criteria),
    groups
  };
}

function syncTemplate(template: SetupChecklistTemplate, groups = template.groups || []) {
  return { ...template, groups, criteria: groups.flatMap((group) => group.criteria) };
}

export default function BrandenSetupBuilderPage() {
  const [user, setUser] = useState<TraderUser | null>(null);
  const [setupTemplateDrafts, setSetupTemplateDrafts] = useState<SetupChecklistTemplate[]>([]);
  const [status, setStatus] = useState("");
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(true);
  const [draggedSetupCriterion, setDraggedSetupCriterion] = useState<DraggedCriterion | null>(null);
  const [dragOverSetupCriterionId, setDragOverSetupCriterionId] = useState("");
  const [importingKnowledgeTemplateId, setImportingKnowledgeTemplateId] = useState("");
  const [expandedKnowledgeSourceIds, setExpandedKnowledgeSourceIds] = useState<string[]>([]);
  const [expandedStrategyExampleIds, setExpandedStrategyExampleIds] = useState<string[]>([]);
  const [expandedSetupTemplateIds, setExpandedSetupTemplateIds] = useState<string[]>([]);

  useEffect(() => {
    let cancelled = false;

    async function loadPageData() {
      setIsLoading(true);
      setError("");
      const setupResponse = await fetch("/api/settings/setup-checklists", { cache: "no-store" });
      const setupData = await setupResponse.json().catch(() => ({}));

      if (cancelled) return;

      if (!setupResponse.ok || !setupData.user) {
        setError("Sign in to view Setup Builder.");
        setIsLoading(false);
        return;
      }

      setUser(setupData.user || null);
      setSetupTemplateDrafts(Array.isArray(setupData.setupChecklists) ? setupData.setupChecklists : []);
      setIsLoading(false);
    }

    loadPageData().catch((loadError) => {
      if (!cancelled) {
        setError(loadError instanceof Error ? loadError.message : "Could not load setup builder.");
        setIsLoading(false);
      }
    });

    return () => {
      cancelled = true;
    };
  }, []);

  const canEdit = user?.id === "branden" && !user.readOnly;

  function addSetupTemplate() {
    setStatus("");
    const template = newSetupTemplate();
    setSetupTemplateDrafts((current) => [...current, template]);
    setExpandedSetupTemplateIds((current) => Array.from(new Set([...current, template.id])));
  }

  function addPresetSetupTemplate(preset: "OTC" | "CANSLIM") {
    setStatus("");
    const template = preset === "OTC" ? otcPresetTemplate() : canslimPresetTemplate();
    setSetupTemplateDrafts((current) => [...current, template]);
    setExpandedSetupTemplateIds((current) => Array.from(new Set([...current, template.id])));
  }

  function updateSetupTemplate(id: string, updates: Partial<SetupChecklistTemplate>) {
    setSetupTemplateDrafts((current) =>
      current.map((template) =>
        template.id === id ? syncTemplate({ ...template, ...updates }, updates.groups || template.groups || []) : template
      )
    );
  }

  function removeSetupTemplate(id: string) {
    setSetupTemplateDrafts((current) => current.filter((template) => template.id !== id));
    setExpandedSetupTemplateIds((current) => current.filter((templateId) => templateId !== id));
  }

  function toggleSetupTemplate(id: string) {
    setExpandedSetupTemplateIds((current) =>
      current.includes(id) ? current.filter((templateId) => templateId !== id) : [...current, id]
    );
  }

  function addSetupKnowledgeSource(templateId: string) {
    const now = new Date().toISOString();
    const sourceId = uniqueId("setup-knowledge");
    setSetupTemplateDrafts((current) =>
      current.map((template) =>
        template.id === templateId
          ? {
              ...template,
              knowledgeSources: [
                ...(template.knowledgeSources || []),
                {
                  id: sourceId,
                  title: "",
                  sourceType: "notes",
                  url: "",
                  content: "",
                  active: true,
                  createdAt: now,
                  updatedAt: now
                }
              ]
            }
          : template
      )
    );
    setExpandedKnowledgeSourceIds((current) => Array.from(new Set([...current, sourceId])));
  }

  function updateSetupKnowledgeSource(
    templateId: string,
    sourceId: string,
    updates: Partial<NonNullable<SetupChecklistTemplate["knowledgeSources"]>[number]>
  ) {
    setSetupTemplateDrafts((current) =>
      current.map((template) =>
        template.id === templateId
          ? {
              ...template,
              knowledgeSources: (template.knowledgeSources || []).map((source) =>
                source.id === sourceId ? { ...source, ...updates, updatedAt: new Date().toISOString() } : source
              )
            }
          : template
      )
    );
  }

  function removeSetupKnowledgeSource(templateId: string, sourceId: string) {
    setSetupTemplateDrafts((current) =>
      current.map((template) =>
        template.id === templateId
          ? { ...template, knowledgeSources: (template.knowledgeSources || []).filter((source) => source.id !== sourceId) }
          : template
      )
    );
    setExpandedKnowledgeSourceIds((current) => current.filter((id) => id !== sourceId));
  }

  function toggleSetupKnowledgeSource(sourceId: string) {
    setExpandedKnowledgeSourceIds((current) =>
      current.includes(sourceId) ? current.filter((id) => id !== sourceId) : [...current, sourceId]
    );
  }

  function reindexSetupKnowledgeSource(templateId: string, sourceId: string) {
    setSetupTemplateDrafts((current) =>
      current.map((template) =>
        template.id === templateId
          ? {
              ...template,
              knowledgeSources: (template.knowledgeSources || []).map((source) =>
                source.id === sourceId
                  ? {
                      ...source,
                      chunks: chunkKnowledgeContent(source),
                      updatedAt: new Date().toISOString()
                    }
                  : source
              )
            }
          : template
      )
    );
    setStatus("Knowledge source re-indexed. Save setup checklists to persist it.");
  }

  function addSetupStrategyExample(templateId: string) {
    const now = new Date().toISOString();
    const exampleId = uniqueId("setup-example");
    setSetupTemplateDrafts((current) =>
      current.map((template) =>
        template.id === templateId
          ? {
              ...template,
              strategyExamples: [
                ...(template.strategyExamples || []),
                {
                  id: exampleId,
                  symbol: "",
                  setupType: "",
                  quality: "good",
                  outcome: "",
                  source: "",
                  sourceUrl: "",
                  notes: "",
                  screenshots: [],
                  active: true,
                  createdAt: now,
                  updatedAt: now
                }
              ]
            }
          : template
      )
    );
    setExpandedStrategyExampleIds((current) => Array.from(new Set([...current, exampleId])));
  }

  function updateSetupStrategyExample(templateId: string, exampleId: string, updates: Partial<SetupStrategyExampleDraft>) {
    setSetupTemplateDrafts((current) =>
      current.map((template) =>
        template.id === templateId
          ? {
              ...template,
              strategyExamples: (template.strategyExamples || []).map((example) =>
                example.id === exampleId ? { ...example, ...updates, updatedAt: new Date().toISOString() } : example
              )
            }
          : template
      )
    );
  }

  function removeSetupStrategyExample(templateId: string, exampleId: string) {
    setSetupTemplateDrafts((current) =>
      current.map((template) =>
        template.id === templateId
          ? { ...template, strategyExamples: (template.strategyExamples || []).filter((example) => example.id !== exampleId) }
          : template
      )
    );
    setExpandedStrategyExampleIds((current) => current.filter((id) => id !== exampleId));
  }

  function toggleSetupStrategyExample(exampleId: string) {
    setExpandedStrategyExampleIds((current) =>
      current.includes(exampleId) ? current.filter((id) => id !== exampleId) : [...current, exampleId]
    );
  }

  async function addSetupStrategyExampleScreenshots(templateId: string, example: SetupStrategyExampleDraft, files: FileList | null) {
    const selectedFiles = Array.from(files || []);
    if (!selectedFiles.length) return;
    setError("");
    setStatus(`Uploading ${selectedFiles.length} example screenshot${selectedFiles.length === 1 ? "" : "s"}...`);
    try {
      const screenshots = (
        await Promise.all(
          selectedFiles.map(async (file) => {
            const dataUrl = await readImageFile(file);
            return dataUrl ? uploadStrategyExampleScreenshot(example.id, dataUrl, file.name) : "";
          })
        )
      ).filter(Boolean);
      updateSetupStrategyExample(templateId, example.id, { screenshots: [...(example.screenshots || []), ...screenshots] });
      setStatus("Strategy example screenshots uploaded. Save setup checklists to persist them.");
    } catch (uploadError) {
      setError(uploadError instanceof Error ? uploadError.message : "Could not upload strategy example screenshots.");
      setStatus("");
    }
  }

  async function importSetupStrategyExamplesCsv(templateId: string, files: FileList | null) {
    const selectedFiles = Array.from(files || []);
    const csvFile = selectedFiles.find((file) => file.name.toLowerCase().endsWith(".csv") || file.type === "text/csv");
    if (!csvFile) {
      setError("Select the CANSLIM examples CSV. You can include matching PNG/JPG files in the same upload.");
      return;
    }
    const imageFiles = selectedFiles.filter((file) => file.type.startsWith("image/") || /\.(png|jpe?g|webp|gif)$/i.test(file.name));

    setStatus(`Importing ${csvFile.name}${imageFiles.length ? ` with ${imageFiles.length} compressed image${imageFiles.length === 1 ? "" : "s"}` : ""}...`);
    setError("");

    const imageEntries = await Promise.all(
      imageFiles.map(async (file) => ({
        name: normalizeFileName(file.name),
        dataUrl: await readImageFile(file)
      }))
    );
    const imageByName = new Map(imageEntries.filter((entry) => entry.name && entry.dataUrl).map((entry) => [entry.name, entry.dataUrl]));
    const text = await readTextFile(csvFile);
    const now = new Date().toISOString();
    const records = csvRecords(text);
    let matchedImageCount = 0;
    const imported = (await Promise.all(records
      .filter((record) => record.review_status === "labeled_seed")
      .map(async (record) => {
        const exampleId = uniqueId("setup-example");
        const page = record.source_page ? `p. ${record.source_page}` : "";
        const imageNameCandidates = [
          normalizeFileName(record.filename || ""),
          normalizeFileName(record.source_image_path || "")
        ].filter(Boolean);
        const screenshotDataUrl = imageNameCandidates.map((name) => imageByName.get(name)).find(Boolean) || "";
        const screenshot = screenshotDataUrl
          ? await uploadStrategyExampleScreenshot(exampleId, screenshotDataUrl, imageNameCandidates[0] || `${record.ticker || "example"}.jpg`)
          : "";
        if (screenshot) matchedImageCount += 1;
        const notes = [
          record.model_lesson ? `Model lesson: ${record.model_lesson}` : "",
          record.base_quality ? `Base quality: ${record.base_quality}` : "",
          record.volume_notes ? `Volume: ${record.volume_notes}` : "",
          record.relative_strength_notes ? `Relative strength: ${record.relative_strength_notes}` : "",
          record.buy_point_notes ? `Buy point: ${record.buy_point_notes}` : "",
          record.failure_warnings ? `Failure warning: ${record.failure_warnings}` : ""
        ].filter(Boolean).join("\n");
        return {
          id: exampleId,
          symbol: String(record.ticker || "").toUpperCase(),
          setupType: record.setup_type || "CANSLIM model example",
          quality: record.setup_type?.toLowerCase().includes("failed") ? "failed" : "good",
          outcome: record.outcome_note || "",
          source: ["CANSLIM model book", page].filter(Boolean).join(" "),
          sourceUrl: record.filename || record.source_image_path || "",
          notes,
          screenshots: screenshot ? [screenshot] : [],
          active: true,
          createdAt: now,
          updatedAt: now
        } satisfies SetupStrategyExampleDraft;
      })
    )).filter((example) => example.symbol || example.setupType || example.notes);

    if (!imported.length) {
      setError("No labeled_seed examples were found in that CSV.");
      setStatus("");
      return;
    }

    const importResult = {
      added: 0,
      updated: 0,
      alreadyHadImages: 0
    };
    setSetupTemplateDrafts((current) => {
      const nextTemplates = current.map((template) => {
        if (template.id !== templateId) return template;
        const existingExamples = template.strategyExamples || [];
        const importedByKey = new Map(
          imported.map((example) => [[example.symbol, example.setupType, example.source, example.sourceUrl].join("|").toLowerCase(), example])
        );
        const existingKeys = new Set(existingExamples.map((example) => [example.symbol, example.setupType, example.source, example.sourceUrl].join("|").toLowerCase()));
        const updatedExamples = existingExamples.map((example) => {
          const key = [example.symbol, example.setupType, example.source, example.sourceUrl].join("|").toLowerCase();
          const importedMatch = importedByKey.get(key);
          const importedScreenshot = importedMatch?.screenshots?.[0];
          if (!importedScreenshot) {
            return example;
          }
          if ((example.screenshots || []).length) {
            importResult.alreadyHadImages += 1;
            return example;
          }
          importResult.updated += 1;
          return {
            ...example,
            screenshots: [importedScreenshot],
            updatedAt: now
          };
        });
        const newExamples = imported.filter((example) => {
          const key = [example.symbol, example.setupType, example.source, example.sourceUrl].join("|").toLowerCase();
          return !existingKeys.has(key);
        });
        importResult.added += newExamples.length;
        return {
          ...template,
          strategyExamples: [...updatedExamples, ...newExamples]
        };
      });
      setStatus(
        `Imported ${imported.length} CANSLIM model examples; added ${importResult.added}, attached images to ${importResult.updated}, already had images on ${importResult.alreadyHadImages}, matched ${matchedImageCount} image${matchedImageCount === 1 ? "" : "s"}. Save Setup Builder to persist them.`
      );
      return nextTemplates;
    });
  }

  async function importSetupKnowledgeFile(templateId: string, files: FileList | null) {
    const file = files?.[0];
    if (!file) return;

    setStatus(`Importing ${file.name}...`);
    setError("");
    setImportingKnowledgeTemplateId(templateId);

    try {
      const formData = new FormData();
      formData.append("file", file);
      const response = await fetch("/api/settings/setup-knowledge/import", {
        method: "POST",
        body: formData
      });
      const data = await response.json().catch(() => ({}));

      if (!response.ok) {
        throw new Error(data.error || "Could not import strategy document.");
      }

      const now = new Date().toISOString();
      const sourceId = uniqueId("setup-knowledge");
      const nextTemplates = setupTemplateDrafts.map((template) =>
        template.id === templateId
          ? {
              ...template,
              knowledgeSources: [
                ...(template.knowledgeSources || []),
                {
                  id: sourceId,
                  title: String(data.title || file.name),
                  sourceType: "document" as const,
                  url: String(data.url || file.name),
                  content: String(data.content || ""),
                  chunks: Array.isArray(data.chunks) ? data.chunks : [],
                  active: true,
                  createdAt: now,
                  updatedAt: now
                }
              ]
            }
          : template
      );
      setSetupTemplateDrafts(nextTemplates);
      setExpandedKnowledgeSourceIds((current) => Array.from(new Set([...current, sourceId])));
      await persistSetupTemplates(
        nextTemplates,
        data.truncated ? `${file.name} imported, truncated to the first 120k characters, and saved.` : `${file.name} imported and saved.`
      );
    } catch (importError) {
      setError(importError instanceof Error ? importError.message : "Could not import strategy document.");
      setStatus("");
    } finally {
      setImportingKnowledgeTemplateId("");
    }
  }

  function addSetupTemplateGradeBand(templateId: string) {
    setSetupTemplateDrafts((current) =>
      current.map((template) =>
        template.id === templateId
          ? {
              ...template,
              gradeBands: [...(template.gradeBands || []), { id: uniqueId("grade"), label: "", minScore: 0, maxScore: null }]
            }
          : template
      )
    );
  }

  function updateSetupTemplateGradeBand(templateId: string, bandId: string, updates: Partial<ChecklistGradeBand>) {
    setSetupTemplateDrafts((current) =>
      current.map((template) =>
        template.id === templateId
          ? {
              ...template,
              gradeBands: (template.gradeBands || defaultChecklistGradeBands).map((band) =>
                band.id === bandId ? { ...band, ...updates } : band
              )
            }
          : template
      )
    );
  }

  function removeSetupTemplateGradeBand(templateId: string, bandId: string) {
    setSetupTemplateDrafts((current) =>
      current.map((template) =>
        template.id === templateId ? { ...template, gradeBands: (template.gradeBands || []).filter((band) => band.id !== bandId) } : template
      )
    );
  }

  function resetSetupTemplateGradeBands(templateId: string) {
    setSetupTemplateDrafts((current) =>
      current.map((template) => (template.id === templateId ? { ...template, gradeBands: defaultChecklistGradeBands } : template))
    );
  }

  function addSetupTemplateGroup(templateId: string, name = "New Group") {
    setSetupTemplateDrafts((current) =>
      current.map((template) => {
        if (template.id !== templateId) return template;
        const groups = [...(template.groups || []), newSetupTemplateGroup(name)];
        return syncTemplate(template, groups);
      })
    );
  }

  function updateSetupTemplateGroup(templateId: string, groupId: string, updates: Partial<SetupChecklistGroup>) {
    setSetupTemplateDrafts((current) =>
      current.map((template) => {
        if (template.id !== templateId) return template;
        const groups = (template.groups || []).map((group) => (group.id === groupId ? { ...group, ...updates } : group));
        return syncTemplate(template, groups);
      })
    );
  }

  function removeSetupTemplateGroup(templateId: string, groupId: string) {
    setSetupTemplateDrafts((current) =>
      current.map((template) => {
        if (template.id !== templateId) return template;
        const groups = (template.groups || []).filter((group) => group.id !== groupId);
        return syncTemplate(template, groups);
      })
    );
  }

  function addSetupTemplateCriteria(templateId: string, groupId: string, inputType: ChecklistInputType = "boolean") {
    setSetupTemplateDrafts((current) =>
      current.map((template) => {
        if (template.id !== templateId) return template;
        const groups = (template.groups || []).map((group) =>
          group.id === groupId ? { ...group, criteria: [...group.criteria, newSetupTemplateCriterion(inputType)] } : group
        );
        return syncTemplate(template, groups);
      })
    );
  }

  function updateSetupTemplateCriteria(
    templateId: string,
    groupId: string,
    criteriaId: string,
    updates: Partial<SetupTemplateCriterion>
  ) {
    setSetupTemplateDrafts((current) =>
      current.map((template) => {
        if (template.id !== templateId) return template;
        const groups = (template.groups || []).map((group) =>
          group.id === groupId
            ? { ...group, criteria: group.criteria.map((item) => (item.id === criteriaId ? { ...item, ...updates } : item)) }
            : group
        );
        return syncTemplate(template, groups);
      })
    );
  }

  function removeSetupTemplateCriteria(templateId: string, groupId: string, criteriaId: string) {
    setSetupTemplateDrafts((current) =>
      current.map((template) => {
        if (template.id !== templateId) return template;
        const groups = (template.groups || []).map((group) =>
          group.id === groupId ? { ...group, criteria: group.criteria.filter((item) => item.id !== criteriaId) } : group
        );
        return syncTemplate(template, groups);
      })
    );
  }

  function dropSetupTemplateCriterion(event: DragEvent<HTMLDivElement>, targetTemplateId: string, targetGroupId: string, targetCriteriaId: string) {
    event.preventDefault();
    const source = draggedSetupCriterion;
    setDraggedSetupCriterion(null);
    setDragOverSetupCriterionId("");
    if (!source || source.templateId !== targetTemplateId || source.criteriaId === targetCriteriaId) return;

    setSetupTemplateDrafts((current) =>
      current.map((template) => {
        if (template.id !== targetTemplateId) return template;
        const sourceGroup = (template.groups || []).find((group) => group.id === source.groupId);
        const movedCriterion = sourceGroup?.criteria.find((criterion) => criterion.id === source.criteriaId);
        const targetGroup = (template.groups || []).find((group) => group.id === targetGroupId);
        const targetIndex = targetGroup?.criteria.findIndex((criterion) => criterion.id === targetCriteriaId) ?? -1;
        if (!movedCriterion || targetIndex < 0) return template;

        const groupsWithoutSource = (template.groups || []).map((group) =>
          group.id === source.groupId ? { ...group, criteria: group.criteria.filter((criterion) => criterion.id !== source.criteriaId) } : group
        );
        const groups = groupsWithoutSource.map((group) => {
          if (group.id !== targetGroupId) return group;
          const criteria = [...group.criteria];
          criteria.splice(targetIndex, 0, movedCriterion);
          return { ...group, criteria };
        });
        return syncTemplate(template, groups);
      })
    );
  }

  async function persistSetupTemplates(templates: SetupChecklistTemplate[], successMessage = "Setup checklists saved.") {
    const invalidTemplate = templates.find((template) => !template.setupName.trim());
    const invalidCriteriaTemplate = templates.find(
      (template) =>
        template.setupName.trim() &&
        !(template.groups || []).some((group) =>
          group.criteria.some((criteria) => criteria.criteria.trim() && numberValue(criteria.points) > 0)
        )
    );
    const invalidGroupTemplate = templates.find(
      (template) => template.setupName.trim() && !(template.groups || []).some((group) => group.name.trim())
    );
    const invalidGradeTemplate = templates.find(
      (template) =>
        template.setupName.trim() &&
        !(template.gradeBands || []).some((band) => band.label.trim() && Number.isFinite(Number(band.minScore)))
    );

    if (!templates.length) return setStatus("Add at least one setup before saving.");
    if (invalidTemplate) return setStatus("Every setup needs a setup name before saving.");
    if (invalidCriteriaTemplate) return setStatus(`${invalidCriteriaTemplate.setupName} needs at least one valid criteria row with points.`);
    if (invalidGroupTemplate) return setStatus(`${invalidGroupTemplate.setupName} needs at least one named criteria group.`);
    if (invalidGradeTemplate) return setStatus(`${invalidGradeTemplate.setupName} needs at least one valid grade rule.`);

    setStatus("Saving setup checklists...");
    const response = await fetch("/api/settings/setup-checklists", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ setupChecklists: templates })
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) return setStatus(data.error || "Could not save setup checklists.");
    setSetupTemplateDrafts(data.setupChecklists || []);
    setStatus(successMessage);
  }

  async function saveSetupTemplates() {
    await persistSetupTemplates(setupTemplateDrafts);
  }

  return (
    <section className="branden-journal-main">
        <section className="trade-entry-form setup-settings-page" aria-label="Setup builder">
          <div className="trade-panel-heading">
            <div>
              <h3>Setup Builder</h3>
              <span>Define strategy criteria, grade rules, and optional Excel tag mappings.</span>
            </div>
            <Link className="trade-muted-button" href="/journal/branden/dashboard">Back to Dashboard</Link>
          </div>
          {isLoading ? <p className="status">Loading setup builder...</p> : null}
          {error ? <p className="status error">{error}</p> : null}
          {!canEdit && user ? <p className="muted">Read-only access. Setup criteria and grade rules cannot be changed.</p> : null}

          {user ? (
            <fieldset className="setup-builder-fieldset" disabled={!canEdit}>
              <div className="setup-template-list">
                {setupTemplateDrafts.map((template) => {
                  const isSetupExpanded = expandedSetupTemplateIds.includes(template.id);
                  return (
                  <article className={["setup-template-card", !isSetupExpanded ? "collapsed" : ""].filter(Boolean).join(" ")} key={template.id}>
                    <div className="setup-template-heading">
                      <label>
                        Setup name
                        <input value={template.setupName} onChange={(event) => updateSetupTemplate(template.id, { setupName: event.target.value })} placeholder="Pullback trend trade" />
                      </label>
                      <div className="setup-template-heading-actions">
                        <div className="setup-template-summary">
                          <span>{(template.knowledgeSources || []).length} knowledge</span>
                          <span>{(template.strategyExamples || []).length} examples</span>
                          <span>{(template.groups || []).reduce((sum, group) => sum + group.criteria.length, 0)} criteria</span>
                        </div>
                        <button className="trade-muted-button" type="button" onClick={() => toggleSetupTemplate(template.id)}>
                          {isSetupExpanded ? "Collapse" : "Expand"}
                        </button>
                        <button className="trade-danger-button" type="button" onClick={() => removeSetupTemplate(template.id)}>Remove Setup</button>
                      </div>
                    </div>
                    {isSetupExpanded ? (
                      <>
                        <label>
                          Description
                          <textarea value={template.description || ""} onChange={(event) => updateSetupTemplate(template.id, { description: event.target.value })} placeholder="What this setup is designed to capture..." />
                        </label>

                    {false ? (
                      <>
                    <div className="setup-builder-section">
                      <div className="setup-builder-subheading">
                        <div>
                          <h4>Strategy Knowledge</h4>
                          <span>Setup-specific source material used by AI trade reviews.</span>
                        </div>
                        <div className="setup-knowledge-summary">
                          <span>{(template.knowledgeSources || []).filter((source) => source.active !== false).length} active</span>
                          <span>{(template.knowledgeSources || []).length} total</span>
                          <span>{(template.knowledgeSources || []).reduce((sum, source) => sum + (source.chunks?.length || 0), 0)} chunks</span>
                        </div>
                      </div>
                      <div className="setup-knowledge-actions">
                        <label className="setup-knowledge-upload trade-muted-button">
                          {importingKnowledgeTemplateId === template.id ? "Importing document..." : "Import PDF / DOCX / TXT"}
                          <input
                            type="file"
                            accept=".pdf,.docx,.txt,.md,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document,text/plain,text/markdown"
                            disabled={importingKnowledgeTemplateId === template.id}
                            onChange={(event) => {
                              void importSetupKnowledgeFile(template.id, event.target.files);
                              event.currentTarget.value = "";
                            }}
                          />
                        </label>
                        <button className="trade-muted-button" type="button" onClick={() => addSetupKnowledgeSource(template.id)}>Add Manual Source</button>
                      </div>
                      <div className="setup-knowledge-list">
                        {(template.knowledgeSources || []).map((source) => {
                          const isExpanded = expandedKnowledgeSourceIds.includes(source.id);
                          const chunks = source.chunks?.length || 0;
                          const words = wordCount(source.content || "");
                          return (
                            <section className={["setup-knowledge-card", source.active === false ? "inactive" : ""].filter(Boolean).join(" ")} key={source.id}>
                              <div className="setup-knowledge-card-head">
                                <div className="setup-knowledge-source-title">
                                  <strong>{source.title || "Untitled source"}</strong>
                                  <div className="setup-knowledge-badges">
                                    <span>{sourceTypeLabel(source.sourceType)}</span>
                                    <span>{source.active === false ? "Excluded from AI reviews" : "Used in AI reviews"}</span>
                                    <span>{chunks} chunk{chunks === 1 ? "" : "s"}</span>
                                    <span>{words} words</span>
                                  </div>
                                </div>
                                <div className="setup-knowledge-card-actions">
                                  <button className="trade-muted-button" type="button" onClick={() => toggleSetupKnowledgeSource(source.id)}>
                                    {isExpanded ? "Collapse" : "Edit"}
                                  </button>
                                  <button className="trade-muted-button" type="button" onClick={() => reindexSetupKnowledgeSource(template.id, source.id)} disabled={!source.content.trim()}>
                                    Re-index
                                  </button>
                                  <button className="trade-danger-button" type="button" onClick={() => removeSetupKnowledgeSource(template.id, source.id)}>Remove</button>
                                </div>
                              </div>
                              <p className="setup-knowledge-preview">{sourcePreview(source)}</p>
                              <div className="setup-knowledge-meta">
                                <span>Reference: {source.url || "None"}</span>
                                <span>Updated: {shortDate(source.updatedAt)}</span>
                              </div>
                              {isExpanded ? (
                                <div className="setup-knowledge-editor">
                                  <div className="setup-knowledge-grid">
                                    <label>
                                      Source title
                                      <input
                                        value={source.title}
                                        onChange={(event) => updateSetupKnowledgeSource(template.id, source.id, { title: event.target.value })}
                                        placeholder="CANSLIM sell rules / OTC trade plan"
                                      />
                                    </label>
                                    <label>
                                      Type
                                      <select
                                        value={source.sourceType}
                                        onChange={(event) =>
                                          updateSetupKnowledgeSource(template.id, source.id, {
                                            sourceType: event.target.value as SetupKnowledgeSourceDraft["sourceType"]
                                          })
                                        }
                                      >
                                        <option value="notes">Notes</option>
                                        <option value="resource">Resource link</option>
                                        <option value="document">Document excerpt</option>
                                      </select>
                                    </label>
                                    <label>
                                      URL / reference
                                      <input
                                        value={source.url}
                                        onChange={(event) => updateSetupKnowledgeSource(template.id, source.id, { url: event.target.value })}
                                        placeholder="https://... or source name"
                                      />
                                    </label>
                                    <label className="setup-knowledge-active-toggle">
                                      Use in AI reviews
                                      <input
                                        type="checkbox"
                                        checked={source.active !== false}
                                        onChange={(event) => updateSetupKnowledgeSource(template.id, source.id, { active: event.target.checked })}
                                      />
                                    </label>
                                  </div>
                                  <label>
                                    Strategy context
                                    <textarea
                                      value={source.content}
                                      onChange={(event) => updateSetupKnowledgeSource(template.id, source.id, { content: event.target.value })}
                                      placeholder="Paste the rules, playbook notes, book/resource summary, examples, invalidation rules, sizing rules, entry/exit rules..."
                                    />
                                  </label>
                                  <p className="muted">
                                    {chunks
                                      ? `Indexed into ${chunks} review chunk${chunks === 1 ? "" : "s"}. Re-index after large edits.`
                                      : "No review chunks yet. Click Re-index after adding strategy context."}
                                  </p>
                                </div>
                              ) : null}
                            </section>
                          );
                        })}
                        {!(template.knowledgeSources || []).length ? <p className="muted">No strategy knowledge added yet.</p> : null}
                      </div>
                    </div>

                    <div className="setup-builder-section">
                      <div className="setup-builder-subheading">
                        <div>
                          <h4>Strategy Examples</h4>
                          <span>Visual case studies for this setup. Use clean winners, failed breakouts, cautionary examples, and your own best/worst trades.</span>
                        </div>
                        <div className="setup-knowledge-summary">
                          <span>{(template.strategyExamples || []).filter((example) => example.active !== false).length} active</span>
                          <span>{(template.strategyExamples || []).length} total</span>
                          <span>{(template.strategyExamples || []).reduce((sum, example) => sum + (example.screenshots?.length || 0), 0)} images</span>
                        </div>
                      </div>
                      <div className="setup-knowledge-actions">
                        <button className="trade-muted-button" type="button" onClick={() => addSetupStrategyExample(template.id)}>Add Strategy Example</button>
                        <label className="setup-knowledge-upload trade-muted-button">
                          Import CSV + images
                          <input
                            type="file"
                            accept=".csv,text/csv,image/*"
                            multiple
                            onChange={(event) => {
                              void importSetupStrategyExamplesCsv(template.id, event.target.files);
                              event.currentTarget.value = "";
                            }}
                          />
                        </label>
                      </div>
                      <div className="setup-example-list">
                        {(template.strategyExamples || []).map((example) => {
                          const isExpanded = expandedStrategyExampleIds.includes(example.id);
                          return (
                            <section className={["setup-example-card", example.active === false ? "inactive" : ""].filter(Boolean).join(" ")} key={example.id}>
                              <div className="setup-knowledge-card-head">
                                <div className="setup-knowledge-source-title">
                                  <strong>{example.symbol || "Untitled example"}</strong>
                                  <div className="setup-knowledge-badges">
                                    <span>{exampleQualityLabel(example.quality)}</span>
                                    <span>{example.active === false ? "Excluded from AI reviews" : "Used in AI reviews"}</span>
                                    <span>{example.screenshots?.length || 0} image{example.screenshots?.length === 1 ? "" : "s"}</span>
                                    {example.setupType ? <span>{example.setupType}</span> : null}
                                  </div>
                                </div>
                                <div className="setup-knowledge-card-actions">
                                  <button className="trade-muted-button" type="button" onClick={() => toggleSetupStrategyExample(example.id)}>
                                    {isExpanded ? "Collapse" : "Edit"}
                                  </button>
                                  <button className="trade-danger-button" type="button" onClick={() => removeSetupStrategyExample(template.id, example.id)}>Remove</button>
                                </div>
                              </div>
                              <p className="setup-knowledge-preview">{examplePreview(example)}</p>
                              <div className="setup-knowledge-meta">
                                <span>Source: {example.source || "None"}</span>
                                <span>Updated: {shortDate(example.updatedAt)}</span>
                              </div>
                              {isExpanded ? (
                                <div className="setup-example-editor">
                                  <div className="setup-example-grid">
                                    <label>
                                      Symbol
                                      <input value={example.symbol} onChange={(event) => updateSetupStrategyExample(template.id, example.id, { symbol: event.target.value.toUpperCase() })} placeholder="NVDA" />
                                    </label>
                                    <label>
                                      Setup type
                                      <input value={example.setupType} onChange={(event) => updateSetupStrategyExample(template.id, example.id, { setupType: event.target.value })} placeholder="Cup with handle / flat base / failed breakout" />
                                    </label>
                                    <label>
                                      Example quality
                                      <select value={example.quality} onChange={(event) => updateSetupStrategyExample(template.id, example.id, { quality: event.target.value as SetupStrategyExampleQuality })}>
                                        <option value="ideal">Ideal</option>
                                        <option value="good">Good</option>
                                        <option value="failed">Failed</option>
                                        <option value="bad">Bad</option>
                                        <option value="cautionary">Cautionary</option>
                                      </select>
                                    </label>
                                    <label>
                                      Outcome
                                      <input value={example.outcome} onChange={(event) => updateSetupStrategyExample(template.id, example.id, { outcome: event.target.value })} placeholder="Big winner / failed breakout / controlled loss" />
                                    </label>
                                    <label>
                                      Source
                                      <input value={example.source} onChange={(event) => updateSetupStrategyExample(template.id, example.id, { source: event.target.value })} placeholder="IBD / TraderLion / My trade / MarketSurge" />
                                    </label>
                                    <label>
                                      Source URL
                                      <input value={example.sourceUrl} onChange={(event) => updateSetupStrategyExample(template.id, example.id, { sourceUrl: event.target.value })} placeholder="https://..." />
                                    </label>
                                    <label className="setup-knowledge-active-toggle">
                                      Use in AI reviews
                                      <input
                                        type="checkbox"
                                        checked={example.active !== false}
                                        onChange={(event) => updateSetupStrategyExample(template.id, example.id, { active: event.target.checked })}
                                      />
                                    </label>
                                  </div>
                                  <label>
                                    Example notes
                                    <textarea
                                      value={example.notes}
                                      onChange={(event) => updateSetupStrategyExample(template.id, example.id, { notes: event.target.value })}
                                      placeholder="What made this valid or invalid? Where was the pivot? What should the AI compare against? What was the sell/risk lesson?"
                                    />
                                  </label>
                                  <div className="setup-example-screenshots">
                                    <div className="setup-example-screenshot-actions">
                                      <strong>Chart screenshots</strong>
                                      <label className="setup-knowledge-upload trade-muted-button">
                                        Add images
                                        <input
                                          type="file"
                                          accept="image/*"
                                          multiple
                                          onChange={(event) => {
                                            void addSetupStrategyExampleScreenshots(template.id, example, event.target.files);
                                            event.currentTarget.value = "";
                                          }}
                                        />
                                      </label>
                                    </div>
                                    <div className="setup-example-screenshot-grid">
                                      {(example.screenshots || []).map((screenshot, index) => (
                                        <div className="setup-example-screenshot" key={`${screenshot.slice(0, 32)}-${index}`}>
                                          <img
                                            src={screenshot}
                                            alt={`${example.symbol || "Strategy"} example ${index + 1}`}
                                            loading="lazy"
                                            decoding="async"
                                          />
                                          <button
                                            className="trade-danger-button"
                                            type="button"
                                            onClick={() =>
                                              updateSetupStrategyExample(template.id, example.id, {
                                                screenshots: example.screenshots.filter((_, screenshotIndex) => screenshotIndex !== index)
                                              })
                                            }
                                          >
                                            Remove
                                          </button>
                                        </div>
                                      ))}
                                      {!example.screenshots?.length ? <p className="muted">No example screenshots attached.</p> : null}
                                    </div>
                                  </div>
                                </div>
                              ) : null}
                            </section>
                          );
                        })}
                        {!(template.strategyExamples || []).length ? <p className="muted">No strategy examples added yet.</p> : null}
                      </div>
                    </div>
                      </>
                    ) : (
                      <div className="setup-builder-section">
                        <div className="setup-builder-subheading">
                          <div>
                            <h4>AI Knowledge Library</h4>
                            <span>Strategy documents and visual examples moved to their own workspace.</span>
                          </div>
                          <div className="setup-knowledge-summary">
                            <span>{(template.knowledgeSources || []).length} knowledge</span>
                            <span>{(template.strategyExamples || []).length} examples</span>
                            <span>{(template.strategyExamples || []).reduce((sum, example) => sum + (example.screenshots?.length || 0), 0)} images</span>
                          </div>
                        </div>
                        <div className="setup-knowledge-actions">
                          <Link className="trade-muted-button" href="/journal/branden/ai-knowledge">Manage AI Knowledge</Link>
                        </div>
                      </div>
                    )}

                    <div className="setup-builder-section">
                      <div className="setup-builder-subheading"><h4>Grade Rules</h4><span>Score is based on checked criteria points for this setup.</span></div>
                      <div className="grade-band-list setup-grade-band-list">
                        {(template.gradeBands || defaultChecklistGradeBands).map((band) => (
                          <div className="grade-band-row" key={band.id}>
                            <label>Grade<input value={band.label} onChange={(event) => updateSetupTemplateGradeBand(template.id, band.id, { label: event.target.value })} placeholder="A+" /></label>
                            <label>From score<input type="number" step="any" inputMode="decimal" value={String(band.minScore)} onChange={(event) => updateSetupTemplateGradeBand(template.id, band.id, { minScore: numberValue(event.target.value) })} /></label>
                            <label>To score<input type="number" step="any" inputMode="decimal" value={band.maxScore === null ? "" : String(band.maxScore)} onChange={(event) => updateSetupTemplateGradeBand(template.id, band.id, { maxScore: event.target.value === "" ? null : numberValue(event.target.value) })} placeholder="No max" /></label>
                            <button className="trade-danger-button" type="button" onClick={() => removeSetupTemplateGradeBand(template.id, band.id)}>Remove</button>
                          </div>
                        ))}
                      </div>
                      <div className="grade-band-actions">
                        <button className="trade-muted-button" type="button" onClick={() => addSetupTemplateGradeBand(template.id)}>Add Grade Rule</button>
                        <button className="trade-muted-button" type="button" onClick={() => resetSetupTemplateGradeBands(template.id)}>Reset Defaults</button>
                      </div>
                    </div>

                    <div className="setup-builder-section">
                      <div className="setup-builder-subheading"><h4>Checklist Criteria</h4><span>Group criteria by workflow, and choose simple yes/no checks or scored point rows.</span></div>
                      <div className="setup-template-groups">
                        {(template.groups || []).map((group) => (
                          <section className="setup-group-card" key={group.id}>
                            <div className="setup-group-head">
                              <input value={group.name} onChange={(event) => updateSetupTemplateGroup(template.id, group.id, { name: event.target.value })} placeholder="Fundamentals" />
                              <button className="trade-danger-button" type="button" onClick={() => removeSetupTemplateGroup(template.id, group.id)}>Remove Group</button>
                            </div>
                            <div className="setup-template-criteria">
                              <div className="setup-template-grid-head"><span>Criteria</span><span>Type</span><span>Points</span><span>Excel field</span><span>Match value</span><span>Action</span></div>
                              {group.criteria.map((criteria) => (
                                <div
                                  className={["setup-template-row", draggedSetupCriterion?.criteriaId === criteria.id ? "dragging" : "", dragOverSetupCriterionId === criteria.id ? "drag-over" : ""].filter(Boolean).join(" ")}
                                  key={criteria.id}
                                  onDragEnter={() => setDragOverSetupCriterionId(criteria.id)}
                                  onDragOver={(event) => { event.preventDefault(); event.dataTransfer.dropEffect = "move"; }}
                                  onDragLeave={(event) => {
                                    if (!event.currentTarget.contains(event.relatedTarget as Node | null)) {
                                      setDragOverSetupCriterionId((current) => current === criteria.id ? "" : current);
                                    }
                                  }}
                                  onDrop={(event) => dropSetupTemplateCriterion(event, template.id, group.id, criteria.id)}
                                >
                                  <input value={criteria.criteria} onChange={(event) => updateSetupTemplateCriteria(template.id, group.id, criteria.id, { criteria: event.target.value })} placeholder="Price closes outside the 50-period Bollinger Band" />
                                  <select value={criteria.inputType || "boolean"} onChange={(event) => updateSetupTemplateCriteria(template.id, group.id, criteria.id, { inputType: event.target.value as ChecklistInputType })}>
                                    <option value="boolean">Yes / No</option>
                                    <option value="points">Points</option>
                                  </select>
                                  <input type="number" min="0" step="any" inputMode="decimal" value={String(criteria.points)} onChange={(event) => updateSetupTemplateCriteria(template.id, group.id, criteria.id, { points: numberValue(event.target.value) })} />
                                  <input value={criteria.importTagKey || ""} onChange={(event) => updateSetupTemplateCriteria(template.id, group.id, criteria.id, { importTagKey: event.target.value })} placeholder="Breakout" />
                                  <input value={criteria.importTagValue || ""} onChange={(event) => updateSetupTemplateCriteria(template.id, group.id, criteria.id, { importTagValue: event.target.value })} placeholder="Yes" />
                                  <div className="setup-criterion-row-actions">
                                    <button
                                      className="setup-criterion-drag-handle"
                                      type="button"
                                      draggable
                                      aria-label={`Reorder ${criteria.criteria || "criteria"}`}
                                      title="Drag to reorder"
                                      onDragStart={(event) => {
                                        setDraggedSetupCriterion({ templateId: template.id, groupId: group.id, criteriaId: criteria.id });
                                        event.dataTransfer.effectAllowed = "move";
                                        event.dataTransfer.setData("text/plain", criteria.id);
                                      }}
                                      onDragEnd={() => { setDraggedSetupCriterion(null); setDragOverSetupCriterionId(""); }}
                                    >
                                      ⋮⋮
                                    </button>
                                    <button className="trade-danger-button" type="button" onClick={() => removeSetupTemplateCriteria(template.id, group.id, criteria.id)}>Remove</button>
                                  </div>
                                </div>
                              ))}
                            </div>
                            <div className="setup-criteria-actions">
                              <button className="trade-muted-button" type="button" onClick={() => addSetupTemplateCriteria(template.id, group.id, "boolean")}>Add Yes/No Criteria</button>
                              <button className="trade-muted-button" type="button" onClick={() => addSetupTemplateCriteria(template.id, group.id, "points")}>Add Points Criteria</button>
                            </div>
                          </section>
                        ))}
                      </div>
                      <button className="trade-muted-button" type="button" onClick={() => addSetupTemplateGroup(template.id)}>Add Criteria Group</button>
                    </div>
                      </>
                    ) : null}
                  </article>
                  );
                })}
                {!setupTemplateDrafts.length ? <p className="muted">No setups yet. Add a setup like OTC or CANSLIM, define the criteria, and optionally map OTC rows to Excel fields.</p> : null}
              </div>

              {canEdit ? (
                <div className="grade-band-actions">
                  <button className="trade-muted-button" type="button" onClick={addSetupTemplate}>Add Setup</button>
                  <button className="trade-muted-button" type="button" onClick={() => addPresetSetupTemplate("OTC")}>Add OTC Preset</button>
                  <button className="trade-muted-button" type="button" onClick={() => addPresetSetupTemplate("CANSLIM")}>Add CANSLIM Preset</button>
                  <button type="button" onClick={saveSetupTemplates}>Save Setup Builder</button>
                  {status ? <span className="status">{status}</span> : null}
                </div>
              ) : null}
            </fieldset>
          ) : null}
        </section>
      </section>
  );
}
