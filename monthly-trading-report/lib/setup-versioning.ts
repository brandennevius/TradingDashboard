import type { SetupChecklistTemplate } from "./types";

function gradingDefinition(template: SetupChecklistTemplate) {
  return JSON.stringify({
    setupName: template.setupName,
    groups: template.groups,
    criteria: template.criteria,
    gradeBands: template.gradeBands
  });
}

/** Names are the existing trade-to-setup key; never reuse a historical name. */
export function versionSetupTemplates(
  existing: SetupChecklistTemplate[],
  drafts: SetupChecklistTemplate[]
): SetupChecklistTemplate[] {
  const result = existing.map((template) => ({ ...template, archived: true }));
  const seen = new Set<string>();
  for (const draft of drafts) {
    if (seen.has(draft.id)) throw new Error("Duplicate setup. Reload Setup Builder and try again.");
    seen.add(draft.id);
    const previous = existing.find((template) => template.id === draft.id);
    if (previous?.archived) throw new Error("This setup version is archived. Reload Setup Builder before saving.");
    const familyId = previous?.familyId || previous?.id || draft.id;
    const familyName = previous?.familyName || previous?.setupName || draft.setupName;
    const previousVersion = previous?.version || 1;
    if (previous && gradingDefinition(previous) === gradingDefinition(draft)) {
      const index = result.findIndex((template) => template.id === previous.id);
      result[index] = { ...draft, familyId, familyName, version: previousVersion, archived: false };
      continue;
    }
    const version = previous ? previousVersion + 1 : 1;
    const baseName = previous && draft.setupName === previous.setupName ? familyName : draft.setupName;
    const setupName = previous ? `${baseName} v${version}` : draft.setupName;
    if (result.some((template) => template.setupName.trim().toLowerCase() === setupName.trim().toLowerCase())) {
      throw new Error(`The setup name "${setupName}" is already in use. Choose a different name.`);
    }
    result.push({
      ...draft,
      id: previous ? `${familyId}:v${version}` : draft.id,
      setupName,
      familyId,
      familyName: baseName,
      version,
      archived: false
    });
  }
  return result;
}
