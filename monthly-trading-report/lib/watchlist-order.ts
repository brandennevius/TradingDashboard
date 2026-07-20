export function reorderWatchlistItems<T extends { id: string }>(items: T[], draggedId: string, targetId: string) {
  const fromIndex = items.findIndex((item) => item.id === draggedId);
  const targetIndex = items.findIndex((item) => item.id === targetId);
  if (fromIndex < 0 || targetIndex < 0 || fromIndex === targetIndex) return items;

  const reordered = [...items];
  const [dragged] = reordered.splice(fromIndex, 1);
  reordered.splice(targetIndex, 0, dragged);
  return reordered;
}

export function applyWatchlistItemOrder<T extends { id: string }>(items: T[], orderedItemIds: string[]) {
  if (orderedItemIds.length !== items.length || new Set(orderedItemIds).size !== items.length) return null;
  const byId = new Map(items.map((item) => [item.id, item]));
  const ordered = orderedItemIds.map((id) => byId.get(id));
  return ordered.every(Boolean) ? ordered as T[] : null;
}
