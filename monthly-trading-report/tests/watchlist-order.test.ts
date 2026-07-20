import assert from "node:assert/strict";
import test from "node:test";
import { applyWatchlistItemOrder, reorderWatchlistItems } from "../lib/watchlist-order";

const items = [{ id: "a" }, { id: "b" }, { id: "c" }, { id: "d" }];

test("dragging a ticker reorders it relative to the drop target", () => {
  assert.deepEqual(reorderWatchlistItems(items, "a", "c").map((item) => item.id), ["b", "c", "a", "d"]);
  assert.deepEqual(reorderWatchlistItems(items, "d", "b").map((item) => item.id), ["a", "d", "b", "c"]);
});

test("persisted priority order must contain every saved ticker exactly once", () => {
  assert.deepEqual(applyWatchlistItemOrder(items, ["d", "b", "a", "c"])?.map((item) => item.id), ["d", "b", "a", "c"]);
  assert.equal(applyWatchlistItemOrder(items, ["a", "b", "c"]), null);
  assert.equal(applyWatchlistItemOrder(items, ["a", "a", "c", "d"]), null);
  assert.equal(applyWatchlistItemOrder(items, ["a", "b", "c", "missing"]), null);
});
