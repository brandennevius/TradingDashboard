import assert from "node:assert/strict";
import test from "node:test";
import {
  completedTradeReviewSectionCount,
  hasCompletedTradeReview,
  minimumCompletedTradeReviewSections
} from "../lib/trade-review";

test("three of six structured review fields count as complete", () => {
  const review = {
    setup: "Breakout setup",
    entry: "",
    exit: "Managed at support",
    didRight: "Honored the stop",
    didWrong: "",
    general: ""
  };

  assert.equal(completedTradeReviewSectionCount(review), 3);
  assert.equal(minimumCompletedTradeReviewSections, 3);
  assert.equal(hasCompletedTradeReview(review), true);
});

test("two structured review fields do not count as complete", () => {
  assert.equal(hasCompletedTradeReview({
    setup: "Breakout setup",
    entry: "",
    exit: "",
    didRight: "Honored the stop",
    didWrong: "",
    general: ""
  }), false);
});

test("blank and whitespace-only fields are not populated", () => {
  assert.equal(hasCompletedTradeReview({
    setup: "Setup",
    entry: "  ",
    exit: "Exit",
    didRight: "\n",
    didWrong: "Review",
    general: ""
  }), true);
});

test("legacy-only notes remain complete for existing trades", () => {
  assert.equal(hasCompletedTradeReview(undefined, "Legacy trade review"), true);
});

test("partial structured reviews cannot be completed by stale legacy notes", () => {
  assert.equal(hasCompletedTradeReview({ setup: "Setup", entry: "Entry" }, "Legacy trade review"), false);
});
