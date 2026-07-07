# Branden Journal Performance Plan

## Goal

Make standalone Branden journal pages feel fast after moving sections out of the dashboard by replacing broad client-side data loading with page-specific, server-shaped, cache-aware data.

## Current bottlenecks

- Several pages call `/api/trades`, which reads every trade and every heavy JSON field, then filters in the browser.
- Open-position pages have a reduced public response, but the backend still loads all trades first.
- Market Gauge makes many browser-side market-data requests and downloads full candle history for each symbol.
- Most standalone pages make `/api/session` plus one or more page-data calls after initial render.
- The static page and route JS sizes are acceptable; runtime API work is the primary bottleneck.

## Phase 1 — highest impact, lowest behavior risk

1. Add database indexes for common journal reads.
2. Add optimized store queries for Branden open positions and calendar rows.
3. Add a lightweight Calendar API that returns only calendar-ready data.
4. Add a server-side aggregated Market Gauge API that fetches/calculates once and returns compact gauge results.
5. Update Calendar, Market Gauge, Open Positions, and Time Stop to use optimized endpoints.

## Phase 2 — reduce broad reads on remaining pages

1. Add Daily Review page-data API scoped by date and portfolio.
2. Add Benchmark summary/read API that returns reduced fields.
3. Keep full `/api/trades` primarily for Trade Log/editor flows.
4. Remove unnecessary client `/api/session` calls by returning user/page data together or server-rendering initial state.

## Phase 3 — structural cleanup

1. Done — extract shared Branden sidebar/navigation to one component.
2. Done — kept the current client editing surfaces client-rendered; server-first rendering is not practical for the trade editor without a larger form rewrite.
3. Done — lazy-load heavy dashboard-only chart/tool components from the main dashboard bundle.
4. Done — add timing/logging around key Branden APIs so slow queries and cold starts are visible in deployment logs.

## Success criteria

- Calendar loads with one lightweight page-data call, not the full trade log.
- Market Gauge loads with one aggregated call, not 15 symbol calls.
- Open Positions and Time Stop read only open Branden positions from the database.
- Trade Log behavior remains unchanged.
- Production build passes before deploy.
