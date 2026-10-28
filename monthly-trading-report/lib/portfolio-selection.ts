"use client";

import { useCallback, useEffect, useState, type SetStateAction } from "react";

export const PORTFOLIO_SELECTION_KEY = "branden-active-portfolio";
const CHANGE_EVENT = "branden-portfolio-change";

export function readPortfolioSelection(fallback = "") {
  try { return localStorage.getItem(PORTFOLIO_SELECTION_KEY) ?? fallback; }
  catch { return fallback; }
}

/** One viewing scope across the dashboard, journal pages, and sidebar. */
export function usePortfolioSelection(initial = "") {
  const [portfolio, setPortfolio] = useState(initial);
  useEffect(() => {
    setPortfolio(readPortfolioSelection(initial));
    const update = () => setPortfolio(readPortfolioSelection(initial));
    window.addEventListener(CHANGE_EVENT, update);
    window.addEventListener("storage", update);
    return () => { window.removeEventListener(CHANGE_EVENT, update); window.removeEventListener("storage", update); };
  }, [initial]);
  const select = useCallback((value: SetStateAction<string>) => {
    const next = typeof value === "function" ? value(readPortfolioSelection()) : value;
    try { localStorage.setItem(PORTFOLIO_SELECTION_KEY, next); } catch { /* In-memory selection still works. */ }
    setPortfolio(next);
    window.dispatchEvent(new Event(CHANGE_EVENT));
  }, []);
  const initialize = useCallback((fallback: string) => {
    // Loading a page must not replace a deliberate selection with its default.
    setPortfolio(readPortfolioSelection(fallback));
  }, []);
  return [portfolio, select, initialize] as const;
}
