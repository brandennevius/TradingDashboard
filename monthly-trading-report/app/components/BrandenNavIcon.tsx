import type { ReactNode } from "react";

type BrandenNavIconName =
  | "dashboard" | "review" | "calendar" | "positions" | "journal"
  | "gauge" | "benchmark" | "sizing" | "setup" | "portfolio" | "settings" | "import";

const icons: Record<BrandenNavIconName, ReactNode> = {
  dashboard: <><rect x="3" y="3" width="7" height="7" rx="1.5" /><rect x="14" y="3" width="7" height="4" rx="1.5" /><rect x="14" y="11" width="7" height="10" rx="1.5" /><rect x="3" y="14" width="7" height="7" rx="1.5" /></>,
  review: <><path d="M9 5H6a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7a2 2 0 0 0-2-2h-3" /><rect x="9" y="3" width="6" height="4" rx="1" /><path d="m8 14 2.5 2.5L16 11" /></>,
  calendar: <><rect x="3" y="5" width="18" height="16" rx="2" /><path d="M7 3v4m10-4v4M3 11h18m-14 4h2m4 0h2m-8 3h2" /></>,
  positions: <><path d="m3 7 9-4 9 4-9 4-9-4Zm0 5 9 4 9-4M3 17l9 4 9-4" /></>,
  journal: <><path d="M12 5C9 3 6 3 3 4v15c3-1 6-1 9 1 3-2 6-2 9-1V4c-3-1-6-1-9 1Zm0 0v15M6 8h3m-3 4h3m6-4h3m-3 4h3" /></>,
  gauge: <><path d="M4.2 19a9 9 0 1 1 15.6 0M5 12H3m4-6L5.5 4.5M12 5V3m5 3 1.5-1.5M19 12h2m-9 3 4-6" /><circle cx="12" cy="16" r="1.5" /></>,
  benchmark: <><path d="M3 3v18h18M6 15l4-5 4 2 6-7M6 18l4-2 4 1 6-4" /></>,
  sizing: <><path d="M12 3v17M7 21h10M4 7h16M6 7l-4 8h8L6 7Zm12 0-4 8h8l-4-8Z" /></>,
  setup: <><path d="M6 3v4m0 4v10M12 3v10m0 4v4M18 3v1m0 4v13" /><circle cx="6" cy="9" r="2" /><circle cx="12" cy="15" r="2" /><circle cx="18" cy="6" r="2" /></>,
  portfolio: <><rect x="3" y="7" width="18" height="14" rx="2" /><path d="M8 7V5a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2M3 12a22 22 0 0 0 18 0m-9 0v3" /></>,
  settings: <><path d="m9.5 4 .6-2h3.8l.6 2 2 .9 1.9-.5 1.9 3.2-1.4 1.5.2 2.2 1.5 1.5-1.9 3.3-2-.5-1.9 1-.6 2h-3.8l-.6-2-2-.9-1.9.5-1.9-3.2 1.4-1.5-.2-2.2-1.5-1.5 1.9-3.3 2 .5 1.9-1Z" transform="translate(0 1)" /><circle cx="12" cy="11.5" r="3" /></>,
  import: <><path d="M14 3H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9l-6-6Zm0 0v6h6M12 12v6m-3-3 3 3 3-3" /></>
};

export default function BrandenNavIcon({ name }: { name: BrandenNavIconName }) {
  return <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" focusable="false">
    {icons[name]}
  </svg>;
}
