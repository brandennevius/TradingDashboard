"use client";

import Link from "next/link";
import { useEffect, useRef, useState, type CSSProperties, type PointerEvent } from "react";

const SIDEBAR_PREFERENCES_KEY = "branden-sidebar-preferences";
const DEFAULT_WIDTH = 264;
const MIN_WIDTH = 220;
const MAX_WIDTH = 420;
const clampWidth = (width: number) => Math.min(MAX_WIDTH, Math.max(MIN_WIDTH, width));

function PanelIcon() {
  return <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" aria-hidden="true">
    <rect x="3" y="4" width="18" height="16" rx="3" />
    <path d="M9 4v16" />
  </svg>;
}

type BrandenSidebarAction = {
  key: string;
  label: string;
  icon: string;
  disabled?: boolean;
  onClick: () => void | Promise<void>;
};

type BrandenSidebarProps = {
  activeHref?: string;
  accountActions?: BrandenSidebarAction[];
};

type BrandenSidebarLinkItem = {
  href: string;
  label: string;
  icon: string;
};

type BrandenSidebarButtonItem = {
  action: BrandenSidebarAction;
  label: string;
  icon: string;
};

type BrandenSidebarItem = BrandenSidebarLinkItem | BrandenSidebarButtonItem;

const brandenSidebarGroups = [
  {
    title: "Overview",
    items: [
      { href: "/journal/branden/dashboard", label: "Dashboard", icon: "D" },
      { href: "/journal/branden/daily-review", label: "Daily Review", icon: "R" },
      { href: "/journal/branden/calendar", label: "Calendar", icon: "C" },
      { href: "/journal/branden/open-positions", label: "Open Positions", icon: "H" },
      { href: "/journal/branden/trade-log", label: "Trade Log", icon: "L" }
    ]
  },
  {
    title: "Tools",
    items: [
      { href: "/journal/branden/market-gauge", label: "Market Gauge", icon: "G" },
      { href: "/journal/branden/benchmark", label: "Benchmark", icon: "X" },
      { href: "/journal/branden/rprp", label: "RPRP Sizer", icon: "R" },
      { href: "/journal/branden/setup-builder", label: "Setup Builder", icon: "S" },
      { href: "/journal/branden/portfolios", label: "Portfolios", icon: "P" }
    ]
  },
  {
    title: "Account",
    items: [{ href: "/journal/branden/settings", label: "Settings", icon: "G" }]
  }
];

export default function BrandenSidebar({ activeHref, accountActions = [] }: BrandenSidebarProps) {
  const [collapsed, setCollapsed] = useState(false);
  const [width, setWidth] = useState(DEFAULT_WIDTH);
  const [closedGroups, setClosedGroups] = useState<Record<string, boolean>>({});
  const [preferencesLoaded, setPreferencesLoaded] = useState(false);
  const [resizing, setResizing] = useState(false);
  const drag = useRef<{ x: number; width: number } | null>(null);

  useEffect(() => {
    try {
      const saved = JSON.parse(localStorage.getItem(SIDEBAR_PREFERENCES_KEY) || "null");
      if (saved && typeof saved === "object") {
        setCollapsed(saved.collapsed === true);
        if (typeof saved.width === "number" && Number.isFinite(saved.width)) setWidth(clampWidth(saved.width));
        if (saved.closedGroups && typeof saved.closedGroups === "object") {
          setClosedGroups(Object.fromEntries(brandenSidebarGroups.map(({ title }) => [title, saved.closedGroups[title] === true])));
        }
      }
    } catch { /* Navigation remains usable when browser storage is unavailable. */ }
    setPreferencesLoaded(true);
  }, []);

  useEffect(() => {
    if (!preferencesLoaded) return;
    try {
      localStorage.setItem(SIDEBAR_PREFERENCES_KEY, JSON.stringify({ collapsed, width, closedGroups }));
    } catch { /* Preferences are optional. */ }
  }, [collapsed, width, closedGroups, preferencesLoaded]);

  function startResize(event: PointerEvent<HTMLDivElement>) {
    if (event.button !== 0) return;
    event.preventDefault();
    drag.current = { x: event.clientX, width };
    event.currentTarget.setPointerCapture(event.pointerId);
    setResizing(true);
  }

  function endResize() {
    drag.current = null;
    setResizing(false);
  }

  return (
    <aside
      className={`branden-journal-sidebar ${collapsed ? "collapsed" : "expanded"}${resizing ? " is-resizing" : ""}`}
      style={{ "--sidebar-width": `${width}px` } as CSSProperties}
    >
      <div className="branden-sidebar-heading">
        {!collapsed ? <span className="branden-sidebar-title">Branden Journal</span> : null}
        <button
          type="button"
          className="branden-sidebar-toggle"
          onClick={() => setCollapsed((value) => !value)}
          aria-label={collapsed ? "Expand navigation" : "Collapse navigation"}
          aria-expanded={!collapsed}
          title={collapsed ? "Expand navigation" : "Collapse navigation"}
        >
          <PanelIcon />
        </button>
      </div>
      <nav className="branden-sidebar-nav" aria-label="Branden journal navigation">
        {brandenSidebarGroups.map((group) => {
          const items: BrandenSidebarItem[] = group.title === "Account"
            ? [...group.items, ...accountActions.map((action) => ({ action, label: action.label, icon: action.icon }))]
            : group.items;
          const groupOpen = collapsed || !closedGroups[group.title];
          const groupId = `branden-nav-${group.title.toLowerCase()}`;

          return (
            <div className="branden-sidebar-group" key={group.title}>
              {!collapsed ? <button
                className="branden-sidebar-group-toggle"
                type="button"
                aria-expanded={groupOpen}
                aria-controls={groupId}
                onClick={() => setClosedGroups((groups) => ({ ...groups, [group.title]: !groups[group.title] }))}
              >
                <span>{group.title}</span>
                <span aria-hidden="true">{groupOpen ? "⌄" : "›"}</span>
              </button> : null}
              <div className="branden-sidebar-group-items" id={groupId} hidden={!groupOpen}>
                {items.map((item) => {
                  const content = <>
                    <span className="branden-sidebar-icon" aria-hidden="true">{item.icon}</span>
                    <span className="branden-sidebar-button-text">{item.label}</span>
                  </>;
                  if ("action" in item) {
                    return <button
                      key={`${group.title}-${item.action.key}`}
                      type="button"
                      aria-label={item.label}
                      title={item.label}
                      disabled={item.action.disabled}
                      onClick={item.action.onClick}
                    >{content}</button>;
                  }
                  return <Link
                    key={item.href}
                    className={item.href === activeHref ? "active" : ""}
                    href={item.href}
                    aria-label={item.label}
                    aria-current={item.href === activeHref ? "page" : undefined}
                    title={item.label}
                  >{content}</Link>;
                })}
              </div>
            </div>
          );
        })}
      </nav>
      {!collapsed ? <div
        className="branden-sidebar-resize"
        role="separator"
        aria-label="Resize navigation"
        aria-orientation="vertical"
        aria-valuemin={MIN_WIDTH}
        aria-valuemax={MAX_WIDTH}
        aria-valuenow={width}
        tabIndex={0}
        onPointerDown={startResize}
        onPointerMove={(event) => {
          if (drag.current) setWidth(clampWidth(drag.current.width + event.clientX - drag.current.x));
        }}
        onPointerUp={endResize}
        onPointerCancel={endResize}
        onLostPointerCapture={endResize}
        onDoubleClick={() => setWidth(DEFAULT_WIDTH)}
        onKeyDown={(event) => {
          if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
          event.preventDefault();
          setWidth((current) => event.key === "Home" ? MIN_WIDTH : event.key === "End" ? MAX_WIDTH : clampWidth(current + (event.key === "ArrowRight" ? 16 : -16)));
        }}
        title="Drag to resize · double-click to reset"
      /> : null}
    </aside>
  );
}
