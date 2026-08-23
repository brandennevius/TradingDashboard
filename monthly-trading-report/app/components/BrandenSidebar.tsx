"use client";

import Link from "next/link";

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
      { href: "/journal/branden/market-review", label: "Market Review", icon: "M" },
      { href: "/journal/branden/calendar", label: "Calendar", icon: "C" },
      { href: "/journal/branden/open-positions", label: "Open Positions", icon: "H" },
      { href: "/journal/branden/trade-log", label: "Trade Log", icon: "L" }
    ]
  },
  {
    title: "Tools",
    items: [
      { href: "/journal/branden/market-gauge", label: "Market Gauge", icon: "G" },
      { href: "/journal/branden/bottom-to-bull", label: "Bottom to Bull", icon: "B" },
      { href: "/journal/branden/time-stop", label: "Time Stop", icon: "T" },
      { href: "/journal/branden/benchmark", label: "Benchmark", icon: "X" },
      { href: "/journal/branden/rprp", label: "RPRP Sizer", icon: "R" },
      { href: "/journal/branden/setup-builder", label: "Setup Builder", icon: "S" },
      { href: "/journal/branden/ai-knowledge", label: "AI Knowledge", icon: "K" },
      { href: "/journal/branden/portfolios", label: "Portfolios", icon: "P" }
    ]
  },
  {
    title: "Account",
    items: [{ href: "/journal/branden/settings", label: "Settings", icon: "G" }]
  }
];

export default function BrandenSidebar({ activeHref, accountActions = [] }: BrandenSidebarProps) {
  return (
    <aside className="branden-journal-sidebar">
      <nav className="branden-sidebar-nav" aria-label="Branden journal navigation">
        {brandenSidebarGroups.map((group) => {
          const items: BrandenSidebarItem[] =
            group.title === "Account"
              ? [
                  ...group.items,
                  ...accountActions.map((action) => ({
                    action,
                    label: action.label,
                    icon: action.icon
                  }))
                ]
              : group.items;

          return (
            <div className="branden-sidebar-group" key={group.title}>
              <span className="branden-sidebar-group-label">{group.title}</span>
              <div className="branden-sidebar-group-items">
                {items.map((item) => {
                  if ("action" in item) {
                    return (
                      <button
                        key={`${group.title}-${item.action.key}`}
                        type="button"
                        disabled={item.action.disabled}
                        onClick={item.action.onClick}
                      >
                        <span className="branden-sidebar-icon">{item.icon}</span>
                        <span className="branden-sidebar-button-text">{item.label}</span>
                        <span className="branden-sidebar-chevron">›</span>
                      </button>
                    );
                  }

                  return (
                    <Link key={`${group.title}-${item.label}`} className={item.href === activeHref ? "active" : ""} href={item.href}>
                      <span className="branden-sidebar-icon">{item.icon}</span>
                      <span className="branden-sidebar-button-text">{item.label}</span>
                      <span className="branden-sidebar-chevron">›</span>
                    </Link>
                  );
                })}
              </div>
            </div>
          );
        })}
      </nav>
    </aside>
  );
}
