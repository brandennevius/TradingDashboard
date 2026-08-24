"use client";

import { createContext, type ReactNode, useContext } from "react";

type BrandenSnapshotActions = {
  canGenerateSnapshot: boolean;
  isGeneratingDailySnapshot: boolean;
  isGeneratingMtdSnapshot: boolean;
  generateDailySnapshot: () => Promise<void>;
  generateMtdSnapshot: () => Promise<void>;
  generateAndSendDailySnapshot: () => Promise<void>;
  generateAndSendMtdSnapshot: () => Promise<void>;
  isMtdEmailConfigured: boolean;
};

const BrandenSnapshotActionsContext = createContext<BrandenSnapshotActions | null>(null);

export function BrandenSnapshotActionsProvider({ children, value }: { children: ReactNode; value: BrandenSnapshotActions }) {
  return <BrandenSnapshotActionsContext.Provider value={value}>{children}</BrandenSnapshotActionsContext.Provider>;
}

export function useBrandenSnapshotActions() {
  const value = useContext(BrandenSnapshotActionsContext);
  if (!value) throw new Error("Snapshot actions must be used within the Branden journal layout.");
  return value;
}
