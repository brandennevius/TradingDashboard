export type TraderUser = {
  id: string;
  name: string;
  readOnly?: boolean;
  journalOwnerId?: string;
};

export type MonthlyReport = {
  id: string;
  userId: string;
  month: string;
  accountSize: number;
  totalReturn: number;
  percentReturn: number;
  netPnl: number;
  totalPayouts: number;
  totalTrades: number;
  winRate: number;
  avgR: number;
  totalR: number;
  avgWinR: number;
  avgLossR: number;
  avgWin: number;
  avgLoss: number;
  avgRisk: number;
  currentRiskPercent: number;
  expectedValueR: number;
  sharpeRatio: number;
  avgTradeLength: number;
  avgSwingLength: number;
  longestWinStreak: number;
  longestLossStreak: number;
  notes: string;
  createdAt: string;
  updatedAt: string;
};

export type MonthlyReportInput = Omit<MonthlyReport, "id" | "createdAt" | "updatedAt">;

export type TradeSide = "LONG" | "SHORT";
export type TradeStatus = "OPEN" | "WIN" | "LOSS" | "BREAKEVEN";
export type ChecklistInputType = "boolean" | "points";

export type TradeChecklistItem = {
  id: string;
  criteria: string;
  points: number;
  met: boolean;
  score?: number;
  inputType?: ChecklistInputType;
  groupName?: string;
  importTagKey?: string;
  importTagValue?: string;
};

export type SetupTemplateCriterion = {
  id: string;
  criteria: string;
  points: number;
  inputType: ChecklistInputType;
  importTagKey?: string;
  importTagValue?: string;
};

export type SetupChecklistGroup = {
  id: string;
  name: string;
  criteria: SetupTemplateCriterion[];
};

export type ChecklistGradeBand = {
  id: string;
  label: string;
  minScore: number;
  maxScore: number | null;
};

export type SetupChecklistTemplate = {
  id: string;
  setupName: string;
  description: string;
  knowledgeSources?: SetupKnowledgeSource[];
  strategyExamples?: SetupStrategyExample[];
  gradeBands: ChecklistGradeBand[];
  criteria: SetupTemplateCriterion[];
  groups: SetupChecklistGroup[];
};

export type SetupKnowledgeSource = {
  id: string;
  title: string;
  sourceType: "notes" | "resource" | "document";
  url: string;
  content: string;
  chunks?: SetupKnowledgeChunk[];
  active?: boolean;
  createdAt: string;
  updatedAt: string;
};

export type SetupKnowledgeChunk = {
  id: string;
  title: string;
  content: string;
  order: number;
};

export type SetupStrategyExampleQuality = "ideal" | "good" | "failed" | "bad" | "cautionary";

export type SetupStrategyExample = {
  id: string;
  symbol: string;
  setupType: string;
  quality: SetupStrategyExampleQuality;
  outcome: string;
  source: string;
  sourceUrl: string;
  notes: string;
  screenshots: string[];
  active?: boolean;
  createdAt: string;
  updatedAt: string;
};

export type TradeLogEntry = {
  id: string;
  userId: string;
  importSource: string;
  importRowKey: string;
  symbol: string;
  side: TradeSide;
  status: TradeStatus;
  entryDate: string;
  exitDate: string;
  openTime: string;
  closeTime: string;
  avgEntry: number;
  exitPrice: number;
  stopPrice: number;
  takeProfitPrice: number;
  shares: number;
  commission: number;
  usedMargin: number;
  risk: number;
  pnl: number;
  rMultiple: number;
  returnPercent: number;
  daysInTrade: number;
  setupTags: string[];
  mistakeTags: string[];
  customTags: string[];
  manualGrade: string;
  portfolioTag: string;
  emotion: string;
  tradeQuality: string;
  checklistItems: TradeChecklistItem[];
  notes: string;
  reviewSections?: TradeReviewSections;
  screenshots: string[];
  chartLinks: string[];
  executions: TradeExecution[];
  hidden: boolean;
  groupId: string;
  groupRole: "none" | "parent" | "child";
  createdAt: string;
  updatedAt: string;
};

export type TradeReviewSections = {
  setup: string;
  entry: string;
  exit: string;
  didRight: string;
  didWrong: string;
  general: string;
};

export type TradeExecution = {
  id: string;
  type: "ENTRY" | "EXIT";
  date: string;
  time: string;
  side: TradeSide;
  shares: number;
  price: number;
  pnl: number;
  commission: number;
  source: string;
  sourceKey: string;
};

export type WatchlistItem = {
  id: string;
  symbol: string;
  side: TradeSide;
  setupTag: string;
  setupGrade: string;
  checklistItems: TradeChecklistItem[];
  plannedEntry: number;
  stopPrice: number;
  takeProfitPrice: number;
  entryCriteria: string;
  entryNotes: string;
  invalidation: string;
  notes: string;
  screenshots: string[];
  chartLinks: string[];
  aiReview?: Record<string, unknown>;
  createdAt: string;
  updatedAt: string;
};

export type WeeklyWatchlist = {
  id: string;
  userId: string;
  weekKey: string;
  year: number;
  weekNumber: number;
  startDate: string;
  endDate: string;
  title: string;
  items: WatchlistItem[];
  createdAt: string;
  updatedAt: string;
};

export type TradeLogInput = Omit<TradeLogEntry, "id" | "createdAt" | "updatedAt" | "hidden" | "groupId" | "groupRole" | "executions" | "reviewSections"> & {
  hidden?: boolean;
  groupId?: string;
  groupRole?: TradeLogEntry["groupRole"];
  executions?: TradeExecution[];
  reviewSections?: TradeReviewSections;
};

export type MarketCycleEntry = {
  id: string;
  userId: string;
  date: string;
  trendDay: number;
  phase: string;
  notes: string;
  createdAt: string;
  updatedAt: string;
};

export type MarketCycleEntryInput = Omit<MarketCycleEntry, "id" | "createdAt" | "updatedAt">;

export type FeedbackTicketKind = "BUG" | "FEATURE";
export type FeedbackTicketStatus = "OPEN" | "IN_PROGRESS" | "COMPLETED";
export type FeedbackMessageAuthor = "CAM" | "ADMIN";

export type FeedbackTicketMessage = {
  id: string;
  author: FeedbackMessageAuthor;
  body: string;
  screenshots: string[];
  createdAt: string;
};

export type FeedbackTicket = {
  id: string;
  kind: FeedbackTicketKind;
  status: FeedbackTicketStatus;
  title: string;
  summary: string;
  details: string;
  expectedBehavior: string;
  reproductionSteps: string;
  businessValue: string;
  screenshots: string[];
  submittedBy: string;
  source: string;
  messages: FeedbackTicketMessage[];
  resolutionNotes: string;
  createdAt: string;
  updatedAt: string;
  completedAt: string;
};
