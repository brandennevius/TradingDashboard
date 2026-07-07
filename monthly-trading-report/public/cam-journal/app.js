const STORAGE_KEY = 'tradeJournalTradesV2';
const SETUPS_KEY = 'tradeJournalSetupsV1';
const PORTFOLIOS_KEY = 'tradeJournalPortfoliosV1';
const TAGS_KEY = 'tradeJournalTagsV1';
const REVIEWS_KEY = 'tradeJournalMonthlyReviewsV1';
const WATCHLIST_KEY = 'tradeJournalWatchlistV1';
const LEGACY_TRADE_KEYS = ['tradeJournalTradesV2','tradeJournalTradesV1','tradeJournalTrades'];
const REMOTE_STATE_ENDPOINT = '/api/cam-journal';
let journalReadOnly = false;
let remoteSaveTimer = null;
let currentSessionUser = null;
let feedbackPollTimer = null;
let localCacheWarningShown = false;
let remoteStateLoaded = false;
let remoteSaveBlockedNoticeShown = false;
let bootingJournal = true;
function loadJson(key, fallback) {
  try {
    const raw = localStorage.getItem(key);
    return raw ? JSON.parse(raw) : fallback;
  } catch (err) {
    console.warn('Could not load', key, err);
    return fallback;
  }
}
function loadTradesFromStorage() {
  for (const key of LEGACY_TRADE_KEYS) {
    const data = loadJson(key, null);
    if (Array.isArray(data) && data.length) return data;
  }
  return [];
}
let portfolios = loadJson(PORTFOLIOS_KEY, []);
let tags = loadJson(TAGS_KEY, { secondary: [], mistakes: [] });
let monthlyReviews = loadJson(REVIEWS_KEY, {});
let watchlistItems = loadJson(WATCHLIST_KEY, []);
let setups = loadJson(SETUPS_KEY, []);
let gradingSetup = null;
let currentSetupScreenshots = [];
let feedbackScreenshots = [];
let feedbackReplyScreenshots = [];
let feedbackTickets = [];
let activeFeedbackTicketId = '';
let trades = loadTradesFromStorage();
let selectedTradeIds = new Set();

const $ = (id) => document.getElementById(id);
const form = $('tradeForm');
let loadingTradeDialog = false;
let tradeDialogOriginalTrade = null;
let tradeDialogSaved = false;
let tradeDialogDeleted = false;
const tradeList = $('tradeList');
let tradeSort = { key: 'date', dir: 'desc' };
let breakdownSort = { key: 'setup', dir: 'asc' };
let gradeReportSort = { key: 'grade', dir: 'asc' };
let monthlyExpectancySort = { key: 'avgR', dir: 'desc' };
let playbookPage = 1;
const PLAYBOOK_PAGE_SIZE = 24;
const GRADE_ORDER = ['A+','A','A-','B+','B','B-','C+','C','Ungraded'];
function gradeRank(value) { const g = String(value || 'Ungraded').trim() || 'Ungraded'; const i = GRADE_ORDER.indexOf(g); return i === -1 ? 999 + g.charCodeAt(0) : i; }
function compareMaybeGrade(av, bv, key, dir) { if (key === 'grade') return (gradeRank(av) - gradeRank(bv)) * dir; if (typeof av === 'number' || typeof bv === 'number') return (Number(av)-Number(bv))*dir; return String(av).localeCompare(String(bv))*dir; }

document.querySelectorAll('.navBtn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.navBtn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
    btn.classList.add('active');
    $(btn.dataset.view).classList.add('active');
  });
});

function currentJournalState() {
  return { trades, setups, portfolios, tags, monthlyReviews, watchlistItems };
}

function jsonByteSize(value) {
  return new Blob([JSON.stringify(value)]).size;
}

function tradesWithoutCachedScreenshots() {
  return trades.map(trade => ({ ...trade, screenshots: [] }));
}

function setupsWithoutCachedScreenshots() {
  return setups.map(setup => ({
    ...setup,
    screenshots: [],
    versions: Array.isArray(setup.versions)
      ? setup.versions.map(version => ({ ...version, screenshots: [] }))
      : setup.versions
  }));
}

function saveLocalCache(key, value, fallbackValue) {
  try {
    localStorage.setItem(key, JSON.stringify(value));
    return true;
  } catch (error) {
    console.warn(`Local cache limit reached for ${key}; saving a lightweight cache instead.`, error);
    try {
      localStorage.setItem(key, JSON.stringify(fallbackValue));
    } catch (fallbackError) {
      console.warn(`Lightweight local cache also failed for ${key}.`, fallbackError);
    }
    if (!localCacheWarningShown) {
      localCacheWarningShown = true;
      showImportStatus('<strong>Using shared journal storage for screenshots.</strong><br><span class="small">This browser has reached its offline cache limit, so screenshot copies will be loaded from the shared database instead.</span>');
    }
    return false;
  }
}

async function saveRemoteStateNow() {
  if (journalReadOnly || !remoteStateLoaded) {
    if (!journalReadOnly && !remoteStateLoaded && !remoteSaveBlockedNoticeShown) {
      remoteSaveBlockedNoticeShown = true;
      showImportStatus('<strong>Shared journal changes are temporarily blocked.</strong><br><span class="small">The complete shared journal did not load, so this browser will not overwrite it with an incomplete cache. Reload after the shared journal is available.</span>');
    }
    return false;
  }
  const state = currentJournalState();
  const stateBytes = jsonByteSize(state);
  if (stateBytes > 3000000) {
    throw new Error(`Journal state is still too large (${(stateBytes / 1000000).toFixed(2)} MB). Embedded screenshots must be migrated first.`);
  }
  const response = await fetch(REMOTE_STATE_ENDPOINT, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ state })
  });
  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    throw new Error(data.error || `Shared journal save failed (${response.status}).`);
  }
  return true;
}

function queueRemoteSave() {
  if (bootingJournal) return;
  if (journalReadOnly || !remoteStateLoaded) {
    void saveRemoteStateNow().catch(error => console.error('Remote journal save blocked', error));
    return;
  }
  clearTimeout(remoteSaveTimer);
  remoteSaveTimer = setTimeout(async () => {
    try {
      await saveRemoteStateNow();
    } catch (err) {
      console.error('Remote journal save failed', err);
      showImportStatus(`<strong>Could not save the shared journal.</strong><br><span class="small">${escapeHtml(err?.message || String(err))}</span>`);
    }
  }, 350);
}

async function flushRemoteSave() {
  clearTimeout(remoteSaveTimer);
  remoteSaveTimer = null;
  return saveRemoteStateNow();
}

function save() {
  saveLocalCache(STORAGE_KEY, trades, tradesWithoutCachedScreenshots());
  queueRemoteSave();
}
function saveSetups() {
  saveLocalCache(SETUPS_KEY, setups, setupsWithoutCachedScreenshots());
  queueRemoteSave();
}
function savePortfolios() {
  saveLocalCache(PORTFOLIOS_KEY, portfolios, portfolios);
  queueRemoteSave();
}
function saveTags() {
  saveLocalCache(TAGS_KEY, tags, tags);
  queueRemoteSave();
}
function saveMonthlyReviews() {
  saveLocalCache(REVIEWS_KEY, monthlyReviews || {}, monthlyReviews || {});
  queueRemoteSave();
}
function saveWatchlist() {
  saveLocalCache(WATCHLIST_KEY, watchlistItems || [], watchlistItems || []);
  queueRemoteSave();
}


function stableId(prefix = 'id') {
  if (window.crypto && crypto.randomUUID) return crypto.randomUUID();
  return prefix + '-' + Date.now().toString(36) + '-' + Math.random().toString(36).slice(2, 10);
}

function tagRecord(input, type = 'tag') {
  if (input && typeof input === 'object') {
    const id = String(input.id || '').trim() || stableId(type);
    const name = String(input.name || '').trim();
    return name ? { id, name } : null;
  }

  const name = String(input || '').trim();
  return name ? { id: stableId(type), name } : null;
}

function normalizeTagList(list, type) {
  const byName = new Map();
  (Array.isArray(list) ? list : []).forEach(item => {
    const record = tagRecord(item, type);
    if (!record) return;
    const key = record.name.toLowerCase();
    if (!byName.has(key)) byName.set(key, record);
  });
  return Array.from(byName.values()).sort((a, b) => a.name.localeCompare(b.name));
}

function tagNames(list) {
  return normalizeTagList(list, 'tag').map(item => item.name);
}

function ensureStableDataIds() {
  let changedTrades = false;
  trades = (trades || []).map(t => {
    let next = { ...t };
    if (!next.id) { next.id = stableId('trade'); changedTrades = true; }
    if (next.setup && !next.setupId) {
      const setup = getSetupByName(next.setup);
      if (setup?.id) { next.setupId = setup.id; changedTrades = true; }
    }
    return next;
  });

  let changedSetups = false;
  setups = (setups || []).map(setup => {
    let next = { ...setup };
    if (!next.id) { next.id = stableId('setup'); changedSetups = true; }
    next = normalizeSetupVersions(next);
    if (!Array.isArray(next.versions) || !next.versions.length) {
      next.versions = [setupSnapshot(next, next.currentVersion || 1)];
      changedSetups = true;
    }
    next.versions = next.versions.map(v => ({
      ...v,
      setupId: v.setupId || next.id,
      setupName: next.name,
      label: `Version ${Number(v.version || 1)}`
    }));
    return next;
  });

  let changedPortfolios = false;
  portfolios = (portfolios || []).map(p => {
    if (typeof p === 'string') { changedPortfolios = true; return { id: stableId('portfolio'), name: p, description: '' }; }
    if (!p.id) { changedPortfolios = true; return { ...p, id: stableId('portfolio') }; }
    return p;
  });

  normalizeTags();
  if (changedTrades) save();
  if (changedSetups) saveSetups();
  if (changedPortfolios) savePortfolios();
  saveTags();
}

function buildFullBackupPayload() {
  ensureStableDataIds();
  return {
    app: 'Trade Journal',
    backupVersion: 1,
    exportedAt: new Date().toISOString(),
    note: 'Full backup includes trades, setups, setup versions, portfolios, tags, screenshots, watchlist items, and frozen grading snapshots.',
    storageKeys: { trades: STORAGE_KEY, setups: SETUPS_KEY, portfolios: PORTFOLIOS_KEY, tags: TAGS_KEY, watchlist: WATCHLIST_KEY },
    data: {
      trades: deepClone(trades || []),
      setups: deepClone(setups || []),
      portfolios: deepClone(portfolios || []),
      tags: deepClone(tags || { secondary: [], mistakes: [] }),
      monthlyReviews: deepClone(monthlyReviews || {}),
      watchlistItems: deepClone(watchlistItems || [])
    }
  };
}

async function screenshotForBackup(shot) {
  const source = screenshotSrc(shot);
  if (!source || source.startsWith('data:image/')) return shot;
  const response = await fetch(source);
  if (!response.ok) throw new Error(`Could not include screenshot in backup (${response.status}).`);
  const dataUrl = await blobToDataUrl(await response.blob());
  return typeof shot === 'string' ? dataUrl : { ...shot, dataUrl, url: '' };
}

async function downloadFullBackup() {
  const payload = buildFullBackupPayload();
  for (const trade of payload.data.trades || []) {
    trade.screenshots = await Promise.all((trade.screenshots || []).map(screenshotForBackup));
  }
  for (const setup of payload.data.setups || []) {
    setup.screenshots = await Promise.all((setup.screenshots || []).map(screenshotForBackup));
    for (const version of setup.versions || []) {
      version.screenshots = await Promise.all((version.screenshots || []).map(screenshotForBackup));
    }
  }
  const date = new Date().toISOString().slice(0,10);
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `trade-journal-full-backup-${date}.json`;
  a.click();
  URL.revokeObjectURL(url);
}

function mergeById(existing, incoming, fallbackKeys = []) {
  const output = [...(existing || [])];
  const findIndex = (item) => output.findIndex(cur => {
    if (item.id && cur.id && item.id === cur.id) return true;
    return fallbackKeys.some(k => item[k] && cur[k] && String(item[k]).toLowerCase() === String(cur[k]).toLowerCase());
  });
  (incoming || []).forEach(item => {
    const idx = findIndex(item);
    if (idx >= 0) output[idx] = { ...output[idx], ...item };
    else output.push(item);
  });
  return output;
}

async function importFullBackupPayload(payload) {
  if (!remoteStateLoaded) throw new Error('The complete shared journal must load before a backup can be restored.');
  const rawData = payload?.data || payload;
  if (!rawData || !Array.isArray(rawData.trades) || !Array.isArray(rawData.setups)) throw new Error('This does not look like a Trade Journal full backup JSON file.');
  const data = await migrateEmbeddedScreenshots(rawData);
  const mode = confirm('Import Full Backup\n\nChoose OK to MERGE. Merge adds new records and updates matching records by stable ID, so it will not create duplicates or wipe unrelated trades.\nChoose Cancel to REPLACE all current journal data with this backup.') ? 'merge' : 'replace';
  if (mode === 'replace') {
    if (!confirm('Replace current journal data with this backup? This will overwrite the data currently saved in this browser.')) return false;
    trades = deepClone(data.trades || []);
    setups = deepClone(data.setups || []);
    portfolios = deepClone(data.portfolios || []);
    tags = deepClone(data.tags || { secondary: [], mistakes: [] });
    monthlyReviews = deepClone(data.monthlyReviews || {});
    watchlistItems = deepClone(data.watchlistItems || []);
  } else {
    trades = mergeById(trades, deepClone(data.trades || []), ['brokerKey','importKey']);
    setups = mergeById(setups, deepClone(data.setups || []), ['name']);
    portfolios = mergeById(portfolios, deepClone(data.portfolios || []), ['name']);
    const incomingTags = data.tags || { secondary: [], mistakes: [] };
    monthlyReviews = { ...(monthlyReviews || {}), ...(data.monthlyReviews || {}) };
    watchlistItems = mergeById(watchlistItems, deepClone(data.watchlistItems || []));
    normalizeTags();
    tags.secondary = normalizeTagList([...(tags.secondary || []), ...((incomingTags.secondary || []))], 'secondary-tag');
    tags.mistakes = normalizeTagList([...(tags.mistakes || []), ...((incomingTags.mistakes || []))], 'mistake-tag');
  }
  ensureStableDataIds();
  normalizeAllSetupVersions();
  save(); saveSetups(); savePortfolios(); saveTags(); saveMonthlyReviews(); saveWatchlist();
  await flushRemoteSave();
  selectedTradeIds.clear();
  renderAll();
  alert('Full backup imported and saved to the shared journal successfully.');
  return true;
}

function money(value) {
  const number = Number(value || 0);
  return number.toLocaleString(undefined, { style: 'currency', currency: 'USD' });
}

function numberOrBlank(value) {
  if (value === '' || value === null || value === undefined) return '';
  const n = Number(value);
  return Number.isFinite(n) ? n : '';
}

function price2(value) {
  if (value === '' || value === null || value === undefined) return '-';
  const n = Number(value);
  return Number.isFinite(n) ? n.toFixed(2) : fmt(value);
}

function price4(value) {
  if (value === '' || value === null || value === undefined) return '';
  const n = Number(value);
  return Number.isFinite(n) ? n.toFixed(4) : value;
}

function cleanTime(value) {
  if (!value) return '';
  const text = String(value).trim();
  const match = text.match(/(\d{1,2}):(\d{2})(?::(\d{2}))?/);
  if (!match) return text;
  const h = match[1].padStart(2, '0');
  const m = match[2];
  const sec = match[3] || '00';
  return `${h}:${m}:${sec}`;
}


function formatDisplayDate(value) {
  if (!value) return '—';
  const text = String(value).trim();
  const iso = text.match(/^(\d{4})-(\d{2})-(\d{2})$/);
  if (iso) return `${iso[2]}/${iso[3]}/${iso[1]}`;
  const slash = text.match(/^(\d{1,2})\/(\d{1,2})\/(\d{4})$/);
  if (slash) return `${slash[1].padStart(2,'0')}/${slash[2].padStart(2,'0')}/${slash[3]}`;
  const d = new Date(text);
  if (!Number.isNaN(d.getTime())) {
    return `${String(d.getMonth()+1).padStart(2,'0')}/${String(d.getDate()).padStart(2,'0')}/${d.getFullYear()}`;
  }
  return text;
}


function formatTradeLogDate(value) {
  if (!value) return '—';
  const text = String(value).trim();
  const iso = text.match(/^(\d{4})-(\d{2})-(\d{2})$/);
  if (iso) return `${iso[2]}-${iso[3]}-${iso[1]}`;
  const slash = text.match(/^(\d{1,2})\/(\d{1,2})\/(\d{4})$/);
  if (slash) return `${slash[1].padStart(2,'0')}-${slash[2].padStart(2,'0')}-${slash[3]}`;
  const d = new Date(text);
  if (!Number.isNaN(d.getTime())) return `${String(d.getMonth()+1).padStart(2,'0')}-${String(d.getDate()).padStart(2,'0')}-${d.getFullYear()}`;
  return text;
}

function formatDisplayTime(value) {
  if (!value) return '—';
  const clean = cleanTime(value);
  const match = String(clean).match(/^(\d{1,2}):(\d{2})(?::(\d{2}))?/);
  if (!match) return String(value);
  let h = Number(match[1]);
  const m = match[2];
  const sec = match[3] || '00';
  const suffix = h >= 12 ? 'pm' : 'am';
  h = h % 12 || 12;
  return `${h}:${m}:${sec} ${suffix}`;
}

function tradeDateValue(t) { return t.date || ''; }

function getReportFilteredTrades() {
  const start = $('reportStartDate')?.value || '';
  const end = $('reportEndDate')?.value || '';
  const tickers = getSelectValues('reportTicker');
  const setupsSelected = getSelectValues('reportSetup');
  const portfoliosSelected = getSelectValues('reportPortfolio');
  const direction = $('reportDirection')?.value || '';
  const resultsSelected = getSelectValues('reportResult');
  const gradesSelected = getSelectValues('reportGrade');
  const secondarySelected = getSelectValues('reportSecondaryTag');
  const mistakeSelected = getSelectValues('reportMistakeTag');
  return trades.filter(t => {
    const d = tradeDateValue(t);
    if (start && d && d < start) return false;
    if (end && d && d > end) return false;
    if (!valueMatchesAny((t.ticker || t.symbol || '').toUpperCase(), tickers)) return false;
    if (!valueMatchesAny(t.setup || '', setupsSelected)) return false;
    if (!valueMatchesAny(t.portfolioTag || '', portfoliosSelected)) return false;
    if (direction && (t.direction || '') !== direction) return false;
    if (!valueMatchesAny(getWinLoss(t), resultsSelected)) return false;
    if (gradesSelected.length) {
      const g = String(t.grade || '').trim() || 'Ungraded';
      if (!gradesSelected.includes(g)) return false;
    }
    if (!valueMatchesAny(t.secondaryTag || '', secondarySelected)) return false;
    if (!valueMatchesAny(t.mistakeTag || '', mistakeSelected)) return false;
    return true;
  });
}

function average(nums) {
  const clean = nums.filter(n => Number.isFinite(n));
  return clean.length ? clean.reduce((a,b)=>a+b,0) / clean.length : 0;
}

function maxConsecutiveLosses(list) {
  const ordered = list.slice().sort((a,b)=>((a.date||'')+(a.entryTime||'')).localeCompare((b.date||'')+(b.entryTime||'')));
  let current = 0, max = 0;
  ordered.forEach(t => {
    if (getWinLoss(t) === 'Loss') { current++; max = Math.max(max, current); }
    else if (getWinLoss(t) === 'Win' || getWinLoss(t) === 'Breakeven') current = 0;
  });
  return max;
}

function maxConsecutiveWins(list) {
  const ordered = list.slice().sort((a,b)=>((a.date||'')+(a.entryTime||'')).localeCompare((b.date||'')+(b.entryTime||'')));
  let current = 0, max = 0;
  ordered.forEach(t => {
    if (getWinLoss(t) === 'Win') { current++; max = Math.max(max, current); }
    else if (getWinLoss(t) === 'Loss' || getWinLoss(t) === 'Breakeven') current = 0;
  });
  return max;
}

function parseTradeDateTime(dateStr, timeStr) {
  if (!dateStr) return null;
  const clean = cleanTime(timeStr || '09:30:00') || '09:30:00';
  const d = new Date(`${dateStr}T${clean}`);
  return Number.isNaN(d.getTime()) ? null : d;
}

function marketMinuteOfDay(d) {
  const mins = d.getHours() * 60 + d.getMinutes();
  return Math.max(0, Math.min(390, mins - 570)); // 9:30am to 4:00pm = 390 minutes
}

function businessDaysBetween(start, end) {
  const a = new Date(start.getFullYear(), start.getMonth(), start.getDate());
  const b = new Date(end.getFullYear(), end.getMonth(), end.getDate());
  let count = 0;
  for (let cur = new Date(a); cur < b; cur.setDate(cur.getDate() + 1)) {
    const day = cur.getDay();
    if (day !== 0 && day !== 6) count++;
  }
  return count;
}

function marketHoldMinutes(t) {
  const open = parseTradeDateTime(t.date, t.entryTime);
  const close = parseTradeDateTime(t.exitDate || t.date, t.exitTime || t.entryTime);
  if (!open || !close || close < open) return null;
  const sameDate = open.toDateString() === close.toDateString();
  if (sameDate) return Math.max(0, marketMinuteOfDay(close) - marketMinuteOfDay(open));
  // Treat each market date crossed as one trading day, then add the exit-day time from the open.
  return businessDaysBetween(open, close) * 390 + marketMinuteOfDay(close);
}

function formatMarketDuration(minutes) {
  if (!Number.isFinite(minutes) || minutes < 0) return '—';
  const total = Math.round(minutes);
  const days = Math.floor(total / 390);
  const rem = total % 390;
  const hours = Math.floor(rem / 60);
  const mins = rem % 60;
  if (days > 0) return `${days} day${days === 1 ? '' : 's'} ${hours} hour${hours === 1 ? '' : 's'} and ${mins} minute${mins === 1 ? '' : 's'}`;
  return `${hours} hour${hours === 1 ? '' : 's'} and ${mins} minute${mins === 1 ? '' : 's'}`;
}

function averageHoldTime(list) {
  const vals = list.map(marketHoldMinutes).filter(v => Number.isFinite(v));
  return vals.length ? average(vals) : 0;
}


function sortedClosedTradesForCharts(list) {
  return list.filter(t => (t.status || 'Closed') === 'Closed')
    .slice()
    .sort((a,b)=>((a.exitDate || a.date || '') + (a.exitTime || a.entryTime || '')).localeCompare((b.exitDate || b.date || '') + (b.exitTime || b.entryTime || '')));
}

function cumulativeSeries(list, mode) {
  const ordered = sortedClosedTradesForCharts(list);
  const series = [];
  let totalPL = 0, totalR = 0, wins = 0, closed = 0;
  const winRs = [], lossRs = [];
  ordered.forEach(t => {
    closed++;
    const rDefined = validRValue(t);
    const r = rDefined === null ? 0 : rDefined;
    totalPL += Number(t.pl || 0);
    totalR += r;
    if (getWinLoss(t) === 'Win') { wins++; if (rDefined !== null) winRs.push(r); }
    if (getWinLoss(t) === 'Loss' && rDefined !== null) lossRs.push(r);
    let value = 0;
    if (mode === 'pl') value = totalPL;
    if (mode === 'winRate') value = closed ? wins / closed * 100 : 0;
    if (mode === 'totalR') value = totalR;
    if (mode === 'avgWinR') value = average(winRs);
    if (mode === 'avgLossR') value = average(lossRs);
    if (mode === 'winLossRatio') { const aw = average(winRs); const al = Math.abs(average(lossRs)); value = al > 0 ? aw / al : 0; }
    const ticker = String(t.ticker || 'N/A').toUpperCase();
    series.push({
      value,
      label: `${formatDisplayDate(t.exitDate || t.date)} · ${ticker}`,
      detail: `Ticker: ${ticker} · Trade PnL ${money(t.pl || 0)} · R ${Number(r || 0).toFixed(2)}`
    });
  });
  return series;
}

function axisValue(value, opts = {}) {
  const n = Number(value || 0);
  if (opts.axis === 'money' || opts.prefix === '$') return money(n).replace('.00','');
  if (opts.suffix === '%') return `${n.toFixed(0)}%`;
  if (opts.suffix === 'R') return `${n.toFixed(1)}R`;
  return n.toFixed(opts.decimals ?? 1);
}

function smoothPath(coords) {
  if (!coords.length) return '';
  if (coords.length === 1) return `M ${coords[0].x} ${coords[0].y}`;
  let d = `M ${coords[0].x.toFixed(1)} ${coords[0].y.toFixed(1)}`;
  for (let i = 0; i < coords.length - 1; i++) {
    const p0 = coords[Math.max(0, i - 1)];
    const p1 = coords[i];
    const p2 = coords[i + 1];
    const p3 = coords[Math.min(coords.length - 1, i + 2)];
    const tension = 0.18;
    const cp1x = p1.x + (p2.x - p0.x) * tension;
    const cp1y = p1.y + (p2.y - p0.y) * tension;
    const cp2x = p2.x - (p3.x - p1.x) * tension;
    const cp2y = p2.y - (p3.y - p1.y) * tension;
    d += ` C ${cp1x.toFixed(1)} ${cp1y.toFixed(1)}, ${cp2x.toFixed(1)} ${cp2y.toFixed(1)}, ${p2.x.toFixed(1)} ${p2.y.toFixed(1)}`;
  }
  return d;
}

function chartDisplayValue(value, opts = {}) {
  const n = Number(value || 0);
  if (opts.axis === 'money' || opts.prefix === '$') return money(n);
  if (opts.suffix === '%') return `${n.toFixed(1)}%`;
  if (opts.suffix === 'R') return `${n.toFixed(2)}R`;
  return n.toFixed(opts.decimals ?? 2);
}

function installChartHover(svg, coords, opts, width, height, padL, padR, padT, padB) {
  if (!svg || !coords.length) return;
  let hover = svg.querySelector('.hover-layer');
  if (!hover) return;
  const show = (evt) => {
    const rect = svg.getBoundingClientRect();
    const xSvg = (evt.clientX - rect.left) / rect.width * width;
    let nearest = coords[0];
    coords.forEach(p => { if (Math.abs(p.x - xSvg) < Math.abs(nearest.x - xSvg)) nearest = p; });
    const tooltipW = opts.largeTooltip ? 420 : 340;
    const tooltipH = opts.largeTooltip ? 134 : 116;
    let tx = nearest.x + 14;
    if (tx + tooltipW > width - 8) tx = nearest.x - tooltipW - 14;
    if (tx < 8) tx = 8;
    let ty = nearest.y - tooltipH - 14;
    if (ty < 8) ty = nearest.y + 18;
    if (ty + tooltipH > height - 8) ty = height - tooltipH - 8;
    const dateText = escapeHtml((nearest.label || '').split(' · ')[0] || 'Data point');
    const valueText = `${escapeHtml(opts.tooltipLabel || 'Value')}: ${escapeHtml(chartDisplayValue(nearest.value, opts))}`;
    const detailText = escapeHtml(nearest.detail || '');
    hover.innerHTML = `
      <line x1="${nearest.x.toFixed(1)}" y1="${padT}" x2="${nearest.x.toFixed(1)}" y2="${height-padB}" class="chart-crosshair" />
      <circle cx="${nearest.x.toFixed(1)}" cy="${nearest.y.toFixed(1)}" r="5" class="chart-hover-point" />
      <g class="chart-tooltip" transform="translate(${tx.toFixed(1)},${ty.toFixed(1)})">
        <rect width="${tooltipW}" height="${tooltipH}" rx="10"></rect>
        <text x="16" y="31" class="tip-date">${dateText}</text>
        <text x="16" y="64" class="tip-value">${valueText}</text>
        <text x="16" y="94" class="tip-detail">${detailText}</text>
      </g>`;
  };
  svg.onmousemove = show;
  svg.onmouseleave = () => { hover.innerHTML = ''; };
}


function splitAreaPaths(coords, baseY) {
  const pos = [], neg = [];
  if (!coords || coords.length < 2) return { pos, neg };
  const addSeg = (bucket, a, b) => {
    if (!a || !b) return;
    bucket.push(`M ${a.x.toFixed(1)} ${a.y.toFixed(1)} L ${b.x.toFixed(1)} ${b.y.toFixed(1)} L ${b.x.toFixed(1)} ${baseY.toFixed(1)} L ${a.x.toFixed(1)} ${baseY.toFixed(1)} Z`);
  };
  for (let i = 0; i < coords.length - 1; i++) {
    const a = coords[i], b = coords[i+1];
    const av = Number(a.value || 0), bv = Number(b.value || 0);
    if (av >= 0 && bv >= 0) { addSeg(pos, a, b); continue; }
    if (av <= 0 && bv <= 0) { addSeg(neg, a, b); continue; }
    const t = (0 - av) / (bv - av);
    const cross = { value: 0, x: a.x + (b.x - a.x) * t, y: baseY };
    if (av >= 0) { addSeg(pos, a, cross); addSeg(neg, cross, b); }
    else { addSeg(neg, a, cross); addSeg(pos, cross, b); }
  }
  return { pos, neg };
}

function renderMiniChart(svgId, series, opts = {}) {
  const svg = $(svgId);
  if (!svg) return;
  svg.innerHTML = '';
  const width = 860, height = 310, padL = 74, padR = 24, padT = 26, padB = 48;
  svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
  const raw = (series || []).map((item, index) => typeof item === 'number' ? {value:item, label:`Point ${index+1}`, detail:String(item)} : item);
  if (!raw.length) {
    svg.innerHTML = `<line x1="${padL}" y1="${height - padB}" x2="${width - padR}" y2="${height - padB}" class="chart-axis" />`;
    return;
  }
  const pointsData = raw.length === 1 ? [{value:0,label:'Start',detail:'Start'}, raw[0]] : raw;
  let min = Math.min(0, ...pointsData.map(p=>Number(p.value || 0)));
  let max = Math.max(0, ...pointsData.map(p=>Number(p.value || 0)));
  if (min === max) { min -= 1; max += 1; }
  const rangePad = (max - min) * 0.10;
  min -= rangePad; max += rangePad;
  const plotW = width - padL - padR;
  const plotH = height - padT - padB;
  const coords = pointsData.map((p, i) => {
    const x = padL + (i / Math.max(1, pointsData.length - 1)) * plotW;
    const y = padT + (1 - ((Number(p.value || 0) - min) / (max - min))) * plotH;
    return {...p, x, y};
  });
  const linePath = smoothPath(coords);
  const baseY = min < 0 && max > 0 ? padT + (1 - ((0 - min) / (max - min))) * plotH : height - padB;
  const areaPath = `${linePath} L ${coords[coords.length - 1].x.toFixed(1)} ${baseY.toFixed(1)} L ${coords[0].x.toFixed(1)} ${baseY.toFixed(1)} Z`;
  const clipId = `clip-${svgId}-${Math.random().toString(36).slice(2,8)}`;
  const positiveAreas = `<clipPath id="${clipId}-pos"><rect x="${padL}" y="${padT}" width="${plotW}" height="${Math.max(0, baseY-padT).toFixed(1)}"></rect></clipPath><path d="${areaPath}" class="chart-area positive-area" clip-path="url(#${clipId}-pos)"/>`;
  const negativeAreas = `<clipPath id="${clipId}-neg"><rect x="${padL}" y="${baseY.toFixed(1)}" width="${plotW}" height="${Math.max(0, height-padB-baseY).toFixed(1)}"></rect></clipPath><path d="${areaPath}" class="chart-area negative-area" clip-path="url(#${clipId}-neg)"/>`;
  const tickCount = 5;
  const ticks = Array.from({length: tickCount}, (_, i) => min + (max-min) * (i/(tickCount-1))).reverse();
  const grids = ticks.map(v => {
    const y = padT + (1 - ((v - min) / (max - min))) * plotH;
    return `<line x1="${padL}" y1="${y.toFixed(1)}" x2="${width-padR}" y2="${y.toFixed(1)}" class="chart-grid"/><text x="${padL-10}" y="${(y+4).toFixed(1)}" text-anchor="end" class="chart-ylabel">${escapeHtml(axisValue(v, opts))}</text>`;
  }).join('');
  const labelCount = Math.min(6, coords.length);
  const xLabs = [];
  for (let i = 0; i < labelCount; i++) {
    const idx = Math.round(i * (coords.length - 1) / Math.max(1, labelCount - 1));
    const p = coords[idx];
    xLabs.push(`<text x="${p.x.toFixed(1)}" y="${height-15}" text-anchor="middle" class="chart-xlabel">${escapeHtml((p.label || '').split(' · ')[0])}</text>`);
  }
  const zeroLine = `<line x1="${padL}" y1="${baseY.toFixed(1)}" x2="${width-padR}" y2="${baseY.toFixed(1)}" class="chart-axis"/>`;
  const hitDots = coords.map((p) => `<circle cx="${p.x.toFixed(1)}" cy="${p.y.toFixed(1)}" r="7" class="chart-dot"><title>${escapeHtml((p.label || 'Data point'))}\n${escapeHtml(chartDisplayValue(p.value, opts))}\n${escapeHtml(p.detail || '')}</title></circle>`).join('');
  svg.innerHTML = `${grids}${zeroLine}${positiveAreas}${negativeAreas}<path d="${linePath}" class="chart-line"/>${xLabs.join('')}<rect x="${padL}" y="${padT}" width="${plotW}" height="${plotH}" class="chart-hitbox"></rect>${hitDots}<g class="hover-layer"></g>`;
  installChartHover(svg, coords, opts, width, height, padL, padR, padT, padB);
}
function monthlyGradeBars(list) {
  const groups = new Map();
  sortedClosedTradesForCharts(list).forEach(t => {
    const month = (t.exitDate || t.date || '').slice(0,7) || 'Unknown';
    const r = Number(getRMultiple(t) || 0);
    groups.set(month, (groups.get(month) || 0) + r);
  });
  return Array.from(groups.entries()).map(([month, value]) => ({month, value}));
}

function monthlyTradeCountBars(list) {
  const groups = new Map();
  list.slice().sort((a,b)=>((a.date||'')+(a.entryTime||'')).localeCompare((b.date||'')+(b.entryTime||''))).forEach(t => {
    const month = (t.date || t.exitDate || '').slice(0,7) || 'Unknown';
    groups.set(month, (groups.get(month) || 0) + 1);
  });
  return Array.from(groups.entries()).map(([month, value]) => ({month, value}));
}

function setupGradeAvgRBars(list) {
  const groups = new Map();
  list.forEach(t => {
    const g = String(t.grade || 'Ungraded').trim() || 'Ungraded';
    if (!groups.has(g)) groups.set(g, []);
    groups.get(g).push(Number(getRMultiple(t) || 0));
  });
  const order = GRADE_ORDER;
  return Array.from(groups.entries())
    .sort((a,b)=>(order.indexOf(a[0]) === -1 ? 99 : order.indexOf(a[0])) - (order.indexOf(b[0]) === -1 ? 99 : order.indexOf(b[0])))
    .map(([month, values]) => ({month, value: average(values)}));
}

function renderBarChart(svgId, bars, opts = {}) {
  const svg = $(svgId); if (!svg) return;
  svg.innerHTML = '';
  const width = opts.width || 760, height = opts.height || 280, padL = 64, padR = 24, padT = 24, padB = 50;
  svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
  if (!bars.length) { svg.innerHTML = `<line x1="${padL}" y1="${height-padB}" x2="${width-padR}" y2="${height-padB}" class="chart-axis"/>`; return; }
  let min = Math.min(0, ...bars.map(b=>Number(b.value || 0)));
  let max = Math.max(0, ...bars.map(b=>Number(b.value || 0)));
  if (min === max) { min -= 1; max += 1; }
  const rangePad = (max - min) * 0.12;
  min -= rangePad; max += rangePad;
  const plotW = width - padL - padR;
  const plotH = height - padT - padB;
  const zeroY = padT + (1 - ((0 - min) / (max - min))) * plotH;
  const slot = plotW / bars.length;
  const barW = Math.max(16, Math.min(58, slot * .58));
  const ticks = Array.from({length: 5}, (_, i) => min + (max-min) * (i/4)).reverse();
  const grid = ticks.map(v=>{ const y=padT + (1 - ((v-min)/(max-min))) * plotH; return `<line x1="${padL}" y1="${y.toFixed(1)}" x2="${width-padR}" y2="${y.toFixed(1)}" class="chart-grid"/><text x="${padL-8}" y="${(y+4).toFixed(1)}" text-anchor="end" class="chart-ylabel">${escapeHtml(axisValue(v,opts))}</text>`; }).join('');
  const rects = bars.map((b,i)=>{
    const x = padL + i*slot + (slot-barW)/2;
    const yVal = padT + (1 - ((Number(b.value || 0) - min) / (max - min))) * plotH;
    const y = Math.min(yVal, zeroY);
    const h = Math.max(2, Math.abs(zeroY - yVal));
    const label = opts.rawLabels ? b.month : (b.month === 'Unknown' ? 'Unknown' : new Date(b.month + '-02').toLocaleDateString(undefined, {month:'short', year:'numeric'}));
    const valueText = chartDisplayValue(b.value, opts);
    return `<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${barW.toFixed(1)}" height="${h.toFixed(1)}" class="chart-bar"><title>${escapeHtml(label)}\n${escapeHtml(opts.tooltipLabel || 'Value')}: ${escapeHtml(valueText)}</title></rect><text x="${(x+barW/2).toFixed(1)}" y="${height-14}" text-anchor="middle" class="chart-xlabel">${escapeHtml(label)}</text>`;
  }).join('');
  svg.innerHTML = `${grid}<line x1="${padL}" y1="${zeroY.toFixed(1)}" x2="${width-padR}" y2="${zeroY.toFixed(1)}" class="chart-axis"/>${rects}`;
}

function updateAtrStats() {
  const entry = Number($('editEntry')?.value || 0);
  const stop = Number($('editStop')?.value || 0);
  const target = Number($('editTarget')?.value || 0);
  const atr = Number($('editStfAtr')?.value || 0);
  const stopPct = (entry > 0 && stop > 0 && atr > 0) ? (Math.abs(entry - stop) / atr * 100) : null;
  const targetPct = (entry > 0 && target > 0 && atr > 0) ? (Math.abs(target - entry) / atr * 100) : null;
  if ($('atrStopPct')) $('atrStopPct').textContent = stopPct === null ? '—' : `${stopPct.toFixed(1)}% of ATR`;
  if ($('atrTargetPct')) $('atrTargetPct').textContent = targetPct === null ? '—' : `${targetPct.toFixed(1)}% of ATR`;
}

function renderDashboardReports() {
  const filtered = getReportFilteredTrades();
  const closed = filtered.filter(t => (t.status || 'Closed') === 'Closed');
  const wins = closed.filter(t => getWinLoss(t) === 'Win');
  const losses = closed.filter(t => getWinLoss(t) === 'Loss');
  const totalR = filtered.reduce((sum, t) => sum + Number(getRMultiple(t) || 0), 0);
  const totalPL = filtered.reduce((sum, t) => sum + Number(t.pl || 0), 0);
  const avgRWin = average(wins.map(validRValue).filter(value => value !== null));
  const avgRLoss = average(losses.map(validRValue).filter(value => value !== null));
  const ratio = Math.abs(avgRLoss) > 0 ? avgRWin / Math.abs(avgRLoss) : 0;

  $('reportTradeCount').textContent = filtered.length;
  $('reportTotalPL').textContent = money(totalPL);
  $('reportWinRate').textContent = closed.length ? Math.round((wins.length / closed.length) * 100) + '%' : '0%';
  $('reportTotalR').textContent = totalR.toFixed(2) + 'R';
  $('reportAvgRWin').textContent = avgRWin.toFixed(2) + 'R';
  $('reportAvgRLoss').textContent = avgRLoss.toFixed(2) + 'R';
  $('reportWinLossRatio').textContent = ratio.toFixed(2);
  $('reportMaxWins').textContent = maxConsecutiveWins(filtered);
  $('reportMaxLosses').textContent = maxConsecutiveLosses(filtered);
  $('reportAvgTimeWins').textContent = formatMarketDuration(averageHoldTime(wins));
  $('reportAvgTimeLosses').textContent = formatMarketDuration(averageHoldTime(losses));
  renderMiniChart('chartTotalPL', cumulativeSeries(filtered, 'pl'), { prefix:'$', decimals:2, axis:'money', tooltipLabel:'Cumulative P&L' });
  renderMiniChart('chartWinRate', cumulativeSeries(filtered, 'winRate'), { suffix:'%', decimals:1, tooltipLabel:'Win Rate' });
  renderMiniChart('chartTotalR', cumulativeSeries(filtered, 'totalR'), { suffix:'R', decimals:2, tooltipLabel:'Cumulative R' });
  renderMiniChart('chartAvgRWin', cumulativeSeries(filtered, 'avgWinR'), { suffix:'R', decimals:2, tooltipLabel:'Avg R Winner' });
  renderMiniChart('chartAvgRLoss', cumulativeSeries(filtered, 'avgLossR'), { suffix:'R', decimals:2, tooltipLabel:'Avg R Loser' });
  renderMiniChart('chartWinLossRatio', cumulativeSeries(filtered, 'winLossRatio'), { decimals:2, tooltipLabel:'Winner ÷ |Loser|' });
  renderBarChart('chartTradesByMonth', monthlyTradeCountBars(filtered), { decimals:0, tooltipLabel:'Trades', width:860, height:310 });
  renderBarChart('chartMonthlyGrades', monthlyGradeBars(filtered), { suffix:'R', decimals:2, tooltipLabel:'Total R' });

  const availableGrades = uniqueSorted(filtered.map(t => String(t.grade || '').trim()).filter(Boolean));
  const order = GRADE_ORDER.filter(g => g !== 'Ungraded');
  const grades = [...order.filter(g => availableGrades.includes(g)), ...availableGrades.filter(g => !order.includes(g))];
  const body = $('gradeReportBody');
  body.innerHTML = '';
  if (!grades.length) {
    body.innerHTML = '<tr><td colspan="5" class="empty">No graded trades match the current filters.</td></tr>';
  }
  const gradeRows = grades.map(g => {
    const group = filtered.filter(t => t.grade === g);
    const groupClosed = group.filter(t => (t.status || 'Closed') === 'Closed');
    const groupWins = groupClosed.filter(t => getWinLoss(t) === 'Win');
    const avgR = average(group.map(validRValue).filter(value => value !== null));
    const gradeTotalR = group.reduce((sum,t)=>sum+Number(validRValue(t)||0),0);
    const winPct = groupClosed.length ? groupWins.length / groupClosed.length * 100 : 0;
    return { grade:g, trades:group.length, avgR, totalR:gradeTotalR, winPct };
  });
  const gdir = gradeReportSort.dir === 'asc' ? 1 : -1;
  gradeRows.sort((a,b) => {
    const av = a[gradeReportSort.key], bv = b[gradeReportSort.key];
    return compareMaybeGrade(av, bv, gradeReportSort.key, gdir);
  });
  gradeRows.forEach(row => {
    const tr = document.createElement('tr');
    tr.innerHTML = `<td><strong>${row.grade}</strong></td><td>${row.trades}</td><td>${row.avgR.toFixed(2)}R</td><td>${row.totalR.toFixed(2)}R</td><td>${Math.round(row.winPct)}%</td>`;
    body.appendChild(tr);
  });
}

function getFilteredTrades() {
  const search = ($('search')?.value || '').toLowerCase();
  const quickSetup = $('setupFilter')?.value || '';
  return getReportFilteredTrades().filter(t => {
    const matchesSearch = [t.ticker, t.symbol, t.instrument, t.setup, t.grade, t.notes, t.status, t.portfolioTag, t.secondaryTag, t.mistakeTag, getWinLoss(t)].join(' ').toLowerCase().includes(search);
    const matchesSetup = !quickSetup || String(t.setup || '').toLowerCase() === String(quickSetup).toLowerCase();
    return matchesSearch && matchesSetup;
  });
}

function fmt(value) { return value === undefined || value === null || value === '' ? '-' : value; }

function hasDefinedR(trade) {
  if (trade.rMultiple !== undefined && trade.rMultiple !== null && String(trade.rMultiple).trim() !== '') {
    return Number.isFinite(Number(trade.rMultiple));
  }
  const risk = Number(trade.risk || 0);
  const plRaw = trade.pl;
  const hasPL = plRaw !== undefined && plRaw !== null && String(plRaw).trim() !== '';
  return risk > 0 && hasPL && Number.isFinite(Number(plRaw));
}

function getRMultiple(trade) {
  if (trade.rMultiple !== undefined && trade.rMultiple !== null && String(trade.rMultiple).trim() !== '') return Number(trade.rMultiple);
  const risk = Number(trade.risk || 0);
  const pl = Number(trade.pl || 0);
  if (risk > 0 && Number.isFinite(pl)) return pl / risk;
  return 0;
}

function validRValue(trade) {
  if (!hasDefinedR(trade)) return null;
  const r = Number(getRMultiple(trade));
  return Number.isFinite(r) ? r : null;
}

function cleanBrokerImportNoteText(text) {
  return String(text || '')
    .split(/\n+/)
    .filter(line => !/Broker import updated this trade with new execution data\.?/i.test(line.trim()))
    .join('\n')
    .trim();
}

function getWinLoss(trade) {
  if ((trade.status || 'Closed') === 'Open') return 'Open';
  const hasR = hasDefinedR(trade);
  const r = getRMultiple(trade);
  if (hasR && Number.isFinite(r)) {
    if (r >= -0.1 && r <= 0.1) return 'BE';
    return r > 0.1 ? 'Win' : 'Loss';
  }
  const pl = Number(trade.pl || 0);
  if (pl > 0) return 'Win';
  if (pl < 0) return 'Loss';
  return 'BE';
}

function tradeSortValue(t, key) {
  const r = getRMultiple(t);
  const map = {
    winloss: getWinLoss(t), status: t.status || 'Closed', date: (t.date || '') + ' ' + (t.entryTime || ''), ticker: (t.ticker || t.symbol || ''),
    entry: Number(t.entry || 0), exit: Number(t.exit || 0), r: Number(r || 0), risk: Number(t.risk || 0), pl: Number(t.pl || 0),
    direction: t.direction || '', setup: t.setup || '', grade: t.grade || 'Ungraded', exitDate: (t.exitDate || '') + ' ' + (t.exitTime || ''), portfolio: t.portfolioTag || ''
  };
  return map[key] ?? '';
}
function applyTradeSort(list) {
  const dir = tradeSort.dir === 'asc' ? 1 : -1;
  return list.slice().sort((a,b) => {
    const av = tradeSortValue(a, tradeSort.key), bv = tradeSortValue(b, tradeSort.key);
    return compareMaybeGrade(av, bv, tradeSort.key, dir);
  });
}
function renderTrades() {
  tradeList.innerHTML = '';
  const filtered = applyTradeSort(getFilteredTrades());
  if (!filtered.length) {
    tradeList.innerHTML = '<tr><td colspan="15" class="empty">No trades yet. Add a trade or import your broker CSV.</td></tr>';
    updateBulkSelectionUi();
    return;
  }

  filtered.forEach((trade) => {
    const row = document.createElement('tr');
    row.className = 'trade-row';
    const winLoss = getWinLoss(trade);
    const r = getRMultiple(trade);
    row.innerHTML = `
      <td><input type="checkbox" class="trade-select" data-id="${trade.id}" ${selectedTradeIds.has(trade.id) ? 'checked' : ''} /></td>
      <td><span class="pill ${winLoss.toLowerCase()}">${winLoss}</span></td>
      <td>${fmt(trade.status || 'Closed')}</td>
      <td>${formatTradeLogDate(trade.date)}${trade.entryTime ? '<br><span class="small">' + formatDisplayTime(trade.entryTime) + '</span>' : ''}</td>
      <td><strong>${fmt(trade.ticker || trade.symbol)}</strong></td>
      <td>${price2(trade.entry)}</td>
      <td>${price2(trade.exit)}</td>
      <td>${r ? r.toFixed(2) + 'R' : '-'}</td>
      <td>${trade.risk ? money(trade.risk) : '-'}</td>
      <td class="${Number(trade.pl || 0) >= 0 ? 'positive' : 'negative'}">${trade.pl !== '' && trade.pl !== undefined ? money(trade.pl) : '-'}</td>
      <td>${fmt(trade.direction)}</td>
      <td>${fmt(trade.setup)}</td>
      <td>${fmt(trade.grade || 'Ungraded')}</td>
      <td>${formatTradeLogDate(trade.exitDate)}${trade.exitTime ? '<br><span class="small">' + formatDisplayTime(trade.exitTime) + '</span>' : ''}</td>
      <td>${fmt(trade.portfolioTag)}</td>
    `;
    row.addEventListener('click', (event) => { if (event.target.closest('input,button,select')) return; openTradeDialog(trade.id); });
    const cb = row.querySelector('.trade-select');
    if (cb) cb.addEventListener('change', (event) => {
      if (event.target.checked) selectedTradeIds.add(trade.id); else selectedTradeIds.delete(trade.id);
      updateBulkSelectionUi();
    });
    tradeList.appendChild(row);
  });
  updateBulkSelectionUi();
}

function updateBulkSelectionUi() {
  const count = Array.from(selectedTradeIds).filter(id => trades.some(t => t.id === id)).length;
  selectedTradeIds = new Set(Array.from(selectedTradeIds).filter(id => trades.some(t => t.id === id)));
  if ($('selectedTradeCount')) $('selectedTradeCount').textContent = `${count} selected`;
  if ($('selectAllTrades')) {
    const visibleIds = getFilteredTrades().map(t => t.id);
    $('selectAllTrades').checked = visibleIds.length > 0 && visibleIds.every(id => selectedTradeIds.has(id));
    $('selectAllTrades').indeterminate = visibleIds.some(id => selectedTradeIds.has(id)) && !$('selectAllTrades').checked;
  }
}

function refreshBulkActionValue() {
  const action = $('bulkActionType')?.value || '';
  const valueSel = $('bulkActionValue');
  if (!valueSel) return;
  const current = valueSel.value;
  let values = [];
  let placeholder = 'Select an action first...';
  if (action === 'setup') {
    values = uniqueSorted(setups.map(s => s.name));
    placeholder = values.length ? 'Select setup...' : 'No saved setups yet';
  } else if (action === 'portfolioTag') {
    values = uniqueSorted(portfolios.map(p => p.name));
    placeholder = values.length ? 'Select portfolio...' : 'No portfolios yet';
  } else if (action === 'merge') {
    valueSel.disabled = true;
    valueSel.innerHTML = '<option value="">Selected trades will be merged into one trade</option>';
    return;
  } else if (action === 'delete') {
    valueSel.disabled = true;
    valueSel.innerHTML = '<option value="">Selected trades will be deleted</option>';
    return;
  }
  valueSel.disabled = !action;
  valueSel.innerHTML = `<option value="">${placeholder}</option>` + values.map(v => `<option value="${escapeHtml(v)}">${escapeHtml(v)}</option>`).join('');
  valueSel.value = values.includes(current) ? current : '';
}

function tradeTimestampMs(trade, preferExit = false) {
  const date = preferExit ? (trade.exitDate || trade.date || '') : (trade.date || trade.exitDate || '');
  const time = preferExit ? (trade.exitTime || trade.entryTime || '') : (trade.entryTime || trade.exitTime || '');
  const parsed = Date.parse(`${date || '1900-01-01'}T${String(time || '00:00:00').padEnd(8, ':00').slice(0, 8)}`);
  return Number.isFinite(parsed) ? parsed : 0;
}

function weightedAverageFromTrades(list, field) {
  let totalWeight = 0;
  let totalValue = 0;
  list.forEach(trade => {
    const value = Number(trade[field]);
    if (!Number.isFinite(value) || value === 0) return;
    const weight = Math.abs(Number(trade.size || trade.shares || trade.quantity || trade.qty || 1)) || 1;
    totalWeight += weight;
    totalValue += value * weight;
  });
  return totalWeight ? totalValue / totalWeight : '';
}

function firstNonEmptyValue(list, keys) {
  for (const trade of list) {
    for (const key of keys) {
      const value = trade?.[key];
      if (value !== undefined && value !== null && String(value).trim() !== '') return value;
    }
  }
  return '';
}

function mergeSelectedTrades(ids) {
  const selected = trades.filter(trade => ids.includes(trade.id));
  if (selected.length < 2) return alert('Select at least two trades to merge.');
  const tickers = uniqueSorted(selected.map(trade => trade.ticker || trade.symbol));
  if (tickers.length > 1 && !confirm(`You selected multiple tickers (${tickers.join(', ')}). Merge them anyway?`)) return;

  const ordered = selected.slice().sort((a, b) => tradeTimestampMs(a) - tradeTimestampMs(b));
  const base = deepClone(ordered[0]);
  const latestExit = selected.slice().sort((a, b) => tradeTimestampMs(b, true) - tradeTimestampMs(a, true))[0];
  const screenshots = [];
  const seenScreenshots = new Set();

  selected.forEach(trade => (trade.screenshots || []).forEach(screenshot => {
    const key = screenshot.id || screenshot.dataUrl || screenshot.name;
    if (key && seenScreenshots.has(key)) return;
    if (key) seenScreenshots.add(key);
    screenshots.push(screenshot);
  }));

  const executions = dedupeExecutions(selected.flatMap(trade => syntheticExecutionsFromTrade(trade)));
  const netPosition = netExecutionPosition(executions);
  const executionSummary = executions.length
    ? summarizeRoundTrip(base.instrument || base.ticker || base.symbol || '', executions, netPosition === 0)
    : null;
  const merged = {
    ...base,
    ...(executionSummary || {}),
    id: base.id,
    ticker: firstNonEmptyValue(ordered, ['ticker', 'symbol']) || base.ticker || base.symbol,
    symbol: firstNonEmptyValue(ordered, ['symbol', 'ticker']) || base.symbol || base.ticker,
    date: ordered[0].date || base.date,
    entryTime: ordered[0].entryTime || base.entryTime,
    exitDate: latestExit?.exitDate || base.exitDate || '',
    exitTime: latestExit?.exitTime || base.exitTime || '',
    status: executions.length
      ? (netPosition === 0 ? 'Closed' : 'Open')
      : (selected.some(trade => String(trade.status || '').toLowerCase() === 'open') && !latestExit?.exitDate ? 'Open' : 'Closed'),
    entry: executionSummary?.entry || weightedAverageFromTrades(selected, 'entry') || base.entry || '',
    exit: executionSummary?.exit || weightedAverageFromTrades(selected, 'exit') || base.exit || '',
    size: executionSummary?.size || selected.reduce((sum, trade) => sum + Math.abs(Number(trade.size || trade.shares || trade.quantity || trade.qty || 0)), 0) || base.size || '',
    pl: executionSummary ? executionSummary.pl : selected.reduce((sum, trade) => sum + Number(trade.pl || 0), 0),
    risk: selected.reduce((sum, trade) => sum + Number(trade.risk || 0), 0) || firstNonEmptyValue(ordered, ['risk']),
    setup: firstNonEmptyValue(ordered, ['setup']) || base.setup || '',
    setupId: firstNonEmptyValue(ordered, ['setupId']) || base.setupId || '',
    portfolioTag: firstNonEmptyValue(ordered, ['portfolioTag']) || base.portfolioTag || '',
    secondaryTag: firstNonEmptyValue(ordered, ['secondaryTag']) || base.secondaryTag || '',
    mistakeTag: firstNonEmptyValue(ordered, ['mistakeTag']) || base.mistakeTag || '',
    grade: firstNonEmptyValue(ordered, ['grade']) || base.grade || '',
    stop: firstNonEmptyValue(ordered, ['stop', 'stopLoss']) || base.stop || base.stopLoss || '',
    stopLoss: firstNonEmptyValue(ordered, ['stopLoss', 'stop']) || base.stopLoss || base.stop || '',
    target: firstNonEmptyValue(ordered, ['target']) || base.target || '',
    stfAtr: firstNonEmptyValue(ordered, ['stfAtr', 'atr']) || base.stfAtr || base.atr || '',
    notes: selected.map(trade => String(trade.notes || '').trim()).filter(Boolean).join('\n\n--- merged trade note ---\n\n'),
    screenshots,
    rawExecutions: executions,
    executionKeys: executions.map(execution => execution.executionKey).filter(Boolean),
    mergedTradeIds: uniqueSorted([...(base.mergedTradeIds || []), ...selected.map(trade => trade.id)]),
    mergedAt: new Date().toISOString()
  };

  const scoreSource = ordered.find(trade => trade.setupScore);
  if (scoreSource) merged.setupScore = deepClone(scoreSource.setupScore);
  if (merged.risk && merged.pl !== '' && merged.pl !== undefined) {
    merged.rMultiple = Number(merged.pl || 0) / Number(merged.risk || 1);
  }

  trades = trades.filter(trade => !ids.includes(trade.id));
  trades.push(merged);
  selectedTradeIds.clear();
  save();
  renderAll();
  alert(`Merged ${selected.length} trades into ${merged.ticker || 'one trade'}.`);
}

function bulkApply(field, value) {
  const ids = Array.from(selectedTradeIds);
  if (!ids.length) return alert('Select at least one trade first.');
  if (field === 'merge') {
    if (!confirm(`Merge ${ids.length} selected trade${ids.length === 1 ? '' : 's'} into one trade? The earliest trade ID will be kept and the others will be removed.`)) return;
    mergeSelectedTrades(ids);
    return;
  }
  if (field === 'delete') {
    if (!confirm(`Delete ${ids.length} selected trade${ids.length === 1 ? '' : 's'}? This cannot be undone.`)) return;
    trades = trades.filter(t => !ids.includes(t.id));
    selectedTradeIds.clear();
    save(); renderAll();
    return;
  }
  if (!value) return alert('Pick a tag first.');
  trades = trades.map(t => ids.includes(t.id) ? { ...t, [field]: value, ...(field === 'setup' ? { setupId: (getSetupByName(value)?.id || t.setupId || '') } : {}) } : t);
  save();
  renderAll();
}


function uniqueSorted(values) {
  return Array.from(new Set(values.map(v => String(v || '').trim()).filter(Boolean))).sort((a,b)=>a.localeCompare(b));
}
function optionList(values, allText, current) {
  return `<option value="">${allText}</option>` + values.map(v => `<option value="${escapeHtml(v)}" ${v===current?'selected':''}>${escapeHtml(v)}</option>`).join('');
}
function dateOptionList(values, allText, current) {
  return `<option value="">${allText}</option>` + values.map(v => `<option value="${escapeHtml(v)}" ${v===current?'selected':''}>${escapeHtml(formatDisplayDate(v))}</option>`).join('');
}

function getSelectValues(id) {
  const el = $(id);
  if (!el) return [];
  if (el.multiple) return Array.from(el.selectedOptions).map(o => o.value).filter(Boolean);
  return el.value ? [el.value] : [];
}
function setSelectValues(id, values) {
  const el = $(id); if (!el) return;
  const set = new Set(Array.isArray(values) ? values : (values ? [values] : []));
  Array.from(el.options).forEach(o => { o.selected = set.has(o.value) && o.value !== ''; });
  if (!el.multiple) el.value = values?.[0] || values || '';
}
function optionListMulti(values, allText, currents=[]) {
  const set = new Set(Array.isArray(currents) ? currents : (currents ? [currents] : []));
  return `<option value="">${allText}</option>` + values.map(v => `<option value="${escapeHtml(v)}" ${set.has(v)?'selected':''}>${escapeHtml(v)}</option>`).join('');
}
function valueMatchesAny(value, selected) {
  if (!selected || !selected.length) return true;
  const raw = String(value || '').trim();
  const v = raw.toLowerCase();
  return selected.some(x => {
    const sx = String(x || '').trim();
    if (sx === '__NONE__') return raw === '';
    return sx.toLowerCase() === v;
  });
}
function tagOptionsWithNone(list, noneLabel) {
  return [{ value: '__NONE__', label: noneLabel }, ...normalizeTagList(list, 'tag').map(item => ({ value: item.name, label: item.name }))];
}
function optionListMultiObjects(items, allText, currents=[]) {
  const set = new Set(Array.isArray(currents) ? currents : (currents ? [currents] : []));
  return `<option value="">${allText}</option>` + items.map(item => `<option value="${escapeHtml(item.value)}" ${set.has(item.value)?'selected':''}>${escapeHtml(item.label)}</option>`).join('');
}
function parseIsoDate(value) {
  const m = String(value || '').match(/^(\d{4})-(\d{2})-(\d{2})$/);
  if (!m) return null;
  return new Date(Number(m[1]), Number(m[2]) - 1, Number(m[3]));
}
function isoDate(d) {
  return `${d.getFullYear()}-${String(d.getMonth()+1).padStart(2,'0')}-${String(d.getDate()).padStart(2,'0')}`;
}
function setReportDateRange(start, end) {
  if ($('reportStartDate')) $('reportStartDate').value = start || '';
  if ($('reportEndDate')) $('reportEndDate').value = end || '';
  if ($('logReportStartDate')) $('logReportStartDate').value = start || '';
  if ($('logReportEndDate')) $('logReportEndDate').value = end || '';
  updateDatePickerLabels();
  refreshReportsAndTradeLog();
}
function reportPairId(id) { return id.startsWith('logReport') ? id.replace('logReport','report') : id.replace('report','logReport'); }
function syncReportPair(changedId) { const other = $(reportPairId(changedId)); const el = $(changedId); if (other && el) { if (el.multiple || other.multiple) { setSelectValues(other.id, getSelectValues(changedId)); refreshMultiSelectUI(other.id); refreshMultiSelectUI(changedId); } else other.value = el.value; } updateDatePickerLabels(); }
function updateDatePickerLabels() {
  document.querySelectorAll('.datePickerBtn').forEach(btn => {
    const target = btn.dataset.dateTarget;
    const value = $(target)?.value || '';
    if (target === 'watchlistDate') {
      btn.textContent = value ? formatDisplayDate(value) : 'Select date';
    } else {
      btn.textContent = value ? formatDisplayDate(value) : (target.toLowerCase().includes('start') ? 'All start dates' : 'All end dates');
    }
  });
}

function initSortableTables() {
  const tradeKeys = ['', 'winloss','status','date','ticker','entry','exit','r','risk','pl','direction','setup','grade','exitDate','portfolio'];
  document.querySelectorAll('#tradeLogView .trade-table thead th').forEach((th, i) => {
    if (!tradeKeys[i] || th.dataset.sortReady === '1') return;
    th.classList.add('sortable-th');
    th.dataset.sortKey = tradeKeys[i];
    th.dataset.sortReady = '1';
    th.addEventListener('click', () => {
      const key = th.dataset.sortKey;
      tradeSort = { key, dir: tradeSort.key === key && tradeSort.dir === 'asc' ? 'desc' : 'asc' };
      renderTrades();
    });
  });
  const gradeKeys = ['grade','trades','avgR','totalR','winPct'];
  document.querySelectorAll('#dashboardView .grade-table thead th').forEach((th, i) => {
    if (!gradeKeys[i] || th.dataset.sortReady === '1') return;
    th.classList.add('sortable-th');
    th.dataset.sortKey = gradeKeys[i];
    th.dataset.sortReady = '1';
    th.addEventListener('click', () => {
      const key = th.dataset.sortKey;
      gradeReportSort = { key, dir: gradeReportSort.key === key && gradeReportSort.dir === 'asc' ? 'desc' : 'asc' };
      renderDashboardReports();
    });
  });

  const expectancyKeys = ['setup','version','grade','trades','avgR','avgPL','winPct','totalR'];
  document.querySelectorAll('#monthlyReviewView .grade-table thead th').forEach((th, i) => {
    if (!expectancyKeys[i] || th.dataset.sortReady === '1') return;
    th.classList.add('sortable-th');
    th.dataset.sortKey = expectancyKeys[i];
    th.dataset.sortReady = '1';
    th.addEventListener('click', () => {
      const key = th.dataset.sortKey;
      monthlyExpectancySort = { key, dir: monthlyExpectancySort.key === key && monthlyExpectancySort.dir === 'asc' ? 'desc' : 'asc' };
      renderMonthlyReview();
    });
  });
  const breakdownKeys = ['setup','grade','trades','totalPL','totalR','avgR','winPct'];
  document.querySelectorAll('#setupCompareBreakdownTable thead th').forEach((th, i) => {
    if (!breakdownKeys[i] || th.dataset.sortReady === '1') return;
    th.classList.add('sortable-th');
    th.dataset.sortKey = breakdownKeys[i];
    th.dataset.sortReady = '1';
    th.addEventListener('click', () => {
      const key = th.dataset.sortKey;
      breakdownSort = { key, dir: breakdownSort.key === key && breakdownSort.dir === 'asc' ? 'desc' : 'asc' };
      renderSetupComparison();
    });
  });
}

function openDatePicker(anchorBtn) {
  closeDatePicker();
  const target = anchorBtn.dataset.dateTarget;
  const isWatchlistPicker = target === 'watchlistDate';
  const current = $(target)?.value || '';
  let viewDate = parseIsoDate(current) || new Date();
  viewDate = new Date(viewDate.getFullYear(), viewDate.getMonth(), 1);
  let pendingDate = current;

  const panel = document.createElement('div');
  panel.className = 'date-picker-popover calendar-style';
  document.body.appendChild(panel);
  panel.addEventListener('click', (e) => e.stopPropagation());
  const rect = anchorBtn.getBoundingClientRect();
  panel.style.left = Math.min(rect.left, window.innerWidth - 378) + 'px';
  panel.style.top = Math.min(rect.bottom + 6, window.innerHeight - 390) + 'px';

  const applyPending = () => {
    if ($(target)) $(target).value = pendingDate || '';
    if (isWatchlistPicker) {
      updateDatePickerLabels();
      closeDatePicker();
      renderWatchlist();
      return;
    }
    syncReportPair(target);
    updateDatePickerLabels();
    closeDatePicker();
    refreshReportsAndTradeLog();
  };

  const render = () => {
    const year = viewDate.getFullYear();
    const month = viewDate.getMonth();
    const first = new Date(year, month, 1);
    const startGrid = new Date(year, month, 1 - first.getDay());
    const monthLabel = first.toLocaleDateString(undefined, { month:'long', year:'numeric' });
    const days = [];
    for (let i=0; i<42; i++) {
      const d = new Date(startGrid); d.setDate(startGrid.getDate()+i);
      const iso = isoDate(d);
      const muted = d.getMonth() !== month ? ' muted-day' : '';
      const selected = iso === pendingDate ? ' selected' : '';
      days.push(`<button type="button" class="cal-day${muted}${selected}" data-date="${iso}">${d.getDate()}</button>`);
    }
    panel.innerHTML = `
      <div class="calendar-panel single-calendar ${isWatchlistPicker ? 'watchlist-calendar' : ''}">
        <div>
          <div class="calendar-head">
            <button type="button" data-cal="prevYear">«</button>
            <button type="button" data-cal="prevMonth">‹</button>
            <strong>${escapeHtml(monthLabel)}</strong>
            <button type="button" data-cal="nextMonth">›</button>
            <button type="button" data-cal="nextYear">»</button>
          </div>
          <div class="calendar-week"><span>Su</span><span>Mo</span><span>Tu</span><span>We</span><span>Th</span><span>Fr</span><span>Sa</span></div>
          <div class="calendar-days">${days.join('')}</div>
          <div class="calendar-footer"><button type="button" data-cal="clear">Clear</button><button type="button" data-cal="today">Today</button></div>
        </div>
        ${isWatchlistPicker ? '' : `<div class="calendar-presets">
          <button type="button" data-range="today">Today</button>
          <button type="button" data-range="yesterday">Yesterday</button>
          <button type="button" data-range="7">Last 7 days</button>
          <button type="button" data-range="30">Last 30 days</button>
          <button type="button" data-range="thisMonth">This Month</button>
          <button type="button" data-range="lastMonth">Last Month</button>
          <button type="button" data-range="12months">Last 12 Months</button>
          <button type="button" data-range="lastYear">Last Year</button>
          <button type="button" data-range="ytd">YTD</button>
          <button type="button" class="apply-date" data-cal="apply">Apply Date</button>
        </div>`}
      </div>`;
    panel.querySelectorAll('.cal-day').forEach(day => day.addEventListener('click', () => { pendingDate = day.dataset.date; applyPending(); }));
    panel.querySelectorAll('[data-cal]').forEach(btn => btn.addEventListener('click', (e) => { e.preventDefault(); e.stopPropagation();
      const action = btn.dataset.cal;
      if (action === 'prevMonth') viewDate = new Date(year, month-1, 1);
      if (action === 'nextMonth') viewDate = new Date(year, month+1, 1);
      if (action === 'prevYear') viewDate = new Date(year-1, month, 1);
      if (action === 'nextYear') viewDate = new Date(year+1, month, 1);
      if (action === 'today') { const now = new Date(); pendingDate = isoDate(now); viewDate = new Date(now.getFullYear(), now.getMonth(), 1); }
      if (action === 'clear') { pendingDate = ''; applyPending(); return; }
      if (action === 'apply') { applyPending(); return; }
      render();
    }));
    panel.querySelectorAll('[data-range]').forEach(btn => btn.addEventListener('click', (e) => { e.preventDefault(); e.stopPropagation();
      const now = new Date(); now.setHours(0,0,0,0);
      let start = '', end = isoDate(now);
      const r = btn.dataset.range;
      if (r === 'today') start = isoDate(now);
      if (r === 'yesterday') { const y = new Date(now); y.setDate(y.getDate()-1); start = end = isoDate(y); }
      if (r === '7') { const d = new Date(now); d.setDate(d.getDate()-6); start = isoDate(d); }
      if (r === '30') { const d = new Date(now); d.setDate(d.getDate()-29); start = isoDate(d); }
      if (r === 'thisMonth') start = isoDate(new Date(now.getFullYear(), now.getMonth(), 1));
      if (r === 'lastMonth') { start = isoDate(new Date(now.getFullYear(), now.getMonth()-1, 1)); end = isoDate(new Date(now.getFullYear(), now.getMonth(), 0)); }
      if (r === '12months') { const d = new Date(now); d.setMonth(d.getMonth()-11); d.setDate(1); start = isoDate(d); }
      if (r === 'lastYear') { start = isoDate(new Date(now.getFullYear()-1,0,1)); end = isoDate(new Date(now.getFullYear()-1,11,31)); }
      if (r === 'ytd') start = isoDate(new Date(now.getFullYear(), 0, 1));
      setReportDateRange(start, end);
      closeDatePicker();
    }));
  };
  render();
  setTimeout(() => document.addEventListener('click', outsideDatePicker, { once: true }), 0);
}
function outsideDatePicker(event) {
  if (!event.target.closest('.date-picker-popover') && !event.target.closest('.datePickerBtn')) closeDatePicker();
  else document.addEventListener('click', outsideDatePicker, { once: true });
}
function closeDatePicker() { document.querySelectorAll('.date-picker-popover').forEach(p => p.remove()); }

const MULTI_SELECT_IDS = ['reportTicker','reportSetup','reportPortfolio','reportResult','reportGrade','reportSecondaryTag','reportMistakeTag','logReportTicker','logReportSetup','logReportPortfolio','logReportResult','logReportGrade','logReportSecondaryTag','logReportMistakeTag','compareGradeA','compareGradeB','playbookSetup','playbookGrade','playbookPortfolio','playbookMistake'];

function labelForMultiSelect(select) {
  const selected = Array.from(select.selectedOptions || []).map(o => o.value).filter(Boolean);
  const allCount = Array.from(select.options || []).filter(o => o.value).length;
  const placeholder = select.options[0]?.textContent || 'All';
  if (!selected.length || selected.length === allCount) return placeholder;
  if (selected.length === 1) return selected[0] === '__NONE__' ? (Array.from(select.options).find(o=>o.value==='__NONE__')?.textContent || 'None') : selected[0];
  return `${selected.length} selected`;
}

function refreshMultiSelectUI(selectId) {
  const select = $(selectId);
  const wrap = document.querySelector(`.multi-check[data-select-id="${selectId}"]`);
  if (!select || !wrap) return;
  const btn = wrap.querySelector('.multi-check-button');
  const optionsBox = wrap.querySelector('.multi-check-options');
  const search = wrap.querySelector('.multi-check-search');
  const query = (search?.value || '').toLowerCase().trim();
  if (btn) btn.textContent = labelForMultiSelect(select);
  if (!optionsBox) return;
  const opts = Array.from(select.options || []).filter(o => o.value);
  const visible = opts.filter(o => !query || String(o.textContent || o.value).toLowerCase().includes(query));
  const selectedValues = new Set(Array.from(select.selectedOptions || []).map(o => o.value));
  const allSelected = opts.length > 0 && opts.every(o => selectedValues.has(o.value));
  optionsBox.innerHTML = `
    <label class="multi-check-row"><input type="checkbox" data-select-all="1" ${allSelected ? 'checked' : ''}><span>Select All</span></label>
    ${visible.length ? visible.map(o => `<label class="multi-check-row"><input type="checkbox" value="${escapeHtml(o.value)}" ${selectedValues.has(o.value) ? 'checked' : ''}><span>${escapeHtml(o.textContent || o.value)}</span></label>`).join('') : '<div class="multi-check-empty">No matches</div>'}
  `;
}

function refreshAllMultiSelectUI() { MULTI_SELECT_IDS.forEach(refreshMultiSelectUI); }

function enhanceMultiSelects() {
  MULTI_SELECT_IDS.forEach(id => {
    const select = $(id);
    if (!select || !select.multiple) return;
    select.classList.add('native-multi-hidden');
    let wrap = document.querySelector(`.multi-check[data-select-id="${id}"]`);
    if (!wrap) {
      wrap = document.createElement('div');
      wrap.className = 'multi-check';
      wrap.dataset.selectId = id;
      wrap.innerHTML = `<button type="button" class="multi-check-button">All</button><div class="multi-check-menu"><input type="text" class="multi-check-search" placeholder="Search"><div class="multi-check-options"></div></div>`;
      select.insertAdjacentElement('afterend', wrap);
      const btn = wrap.querySelector('.multi-check-button');
      const search = wrap.querySelector('.multi-check-search');
      const optionsBox = wrap.querySelector('.multi-check-options');
      btn.addEventListener('click', (e) => {
        e.stopPropagation();
        document.querySelectorAll('.multi-check.open').forEach(m => { if (m !== wrap) m.classList.remove('open'); });
        wrap.classList.toggle('open');
        refreshMultiSelectUI(id);
        if (wrap.classList.contains('open')) setTimeout(() => search?.focus(), 0);
      });
      search.addEventListener('input', () => refreshMultiSelectUI(id));
      optionsBox.addEventListener('change', (e) => {
        const target = e.target;
        if (!target || target.tagName !== 'INPUT') return;
        const opts = Array.from(select.options || []).filter(o => o.value);
        if (target.dataset.selectAll) {
          opts.forEach(o => { o.selected = target.checked; });
        } else {
          const opt = opts.find(o => o.value === target.value);
          if (opt) opt.selected = target.checked;
        }
        select.dispatchEvent(new Event('change', { bubbles: true }));
        refreshMultiSelectUI(id);
      });
    }
    refreshMultiSelectUI(id);
  });
}

document.addEventListener('click', (e) => {
  if (!e.target.closest('.multi-check')) document.querySelectorAll('.multi-check.open').forEach(m => m.classList.remove('open'));
});

function refreshReportFilterOptions() {
  updateDatePickerLabels();
  const tickers = uniqueSorted(trades.map(t => (t.ticker || t.symbol || '').toUpperCase()));
  ['reportTicker','logReportTicker'].forEach(id => { const sel = $(id); if (!sel) return; const cur = getSelectValues(id); sel.innerHTML = optionListMulti(tickers, 'All tickers', cur); setSelectValues(id, cur.filter(v=>tickers.includes(v))); });
  const setupNames = uniqueSorted([...setups.map(s=>s.name), ...trades.map(t=>t.setup)]);
  ['reportSetup','logReportSetup'].forEach(id => { const sel = $(id); if (!sel) return; const cur = getSelectValues(id); sel.innerHTML = optionListMulti(setupNames, 'All setups', cur); setSelectValues(id, cur.filter(v=>setupNames.includes(v))); });
  const portfolioNames = uniqueSorted(portfolios.map(p=>p.name));
  ['reportPortfolio','logReportPortfolio'].forEach(id => { const sel = $(id); if (!sel) return; const cur = getSelectValues(id); sel.innerHTML = optionListMulti(portfolioNames, 'All portfolios', cur); setSelectValues(id, cur.filter(v=>portfolioNames.includes(v))); });
  normalizeTags();
  const secondaryNames = tagNames(tags.secondary);
  const mistakeNames = tagNames(tags.mistakes);
  ['reportSecondaryTag','logReportSecondaryTag'].forEach(id => { const sel = $(id); if (!sel) return; const cur = getSelectValues(id); const opts = tagOptionsWithNone(tags.secondary, 'No Secondary Tag'); sel.innerHTML = optionListMultiObjects(opts, 'All secondary tags', cur); setSelectValues(id, cur.filter(v=>v==='__NONE__' || secondaryNames.includes(v))); });
  ['reportMistakeTag','logReportMistakeTag'].forEach(id => { const sel = $(id); if (!sel) return; const cur = getSelectValues(id); const opts = tagOptionsWithNone(tags.mistakes, 'No Mistake'); sel.innerHTML = optionListMultiObjects(opts, 'All mistake tags', cur); setSelectValues(id, cur.filter(v=>v==='__NONE__' || mistakeNames.includes(v))); });
  const grades = uniqueSorted(trades.map(t => (t.grade || '').trim() || 'Ungraded'));
  const ordered = GRADE_ORDER.filter(g=>grades.includes(g));
  const extras = grades.filter(g=>!ordered.includes(g));
  const gradeOptions = [...ordered, ...extras];
  ['reportGrade','logReportGrade'].forEach(id => { const sel = $(id); if (!sel) return; const cur = getSelectValues(id); sel.innerHTML = optionListMulti(gradeOptions, 'All grades', cur); setSelectValues(id, cur.filter(v=>gradeOptions.includes(v))); });
  ['reportDirection'].forEach(id => { const other = $(reportPairId(id)); if (other && $(id)) other.value = $(id).value; });
  ['reportResult','logReportResult'].forEach(id => {
    const selected = getSelectValues(id);
    setSelectValues(id, selected.filter(value => ['Win','Loss','BE','Open'].includes(value)));
  });
  enhanceMultiSelects();
  refreshComparisonOptions();
}


function fillPortfolioForm(portfolio = {id:'', name:'', description:''}) {
  if (!$('portfolioForm')) return;
  $('portfolioId').value = portfolio.id || '';
  $('portfolioName').value = portfolio.name || '';
  $('portfolioDescription').value = portfolio.description || '';
}
function renderPortfolios() {
  const list = $('portfolioList'); if (!list) return;
  if (!portfolios.length) { list.innerHTML = '<p class="empty">No portfolios yet. Add one above, like Main, IRA, Cash, or Challenge Account.</p>'; return; }
  list.innerHTML = '';
  portfolios.forEach(portfolio => {
    const count = trades.filter(t => String(t.portfolioTag || '').toLowerCase() === String(portfolio.name || '').toLowerCase()).length;
    const div = document.createElement('div');
    div.className = 'setup-card';
    div.innerHTML = `<div><h3>${escapeHtml(portfolio.name)}</h3><p>${escapeHtml(portfolio.description || '')}</p><span class="small">Trades tagged here: ${count}</span></div><div><button type="button" class="editPortfolioBtn">Edit</button><button type="button" class="deletePortfolioBtn danger">Delete</button></div>`;
    div.querySelector('.editPortfolioBtn').addEventListener('click', () => fillPortfolioForm(portfolio));
    div.querySelector('.deletePortfolioBtn').addEventListener('click', () => { if(confirm('Delete this portfolio? Existing trades will keep their portfolio tag text.')) { portfolios = portfolios.filter(p => p.id !== portfolio.id); savePortfolios(); renderAll(); } });
    list.appendChild(div);
  });
}
function refreshPortfolioOptions() {
  const names = uniqueSorted(portfolios.map(p => p.name));
  ['editPortfolioTag','portfolioTag'].forEach(id => {
    const sel = $(id);
    if (!sel || sel.tagName !== 'SELECT') return;
    const cur = sel.value;
    sel.innerHTML = `<option value="">${id === 'bulkPortfolioTag' ? 'Portfolio tag...' : 'No portfolio'}</option>` + names.map(n => `<option ${n===cur?'selected':''}>${escapeHtml(n)}</option>`).join('');
    sel.value = names.includes(cur) ? cur : '';
  });
  refreshBulkActionValue();
}


function setupComparisonStats(list) {
  const closed = list.filter(t => (t.status || 'Closed') === 'Closed');
  const wins = closed.filter(t => getWinLoss(t) === 'Win');
  const totalPL = list.reduce((sum,t)=>sum+Number(t.pl||0),0);
  const totalR = list.reduce((sum,t)=>sum+Number(getRMultiple(t)||0),0);
  const avgR = average(list.map(getRMultiple));
  const winPct = closed.length ? wins.length / closed.length * 100 : 0;
  return { trades:list.length, totalPL, totalR, avgR, winPct };
}
function statsHtml(stats) {
  return `<div><span>Trades</span><strong>${stats.trades}</strong></div><div><span>Total PnL</span><strong>${money(stats.totalPL)}</strong></div><div><span>Total R</span><strong>${stats.totalR.toFixed(2)}R</strong></div><div><span>Avg R</span><strong>${stats.avgR.toFixed(2)}R</strong></div><div><span>Win %</span><strong>${stats.winPct.toFixed(1)}%</strong></div>`;
}
function refreshComparisonVersionOptions(side) {
  const setupName = $(`compareSetup${side}`)?.value || '';
  const sel = $(`compareVersion${side}`); if (!sel) return;
  const cur = sel.value || 'current';
  const setup = getSetupByName(setupName);
  let html = '<option value="current">Current Version</option><option value="past">All Past Versions</option><option value="all">All Versions</option>';
  if (setup) {
    normalizeSetupVersions(setup);
    html += (setup.versions || []).map(v => `<option value="${Number(v.version)}">${setupVersionLabel(v.version)}</option>`).join('');
  }
  sel.innerHTML = html;
  const values = Array.from(sel.options).map(o=>o.value);
  sel.value = values.includes(cur) ? cur : 'current';
}
function refreshComparisonOptions() {
  normalizeAllSetupVersions();
  const names = uniqueSorted([...setups.map(s=>s.name), ...trades.map(t=>t.setup)]);
  ['compareSetupA','compareSetupB','breakdownSetupFilter'].forEach(id => {
    const sel = $(id); if (!sel) return;
    const cur = sel.value;
    sel.innerHTML = '<option value="">All setups</option>' + names.map(n=>`<option ${n===cur?'selected':''}>${escapeHtml(n)}</option>`).join('');
    sel.value = names.includes(cur) ? cur : '';
  });
  ['A','B'].forEach(refreshComparisonVersionOptions);
  const order = GRADE_ORDER.filter(g => g !== 'Ungraded');
  ['compareGradeA','compareGradeB'].forEach(id => {
    const side = id.endsWith('A') ? 'A' : 'B';
    const setupName = $(`compareSetup${side}`)?.value || '';
    const relevant = setupName ? trades.filter(t => String(t.setup||'').toLowerCase() === setupName.toLowerCase()) : trades;
    const grades = uniqueSorted(relevant.map(t => t.grade || '').filter(Boolean));
    const ordered = [...order.filter(g=>grades.includes(g)), ...grades.filter(g=>!order.includes(g))];
    const sel = $(id); if (!sel) return;
    const cur = getSelectValues(id);
    sel.innerHTML = '<option value="">All grades</option>' + ordered.map(g=>`<option value="${escapeHtml(g)}">${escapeHtml(g)}</option>`).join('');
    setSelectValues(id, cur.filter(g => ordered.includes(g)));
  });
  refreshAllMultiSelectUI();
}
function compareFilteredTrades(setupName, grades, versionFilter) {
  const gradeValues = Array.isArray(grades) ? grades : (grades ? [grades] : []);
  const setup = getSetupByName(setupName);
  const vf = versionFilter || 'current';
  return trades.filter(t => {
    const setupOk = !setupName || String(t.setup||'').toLowerCase() === setupName.toLowerCase();
    const gradeOk = !gradeValues.length || gradeValues.includes(t.grade || 'Ungraded');
    let versionOk = true;
    if (setupName && vf !== 'all') {
      const tradeVersion = tradeSetupVersion(t, setup);
      const currentVersion = currentVersionForSetup(setup);
      if (vf === 'current') versionOk = tradeVersion === currentVersion;
      else if (vf === 'past') versionOk = tradeVersion !== currentVersion;
      else if (vf === 'all') versionOk = true;
      else versionOk = tradeVersion === Number(vf);
    }
    return setupOk && gradeOk && versionOk;
  });
}
function compareLineSeries(list, mode) {
  const sorted = sortedClosedTradesForCharts(list);
  let pl=0, r=0, wins=0, count=0;
  return sorted.map((t,i) => {
    count += 1; pl += Number(t.pl || 0); r += Number(getRMultiple(t)||0); if (getWinLoss(t)==='Win') wins += 1;
    const value = mode === 'pl' ? pl : mode === 'r' ? r : mode === 'avgR' ? r/count : (wins/count*100);
    return { value, label:`Trade ${i+1}`, detail:`${t.ticker || ''} ${formatDisplayDate(t.exitDate || t.date)}` };
  });
}
function renderCompareCharts(prefix, list) {
  renderMiniChart(`chartCompare${prefix}PL`, compareLineSeries(list,'pl'), { prefix:'$', decimals:2, axis:'money', tooltipLabel:'Total PnL', largeTooltip:true });
  renderMiniChart(`chartCompare${prefix}R`, compareLineSeries(list,'r'), { suffix:'R', decimals:2, tooltipLabel:'Total R', largeTooltip:true });
  renderMiniChart(`chartCompare${prefix}AvgR`, compareLineSeries(list,'avgR'), { suffix:'R', decimals:2, tooltipLabel:'Average R', largeTooltip:true });
  renderMiniChart(`chartCompare${prefix}Win`, compareLineSeries(list,'win'), { suffix:'%', decimals:1, tooltipLabel:'Win %', largeTooltip:true });
}
function renderSetupComparison() {
  const setupA = $('compareSetupA')?.value || '';
  const setupB = $('compareSetupB')?.value || '';
  const gradeA = getSelectValues('compareGradeA');
  const gradeB = getSelectValues('compareGradeB');
  const versionA = $('compareVersionA')?.value || 'current';
  const versionB = $('compareVersionB')?.value || 'current';
  const versionLabelA = versionA === 'current' ? 'Current Version' : versionA === 'past' ? 'All Past Versions' : versionA === 'all' ? 'All Versions' : setupVersionLabel(versionA);
  const versionLabelB = versionB === 'current' ? 'Current Version' : versionB === 'past' ? 'All Past Versions' : versionB === 'all' ? 'All Versions' : setupVersionLabel(versionB);
  const listA = compareFilteredTrades(setupA, gradeA, versionA);
  const listB = compareFilteredTrades(setupB, gradeB, versionB);
  if ($('compareTitleA')) $('compareTitleA').textContent = `${setupA || 'All setups'} · ${versionLabelA} · ${gradeA.length ? gradeA.join(', ') : 'All grades'}`;
  if ($('compareTitleB')) $('compareTitleB').textContent = `${setupB || 'All setups'} · ${versionLabelB} · ${gradeB.length ? gradeB.join(', ') : 'All grades'}`;
  if ($('compareStatsA')) $('compareStatsA').innerHTML = statsHtml(setupComparisonStats(listA));
  if ($('compareStatsB')) $('compareStatsB').innerHTML = statsHtml(setupComparisonStats(listB));
  renderCompareCharts('A', listA);
  renderCompareCharts('B', listB);
  const setupATradesAllGrades = setupA ? trades.filter(t => String(t.setup||'').toLowerCase() === setupA.toLowerCase()) : trades;
  const setupBTradesAllGrades = setupB ? trades.filter(t => String(t.setup||'').toLowerCase() === setupB.toLowerCase()) : trades;
  renderBarChart('chartSetupCompareGradesA', setupGradeAvgRBars(setupATradesAllGrades), { suffix:'R', decimals:2, tooltipLabel:'Avg R', rawLabels:true });
  renderBarChart('chartSetupCompareGradesB', setupGradeAvgRBars(setupBTradesAllGrades), { suffix:'R', decimals:2, tooltipLabel:'Avg R', rawLabels:true });
  const setupForBreakdown = $('breakdownSetupFilter')?.value || '';
  const setupTrades = setupForBreakdown ? trades.filter(t => String(t.setup||'').toLowerCase() === setupForBreakdown.toLowerCase()) : trades;
  const body = $('setupCompareBody'); if (!body) return;
  const groups = new Map();
  setupTrades.forEach(t => { const key = `${t.setup || 'Other'}|||${t.grade || 'Ungraded'}`; if (!groups.has(key)) groups.set(key, []); groups.get(key).push(t); });
  body.innerHTML = '';
  if (!groups.size) { body.innerHTML = '<tr><td colspan="7" class="empty">No trades match this setup yet.</td></tr>'; return; }
  const rows = Array.from(groups.entries()).map(([key, group]) => {
    const [setup, grade] = key.split('|||'); const st = setupComparisonStats(group);
    return { setup, grade, trades: st.trades, totalPL: st.totalPL, totalR: st.totalR, avgR: st.avgR, winPct: st.winPct };
  });
  const bdir = breakdownSort.dir === 'asc' ? 1 : -1;
  rows.sort((a,b) => {
    const av = a[breakdownSort.key], bv = b[breakdownSort.key];
    return compareMaybeGrade(av, bv, breakdownSort.key, bdir);
  });
  rows.forEach(st => {
    const tr = document.createElement('tr');
    tr.innerHTML = `<td>${escapeHtml(st.setup)}</td><td>${escapeHtml(st.grade)}</td><td>${st.trades}</td><td>${money(st.totalPL)}</td><td>${st.totalR.toFixed(2)}R</td><td>${st.avgR.toFixed(2)}R</td><td>${st.winPct.toFixed(1)}%</td>`;
    body.appendChild(tr);
  });
}

function normalizeTags() {
  if (!tags || Array.isArray(tags)) tags = { secondary: [], mistakes: [] };
  tags.secondary = normalizeTagList(tags.secondary || [], 'secondary-tag');
  tags.mistakes = normalizeTagList(tags.mistakes || [], 'mistake-tag');
}
function refreshTagOptions() {
  normalizeTags();
  const secondary = tagNames(tags.secondary);
  const mistakes = tagNames(tags.mistakes);
  const sec = $('editSecondaryTag');
  if (sec) { const cur = sec.value; sec.innerHTML = '<option value="">No secondary tag</option>' + secondary.map(n=>`<option ${n===cur?'selected':''}>${escapeHtml(n)}</option>`).join(''); sec.value = secondary.includes(cur) ? cur : ''; }
  const mis = $('editMistakeTag');
  if (mis) { const cur = mis.value; mis.innerHTML = '<option value="">No mistake tag</option>' + mistakes.map(n=>`<option ${n===cur?'selected':''}>${escapeHtml(n)}</option>`).join(''); mis.value = mistakes.includes(cur) ? cur : ''; }
}
function renderTags() {
  normalizeTags();
  const renderList = (id, list, type) => {
    const el = $(id); if (!el) return;
    if (!list.length) { el.innerHTML = '<p class="empty">No tags yet.</p>'; return; }
    el.innerHTML = '';
    list.forEach(record => {
      const name = record.name;
      const row = document.createElement('div'); row.className = 'tag-chip-row';
      row.innerHTML = `<span class="tag-chip">${escapeHtml(name)}</span><button type="button" class="editTagBtn">Edit</button><button type="button" class="danger">Delete</button>`;
      row.querySelector('.editTagBtn').addEventListener('click', () => {
        const next = prompt('Rename tag:', name);
        if (!next || !next.trim()) return;
        const clean = next.trim();
        const field = type === 'secondary' ? 'secondaryTag' : 'mistakeTag';
        tags[type] = normalizeTagList(tags[type].map(t => t.id === record.id ? { ...t, name: clean } : t), `${type}-tag`);
        trades = trades.map(t => String(t[field] || '') === name ? { ...t, [field]: clean } : t);
        saveTags(); save(); renderAll();
      });
      row.querySelector('button.danger').addEventListener('click', () => {
        if (!confirm('Delete this tag? Existing trades will keep the tag text.')) return;
        tags[type] = tags[type].filter(t => t.id !== record.id); saveTags(); renderAll();
      });
      el.appendChild(row);
    });
  };
  renderList('secondaryTagList', tags.secondary, 'secondary');
  renderList('mistakeTagList', tags.mistakes, 'mistakes');
}


function monthKeyForTrade(t) {
  return String(t.exitDate || t.date || '').slice(0,7);
}
function currentMonthKey() {
  const d = new Date();
  return `${d.getFullYear()}-${String(d.getMonth()+1).padStart(2,'0')}`;
}
function groupStats(list) {
  const closed = list.filter(t => (t.status || 'Closed') === 'Closed');
  const wins = closed.filter(t => getWinLoss(t) === 'Win');
  const totalPL = list.reduce((sum,t)=>sum+Number(t.pl||0),0);
  const totalR = list.reduce((sum,t)=>sum+Number(getRMultiple(t)||0),0);
  return { trades:list.length, totalPL, totalR, avgR: average(list.map(getRMultiple)), avgPL: average(list.map(t=>Number(t.pl||0))), winPct: closed.length ? wins.length/closed.length*100 : 0 };
}
function setupNameForTrade(t) { return String(t.setup || 'Other').trim() || 'Other'; }
function versionLabelForTrade(t) {
  const setup = getSetupByName(t.setup || '');
  return setupVersionLabel(tradeSetupVersion(t, setup));
}
function renderExpectancyBody(list, bodyId='expectancyBody') {
  const body = $(bodyId); if (!body) return;
  const groups = new Map();
  list.forEach(t => {
    const key = `${setupNameForTrade(t)}|||${versionLabelForTrade(t)}|||${String(t.grade || '').trim() || 'Ungraded'}`;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(t);
  });
  body.innerHTML = '';
  if (!groups.size) { body.innerHTML = '<tr><td colspan="8" class="empty">No trades available for this review.</td></tr>'; return; }

  const rows = Array.from(groups.entries()).map(([key, group]) => {
    const [setup, version, grade] = key.split('|||');
    const st = groupStats(group);
    return { setup, version, grade, ...st };
  });

  const sort = bodyId === 'expectancyBody' ? monthlyExpectancySort : { key: 'avgR', dir: 'desc' };
  const dir = sort.dir === 'asc' ? 1 : -1;
  rows.sort((a,b) => compareMaybeGrade(a[sort.key], b[sort.key], sort.key, dir));

  rows.forEach(row => {
    const tr = document.createElement('tr');
    tr.innerHTML = `<td>${escapeHtml(row.setup)}</td><td>${escapeHtml(row.version)}</td><td>${escapeHtml(row.grade)}</td><td>${row.trades}</td><td>${row.avgR.toFixed(2)}R</td><td>${money(row.avgPL)}</td><td>${row.winPct.toFixed(1)}%</td><td>${row.totalR.toFixed(2)}R</td>`;
    body.appendChild(tr);
  });
}
function renderMonthlyReview() {
  const monthInput = $('monthlyReviewMonth'); if (!monthInput) return;
  if (!monthInput.value) monthInput.value = currentMonthKey();
  const month = monthInput.value;
  const list = trades.filter(t => monthKeyForTrade(t) === month);
  const st = groupStats(list);
  const closed = list.filter(t => (t.status || 'Closed') === 'Closed');
  const wins = closed.filter(t => getWinLoss(t) === 'Win');
  if ($('monthlyPnL')) $('monthlyPnL').textContent = money(st.totalPL);
  if ($('monthlyR')) $('monthlyR').textContent = st.totalR.toFixed(2) + 'R';
  if ($('monthlyWinRate')) $('monthlyWinRate').textContent = closed.length ? Math.round(wins.length/closed.length*100) + '%' : '0%';
  const setupGroups = new Map();
  list.forEach(t => { const k = setupNameForTrade(t); if (!setupGroups.has(k)) setupGroups.set(k, []); setupGroups.get(k).push(t); });
  const rankedSetups = Array.from(setupGroups.entries()).map(([name, group]) => ({name, st: groupStats(group)})).sort((a,b)=>b.st.totalR-a.st.totalR);
  if ($('monthlyBestSetup')) $('monthlyBestSetup').textContent = rankedSetups[0] ? `${rankedSetups[0].name} (${rankedSetups[0].st.totalR.toFixed(2)}R)` : '—';
  const mistakeGroups = new Map();
  list.filter(t => t.mistakeTag).forEach(t => { const k=t.mistakeTag; if(!mistakeGroups.has(k)) mistakeGroups.set(k, []); mistakeGroups.get(k).push(t); });
  const rankedMistakes = Array.from(mistakeGroups.entries()).map(([name, group]) => ({name, count: group.length, st: groupStats(group)})).sort((a,b)=>b.count-a.count || a.st.totalR-b.st.totalR);
  if ($('monthlyWorstMistake')) $('monthlyWorstMistake').textContent = rankedMistakes[0] ? `${rankedMistakes[0].name} (${rankedMistakes[0].count})` : 'No Mistakes Tagged';
  if ($('monthlyFocus')) $('monthlyFocus').textContent = rankedMistakes[0] ? `Reduce ${rankedMistakes[0].name}` : (rankedSetups[0] ? `Scale ${rankedSetups[0].name}` : 'Add Trades');
  renderExpectancyBody(list);
  if ($('monthlyReviewNotes')) $('monthlyReviewNotes').value = (monthlyReviews || {})[month] || '';
}
function renderPlaybookFilters() {
  const setupSel = $('playbookSetup');
  const gradeSel = $('playbookGrade');
  const portfolioSel = $('playbookPortfolio');
  const mistakeSel = $('playbookMistake');
  if (setupSel) {
    const cur = getSelectValues('playbookSetup');
    const names = uniqueSorted([...setups.map(s=>s.name), ...trades.map(t=>setupNameForTrade(t) || t.setup)]);
    setupSel.innerHTML = optionListMulti(names, 'All setups', cur);
    setSelectValues('playbookSetup', cur.filter(v => names.includes(v)));
  }
  if (gradeSel) {
    const cur = getSelectValues('playbookGrade');
    const grades = uniqueSorted(trades.map(t => String(t.grade || '').trim() || 'Ungraded'));
    const ordered = GRADE_ORDER.filter(g=>grades.includes(g)).concat(grades.filter(g=>!GRADE_ORDER.includes(g)));
    gradeSel.innerHTML = optionListMulti(ordered, 'All grades', cur);
    setSelectValues('playbookGrade', cur.filter(v => ordered.includes(v)));
  }
  if (portfolioSel) {
    const cur = getSelectValues('playbookPortfolio');
    const portfoliosForFilter = uniqueSorted([...portfolios.map(p=>p.name), ...trades.map(t=>t.portfolioTag)]);
    portfolioSel.innerHTML = optionListMulti(portfoliosForFilter, 'All portfolios', cur);
    setSelectValues('playbookPortfolio', cur.filter(v => portfoliosForFilter.includes(v)));
  }
  if (mistakeSel) {
    const cur = getSelectValues('playbookMistake');
    normalizeTags();
    const mistakesForFilter = uniqueSorted([...tagNames(tags.mistakes), ...trades.map(t=>t.mistakeTag)]);
    mistakeSel.innerHTML = optionListMultiObjects(tagOptionsWithNone(mistakesForFilter, 'No Mistake'), 'All mistake tags', cur);
    setSelectValues('playbookMistake', cur.filter(v => v === '__NONE__' || mistakesForFilter.includes(v)));
  }
  refreshMultiSelectUI('playbookSetup');
  refreshMultiSelectUI('playbookGrade');
  refreshMultiSelectUI('playbookPortfolio');
  refreshMultiSelectUI('playbookMistake');
}
function screenshotSrc(shot) {
  if (!shot) return '';
  if (typeof shot === 'string') return shot;
  return shot.dataUrl || shot.data || shot.url || shot.src || '';
}

function camScreenshotId(shot) {
  const source = screenshotSrc(shot);
  const match = source.match(/\/api\/cam-journal\/screenshots\/([^/?#]+)/);
  return match ? decodeURIComponent(match[1]) : '';
}
function playbookShotIndex(trade, shots) {
  const idx = Number(trade.playbookScreenshotIndex);
  if (Number.isInteger(idx) && idx >= 0 && idx < shots.length) return idx;
  return 0;
}
function playbookTradeTimestamp(t) {
  const d = parseTradeDateTime(t.exitDate || t.date, t.exitTime || t.entryTime || '16:00:00');
  if (d && !Number.isNaN(d.getTime())) return d.getTime();
  const fallback = new Date(t.exitDate || t.date || 0);
  return Number.isNaN(fallback.getTime()) ? 0 : fallback.getTime();
}
function renderPlaybookPagination(total) {
  const pagerTargets = [
    { pager: $('playbookPaginationTop'), summary: $('playbookPageSummaryTop') },
    { pager: $('playbookPagination'), summary: $('playbookPageSummary') }
  ].filter(x => x.pager);
  if (!pagerTargets.length) return;
  const pages = Math.max(1, Math.ceil(total / PLAYBOOK_PAGE_SIZE));
  playbookPage = Math.min(Math.max(1, playbookPage), pages);
  const start = total ? ((playbookPage - 1) * PLAYBOOK_PAGE_SIZE) + 1 : 0;
  const end = Math.min(total, playbookPage * PLAYBOOK_PAGE_SIZE);
  const summaryText = total ? `Showing ${start}-${end} of ${total} playbook trades` : 'No playbook trades found';

  pagerTargets.forEach(({ pager, summary }) => {
    pager.innerHTML = '';
    if (summary) summary.textContent = summaryText;
    if (pages <= 1) return;

    const makeBtn = (label, page, disabled=false, active=false) => {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'pager-btn' + (active ? ' active' : '');
      btn.textContent = label;
      btn.disabled = disabled;
      btn.addEventListener('click', () => { playbookPage = page; renderPlaybook(); });
      pager.appendChild(btn);
    };

    makeBtn('‹', Math.max(1, playbookPage - 1), playbookPage === 1);
    const pagesToShow = [];
    for (let i = 1; i <= pages; i++) {
      if (i === 1 || i === pages || Math.abs(i - playbookPage) <= 2) pagesToShow.push(i);
    }
    let last = 0;
    pagesToShow.forEach(page => {
      if (last && page - last > 1) {
        const dots = document.createElement('span');
        dots.className = 'pager-dots';
        dots.textContent = '…';
        pager.appendChild(dots);
      }
      makeBtn(String(page), page, false, page === playbookPage);
      last = page;
    });
    makeBtn('›', Math.min(pages, playbookPage + 1), playbookPage === pages);
  });
}
function renderPlaybook() {
  const wrap = $('playbookCards'); if (!wrap) return;
  renderPlaybookFilters();
  const setupsSelected = getSelectValues('playbookSetup');
  const gradesSelected = getSelectValues('playbookGrade');
  const portfoliosSelected = getSelectValues('playbookPortfolio');
  const mistakesSelected = getSelectValues('playbookMistake');
  const onlyShots = $('playbookScreenshots')?.value === 'yes';
  let list = trades.filter(t => {
    if (!valueMatchesAny(setupNameForTrade(t), setupsSelected)) return false;
    if (!valueMatchesAny(String(t.grade || '').trim() || 'Ungraded', gradesSelected)) return false;
    if (!valueMatchesAny(t.portfolioTag || '', portfoliosSelected)) return false;
    if (!valueMatchesAny(t.mistakeTag || '', mistakesSelected)) return false;
    if (onlyShots && !(Array.isArray(t.screenshots) && t.screenshots.length)) return false;
    return true;
  }).sort((a,b)=>playbookTradeTimestamp(b)-playbookTradeTimestamp(a) || String(b.id||'').localeCompare(String(a.id||'')));
  renderPlaybookPagination(list.length);
  const paged = list.slice((playbookPage - 1) * PLAYBOOK_PAGE_SIZE, playbookPage * PLAYBOOK_PAGE_SIZE);
  wrap.innerHTML = '';
  if (!paged.length) { wrap.innerHTML = '<p class="empty">No playbook examples match those filters yet.</p>'; return; }
  paged.forEach(t => {
    const shots = Array.isArray(t.screenshots) ? t.screenshots : [];
    const selectedIndex = playbookShotIndex(t, shots);
    const card = document.createElement('div'); card.className = 'playbook-card';
    const thumb = screenshotSrc(shots[selectedIndex]);
    const shotOptions = shots.length > 1 ? `<label class="playbook-shot-picker">Card Screenshot<select class="playbookShotSelect">${shots.map((shot, idx) => `<option value="${idx}" ${idx===selectedIndex?'selected':''}>${escapeHtml(shot.name || `Screenshot ${idx+1}`)}</option>`).join('')}</select></label>` : '';
    card.innerHTML = `<div class="playbook-thumb">${thumb ? `<img src="${thumb}" alt="${escapeHtml(t.ticker || 'Trade screenshot')}" />` : '<span>No Screenshot</span>'}</div><div class="playbook-meta"><h3>${escapeHtml(t.ticker || 'Trade')}</h3><p>${escapeHtml(setupNameForTrade(t))} · ${escapeHtml(t.grade || 'Ungraded')} · ${Number(getRMultiple(t)||0).toFixed(2)}R</p><p>${formatDisplayDate(t.date)} → ${formatDisplayDate(t.exitDate)}</p>${shotOptions}<button type="button" class="openPlaybookTrade">Open Trade</button></div>`;
    const thumbImg = card.querySelector('.playbook-thumb img');
    if (thumbImg) thumbImg.addEventListener('click', () => {
      const large = $('largeScreenshot');
      const dialog = $('imageDialog');
      if (large && dialog) {
        large.src = thumb;
        large.alt = `${t.ticker || 'Trade'} playbook screenshot`;
        dialog.showModal();
      }
    });
    const picker = card.querySelector('.playbookShotSelect');
    if (picker) picker.addEventListener('change', () => {
      t.playbookScreenshotIndex = Number(picker.value);
      save();
      renderPlaybook();
    });
    card.querySelector('.openPlaybookTrade').addEventListener('click', () => openTradeDialog(t.id));
    wrap.appendChild(card);
  });
}
function healthIssuesForTrade(t) {
  const issues = [];
  if (!Number(t.risk || 0)) issues.push('Missing Risk');
  if (!setupNameForTrade(t) || setupNameForTrade(t) === 'Other') issues.push('Missing Setup');
  if (!(String(t.grade || '').trim())) issues.push('Ungraded');
  if (!(Array.isArray(t.screenshots) && t.screenshots.length)) issues.push('No Screenshots');
  if (!Number(t.stop || 0)) issues.push('Missing Stop');
  if (!String(t.notes || '').trim()) issues.push('Missing Notes');
  return issues;
}
function renderDataHealth() {
  if (!$('healthIssueBody')) return;
  const all = trades.map(t => ({ trade:t, issues:healthIssuesForTrade(t) })).filter(x=>x.issues.length);
  const count = name => all.filter(x=>x.issues.includes(name)).length;
  if ($('healthMissingRisk')) $('healthMissingRisk').textContent = count('Missing Risk');
  if ($('healthMissingSetup')) $('healthMissingSetup').textContent = count('Missing Setup');
  if ($('healthUngraded')) $('healthUngraded').textContent = count('Ungraded');
  if ($('healthNoScreenshots')) $('healthNoScreenshots').textContent = count('No Screenshots');
  if ($('healthMissingStop')) $('healthMissingStop').textContent = count('Missing Stop');
  if ($('healthMissingNotes')) $('healthMissingNotes').textContent = count('Missing Notes');
  const body = $('healthIssueBody'); body.innerHTML = '';
  if (!all.length) { body.innerHTML = '<tr><td colspan="6" class="empty">Data health looks clean.</td></tr>'; return; }
  all.slice(0,100).forEach(({trade, issues}) => {
    const tr = document.createElement('tr');
    tr.innerHTML = `<td>${formatDisplayDate(trade.date)}</td><td>${escapeHtml(trade.ticker || '')}</td><td>${escapeHtml(issues.join(', '))}</td><td>${escapeHtml(setupNameForTrade(trade))}</td><td>${escapeHtml(trade.grade || 'Ungraded')}</td><td><button type="button">Open</button></td>`;
    tr.querySelector('button').addEventListener('click', () => openTradeDialog(trade.id));
    body.appendChild(tr);
  });
}

function todayKey() {
  const date = new Date();
  return `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}-${String(date.getDate()).padStart(2, '0')}`;
}

function dateFromKey(key) {
  const match = String(key || '').match(/^(\d{4})-(\d{2})-(\d{2})$/);
  if (!match) return null;
  return new Date(Number(match[1]), Number(match[2]) - 1, Number(match[3]));
}

function watchlistInRange(item, selectedDate, range) {
  const itemDate = dateFromKey(item.date);
  const selected = dateFromKey(selectedDate) || new Date();
  if (!itemDate) return false;
  if (range === 'all') return true;
  if (range === 'month') return itemDate.getFullYear() === selected.getFullYear() && itemDate.getMonth() === selected.getMonth();
  if (range === 'week') {
    const end = new Date(selected.getFullYear(), selected.getMonth(), selected.getDate(), 23, 59, 59, 999);
    const start = new Date(end);
    start.setDate(start.getDate() - 6);
    start.setHours(0, 0, 0, 0);
    return itemDate >= start && itemDate <= end;
  }
  return String(item.date || '') === String(selectedDate || '');
}

function getFilteredWatchlistItems() {
  const selectedDate = $('watchlistDate')?.value || todayKey();
  const range = $('watchlistRange')?.value || 'day';
  const sort = $('watchlistSort')?.value || 'dateDesc';
  return (watchlistItems || [])
    .filter(item => watchlistInRange(item, selectedDate, range))
    .slice()
    .sort((a, b) => {
      if (sort === 'tickerAsc' || sort === 'tickerDesc') {
        const tickerOrder = String(a.ticker || '').localeCompare(String(b.ticker || ''));
        return sort === 'tickerAsc' ? tickerOrder : -tickerOrder;
      }
      const dateOrder = String(a.date || '').localeCompare(String(b.date || ''));
      if (dateOrder !== 0) return sort === 'dateAsc' ? dateOrder : -dateOrder;
      const createdOrder = String(a.createdAt || '').localeCompare(String(b.createdAt || ''));
      return sort === 'dateAsc' ? createdOrder : -createdOrder;
    });
}

function renderWatchlist() {
  const body = $('watchlistBody');
  if (!body) return;
  if ($('watchlistDate') && !$('watchlistDate').value) $('watchlistDate').value = todayKey();
  const items = getFilteredWatchlistItems();
  const range = $('watchlistRange')?.value || 'day';
  const selectedDate = $('watchlistDate')?.value || todayKey();
  if ($('watchlistSummary')) {
    const label = range === 'day'
      ? formatDisplayDate(selectedDate)
      : range === 'week'
        ? `7-day window ending ${formatDisplayDate(selectedDate)}`
        : range === 'month'
          ? `month containing ${formatDisplayDate(selectedDate)}`
          : 'all dates';
    $('watchlistSummary').textContent = `Showing ${items.length} watchlist idea${items.length === 1 ? '' : 's'} for ${label}.`;
  }
  body.innerHTML = '';
  if (!items.length) {
    body.innerHTML = '<tr><td colspan="5" class="empty">No watchlist ideas for this filter.</td></tr>';
    return;
  }
  items.forEach(item => {
    const row = document.createElement('tr');
    const added = item.createdAt ? new Date(item.createdAt) : null;
    const addedText = added && !Number.isNaN(added.getTime())
      ? `${formatDisplayDate(item.createdAt)} ${added.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })}`
      : '—';
    row.innerHTML = `<td>${formatDisplayDate(item.date)}</td><td><strong>${escapeHtml(String(item.ticker || '').toUpperCase())}</strong></td><td>${escapeHtml(item.note || '')}</td><td>${escapeHtml(addedText)}</td><td><button type="button" class="danger smallBtn">Delete</button></td>`;
    row.querySelector('button').addEventListener('click', () => {
      if (!confirm('Delete this watchlist idea?')) return;
      watchlistItems = watchlistItems.filter(candidate => candidate.id !== item.id);
      saveWatchlist();
      renderWatchlist();
    });
    body.appendChild(row);
  });
}

function renderAll() {
  normalizeTags();
  normalizeAllSetupVersions();
  refreshSetupOptions();
  refreshPortfolioOptions();
  refreshTagOptions();
  refreshReportFilterOptions();
  renderDashboardReports();
  renderTrades();
  renderSetups();
  renderPortfolios();
  renderTags();
  renderSetupComparison();
  renderMonthlyReview();
  renderPlaybook();
  renderWatchlist();
  renderDataHealth();
}

function formValue(id) { return $(id).value; }

if (form) form.addEventListener('submit', (event) => {
  event.preventDefault();
  const trade = {
    id: crypto.randomUUID(),
    date: formValue('date'),
    entryTime: cleanTime(formValue('entryTime')),
    exitDate: formValue('exitDate'),
    exitTime: cleanTime(formValue('exitTime')),
    ticker: formValue('ticker').toUpperCase().trim(),
    setup: formValue('setup'),
    grade: formValue('grade'),
    direction: formValue('direction'),
    entry: formValue('entry'),
    exit: formValue('exit'),
    stop: formValue('stop'),
    size: formValue('size'),
    risk: formValue('risk'),
    pl: formValue('pl'),
    rMultiple: formValue('rMultiple'),
    portfolioTag: formValue('portfolioTag'),
    emotion: formValue('emotion'),
    status: formValue('status'),
    checklist: formValue('checklist'),
    notes: formValue('notes'),
    screenshots: [],
    target: '',
    stfAtr: ''
  };
  trades.push(trade);
  save();
  form.reset();
  if ($('date')) $('date').valueAsDate = new Date();
  renderAll();
});


function openTradeDialog(id) {
  const trade = trades.find(t => t.id === id);
  if (!trade) return;
  loadingTradeDialog = true;
  tradeDialogOriginalTrade = JSON.parse(JSON.stringify(trade));
  tradeDialogSaved = false;
  tradeDialogDeleted = false;
  $('editId').value = trade.id;
  $('editTitle').textContent = `Edit ${trade.ticker || 'Trade'}`;
  const viewMap = {
    viewDate: formatDisplayDate(trade.date),
    viewEntryTime: formatDisplayTime(trade.entryTime),
    viewExitDate: formatDisplayDate(trade.exitDate),
    viewExitTime: formatDisplayTime(trade.exitTime),
    viewTicker: (trade.ticker || '—'),
    viewDirection: (trade.direction || '—'),
    viewSize: (trade.size || '—'),
    viewPl: money(trade.pl || 0),
    viewRMultiple: getRMultiple(trade) ? `${Number(getRMultiple(trade)).toFixed(2)}R` : '—'
  };
  Object.entries(viewMap).forEach(([id, value]) => { if ($(id)) $(id).textContent = value; });
  const map = {
    editSetup:'setup', editEntry:'entry', editExit:'exit', editStop:'stop', editTarget:'target', editRisk:'risk', editStfAtr:'stfAtr',
    editPortfolioTag:'portfolioTag', editGrade:'grade', editEmotion:'emotion', editSecondaryTag:'secondaryTag', editMistakeTag:'mistakeTag', editChecklist:'checklist', editNotes:'notes'
  };
  Object.entries(map).forEach(([inputId, field]) => {
    let value = trade[field] ?? '';
    if (field === 'entry' || field === 'exit' || field === 'stop' || field === 'target' || field === 'stfAtr') value = price4(value);
    if (field === 'notes' && String(value).startsWith('Imported from broker statement.')) value = '';
    $(inputId).value = value;
  });
  updateAtrStats();
  renderScreenshotGallery(trade);
  renderExecutionTable(trade);
  $('tradeDialog').showModal();
  setTimeout(() => { loadingTradeDialog = false; }, 0);
}

function renderExecutionTable(trade) {
  const body = $('executionTableBody');
  const empty = $('executionEmpty');
  if (!body) return;
  const executions = dedupeExecutions(syntheticExecutionsFromTrade(trade));
  body.innerHTML = '';
  if (!executions.length) {
    if (empty) empty.classList.remove('hidden');
    return;
  }

  if (empty) empty.classList.add('hidden');
  let runningPosition = 0;
  executions.forEach(execution => {
    runningPosition += Number(execution.signedQty || 0);
    const row = document.createElement('tr');
    row.innerHTML = `
      <td>${escapeHtml(execution.action === 'BOT' ? 'Buy' : 'Sell')}</td>
      <td>${escapeHtml(execution.date ? formatDisplayDate(execution.date) : '—')}<div class="small">${escapeHtml(formatDisplayTime(execution.time || ''))}</div></td>
      <td>${Number(execution.qty || 0)}</td>
      <td>${price4(execution.price)}</td>
      <td>${runningPosition}</td>
      <td>${money(execution.amount || 0)}</td>
    `;
    body.appendChild(row);
  });
}

function renderScreenshotGallery(trade) {
  const gallery = $('screenshotGallery');
  if (!gallery) return;
  const shots = trade.screenshots || [];
  gallery.innerHTML = '';
  if (!shots.length) {
    gallery.innerHTML = '<p class="small">No screenshots uploaded yet.</p>';
    return;
  }
  shots.forEach((shot, index) => {
    const wrap = document.createElement('div');
    wrap.className = 'screenshot-thumb-wrap';
    const img = document.createElement('img');
    img.className = 'screenshot-thumb';
    img.src = shot.dataUrl || shot;
    img.alt = shot.name || `Screenshot ${index + 1}`;
    img.addEventListener('click', () => {
      $('largeScreenshot').src = img.src;
      $('imageDialog').showModal();
    });
    const setPlaybook = document.createElement('button');
    setPlaybook.type = 'button';
    setPlaybook.className = 'set-playbook-shot';
    setPlaybook.textContent = Number(trade.playbookScreenshotIndex) === index ? 'Playbook ✓' : 'Use In Playbook';
    setPlaybook.title = 'Use this screenshot as the Playbook card image';
    setPlaybook.addEventListener('click', (event) => {
      event.stopPropagation();
      const current = trades.find(t => t.id === trade.id);
      if (!current) return;
      current.playbookScreenshotIndex = index;
      save();
      renderScreenshotGallery(current);
      renderPlaybook();
    });
    const remove = document.createElement('button');
    remove.type = 'button';
    remove.className = 'remove-shot';
    remove.textContent = '×';
    remove.title = 'Remove screenshot';
    remove.addEventListener('click', (event) => {
      event.stopPropagation();
      if (!confirm('Remove this screenshot from the trade?')) return;
      const current = trades.find(t => t.id === trade.id);
      if (!current) return;
      current.screenshots = (current.screenshots || []).filter((_, i) => i !== index);
      if (Number(current.playbookScreenshotIndex) === index) current.playbookScreenshotIndex = 0;
      else if (Number(current.playbookScreenshotIndex) > index) current.playbookScreenshotIndex = Number(current.playbookScreenshotIndex) - 1;
      save();
      renderScreenshotGallery(current);
      renderPlaybook();
    });
    wrap.appendChild(img);
    wrap.appendChild(setPlaybook);
    wrap.appendChild(remove);
    gallery.appendChild(wrap);
  });
}

function calculateStopFromRisk() {
  const trade = trades.find(t => t.id === $('editId').value) || {};
  const entry = Number($('editEntry').value || trade.entry);
  const size = Math.abs(Number(trade.size));
  const risk = Math.abs(Number($('editRisk').value));
  const direction = trade.direction || 'Long';
  if (!Number.isFinite(entry) || entry <= 0 || !Number.isFinite(size) || size <= 0 || !Number.isFinite(risk) || risk <= 0) {
    alert('Enter a valid entry, shares/contracts, and $ risk first.');
    return;
  }
  const riskPerShare = risk / size;
  const stop = direction === 'Short' ? entry + riskPerShare : entry - riskPerShare;
  $('editStop').value = stop.toFixed(4);
}

$('closeDialog').addEventListener('click', () => $('tradeDialog').close());

$('tradeDialog').addEventListener('click', (event) => {
  if (event.target === $('tradeDialog')) $('tradeDialog').close();
});

$('tradeDialog').addEventListener('close', () => {
  if (!tradeDialogSaved && !tradeDialogDeleted && tradeDialogOriginalTrade) {
    const idx = trades.findIndex(t => t.id === tradeDialogOriginalTrade.id);
    if (idx !== -1) {
      trades[idx] = JSON.parse(JSON.stringify(tradeDialogOriginalTrade));
      save();
      renderAll();
    }
  }
  tradeDialogOriginalTrade = null;
  tradeDialogSaved = false;
  tradeDialogDeleted = false;
});

$('editForm').addEventListener('submit', (event) => {
  event.preventDefault();
  const id = $('editId').value;
  const index = trades.findIndex(t => t.id === id);
  if (index === -1) return;
  const existing = trades[index];
  trades[index] = {
    ...existing,
    setup: $('editSetup').value.trim() || 'Other',
    setupId: getSetupByName($('editSetup').value.trim())?.id || existing.setupId || '',
    stop: $('editStop').value,
    target: $('editTarget').value,
    risk: $('editRisk').value,
    rMultiple: '',
    stfAtr: $('editStfAtr').value,
    portfolioTag: $('editPortfolioTag').value.trim(),
    grade: $('editGrade').value,
    emotion: $('editEmotion').value,
    secondaryTag: $('editSecondaryTag')?.value || '',
    mistakeTag: $('editMistakeTag')?.value || '',
    checklist: $('editChecklist').value,
    notes: $('editNotes').value,
    screenshots: existing.screenshots || []
  };
  tradeDialogSaved = true;
  save();
  $('tradeDialog').close();
  renderAll();
});

function deleteCurrentTradeFromDialog() {
  const id = $('editId').value;
  if (!confirm('Delete this trade?')) return;
  tradeDialogDeleted = true;
  trades = trades.filter(t => t.id !== id);
  save();
  $('tradeDialog').close();
  renderAll();
}
$('deleteFromDialog').addEventListener('click', deleteCurrentTradeFromDialog);
if ($('deleteFromDialogTop')) $('deleteFromDialogTop').addEventListener('click', deleteCurrentTradeFromDialog);


$('calcStopBtn').addEventListener('click', calculateStopFromRisk);
['editEntry','editStop','editTarget','editStfAtr'].forEach(id => { const el = $(id); if (el) el.addEventListener('input', updateAtrStats); });
$('screenshotInput').addEventListener('change', async (event) => {
  const id = $('editId').value;
  const trade = trades.find(t => t.id === id);
  if (!trade) return;
  const files = Array.from(event.target.files || []);
  if (!files.length) return;
  trade.screenshots = trade.screenshots || [];
  try {
    const added = await filesToImageObjects(files, 'trade', trade.id);
    trade.screenshots.push(...added);
    save();
    renderScreenshotGallery(trade);
  } catch (error) {
    console.error('Screenshot processing failed', error);
    alert(error?.message || 'Could not process this screenshot.');
  }
  event.target.value = '';
});
$('closeImageDialog').addEventListener('click', () => $('imageDialog').close());
$('imageDialog').addEventListener('click', (event) => {
  if (event.target.id === 'imageDialog') $('imageDialog').close();
});

$('search').addEventListener('input', renderTrades);
if ($('setupFilter')) $('setupFilter').addEventListener('change', renderTrades);
function refreshReportsAndTradeLog() {
  renderDashboardReports();
  renderTrades();
}
['reportStartDate','reportEndDate','reportTicker','reportSetup','reportPortfolio','reportDirection','reportResult','reportGrade','reportSecondaryTag','reportMistakeTag','logReportStartDate','logReportEndDate','logReportTicker','logReportSetup','logReportPortfolio','logReportDirection','logReportResult','logReportGrade','logReportSecondaryTag','logReportMistakeTag'].forEach(id => {
  const el = $(id);
  if (!el) return;
  const handler = () => { syncReportPair(id); refreshReportsAndTradeLog(); };
  el.addEventListener('input', handler);
  el.addEventListener('change', handler);
});
if ($('resetLogReportFilters')) $('resetLogReportFilters').addEventListener('click', () => $('resetReportFilters').click());
$('resetReportFilters').addEventListener('click', () => {
  ['reportStartDate','reportEndDate','reportDirection','logReportStartDate','logReportEndDate','logReportDirection'].forEach(id => { if ($(id)) $(id).value = ''; });
  MULTI_SELECT_IDS.forEach(id => setSelectValues(id, []));
  updateDatePickerLabels();
  refreshAllMultiSelectUI();
  refreshReportsAndTradeLog();
});

if ($('clearBtn')) $('clearBtn').addEventListener('click', () => {
  const first = confirm('Clear All Data?\n\nThis will permanently delete every saved trade from this browser. This cannot be undone.');
  if (!first) return;
  const typed = prompt('Type DELETE to confirm clearing all journal data.');
  if (typed !== 'DELETE') return;
  trades = [];
  setups = [];
  portfolios = [];
  tags = { secondary: [], mistakes: [] };
  monthlyReviews = {};
  watchlistItems = [];
  localStorage.removeItem(STORAGE_KEY);
  localStorage.removeItem(SETUPS_KEY);
  localStorage.removeItem(PORTFOLIOS_KEY);
  localStorage.removeItem(TAGS_KEY);
  localStorage.removeItem(REVIEWS_KEY);
  localStorage.removeItem(WATCHLIST_KEY);
  queueRemoteSave();
  renderAll();
});

function setupNameForExport(trade) {
  const linked = (trade?.setupId) ? setups.find(s => s.id === trade.setupId) : null;
  return linked?.name || trade?.setupScore?.setupName || trade?.setup || '';
}

function exportJournalCsv() {
  const headers = ['Date','Ticker','Quantity','Price Entered','Stop Loss','Target','','ATR','','Setup','Grade','Date closed','Price closed'];
  const csvValue = (value) => '"' + String(value ?? '').replaceAll('\r', ' ').replaceAll('\n', ' ').replaceAll('\"', '""') + '"';
  const rows = (trades || []).map(t => [
    t.date || '',
    t.ticker || '',
    t.size || '',
    t.entry || '',
    t.stop || '',
    t.target || '',
    '',
    t.stfAtr || '',
    '',
    setupNameForExport(t),
    t.grade || '',
    t.exitDate || '',
    t.exit || ''
  ].map(csvValue).join(','));
  const csv = [headers.map(csvValue).join(','), ...rows].join('\n');
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'trade-journal-export.csv';
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

if ($('exportBtn')) $('exportBtn').addEventListener('click', exportJournalCsv);
if ($('settingsExportBtn')) $('settingsExportBtn').addEventListener('click', exportJournalCsv);
if ($('backupExportBtn')) $('backupExportBtn').addEventListener('click', async () => {
  try {
    await downloadFullBackup();
  } catch (error) {
    console.error('Full backup export failed', error);
    alert(error?.message || 'Could not export the full backup.');
  }
});
if ($('backupImportInput')) $('backupImportInput').addEventListener('change', event => {
  const file = event.target.files && event.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = async () => {
    try { await importFullBackupPayload(JSON.parse(reader.result)); }
    catch (err) { console.error(err); alert(err.message || 'Could not import this backup file.'); }
    event.target.value = '';
  };
  reader.readAsText(file);
});

function clearReportAndTradeLogFiltersForImport() {
  [
    'reportStartDate', 'reportEndDate', 'reportDirection', 'reportResult',
    'logReportStartDate', 'logReportEndDate', 'logReportDirection', 'logReportResult'
  ].forEach(id => {
    if ($(id)) $(id).value = '';
  });
  if ($('search')) $('search').value = '';
  if ($('setupFilter')) $('setupFilter').value = '';
  if (typeof MULTI_SELECT_IDS !== 'undefined') {
    MULTI_SELECT_IDS.forEach(id => setSelectValues(id, []));
  }
  updateDatePickerLabels();
  refreshAllMultiSelectUI();
}

async function importBrokerCsvFile(file) {
  if (!file) return;
  try {
    const rows = parseCsv(await file.text());
    const executions = statementExecutions(rows);
    if (!executions.length) {
      showImportStatus('<strong>No broker trades found.</strong><br><span class="small">This importer looks for Thinkorswim/Schwab account statement rows with TYPE = TRD. If this file has trades, send the CSV format so another parser can be added.</span>');
      return;
    }

    const importedTrades = groupExecutionsIntoTrades(executions);
    const result = upsertImportedTrades(importedTrades);
    clearReportAndTradeLogFiltersForImport();
    selectedTradeIds.clear();
    save();
    renderAll();
    const closed = importedTrades.filter(trade => trade.status === 'Closed').length;
    const open = importedTrades.filter(trade => trade.status === 'Open').length;
    const totalPL = importedTrades.reduce((sum, trade) => sum + Number(trade.pl || 0), 0);
    showImportStatus(`<strong>Import complete:</strong> ${executions.length} executions were checked against your journal. New: ${result.added}. Updated: ${result.updated}. Duplicates ignored: ${result.ignored}. Open trades closed by this upload: ${result.closedOpen}.<br><span class="small">This upload contained ${importedTrades.length} grouped trades. Closed in file: ${closed}. Open in file: ${open}. File P/L: ${money(totalPL)}. Your setup, grade, checklist, emotion, and custom notes are preserved when an imported trade is updated.</span>`);
  } catch (error) {
    console.error('Broker CSV import failed', error);
    showImportStatus(`<strong>Broker CSV import failed.</strong><br><span class="small">${escapeHtml(error?.message || String(error))}</span>`);
  }
}

if ($('settingsImportInput')) $('settingsImportInput').addEventListener('change', async event => {
  const file = event.target.files && event.target.files[0];
  await importBrokerCsvFile(file);
  event.target.value = '';
});

function parseCsv(text) {
  text = text.replace(/^\uFEFF/, '');
  const rows = [];
  let row = [], cell = '', inQuotes = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    const next = text[i + 1];
    if (ch === '"') {
      if (inQuotes && next === '"') { cell += '"'; i++; }
      else { inQuotes = !inQuotes; }
    } else if (ch === ',' && !inQuotes) {
      row.push(cell); cell = '';
    } else if ((ch === '\n' || ch === '\r') && !inQuotes) {
      if (ch === '\r' && next === '\n') i++;
      row.push(cell); rows.push(row); row = []; cell = '';
    } else {
      cell += ch;
    }
  }
  if (cell.length || row.length) { row.push(cell); rows.push(row); }
  return rows.filter(r => r.some(c => String(c).trim() !== ''));
}

function cleanNumber(value) {
  if (value === undefined || value === null) return 0;
  const cleaned = String(value).replace(/[$,()]/g, '').trim();
  if (!cleaned || cleaned === '--') return 0;
  const sign = String(value).includes('(') && String(value).includes(')') ? -1 : 1;
  const n = Number(cleaned);
  return Number.isFinite(n) ? n * sign : 0;
}

function normalizeDate(mmddyy) {
  const parts = String(mmddyy).split('/');
  if (parts.length !== 3) return mmddyy;
  const [m, d, yy] = parts.map(x => x.padStart(2, '0'));
  const year = Number(yy) < 70 ? '20' + yy : '19' + yy;
  return `${year}-${m}-${d}`;
}

function parseTradeDescription(description) {
  const text = String(description || '').trim();
  const match = text.match(/^(BOT|SOLD)\s+([+-]?\d+)\s+(.+?)\s+@(\.?\d+(?:\.\d+)?)/i);
  if (!match) return null;
  const action = match[1].toUpperCase();
  const qty = Math.abs(Number(match[2]));
  const instrument = match[3].trim();
  const price = Number(match[4]);
  const ticker = (instrument.split(/\s+/)[0] || '').toUpperCase();
  const isOption = /\b(CALL|PUT)\b/i.test(instrument);
  const signedQty = action === 'BOT' ? qty : -qty;
  return { action, qty, signedQty, instrument, ticker, price, isOption };
}

function statementExecutions(rows) {
  const executions = [];
  let header = null;
  for (const row of rows) {
    const firstCell = String(row[0] || '').trim();
    const secondCell = String(row[1] || '').trim();
    const typeCell = String(row[2] || '').trim().toUpperCase();
    if (firstCell === 'Futures Statements') break;
    if (firstCell.toUpperCase() === 'DATE' && secondCell.toUpperCase() === 'TIME' && typeCell === 'TYPE') {
      header = row;
      continue;
    }
    if (!header || typeCell !== 'TRD') continue;
    const desc = parseTradeDescription(row[4]);
    if (!desc) continue;
    executions.push({
      date: normalizeDate(row[0]),
      time: row[1] || '',
      type: row[2],
      ref: row[3] || '',
      description: row[4] || '',
      miscFees: cleanNumber(row[5]),
      commissions: cleanNumber(row[6]),
      amount: cleanNumber(row[7]),
      balance: cleanNumber(row[8]),
      ...desc,
      executionKey: [normalizeDate(row[0]), row[1] || '', row[3] || '', row[4] || '', row[7] || ''].join('|')
    });
  }
  return executions;
}

function summarizeRoundTrip(key, execs, closed=true) {
  const first = execs[0];
  const last = execs[execs.length - 1];
  const openingSign = Math.sign(first.signedQty);
  const direction = openingSign > 0 ? 'Long' : 'Short';
  const buys = execs.filter(e => e.action === 'BOT');
  const sells = execs.filter(e => e.action === 'SOLD');
  const boughtQty = buys.reduce((s,e)=>s+e.qty,0);
  const soldQty = sells.reduce((s,e)=>s+e.qty,0);
  const openingExecs = direction === 'Long' ? buys : sells;
  const closingExecs = direction === 'Long' ? sells : buys;
  const openingQty = openingExecs.reduce((s,e)=>s+e.qty,0);
  const closingQty = closingExecs.reduce((s,e)=>s+e.qty,0);
  const entry = openingQty ? openingExecs.reduce((s,e)=>s+e.qty*e.price,0) / openingQty : '';
  const exit = closingQty ? closingExecs.reduce((s,e)=>s+e.qty*e.price,0) / closingQty : '';
  const totalCash = execs.reduce((s,e)=>s+e.amount+e.miscFees+e.commissions,0);
  const pl = closed ? totalCash : '';
  const sourceLines = execs.map(e => `${e.date} ${e.time} ${e.description} amount ${money(e.amount)}`).join('\n');
  const executionKeys = execs.map(e => e.executionKey);
  const openingKeys = openingExecs.map(e => e.executionKey);
  return {
    id: crypto.randomUUID(),
    date: first.date,
    entryTime: first.time,
    exitDate: closed ? last.date : '',
    exitTime: closed ? last.time : '',
    ticker: first.ticker,
    instrument: first.instrument,
    setup: 'Other',
    grade: '',
    direction,
    entry: numberOrBlank(entry),
    exit: numberOrBlank(exit),
    stop: '',
    size: Math.max(boughtQty, soldQty),
    risk: '',
    pl: closed ? Number(pl.toFixed(2)) : '',
    rMultiple: '',
    portfolioTag: '',
    emotion: '',
    status: closed ? 'Closed' : 'Open',
    checklist: '',
    notes: '',
    source: 'Broker CSV import',
    screenshots: [],
    importOpenKey: `${first.instrument}|${direction}|${openingKeys.join('~')}`,
    importTradeKey: `${first.instrument}|${executionKeys.join('~')}`,
    executionKeys,
    rawExecutions: execs.map(execution => ({
      date: execution.date,
      time: execution.time,
      action: execution.action,
      qty: Number(execution.qty || 0),
      signedQty: Number(execution.signedQty || 0),
      price: Number(execution.price || 0),
      instrument: execution.instrument,
      ticker: execution.ticker,
      amount: Number(execution.amount || 0),
      miscFees: Number(execution.miscFees || 0),
      commissions: Number(execution.commissions || 0),
      description: execution.description || '',
      executionKey: execution.executionKey || [execution.date, execution.time, execution.description, execution.amount].join('|')
    }))
  };
}

function preserveManualFields(existing, imported) {
  const manualFields = ['setup','grade','stop','target','risk','stfAtr','rMultiple','portfolioTag','secondaryTag','mistakeTag','emotion','checklist'];
  const merged = { ...imported, id: existing.id || imported.id };
  manualFields.forEach(field => {
    if (existing[field] !== undefined && existing[field] !== null && String(existing[field]).trim() !== '' && !(field === 'setup' && existing[field] === 'Other')) {
      merged[field] = existing[field];
    }
  });
  if (existing.notes !== undefined && existing.notes !== null) {
    merged.notes = cleanBrokerImportNoteText(existing.notes);
  }
  return merged;
}

function syntheticExecutionsFromTrade(trade) {
  if (Array.isArray(trade.rawExecutions) && trade.rawExecutions.length) {
    return trade.rawExecutions.map(execution => ({ ...execution }));
  }

  const instrument = trade.instrument || trade.ticker || trade.symbol || '';
  const ticker = trade.ticker || trade.symbol || String(instrument).split(/\s+/)[0] || '';
  const qty = Math.abs(Number(trade.size || trade.quantity || trade.shares || trade.qty || 0));
  const entry = Number(trade.entry || 0);
  const exit = Number(trade.exit || 0);
  if (!qty || !entry) return [];

  const isLong = String(trade.direction || 'Long').toLowerCase() !== 'short';
  const openAction = isLong ? 'BOT' : 'SOLD';
  const closeAction = isLong ? 'SOLD' : 'BOT';
  const executions = [{
    date: trade.date || '',
    time: trade.entryTime || '',
    action: openAction,
    qty,
    signedQty: isLong ? qty : -qty,
    price: entry,
    instrument,
    ticker,
    amount: isLong ? -(qty * entry) : qty * entry,
    miscFees: 0,
    commissions: 0,
    description: `${openAction} ${isLong ? '+' : '-'}${qty} ${instrument} @${entry}`,
    executionKey: trade.executionKeys?.[0] || `legacy-open|${trade.id}`
  }];

  if (String(trade.status || '').toLowerCase() === 'closed' && exit) {
    executions.push({
      date: trade.exitDate || trade.date || '',
      time: trade.exitTime || '',
      action: closeAction,
      qty,
      signedQty: isLong ? -qty : qty,
      price: exit,
      instrument,
      ticker,
      amount: isLong ? qty * exit : -(qty * exit),
      miscFees: 0,
      commissions: 0,
      description: `${closeAction} ${isLong ? '-' : '+'}${qty} ${instrument} @${exit}`,
      executionKey: trade.executionKeys?.[1] || `legacy-close|${trade.id}`
    });
  }

  return executions;
}

function dedupeExecutions(executions) {
  const seen = new Set();
  return (executions || []).filter(execution => {
    const key = execution.executionKey || [
      execution.date,
      execution.time,
      execution.action,
      execution.qty,
      execution.instrument,
      execution.price,
      execution.amount
    ].join('|');
    if (seen.has(key)) return false;
    seen.add(key);
    execution.executionKey = key;
    return true;
  }).sort((a, b) => ((a.date || '') + (a.time || '')).localeCompare((b.date || '') + (b.time || '')));
}

function netExecutionPosition(executions) {
  return (executions || []).reduce((sum, execution) => {
    const signedQty = Number(execution.signedQty);
    if (Number.isFinite(signedQty) && signedQty !== 0) return sum + signedQty;
    return sum + (String(execution.action).toUpperCase() === 'BOT' ? Number(execution.qty || 0) : -Number(execution.qty || 0));
  }, 0);
}

function reconcileWithExistingOpen(imported) {
  const importedExecutions = imported.rawExecutions || [];
  if (!importedExecutions.length) return imported;

  const candidateIndex = trades.findIndex(trade => {
    if (String(trade.status || '').toLowerCase() !== 'open') return false;
    const sameInstrument = trade.instrument && imported.instrument && trade.instrument === imported.instrument;
    const sameTicker = String(trade.ticker || trade.symbol || '').toUpperCase() === String(imported.ticker || '').toUpperCase();
    const bothHaveInstruments = Boolean(trade.instrument && imported.instrument);
    if (bothHaveInstruments ? !sameInstrument : !sameTicker) return false;
    const existingExecutions = syntheticExecutionsFromTrade(trade);
    if (!existingExecutions.length) return false;
    const existingNet = netExecutionPosition(existingExecutions);
    const incomingNet = netExecutionPosition(importedExecutions);
    return existingNet !== 0 && incomingNet !== 0;
  });
  if (candidateIndex === -1) return imported;

  const existing = trades[candidateIndex];
  const combined = dedupeExecutions([...syntheticExecutionsFromTrade(existing), ...importedExecutions]);
  const netPosition = netExecutionPosition(combined);
  const rebuilt = summarizeRoundTrip(imported.instrument || existing.instrument || imported.ticker, combined, netPosition === 0);
  const merged = preserveManualFields(existing, rebuilt);
  merged.id = existing.id;
  merged.screenshots = existing.screenshots || [];
  merged.notes = existing.notes || merged.notes || '';
  merged.setupScore = existing.setupScore;
  merged.playbookScreenshotIndex = existing.playbookScreenshotIndex;
  merged._reconciledExistingIndex = candidateIndex;
  return merged;
}

function upsertImportedTrades(importedTrades) {
  let added = 0, updated = 0, ignored = 0, closedOpen = 0;

  function executionOverlapInfo(existing, importedKeys) {
    const existingKeys = new Set(existing.executionKeys || []);
    if (!existingKeys.size || !importedKeys.size) return { overlap: 0, existingKeys };
    let overlap = 0;
    importedKeys.forEach(key => { if (existingKeys.has(key)) overlap++; });
    return { overlap, existingKeys };
  }

  function rebuildExistingWithImported(existing, imported) {
    const combined = dedupeExecutions([
      ...syntheticExecutionsFromTrade(existing),
      ...(imported.rawExecutions || [])
    ]);
    const netPosition = netExecutionPosition(combined);
    const rebuilt = summarizeRoundTrip(
      existing.instrument || imported.instrument || existing.ticker || imported.ticker,
      combined,
      netPosition === 0
    );
    const merged = preserveManualFields(existing, rebuilt);
    merged.id = existing.id;
    merged.screenshots = existing.screenshots || [];
    merged.setupScore = existing.setupScore;
    merged.playbookScreenshotIndex = existing.playbookScreenshotIndex;
    if (String(existing.status || '').toLowerCase() === 'open' && String(merged.status || '').toLowerCase() === 'closed') closedOpen++;
    return merged;
  }

  for (let imported of importedTrades) {
    imported = reconcileWithExistingOpen(imported);
    const importedKeys = new Set(imported.executionKeys || []);
    const reconciledExistingIndex = Number.isInteger(imported._reconciledExistingIndex)
      ? imported._reconciledExistingIndex
      : null;
    let index = reconciledExistingIndex ?? trades.findIndex(trade => trade.importTradeKey && trade.importTradeKey === imported.importTradeKey);
    if (imported._reconciledExistingIndex !== undefined) delete imported._reconciledExistingIndex;

    if (index === -1 && imported.importOpenKey) {
      index = trades.findIndex(t => t.importOpenKey && t.importOpenKey === imported.importOpenKey);
      if (index !== -1 && trades[index].status === 'Open' && imported.status === 'Closed') closedOpen++;
    }

    if (index === -1 && importedKeys.size) {
      index = trades.findIndex(t => {
        const info = executionOverlapInfo(t, importedKeys);
        return info.overlap > 0;
      });
    }

    if (index !== -1 && importedKeys.size) {
      const existing = trades[index];
      const { overlap, existingKeys } = executionOverlapInfo(existing, importedKeys);
      if (overlap === importedKeys.size && existingKeys.size >= importedKeys.size) {
        ignored++;
        continue;
      }
      if (overlap > 0 || reconciledExistingIndex !== null) {
        const before = JSON.stringify(existing);
        trades[index] = rebuildExistingWithImported(existing, imported);
        if (before === JSON.stringify(trades[index])) ignored++;
        else updated++;
        continue;
      }
    }

    if (index === -1) {
      trades.push(imported);
      added++;
    } else {
      const before = JSON.stringify(trades[index]);
      trades[index] = preserveManualFields(trades[index], imported);
      if (before === JSON.stringify(trades[index])) ignored++;
      else updated++;
    }
  }
  return { added, updated, ignored, closedOpen };
}

function groupExecutionsIntoTrades(executions) {
  const byInstrument = new Map();
  executions.forEach(e => {
    const key = e.instrument;
    if (!byInstrument.has(key)) byInstrument.set(key, []);
    byInstrument.get(key).push(e);
  });
  const imported = [];
  for (const [key, list] of byInstrument.entries()) {
    list.sort((a,b) => (a.date + a.time).localeCompare(b.date + b.time));
    let position = 0;
    let bucket = [];
    for (const e of list) {
      const before = position;
      position += e.signedQty;
      bucket.push(e);
      if (before !== 0 && position === 0) {
        imported.push(summarizeRoundTrip(key, bucket, true));
        bucket = [];
      }
    }
    if (bucket.length) imported.push(summarizeRoundTrip(key, bucket, false));
  }
  imported.sort((a,b) => ((a.date || '') + (a.entryTime || '')).localeCompare((b.date || '') + (b.entryTime || '')));
  return imported;
}

function showImportStatus(message) {
  const box = $('importStatus');
  box.innerHTML = message;
  box.classList.remove('hidden');
}

if ($('importInput')) $('importInput').addEventListener('change', async event => {
  const file = event.target.files && event.target.files[0];
  await importBrokerCsvFile(file);
  event.target.value = '';
});



const GRADE_CHOICES = ['A+','A','A-','B+','B','B-','C+','C'];
function defaultGradeRules() {
  return [
    { grade: 'A+', min: 90, max: 100 },
    { grade: 'A', min: 80, max: 90 },
    { grade: 'B+', min: 70, max: 80 },
    { grade: 'B', min: 60, max: 70 },
    { grade: 'C', min: 0, max: 60 }
  ];
}
function normalizeGradeRules(rules) {
  if (!Array.isArray(rules) || !rules.length) return defaultGradeRules();
  if (typeof rules[0] === 'object') return rules.slice(0,5).concat(defaultGradeRules()).slice(0,5).map((r,i)=>{
    const def = defaultGradeRules()[i];
    let min = Number(r.min ?? def.min);
    let max = Number(r.max ?? def.max);
    if (max === 101) max = 100;
    if (max === 59.99) max = 60;
    return { grade: r.grade || def.grade, min, max };
  });
  const parsed = rules.map((txt, i) => {
    const m = String(txt||'').match(/(A\+|A-|A|B\+|B-|B|C\+|C).*?(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)/i);
    return m ? { grade: m[1].toUpperCase(), min: Number(m[2]), max: Number(m[3]) } : defaultGradeRules()[i];
  });
  return parsed.concat(defaultGradeRules()).slice(0,5);
}
function escapeHtml(value) {
  return String(value ?? '').replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');
}
function renderGradeRuleRows(rules) {
  const box = $('gradeRulesArea'); if (!box) return;
  const normalized = normalizeGradeRules(rules);
  box.innerHTML = '';
  normalized.forEach((r, i) => {
    const div = document.createElement('div');
    div.className = 'grade-rule-grid';
    div.innerHTML = `<select class="gradeRuleGrade">${GRADE_CHOICES.map(g=>`<option ${g===r.grade?'selected':''}>${g}</option>`).join('')}</select><input class="gradeRuleMin" type="number" min="0" max="100" step="1" value="${r.min}"><input class="gradeRuleMax" type="number" min="0" max="100" step="1" value="${r.max}">`;
    box.appendChild(div);
  });
}
function readGradeRules() {
  return Array.from(document.querySelectorAll('#gradeRulesArea .grade-rule-grid')).map(row => ({
    grade: row.querySelector('.gradeRuleGrade').value,
    min: Number(row.querySelector('.gradeRuleMin').value || 0),
    max: Number(row.querySelector('.gradeRuleMax').value || 0)
  }));
}
function gradeFromSetupPercent(setup, percent) {
  const rules = normalizeGradeRules(setup?.gradeRules);
  const p = Math.min(100, Math.max(0, Number(percent || 0)));
  const match = rules.find(r => p >= Number(r.min) && (p < Number(r.max) || (Number(r.max) >= 100 && p <= Number(r.max))));
  if (match) return match.grade;
  return p >= 90 ? 'A+' : p >= 80 ? 'A' : p >= 70 ? 'B+' : p >= 60 ? 'B' : 'C';
}
function renderImageGallery(galleryId, shots, onRemove) {
  const gallery = $(galleryId); if (!gallery) return;
  if (!shots || !shots.length) { gallery.innerHTML = '<p class="small">No screenshots uploaded yet.</p>'; return; }
  gallery.innerHTML = '';
  shots.forEach((shot, index) => {
    const wrap = document.createElement('div'); wrap.className = 'screenshot-thumb-wrap';
    const img = document.createElement('img'); img.className = 'screenshot-thumb'; img.src = shot.dataUrl; img.alt = shot.name || 'Screenshot';
    img.addEventListener('click', () => { $('largeScreenshot').src = shot.dataUrl; $('imageDialog').showModal(); });
    const remove = document.createElement('button'); remove.type = 'button'; remove.className = 'removeShot'; remove.textContent = '×';
    remove.addEventListener('click', () => onRemove(index));
    wrap.appendChild(img); wrap.appendChild(remove); gallery.appendChild(wrap);
  });
}

function blobToDataUrl(blob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result || ''));
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(blob);
  });
}

function dataUrlToBlob(dataUrl) {
  const [header, encoded = ''] = String(dataUrl || '').split(',');
  const mime = header.match(/^data:([^;]+)/)?.[1] || 'image/jpeg';
  const bytes = atob(encoded);
  const output = new Uint8Array(bytes.length);
  for (let index = 0; index < bytes.length; index += 1) output[index] = bytes.charCodeAt(index);
  return new Blob([output], { type: mime });
}

async function compressImageBlob(blob, maxDimension = 1600, targetBytes = 450000) {
  if (!blob || !String(blob.type || '').startsWith('image/')) return blob;
  const objectUrl = URL.createObjectURL(blob);
  try {
    const image = await new Promise((resolve, reject) => {
      const element = new Image();
      element.onload = () => resolve(element);
      element.onerror = () => reject(new Error('Could not read this screenshot.'));
      element.src = objectUrl;
    });
    const scale = Math.min(1, maxDimension / Math.max(image.naturalWidth || image.width, image.naturalHeight || image.height));
    const width = Math.max(1, Math.round((image.naturalWidth || image.width) * scale));
    const height = Math.max(1, Math.round((image.naturalHeight || image.height) * scale));
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const context = canvas.getContext('2d');
    context.fillStyle = '#ffffff';
    context.fillRect(0, 0, width, height);
    context.drawImage(image, 0, 0, width, height);

    let quality = 0.82;
    let compressed = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', quality));
    while (compressed && compressed.size > targetBytes && quality > 0.48) {
      quality -= 0.08;
      compressed = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', quality));
    }
    return compressed || blob;
  } finally {
    URL.revokeObjectURL(objectUrl);
  }
}

async function imageFileToObject(file) {
  const compressed = await compressImageBlob(file);
  return {
    name: String(file.name || 'screenshot').replace(/\.[^.]+$/, '') + '.jpg',
    dataUrl: await blobToDataUrl(compressed),
    addedAt: new Date().toISOString(),
    originalBytes: Number(file.size || 0),
    storedBytes: Number(compressed.size || 0)
  };
}

async function uploadCamScreenshot(entityType, entityId, blob, fileName) {
  if (!remoteStateLoaded) throw new Error('The complete shared journal must load before screenshots can be saved.');
  const formData = new FormData();
  formData.append('file', blob, fileName || 'screenshot.jpg');
  formData.append('entityType', entityType);
  formData.append('entityId', entityId);
  const response = await fetch('/api/cam-journal/screenshots', { method: 'POST', body: formData });
  const data = await response.json().catch(() => ({}));
  if (!response.ok || !data.url) throw new Error(data.error || 'Could not save screenshot.');
  return { id: data.id, name: fileName || 'screenshot.jpg', dataUrl: data.url, url: data.url, addedAt: new Date().toISOString() };
}

async function uploadScreenshotObject(entityType, entityId, shot) {
  const source = screenshotSrc(shot);
  if (!source.startsWith('data:image/')) return shot;
  const compressed = await compressImageBlob(dataUrlToBlob(source));
  const name = String(shot?.name || 'screenshot').replace(/\.[^.]+$/, '') + '.jpg';
  return uploadCamScreenshot(entityType, entityId, compressed, name);
}

async function filesToImageObjects(fileList, entityType, entityId) {
  const out = [];
  for (const file of Array.from(fileList || [])) {
    if (!file.type.startsWith('image/')) continue;
    const compressed = await compressImageBlob(file);
    out.push(await uploadCamScreenshot(
      entityType,
      entityId,
      compressed,
      String(file.name || 'screenshot').replace(/\.[^.]+$/, '') + '.jpg'
    ));
  }
  return out;
}

async function migrateEmbeddedScreenshots(data) {
  const next = deepClone(data);
  for (const trade of next.trades || []) {
    const migrated = [];
    for (const shot of trade.screenshots || []) migrated.push(await uploadScreenshotObject('trade', trade.id, shot));
    trade.screenshots = migrated;
  }
  for (const setup of next.setups || []) {
    const migrated = [];
    for (const shot of setup.screenshots || []) migrated.push(await uploadScreenshotObject('setup', setup.id, shot));
    setup.screenshots = migrated;
    for (const version of setup.versions || []) {
      const versionShots = [];
      for (const shot of version.screenshots || []) {
        versionShots.push(await uploadScreenshotObject('setup-version', `${setup.id}:${version.version}`, shot));
      }
      version.screenshots = versionShots;
    }
  }
  return next;
}


function deepClone(obj) { return JSON.parse(JSON.stringify(obj || {})); }
function setupSnapshot(setup, versionNumber) {
  return {
    version: Number(versionNumber || setup?.currentVersion || 1),
    label: `Version ${Number(versionNumber || setup?.currentVersion || 1)}`,
    createdAt: new Date().toISOString(),
    description: setup?.description || '',
    gradeRules: deepClone(setup?.gradeRules || defaultGradeRules()),
    screenshots: deepClone(setup?.screenshots || []),
    enableEtf: setup?.enableEtf !== false,
    criteria: deepClone(setup?.criteria || { stf: [], mtf: [], etf: [] })
  };
}
function normalizeSetupVersions(setup) {
  if (!setup) return setup;
  if (!setup.id) setup.id = crypto.randomUUID();
  if (!Array.isArray(setup.versions)) setup.versions = [];
  setup.versions = setup.versions
    .filter(Boolean)
    .map((v, i) => ({ ...v, version: Number(v.version || i + 1), label: `Version ${Number(v.version || i + 1)}` }))
    .sort((a,b) => Number(a.version || 0) - Number(b.version || 0));
  if (!setup.versions.length) {
    setup.currentVersion = Number(setup.currentVersion || 1);
    setup.versions = [setupSnapshot(setup, setup.currentVersion)];
  }
  const highest = Math.max(...setup.versions.map(v => Number(v.version || 1)), Number(setup.currentVersion || 1));
  setup.currentVersion = highest;
  // If the saved setup claims a newer current version than the history contains, preserve it as a snapshot instead of losing it.
  if (!setup.versions.some(v => Number(v.version) === highest)) {
    setup.versions.push(setupSnapshot(setup, highest));
  }
  setup.versions = setup.versions
    .map((v, i) => ({ ...v, version: Number(v.version || i + 1), label: `Version ${Number(v.version || i + 1)}` }))
    .sort((a,b) => Number(a.version || 0) - Number(b.version || 0));
  return setup;
}
function normalizeAllSetupVersions() { setups = (setups || []).map(normalizeSetupVersions); }
function setupCoreChanged(oldSetup, newSetup) {
  const pick = (s) => JSON.stringify({ criteria: s?.criteria || {}, gradeRules: normalizeGradeRules(s?.gradeRules), enableEtf: s?.enableEtf !== false });
  return pick(oldSetup) !== pick(newSetup);
}
function currentVersionForSetup(setup) { normalizeSetupVersions(setup); return Number(setup?.currentVersion || 1); }
function tradeSetupVersion(trade, setup) { return Number(trade?.setupScore?.setupVersion || trade?.setupVersion || currentVersionForSetup(setup)); }
function setupVersionLabel(n) { return `Version ${Number(n || 1)}`; }
function gradeRuleLabel(r) {
  return `${r.grade}: ≥${Number(r.min || 0)}% To <${Number(r.max || 0)}%`;
}
function criteriaSignature(c) {
  return `${String(c.text||'').trim()}|||${Number(c.max||0)}|||${c.important ? '1' : '0'}`;
}
function describeVersionChanges(prev, cur) {
  if (!prev) return ['Initial setup version.'];
  const changes = [];
  const prevRules = normalizeGradeRules(prev.gradeRules);
  const curRules = normalizeGradeRules(cur.gradeRules);
  curRules.forEach(rule => {
    const old = prevRules.find(r => r.grade === rule.grade);
    if (!old) changes.push(`Grade ${rule.grade} Rule Added: ${gradeRuleLabel(rule)}`);
    else if (Number(old.min) !== Number(rule.min) || Number(old.max) !== Number(rule.max)) {
      changes.push(`${rule.grade} Grade % Changed: ≥${Number(old.min)}% To <${Number(old.max)}% → ≥${Number(rule.min)}% To <${Number(rule.max)}%`);
    }
  });
  prevRules.forEach(rule => { if (!curRules.some(r => r.grade === rule.grade)) changes.push(`Grade ${rule.grade} Rule Removed`); });
  if ((prev.enableEtf !== false) !== (cur.enableEtf !== false)) changes.push(`Execution Time Frame ${cur.enableEtf !== false ? 'Enabled' : 'Hidden'}`);
  ['stf','mtf','etf'].forEach(sec => {
    const title = sec === 'stf' ? 'Scanning Time Frame' : sec === 'mtf' ? 'Monitoring Time Frame' : 'Execution Time Frame';
    const oldRows = prev.criteria?.[sec] || [];
    const newRows = cur.criteria?.[sec] || [];
    const oldTexts = new Map(oldRows.map(c => [String(c.text||'').trim().toLowerCase(), c]));
    const newTexts = new Map(newRows.map(c => [String(c.text||'').trim().toLowerCase(), c]));
    newRows.forEach(c => {
      const key = String(c.text||'').trim().toLowerCase();
      const old = oldTexts.get(key);
      if (!old) changes.push(`${title} Criteria Added: ${c.text}`);
      else {
        if (Number(old.max||0) !== Number(c.max||0)) changes.push(`${title} Max Points Changed: ${c.text} (${Number(old.max||0)} → ${Number(c.max||0)})`);
        if (!!old.important !== !!c.important) changes.push(`${title} Star Changed: ${c.text} (${old.important ? 'Starred' : 'Not Starred'} → ${c.important ? 'Starred' : 'Not Starred'})`);
      }
    });
    oldRows.forEach(c => {
      const key = String(c.text||'').trim().toLowerCase();
      if (!newTexts.has(key)) changes.push(`${title} Criteria Removed: ${c.text}`);
    });
  });
  return changes.length ? changes : ['No criteria or grade-rule changes recorded.'];
}
function versionSectionTitle(sec) {
  return sec === 'stf' ? 'Scanning Time Frame' : sec === 'mtf' ? 'Monitoring Time Frame' : 'Execution Time Frame';
}
function renderVersionDetails(setup, version, previousVersion) {
  const changes = describeVersionChanges(previousVersion, version);
  const sections = ['stf','mtf','etf'].filter(sec => !(sec === 'etf' && version.enableEtf === false));
  return `
    <div class="version-detail-grid">
      <div class="version-detail-card">
        <h4>Created</h4>
        <p>${version.createdAt ? new Date(version.createdAt).toLocaleString() : 'Unknown'}</p>
      </div>
      <div class="version-detail-card">
        <h4>Status</h4>
        <p>${Number(version.version) === Number(setup.currentVersion) ? 'Current Version' : 'Past Version'}</p>
      </div>
    </div>
    <div class="version-block">
      <h4>Changes</h4>
      <ul>${changes.map(c => `<li>${escapeHtml(c)}</li>`).join('')}</ul>
    </div>
    <div class="version-block">
      <h4>Grade Rules</h4>
      <ul>${normalizeGradeRules(version.gradeRules).map(r => `<li>${escapeHtml(gradeRuleLabel(r))}</li>`).join('')}</ul>
    </div>
    ${sections.map(sec => `
      <div class="version-block">
        <h4>${versionSectionTitle(sec)}</h4>
        ${(version.criteria?.[sec] || []).length ? `<ul>${(version.criteria?.[sec] || []).map(c => `<li>${c.important ? '⭐ ' : ''}${escapeHtml(c.text || 'Untitled Criteria')} <span class="muted">(${Number(c.max || 0)} pts)</span></li>`).join('')}</ul>` : '<p class="small">No criteria saved for this time frame.</p>'}
      </div>
    `).join('')}
  `;
}
function viewSetupVersions(setup) {
  normalizeSetupVersions(setup);
  const dialog = $('versionDialog');
  const content = $('versionDialogContent');
  const title = $('versionDialogTitle');
  const versions = (setup.versions || []).slice().sort((a,b) => Number(a.version || 0) - Number(b.version || 0));
  if (!dialog || !content || !title) {
    const lines = [`${setup.name} Version History`, ''];
    versions.forEach((v, idx) => {
      const prev = idx > 0 ? versions[idx - 1] : null;
      lines.push(`${setupVersionLabel(v.version)}${Number(v.version) === Number(setup.currentVersion) ? ' (Current)' : ''}`);
      lines.push(`Created: ${v.createdAt ? new Date(v.createdAt).toLocaleString() : 'Unknown'}`);
      lines.push('Changes:');
      describeVersionChanges(prev, v).forEach(change => lines.push(`  • ${change}`));
      lines.push('');
    });
    alert(lines.join('\n'));
    return;
  }
  title.textContent = `${setup.name} Version History`;
  content.innerHTML = versions.map((v, idx) => {
    const prev = idx > 0 ? versions[idx - 1] : null;
    const isCurrent = Number(v.version) === Number(setup.currentVersion);
    return `
      <details class="version-item" ${idx === versions.length - 1 ? 'open' : ''}>
        <summary>
          <span>${setupVersionLabel(v.version)}${isCurrent ? ' · Current' : ''}</span>
          <span class="small">${v.createdAt ? new Date(v.createdAt).toLocaleDateString() : 'Unknown Date'}</span>
        </summary>
        ${renderVersionDetails(setup, v, prev)}
      </details>
    `;
  }).join('') || '<p class="empty">No versions saved yet.</p>';
  dialog.showModal();
}


function defaultCriteria(section) {
  return [{ text: '', important: false, max: 1 }];
}
function ensureDefaultSetups() {
  normalizeAllSetupVersions();
  if (setups.length) { saveSetups(); return; }
  const firstSetup = normalizeSetupVersions({ id: crypto.randomUUID(), name: 'BBC', description: 'Bollinger Band Capitulation reversal setup.', gradeRules: defaultGradeRules(), screenshots: [], enableEtf: true, criteria: { stf: defaultCriteria('stf'), mtf: defaultCriteria('mtf'), etf: defaultCriteria('etf') } });
  setups.push(firstSetup);
  saveSetups();
saveTags();
}
function renderCriteriaRows(section, rows) {
  const box = $('criteria-' + section); if (!box) return;
  box.innerHTML = '';
  (rows || []).forEach((c, i) => {
    c.max = Number(c.max || (c.important ? 2 : 1));
    const div = document.createElement('div');
    div.className = 'criteria-row';
    div.innerHTML = `<button type="button" class="starBtn ${c.important ? 'active' : ''}" title="Mark as important">⭐</button><input type="text" class="criteriaText" value="${escapeHtml(c.text)}" placeholder="Enter Criteria Here" /><label>Max Points<input type="number" class="criteriaMax" min="0" step="1" value="${c.max}" /></label><button type="button" class="removeCriteria danger">Remove</button>`;
    div.querySelector('.starBtn').addEventListener('click', () => {
      const btn = div.querySelector('.starBtn');
      btn.classList.toggle('active');
      if (btn.classList.contains('active') && Number(div.querySelector('.criteriaMax').value || 0) < 2) div.querySelector('.criteriaMax').value = 2;
    });
    div.querySelector('.removeCriteria').addEventListener('click', () => { rows.splice(i,1); renderCriteriaRows(section, rows); });
    box.appendChild(div);
  });
}
function readCriteriaRows(section) {
  return Array.from(document.querySelectorAll(`#criteria-${section} .criteria-row`)).map(row => ({
    text: row.querySelector('.criteriaText').value.trim(),
    important: row.querySelector('.starBtn').classList.contains('active'),
    max: Math.max(0, Number(row.querySelector('.criteriaMax').value || 0))
  })).filter(c => c.text);
}
function readSetupDraft() {
  const id = $('setupId')?.value || crypto.randomUUID();
  const existing = setups.find(s => s.id === id);
  return { id, name: $('setupName')?.value.trim() || '', description: $('setupDescription')?.value || '', gradeRules: readGradeRules(), screenshots: currentSetupScreenshots || [], enableEtf: $('enableEtf')?.checked ?? true, criteria: { stf: readCriteriaRows('stf'), mtf: readCriteriaRows('mtf'), etf: readCriteriaRows('etf') }, versions: existing?.versions || [], currentVersion: existing?.currentVersion || 1 };
}
function fillSetupForm(setup, mode = 'edit') {
  if (!$('setupForm')) return;
  if ($('setupSaveMode')) $('setupSaveMode').value = mode;
  $('setupId').value = setup.id || '';
  $('setupName').value = setup.name || '';
  $('setupDescription').value = setup.description || '';
  renderGradeRuleRows(setup.gradeRules || defaultGradeRules());
  currentSetupScreenshots = JSON.parse(JSON.stringify(setup.screenshots || []));
  renderSetupScreenshotGallery();
  $('enableEtf').checked = setup.enableEtf !== false;
  renderCriteriaRows('stf', JSON.parse(JSON.stringify(setup.criteria?.stf || [])));
  renderCriteriaRows('mtf', JSON.parse(JSON.stringify(setup.criteria?.mtf || [])));
  renderCriteriaRows('etf', JSON.parse(JSON.stringify(setup.criteria?.etf || [])));
  const submitButton = document.querySelector('#setupForm button[type="submit"]');
  if (submitButton) {
    submitButton.textContent = mode === 'newVersion'
      ? `Save ${setupVersionLabel(Number(setup.currentVersion || 1) + 1)}`
      : 'Save Setup';
  }
}
function renderSetupScreenshotGallery() { renderImageGallery('setupScreenshotGallery', currentSetupScreenshots, (index) => { currentSetupScreenshots.splice(index,1); renderSetupScreenshotGallery(); }); }
function renderFeedbackScreenshotGallery() {
  renderImageGallery('feedbackScreenshotGallery', feedbackScreenshots, (index) => {
    feedbackScreenshots.splice(index, 1);
    renderFeedbackScreenshotGallery();
  });
}
function renderFeedbackReplyScreenshotGallery() {
  renderImageGallery('feedbackReplyScreenshotGallery', feedbackReplyScreenshots, (index) => {
    feedbackReplyScreenshots.splice(index, 1);
    renderFeedbackReplyScreenshotGallery();
  });
}
function feedbackStatusLabel(status) {
  const value = String(status || 'OPEN').toUpperCase();
  return value === 'IN_PROGRESS' ? 'In Progress' : value === 'COMPLETED' ? 'Completed' : 'Open';
}
function feedbackStatusClass(status) {
  return String(status || 'OPEN').toLowerCase();
}
function resetFeedbackForm() {
  $('feedbackForm').reset();
  $('feedbackKind').value = 'BUG';
  document.querySelectorAll('.feedbackKindBtn').forEach(item => item.classList.toggle('active', item.dataset.kind === 'BUG'));
  $('bugFields').classList.remove('hidden');
  $('featureFields').classList.add('hidden');
  feedbackScreenshots = [];
  renderFeedbackScreenshotGallery();
}
function activeFeedbackTicket() {
  return feedbackTickets.find(ticket => ticket.id === activeFeedbackTicketId) || null;
}
function renderFeedbackThreadList() {
  const list = $('feedbackThreadList');
  if (!list) return;
  if (!feedbackTickets.length) {
    list.innerHTML = '<p class="small">No threads yet.</p>';
    return;
  }
  list.innerHTML = '';
  feedbackTickets.forEach(ticket => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = `feedback-thread-item ${ticket.id === activeFeedbackTicketId ? 'active' : ''}`;
    btn.innerHTML = `<strong>${escapeHtml(ticket.title)}</strong><span>${escapeHtml(feedbackStatusLabel(ticket.status))}</span><p>${escapeHtml(ticket.summary || '')}</p>`;
    btn.addEventListener('click', () => {
      activeFeedbackTicketId = ticket.id;
      renderFeedbackThreadView();
      renderFeedbackThreadList();
    });
    list.appendChild(btn);
  });
}
function renderFeedbackMessageList(ticket) {
  const list = $('feedbackMessageList');
  if (!list) return;
  if (!ticket || !ticket.messages || !ticket.messages.length) {
    list.innerHTML = '<p class="small">No messages yet.</p>';
    return;
  }
  list.innerHTML = '';
  ticket.messages.forEach(message => {
    const wrap = document.createElement('div');
    wrap.className = `feedback-message ${message.author === 'ADMIN' ? 'admin' : 'cam'}`;
    const shots = Array.isArray(message.screenshots) && message.screenshots.length
      ? `<div class="feedback-message-shots">${message.screenshots.map((shot, index) => `<img src="${shot}" alt="Attachment ${index + 1}" data-shot="${index}" />`).join('')}</div>`
      : '';
    wrap.innerHTML = `<div class="feedback-message-head"><strong>${message.author === 'ADMIN' ? 'Codex' : 'Cam'}</strong><span>${new Date(message.createdAt).toLocaleString()}</span></div><p>${escapeHtml(message.body)}</p>${shots}`;
    wrap.querySelectorAll('img').forEach((img, index) => img.addEventListener('click', () => {
      $('largeScreenshot').src = message.screenshots[index];
      $('imageDialog').showModal();
    }));
    list.appendChild(wrap);
  });
}
function renderFeedbackThreadView() {
  const ticket = activeFeedbackTicket();
  const newPanel = $('feedbackNewTicketPanel');
  const threadPanel = $('feedbackThreadPanel');
  const emptyState = $('feedbackThreadEmpty');
  if (!newPanel || !threadPanel || !emptyState) return;
  if (!ticket) {
    newPanel.classList.remove('hidden');
    threadPanel.classList.add('hidden');
    emptyState.classList.add('hidden');
    return;
  }
  newPanel.classList.add('hidden');
  threadPanel.classList.remove('hidden');
  emptyState.classList.add('hidden');
  $('feedbackActiveTitle').textContent = ticket.title || 'Conversation';
  $('feedbackActiveMeta').textContent = `${ticket.kind} · ${ticket.submittedBy} · ${new Date(ticket.updatedAt).toLocaleString()}`;
  const pill = $('feedbackActiveStatus');
  pill.textContent = feedbackStatusLabel(ticket.status);
  pill.className = `feedback-status-pill ${feedbackStatusClass(ticket.status)}`;
  renderFeedbackMessageList(ticket);
  renderFeedbackReplyScreenshotGallery();
}
async function loadFeedbackTickets(nextSelectedId) {
  try {
    const response = await fetch('/api/tickets');
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Could not load support threads.');
    feedbackTickets = Array.isArray(data.tickets) ? data.tickets : [];
    if (nextSelectedId) activeFeedbackTicketId = nextSelectedId;
    if (!activeFeedbackTicketId && feedbackTickets.length) activeFeedbackTicketId = feedbackTickets[0].id;
    if (activeFeedbackTicketId && !feedbackTickets.some(ticket => ticket.id === activeFeedbackTicketId)) {
      activeFeedbackTicketId = feedbackTickets[0]?.id || '';
    }
    renderFeedbackThreadList();
    renderFeedbackThreadView();
  } catch (err) {
    console.error('Feedback thread load failed', err);
    const list = $('feedbackThreadList');
    if (list) list.innerHTML = '<p class="small">Could not load support threads.</p>';
  }
}
function newSetupDraft() { return { id: '', name: '', description: '', gradeRules: defaultGradeRules(), screenshots: [], enableEtf: true, criteria: { stf: defaultCriteria('stf'), mtf: defaultCriteria('mtf'), etf: [] } }; }
function startNewSetupVersion(setup) {
  fillSetupForm(deepClone(setup), 'newVersion');
  window.scrollTo({ top: 0, behavior: 'smooth' });
}
function cancelSetupEdit() {
  fillSetupForm(newSetupDraft(), 'edit');
  if ($('setupScreenshotInput')) $('setupScreenshotInput').value = '';
}
function renderSetups() {
  const list = $('setupList'); if (!list) return;
  if (!setups.length) { list.innerHTML = '<p class="empty">No setups yet. Create your first setup above.</p>'; return; }
  list.innerHTML = '';
  setups.forEach(setup => {
    const max = getSetupMax(setup);
    const div = document.createElement('div');
    div.className = 'setup-card';
    div.innerHTML = `<div><h3>${setup.name}</h3><p>${setup.description || ''}</p><span class="small">${setupVersionLabel(currentVersionForSetup(setup))} · Max score: ${max} points · ETF ${setup.enableEtf === false ? 'hidden' : 'shown'} · Screenshots: ${(setup.screenshots||[]).length}</span></div><div><button type="button" class="editSetupBtn">Edit Current</button><button type="button" class="newVersionBtn">Create New Version</button><button type="button" class="viewVersionsBtn">View Versions</button><button type="button" class="deleteSetupBtn danger">Delete</button></div>`;
    div.querySelector('.editSetupBtn').addEventListener('click', () => fillSetupForm(setup, 'edit'));
    div.querySelector('.newVersionBtn').addEventListener('click', () => startNewSetupVersion(setup));
    div.querySelector('.viewVersionsBtn').addEventListener('click', () => viewSetupVersions(setup));
    div.querySelector('.deleteSetupBtn').addEventListener('click', () => { if(confirm('Delete this setup?')) { setups = setups.filter(s => s.id !== setup.id); saveSetups();
saveTags(); renderAll(); } });
    list.appendChild(div);
  });
}
function refreshSetupOptions() {
  const savedNames = uniqueSorted(setups.map(s=>s.name));
  const defaultNames = Array.from(new Set([...savedNames, 'BBC','ILEL','Iron Condor','Other']));
  const optionHtml = defaultNames.map(n => `<option>${escapeHtml(n)}</option>`).join('');
  const savedOptionHtml = savedNames.map(n => `<option>${escapeHtml(n)}</option>`).join('');
  const sel = $('setup');
  if (sel) { const current = sel.value; sel.innerHTML = optionHtml; sel.value = defaultNames.includes(current) ? current : (defaultNames[0] || 'Other'); }
  const editSel = $('editSetup');
  if (editSel) {
    const current = editSel.value;
    editSel.innerHTML = savedNames.length ? '<option value="">Select saved setup</option>' + savedOptionHtml : '<option value="">No saved setups yet</option>';
    editSel.value = savedNames.includes(current) ? current : '';
  }
  const filter = $('setupFilter');
  if (filter) { const fcur = filter.value; filter.innerHTML = '<option value="">All setups</option>' + optionHtml; filter.value = fcur; }
  refreshBulkActionValue();
}
function getSetupByName(name) { return setups.find(s => s.name.toLowerCase() === String(name||'').trim().toLowerCase()); }
function getSetupMax(setup) { return ['stf','mtf','etf'].reduce((sum, sec) => sum + ((sec==='etf' && setup.enableEtf===false) ? 0 : (setup.criteria?.[sec]||[]).reduce((s,c)=>s+Number(c.max||0),0)), 0); }
function gradeFromPercent(percent) { return gradeFromSetupPercent(gradingSetup, percent); }
function setupSections(setup) { return ['stf','mtf','etf'].filter(sec => !(sec === 'etf' && setup.enableEtf === false)); }
function buildFrozenGradingSetup(trade, currentSetup) {
  // Once a trade has been graded, keep the exact criteria/rules that existed on grading day.
  // The displayed setup name can still update through setupId renames, but the scorecard itself is frozen.
  if (trade?.setupScore?.criteriaSnapshot) {
    return {
      id: trade.setupScore.setupId || currentSetup?.id || trade.setupId || crypto.randomUUID(),
      name: currentSetup?.name || trade.setupScore.setupName || trade.setup || 'Saved setup',
      criteria: JSON.parse(JSON.stringify(trade.setupScore.criteriaSnapshot || {})),
      gradeRules: JSON.parse(JSON.stringify(trade.setupScore.gradeRulesSnapshot || currentSetup?.gradeRules || defaultGradeRules())),
      enableEtf: trade.setupScore.enableEtf !== false,
      currentVersion: Number(trade.setupScore.setupVersion || currentSetup?.currentVersion || 1)
    };
  }
  return currentSetup;
}
function openGradeDialogForTrade(tradeId) {
  const trade = trades.find(t => t.id === tradeId); if (!trade) return;
  const currentSetup = getSetupByName($('editSetup')?.value || trade.setup);
  if (!currentSetup && !trade.setupScore?.criteriaSnapshot) { alert('No saved setup matches this trade setup tag. Create it in Setup Builder first, or make sure the names match exactly.'); return; }
  const setup = buildFrozenGradingSetup(trade, currentSetup);
  gradingSetup = setup;
  $('gradeTradeId').value = tradeId;
  $('gradeDialogTitle').textContent = `Grade ${trade.ticker || 'Trade'} · ${setup.name} ${setupVersionLabel(setup.currentVersion || trade.setupScore?.setupVersion || 1)}${trade.setupScore?.criteriaSnapshot ? ' · Saved criteria' : ''}`;
  const area = $('gradeCriteriaArea'); area.innerHTML = '';
  const savedScores = trade.setupScore?.scores || {};
  setupSections(setup).forEach(sec => {
    const card = document.createElement('section'); card.className = 'card grade-section';
    const title = sec === 'stf' ? 'Scanning Time Frame' : sec === 'mtf' ? 'Monitoring Time Frame' : 'Execution Time Frame';
    card.innerHTML = `<h3>${title}</h3>`;
    (setup.criteria?.[sec] || []).forEach((c, i) => {
      const key = `${sec}-${i}`;
      const row = document.createElement('div'); row.className = 'grade-row';
      row.innerHTML = `<div>${c.important ? '⭐ ' : ''}${c.text}</div><label>Score / ${c.max}<input type="number" min="0" max="${c.max}" step="1" data-max="${c.max}" data-key="${key}" value="${savedScores[key] ?? ''}" /></label>`;
      card.appendChild(row);
    });
    area.appendChild(card);
  });
  area.querySelectorAll('input').forEach(inp => inp.addEventListener('input', updateGradeDialogTotal));
  updateGradeDialogTotal();
  $('gradeDialog').showModal();
}
function updateGradeDialogTotal() {
  const inputs = Array.from(document.querySelectorAll('#gradeCriteriaArea input'));
  const earned = inputs.reduce((s,i)=>s+Math.min(Number(i.value||0), Number(i.dataset.max||0)),0);
  const max = inputs.reduce((s,i)=>s+Number(i.dataset.max||0),0);
  const pct = max ? earned / max * 100 : 0;
  $('gradeTotalScore').textContent = earned;
  $('gradeMaxScore').textContent = max;
  $('gradeResult').textContent = max ? gradeFromPercent(pct) : 'Ungraded';
}

if ($('portfolioForm')) $('portfolioForm').addEventListener('submit', (event) => {
  event.preventDefault();
  const portfolio = { id: $('portfolioId').value || crypto.randomUUID(), name: $('portfolioName').value.trim(), description: $('portfolioDescription').value.trim() };
  if (!portfolio.name) return alert('Please name the portfolio.');
  const index = portfolios.findIndex(p => p.id === portfolio.id || p.name.toLowerCase() === portfolio.name.toLowerCase());
  if (index >= 0) portfolios[index] = { ...portfolio, id: portfolios[index].id };
  else portfolios.push({ ...portfolio, id: crypto.randomUUID() });
  savePortfolios(); fillPortfolioForm(); renderAll();
});
if ($('newPortfolioBtn')) $('newPortfolioBtn').addEventListener('click', () => fillPortfolioForm());

if ($('setupForm')) $('setupForm').addEventListener('submit', (event) => {
  event.preventDefault();
  let setup = readSetupDraft();
  if (!setup.name) return alert('Please name the setup.');
  const mode = $('setupSaveMode')?.value || 'edit';
  const index = setups.findIndex(s => s.id === setup.id || s.name.toLowerCase() === setup.name.toLowerCase());
  if (index >= 0) {
    const old = normalizeSetupVersions(setups[index]);
    const stableId = old.id || setup.id || crypto.randomUUID();
    let versions = deepClone(old.versions || []);
    const highestSavedVersion = versions.length ? Math.max(...versions.map(v => Number(v.version || 1))) : Number(old.currentVersion || 1);
    let nextVersion = Number(old.currentVersion || highestSavedVersion || 1);
    if (mode === 'newVersion') {
      nextVersion = Math.max(highestSavedVersion, Number(old.currentVersion || 1)) + 1;
      versions.push(setupSnapshot({ ...setup, id: stableId, currentVersion: nextVersion }, nextVersion));
    } else {
      nextVersion = Math.max(Number(old.currentVersion || 1), 1);
      const currentSnapshot = setupSnapshot({ ...setup, id: stableId, currentVersion: nextVersion }, nextVersion);
      const existingIndex = versions.findIndex(version => Number(version.version) === nextVersion);
      if (existingIndex >= 0) {
        versions[existingIndex] = {
          ...versions[existingIndex],
          ...currentSnapshot,
          createdAt: versions[existingIndex].createdAt || currentSnapshot.createdAt,
          editedAt: new Date().toISOString()
        };
      } else {
        versions.push(currentSnapshot);
      }
    }
    versions = versions.sort((a,b) => Number(a.version || 0) - Number(b.version || 0));
    setup = { ...setup, id: stableId, currentVersion: nextVersion, versions };
    setups[index] = setup;
    if (old.name !== setup.name) {
      trades = trades.map(t => ((t.setupId && t.setupId === stableId) || String(t.setup||'').toLowerCase() === String(old.name||'').toLowerCase()) ? { ...t, setup: setup.name, setupId: stableId, setupScore: t.setupScore ? { ...t.setupScore, setupName: setup.name, setupId: stableId } : t.setupScore } : t);
      save();
    }
  } else {
    setup = normalizeSetupVersions({ ...setup, id: setup.id || crypto.randomUUID(), currentVersion: 1, versions: [] });
    setups.push(setup);
  }
  saveSetups();
  saveTags();
  fillSetupForm(newSetupDraft(), 'edit');
  renderAll();
});
document.querySelectorAll('.addCriteriaBtn').forEach(btn => btn.addEventListener('click', () => {
  const sec = btn.dataset.section;
  const draft = readSetupDraft();
  const mode = $('setupSaveMode')?.value || 'edit';
  draft.criteria[sec].push({ text: '', important: false, max: 1 });
  fillSetupForm(draft, mode);
}));
if ($('newSetupBtn')) $('newSetupBtn').addEventListener('click', () => fillSetupForm(newSetupDraft(), 'edit'));
if ($('cancelSetupBtn')) $('cancelSetupBtn').addEventListener('click', cancelSetupEdit);
if ($('setupScreenshotInput')) $('setupScreenshotInput').addEventListener('change', async (event) => {
  let setupId = $('setupId')?.value || '';
  if (!setupId) {
    setupId = stableId('setup');
    if ($('setupId')) $('setupId').value = setupId;
  }
  const added = await filesToImageObjects(event.target.files, 'setup', setupId);
  currentSetupScreenshots.push(...added);
  renderSetupScreenshotGallery();
  event.target.value = '';
});
if ($('feedbackLauncher')) $('feedbackLauncher').addEventListener('click', () => {
  loadFeedbackTickets();
  renderFeedbackScreenshotGallery();
  renderFeedbackReplyScreenshotGallery();
  $('feedbackDialog').showModal();
  clearInterval(feedbackPollTimer);
  feedbackPollTimer = setInterval(() => loadFeedbackTickets(activeFeedbackTicketId), 15000);
});
if ($('closeFeedbackDialog')) $('closeFeedbackDialog').addEventListener('click', () => { clearInterval(feedbackPollTimer); $('feedbackDialog').close(); });
if ($('cancelFeedbackBtn')) $('cancelFeedbackBtn').addEventListener('click', () => { clearInterval(feedbackPollTimer); $('feedbackDialog').close(); });
if ($('newFeedbackThreadBtn')) $('newFeedbackThreadBtn').addEventListener('click', () => {
  activeFeedbackTicketId = '';
  resetFeedbackForm();
  renderFeedbackThreadView();
  renderFeedbackThreadList();
});
if ($('backToFeedbackNewBtn')) $('backToFeedbackNewBtn').addEventListener('click', () => {
  activeFeedbackTicketId = '';
  resetFeedbackForm();
  renderFeedbackThreadView();
  renderFeedbackThreadList();
});
document.querySelectorAll('.feedbackKindBtn').forEach(btn => btn.addEventListener('click', () => {
  const kind = btn.dataset.kind === 'FEATURE' ? 'FEATURE' : 'BUG';
  $('feedbackKind').value = kind;
  document.querySelectorAll('.feedbackKindBtn').forEach(item => item.classList.toggle('active', item === btn));
  $('bugFields').classList.toggle('hidden', kind !== 'BUG');
  $('featureFields').classList.toggle('hidden', kind !== 'FEATURE');
}));
if ($('feedbackScreenshotInput')) $('feedbackScreenshotInput').addEventListener('change', async (event) => {
  const added = await Promise.all(Array.from(event.target.files || []).map(imageFileToObject));
  feedbackScreenshots.push(...added);
  renderFeedbackScreenshotGallery();
  event.target.value = '';
});
if ($('feedbackReplyScreenshotInput')) $('feedbackReplyScreenshotInput').addEventListener('change', async (event) => {
  const added = await Promise.all(Array.from(event.target.files || []).map(imageFileToObject));
  feedbackReplyScreenshots.push(...added);
  renderFeedbackReplyScreenshotGallery();
  event.target.value = '';
});
if ($('feedbackForm')) $('feedbackForm').addEventListener('submit', async (event) => {
  event.preventDefault();
  const kind = $('feedbackKind').value === 'FEATURE' ? 'FEATURE' : 'BUG';
  const payload = {
    kind,
    title: $('feedbackTitle').value.trim(),
    summary: $('feedbackSummary').value.trim(),
    details: kind === 'BUG' ? $('feedbackDetails').value.trim() : $('feedbackFeatureDetails').value.trim(),
    expectedBehavior: kind === 'BUG' ? $('feedbackExpected').value.trim() : '',
    reproductionSteps: kind === 'BUG' ? $('feedbackSteps').value.trim() : '',
    businessValue: kind === 'FEATURE' ? $('feedbackBusinessValue').value.trim() : '',
    screenshots: feedbackScreenshots.map((shot) => shot.dataUrl || shot),
    source: 'cam-journal'
  };
  try {
    const response = await fetch('/api/tickets', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Could not submit ticket.');
    showImportStatus(`<strong>Ticket submitted.</strong><br><span class="small">${escapeHtml(data.ticket?.title || payload.title)} is now in the admin review queue.</span>`);
    resetFeedbackForm();
    await loadFeedbackTickets(data.ticket?.id || '');
    activeFeedbackTicketId = data.ticket?.id || activeFeedbackTicketId;
    renderFeedbackThreadView();
    renderFeedbackThreadList();
  } catch (err) {
    console.error('Feedback submit failed', err);
    showImportStatus('<strong>Could not submit ticket.</strong><br><span class="small">Try again in a moment. If it keeps failing, take a screenshot and report it directly.</span>');
  }
});
if ($('sendFeedbackReplyBtn')) $('sendFeedbackReplyBtn').addEventListener('click', async () => {
  const ticket = activeFeedbackTicket();
  const body = $('feedbackReplyBody').value.trim();
  if (!ticket) return;
  if (!body) return alert('Please write a reply before sending.');
  try {
    const response = await fetch(`/api/tickets/${ticket.id}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        body,
        screenshots: feedbackReplyScreenshots.map((shot) => shot.dataUrl || shot)
      })
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Could not send reply.');
    $('feedbackReplyBody').value = '';
    feedbackReplyScreenshots = [];
    renderFeedbackReplyScreenshotGallery();
    await loadFeedbackTickets(ticket.id);
    showImportStatus(`<strong>Reply sent.</strong><br><span class="small">${escapeHtml(ticket.title)} has been updated.</span>`);
  } catch (err) {
    console.error('Feedback reply failed', err);
    showImportStatus('<strong>Could not send reply.</strong><br><span class="small">Try again in a moment.</span>');
  }
});
if ($('gradeFromSetupBtn')) $('gradeFromSetupBtn').addEventListener('click', () => openGradeDialogForTrade($('editId').value));
if ($('closeGradeDialog')) $('closeGradeDialog').addEventListener('click', () => $('gradeDialog').close());
if ($('closeVersionDialog')) $('closeVersionDialog').addEventListener('click', () => $('versionDialog').close());
if ($('cancelGradeBtn')) $('cancelGradeBtn').addEventListener('click', () => $('gradeDialog').close());
if ($('gradeForm')) $('gradeForm').addEventListener('submit', (event) => {
  event.preventDefault();
  const id = $('gradeTradeId').value; const idx = trades.findIndex(t=>t.id===id); if (idx === -1 || !gradingSetup) return;
  const inputs = Array.from(document.querySelectorAll('#gradeCriteriaArea input'));
  const scores = {}; let earned = 0, max = 0;
  inputs.forEach(i => { const val = Math.min(Number(i.value || 0), Number(i.dataset.max || 0)); scores[i.dataset.key] = val; earned += val; max += Number(i.dataset.max||0); });
  const grade = max ? gradeFromPercent(earned / max * 100) : '';
  trades[idx].setup = gradingSetup.name;
  trades[idx].setupId = gradingSetup.id;
  trades[idx].grade = grade;
  trades[idx].setupScore = { setupId: gradingSetup.id, setupName: gradingSetup.name, setupVersion: Number(gradingSetup.currentVersion || 1), versionLabel: setupVersionLabel(gradingSetup.currentVersion || 1), earned, max, grade, scores, criteriaSnapshot: JSON.parse(JSON.stringify(gradingSetup.criteria || {})), gradeRulesSnapshot: JSON.parse(JSON.stringify(gradingSetup.gradeRules || [])), enableEtf: gradingSetup.enableEtf !== false, gradedAt: new Date().toISOString() };
  save(); $('gradeDialog').close(); $('editGrade').value = grade; $('editSetup').value = gradingSetup.name; renderAll();
});

if ($('selectAllTrades')) $('selectAllTrades').addEventListener('change', (event) => {
  const visible = getFilteredTrades().map(t => t.id);
  if (event.target.checked) visible.forEach(id => selectedTradeIds.add(id));
  else visible.forEach(id => selectedTradeIds.delete(id));
  renderTrades();
});
if ($('clearSelectedTrades')) $('clearSelectedTrades').addEventListener('click', () => { selectedTradeIds.clear(); renderTrades(); });
if ($('bulkActionType')) $('bulkActionType').addEventListener('change', () => refreshBulkActionValue());
if ($('bulkSubmit')) $('bulkSubmit').addEventListener('click', () => {
  const field = $('bulkActionType')?.value;
  const value = $('bulkActionValue')?.value;
  if (!field) return alert('Choose Assign Setup, Assign Portfolio, Merge Trades, or Delete Trades first.');
  bulkApply(field, value);
});
if ($('secondaryTagForm')) $('secondaryTagForm').addEventListener('submit', (event) => {
  event.preventDefault(); normalizeTags();
  const name = $('secondaryTagName').value.trim(); if (!name) return;
  if (!tags.secondary.some(t => t.name.toLowerCase() === name.toLowerCase())) tags.secondary.push({ id: stableId('secondary-tag'), name });
  $('secondaryTagName').value = ''; saveTags(); renderAll();
});
if ($('mistakeTagForm')) $('mistakeTagForm').addEventListener('submit', (event) => {
  event.preventDefault(); normalizeTags();
  const name = $('mistakeTagName').value.trim(); if (!name) return;
  if (!tags.mistakes.some(t => t.name.toLowerCase() === name.toLowerCase())) tags.mistakes.push({ id: stableId('mistake-tag'), name });
  $('mistakeTagName').value = ''; saveTags(); renderAll();
});

function applyReadOnlyUi() {
  if (!journalReadOnly) return;
  document.body.classList.add('readonly-journal');
  [
    'importInput',
    'settingsImportInput',
    'backupImportInput',
    'clearBtn',
    'bulkSubmit',
    'tradeForm',
    'editForm',
    'setupForm',
    'portfolioForm',
    'secondaryTagForm',
    'mistakeTagForm',
    'watchlistForm',
    'gradeForm',
    'saveMonthlyReviewNotes'
  ].forEach(id => {
    const el = $(id);
    if (!el) return;
    if (el.tagName === 'FORM') {
      el.querySelectorAll('input,select,textarea,button').forEach(control => {
        if (['closeDialog', 'closeGradeDialog', 'cancelGradeBtn', 'closeVersionDialog'].includes(control.id)) return;
        control.disabled = true;
      });
    } else {
      el.disabled = true;
    }
  });
  document.querySelectorAll('#newSetupBtn, #newPortfolioBtn, .addCriteriaBtn, .newVersionBtn, #watchlistBody button').forEach(btn => {
    btn.disabled = true;
    btn.title = 'Read only';
  });
}

async function loadRemoteJournalState() {
  try {
    const response = await fetch(REMOTE_STATE_ENDPOINT);
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Could not load remote journal.');
    const state = data.state || {};
    remoteStateLoaded = true;
    remoteSaveBlockedNoticeShown = false;
    journalReadOnly = Boolean(data.readOnly);
    currentSessionUser = data.user || null;
    trades = Array.isArray(state.trades) ? state.trades : trades;
    setups = Array.isArray(state.setups) ? state.setups : setups;
    portfolios = Array.isArray(state.portfolios) ? state.portfolios : portfolios;
    tags = state.tags && typeof state.tags === 'object' ? state.tags : tags;
    monthlyReviews = state.monthlyReviews && typeof state.monthlyReviews === 'object' ? state.monthlyReviews : monthlyReviews;
    watchlistItems = Array.isArray(state.watchlistItems) ? state.watchlistItems : watchlistItems;
    saveLocalCache(STORAGE_KEY, trades, tradesWithoutCachedScreenshots());
    saveLocalCache(SETUPS_KEY, setups, setupsWithoutCachedScreenshots());
    saveLocalCache(PORTFOLIOS_KEY, portfolios, portfolios);
    saveLocalCache(TAGS_KEY, tags, tags);
    saveLocalCache(REVIEWS_KEY, monthlyReviews || {}, monthlyReviews || {});
    saveLocalCache(WATCHLIST_KEY, watchlistItems || [], watchlistItems || []);
  } catch (err) {
    remoteStateLoaded = false;
    console.error('Remote journal load failed', err);
    showImportStatus('<strong>Could not load the shared database copy.</strong><br><span class="small">Showing browser cached journal data if available.</span>');
  }
}

async function bootJournal() {
  await loadRemoteJournalState();
  ensureDefaultSetups();
  ensureStableDataIds();
  fillSetupForm(newSetupDraft());
  fillPortfolioForm();
  if ($('date')) $('date').valueAsDate = new Date();
  initSortableTables();
  renderAll();
  applyReadOnlyUi();
  bootingJournal = false;
}


['monthlyReviewMonth'].forEach(id => { if ($(id)) $(id).addEventListener('change', renderMonthlyReview); });
if ($('saveMonthlyReviewNotes')) $('saveMonthlyReviewNotes').addEventListener('click', () => { const m = $('monthlyReviewMonth')?.value || currentMonthKey(); monthlyReviews[m] = $('monthlyReviewNotes')?.value || ''; saveMonthlyReviews(); alert('Monthly notes saved.'); });
['playbookSetup','playbookGrade','playbookPortfolio','playbookMistake','playbookScreenshots'].forEach(id => { if ($(id)) $(id).addEventListener('change', () => { playbookPage = 1; renderPlaybook(); }); });

if ($('watchlistDate')) {
  $('watchlistDate').value = $('watchlistDate').value || todayKey();
  $('watchlistDate').addEventListener('change', renderWatchlist);
  updateDatePickerLabels();
}
['watchlistRange','watchlistSort'].forEach(id => { if ($(id)) $(id).addEventListener('change', renderWatchlist); });
if ($('watchlistForm')) $('watchlistForm').addEventListener('submit', event => {
  event.preventDefault();
  const ticker = String($('watchlistTicker')?.value || '').trim().toUpperCase();
  if (!ticker) return;
  watchlistItems.push({
    id: stableId('watchlist'),
    date: $('watchlistDate')?.value || todayKey(),
    ticker,
    note: String($('watchlistNote')?.value || '').trim(),
    createdAt: new Date().toISOString()
  });
  $('watchlistTicker').value = '';
  $('watchlistNote').value = '';
  saveWatchlist();
  renderWatchlist();
});

bootJournal();
document.querySelectorAll('.datePickerBtn').forEach(btn => btn.addEventListener('click', (event) => { event.stopPropagation(); openDatePicker(btn); }));
['compareSetupA','compareSetupB','compareVersionA','compareVersionB','compareGradeA','compareGradeB','breakdownSetupFilter'].forEach(id => { if ($(id)) $(id).addEventListener('change', () => { refreshComparisonOptions(); renderSetupComparison(); }); });
if ($('resetComparisonFilters')) $('resetComparisonFilters').addEventListener('click', () => { ['compareSetupA','compareSetupB','breakdownSetupFilter'].forEach(id => { if ($(id)) $(id).value = ''; }); ['compareVersionA','compareVersionB'].forEach(id => { if ($(id)) $(id).value = 'current'; }); ['compareGradeA','compareGradeB'].forEach(id => { if ($(id)) setSelectValues(id, []); }); refreshComparisonOptions(); renderSetupComparison(); });


// Version 30: Daily rotating quote in the sidebar. No database needed; quotes live in this file.
const DAILY_QUOTES = [
  "The goal of trading is not to be right. The goal is to make money when you're right and lose little when you're wrong.",
  "A good trade is one that follows your process, regardless of the outcome.",
  "Your edge isn't found in a chart pattern. It's found in your ability to execute the same process repeatedly.",
  "The market pays discipline, not predictions.",
  "One A+ trade can do more for your account than ten impulsive trades.",
  "Protecting capital is not being defensive—it's preserving opportunity.",
  "Every trade is uncertain. Your job is to manage risk, not certainty.",
  "Patience is a position.",
  "The trader who waits for confirmation often outperforms the trader who chases anticipation.",
  "Consistency is built from thousands of small decisions, not one big winner.",
  "Your next trade doesn't know what happened on your last trade.",
  "When in doubt, sit out.",
  "The market will be open tomorrow. Your capital needs to be there too.",
  "The difference between a professional and an amateur is that the professional can follow their rules when they don't feel like it.",
  "You don't become profitable by finding the perfect setup. You become profitable by eliminating the trades that never met your criteria in the first place.",
  "The fastest way to improve your win rate is not finding better trades—it's refusing to take worse ones.",
  "An A+ setup with a full loss is a successful trade. A C setup with a small gain is a mistake.",
  "Your account grows when your standards rise.",
  "The market rewards the trader who can do nothing while waiting and everything when it's time to act.",
  "You are only one or two rule changes away from becoming the trader you want to be.",
  "The best traders think in probabilities, not certainties.",
  "Great traders are great risk managers first.",
  "Build a playbook of your best setups and trade those relentlessly.",
  "Consistency comes from repeating what works.",
  "Trade the setup, not your opinion.",
  "Your job is not to make money. Your job is to trade well.",
  "Be 1% Better Today Than You Were Yesterday.",
  "Find the broken slot machine.",
  "The real improvement doesn't come from the actual day. It comes from the review process and practice.",
  "99% of our learning comes with hindsight."
];

function renderDailyQuote(){
  const el = document.getElementById('dailyQuote');
  if (!el || !DAILY_QUOTES.length) return;
  const start = Date.UTC(2026, 0, 1);
  const now = new Date();
  const todayUtc = Date.UTC(now.getFullYear(), now.getMonth(), now.getDate());
  const dayIndex = Math.floor((todayUtc - start) / 86400000);
  const quote = DAILY_QUOTES[((dayIndex % DAILY_QUOTES.length) + DAILY_QUOTES.length) % DAILY_QUOTES.length];
  el.textContent = `“${quote}”`;
}
renderDailyQuote();
