import { NextResponse } from "next/server";
import { getSessionUser } from "@/lib/auth";
import { getCamJournalScreenshot, getWeeklyWatchlists, saveWeeklyWatchlists } from "@/lib/store";
import type { WatchlistItem, WeeklyWatchlist } from "@/lib/types";

function isoWeek(date = new Date()) {
  const utc = new Date(Date.UTC(date.getFullYear(), date.getMonth(), date.getDate()));
  const day = utc.getUTCDay() || 7;
  utc.setUTCDate(utc.getUTCDate() + 4 - day);
  const year = utc.getUTCFullYear();
  const yearStart = new Date(Date.UTC(year, 0, 1));
  const weekNumber = Math.ceil(((utc.getTime() - yearStart.getTime()) / 86400000 + 1) / 7);
  const monday = new Date(utc);
  monday.setUTCDate(utc.getUTCDate() - ((utc.getUTCDay() || 7) - 1));
  const friday = new Date(monday);
  friday.setUTCDate(monday.getUTCDate() + 4);
  return {
    year,
    weekNumber,
    weekKey: `${year}-W${String(weekNumber).padStart(2, "0")}`,
    startDate: monday.toISOString().slice(0, 10),
    endDate: friday.toISOString().slice(0, 10)
  };
}

function newYorkDate() {
  const parts = new Intl.DateTimeFormat("en-US", {
    timeZone: "America/New_York",
    year: "numeric",
    month: "2-digit",
    day: "2-digit"
  }).formatToParts(new Date());
  const value = Object.fromEntries(parts.map((part) => [part.type, part.value]));
  return new Date(`${value.year}-${value.month}-${value.day}T12:00:00Z`);
}

function currentOrUpcomingTradingWeek() {
  const date = newYorkDate();
  const day = date.getUTCDay();
  if (day === 6) date.setUTCDate(date.getUTCDate() + 2);
  if (day === 0) date.setUTCDate(date.getUTCDate() + 1);
  return isoWeek(date);
}

function createWatchlist(ownerId: string, week: ReturnType<typeof isoWeek>) {
  const now = new Date().toISOString();
  return {
    id: `${ownerId}-${week.weekKey}`,
    userId: ownerId,
    ...week,
    title: `W${week.weekNumber} Watchlist`,
    items: [],
    createdAt: now,
    updatedAt: now
  } satisfies WeeklyWatchlist;
}

export async function GET(request: Request) {
  const user = await getSessionUser();
  if (!user) return NextResponse.json({ error: "Unauthorized." }, { status: 401 });

  const screenshotId = new URL(request.url).searchParams.get("screenshotId");
  if (screenshotId) {
    const screenshot = await getCamJournalScreenshot(screenshotId);
    if (!screenshot || screenshot.entityType !== "watchlist-item") {
      return NextResponse.json({ error: "Watchlist screenshot not found." }, { status: 404 });
    }
    return new NextResponse(new Uint8Array(screenshot.imageData), {
      headers: {
        "Content-Type": screenshot.mimeType,
        "Cache-Control": "private, no-store",
        "X-Content-Type-Options": "nosniff"
      }
    });
  }

  const ownerId = user.journalOwnerId || user.id;
  try {
    let watchlists = await getWeeklyWatchlists(ownerId);
    const current = currentOrUpcomingTradingWeek();
    if (!watchlists.some((watchlist) => watchlist.weekKey === current.weekKey) && !user.readOnly) {
      watchlists = await saveWeeklyWatchlists(ownerId, [
        ...watchlists,
        createWatchlist(ownerId, current)
      ]);
    }
    return NextResponse.json({ user, watchlists, currentWeekKey: current.weekKey });
  } catch (error) {
    return NextResponse.json({ error: error instanceof Error ? error.message : "Could not load watchlists." }, { status: 500 });
  }
}

export async function PUT(request: Request) {
  const user = await getSessionUser();
  if (!user) return NextResponse.json({ error: "Unauthorized." }, { status: 401 });
  if (user.readOnly) return NextResponse.json({ error: "This account is read-only." }, { status: 403 });

  const body = await request.json();
  const watchlists = Array.isArray(body.watchlists) ? (body.watchlists as WeeklyWatchlist[]) : [];
  try {
    const saved = await saveWeeklyWatchlists(user.id, watchlists);
    return NextResponse.json({ watchlists: saved });
  } catch (error) {
    return NextResponse.json({ error: error instanceof Error ? error.message : "Could not save watchlists." }, { status: 500 });
  }
}

export async function PATCH(request: Request) {
  const user = await getSessionUser();
  if (!user) return NextResponse.json({ error: "Unauthorized." }, { status: 401 });
  if (user.readOnly) return NextResponse.json({ error: "This account is read-only." }, { status: 403 });

  const body = await request.json();
  const incomingWatchlist = body.watchlist as Partial<WeeklyWatchlist> | undefined;
  const incomingItem = body.item as WatchlistItem | undefined;
  const weekKey = String(body.weekKey || incomingWatchlist?.weekKey || "");
  if (!weekKey || !incomingItem?.id) {
    return NextResponse.json({ error: "A week and watchlist item are required." }, { status: 400 });
  }

  const ownerId = user.journalOwnerId || user.id;
  try {
    const now = new Date().toISOString();
    const current = await getWeeklyWatchlists(ownerId);
    const existingWeek = current.find((watchlist) => watchlist.weekKey === weekKey);
    const fallbackWeek = incomingWatchlist;
    if (!existingWeek && (!fallbackWeek?.year || !fallbackWeek.weekNumber)) {
      return NextResponse.json({ error: "Could not find that watchlist week." }, { status: 404 });
    }

    const mergedWeek: WeeklyWatchlist = existingWeek
      ? {
          ...existingWeek,
          updatedAt: now,
          items: existingWeek.items.some((item) => item.id === incomingItem.id)
            ? existingWeek.items.map((item) => (item.id === incomingItem.id ? { ...incomingItem, updatedAt: now } : item))
            : [...existingWeek.items, { ...incomingItem, updatedAt: now }]
        }
      : {
          id: String(fallbackWeek?.id || `${ownerId}-${weekKey}`),
          userId: ownerId,
          weekKey,
          year: Number(fallbackWeek?.year),
          weekNumber: Number(fallbackWeek?.weekNumber),
          startDate: String(fallbackWeek?.startDate || ""),
          endDate: String(fallbackWeek?.endDate || ""),
          title: String(fallbackWeek?.title || `W${fallbackWeek?.weekNumber} Watchlist`),
          items: [{ ...incomingItem, updatedAt: now }],
          createdAt: String(fallbackWeek?.createdAt || now),
          updatedAt: now
        };

    const nextWatchlists = existingWeek
      ? current.map((watchlist) => (watchlist.weekKey === weekKey ? mergedWeek : watchlist))
      : [...current, mergedWeek];
    const saved = await saveWeeklyWatchlists(ownerId, nextWatchlists);
    const savedWeek = saved.find((watchlist) => watchlist.weekKey === weekKey);
    const savedItem = savedWeek?.items.find((item) => item.id === incomingItem.id);
    return NextResponse.json({ watchlist: savedWeek, item: savedItem });
  } catch (error) {
    return NextResponse.json({ error: error instanceof Error ? error.message : "Could not save ticker." }, { status: 500 });
  }
}
