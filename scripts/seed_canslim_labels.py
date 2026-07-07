from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INDEX_FILE = ROOT / "data" / "canslim_reference" / "chart_setup_index.csv"

LABELS = {
    "canslim-001": {
        "source_page": "11",
        "setup_type": "cup with handle",
        "ticker": "Tennessee Coal & Iron",
        "timeframe": "weekly",
        "base_quality": "Shorter constructive cup-with-handle after earlier consolidation; handle formed near the pivot with tight price action.",
        "volume_notes": "Volume expanded on the breakout after a quieter base area.",
        "relative_strength_notes": "",
        "buy_point_notes": "Buy marked as price clears the handle area and starts a sustained advance.",
        "failure_warnings": "Avoid late entries after the move is extended far above the moving average.",
        "model_lesson": "A compact weekly cup-with-handle can launch a major move when the breakout comes with volume support.",
        "outcome_note": "Book chart notes a 265% advance in 39 weeks.",
        "confidence": "high",
        "review_status": "labeled_seed",
    },
    "canslim-002": {
        "source_page": "12",
        "setup_type": "double bottom; cup with handle add-on",
        "ticker": "Northern Pacific",
        "timeframe": "weekly",
        "base_quality": "Long double-bottom structure followed by a later 10-week cup-with-handle add-on area.",
        "volume_notes": "Volume increased materially as the advance developed.",
        "relative_strength_notes": "Relative strength line was uptrending versus the Dow, showing outperformance.",
        "buy_point_notes": "Initial buy marked after the double-bottom breakout; add-on marked near the later 10-week cup-with-handle.",
        "failure_warnings": "Chart later shows a climax top, so strength required sell discipline after the advance became vertical.",
        "model_lesson": "Strong leaders can offer an initial base breakout and later add-on entries while relative strength confirms leadership.",
        "outcome_note": "Book chart notes an 1181% advance in 29 weeks.",
        "confidence": "high",
        "review_status": "labeled_seed",
    },
    "canslim-003": {
        "source_page": "13",
        "setup_type": "long base; high tight flag",
        "ticker": "Bethlehem Steel",
        "timeframe": "weekly",
        "base_quality": "Large multi-month base during a difficult market period, followed by a powerful advance and a later 6-week high tight flag.",
        "volume_notes": "Volume expanded during the breakout and early advance.",
        "relative_strength_notes": "Relative strength line surged with price after the market correction ended.",
        "buy_point_notes": "First buy marked as price clears the long base; second buy marked on the first pullback to the 10-week moving average.",
        "failure_warnings": "Later climax-top area is marked as a sell zone after a very steep run.",
        "model_lesson": "Major leaders can emerge from long bases after market corrections and may offer early 10-week pullback entries.",
        "outcome_note": "Book chart notes a 1479% advance in 99 weeks.",
        "confidence": "high",
        "review_status": "labeled_seed",
    },
    "canslim-004": {
        "source_page": "14",
        "setup_type": "long base; tight-price add-on",
        "ticker": "General Motors",
        "timeframe": "weekly",
        "base_quality": "Nine-month base before the buy point, then a later short base with four tight weekly closes.",
        "volume_notes": "Big volume appeared at the buy point.",
        "relative_strength_notes": "Relative strength improved sharply as the stock broke out and advanced.",
        "buy_point_notes": "Buy marked on breakout from the long base; add-on marked after the tight-price base.",
        "failure_warnings": "Avoid chasing after the stock becomes extended from the moving average.",
        "model_lesson": "A long base with volume confirmation can start a major move; tight closes can create a controlled add-on point.",
        "outcome_note": "Book chart notes a 471% advance in 39 weeks.",
        "confidence": "high",
        "review_status": "labeled_seed",
    },
    "canslim-005": {
        "source_page": "14",
        "duplicate_of": "canslim-004",
        "setup_type": "duplicate",
        "ticker": "General Motors",
        "timeframe": "weekly",
        "base_quality": "Duplicate capture of canslim-004.",
        "volume_notes": "Duplicate capture of canslim-004.",
        "relative_strength_notes": "Duplicate capture of canslim-004.",
        "buy_point_notes": "Duplicate capture of canslim-004.",
        "failure_warnings": "Do not count as a separate model example.",
        "model_lesson": "Duplicate image retained for source traceability.",
        "outcome_note": "Duplicate of page 14 General Motors example.",
        "confidence": "high",
        "review_status": "duplicate",
    },
    "canslim-006": {
        "source_page": "15",
        "setup_type": "cup with handle; saucer with handle",
        "ticker": "S.S. Kresge",
        "timeframe": "weekly",
        "base_quality": "Early 14-week cup-with-handle followed by a longer 22-week saucer-with-handle as the stock stair-stepped higher.",
        "volume_notes": "Volume increased at key breakout areas and later rose again during the advance.",
        "relative_strength_notes": "Relative strength line trended higher with the stock during the long advance.",
        "buy_point_notes": "Buy points are marked at the cup-with-handle breakout and later saucer-with-handle breakout; add-on noted near the 10-week line.",
        "failure_warnings": "The chart notes selling into a prolonged move if price rises too far above its trend line.",
        "model_lesson": "Multiple constructive bases can appear during a long leader run; add-ons should be tied to proper bases or 10-week support.",
        "outcome_note": "Book chart notes an 836% advance in 154 weeks.",
        "confidence": "high",
        "review_status": "labeled_seed",
    },
    "canslim-007": {
        "source_page": "15",
        "duplicate_of": "canslim-006",
        "setup_type": "duplicate",
        "ticker": "S.S. Kresge",
        "timeframe": "weekly",
        "base_quality": "Duplicate capture of canslim-006.",
        "volume_notes": "Duplicate capture of canslim-006.",
        "relative_strength_notes": "Duplicate capture of canslim-006.",
        "buy_point_notes": "Duplicate capture of canslim-006.",
        "failure_warnings": "Do not count as a separate model example.",
        "model_lesson": "Duplicate image retained for source traceability.",
        "outcome_note": "Duplicate of page 15 S.S. Kresge example.",
        "confidence": "high",
        "review_status": "duplicate",
    },
    "canslim-008": {
        "source_page": "16",
        "setup_type": "long base breakout",
        "ticker": "Utah Securities",
        "timeframe": "weekly",
        "base_quality": "Long 54-week base leading to a breakout and sustained advance.",
        "volume_notes": "Buy week showed a large volume increase.",
        "relative_strength_notes": "Relative strength line rose strongly after the breakout.",
        "buy_point_notes": "Buy marked as the stock clears the long base; chart notes that a normal pullback may revisit the buy area without necessarily failing.",
        "failure_warnings": "The example later marks a climax top after the advance becomes steep.",
        "model_lesson": "A proper base breakout may briefly pull back toward the pivot, so execution should distinguish normal retests from true failure.",
        "outcome_note": "Book chart notes a 538% advance in 63 weeks.",
        "confidence": "high",
        "review_status": "labeled_seed",
    },
    "canslim-009": {
        "source_page": "16",
        "duplicate_of": "canslim-008",
        "setup_type": "duplicate",
        "ticker": "Utah Securities",
        "timeframe": "weekly",
        "base_quality": "Duplicate capture of canslim-008.",
        "volume_notes": "Duplicate capture of canslim-008.",
        "relative_strength_notes": "Duplicate capture of canslim-008.",
        "buy_point_notes": "Duplicate capture of canslim-008.",
        "failure_warnings": "Do not count as a separate model example.",
        "model_lesson": "Duplicate image retained for source traceability.",
        "outcome_note": "Duplicate of page 16 Utah Securities example.",
        "confidence": "high",
        "review_status": "duplicate",
    },
    "canslim-010": {
        "source_page": "17",
        "setup_type": "cup with handle; double bottom with handle; flat base",
        "ticker": "Du Pont de Nemours",
        "timeframe": "weekly",
        "base_quality": "Sequence of constructive bases, including a 9-week cup-with-handle, a 35-week double-bottom-with-handle, a 20-week cup-with-handle, and a later flat base.",
        "volume_notes": "Volume picked up around several buy areas.",
        "relative_strength_notes": "Relative strength advanced with price through the base sequence.",
        "buy_point_notes": "Multiple buy points are marked as the stock emerges from each constructive base and later flat base.",
        "failure_warnings": "The chart marks a climax top after an excessive split-adjusted advance.",
        "model_lesson": "A true market leader can build repeated proper bases during a long advance; each new entry still needs a defined pivot and sell plan.",
        "outcome_note": "Book chart notes a 1074% advance in 225 weeks.",
        "confidence": "high",
        "review_status": "labeled_seed",
    },
}


def label(
    page: str,
    ticker: str,
    setup_type: str,
    outcome: str,
    lesson: str,
    *,
    base: str = "Annotated weekly model-book leader example with one or more constructive bases during a sustained advance.",
    volume: str = "Volume should be checked around the marked buy/add areas for confirmation.",
    rs: str = "Relative strength line generally confirms leadership during the advance.",
    buy: str = "Use only the marked base breakout, pullback, or add-on area; avoid late entries after extension.",
    warning: str = "Several examples later mark climax or extended areas, so entry quality still requires sell-rule discipline.",
    confidence: str = "medium",
) -> dict[str, str]:
    return {
        "source_page": page,
        "setup_type": setup_type,
        "ticker": ticker,
        "timeframe": "weekly",
        "base_quality": base,
        "volume_notes": volume,
        "relative_strength_notes": rs,
        "buy_point_notes": buy,
        "failure_warnings": warning,
        "model_lesson": lesson,
        "outcome_note": outcome,
        "confidence": confidence,
        "review_status": "labeled_seed",
    }


def duplicate(page: str, duplicate_of: str, ticker: str) -> dict[str, str]:
    return {
        "source_page": page,
        "duplicate_of": duplicate_of,
        "setup_type": "duplicate",
        "ticker": ticker,
        "timeframe": "weekly",
        "base_quality": f"Duplicate capture of {duplicate_of}.",
        "volume_notes": f"Duplicate capture of {duplicate_of}.",
        "relative_strength_notes": f"Duplicate capture of {duplicate_of}.",
        "buy_point_notes": f"Duplicate capture of {duplicate_of}.",
        "failure_warnings": "Do not count as a separate model example.",
        "model_lesson": "Duplicate image retained for source traceability.",
        "outcome_note": f"Duplicate of page {page} {ticker} example.",
        "confidence": "high",
        "review_status": "duplicate",
    }


LABELS.update(
    {
        "canslim-011": duplicate("17", "canslim-010", "Du Pont de Nemours"),
        "canslim-012": label(
            "19",
            "Burroughs Adding Machines",
            "saucer with handle; tight-area add-ons",
            "Book chart notes a 1392% advance in 168 weeks.",
            "Long, orderly bases and tight follow-on areas can support repeated entries in a durable institutional leader.",
            base="Long saucer-with-handle structure followed by later tight areas and add-on opportunities.",
        ),
        "canslim-013": duplicate("19", "canslim-012", "Burroughs Adding Machines"),
        "canslim-014": label(
            "24",
            "Deere & Co.",
            "cup with handle; add-ons",
            "Book chart notes a 307% advance in 104 weeks.",
            "A leader can recover from a market correction, build a proper base, and then offer add-ons near controlled pullbacks.",
        ),
        "canslim-015": label(
            "25",
            "Schenley Distilling",
            "base breakout; 10-week add-ons",
            "Book chart notes an 1164% advance in 185 weeks.",
            "Repeated support near the 10-week line can give lower-risk add points after a valid breakout.",
        ),
        "canslim-016": label(
            "26",
            "Conde Nast Publications",
            "cup with handle; base breakout",
            "Book chart notes a 514% advance in 101 weeks.",
            "A long base can turn into a major leader when the breakout aligns with improving relative strength and volume.",
        ),
        "canslim-017": label(
            "27",
            "Gimbel Bros.",
            "saucer/base breakout",
            "Book chart notes a 674% advance in 103 weeks.",
            "Rounded, constructive bases can work when price tightens and breaks out with institutional demand.",
        ),
        "canslim-018": label(
            "28",
            "Outboard Marine",
            "long leader run; add-on bases",
            "Book chart notes a 720% advance in 177 weeks.",
            "A major winner may advance through multiple base-and-add phases, but each add still needs a defined setup.",
        ),
        "canslim-019": label(
            "29",
            "Kaiser Aluminum",
            "base breakout; climax-top sell area",
            "Book chart notes a 377% advance in 93 weeks.",
            "The same chart can teach both proper entry from a base and the need to sell into late-stage climactic action.",
        ),
        "canslim-020": duplicate("29", "canslim-019", "Kaiser Aluminum"),
        "canslim-021": label(
            "30",
            "Thiokol Chemical",
            "IPO base; add-on bases",
            "Book chart notes an 800% advance in 109 weeks.",
            "Newer leaders can build early IPO-style bases and then stair-step higher through secondary bases.",
        ),
        "canslim-022": duplicate("30", "canslim-021", "Thiokol Chemical"),
        "canslim-023": label(
            "31",
            "Brunswick",
            "long leader run; repeated bases",
            "Book chart notes a 1500% advance in 162 weeks.",
            "A long-duration winner can show multiple valid bases, but add-ons should remain tied to structure rather than emotion.",
        ),
        "canslim-024": duplicate("31", "canslim-023", "Brunswick"),
        "canslim-025": label(
            "32",
            "Zenith Radio",
            "base breakout; add-ons",
            "Book chart notes a 493% advance in 66 weeks.",
            "Explosive advances often start from quieter bases, then offer add points as the 10-week line catches up.",
        ),
        "canslim-026": duplicate("32", "canslim-025", "Zenith Radio"),
        "canslim-027": label(
            "33",
            "Texas Instruments",
            "long base breakout; add-ons",
            "Book chart notes a 772% advance in 116 weeks.",
            "Strong leadership can persist through normal pullbacks when price respects the moving-average structure.",
        ),
        "canslim-028": duplicate("33", "canslim-027", "Texas Instruments"),
        "canslim-029": label(
            "39",
            "National Airlines",
            "cup with handle; trendline/add-on entries",
            "Book chart notes a 1004% advance in 179 weeks.",
            "A proper base can start the move, while later controlled pullbacks and trendline areas can support add decisions.",
        ),
        "canslim-030": label(
            "40",
            "Northwest Airlines",
            "base breakout; add-on bases",
            "Book chart notes a 1240% advance in 186 weeks.",
            "A leader can compound through repeated bases if entries are taken near pivots and not after obvious extension.",
        ),
        "canslim-031": label(
            "41",
            "Xerox",
            "IPO-style leader; base breakout",
            "Book chart notes a 680% advance in 158 weeks.",
            "Early-stage growth leaders often offer their best entries after a constructive base and before the move is widely obvious.",
        ),
        "canslim-032": label(
            "42",
            "Syntex",
            "high tight flag; base breakout",
            "Book chart notes a 451% advance in 25 weeks.",
            "Very powerful leaders can form high tight areas, but they require unusually strong price and volume confirmation.",
        ),
        "canslim-033": label(
            "43",
            "Rollins",
            "base breakout; climactic advance",
            "Book chart notes a 254% advance in 35 weeks.",
            "A quiet base can produce a fast advance, but vertical moves need disciplined sell rules.",
        ),
        "canslim-034": label(
            "44",
            "Simmonds Precision Products",
            "base breakout; add-ons",
            "Book chart notes a 672% advance in 28 weeks.",
            "Strong earnings/relative strength context plus decisive price action can produce rapid leader moves.",
        ),
        "canslim-035": label(
            "45",
            "Monogram Industries",
            "base breakout; 10-week support",
            "Book chart notes an 807% advance in 57 weeks.",
            "Support and add points around the 10-week line can help manage a fast leader after the initial breakout.",
        ),
        "canslim-036": label(
            "46",
            "Digital Equipment",
            "base breakout; pullback add-on",
            "Book chart notes a 1343% advance in 156 weeks.",
            "A new technology leader can become a long winner when bases form above rising moving averages.",
        ),
        "canslim-037": label(
            "47",
            "Loews",
            "long base breakout; sell-rule example",
            "Book chart notes a 1023% advance in 101 weeks.",
            "The model shows both entry discipline during the base and exit discipline after the advance matures.",
        ),
        "canslim-038": {
            "setup_type": "non-chart artifact",
            "ticker": "",
            "timeframe": "",
            "base_quality": "Image appears to be a file-display placeholder rather than a chart screenshot.",
            "volume_notes": "",
            "relative_strength_notes": "",
            "buy_point_notes": "",
            "failure_warnings": "Exclude from model-example counts.",
            "model_lesson": "Not usable as a chart model example.",
            "outcome_note": "",
            "confidence": "high",
            "review_status": "exclude",
        },
        "canslim-039": label("49", "Skyline", "base breakout; 10-week support", "Book chart notes a 715% advance in 98 weeks.", "Constructive bases and controlled support at moving averages can sustain a large advance."),
        "canslim-040": label("50", "Redman Industries", "base breakout; add-on bases", "Book chart notes a 683% advance in 49 weeks.", "Repeated tight setups during an advance can provide add points when the stock stays under institutional accumulation."),
        "canslim-041": label("51", "Levitz Furniture", "base breakout; add-on entries", "Book chart notes a 608% advance in 87 weeks.", "A long winner can offer several lower-risk entries as long as each one has a nearby invalidation point."),
        "canslim-042": label("52", "Rite Aid", "cup with handle; trendline entry", "Book chart notes a 421% advance in 71 weeks.", "A cup-with-handle plus rising relative strength can mark an actionable leadership setup."),
        "canslim-043": label("53", "McDonald's", "long base; add-on entries", "Book chart notes a 422% advance in 108 weeks.", "A long base can be valid when the stock emerges with improving price strength and then respects follow-on support."),
        "canslim-044": label("55", "Sea Containers", "deep base; high-risk leader", "Book chart notes a 4048% advance in 389 weeks.", "Deep or volatile leaders require extra selectivity; the reward can be large, but the setup must still define risk."),
        "canslim-045": label("56", "FlightSafety International", "cup with handle; add-ons", "Book chart notes a 958% advance in 195 weeks.", "Quiet handles and 10-week support can help separate normal pullbacks from failed moves."),
        "canslim-046": label("57", "Wang Laboratories", "base breakout; add-on bases", "Book chart notes a 1348% advance in 138 weeks.", "A technology leader can run through several constructive consolidations when demand remains persistent."),
        "canslim-047": label("58", "Resorts International", "speculative leader; climax top", "Book chart notes a 6330% advance in 74 weeks.", "Speculative leaders can move extremely fast, so position management and climax-top recognition matter as much as entry."),
        "canslim-048": label("59", "Texas Oil & Gas", "base breakout; add-ons", "Book chart notes a 529% advance in 101 weeks.", "Commodity-related leaders still need the same base, volume, and risk-location discipline."),
        "canslim-049": label("60", "Global Marine", "base breakout; add-ons", "Book chart notes a 782% advance in 94 weeks.", "An industry leader can build a series of add-on opportunities when group strength and price action align."),
        "canslim-050": label("61", "Pic N Save", "base breakout; long leader run", "Book chart notes a 564% advance in 206 weeks.", "Retail leaders can compound for years when bases remain orderly and sell signals are respected."),
        "canslim-051": label("62", "Wal-Mart Stores", "base breakout; add-ons", "Book chart notes an 882% advance in 158 weeks.", "Large retail winners often advance through repeated base breakouts and controlled pullbacks."),
        "canslim-052": label("63", "The Limited", "base breakout; high tight area", "Book chart notes a 673% advance in 71 weeks.", "Strong retail leaders may form tight, powerful continuation areas after an initial breakout."),
        "canslim-053": label("64", "Home Depot", "IPO base; base breakout", "Book chart notes an 892% advance in 64 weeks.", "New public leaders can produce major moves from early bases when growth and demand are clear."),
        "canslim-054": label("65", "Price Co.", "base breakout; add-on bases", "Book chart notes a 417% advance in 60 weeks.", "A leader can offer several entries during an advance, but each must be judged against base quality and extension."),
        "canslim-055": label("66", "Stop & Shop", "base breakout; 10-week add-ons", "Book chart notes a 536% advance in 74 weeks.", "Pullbacks to rising moving averages can be usable only when the broader setup remains intact."),
        "canslim-056": label("67", "Digital Switch", "base breakout; fast leader", "Book chart notes an 843% advance in 46 weeks.", "Fast leaders demand early entry and clear sell rules because later entry points become extended quickly."),
        "canslim-057": label("68", "Pulte Home", "cup with handle; base breakout", "Book chart notes a 733% advance in 47 weeks.", "A cyclical leader can be actionable when the base is proper and breakout demand is visible."),
        "canslim-058": label("69", "Liz Claiborne", "base breakout; add-on entries", "Book chart notes a 211% advance in 43 weeks.", "Even shorter leader moves can be useful models when the entry and risk point are clean."),
        "canslim-059": label("70", "Franklin Resources", "base breakout; add-ons", "Book chart notes an 811% advance in 78 weeks.", "Financial leaders can run strongly when earnings acceleration and base breakouts align."),
        "canslim-060": label("71", "Microsoft", "IPO/base breakout; fast leader", "Book chart notes a 272% advance in 30 weeks.", "Early entries in exceptional growth leaders can matter more than waiting for obvious consensus.",
                               base="Early Microsoft leader example with a base breakout and rapid advance."),
        "canslim-061": label("72", "Adobe Systems", "IPO/base breakout; fast leader", "Book chart notes a 307% advance in 23 weeks.", "Young software leaders can advance quickly after proper early bases; late entries carry higher risk."),
        "canslim-062": label("73", "Costco Wholesale", "base breakout; long leader run", "Book chart notes a 712% advance in 163 weeks.", "High-quality retail leaders can build many add-on areas if the long-term trend remains intact."),
        "canslim-063": label("74", "Microsoft", "long base; add-on entries", "Book chart notes a 571% advance in 121 weeks.", "A proven leader can build a second major run from later bases when growth and relative strength persist."),
        "canslim-064": label("75", "American Power Conversion", "base breakout; add-ons", "Book chart notes a 745% advance in 96 weeks.", "Strong leaders often retest or pause near moving averages; add only when structure and demand remain favorable."),
        "canslim-065": label("76", "Amgen", "base breakout; add-ons", "Book chart notes a 681% advance in 96 weeks.", "Biotech leaders can trend powerfully, but bases and pullbacks must be judged with extra attention to volatility."),
        "canslim-066": label("77", "United States Surgical", "base breakout; add-ons", "Book chart notes a 786% advance in 93 weeks.", "Healthcare leaders can build multiple bases; the best entries keep risk close to the pivot or 10-week line."),
        "canslim-067": label("80", "JLG Industries", "base breakout; 3-weeks-tight", "Book chart notes a 670% advance in 53 weeks.", "Tight price areas during an uptrend can become add-on points when supported by volume and trend."),
        "canslim-068": label("81", "Charles Schwab", "base breakout; late-stage sell area", "Book chart notes a 408% advance in 26 weeks.", "Powerful financial leaders can move fast, making late-stage extension and sell-rule discipline critical."),
        "canslim-069": label("82", "America Online", "base breakout; fast internet leader", "Book chart notes a 456% advance in 23 weeks.", "Hyper-growth leaders can accelerate quickly from proper bases, but late entries after vertical moves are dangerous."),
        "canslim-070": label("90", "Southwestern Energy", "cup with handle; add-on bases", "Book chart notes a 565% advance in 83 weeks.", "Energy leaders can offer proper bases and add-on points when price, volume, and group strength align."),
        "canslim-071": label("105", "Precision Castparts", "base breakout; tight action", "Book chart notes a 259% advance in 115 weeks.", "A more orderly leader can still be valuable when entries are taken from clean bases and tight support areas."),
        "canslim-072": label("106", "Intuitive Surgical", "IPO/base breakout; add-ons", "Book chart notes a 418% advance in 123 weeks.", "Innovative growth leaders can form early bases and later continuation setups; each add should be tied to structure."),
        "canslim-073": label("108", "First Solar", "IPO/base breakout; climax sell area", "Book chart notes an 807% advance in 47 weeks.", "A spectacular leader can still require strict entry and sell discipline, especially after a steep late-stage run."),
    }
)


def ensure_columns(fieldnames: list[str]) -> list[str]:
    desired = [
        "source_page",
        "duplicate_of",
        "outcome_note",
        "confidence",
    ]
    for column in desired:
        if column not in fieldnames:
            insert_at = fieldnames.index("setup_type") if column in {"source_page", "duplicate_of"} else fieldnames.index("review_status")
            fieldnames.insert(insert_at, column)
    return fieldnames


def main() -> None:
    with INDEX_FILE.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = ensure_columns(list(reader.fieldnames or []))
        rows = list(reader)

    for row in rows:
        row.setdefault("source_page", "")
        row.setdefault("duplicate_of", "")
        row.setdefault("outcome_note", "")
        row.setdefault("confidence", "")
        label = LABELS.get(row["image_id"])
        if label:
            row.update(label)

    with INDEX_FILE.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Seeded {len(LABELS)} labels in {INDEX_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
