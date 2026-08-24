# Shakeout and Re-entry Study

**Portfolio:** CF Statement  
**Trade window:** June 1–August 11, 2026  
**Market data through:** August 10, 2026 close (the August 11 session was not yet complete)  
**Sources:** Branden trade-log export and journal table; FMP daily OHLCV history

## Executive conclusion

The concern is real. Among 28 mature CANSLIM losses or breakeven exits, 17 later traded at least 2 ATR above the original entry, 19 later gained at least 5% above entry, and 17 later gained at least 10%. The median maximum move after exit was **14.3% above the original entry** and **16.5% above the exit-day close**. This is too frequent to dismiss as hindsight or a few memorable examples.

The study points to two practical changes:

1. **For qualified CANSLIM entries, replace the typical ~1.35 ATR stop with a smaller position sized for 2.5–3.0 ATR of price room.** A close-confirmed 2.5 ATR threshold produced the best 20-session average result in this sample, but it needs a 3 ATR catastrophic hard stop because close-only exits can gap past the intended risk.
2. **After every stop or breakeven exit, automatically monitor the stock for 20 sessions. Make a daily close back above the original entry/pivot the primary re-entry signal.** This was materially better than moving-average reclaims or generic short-term breakouts in this sample.

## Scope and data quality

- The export contained 82 rows. There were 55 closed, exchange-traded long positions opened since June 1 after excluding FX and dot-index symbols.
- FMP daily prices aligned with 51 trades. Four trades—HOOD on June 26, APH on June 16, ARM on June 10, and NVO on June 8—had entry prices that did not align with FMP's same-day range and were excluded from price-sensitive comparisons.
- The main shakeout/re-entry cohort is 28 mature CANSLIM losses or breakeven exits with at least 10 post-exit sessions.
- The stop comparison uses 28 matched CANSLIM trades with at least 20 sessions after entry. It includes winners, losses, and breakevens to avoid testing stops only on trades already known to have failed.
- Repeated entries in the same symbol are counted as separate trading decisions. Therefore, the results describe the journal's decisions, not 28 independent companies.
- Exact exits came from the richer journal table. Approximate shares and planned stop distances were inferred from entry, exit, P&L, and stated dollar risk. Rounded display values introduce some noise, and implausible inferred stop distances were excluded from the median.

## 1. How to stay in without increasing account risk

### The main finding: the existing stop geometry is too tight for the volatility

The median inferred planned stop among mature CANSLIM losses/breakevens was **1.35 ATR**. That is approximately where this sample remained vulnerable to normal daily noise.

| Stop method | Reached +2 ATR before stop | 20-session mean | 20-session median | Win rate |
|---|---:|---:|---:|---:|
| 1.0 ATR hard stop | 32% | -0.34R | -1.00R | 25% |
| 1.5 ATR hard stop | 44% | -0.13R | -1.00R | 39% |
| 2.0 ATR hard stop | 59% | +0.11R | +0.02R | 54% |
| 2.5 ATR hard stop | 69% | +0.15R | +0.23R | 61% |
| 3.0 ATR hard stop | 78% of resolved cases | +0.16R | +0.26R | 64% |
| 2.5 ATR close-confirmed stop | 75% of resolved cases | **+0.22R** | **+0.31R** | **64%** |

Returns assume fixed dollar risk: position size shrinks as the stop widens, and an unresolved trade is marked at the 20th-session close. Hard-stop fills are assumed at the stop price, so real gaps and slippage would make those results slightly worse. The close-confirmed 2.5 ATR rule had a worst result of -1.12R because the close can occur beyond the threshold.

Two closes below the 20-day moving average reached +2 ATR first in only 46% of cases. A five-day structure stop with a 0.25 ATR buffer did better at 61%, but still lagged the 2.5–3.0 ATR methods. The moving average is useful as confirmation, not as the only initial stop.

### Recommended position architecture

Use this only for valid A/B-quality CANSLIM entries with acceptable market conditions—not as permission to give low-quality entries more room.

1. Calculate ATR(14) before entry.
2. Set a **catastrophic hard stop at 3.0 ATR** below entry or below a clearly defined structural low if that is farther away.
3. Use **2.5 ATR below entry on a closing basis** as the normal failure threshold.
4. Size from the catastrophic stop: `shares = allowed dollar risk / (entry − 3 ATR stop)`.
5. The journal's median 1.35 ATR stop divided by the 3 ATR catastrophic distance implies approximately **45% of the old share count** for equal maximum dollar risk. A 2.5 ATR position without the hard-stop buffer would imply about 54%.
6. Add only after the position proves itself and only after recalculating total open risk: a close above the pivot/original entry followed by either a second close above it or a +1 ATR advance. Combined core and add-on risk should remain within the original 1R budget.

This is the crucial distinction: **wider price risk must be paired with smaller size**. Keeping the old share count and simply widening the stop would almost double dollar risk.

### A useful split-position version

- **Core risk allocation (0.5R):** sized for the 3 ATR catastrophic stop; managed on the 2.5 ATR close threshold.
- **Tactical add risk allocation (up to 0.5R):** added only after confirmation; can use the reclaim-day low or roughly 1 ATR below the add price. Reduce the add if the core still carries more than 0.5R of open risk.
- If the stock reaches +2 ATR, take a partial profit or raise risk on part of the position. CRDO and ANET later gave back large portions of their post-exit advances, showing that correct re-entry alone does not prevent round trips.

## 2. How to recognize a shakeout and get back in

### Re-entry rule comparison

A re-entry was considered successful when it reached +2 ATR before falling 1 ATR from the trigger. Only mature signals with enough forward data were scored.

| Re-entry trigger | Triggers | Resolved W-L | Success rate | Median next-10-session maximum gain | Median drawdown |
|---|---:|---:|---:|---:|---:|
| **Close reclaims original entry** | 17 | **8-5** | **62%** | **+11.2%** | **-1.8%** |
| Entry reclaim plus prior-day-high confirmation | 13 | 5-4 | 56% | +10.1% | -3.9% |
| 20-day MA reclaim | 20 | 6-8 | 43% | +9.7% | -5.1% |
| Close above exit-day high | 22 | 7-9 | 44% | +10.4% | -6.0% |
| Undercut and reclaim of exit-day low | 19 | 6-10 | 38% | +8.8% | -7.3% |
| 10-day MA reclaim with strength | 25 | 8-14 | 36% | +10.3% | -9.1% |
| Fresh five-day high above 20-day MA | 22 | 5-9 | 36% | +12.2% | -6.6% |

The original entry/pivot is the most informative reference because it represents the price that first justified the trade. Generic moving-average signals fired more often but produced more false starts and much deeper drawdowns.

### Recommended 20-session re-entry workflow

Immediately after any stop or breakeven exit:

1. Move the symbol to a **Shakeout Watch** list for 20 trading sessions. Do not archive it mentally because the position is gone.
2. Record five levels: original entry/pivot, exit-day high, exit-day low, 10-day MA, and 20-day MA.
3. **Primary trigger:** a daily close back above the original entry/pivot after trading below it. Take a half-risk starter on the next session if price is not more than roughly 0.5 ATR extended.
4. **Secondary trigger for breakeven exits that never lose the pivot:** a close above the exit-day high or a fresh five-day high while above the 20-day MA. This would have helped with TWLO and the July 1 SNOW exit, where a strict “cross back above entry” trigger never occurred.
5. Put the re-entry stop below the reclaim-day low or approximately 1 ATR below the trigger—whichever gives the trade legitimate room—and size so the starter risks no more than 0.5R.
6. Add the remaining half after a second close above the pivot or a +1 ATR move. Do not add merely because the price is lower.
7. At +2 ATR, realize part of the gain and trail the remainder with the 10-day line or a two-close rule. This protects against the CRDO/ANET type of large advance that later fades.
8. Allow no more than two re-entry attempts per symbol without a completely new base. Stop watching after 20 sessions, a decisive 50-day-line failure, or a material fundamental change.

### The most important trade examples

| Trade | Exit | Objective re-entry evidence | August 10 close | Gain vs original entry |
|---|---:|---|---:|---:|
| SNOW, June 18 entry | 229.43 on June 24 | Entry, 10-day, 20-day, exit-high, and five-day-high reclaim all aligned June 26 | 334.70 | **+43.8%** |
| TWLO, June 18 entry | 187.50 on June 24 | Exit-day-high break June 29; 20-day reclaim June 30 | 250.06 | **+33.5%** |
| NTAP, July 2 entry | 160.94 on July 8 | Exit-day-high/five-day-high break July 9; entry/20-day reclaim July 17 | 198.72 | **+23.0%** |
| W, July 9 entry | 89.37 on July 10 | Entry, 10-day, and 20-day reclaim aligned July 15 | 103.26 | **+16.0%** |
| DXCM, July 10 entry | 74.85 on July 17 | 20-day reclaim July 27; 10-day July 28; entry/exit-high reclaim July 31 | 87.65 | **+15.9%** |
| S, July 16 entry | 19.07 on July 17 | 20-day reclaim July 29; 10-day/five-day high July 31; entry reclaim August 3 | 22.23 | **+11.4%** |

ANET reached 18.5% above entry after its loss but was only 5.6% above entry by August 10. CRDO reached roughly 31% above its June 9 entries but was only about 1%–2% above them by August 10. These are the strongest arguments for taking something off at +2 ATR after re-entry instead of relying only on an open-ended trailing exit.

## A concise rule card

### Initial entry

- Valid CANSLIM setup and acceptable market condition.
- Normal failure threshold: close 2.5 ATR below entry.
- Catastrophic hard stop: 3 ATR below entry.
- Size from the 3 ATR stop; approximately 45% of the old share count at the sample's median prior stop distance.
- Add only after confirmation.

### After exit

- Automatic 20-session Shakeout Watch.
- Primary re-entry: close back above original entry/pivot.
- Secondary: exit-day-high/five-day-high break above 20-day MA if price never lost the pivot.
- Begin at half risk, add after confirmation.
- Partial at +2 ATR; trail the remainder.
- Maximum two attempts without a new base.

## Limitations

This is a small, overlapping sample from one market period. Daily bars do not reveal the intraday order in which a target and stop were touched, hard-stop simulations assume a fill at the stop price, and the 20-session mark is a standardized comparison rather than a complete sell strategy. The results support a controlled live trial—not an immediate wholesale change at full size. A sensible next step is to paper-track or use half-normal account risk for the next 20 qualified CANSLIM trades and compare the new rules with the old process.

This report is decision support and trading-process research, not a guarantee of future performance or personalized investment advice.
