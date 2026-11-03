# Kalshi live-event market scan (2026-11-03)

Question: where on Kalshi does a machine that watches or listens to a live stream have an edge, now that the college pre-game replay showed the score-only model has none (BENCHMARKS.md, "Replay against Kalshi's own 2025-26 prices")?

Method: every series on the public API (15,663), settled markets per series via `/markets?status=settled` and `/historical/markets`, trade tapes via `/markets/trades` and `/historical/trades`, hourly candles via the candlestick endpoints. All public, no key. Scripts in the session scratchpad; outputs summarised here.

## Volume by category since May 2026 (settled markets, contracts)

| category | volume |
|---|---|
| Mentions (will X say Y during event Z) | 525M, 32,957 markets, 1,744 events, ~66 events and ~540 YES resolutions per week |
| Entertainment (Rotten Tomatoes, reality-TV eliminations, charts) | 379M |
| Politics | 279M |
| Science and Technology (launches, LLM rankings) | 78M |

Roughly 29% of non-sports, non-mention volume is in markets that settle on something visible on a live or first-air broadcast (reality-TV eliminations and winners, celebrity attendance, launches).

## Mention markets

- 82,381 settled markets since Feb 2025, 1.31B contracts. Median market 5,300 contracts; 19% above 20,000. Biggest series: KXTRUMPMENTION (261M), KXWCMENTION (150M), KXTRUMPSAY, KXNBAMENTION, KXNFLMENTION, KXSECPRESSMENTION, KXFEDMENTION.
- **Pre-event pricing is calibrated.** 2,500-market sample with volume >= 1,000: 6 h before close mean price 0.481 vs YES rate 0.481, Brier 0.153; spreads 4c. Buying NO or YES blindly in any price band loses 4-10c per contract. There is no slow edge.
- **The move at resolution is the only edge, and it is a speed race.** On 463 high-volume YES markets the price goes from the last trade under 0.80 to the first trade at or above 0.90 in a median of 8 s (41% within 5 s, 59% within 30 s). On 148 typical markets (3k-20k volume) the median is 15 s. Money traded at uncertain prices during that move (a lower bound on what a faster bot could have taken): median $41, mean $80, p90 $192 per typical market; median $84, mean $384, p90 $704 on the high-volume ones; essentially all of it is YES takers sweeping asks. Five minutes before the move the median price is 0.63-0.67, so the outcome is genuinely open.
- Bots already exist (Sonist and others watch audio and captions); a White House teleprompter operator is under CFTC investigation for trading these, which says the markets are soft enough that information wins. Resolution disputes (who counts as the speaker, word variants) are the main model risk.

## Visually resolved markets (detector territory)

| series | median market volume | price 5 min before | move time | money in the move (median / max) | late money, 10 min before |
|---|---|---|---|---|---|
| Dancing with the Stars elimination | 104k | 0.59 | 0.5 s | $128 / $763 | $5,447 |
| Big Brother elimination | 125k | 0.62 | 5 s | $61 / $927 | $2,204 |
| World Cup celebrity attendance | 142k | 0.74 | 34 s | $219 / $712 | $1,000 |
| Love Island USA weekly elimination | 419k | 0.79 | 200 s | $278 / $3,042 | $733 |
| Starship launch outcome | 115k | 0.90 | minutes | $237 / $2,348 | $165 |
| Met Gala attendance | 91k | 0.86 | hours | $25 / $583 | $5 |
| Survivor winner (finale) | 2.9M | 0.83 | days | $20k / $40k | $1,428 |

Reality-TV live eliminations move in under 5 s (bots or fast viewers already there); attendance markets move in tens of seconds and settle on a face appearing on a broadcast, which is literally a detection problem; launch outcomes (catch, explosion) are a handful a year.

## Verdict

1. There is no pre-event mispricing to exploit in mention markets; the edge is purely latency at the moment the word is spoken or the face appears. The race is against humans and existing bots, not data feeds, which is the one structural difference from sports.
2. Realistic size: tens of dollars per typical market, hundreds on big events, with ~540 YES resolutions a week. Capturable share depends entirely on being faster than the bots already there (sub-second speech recognition on the lowest-latency feed, vision for speaker identity and on-screen text).
3. Detector/tracker relevance: speaker identification and OCR as a second sensor for mention markets; face recognition for attendance markets; event detection for launches and eliminations. The sports player tracker and court calibration have no role here.
4. Before building: paper-trade the resolution moment with a live ASR bot on one week of events and measure fill prices against this tape. If the bot cannot buy below 0.90 on most YES resolutions, the existing bots are faster and the edge is gone.
