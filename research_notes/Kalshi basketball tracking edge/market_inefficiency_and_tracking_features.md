# Where quantitative models have shown persistent edges in basketball betting, and what tracking data adds beyond box scores (as of 2026-11-01)

Scope note for the writer: this file covers five key questions. Every number below is tagged with its source. Where I could only read an abstract or a search-engine summary of a paper (several PDFs failed to parse and ScienceDirect/ats.io returned 403s), I say so. Calculations I did myself are labeled "my calculation." Anecdotes and affiliate/blog claims are labeled as such and should be weighted lightly.

---

## Key Question 1: Which basketball market segments are least efficient? (NBA vs NCAA vs WNBA; spreads/totals vs props vs live; early vs closing lines; prediction markets vs sportsbooks)

### Takeaway
Peer-reviewed work consistently finds NBA closing spreads and moneylines are efficient to within the vig (inefficiencies, where found, live in opening/early lines, week-1 totals, and large home underdogs), while college basketball, WNBA, player props, and in-game prediction-market prices show measurable but mostly small, short-lived, or liquidity-capped inefficiencies. The most robust 2024-2026 finding is that Kalshi in-game NBA prices underreact to public information (about 0.64-for-1 on impact with predictable drift), and that Polymarket NBA arbitrage exists but is retail-scale and seconds-long.

### Cited Findings

**NBA pre-game spreads and totals (academic)**
- Compton & Sigler, using the Even-Noble log-likelihood test on 2000-01 through 2007-08 data, found NBA point-spread and totals markets efficient; the lone exception was that betting home underdogs of 10+ points was profitable — [The Sport Journal](https://thesportjournal.org/article/nba-gambling-inefficiencies-a-second-look/)
- "Learning, price formation and the early season bias in the NBA" (Finance Research Letters): totals lines are significantly biased early each season while sides are not; a strategy against the closing totals line early in the season won 56.72% of the time, but the authors state no profitable strategy exists at the closing line in general because biases are removed as books respond to information and bettor action — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1544612307000177); also indexed at [ETSU Works](https://dc.etsu.edu/etsu-works/17780)
- Girdner, Davis, Fodor & Kirch, *Journal of Prediction Markets* 7(2), 2013, NBA 2009-2012: totals market "generally very efficient," but in Week 1, 58.2% of games went under the total (break-even 52.38%); betting all Week 1 unders returned an average 11.1% per game. The abstract gives no test statistic — [UBPLJ](https://www.ubplj.org/index.php/jpm/article/view/601)
- "Player absence and betting lines in the NBA" (Finance Research Letters, 2015): opening lines show exploitable biases around player absences, but most are eliminated by close; the closing line is described as a 50-50 bet. The same paper cites Gandar et al. (1998) showing open-to-close NBA line movement significantly improves forecast accuracy — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1544612315000227) (abstract only; full page returned 403)
- A weak-form test across major North American sports (preprint) reported statistically significant inefficiencies in NFL, NCAAF, NCAAB and MLB but could not demonstrate significant inefficiencies in the NBA or NHL; NCAAB returns varied widely and extreme college longshots had very poor returns — [ResearchGate, "On the Efficiency of Sports Betting Markets"](https://www.researchgate.net/publication/365618651_On_the_Efficiency_of_Sports_Betting_Markets) (search summary only; PDF not read)
- Robbins' weak-form efficiency paper: pro basketball returns are generally negative and betting underdogs/longshots in basketball is a losing proposition — [ECU PDF](https://myweb.ecu.edu/robbinst/PDFs/Weak%20Form%20Efficiency%20in%20Sports%20Betting%20Markets.pdf) (search summary only; PDF failed to parse)
- Karl Whelan's analysis of 3,670 NBA moneyline games (2022-23 to 2024-25, ~10 US books per bet, quotes at 24h/18h/12h/6h/1h/10min): about half of all bets had positive CLV; three of five positive-CLV deciles did not profit; the 9th decile averaged +5% CLV but only +0.2% return; only the top decile profited meaningfully (+11.4%). In the 24-hour sample the top decile returned ~16% while the bottom decile lost ~25%; he estimates NBA moneyline margin at ~4.5% — [Karl Whelan](https://www.karlwhelan.com/?p=2595)
- An older thesis found NBA spreads weak-form efficient but rejected semi-strong efficiency — [Lust, Miami Univ. thesis](https://etd.ohiolink.edu/acprod/odb_etd/ws/send_file/send?accession=miami1533158563565605&disposition=inline); a 34-year MDPI study tested "hot hand" effects in NBA spread betting — [MDPI](https://www.mdpi.com/2227-7072/2/4/359) (titles only; findings not read)
- COVID-era: "The Impact of COVID-19 on Sports Betting Markets" reports NBA moneyline markets "became very inefficient" during the disruption — [arXiv 2109.07581](https://arxiv.org/pdf/2109.07581) (search summary)

**College basketball**
- Colquitt, Godwin & Caudill (2001, *Journal of Business Finance & Accounting*): evidence of inefficiency in men's college basketball was limited — [Wiley](https://onlinelibrary.wiley.com/doi/10.1111/1468-5957.00372)
- A later paper examined efficiency in the NCAA tournament market specifically — [ResearchGate, "Efficiency in the madness?"](https://www.researchgate.net/publication/339775792_Efficiency_in_the_madness_examining_the_betting_market_for_the_ncaa_men's_basketball_tournament) (title only)
- Whelan notes that in college basketball, early bets on underdogs generate positive CLV, but closing odds are so poor the CLV does not translate into wins — [Karl Whelan](https://www.karlwhelan.com/?p=2595)
- Ramesh et al. (2019, arXiv, Rice): a non-parametric win-probability model applied to Vegas lines 1990-2017 (16 casinos; NCAAB restricted to games with a Power Five team) reported optimal ROI of 4.07% (simple) / 13.02% (weighted) on NCAAB (645 / 284 bets) vs 9.04% / 11.18% on NBA (1,500 / 619 bets) and 17.01% / 12.98% on WNBA (151 / 216 bets). Important caveat: thresholds were optimized in-sample and the bootstrap resamples the same sample; there is no true holdout. Without thresholds, NCAAB ROI under the simple model was -1.10% — [arXiv 1910.08858v2](https://arxiv.org/html/1910.08858v2)

**WNBA**
- Paul & Weinbach (2014, *International Journal of Financial Studies* 2(2):193-202), WNBA 2007-2012: a regression rejects efficiency at the 10% level, but simple betting strategies did not earn statistically significant returns; bettors strongly prefer the best teams, especially on the road, yet fading popular sides was not significantly profitable — [MDPI](https://www.mdpi.com/2227-7072/2/2/193); [IDEAS](https://ideas.repec.org/a/gam/jijfss/v2y2014i2p193-202d35451.html)
- Kochman & Goodwin (Kennesaw State) tested WNBA spreads on the premise that emerging markets lack participation needed for efficient pricing — [Kennesaw DigitalCommons](https://digitalcommons.kennesaw.edu/facpubs/2419) (findings not visible)
- The WNBA signed with Genius Sports to implement Second Spectrum tracking — [BettingOdds.com](https://www.bettingodds.com/us/news/wnba-partners-with-genius-sports-to-implement-second-spectrum-tracking-data)

**Live / in-play**
- Angelini & De Angelis (June 2026, arXiv): using a real-time prediction market with binary payoffs and high-frequency prices, a one-minute change in an out-of-sample benchmark probability is associated with only ~0.64-for-1 contemporaneous price change; the unabsorbed portion predicts drift over the following several minutes; underreaction is much larger when liquidity is low and for salient signals. The search summary of the same paper reports Kalshi NBA pre-game calibration across 1,421 games with Brier 0.204 at 24h before tip improving to 0.199 at close — [arXiv 2606.07811](https://arxiv.org/abs/2606.07811); Brier figures via [arXiv PDF](https://arxiv.org/pdf/2606.07811) (search summary)
- A review of in-play betting notes there is "still very little publicly available academic research on in-play betting," mostly on exchanges and mostly football; in-play inefficiency correlates with surprise — [Advisor Perspectives](https://www.advisorperspectives.com/articles/2020/10/05/are-sports-betting-markets-efficient) (search summary)
- Covers reports books "tend to set strict limits on in-play betting" — [Covers](https://www.covers.com/industry/how-sportsbook-limits-work-behind-the-scenes-july-14-2025)

**Prediction markets vs sportsbooks (2024-2026)**
- Cheng, Yang & Zou (UCLA, arXiv 2605.00864, Polymarket NBA, Feb 4 - Mar 4 2026, 173 games, ~75M order-book snapshots, zero trading fees at the time): pre-game single-market arbitrage episodes: 0; in-game: 7 valid episodes, median duration 3.6 s, median yield ~11%, capped profit $210 total. Combinatorial (moneyline vs spread): 11 pre-game episodes (0.0034% of time, median 11 s, median 309 bps) and 279 in-game (0.176% of time, median 16 s, median 101 bps); in 76.9% of episodes a $100 budget could not be fully deployed (avg executable ~14.8 synthetic shares). Median bid-ask: 392 bps pre-game vs 1,031 bps in-game. Player-prop markets averaged only ~13 trades and $253 volume per market vs ~10,412 trades / $1.43M for full-game moneylines. Polymarket NBA volume grew from $51M (2024) to $0.89B (2025) — [arXiv 2605.00864](https://arxiv.org/html/2605.00864v1)
- Citizens analysis (Dec 26, 2025, NFL Week 17, 26 data points, pre-game moneylines and totals): Kalshi odds were 8% worse than FanDuel and 6% worse than DraftKings including fees. Methodology not defined — [CDC Gaming / NEXT.io](https://cdcgaming.com/brief/kalshi-pricing-remains-worse-than-sportsbooks-says-citizens/)
- Earlier NFL-week analysis reported Kalshi ~7% worse on moneylines and ~10% worse on totals than sportsbooks (2025, football) — [Yogonet](https://www.yogonet.com/international/news/2025/09/09/115258-kalshi-trading-hits-441m-in-first-nfl-week-but-analysts-flag-pricing-gap-with-sportsbooks)
- Blockworks report (secondary; I could not fetch it): Kalshi and FanDuel prices correlate up to 0.996; arb gaps small and short-lived; Kalshi prices deviate ~5.5 pp from outcomes two days out (basketball pooled with NFL) — [Blockworks](https://app.blockworksresearch.com/research/from-betting-to-trading-how-kalshi-is-reshaping-sports-markets) (search summary)
- Claremont thesis (NFL 2025-26): after controlling for how each platform spreads quoted probabilities, the accuracy gap between Kalshi and DraftKings largely disappears — [Claremont](https://scholarship.claremont.edu/cmc_theses/4214) (search summary)
- Le (arXiv 2602.19520, v2 Aug 2026): 353M trades across 429k binary contracts on Kalshi and Polymarket; the most robust miscalibration is underconfidence in *political* markets; about half of raw calibration-slope variation looks like estimation noise. No sports-specific result is given in the abstract — [arXiv 2602.19520](https://arxiv.org/abs/2602.19520)
- Pew: prediction-market volume doubled May-July 2026, driven by sports; sports is the most traded topic on both platforms — [Pew Research](https://www.pewresearch.org/short-reads/2026/09/23/prediction-markets-trading-volume-doubled-between-may-and-july-largely-driven-by-sports/)
- Vendor comparisons (e.g., "77% better odds across 5,000+ identical markets," Brier ~0.09) exist but are marketing-grade with no methodology — [Keyrock](https://keyrock.com/knowledge-hub/polymarket-vs-kalshi/); treat as unverified

### Inferences
- The ordering of inefficiency implied by the literature, from most to least efficient: NBA closing spreads/moneylines > NBA closing totals (except week 1) > NCAAB (significant weak-form inefficiency in one cross-sport study, but limited in Colquitt et al.) ≈ WNBA (regression rejects efficiency at 10% but no significant profits) > opening/early-week NBA lines (documented exploitable biases) > illiquid prediction-market props and late in-game exchange prices (measurable mispricing, but seconds-long and capped at tens of dollars).
- Early-week vs closing: the academic consensus (Gandar 1998; player-absence paper; Whelan's 24h decile spread) is that the signal is in the move from open to close, which means an edge is most likely harvested by betting openers against a model and then measured by CLV, not by betting closers.
- The Kalshi in-game underreaction result (0.64 impact ratio with multi-minute drift) is the most actionable 2026 academic finding for basketball, because basketball generates dense public signals and the drift is predictable from a public benchmark; but it is largest precisely where liquidity is thin, so capacity is the constraint.
- Prediction-market pre-game prices appear to be roughly as calibrated as books (Brier 0.199 at close on Kalshi) but *worse priced after fees* by 6-8% in the only independent head-to-head I found (NFL, Citizens); there is no published NBA-specific head-to-head.

### Gaps
- No peer-reviewed study directly compares NBA vs NCAAB vs WNBA efficiency on the same methodology with post-2018 (post-PASPA) data.
- No peer-reviewed study of in-play *sportsbook* NBA markets; the in-play evidence is from exchanges/prediction markets.
- No NBA-specific Kalshi-vs-sportsbook closing-price comparison; the Blockworks NBA figure is pooled with NFL and I could not access the report.
- Full text of the Robbins and cross-sport weak-form papers could not be parsed; effect sizes for NCAAB vs NBA are from abstracts only.

---

## Key Question 2: Which tracking-derived features have demonstrated out-of-sample predictive value for game outcomes, totals, or player stats?

### Takeaway
I found no peer-reviewed, out-of-sample evidence that tracking-derived features (spacing, speed/distance, closeouts, lineup matchups) improve prediction of NBA game outcomes or totals *relative to the betting market*. The best-documented tracking work either (a) values possessions/shots in real time (EPV, qSQ), (b) predicts talent rather than games (Stats Perform: log-loss 0.30 vs 0.40 for play-by-play), or (c) tests fatigue and finds effects that are statistically significant but explain under 1% of variance; the schedule features that do move team net rating (rest difference, 3-in-4, three-hour westward travel) are box-score-derivable and are already priced by books.

### Cited Findings

**Tracking data for game-level prediction**
- Stats Perform's Sloan paper using college tracking data to predict whether players reach the NBA reports tracking-based forecasts with lower log-loss than play-by-play-based forecasts (0.30 vs 0.40). This is talent prediction, not game prediction — [Sloan Sports Conference](https://www.sloansportsconference.com/research-papers/predicting-nba-talent-from-enormous-amounts-of-college-basketball-tracking-data)
- Cervone et al. (Sloan 2014) POINTWISE/EPV predicts expected points from optical tracking in real time, at the possession level, not game outcomes — cited in [HDSR "Recreating the Game"](https://hdsr.mitpress.mit.edu/pub/kxks56er/release/5)
- A ScienceDirect study of 632 NBA games from 2015-16 found shooter velocity and defender distance among the factors distinguishing made from missed shots — [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S3050544525000040) (search summary)
- Second Spectrum's qSQ is a shot-level expected eFG% for an average player given tracking context; qSI = actual eFG minus qSQ isolates shooter skill — [ESPN](https://africa.espn.com/nba/playoffs/2016/story/_/id/15530589/what-advanced-tracking-data-reveals-nba-shooters). I found no reliable source on how fast qSQ/qSI stabilize or whether they out-predict box-score shooting for betting purposes; blog claims (e.g., "most powerful predictor of regression," a 0.47 out-of-sample correlation) cite no study — [TechBuzz Ireland](https://techbuzzireland.com/2026/02/02/beyond-the-box-score-feature-engineering-for-predictive-sports-models-focusing-on-nba-player-props-and-advanced-metrics/) (unverified blog)
- A Bryant University honors thesis reports ~65% game-prediction accuracy mostly from box-score/player stats, and mentions RAPTOR blends box score with tracking — [Bryant DigitalCommons](https://digitalcommons.bryant.edu/cgi/viewcontent.cgi?article=1000&context=honors_data_science) (student work)
- Genius Sports acquired Second Spectrum (~$200M, 2021); Second Spectrum is the NBA's official optical tracking provider; Genius supplies official data to 400+ sportsbook operators, and analysts expect the tracking feed to power in-play pricing — [Sportico](https://www.sportico.com/business/finance/2021/genius-buys-second-spectrum-1234632011/); [Covers](https://www.covers.com/industry/genius-sports-betting-advanced-cross-selling-november-2022); [SBC News](https://sbcnews.co.uk/sportsbook/2021/05/07/genius-bolsters-sports-portfolio-with-200m-second-spectrum-acquisition/)

**Fatigue, speed/distance (tracking-based)**
- Stephen, Yep & Fain (CMU, arXiv 2112.14649, advisers Yurko & Horowitz), using Second Spectrum in-game distance via nbastatR, 2015 season windowed regression n = 19,751 player-games, schedule data 2010-2019:
  - In-game distance vs season average is *positively* associated with points (p = 1e-12 for 12+ min players) and FG%, but the largest adjusted R² is 0.0013 (under 1% of variance).
  - Prior 5- and 7-day distance load, after adjusting for playing time, has a small negative coefficient on next-game distance (p = .003 and 1.1e-6), which the authors say may be regression to the mean; prior load does not reduce points.
  - Kawhi Leonard load-management case: no significant PIE difference after extra rest.
  - Schedule model on net-rating residuals: +0.35 points per extra day of rest difference; -1.74 for traveling three hours back (westward); -1.29 for third game in four days; second leg of back-to-back and flight-distance metrics not significant; these factors explain under 0.1% of variance. Authors conclude the impact of cumulative fatigue is "minimal" and "overstated" — [ar5iv 2112.14649](https://ar5iv.labs.arxiv.org/html/2112.14649)

**Rest, travel, circadian (box-score / schedule-derived, not tracking)**
- Steenland & Deddens (1997, *Sleep*), 8,495 games: peak performance at 3 days between games; cost of short gaps attributed to physical recovery rather than jet lag; few consistent distance/direction effects — via [search summary](https://cosmosmagazine.com/health/body-and-mind/circadian-rhythms-time-zone-jumping-impacts-nba-performance-study)
- *Chronobiology International* (2022, 2014-2018 data): greater circadian misalignment and greater travel distance both reduce performance, and distance matters more than misalignment; a separate 25,016-game paper finds away performance depends more on travel fatigue than phase shift — [Informa](https://informahealthcare.com/doi/full/10.1080/07420528.2022.2113093); [Informa 2024](https://informahealthcare.com/doi/full/10.1080/07420528.2024.2325641) (search summaries)
- *Journal of Clinical Sleep Medicine* (2021), 2013-2020 seasons: travel distance/direction on back-to-backs; accumulated travel fatigue and chronic circadian desynchronization disturb sleep and recovery — [Univ. of Arizona](https://experts.arizona.edu/en/publications/impacts-of-travel-distance-and-travel-direction-on-back-to-back-g/)
- *Scientific Reports* (2020) used the 2020 COVID bubble as a natural experiment separating travel from time-zone effects — [PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7732833/)
- Entine (Wharton): the NBA schedule favors home teams by limiting their back-to-backs; visitors on the second night of a back-to-back score about one point less — [Wharton PDF](https://faculty.wharton.upenn.edu/wp-content/uploads/2012/04/Nba.pdf)
- Betting-industry sources say books shorten lines ~1-2 points for a team on the second night of a back-to-back; several call spreads the most efficient market for this factor. These are blogs, not studies — [NBAStatsForBetting](https://nbastatsforbetting.com/schedule-rest-back-to-back/) (anecdotal)
- Shooting under fatigue (lab, not NBA): long-range accuracy declines significantly after fatigue in college players; mid-range does not — [PeerJ/PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12435330/); 3-point accuracy declines with basketball-induced fatigue in high-level players — [PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10974731/)

**Lineups / absences**
- Opening lines around player absences are biased and exploitable; most bias is gone by close — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1544612315000227) (abstract)
- The NBA asked sportsbooks not to offer unders on two-way or 10-day players (an integrity rule that also removes a class of minutes-driven prop bets) — [Action Network](https://www.actionnetwork.com/nba/nba-asks-sportsbooks-not-to-offer-unders-on-2-way-or-10-day-players)

### Inferences
- The tracking features with documented *in-game* value (defender distance, shooter velocity, qSQ) operate at the shot/possession level; their game-level predictive lift against a market that already prices season-to-date efficiency is undemonstrated in public literature. The public evidence base supports tracking data for valuation and live state estimation more than for pre-game pricing.
- Fatigue/speed-distance features are the best-studied tracking inputs and the result is negative for betting purposes: significant p-values but R² of 0.001, with the practically meaningful schedule effects (3-in-4 at -1.29, westward 3-hour travel at -1.74, rest at +0.35/day) requiring no tracking data at all and being well known to books.
- Since Genius/Second Spectrum sells the same tracking feed to 400+ books, any edge from tracking must come from better modeling of the feed, not from access to it; live underreaction (Key Question 1) is where a faster tracking-informed win-probability model could plausibly show CLV.
- Player-absence and lineup news is the one input class with academic evidence of exploitable *opening* line bias; a tracking-informed lineup-impact model (e.g., RAPM-style with on/off from tracking) would be measured by CLV on openers.

### Gaps
- No Sloan/JQAS/arXiv paper found that tests tracking-derived features against the closing line for NBA sides or totals out-of-sample. Searches for Sloan "tracking data" + "game outcome" returned talent prediction and possession-value papers only.
- No published study of tracking-derived minutes or points/rebounds projections vs sportsbook prop lines.
- Stabilization/reliability of qSQ, qSI, spacing, or closeout metrics is not documented in accessible sources.
- Pace: I did not find a study isolating tracking-derived pace (possession speed, time-to-shot) as a totals predictor beyond box-score pace; this remains unverified.

---

## Key Question 3: Player props: evidence of inefficiency, limits, and how books/exchanges price them

### Takeaway
There is no peer-reviewed study of sportsbook NBA player-prop efficiency; what exists is (a) margin data showing props carry ~8% hold vs ~5% on mainlines, (b) consistent reporting that prop limits are low (as low as ~$100) and are the first thing cut for winning accounts, and (c) Polymarket data showing prop markets are extremely thin (~13 trades, ~$253 per market). Props are plausibly the least efficiently priced segment, but the hold and limits mean the extractable dollar edge is small.

### Cited Findings
- A 2026 Mendeley Data margin study (4M+ odds, 22 leagues, 52 bookmakers) reports average margins of 7.93% on player props vs 4.77% on head-to-head and 5.45% on totals; licensed US books averaged 4.72% across markets vs 4.55% offshore; NBA/WNBA assists props "can exceed 7%" (source for the last figure unnamed) — [Low Juice Sportsbooks](https://lowjuicesportsbooks.com/prop-market-margins-in-mlb-and-nba/) (secondary; I could not locate the Mendeley dataset)
- ESPN: books protect themselves on player props with smaller limits and more juice than on spreads; the offering is too large every week for one book to price efficiently, so they minimize liability with low limits — [ESPN Chalk](https://africa.espn.com/chalk/story/_/id/34371732/the-future-nfl-betting-why-player-props-not-point-spreads-expected-king) (NFL context)
- Covers (July 2025): limits on niche sports and specific player props are "sometimes as low as $100," versus "$100,000 or more" on popular NFL markets; successful players face stricter limits; books "tend to set strict limits on in-play betting" — [Covers](https://www.covers.com/industry/how-sportsbook-limits-work-behind-the-scenes-july-14-2025)
- Action Network: books limit advantage players who specialize in props; a bettor who got $200 per prop may later get $20 — [Action Network](https://www.actionnetwork.com/education/limits)
- Massachusetts Gaming Commission discussion: BetMGM's listed limiting triggers include exploiting odds errors, syndicate betting, court-siding, bets on low-liquidity high-volatility markets, and inconsistent bet amounts; operators say limits rest on behaviors, not results — [BettingUSA](https://www.bettingusa.com/sportsbooks-ban-smart-customers/)
- Industry sources state books track CLV rather than win-loss to identify sharps, and that a "soft limit" on props arrives first while mainlines are left alone — [Bettingsite.net](https://www.bettingsite.net/blog/sportsbook-limiting-winning-bettors/); [Track360](https://track360.io/blog/sportsbook-bet-limiting-sharp-player-liability-operator-guide-2026) (industry blogs)
- Polymarket NBA (Feb-Mar 2026): 5,074 player-prop markets averaged ~13 trades and $253 volume each, vs ~10,412 trades and $1.43M for full-game moneylines; props were not tested for arbitrage — [arXiv 2605.00864](https://arxiv.org/html/2605.00864v1)
- Post-Rozier/Porter: the NBA and sportsbooks are reviewing prop markets; an NBA spokesperson said individual-performance props raise heightened integrity concerns; some prop types were removed — [Yahoo Sports](https://sports.yahoo.com/article/nba-sportsbooks-reportedly-reviewing-prop-155600296.html); the NBA asked books not to offer unders on two-way/10-day players — [Action Network](https://www.actionnetwork.com/nba/nba-asks-sportsbooks-not-to-offer-unders-on-2-way-or-10-day-players); calls for prop bet caps (e.g., $1,000) followed the Malik Beasley case — [SportsGrid](https://www.sportsgrid.com/nba/video/malik-beasley-case-fuels-calls-for-nba-prop-bet-caps)
- Kalshi: player props "sometimes use slightly different multipliers" in the fee schedule; live markets use the same schedule as pre-game — [Prediction.com fee guide](https://prediction.com/blog/kalshi-fees-complete-guide-2026) (third-party)

### Inferences
- A prop edge must exceed roughly 8% hold on a two-way line (my reading of the Mendeley margins), i.e., a true hit rate of about 54% at typical -120/-120 pricing, before it is positive EV; at -110/-110 the figure is 52.4%. The higher hold roughly doubles the edge required versus mainlines.
- Prop capacity at books is bounded by low limits that shrink further once CLV is detected; the exchange alternative (Polymarket/Kalshi) currently has negligible prop liquidity, so there is no present venue to scale a prop edge.
- The integrity-driven removal of unders on marginal players removes the prop class most tied to minutes uncertainty, which is the class where a minutes/rotation model would have had the most informational advantage.

### Gaps
- No peer-reviewed, out-of-sample study of NBA player-prop pricing efficiency at sportsbooks was found.
- The Mendeley 2026 margin dataset was not located directly; its figures are reported via a low-juice affiliate site.
- No source documented the actual prop limits of specific US books (ESPN Bet, DraftKings, FanDuel) for NBA in 2025-26.

---

## Key Question 4: How closing line value is computed and used; what CLV implies about edge; how many bets separate skill from luck

### Takeaway
CLV is computed as the improvement of the odds taken over the (no-vig, sharp-consensus) closing odds; it is used because it converges on skill far faster than P&L, but Whelan's 3,670-game NBA dataset shows it is not sufficient — positive CLV only turned into profit in the top CLV decile. Detecting a realistic 2-3% ROI edge with results alone requires roughly 3,000-9,000 bets (my calculation; consistent with third-party rules of thumb), which is why practitioners track CLV instead.

### Cited Findings
- CLV definition: percentage improvement in decimal odds taken versus the closing odds; Whelan uses the median across ~10 licensed US books of the last quote before tip — [Karl Whelan](https://www.karlwhelan.com/?p=2595)
- Whelan (NBA moneylines, 2022-23 to 2024-25, 3,670 games, 34,944 quotes per timestamp): positive-CLV bets outperform negative-CLV bets, but three of five positive-CLV deciles did not profit; 9th decile +5% CLV yielded +0.2%; top decile +11.4%; at 24h the top decile returned ~16% and the bottom lost ~25%; NBA moneyline margin ~4.5%, so edge must exceed the margin, not just be positive — [Karl Whelan](https://www.karlwhelan.com/?p=2595)
- Books themselves reportedly monitor CLV rather than win-loss to flag sharps — [Bettingsite.net](https://www.bettingsite.net/blog/sportsbook-limiting-winning-bettors/) (industry blog)
- Practitioner guidance recommends measuring against a no-vig closing line from a sharp source (Pinnacle or a consensus such as the "Unabated Line," a vig-free consensus from the sharpest books) — [Dealroom profile of Unabated](https://app.dealroom.co/companies/unabated_products); multiple third-party guides — [sports-models-kb](https://sports-models-kb.pages.dev/closing-line-value) (unofficial; I could not locate Pinnacle's primary article)
- Third-party rules of thumb for sample size (unsourced; treat as rough): 200 bets detects ~3% edge, 500 bets ~2%, 1,500 bets ~1%, each at ~80% confidence — [Domain-Knowledge-Notebook / search summary](https://ercbk.github.io/Domain-Knowledge-Notebook/qmd/sports-betting.html); other guides say 500-1,000 or 200-500 bets for CLV to be reliable — [DeucesCracked](https://www.deucescracked.com/blog/closing-line-value-clv-sports-betting-explained-2026) (affiliate)
- One tracked dataset of 12,437 bets reports the correlation between CLV and realized ROI rising from 0.42 at 200 bets to 0.97 at 5,000 bets — [Sharpery](https://sharpery.vercel.app/research/clv-long-term-profitability) (unverified, self-published)
- Calibration benchmark: Kalshi NBA pre-game Brier 0.204 (24h) to 0.199 (close) over 1,421 games — [arXiv 2606.07811](https://arxiv.org/pdf/2606.07811) (search summary); Angelini & De Angelis's own validation uses an out-of-sample benchmark probability model and measures price response ratio and subsequent drift — [arXiv 2606.07811](https://arxiv.org/abs/2606.07811)
- Academic validation examples: Even-Noble log-likelihood tests for spread efficiency — [The Sport Journal](https://thesportjournal.org/article/nba-gambling-inefficiencies-a-second-look/); bootstrap percentile intervals with Bonferroni across sports (but in-sample threshold optimization) — [arXiv 1910.08858v2](https://arxiv.org/html/1910.08858v2); Sill (Sloan 2010) regularized adjusted +/- with explicit out-of-sample testing — cited in [HDSR](https://hdsr.mitpress.mit.edu/pub/kxks56er/release/5)

### Inferences (my calculations, standard binomial power; verify before quoting)
- Break-even win rate at -110 = 110/210 = 52.38%. ROI at -110 given true win rate p is 1.909p - 1: p = 53% gives +1.2%, 54% gives +3.1%, 55% gives +5.0%, 56% gives +6.9%.
- Bets needed to reject "no edge" with results alone (two-sided alpha 0.05, 80% power, per-bet SD ~0.5 on the win indicator): n ≈ (1.4 / (p - 0.5238))². p = 56%: ~1,500 bets; 55%: ~2,900; 54%: ~7,500; 53%: ~51,000. At one NBA game per day, 55% skill takes roughly two full seasons of 5+ bets per day to confirm by P&L.
- Expressed in ROI with per-bet SD ~1 unit: n ≈ (2.8 / ROI)²: 5% ROI ~3,100 bets; 3% ~8,700; 2% ~19,600; 1% ~78,000. The third-party "200 bets for 3%" rule of thumb is therefore far too optimistic for P&L; it is closer to right for CLV, whose per-bet variance is much smaller because it compares two prices rather than a price to a binary outcome.
- CLV-to-edge mapping: average CLV approximates expected ROI *only if* the closing line is unbiased and the CLV is measured against a no-vig close. Whelan's NBA data imply that a mean CLV of about +5% on vigged US moneylines corresponded to roughly 0% realized ROI, so the vig (~4.5%) must be subtracted; CLV above roughly +5% was where profit appeared. (Inference from Whelan's decile table, not a stated rule.)
- A sound validation stack, assembled from the sources: (1) no-vig CLV vs sharp consensus close on every bet; (2) Brier/log loss of model probabilities vs no-vig closing probabilities on held-out seasons; (3) calibration plot by probability bucket; (4) P&L with bootstrap intervals, acknowledging it needs thousands of bets; (5) separate reporting by market (openers vs closers, segment) because CLV at soft books vs Pinnacle close is inflated (noted by [DeucesCracked](https://www.deucescracked.com/blog/closing-line-value-explained-sports-betting-sharp-test-2026), affiliate).

### Gaps
- I could not locate Pinnacle's primary CLV article or any primary Rufus Peabody/Unabated statement with a specific bets-needed figure; a claimed "100,000 NFL bets" Peabody study has no primary source — [PIPER blog](https://piper-seven.vercel.app/blog/what-is-clv-closing-line-value) (unverified).
- No source gave an NBA-specific conversion of spread half-points to win probability (needed to turn spread CLV into % edge); left unverified.
- No academic paper found that formally derives the CLV-to-ROI mapping for NBA; Whelan's is the only dataset-based treatment I found.

---

## Key Question 5: Economics: hold by market type, Kalshi fee drag, account limits vs exchange model, and documented ROI of syndicates/academic strategies

### Takeaway
Mainline sportsbook hold is ~4.5-5.5% (props ~8%); Kalshi's taker fee peaks at 1.75 cents per contract at 50 cents (3.5% of stake, requiring a 51.75% true probability vs 52.38% at -110) but a 1-cent spread or an early exit erases that advantage, and independent NFL checks found Kalshi 6-8% worse than books after fees. Documented ROIs range from Whelan's top-CLV-decile +11.4% (NBA, in-sample sort) to Ramesh et al.'s 9-13% NBA ROI (in-sample optimized); I found no primary-source syndicate ROI figures.

### Cited Findings

**Hold / vig**
- NBA moneyline margin ~4.5% across US books — [Karl Whelan](https://www.karlwhelan.com/?p=2595)
- Multi-league margins: head-to-head 4.77%, totals 5.45%, player props 7.93%; licensed US books 4.72% overall — [Low Juice Sportsbooks citing Mendeley 2026](https://lowjuicesportsbooks.com/prop-market-margins-in-mlb-and-nba/)
- Spread bets pay 100/110 (vig embedded; ~4.5% randomized baseline loss on spreads in Ramesh et al.) — [arXiv 1910.08858v2](https://arxiv.org/html/1910.08858v2)

**Kalshi fee structure (official page returned HTTP 429; figures below are from third-party summaries that agree on the formula)**
- Standard taker fee = round-up(0.07 × C × P × (1 - P)), C = contracts, P = price in dollars; peaks at 1.75 cents per contract at 50 cents; 0.63 cents at 10 or 90 cents; 1.47 cents at 30 or 70 cents. Maker fee on designated markets ≈ one quarter of taker (0.0175 coefficient). Fee schedule effective Feb 5, 2026 with a July 2026 revision keeping 0.07 — [search summary of multiple guides, e.g., OddsShopper](https://www.oddsshopper.com/articles/betting-101/kalshi-fees), [Ballislife](https://ballislife.com/prediction-markets/kalshi/fees/), [0xinsider](https://0xinsider.com/learn/kalshi-fees-explained)
- Prediction.com's guide (updated July 27, 2026) gives the same formula with rounding to the nearest cent but its table implies ~6% for sports (peak ~1.5% at 50 cents) and 7% for crypto; its own worked examples are internally inconsistent; it says live markets use the pre-game schedule, props sometimes use different multipliers, combos have their own schedule, and maker fees round to ~$0 for small orders — [Prediction.com](https://prediction.com/blog/kalshi-fees-complete-guide-2026)
- One source claims a $0.07-per-contract cap; this conflicts with the formula (peak ~$0.0175) and is likely wrong — [Tech-Insider](https://tech-insider.org/sports-prediction-markets/)
- Kalshi 2025 fee revenue: $263.5M, 89% from sports — [CDC Gaming / InGame](https://cdcgaming.com/brief/kalshis-2025-fee-revenue-was-263-5m-with-89-coming-from-sports/)
- Polymarket charged zero trading fees on NBA markets in Feb-Mar 2026 — [arXiv 2605.00864](https://arxiv.org/html/2605.00864v1)
- Independent price checks: Kalshi NFL pre-game odds 8% worse than FanDuel and 6% worse than DraftKings including fees (Citizens, Dec 2025) — [CDC Gaming](https://cdcgaming.com/brief/kalshi-pricing-remains-worse-than-sportsbooks-says-citizens/); ~7% worse moneylines, ~10% worse totals in early NFL 2025 — [Yogonet](https://www.yogonet.com/international/news/2025/09/09/115258-kalshi-trading-hits-441m-in-first-nfl-week-but-analysts-flag-pricing-gap-with-sportsbooks)

**Limits vs exchange model**
- Props limits as low as ~$100; NFL mainlines $100k+; winners get reduced limits; in-play strictly limited — [Covers](https://www.covers.com/industry/how-sportsbook-limits-work-behind-the-scenes-july-14-2025)
- Retail books limit aggressively; market-maker books (Pinnacle named) tolerate sharp action longer — [Bookmakers Review](https://www.bookmakersreview.com/betting/sharpest-sportsbooks/); [Covers](https://www.covers.com/industry/how-sportsbook-limits-work-behind-the-scenes-july-14-2025)
- Exchange capacity constraint (Polymarket NBA): in 76.9% of arb episodes a $100 budget could not be fully deployed; props ~$253 average volume per market — [arXiv 2605.00864](https://arxiv.org/html/2605.00864v1)
- Kalshi in-game underreaction is largest when liquidity is low — [arXiv 2606.07811](https://arxiv.org/abs/2606.07811)

**Documented ROI**
- Whelan NBA moneylines: top CLV decile +11.4% (close) / ~+16% (24h); average bettor loses ~4.5% — [Karl Whelan](https://www.karlwhelan.com/?p=2595)
- Ramesh et al.: NBA 9.04-11.18%, NCAAB 4.07-13.02%, WNBA 12.98-17.01%, NFL 9.26-16.57% optimal ROI; bootstrap 99% lower bounds NBA 1.77-3.41%, NCAAB 1.20-2.63%, WNBA 6.80-7.69%; thresholds optimized in-sample — [arXiv 1910.08858v2](https://arxiv.org/html/1910.08858v2)
- Week-1 NBA unders: +11.1% per game, 2009-2012 — [UBPLJ](https://www.ubplj.org/index.php/jpm/article/view/601)
- Early-season anti-totals strategy: 56.72% win rate — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1544612307000177)
- Polymarket NBA arb: $210 capped single-market profit and $527 capped combinatorial profit over 173 games — [arXiv 2605.00864](https://arxiv.org/html/2605.00864v1)
- A personal GitHub experiment (2013-14 to 2016-17) found model spread picks win about half the time and lose the vig — [GitHub](https://github.com/DeliciousPipe1326/nba-beat-the-close/blob/main/docs/experiments/other_markets/RESULTS.md) (anecdotal)

### Inferences (worked examples are my calculations using the 0.07 taker formula; re-check against the official Kalshi PDF)
- **Kalshi taker at 50 cents, held to settlement:** fee 1.75 cents; EV = p × $1 - $0.50 - $0.0175, so break-even p = 51.75% versus 52.38% at -110. Fee is 3.5% of stake but only 1.75% of notional, so the required edge over a fair 50% price is 1.75 points vs 2.38 points at -110.
- **But crossing a 1-cent spread removes the advantage:** if fair is 50 and you lift 51, break-even = 0.51 + 0.07 × 0.51 × 0.49 = 52.75%, worse than -110. Kalshi only beats -110 if you can get filled at or within about half a cent of fair.
- **Maker at 50 cents** (0.0175 coefficient, where offered): fee ≈ 0.44 cents, break-even ≈ 50.44%; this is the cheapest venue in the comparison, with the caveat that resting orders get adversely selected.
- **Round-trip in-game trading** (enter and exit as taker near 50 cents): about 1.75 + 1.75 = 3.5 cents, i.e., 7% of a 50-cent stake, so a live strategy exploiting the 0.64 underreaction must capture more than ~3.5 cents of drift per contract, net of spread, to profit; the paper's drift is "over the following several minutes" with magnitude not stated in the abstract.
- **Favorite at 70 cents:** fee 1.47 cents, break-even 71.47% (1.47 points of edge). **Longshot at 10 cents:** fee 0.63 cents, break-even 10.63%, which is 6.3% of stake; per dollar risked, Kalshi fees are heaviest on longshots and lightest on heavy favorites.
- **Per-dollar comparison summary:** required edge in probability points over fair: Kalshi maker ~0.4; Kalshi taker at fair ~1.75; -110 book ~2.4; -120/-120 prop ~4.5 (my calculation: 120/220 = 54.5%); Polymarket at zero fee = spread only.
- **Why Kalshi can still be worse:** the Citizens and Yogonet checks show Kalshi *prices* (not just fees) were 6-10% worse than books on NFL, meaning the fair price is not reliably available; the structural advantage of the exchange is capacity without being limited, not better prices.
- **Economics of a documented strategy at scale:** at Covers' ~$100 prop limits, even a 10% ROI prop edge yields ~$10 per bet; at Polymarket's $253 average prop volume, you are the market. Mainline NBA sides at books carry high limits but the required edge (2.4 pts) exceeds what any peer-reviewed closing-line study has found.

### Gaps
- The official Kalshi fee schedule PDF could not be fetched (HTTP 429); the 0.07 vs ~0.06 sports multiplier discrepancy between third-party guides is unresolved, as is which sports markets currently carry maker fees.
- No primary-source documentation of syndicate ROI (ESPN/Bloomberg/Ringer reporting on specific groups did not surface in my searches); all ROI figures above are academic or individual.
- No NBA-specific Kalshi-vs-sportsbook price gap measurement; the 6-10% figures are NFL.
- No source quantified in-play sportsbook limits for NBA, nor the typical limit trajectory (initial vs post-flag) for NBA mainlines at US retail books.
