# Legal, practical sources of NCAA D-I men's basketball broadcast video and video-derived datasets (as of 2 Nov 2026)

Scope note: findings below were verified on 2026-11-02 by fetching primary pages, the Internet Archive metadata/search APIs, the Hugging Face API, and YouTube's oEmbed endpoint where possible. "Verified live" means the URL returned content today. Nothing here is legal advice; where a page's terms are quoted, the quote is what the operator's contract says, which is distinct from what copyright law (e.g., fair use) might ultimately allow.

## Summary table of sources (quick reference)

### Takeaway
There is no openly licensed, large, annotated NCAA-broadcast dataset. The two usable public research sets are SportsMOT (15 basketball train clips, mixed NCAA/NBA/Olympic, CC BY-NC 4.0, directly downloadable) and a 2.6 GB course subset of the Stanford/Google NCAA dataset (full release not found). Raw full-game NCAA broadcast video is legitimately *viewable* at 1080p on the official March Madness and ACC Digital Network YouTube channels, but every streaming service's contract (YouTube, Disney/ESPN+, Peacock, BTN) forbids downloading/recording/scraping and, except YouTube, explicitly forbids ML training.

### Cited Findings
| Source | Type | Size | Res/fps | Annotations | License / terms | Broadcast view? | Obtain |
|---|---|---|---|---|---|---|---|
| Stanford/Google NCAA (Ramanathan et al. CVPR'16) | event detection + player boxes/tracks | 257 games, 14,548 clips x 20 frames | not stated (YouTube source) | 11 event classes, per-frame player boxes + track ids | none stated; official release not found | yes (YouTube broadcasts) | Only a 2.6 GB Toronto course subset is live — [course PDF](https://www.cs.toronto.edu/~urtasun/courses/CSC2541_Winter17/project_1.pdf), [data.7z](https://www.cs.toronto.edu/~namdar/sports_course/ncaa/data.7z) |
| SportsMOT (ICCV'23) | MOT | 240 clips (80 basketball: 15/15/50 train/val/test) | 1280x720 @ 25 fps | boxes + track ids (MOT17 format); test GT withheld | CC BY-NC 4.0; no redistribution | yes ("official recording" YouTube broadcasts) | [HF MCG-NJU/SportsMOT](https://huggingface.co/datasets/MCG-NJU/SportsMOT), [GitHub](https://github.com/MCG-NJU/SportsMOT) |
| TrackID3x3 (MMSports'25) | MOT + ID + pose | 3x3 basketball | fixed/drone cams | boxes, ids, pose | see repo | no | [arXiv 2503.18282](https://arxiv.org/abs/2503.18282v2) |
| TeamTrack | MOT | 279.9K frames / 4.37M boxes, soccer/basketball/handball | fisheye/drone full-pitch | boxes + ids | see paper | no | [review](https://www.themoonlight.io/review/teamtrack-a-dataset-for-multi-sport-multi-object-tracking-in-full-pitch-videos) |
| BasketEvent (arXiv Jul 2026) | PBP-aligned events | NBA, 1,000 manually timed samples | n/a | player-grounded events | "will be made publicly available" | yes (NBA) | [arXiv 2607.21267](https://arxiv.org/abs/2607.21267) |
| March Madness YouTube channel | raw video | dozens of classic/recent full games | 1080p (per mirror metadata) | none | YouTube ToS (no download) | yes | [@MarchMadness](https://www.youtube.com/@MarchMadness) |
| ACC Digital Network YouTube | raw video | full-game replays, 2024-25 and 2025-26 | 1080p (per mirror metadata) | none | YouTube ToS | yes | [@ACCDigitalNetwork](https://www.youtube.com/@ACCDigitalNetwork) |
| Internet Archive "MCBK_*" items | raw video | e.g., 179 full games in one item; ~1-2 TB per item, many items | 1920x1080 | none | no license; user-uploaded copyrighted broadcasts | yes (Stadium network) | [example item](https://archive.org/details/MCBK_2018-19_201905) |
| Veritone (NCAA footage agent) | commercial licensing | NCAA championship archive | broadcast masters | none | paid license; no public purchase | yes | [Veritone/NCAA](https://unlock.veritone.com/ncaa_video_licensing) |

### Inferences
- For a tracking/calibration project, the realistic *permitted* path is: SportsMOT (train/val, non-commercial) + the Toronto subset for boxes/tracks, plus streaming official YouTube full games for qualitative testing without downloading, or asking Veritone/ACC/NCAA Digital for a research license.

### Gaps
- No source was found that confirms an NCAA-specific annotated dataset released after 2023.

## Q1. Stanford/Google "NCAA basketball" dataset (Ramanathan et al., CVPR 2016): still downloadable? contents? terms?

### Takeaway
The full 257-game dataset has no official download page that I could find (Google Research's publication page lists none), and a 2023 paper states it is no longer available; what *is* live is a 2.6 GB course subset hosted by the University of Toronto (~50 clips per action x 11 actions, PNG frames with per-frame player boxes and track ids), served without any stated license.

### Cited Findings
- The paper is "Detecting events and key actors in multi-person videos" (Ramanathan, Huang, Abu-El-Haija, Gorban, Murphy, Fei-Fei), CVPR 2016 — [arXiv 1511.02917](https://arxiv.org/pdf/1511.02917); [Stanford PDF](https://cs.stanford.edu/groups/vision/pdf/ramanathan2016cvpr.pdf); [Google Research page](https://research.google/pubs/pub45489/).
- Dataset composition per the paper: a subset of the 296 NCAA games available on YouTube, keeping the most recent 257 games; 14K event annotations across 11 event classes — [arXiv 1511.02917](https://arxiv.org/pdf/1511.02917).
- Google Research's publication page says the authors "collected a new basketball dataset comprising 257 basketball games with 14K event annotations" but provides no dataset name, download link, project page or release statement; its only "Download" link is to the arXiv paper — [Google Research pub45489](https://research.google/pubs/pub45489/) (fetched 2026-11-02).
- The Toronto CSC2541 (Winter 2017) course handout describes the data: "Video clips corresponding to 11 basketball actions (e.g., shoot, pass) are annotated across 257 NCAA games. In total there are 14548 clips annotated. Each clip corresponds to 20 frames of a 4 second segment of the game. Moreover, each frame is annotated with bounding boxes and tracking data for all the players." — [CSC2541 project_1.pdf](https://www.cs.toronto.edu/~urtasun/courses/CSC2541_Winter17/project_1.pdf).
- Course subset contents: "for each of the 11 actions, we have randomly chosen about 50 clips consisting of at most 20 consecutive frames"; layout is data/game-id/clip-id/NN.png with NN-info.csv giving (x, y, w, h) boxes plus player tracking ids, and data/events.pkl mapping actions to clips; game ids are YouTube video ids (example game name "0En5pOUZN5M") — [CSC2541 project_1.pdf](https://www.cs.toronto.edu/~urtasun/courses/CSC2541_Winter17/project_1.pdf).
- The subset archive is live: HTTP 200, Content-Length 2,771,555,302 bytes (~2.6 GiB), Last-Modified 15 Feb 2017 — [data.7z](https://www.cs.toronto.edu/~namdar/sports_course/ncaa/data.7z) (HEAD request 2026-11-02).
- A survey describes the same dataset as 257 untrimmed NCAA game videos (~1.5 h each), 14,548 segments, and says it "also provides 9,000 frames with bounding boxes of players" and lists 14 categories (e.g., 3-point success/fail, steal, slam dunk) — [arXiv 2206.01038 survey](https://arxiv.org/pdf/2206.01038); this conflicts with the paper's 11 event classes and the Toronto handout's "every frame annotated" claim.
- A 2023 arXiv benchmark paper surfaced in search states that the Ramanathan et al. dataset is "currently not available for download" (attribution from search snippet; the result list pointed to [arXiv 2305.13124](https://arxiv.org/pdf/2305.13124)) — treat as secondary.
- An "NCAA+" variant extending each clip with pre/post-event frames is described (250 games, ~1.5 h average) — [arXiv 1903.06879](https://arxiv.org/pdf/1903.06879); no download found.

### Inferences
- The 9,000-frame figure likely refers to frames with *manually* drawn boxes; the rest of the per-frame boxes/ids in the Toronto subset were presumably produced by the authors' tracker (the handout calls it "tracking data"). Expect tracker-quality, not human-quality, boxes.
- Because the source games are YouTube broadcasts and the paper's pipeline worked on broadcast frames, the footage is the standard sideline broadcast view; the Toronto PNGs' pixel resolution is not documented.
- The Toronto host states no license; the underlying frames are NCAA/broadcaster copyright. Using it for internal, non-commercial research mirrors how the field has treated it, but there is no written permission.

### Gaps
- No official Stanford/Google hosting (or mirrors on Kaggle/HF/datasetninja) was found for the full 257-game set; the Toronto 2.6 GB subset is the only live copy located.
- Frame resolution of the Toronto PNGs and whether clips are 720p were not verified (would require downloading the 2.6 GB archive).
- The discrepancy between 11 vs 14 classes and "every frame boxed" vs "9,000 boxed frames" was not resolved from primary text (the paper PDF text was not fully parsed).

## Q2. SportsMOT and SportsMOT-like datasets: NCAA clip counts per split, resolution, train availability, successors

### Takeaway
SportsMOT has exactly 80 basketball clips split 15 train / 15 val / 50 test (computed from the official split files), all 1280x720 @ 25 fps, with train and val ground truth freely downloadable from the official Hugging Face repo under CC BY-NC 4.0; basketball footage is a mix of NCAA, NBA and Olympic broadcasts with no per-clip league label. No "SportsMOT-X" or "BasketTrack" exists in any source found; the 2025 successors (TeamTrack, TrackID3x3) are not broadcast footage.

### Cited Findings
- Official description: "We provide 240 sports video clips of 3 categories (i.e., basketball, football and volleyball)"; source videos filtered to "720P resolution, 25 FPS, and official recording"; clips "cut into clips of average 485 frames manually"; sample seqinfo.ini lists imWidth=1280, imHeight=720, frameRate=25 — [GitHub MCG-NJU/SportsMOT](https://github.com/MCG-NJU/SportsMOT).
- Footage origin (GitHub): "collected from Olymplic Games, NCAA Championship, and NBA on YouTube" — [GitHub MCG-NJU/SportsMOT](https://github.com/MCG-NJU/SportsMOT). The HF README phrases it as "videos of high-quality professional games including NCAA, Premier League, and Olympics from MultiSports" — [HF README](https://huggingface.co/datasets/MCG-NJU/SportsMOT/resolve/main/README.md).
- Per-sport statistics on GitHub: average frames per clip basketball 845.4, volleyball 360.4, football 673.9; boxes per frame 9.1 / 11.2 / 12.8 respectively (column alignment slightly uncertain in the scraped table) — [GitHub MCG-NJU/SportsMOT](https://github.com/MCG-NJU/SportsMOT).
- Split composition computed from official split files (splits_txt/basketball.txt, train.txt, val.txt, test.txt on HF, fetched 2026-11-02): 80 basketball names; train/basketball = 15, val/basketball = 15, test/basketball = 50 (same 15/15/50 for volleyball and football; totals 45/45/150) — [HF splits_txt](https://huggingface.co/datasets/MCG-NJU/SportsMOT/tree/main/splits_txt).
- Official HF repo status via API (2026-11-02): gated: False, private: False, 3,862 downloads, last modified 2024-07-21; files dataset/train.tar 6.70 GB, dataset/val.tar 6.59 GB, dataset/test.tar 23.21 GB plus conversion scripts — [HF API](https://huggingface.co/api/datasets/MCG-NJU/SportsMOT). The README front-matter still carries an "Acknowledge license to accept the repository" gating prompt — [HF README](https://huggingface.co/datasets/MCG-NJU/SportsMOT/resolve/main/README.md).
- License/terms: "SportsMOT is licensed under a Creative Commons Attribution-NonCommercial 4.0 International License"; users agree not to "distribute the SportsMOT dataset without prior written permission"; full data historically via Codalab competition ("Participate" / "Get Data"), with an example subset on OneDrive/Baidu — [GitHub MCG-NJU/SportsMOT](https://github.com/MCG-NJU/SportsMOT).
- Annotation format: MOT Challenge 17 layout (gt/, img1/ JPEG frames, seqinfo.ini); example GT line "1, 7, 749, 217, 34, 125, 1, 1, 1" — [GitHub MCG-NJU/SportsMOT](https://github.com/MCG-NJU/SportsMOT). Test ground truth is withheld for benchmark evaluation — [datasetninja](https://datasetninja.com/sports-mot).
- Third-party HF mirrors exist (community uploads, not official) — [Lekim89/sportsmot](https://huggingface.co/datasets/Lekim89/sportsmot), [Hellfire98/sportsmot](https://huggingface.co/datasets/Hellfire98/sportsmot).
- Successors: TrackID3x3 (arXiv 2503.18282, MMSports'25) is "the first publicly available comprehensive dataset specifically designed for multi-player tracking, player identification, and pose estimation in 3x3 basketball", with Indoor fixed-camera, Outdoor fixed-camera and Drone subsets; code/data at github.com/open-starlab/TrackID3x3 — [arXiv 2503.18282v2](https://arxiv.org/abs/2503.18282v2).
- TeamTrack: "over 279,900 frames and 4,374,900 annotated bounding boxes across soccer, basketball, and handball", full-pitch fisheye/drone cameras (not broadcast) — [TeamTrack review](https://www.themoonlight.io/review/teamtrack-a-dataset-for-multi-sport-multi-object-tracking-in-full-pitch-videos).
- DeepSportradar / DeepSport: ball 3D localization, camera calibration, instance segmentation, re-ID from professional (Belgian league) basketball, not NCAA — [arXiv 2208.08190](https://arxiv.org/pdf/2208.08190.pdf); [datasetninja deepsportradar](https://cdn.datasetninja.com/deep-sport-radar).
- Searches for "SportsMOT-X" and "BasketTrack" returned no matching dataset — [search via TrackID3x3 comparison table](https://arxiv.org/html/2503.18282v2).

### Inferences
- Only 15 basketball training clips (~15 x 845 frames ≈ 12.7K frames) means SportsMOT alone is thin for training an NCAA-specific detector/tracker; it is better used as a validation benchmark plus fine-tuning seed.
- Since basketball clips come from "NCAA Championship, NBA and Olympics", the NCAA share must be identified by inspecting clips (YouTube ids in sequence names like v_-6Os86HzwCs_c001 can be looked up).
- The HF official repo being ungated with train/val/test tars means the train split is directly obtainable without Codalab as of today.

### Gaps
- Exact number of NCAA-origin basketball clips (vs NBA/Olympic) is not documented anywhere found.
- Whether test.tar on HF includes ground truth was not checked (likely images only).
- MultiSports' own basketball league composition was not verified.

## Q3. NCAA play-by-play-aligned video datasets, Kaggle datasets with college video, university research releases

### Takeaway
Several papers built PBP-aligned NCAA broadcast corpora (25 full 2015 NCAA games at 720p; Hudl-sourced NCAA/WNCAA frames; a CVPR 2020 college calibration set), but none was found publicly released; Kaggle hosts no NCAA game-video dataset I could locate, and the only PBP-aligned broadcast release in progress (BasketEvent, July 2026) is NBA, with data "to be made publicly available."

### Cited Findings
- "Leveraging Contextual Cues for Generating Basketball Highlights" (2016): 25 full NCAA broadcast games from 2015 YouTube uploads, ~35 hours of 720p video, with play-by-play aligned to video via OCR; the paper says the dataset will be made public — [arXiv 1606.08955](https://arxiv.org/pdf/1606.08955). No release link was found.
- "A Simple and Effective Temporal Grounding Pipeline for Basketball Broadcast Footage" (2024): aligns PBP to broadcast frames using training frames from NBA, WNBA, Euroleague, NCAA, WNCAA and US high-school leagues (Hudl-sourced); describes a method, not a released corpus — [arXiv 2411.00862](https://arxiv.org/html/2411.00862v1).
- BasketEvent (submitted 23 Jul 2026, arXiv:2607.21267v1): "curated from real NBA broadcasts", pairs broadcast clips with official PBP, 1,000 manually annotated samples with precise event intervals; "Our data, code, and models will be made publicly available"; no URL yet; article badge CC BY 4.0 but dataset license unstated — [arXiv 2607.21267](https://arxiv.org/abs/2607.21267).
- Sha et al., "End-to-End Camera Calibration for Broadcast Videos" (CVPR 2020) evaluated on "a new college basketball dataset" — [CVPR 2020 PDF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Sha_End-to-End_Camera_Calibration_for_Broadcast_Videos_CVPR_2020_paper.pdf); no public release found.
- Stats Perform describes using "enormous amounts of college basketball tracking data" (proprietary) — [Stats Perform](https://www.statsperform.com/resource/predicting-nba-talent-from-enormous-amounts-of-college-basketball-tracking-data).
- Hugging Face small commercial sets: [TrainingDataPro/basketball_tracking](https://huggingface.co/datasets/TrainingDataPro/basketball_tracking) and [UniqueData/basketball_tracking](https://huggingface.co/datasets/UniqueData/basketball_tracking) (ball-box screenshots; not player tracking, league unspecified).
- Kaggle: searches for college/NCAA basketball game video on Kaggle returned no dataset listings (only papers) — [search results](https://arxiv.org/pdf/2206.01038).
- NCAA Division III launched a streaming initiative with NCAA Digital (TNT Sports) and Hudl for 200+ free games on ncaa.com for 2024 fall championships (streams, not datasets) — [NCAA news](https://ncaanewsarchive.s3.amazonaws.com/sa/2024-11-04_media-center-division-iii-announces-new-livestreaming-initiative-for-2024-fall-championships.html).

### Inferences
- Synergy/Hudl/ShotTracker college video+PBP is commercially licensed to schools; no academic release surfaced, so a researcher would need a direct agreement.
- The most practical PBP alignment route is to build it yourself: official PBP from public box-score sites + an official YouTube full game viewed in-browser, or BasketEvent's forthcoming NBA tooling adapted to college.

### Gaps
- Could not confirm whether the 25-game 2016 NCAA highlights dataset was ever released (no project page found).
- Did not search Kaggle's internal search directly (web search only); a dataset could exist under a non-obvious name.
- No university research group (CMU/UT/Stanford) NCAA video release was found beyond the Stanford/Google set.

## Q4. Official channels, archives and Creative Commons footage: where full/extended NCAA broadcast video is legitimately accessible

### Takeaway
Full games are officially published on YouTube by the NCAA's March Madness channel (classic and recent tournament games) and by the ACC Digital Network (regular-season full-game replays, 2024-25 and 2025-26) at 1080p; the Internet Archive holds thousands of hours of user-uploaded NCAA broadcasts (e.g., 179 Stadium-network games in one item) but with no license and no rights-holder involvement, and I found no Creative Commons-licensed D-I/D-II/D-III full-game streams.

### Cited Findings
- NCAA Digital (Turner Sports) launched the dedicated March Madness YouTube channel in March 2015 — [NCAA.com announcement](https://www.ncaa.com/news/basketball-men/article/2015-03-02/ncaa-and-turner-sports-announce-march-madness-youtube-channel); YouTube's blog on the partnership — [blog.youtube](https://blog.youtube/news-and-events/get-more-into-ncaa-march-madness-on/).
- Verified live official full games (YouTube oEmbed, 2026-11-02): "Duke vs. Kansas: 1991 National Championship — Coach K's first title | FULL GAME" by March Madness — [youtube.com/watch?v=Y3zEz5-1fXY](https://www.youtube.com/watch?v=Y3zEz5-1fXY); "North Carolina vs. Houston: 1982 Final Four | FULL GAME" by March Madness — [youtube.com/watch?v=IeKjDIn4NA4](https://www.youtube.com/watch?v=IeKjDIn4NA4); channel [@MarchMadness](https://www.youtube.com/@MarchMadness).
- Verified live: "North Carolina vs. Duke Full Game Replay | 2024-25 ACC Men's Basketball" by ACC Digital Network — [youtube.com/watch?v=DWnd-iDnA1M](https://www.youtube.com/watch?v=DWnd-iDnA1M); channel [@ACCDigitalNetwork](https://www.youtube.com/@ACCDigitalNetwork). Third-party listings show many more ACC full-game replays for 2024-25 and 2025-26 (e.g., Citadel vs Boston College ~76 min; Boston College vs Louisville 2025-26) — [spotsports listing](https://spotsports.com/videos/271b275968), [spotsports 2025-26](https://spotsports.com/videos/244d27de29); a fan forum says replays appear ~2 days after games — [Bear Insider](https://bearinsider.com/forums/3/topics/122427).
- Internet Archive mirror metadata for those YouTube videos reports 1920x1080 files of 4,300-4,780 s duration (i.e., the YouTube uploads are 1080p, 72-80 min) — [IA youtube-Y3zEz5-1fXY](https://archive.org/details/youtube-Y3zEz5-1fXY), [IA youtube-DWnd-iDnA1M](https://archive.org/details/youtube-DWnd-iDnA1M). These IA items were uploaded by a private individual (collection "mirrortube"), carry an uploader-applied "CC BY 4.0" tag, and are not rights-holder releases — [IA metadata](https://archive.org/metadata/youtube-DWnd-iDnA1M).
- Internet Archive bulk uploads: an advanced search for identifier:MCBK* returned 72 items; e.g., MCBK_2018-19 (subject "NCAA Basketball", collection television_inbox) contains 179 MP4 full games at 1920x1080 from the Stadium network (filenames like ncaam.2018.12.22.ut.martin.at.fresno.st-stadium.mp4), 1.27 TB total; sibling items MCBK_2016-17_x, MCBK_2018-2019_x, College2017-18 are 0.4-2.0 TB each; none lists a license — [IA advancedsearch](https://archive.org/advancedsearch.php?q=identifier%3AMCBK*&fl%5B%5D=identifier&fl%5B%5D=title&fl%5B%5D=item_size&rows=50&output=json), [IA MCBK_2018-19_201905](https://archive.org/details/MCBK_2018-19_201905).
- A broad IA search ("college basketball" OR "NCAA basketball" OR "NCAA tournament", mediatype movies) returns 3,278 items, mostly unrelated TV news; a VHS-to-DVD transfer of the 1996 Final Four semifinal exists at only 320x240 — [IA dvd-transfer-30](https://archive.org/details/dvd-transfer-30).
- NCAA championship footage licensing: Veritone became "global archive of record and exclusive video licensing agent for the NCAA" (Aug 2024) — [SVG News](https://www.sportsvideo.org/2024/08/13/veritone-named-global-archive-of-record-and-exclusive-video-licensing-agent-for-the-ncaa); the NCAA's guide says there is "NOT an option available for the general public/fans to purchase NCAA championships footage/DVDs", directing requests to Veritone (sports@veritone.com) — [NCAA Quick Reference Guide](https://www.ncaa.com/_flysystem/public-s3/files/Quick%20Reference%20Guide.pdf); licensing portal — [Veritone](https://unlock.veritone.com/ncaa_video_licensing).
- University libraries: Notre Dame's archive catalogs full ESPN Classic game broadcasts (e.g., ND vs Cincinnati 2010) as "Staff Only" items requested through the library — [ND ArchivesSpace](https://archivesspace.library.nd.edu/repositories/2/archival_objects/1805242); Marquette lists early-2000s broadcasts on VHS/local-network MP4 — [Marquette archives](https://www.marquette.edu/library/archives/SuperA/univ-a-4-5s17-2000-2010.php); a WKU fan archive indexes full games on YouTube/Vimeo — [WKU archive (Medium)](https://medium.com/the-red-towel/wku-video-archives-full-game-mens-basketball-library-e19d1eb9bf4a).
- YouTube Creative Commons rule: creators may apply CC BY only to original content; "If there's a Content ID claim on your video, you cannot mark your video with the Creative Commons license" — [YouTube Help: License types](https://support.google.com/youtube/answer/2797468?hl=en). No CC-licensed full college games were found in searches — [search results](https://ncaanewsarchive.s3.amazonaws.com/sa/2024-11-04_media-center-division-iii-announces-new-livestreaming-initiative-for-2024-fall-championships.html).
- D-III championship streaming: 200+ games free on ncaa.com via Hudl/TNT Sports (fall 2024) — [NCAA news](https://ncaanewsarchive.s3.amazonaws.com/sa/2024-11-04_media-center-division-iii-announces-new-livestreaming-initiative-for-2024-fall-championships.html).

### Inferences
- The ACC Digital Network full replays are the best legitimate source of *current-season, standard broadcast-view* D-I footage that can at least be streamed and frame-analyzed in-browser; March Madness classics skew to pre-2010 production (4:3, SD upscales) except recent tournament replays.
- The IA "MCBK" collections are technically the largest NCAA full-game trove online (likely >10 TB at 1080p), but they are unauthorized recordings of Stadium/other broadcasts; IA hosting confers no license, and the "CC BY 4.0" tag on mirrortube items is an uploader setting, not the rights holder's license.
- Because YouTube blocks CC on Content-ID-claimed videos, conference/school broadcast rights effectively preclude CC licensing of real game streams; the realistic "permission" route is a written research license from a conference (ACC/ESPN), a school's athletic department (which owns its own in-house stream productions), or Veritone/NCAA.

### Gaps
- Could not fetch the March Madness or ACC Digital Network channel listings directly (JS-rendered), so the total count of full games and their native frame rates are unverified.
- Did not confirm whether any school's in-house D-II/D-III YouTube streams carry a CC license (YouTube's license filter could not be queried server-side).
- The 506sports lower-division broadcast index was behind a bot challenge and could not be read.

## Q5. Terms governing recording/downloading from streaming services (ESPN+, Peacock, BTN/B1G+, SEC Network, YouTube, NCAA.com)

### Takeaway
Every relevant service contract prohibits copying, recording, downloading outside the app, DRM circumvention and automated scraping, and Disney/ESPN+, Peacock and BTN additionally prohibit using content to train AI/ML models; YouTube's terms bar downloading and automated access and separately bar collecting information that identifies a person (including faces). Recording is therefore "technically possible" but a contract violation on all of these platforms, independent of any copyright analysis.

### Cited Findings
- Disney+, ESPN, and Hulu Subscriber Agreement (Updated Oct 17, 2025), which covers "ESPN+ and other digital services within the ESPN Service": Sec. 1(a) use is "for their personal, noncommercial use only"; Sec. 3(b)(ii) you may not "copy the Content (except as expressly permitted by this Agreement)"; Sec. 4(g) "you may not record, copy, store, or redistribute any Content"; Sec. 3(b)(i) no "circumvent or disable any content protection system or digital rights management technology"; Sec. 3(c)(viii) no "access, monitor, copy, or extract the Services using a robot, spider, script or other automated means"; Sec. 3(b)(v) bars "fine-tuning, training, testing, benchmarking or validation of any artificial intelligence or machine learning tool" — [disneyplus.com subscriber agreement](https://www.disneyplus.com/welcome/subscriber-agreement). A third-party tracker reports a Feb 5, 2026 revision — [ConductAtlas history](https://conductatlas.com/platform/disney/disney-subscriber-agreement/history/).
- Disney Terms of Use (Last Updated May 24, 2024), the umbrella terms for Disney/ESPN products: Sec. 2.B prohibits "copy the Disney Product (except as expressly permitted by us)", "rebroadcast, transmit or perform", "circumvent or disable any content protection system or digital rights management technology", and "access, monitor, copy or extract the Disney Products using a robot, spider, script, or other automated means" including "for the purposes of creating or developing any AI Tool, data mining or web scraping"; Sec. 2.A license excludes "training, testing" of AI — [disneytermsofuse.com](https://disneytermsofuse.com/english/).
- Peacock Terms of Service (last updated Dec 3, 2025): Sec. 5.1 prohibits "copy, download, stream, capture, reproduce, duplicate, archive, distribute" content "unless expressly permitted by us in writing"; bars "robots, spiders, crawlers, devices, scripts, or other data gathering or extraction software" used "to access, acquire, copy, monitor, scrape, data mine, or aggregate any Content"; bars DRM circumvention; and bars use of "any Content for the purpose of directly or indirectly training, developing, or improving ... any artificial intelligence tool, model, system, or platform"; Sec. 4.4 in-app downloads are plan-limited and expire on cancellation — [peacocktv.com/terms](https://www.peacocktv.com/terms).
- BTN Terms of Use (Last Revised Dec 5, 2025; "The BTN Services are provided by Big Ten Network, LLC"), linked from the B1G+ site: Sec. 4 "Except as expressly permitted in these Terms of Use, you may not copy, download, stream, scrape, capture, reproduce ..." and "you may not and shall not copy, data mine, scrape or in any way extract or use the Content ... for the purpose of training any artificial intelligence algorithm, system, model or tool"; Sec. 16 bars "scripts, spiders, offline readers or bots" — [btn.com terms](https://btn.com/about/btn-terms-of-use/); B1G+ site links to these terms — [bigtenplus.com/terms](https://www.bigtenplus.com/terms).
- YouTube Terms of Service (Effective Dec 15, 2023): "You may view or listen to Content for your personal, non-commercial use"; you may not "access, reproduce, download, distribute, transmit, broadcast, display, sell, license, alter, modify" the Service or Content except "(a) as expressly authorized by the Service; or (b) with prior written permission from YouTube and, if applicable, the respective rights holders"; may not circumvent features that "prevent or restrict the copying or other use of Content"; may not "access the Service using any automated means (such as robots, botnets or scrapers)" except search engines per robots.txt or with permission; may not collect "any information that might identify a person (for example, usernames or faces)" — [youtube.com/t/terms](https://www.youtube.com/t/terms). Older (2016-17) versions phrased it as no download "unless you see a 'download' or similar link" — [Wayback 2017](https://web-wp.archive.org/web/20170811164130/https:/www.youtube.com/t/terms).
- ESPN Unlimited bundle terms point back to the same Disney/ESPN/Hulu Subscriber Agreement and restrict content to "personal and non-commercial use only" — [plus.espn.com FOX One additional terms](https://plus.espn.com/foxoneadditionalterms).
- An older ESPN (ESPNMotion) EULA: "Copying or reproduction of the content or software to any other server or location for further reproduction or redistribution is expressly prohibited" — [espn.com EULA](https://www.espn.com/motion/eula_plaintext.html).

### Inferences
- SEC Network and ACC Network are ESPN-operated linear/streaming channels, so viewing them via ESPN+/ESPN app falls under the Disney/ESPN/Hulu Subscriber Agreement quoted above (inference: the agreement defines the "ESPN Service" broadly; I did not fetch a SEC Network-specific terms page).
- The AI/ML-training clauses in the Disney, Peacock and BTN contracts mean that even a privately made screen recording used to train a tracker is a direct breach, not merely a grey area; YouTube's terms do not name AI training but its download, automation and face-collection clauses cover the same activity.
- "Technically possible vs permitted": screen capture or yt-dlp of these services is trivially possible; none of the contracts permits it, and all reserve termination rights. Fair use is a defense in a copyright claim, not a license, and does not override the contractual prohibitions a subscriber accepted.

### Gaps
- NCAA.com / March Madness Live terms of use could not be fetched (page returned empty); the operator (NCAA Digital, managed by TNT Sports / Warner Bros. Discovery) and its exact clauses are unverified.
- FloHoops, ESPN+'s school-produced "ESPN+ Extra"/ACCNX streams, and conference-run platforms (e.g., Mountain West/Stadium) were not individually checked.
- The last ~15% of the Disney subscriber agreement and BTN terms pages were not read (truncated at 100K chars).

## Q6. Practical: resolution/frame rate of these sources and whether they show the standard sideline broadcast camera

### Takeaway
Official YouTube full games (March Madness, ACC Digital Network) are 1080p standard broadcast productions (sideline main camera with cut-ins); SportsMOT is 720p/25 fps broadcast cuts; the Internet Archive "MCBK" Stadium recordings are 1080p broadcast; the Stanford/Toronto subset is 20-frame broadcast clips of unknown pixel size; TeamTrack/TrackID3x3 are fixed/fisheye/drone, not broadcast.

### Cited Findings
- March Madness and ACC Digital Network YouTube full games: mirror file metadata 1920x1080, durations 4,300-4,780 s (72-80 min) — [IA youtube-DWnd-iDnA1M](https://archive.org/metadata/youtube-DWnd-iDnA1M), [IA youtube-Y3zEz5-1fXY](https://archive.org/metadata/youtube-Y3zEz5-1fXY). The 1982 Final Four game is a broadcast-era recording upscaled to 1080p container — [IA youtube-IeKjDIn4NA4](https://archive.org/metadata/youtube-IeKjDIn4NA4).
- SportsMOT: 1280x720, 25 fps, "official recording" broadcast search results; average basketball clip 845 frames (~34 s) — [GitHub MCG-NJU/SportsMOT](https://github.com/MCG-NJU/SportsMOT).
- IA MCBK_2018-19 Stadium recordings: 1920x1080 MP4, ~6,700-7,200 s per game (full broadcasts incl. breaks) — [IA metadata](https://archive.org/metadata/MCBK_2018-19_201905).
- The 2016 NCAA highlights corpus was 720p YouTube broadcasts — [arXiv 1606.08955](https://arxiv.org/pdf/1606.08955).
- Stanford/Toronto subset: 20 frames per 4-second clip (i.e., 5 fps sampling), PNG frames; no pixel size stated — [CSC2541 project_1.pdf](https://www.cs.toronto.edu/~urtasun/courses/CSC2541_Winter17/project_1.pdf).
- TrackID3x3 camera setups are indoor fixed, outdoor fixed and drone; TeamTrack is fisheye/drone — [arXiv 2503.18282v2](https://arxiv.org/html/2503.18282v2).

### Inferences
- The Stanford subset's 5 fps sampling makes it unsuitable for training a high-frame-rate tracker/motion model but fine for detector pretraining and event classification.
- For court-calibration work, only broadcast-view sources (YouTube official, SportsMOT, Stanford subset) apply; none ships camera parameters, so homographies must be annotated or estimated (Sha et al. 2020 did this on an unreleased college set).

### Gaps
- Native frame rate (30 vs 60 fps) of the ACC Digital Network and March Madness YouTube uploads was not verified (oEmbed and IA metadata report resolution and duration only).
- The Toronto subset's image resolution was not checked.
