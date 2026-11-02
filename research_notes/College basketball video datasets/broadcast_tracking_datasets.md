# Public basketball player-tracking, detection, pose, jersey and video+positional datasets from broadcast-style video (as of 2026-11-02)

Scope note: inventory of datasets usable for player detection, MOT, re-ID, jersey-number recognition and court-plane localisation from TV-style basketball footage, plus datasets pairing video with official positional data. "Verified" below means read on the source page or in the paper text during this session; items marked UNVERIFIED were only seen in search snippets or secondary pages. Roboflow Universe project pages returned HTTP 403 to the fetcher, and the Kaggle dataset page rendered empty, so figures for those come from secondary sources.

## Key question 1: SportsMOT (composition, splits, sources) and other basketball MOT sets

### Takeaway
SportsMOT is still the only sizeable public MOT benchmark with broadcast-view 5v5 basketball; its basketball clips come from NCAA tournament footage (via MultiSports), not the NBA, and it is CC BY-NC 4.0 with no redistribution. Newer basketball MOT sets (TeamTrack, TrackID3x3, the 3x3 pose-MOT set) are fixed-camera/fisheye/drone, not broadcast, and no 2025-2026 broadcast 5v5 basketball MOT successor was found.

### Cited Findings

**SportsMOT (ICCV 2023, MCG-NJU)** — https://github.com/MCG-NJU/SportsMOT ; paper https://arxiv.org/abs/2304.05170
- 240 clips over basketball, volleyball, football; 150,379 frames, 6,015 s, 1,629,490 boxes, 3,401 tracks; split 45 train / 45 val / 150 test sequences (split is for the whole dataset; the paper does not give per-sport clip counts) — [SportsMOT paper v2, Table 1 and Sec. 3.2](https://arxiv.org/html/2304.05170v2)
- Per-sport averages (Table 2): basketball 845.4 frames/clip, 10 tracks/clip, track gap length 68.7, track length 767.9, 9.1 boxes/frame; volleyball 360.4 / 12 / 38.2 / 335.9 / 11.2; football 673.9 / 20.5 / 116.1 / 422.1 / 12.8 — [SportsMOT paper v2](https://arxiv.org/html/2304.05170v2)
- Source footage: "high-quality professional games including NCAA, Premier League, and Olympics from MultiSports"; "Only the overhead shots of sports game scenes are used" (i.e., the main broadcast camera), 720p, 25 fps — [SportsMOT paper v2 Sec. 3.1](https://arxiv.org/html/2304.05170v2). The GitHub README lists sources as "Olymplic Games, NCAA Championship, and NBA" on YouTube, 720P/25FPS official recordings — [SportsMOT GitHub](https://github.com/MCG-NJU/SportsMOT). The paper text itself does not mention the NBA; the README does. Treat the basketball portion as NCAA (flag: college footage), with possible NBA clips per the README.
- Annotations: only players on the court (no referees, spectators, coaches; ball not boxed); boxes cover whole limbs/torso; occluded players still boxed while part of body visible, skipped if half the torso is out of view; KCF-propagated then hand-refined; boxes with w<5 or h<5 removed; MOT17 format; seqinfo 1280x720 25 fps — [SportsMOT paper v2](https://arxiv.org/html/2304.05170v2); [SportsMOT GitHub](https://github.com/MCG-NJU/SportsMOT)
- No team labels, jersey numbers, pose, ball or court calibration are provided (format is MOT17 gt.txt only) — [SportsMOT GitHub](https://github.com/MCG-NJU/SportsMOT)
- License (exact): "SportsMOT is licensed under a Creative Commons Attribution-NonCommercial 4.0 International License." plus "You agree not to distribute the SportsMOT dataset without prior written permission." — [SportsMOT GitHub](https://github.com/MCG-NJU/SportsMOT)
- Access: register on Codalab and download from the competition "Participate/Get Data" section; an example subset is on OneDrive/Baidu. Test GT withheld (server evaluation) — [SportsMOT GitHub](https://github.com/MCG-NJU/SportsMOT)
- Hugging Face mirror (unofficial): Hellfire98/sportsmot, 36.1 GB, train 45 + val 45 with GT, test 150 without GT, states CC BY-NC 4.0 — [HF mirror](https://huggingface.co/datasets/Hellfire98/sportsmot). Note the official terms forbid redistribution without permission, so this mirror's legitimacy is questionable.
- Known issues: paper reports basketball is the hardest category (HOTA 66.17 for their baseline); 14 open GitHub issues (contents not read) — [SportsMOT paper v2](https://arxiv.org/html/2304.05170v2); [SportsMOT GitHub](https://github.com/MCG-NJU/SportsMOT)

**MultiSports (the parent of SportsMOT's clips)** — referenced as the source of SportsMOT video; it is a spatio-temporal action localisation dataset ("NCAA, Premier League, and Olympics") — [SportsMOT paper v2](https://arxiv.org/html/2304.05170v2). Not independently fetched this session (see Gaps).

**TeamTrack (2024, Scott et al.)** — https://atomscott.github.io/TeamTrack/ ; paper https://arxiv.org/abs/2404.13868
- Full-pitch/court video for soccer, basketball, handball. Basketball subset: 1,296,900 boxes total (346,500 fisheye; 475,200 side-view drone; 475,200 top-view drone); 65.5 min total (17.5 fisheye, 24 + 24 drone); fisheye Z CAM E2-F6 6K indoor at "B University", drone DJI Mavic 3 4K outdoor at "C University" — [TeamTrack project page](https://atomscott.github.io/TeamTrack/)
- NOT broadcast: fisheye and drone cameras, university-level play. Fps, track counts, court coordinates and dataset license not stated on the page (only the website is CC BY-SA 4.0). Download via Kaggle / Google Drive links on the page — [TeamTrack project page](https://atomscott.github.io/TeamTrack/)

**TrackID3x3 (MMSports 2025, Nagoya Univ. / open-starlab)** — https://github.com/open-starlab/TrackID3x3 ; paper https://arxiv.org/abs/2503.18282
- 3x3 basketball; subsets Indoor (fixed consumer camera, 7,531 frames), Outdoor (iPhone 13 fixed, 143,276 frames), Drone (DJI Air 2S, 127,119 frames); v1 of the paper listed 155,797 frames total, v2 table totals 277,926 — [arXiv 2503.18282 via search summary](https://arxiv.org/abs/2503.18282v2) (frame numbers UNVERIFIED against full paper text)
- Annotations: boxes for the 6 on-court players on all frames; 10 pose keypoints on some frames; track IDs + identity (identity mainly from jersey numbers); jersey numbers Outdoor only via jersey-number-pipeline; court localisation is the only manual step; player attributes/colour histograms — [TrackID3x3 GitHub](https://github.com/open-starlab/TrackID3x3)
- License (exact): dataset "CC BY 4.0, unless otherwise stated"; code Apache-2.0 (own), MIT (BoT-SORT), Apache-2.0 (CAMELTrack), CC BY-NC 3.0 (jersey-number-pipeline) — [TrackID3x3 GitHub](https://github.com/open-starlab/TrackID3x3)
- Access: Google Drive folder, no request form. Not broadcast; not NCAA — [TrackID3x3 GitHub](https://github.com/open-starlab/TrackID3x3)

**3x3 fixed-camera pose/MOT set (arXiv 2412.06258, "Enhanced MOT Using Pose-based Virtual Markers in 3x3 Basketball")** — 42 videos, 7,531 frames, 6 keypoints per player (head, elbows, ankles, hip centre); fixed camera; not broadcast — [arXiv 2412.06258 via search](https://arxiv.org/pdf/2412.06258). Likely the same footage as TrackID3x3 Indoor (identical 7,531 frame count) — inference.

**NCAA basketball event/actor dataset (Ramanathan et al., CVPR 2016, "Detecting events and key actors in multi-person videos")** — https://arxiv.org/abs/1511.02917
- Built from 296 NCAA games on YouTube, most recent 257 used; 11 event classes; boxes for all players on a 9,000-frame training subset (AMT); a Toronto course handout describes 14,548 annotated clips across 257 games with per-frame boxes and tracking IDs — [arXiv 1511.02917](https://arxiv.org/pdf/1511.02917); [Toronto CSC2541 project 1](https://www.cs.toronto.edu/~urtasun/courses/CSC2541_Winter17/project_1.pdf)
- Broadcast view, NCAA (college) footage. Official hosting (basketballattention.appspot.com) and the Toronto 2.6 GB mirror (cs.toronto.edu/~namdar/sports_course/ncaa/data.7z) are of unknown current status; no license found — UNVERIFIED availability (search results only).

**UniqueData / TrainingDataPro "basketball_tracking" (Hugging Face)** — https://huggingface.co/datasets/TrainingDataPro/basketball_tracking
- Sample-only commercial set: screenshots from basketball game videos, ball bounding boxes with "occluded" and "basket" attributes, CVAT annotations.xml, 206 MB, license cc-by-nc-nd-4.0, full version paid on request — [HF page](https://huggingface.co/datasets/TrainingDataPro/basketball_tracking). No player tracks.

**Roboflow Universe basketball detection sets (UNVERIFIED — pages returned 403)**
- "Basketball Broadcast" (workspace basketball-w376e) has versions "Ball and Made Basket" and "Player and Referee" — [search summary](https://universe.roboflow.com/basketball-w376e/basketball-broadcast)
- "NBA-Player-Detector" (francisco-zenteno-uryfd), 3 classes; "NBA Players Detection" 68 images; "basketball_player_segmentation" (mr-iexff) listed CC BY 4.0; "basketball-player-detection" (roboflow-jvuqo) includes action classes (block, box-out, defence, dribble ...) — [search results](https://universe.roboflow.com/roboflow-jvuqo/basketball-player-detection-imrui)
- Roboflow's Skalski basketball tutorial trains on NBA footage and uses a court-keypoint dataset; the write-up references a "nba-court" dataset (betatracker) with boxes on baseline markers and paint corners — [Roboflow blog](https://blog.roboflow.com/identify-basketball-players/); [search summary](https://universe.roboflow.com/betatracker/nba-court)

**Related not-basketball MOT (for context only)**: DanceTrack is not basketball; SoccerNet-Tracking/GSR are soccer broadcast (GSR: 57 train / 59 val / 50 test 30-s clips) — [SoccerNet GSR via search](https://openaccess.thecvf.com/content/CVPR2024W/CVsports/papers/Somers_SoccerNet_Game_State_Reconstruction_End-to-End_Athlete_Tracking_and_Identification_on_CVPRW_2024_paper.pdf)

### Inferences
- For NCAA-domain training, SportsMOT basketball is the single best public MOT source: it is the standard main-camera broadcast view and is NCAA-derived. Expect ~845 frames and ~10 tracks per clip; with 240 clips split roughly evenly across three sports (not confirmed), the basketball portion is likely on the order of 60-80 clips, but this is an estimate, not a published figure.
- Because SportsMOT has only player boxes/IDs, team, jersey, ball and court annotations have to come from other sets (DeepSportradar for calibration/ball, E-BARD/Roboflow for jersey numbers) or be self-labelled.
- No public 5v5 broadcast basketball MOT dataset newer than SportsMOT was found in searches covering 2024-2026; the 2025-2026 basketball dataset activity is in 3x3 (TrackID3x3) and in video-QA (BasketballBench).

### Gaps
- Exact per-sport clip counts for SportsMOT train/val/test are not in the paper or README; the splits_txt files do not encode sport in the filename. Would need to download and inspect.
- Whether SportsMOT basketball includes any NBA clips (README says NBA; paper says NCAA/Olympics/Premier League via MultiSports) is unresolved.
- MultiSports page/licence not fetched this session.
- Roboflow page-level image counts/licences could not be verified (HTTP 403).

## Key question 2: DeepSportradar (MMSports 2022-2024): contents, league, license, availability

### Takeaway
DeepSportradar-v1 is a French LNB Pro A dataset captured by the Keemotion/Synergy automated sideline camera system (centre-line, high-resolution raw instants with exact camera calibration), not TV broadcast; it supports instance segmentation, ball 3D, camera calibration and re-ID, is CC BY-NC-SA 4.0 on Kaggle, and the challenge series ran 2022 and 2023 with no basketball tracking task and no 2024 edition found.

### Cited Findings
- Four basketball tasks (ball 3D localisation, camera calibration, player instance segmentation, player re-identification) with "high-resolution raw images, camera parameters and high quality annotations"; challenge at MMSports @ ACM MM 2022 — [arXiv 2208.08190](https://arxiv.org/abs/2208.08190)
- Capture system: "Both datasets were acquired during professional basketball matches with the Keemotion/Synergy Automated Camera System that offers a sideline view of the court from a position aligned with the court center line"; "made freely available to the research community by Sportradar" — [DeepSportradar-v1 paper PDF text](https://arxiv.org/pdf/2208.08190)
- DeepSport dataset (seg/ball/calib): raw instants from multi-camera arrays, camera resolutions 2-5 Mpx, 80-150 px/m, 15 arenas, 37 LNB Pro A games; 3 arenas held out for test; 3D world origin at far-left court corner, cm units; lens distortion assumed nil; the paper thanks the LNB (Ligue Nationale de Basketball) for allowing image sharing — [paper PDF text](https://arxiv.org/pdf/2208.08190); [Dataset Ninja](https://datasetninja.com/deep-sport-radar)
- Camera calibration task: 728 views (image + calibration pairs) randomly generated from DeepSport instants; train 480 / val 164 / test 84 plus 84-image secret challenge split; val arenas ks-fr-nantes, ks-fr-blois, ks-fr-fos; test/challenge rendered at 920x540; infinite random views can be generated — [paper PDF text](https://arxiv.org/pdf/2208.08190)
- Instance segmentation: 1,456 images, 4,104 mask instances (human 3,766 across 614 images; ball 338), 58% of images unlabeled in the Ninja release, 2 classes; splits 1,132/168/156 in the Ninja mirror; the paper's own subsets contain 1,674 + 344 annotations and a 64-image test (477 humans) — [Dataset Ninja](https://datasetninja.com/deep-sport-radar); [paper PDF text](https://arxiv.org/pdf/2208.08190)
- Ball 3D: two points per ball (centre and vertical ground projection) enabling 3D via known calibration; baseline MADE 2.12 px — [Dataset Ninja](https://datasetninja.com/deep-sport-radar); [paper PDF text](https://arxiv.org/pdf/2208.08190)
- Player ReID dataset (separate "Basketball ReID", originally VIPriors 2021): 99 sequences from 97 LNB Pro A games in 29 arenas, 20 frames at 10 fps per sequence, ~10 identities per sequence, 18,703 thumbnails; train 45 seq / 436 IDs / 8,569 images; test 5 seq (query 50, gallery 910); challenge 49 seq (query 468, gallery 8,703, IDs undisclosed); crops include players, coaches, referees; baseline 65% mAP — [paper PDF text, Table 2](https://arxiv.org/pdf/2208.08190)
- License: CC BY-NC-SA 4.0 (Dataset Ninja mirror of the Kaggle dataset) — [Dataset Ninja](https://datasetninja.com/deep-sport-radar). Kaggle source: https://www.kaggle.com/datasets/deepsportradar/basketball-instants-dataset (page did not render; license there UNVERIFIED but expected to match)
- Access: Kaggle dataset + per-task toolkits on GitHub (instance-segmentation-challenge, player-reidentification-challenge, camera-calibration-challenge, ball-3d-localization-challenge); EvalAI for submissions — [DeepSportradar GitHub org](https://github.com/DeepSportradar)
- 2023 edition: 3 challenges, 2 basketball (instance segmentation, re-identification) + cricket bowl release; EvalAI; rule that solutions must not use data beyond the toolkit — [challenge page via search](https://deepsportradar.github.io/challenge.html)
- No player-tracking challenge and no 2024 edition appear on the GitHub org (latest activity October 2023) — [DeepSportradar GitHub org](https://github.com/DeepSportradar)
- Keypoints: DeepSportRadar basketball instants annotate 4 keypoints per player on >700 images (per a 2024 paper's comparison) — [arXiv 2412.06258 via search](https://arxiv.org/pdf/2412.06258) (UNVERIFIED against the original release)
- Court-calibration limitation: "The training set does not contain a wide variety of arenas. It is therefore hard to generalize to various court color designs." — [KaliCalib, arXiv 2209.07795](https://arxiv.org/pdf/2209.07795)

### Inferences
- The camera is a fixed, centre-line sideline panorama at 2-5 Mpx, so it is closer to broadcast than TeamTrack's fisheye/drone views but lacks broadcast zoom/pan, scorebug overlays and cuts; calibration models trained on it will need domain adaptation for TV footage.
- The NC-SA license blocks commercial use of derived models in a strict reading; fine for research/eval.
- No NCAA footage; French professional league only.

### Gaps
- Kaggle page could not be read; exact license text there unverified (relying on Dataset Ninja's CC BY-NC-SA 4.0).
- Whether the 4-keypoint pose labels are in the Kaggle release or only in the VIPriors/other toolkits is unconfirmed.

## Key question 3: Jersey-number, basketball re-ID, basketball pose, and ball-detection datasets

### Takeaway
Basketball-specific jersey-number data is small and scattered (E-BARD ~3.6k 16-frame stubs CC BY 4.0 from NBA footage; a ~3.6k-image Roboflow NBA-Playoffs OCR set; older S2JN/Toutiao sets not publicly confirmed); SoccerNet-Jersey and hockey sets are the transfer options. For re-ID the only public basketball set is DeepSportradar ReID (LNB). For pose, nothing broadcast-basketball-specific exists beyond SpaceJam's 2D joints and the NBA2K synthetic set; for ball detection WASB-SBDT provides 275k NBA-broadcast frames with ball centres.

### Cited Findings

**Jersey numbers — basketball**
- E-BARD Jersey Number Recognition (HF, GabrieleGiudici): 3,633 manually labelled 16-frame stubs of ByteTrack-tracked players, 57 distinct numbers out of 101 classes, 300 "NaN" (not visible); pose-enhanced subset 983 stubs via MediaPipe torso crops; 2.17 GB; license CC-BY-4.0; footage not named on the HF page — [HF E-BARD](https://huggingface.co/datasets/GabrieleGiudici/E-BARD-JerseyNumberRecognition). The E-BARD project extends BARD, "classifying primary actions in NBA videos", with dense labels for players, referees, ball, hoop, team colours and jersey numbers; benchmarks YOLO/RF-DETR, CLIP/SigLIP2 for team, olmOCR/Qwen2.5-VL for numbers; code at github.com/GabrieleGiudic/E-BARD; license/venue not stated — [BODaI-Lab page](https://bodai.unibs.it/?p=9917). So the footage is NBA broadcast (inference from BARD description).
- Roboflow "basketball-jersey-numbers-ocr": ~3.6k crops from the 2025 NBA Playoffs, pre-labelled with SmolVLM2 then manually corrected; incomplete coverage of 00-99 and class-imbalanced; pre-fine-tune SmolVLM2 accuracy ~56% — [Roboflow blog](https://blog.roboflow.com/identify-basketball-players/); listed at ~3.62k images under account Vicroe / workspace roboflow-jvuqo v3 — [search summary](https://universe.roboflow.com/vicroe). License UNVERIFIED (page 403).
- Roboflow "Jersey Number Annotations Close Up" (cbb-analytics-computer-vision workspace — name suggests college basketball) exists; contents UNVERIFIED — [search result](https://universe.roboflow.com/cbb-analytics-computer-vision/jersey-number-annotations-close-up)
- S2JN (2021, SciTePress): 1,872 basketball images + 851 basketball images (different arena) + 1,317 ice hockey images with jersey-number boxes; no public download confirmed — [SciTePress paper](https://www.scitepress.org/Papers/2021/103417/103417.pdf)
- Toutiao AI Lab (Li et al. 2018, STN-based jersey recognition) said they would release a dataset; release unconfirmed — [li2018jersey.pdf](https://sites.cs.ucsb.edu/~lilei/pubs/li2018jersey.pdf)
- Generalized JNR (arXiv 2406.01033, 2024) combines SoccerNet with jersey images from football, basketball, volleyball, baseball; availability not checked — [arXiv 2406.01033](https://arxiv.org/abs/2406.01033)

**Jersey numbers — transfer sets (soccer/hockey)**
- SoccerNet Jersey (sn-jersey): 2,853 player tracklets extracted from SoccerNet tracking videos (derived from sn-gamestate/sn-tracking), challenge set 1,211 tracklets with hidden labels; JSON {player_id: number, -1 = not visible}; download via SoccerNet pip `downloadDataTask(task="jersey-2023")`; no license on the repo page, only a citation request — [sn-jersey GitHub](https://github.com/SoccerNet/sn-jersey). A later paper reports 4,064 tracklets / ~2.05M images across train/val/test/challenge — [arXiv 2309.06285](https://arxiv.org/pdf/2309.06285). SoccerNet gated data requires a Hugging Face account and acceptance of the SoccerNet NDA (old form deprecated) — [soccer-net.org/data](https://www.soccer-net.org/data). Soccer broadcast, not basketball.
- Hockey: Chan, Levine & Javan 2020 NHL broadcast tracklet set (unbalanced, incomplete number coverage, uses Sportlogiq IP with permission) — [arXiv 2009.02429](https://arxiv.org/pdf/2009.02429); Koshkina & Elder 2024 image-level set from University Hockey (9 games, stationary wide camera) + McGill NHL tracking dataset (8 broadcast games), code/models/data on GitHub, license unstated — [arXiv 2405.13896](https://arxiv.org/html/2405.13896v1)

**Basketball re-ID**
- DeepSportradar Player ReID (details in Key question 2): 18,703 thumbnails, 436 train IDs, LNB Pro A, same-camera 20-frame sequences; recent pose-injected embeddings reach 80.6% mAP vs 77.1% on its test set — [paper PDF text](https://arxiv.org/pdf/2208.08190); [arXiv 2208.08190 via search](https://arxiv.org/pdf/2208.08190.pdf)
- "Deep-SportLab / DeepSport-Basketball" indoor broadcast-camera set of 672 frames cited by TrackID3x3 as prior work — [arXiv 2503.18282 via search](https://arxiv.org/pdf/2503.18282v2) (UNVERIFIED)
- Sports Re-ID (arXiv 2206.02373) shows surveillance re-ID degrades on broadcast team sports without modification; evaluated on SoccerNet ReID, not basketball — [arXiv 2206.02373](https://arxiv.org/html/2206.02373v2)
- No broadcast NBA/NCAA re-ID benchmark with identity labels was found.

**Basketball pose**
- SpaceJam (simonefrancia): ~32,560 examples total; 16-frame single-player action clips (.mp4) plus a joint dataset of 2D (x,y) image-plane joints (.npy); MIT license; Google Drive download; source footage and pose extractor not stated — [SpaceJam GitHub](https://github.com/simonefrancia/SpaceJam)
- NBA2K dataset (Zhu et al., ECCV 2020 "Reconstructing NBA Players"): synthetic captures from NBA 2K19 (2ku/normal/rest-pose modes), 28 synthetic players, RGB + meshes + textures + 3D bone transforms + 4x4 camera projection + 3D joints + court line masks; >100 GB; access via Google form; license CC BY-NC-SA 4.0 (University of Washington); real-NBA-likeness data withheld "due to copyright issues" — [NBA2K-dataset GitHub](https://github.com/luyangzhu/NBA2K-dataset)
- TrackID3x3 (10 keypoints, 3x3, fixed/drone) and the 6-keypoint 3x3 set — see Key question 1.
- SportsPose is a general 3D sports pose dataset (throw, soccer, tennis, jump, volleyball), no basketball-specific subset — [arXiv 2507.20763 via search](https://arxiv.org/pdf/2507.20763)
- BASKET (CVPR 2025, UNC): ~4,477 h of video, 32,232 players, 21 pro/amateur leagues over six seasons, per-player 1-5 ratings on 20 skills; no boxes/pose; request form; license unstated; leagues unnamed — [BASKET project page](https://sites.google.com/cs.unc.edu/basket). Useful as raw broadcast video volume, not as a tracking label source.
- Basketball-51 (2021): clips cut from a live TV broadcast, labelled by scoring activity type; activity recognition only — [library record](https://katalog.lib.cas.cz/KNAV/EdsRecord/edsair,edsair.doi...........d65fc8fef785206542f526db5fd01075)
- NPUBasketball: no results found in searches this session.

**Ball detection**
- WASB-SBDT (BMVC 2023, NTT): Basketball dataset built from the video clips of Yan et al. ECCV 2020 [91] (the "NBA dataset" for group activity recognition); authors manually annotated 2D ball position in 45% (81/181 games) -> 275,328 annotated images in 3,824 clips; 1920x1080, fps N/A; train 70 games / 3,392 clips / 244,224 frames, test 11 games / 432 clips / 31,104 frames; mean ball displacement 33.7 px (largest of the five sports); "camera frequently moves and zooms in rapidly" — [WASB-SBDT paper PDF text, Sec. 4.1 and Table 1](https://arxiv.org/pdf/2311.05237). Code MIT; dataset license and download path not in README (GET_STARTED.md not read) — [WASB-SBDT GitHub](https://github.com/nttcom/WASB-SBDT). Single-ball (x,y) per frame — [search summary](https://arxiv.org/abs/2311.05237v1)
- DeepSportradar ball 3D (see Key question 2) and UniqueData ball boxes (Key question 1).

### Inferences
- The WASB basketball subset is, by frame count (275k) and camera behaviour (pan/zoom, NBA broadcast via Yan et al.'s NBA dataset), the largest public broadcast-basketball frame set with any annotation; worth checking whether the underlying NBA-dataset clips are still distributed by the ECCV 2020 authors.
- Jersey-number models for NCAA will need self-labelled college data; E-BARD and the Roboflow playoffs set give NBA uniforms only, and college uniforms/number fonts differ.
- Basketball re-ID from broadcast must currently be trained on DeepSportradar (centre-line camera) or SoccerNet ReID and then adapted.

### Gaps
- Licenses for Roboflow NBA jersey OCR set, S2JN, Toutiao, Koshkina hockey set, WASB basketball data, BASKET and MultiSports could not be verified.
- Whether WASB redistributes the NBA clips or only annotations is unconfirmed.
- No per-frame resolution/fps for E-BARD; no jersey-number count distribution beyond "57 of 101".

## Key question 4: Video paired with official positional tracking (SportVU, Second Spectrum, Sportradar/Genius, FIBA/EuroLeague, mirrors)

### Takeaway
No public dataset pairs official optical-tracking positions with the matching broadcast video for any basketball league; the only open official tracking is the 2015-16 NBA SportVU logs (no video, no stated license), and all newer NBA/WNBA/FIBA tracking (Second Spectrum/Genius, Sportradar) is proprietary. DeepSportradar is the only release with exact camera calibration for real pro games, and it uses an automated arena camera rather than TV feeds.

### Cited Findings
- SportVU 2015-16 (HF dcayton/nba_tracking_data_15_16): "600+" games, moments at 25 Hz with ball x/y/z and 10 players' team/player IDs and x/y/z, merged with play-by-play; configs tiny/small/medium/large (5/25/100/all games); sources credited as linouk23/NBA-Player-Movements and sumitrodatta/nba-alt-awards; no license stated; no video; some events lack coordinates; coordinates bleed across event boundaries; "the last iteration of publicly available tracking data", last eight seasons private — [HF nba_tracking_data_15_16](https://huggingface.co/datasets/dcayton/nba_tracking_data_15_16)
- linouk23/NBA-Player-Movements: raw SportVU logs under data/2016.NBA.Raw.SportVU.Game.Logs, JSON per game, event-indexed; no license file; no video — [GitHub](https://github.com/linouk23/NBA-Player-Movements)
- No paired NBA video: a 2015 Stanford project attempted camera calibration from NBA broadcast frames as an open alternative to SportVU but released no paired set — [Stanford CS231a project](https://stanford.edu/class/cs231a/prev_projects_2015/basketball.pdf)
- NCAA: Stats Perform generated player/ball tracking from broadcast video of NCAA D-I men's games (>650,000 possessions, >300M frames) internally; proprietary, not released — [Stats Perform](https://www.statsperform.com/resource/predicting-nba-talent-from-enormous-amounts-of-college-basketball-tracking-data)
- NBA/WNBA: Second Spectrum (Genius Sports, acquired 2021 for $200M) is the official NBA tracking provider and became WNBA optical tracking provider from 2024 (3D pose + ball); NBA "Dragon" mesh platform in development; no public/academic release found — [SVG](https://www.sportsvideo.org/2024/03/07/wnba-and-genius-sports-bring-data-tracking-to-every-wnba-arena/); [SBC News](https://sbcnews.co.uk/sportsbook/2021/05/07/genius-bolsters-sports-portfolio-with-200m-second-spectrum-acquisition/)
- FIBA: Genius Sports is FIBA's Official Data & Video Capture and Production Partner from 2025 for 10 years, including AI player tracking; FIBA also runs an Approval Program for tracking vendors; nothing public — [SVG Europe](https://www.svgeurope.org/blog/headlines/fiba-partners-with-genius-sports-to-deliver-ai-powered-player-tracking-and-statistics/); [FIBA](https://about.fiba.basketball/en/news/fiba-launches-approval-program-for-player-tracking-solutions-and-holds-inaugural-test-event)
- EuroLeague: only unofficial box-score/shot API wrappers (euroleague-api on PyPI); no tracking release — [PyPI euroleague-api](https://pypi.org/project/euroleague-api/0.0.12)
- Sportradar: its public release is DeepSportradar (LNB Pro A, arena cameras with calibration), not broadcast-plus-tracking — [DeepSportradar paper](https://arxiv.org/pdf/2208.08190)
- BasketballBench (arXiv 2608.23435, Aug 2026): 2,501 possession-level NBA broadcast clips from the 2025-26 season (test split of a "BasketEvent" set), each aligned to its source game and structured play-by-play records via scorebug period/game-clock; 7,980 QA items; official play-by-play, rosters, 530 player profiles; no boxes/tracks/calibration described; no download link on the arXiv page; paper license CC BY 4.0 — [arXiv 2608.23435](https://arxiv.org/abs/2608.23435); [HTML v1 via search](https://arxiv.org/html/2608.23435v1)
- Soccer analogues that do have video + game state (for method transfer): SoccerNet-GSR (broadcast, 30-s clips), SoccerTrack v2 (fixed 4K panoramic, 10 university matches, 932 min), FOOTPASS (54 full broadcast matches aligned to play-by-play and all-player game state) — [SoccerTrack v2](https://arxiv.org/abs/2508.01802); [FOOTPASS](https://arxiv.org/pdf/2511.16183); [SoccerNet GSR](https://www.soccer-net.org/tasks/game-state-reconstruction)

### Inferences
- The only feasible route to video-to-court supervision from official data is to self-pair the 2015-16 SportVU logs with 2015-16 NBA broadcast recordings obtained separately (clock-OCR alignment via game clock/shot clock, which the moments contain); no one has published such a pairing. Rights to the broadcast video would remain with the NBA/broadcasters.
- Play-by-play alignment to broadcast clips at possession level (as BasketballBench does via scorebug OCR) is demonstrated for 2025-26 NBA footage, so clock-based alignment is practical.
- For NCAA there is no official positional data at all in the public domain; court-plane supervision must come from self-labelled court keypoints/homographies (e.g., Roboflow court-keypoint sets or DeepSportradar-style line labels).

### Gaps
- No SportVU video pairing exists publicly; license of the SportVU logs is unstated (they were scraped from the NBA stats API, so terms are murky).
- BasketballBench/BasketEvent release location and license not found.
- No CMU/Second Spectrum academic release found; searches returned only commercial announcements.

## Key question 5: Camera viewpoint and NCAA inclusion — summary table

### Takeaway
Only SportsMOT (basketball subset) and the Ramanathan CVPR 2016 set are standard sideline broadcast views with NCAA footage; everything else is NBA broadcast (E-BARD, Roboflow NBA sets, WASB basketball, BasketballBench), European pro arena-camera (DeepSportradar), or fixed/fisheye/drone amateur footage (TeamTrack, TrackID3x3).

### Cited Findings
| Dataset | Camera | League | NCAA? | Core labels | License | Access |
|---|---|---|---|---|---|---|
| SportsMOT | main broadcast ("overhead shots"), 720p/25fps | NCAA (+Olympics/EPL for other sports; README also says NBA) | Yes | boxes + track IDs, players only | CC BY-NC 4.0, no redistribution | Codalab — [src](https://github.com/MCG-NJU/SportsMOT) |
| Ramanathan NCAA 2016 | broadcast (YouTube) | NCAA | Yes | events, boxes on 9k frames, clip-level tracks (per course handout) | unstated | mirrors of uncertain status — [src](https://arxiv.org/pdf/1511.02917) |
| DeepSportradar-v1 | fixed centre-line sideline arena cameras, 2-5 Mpx, calibrated | LNB Pro A (France) | No | masks, ball 3D, camera P matrix, re-ID IDs, (4 kp) | CC BY-NC-SA 4.0 | Kaggle + toolkits — [src](https://datasetninja.com/deep-sport-radar) |
| WASB-SBDT basketball | NBA broadcast (pan/zoom), 1080p | NBA | No | ball (x,y) per frame | code MIT; data license unstated | GitHub — [src](https://arxiv.org/pdf/2311.05237) |
| E-BARD JNR | NBA broadcast crops | NBA | No | jersey number per 16-frame stub | CC BY 4.0 | HF — [src](https://huggingface.co/datasets/GabrieleGiudici/E-BARD-JerseyNumberRecognition) |
| Roboflow NBA jersey OCR | NBA broadcast crops (2025 Playoffs) | NBA | No | jersey number text | unverified | Roboflow — [src](https://blog.roboflow.com/identify-basketball-players/) |
| TeamTrack basketball | fisheye 6K / drone 4K | university (Japan) | No | boxes + IDs | unstated | Kaggle/Drive — [src](https://atomscott.github.io/TeamTrack/) |
| TrackID3x3 | fixed consumer / drone | 3x3 amateur | No | boxes, IDs, jersey (outdoor), 10 kp | CC BY 4.0 | Drive — [src](https://github.com/open-starlab/TrackID3x3) |
| NBA2K | synthetic game engine | n/a | No | 3D pose, mesh, camera, court lines | CC BY-NC-SA 4.0 | request form — [src](https://github.com/luyangzhu/NBA2K-dataset) |
| SportVU 2015-16 | no video | NBA | No | 25 Hz xyz ball + xy players, PBP | unstated | HF/GitHub — [src](https://huggingface.co/datasets/dcayton/nba_tracking_data_15_16) |
| BasketballBench | NBA broadcast possessions | NBA 2025-26 | No | PBP alignment, QA | paper CC BY 4.0; data n/a | not found — [src](https://arxiv.org/abs/2608.23435) |

### Inferences
- For an NCAA broadcast pipeline, pretrain detection/MOT on SportsMOT, calibration on DeepSportradar plus self-labelled NCAA court keypoints, jersey OCR on E-BARD/Roboflow NBA crops plus self-labelled college crops, and ball detection on WASB basketball; expect a domain gap on every component except SportsMOT.

### Gaps
- No public dataset combines broadcast NCAA video with court-plane positions; this must be built in-house.
