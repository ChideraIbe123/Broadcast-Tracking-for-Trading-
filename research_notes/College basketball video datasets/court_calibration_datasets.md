# Basketball court calibration / keypoint / homography datasets — inventory as of 2026-11-02

Scope: every publicly reachable dataset (or model with recoverable training data) that gives, for basketball images/frames, court keypoints/landmarks, homographies or camera parameters, court-line/region segmentation, or field-registration ground truth. Verified directly on dataset pages where possible (Roboflow Universe pages were read in a live browser on 2026-11-02 because they block scripted fetches; Kaggle pages could not be rendered, see Gaps). "Footage judged from thumbnails" = visual inspection of the Universe preview strip on 2026-11-02; treat as approximate.

**Headline on NCAA/college footage:** Only ONE dataset with court-geometry labels was found whose images are confirmed as NCAA/college games: the private "College Basketball" homography set of Sha et al. (CVPR 2020, Stats Perform) — 13 NCAA games, 640 images, NOT released. On Roboflow Universe, the only college-flavored sets with court geometry are `keltons-workspace-tjkou/cmu-basketball-court-keypoints` (92 images, elevated fixed-camera view of a CMU gym — Division III, not broadcast) and, arguably, none with D-I broadcast look. Two Universe projects contain NCAA broadcast frames but NO court geometry labels (`ncaa-duke-player-detection/duke-ncaa-basketball-detection`, player boxes; `college-basketball-computer-vision/basketball-court-view-detector`, court-view vs non-court-view classification). Everything else is NBA broadcast, FIBA/European pro (LNB Pro A, Paraíba), amateur/high-school fixed camera, or phone video.

---

## Key question 1 — Roboflow Universe: basketball court keypoint / homography / court detection / segmentation projects

### Takeaway
Roboflow Universe hosts roughly 40 basketball court projects; the de-facto standard keypoint set is Roboflow's own `roboflow-jvuqo/basketball-court-detection-2` (850 NBA-broadcast images, 33 numbered landmarks, CC BY 4.0), with several forks/re-exports. The largest sets by count (statslane 8.6k, statslanes 3k, nxt1 2.3k, hs-basketball 861) are fixed-camera amateur/high-school footage, not broadcast. Nearly all are CC BY 4.0; none ship an explicit court-coordinate table on the Universe page (the index→court-location map must be reconstructed), and no project is identifiably NCAA broadcast.

### Cited Findings

**A. Keypoint-detection projects (court landmarks)** — all values read from the Universe project page on 2026-11-02 unless noted.

| # | Project (workspace/slug) | Images | License (exact page string) | Classes / schema | Updated / cite date | Footage (judged from thumbnails or filenames) |
|---|---|---|---|---|---|---|
| 1 | [roboflow-jvuqo/basketball-court-detection-2](https://universe.roboflow.com/roboflow-jvuqo/basketball-court-detection-2) | 850 (22 versions; v19 had 1,460 after augmentation; v21/v22 = 850 at 960×960 and 576×576 stretch) | "CC BY 4.0" | 1 class `court`; 33 landmarks per Roboflow's RF-DETR blog ([source](https://roboflow.com/blog/train-rf-detr-keypoint), post dated Jun 26 2026); keypoint names are 2-digit codes "01","02","04",…,"41" per the ONNX re-export's keypoints_metadata.json ([source](https://huggingface.co/MTerryJack/Element-PitchCalib-Basketball-Court/raw/main/keypoints_metadata.json)); a fork exported in YOLOv8 format has `kpt_shape: [48, 3]` ([source](https://huggingface.co/datasets/koppolusameer/basketball-keypoint-detection-yolov8/raw/main/data.yaml)) — i.e. the fork pads to 48 slots | Updated 2 months ago; BibTeX year 2026 month aug; models incl. yolov11x-pose, rfdetr-keypoint-preview; 36 stars, 16.3k views, 291 downloads | **NBA broadcast**: filenames in the HF export are "boston-celtics-new-york-knicks-game-1-q1…", "cleveland-cavaliers-indiana-pacers-game-2…", "denver-nuggets-los-angeles-clippers-game-1…" (2025 NBA playoffs) ([source](https://huggingface.co/api/datasets/koppolusameer/basketball-keypoint-detection-yolov8)); thumbnail = pro arena broadcast. Used by Roboflow's 3-second-violation post (dataset v13) ([source](https://roboflow.com/blog/detect-3-second-violation-ai-basketball), Jul 22 2025). Split in the 850-image fork: train 610 / valid 124 / test 116. No description published; no court-coordinate table on page. |
| 2 | [samet-mmrat/basketball-court-detection-2-axedc](https://universe.roboflow.com/samet-mmrat/basketball-court-detection-2-axedc) | 850 | "CC BY 4.0" | `court` (fork of #1) | Updated 9 months ago; cite 2026-feb; yolov11m-pose | Same images as #1 |
| 3 | [sameer-prasad-koppolu/basketball-court-detection-2-btzpp](https://universe.roboflow.com/sameer-prasad-koppolu/basketball-court-detection-2-btzpp) | 850 | "CC BY 4.0" | `court` (fork of #1; 48-slot YOLO export) | Updated 4 months ago; yolov11n-pose | Same images as #1; mirrored to HF (see KQ4) |
| 4 | [cocojumbopremium/homography-enrqp](https://universe.roboflow.com/cocojumbopremium/homography-enrqp) ("Homography") | 860 | "CC BY 4.0" | `court`; 20 dataset versions, 2 models (yolov8m, yolov11m-pose) | Updated 9 months ago; cite 2026-feb | Thumbnail is a close-up of a blue court; likely a re-annotation/extension of #1 (850→860) but not verifiable |
| 5 | [priver-one/basketball_courts](https://universe.roboflow.com/priver-one/basketball_courts) | 1,120 | "CC BY 4.0" | `basketball` (single class); roboflow-3-n-pose | Updated a year ago; cite 2025-jun | **NBA broadcast** (thumbnail shows pro arena with broadcast score bug) |
| 6 | [basketball-w376e/key-point-tq6ks](https://universe.roboflow.com/basketball-w376e/key-point-tq6ks) ("key point") | 211 | "CC BY 4.0" | 2 classes `court-keypoints`, `Whole-Court`; roboflow-3-n-pose; mAP50 86.1% | Updated 2 years ago; cite 2024-aug | Pro broadcast look (same workspace as "Basketball Broadcast" 1,853-image OD set) |
| 7 | [sportanalytics-sykyt/nba-court-keypoint-detection-u9n5v-gwoaa](https://universe.roboflow.com/sportanalytics-sykyt/nba-court-keypoint-detection-u9n5v-gwoaa) | 80 | "CC BY 4.0" | `court`; roboflow-3-n-pose | Updated a year ago; cite 2025-may | NBA (by name) |
| 8 | [easyaiedu/nba-court-keypoint-detection-u9n5v](https://universe.roboflow.com/easyaiedu/nba-court-keypoint-detection-u9n5v) | 80 | "CC BY 4.0" | `court`; no model | Updated 2 years ago; cite 2025-jan | NBA (by name); same slug root as #7 (fork) |
| 9 | [nbavideolabeling/nba-court-keypoint-detection](https://universe.roboflow.com/nbavideolabeling/nba-court-keypoint-detection) | 198 | "CC BY 4.0" | 2 classes `0`, `court`; 6 versions, 4 models (yolov8x) | Updated ~30 days ago; cite 2026-oct | **NBA broadcast** (thumbnail) |
| 10 | [fyp-3bwmg/basketball-court-detection-fevi7](https://universe.roboflow.com/fyp-3bwmg/basketball-court-detection-fevi7) | 203 | "CC BY 4.0" | `basketball_court`; yolov8x; 9 versions, 7 models | Updated 2 years ago; cite 2025-mar | **Outdoor/amateur court, phone-style** (thumbnail shows outdoor green/red court) |
| 11 | [fyp-3bwmg/basketball-court-detection-2-gzt6u](https://universe.roboflow.com/fyp-3bwmg/basketball-court-detection-2-gzt6u) | 100 | "CC BY 4.0" | `basketball_court` | cite 2025-mar | Same author as #10 |
| 12 | [fyp-3bwmg/basketball-court-detection-2-ku23t](https://universe.roboflow.com/fyp-3bwmg/basketball-court-detection-2-ku23t) | 203 | "CC BY 4.0" | `basketball_court` | cite 2025-mar | Duplicate of #10 |
| 13 | [vayvay/basketball_keypoint_detection](https://universe.roboflow.com/vayvay/basketball_keypoint_detection) | 351 | "CC BY 4.0" | `court`; yolov8n; 7 versions | Updated 2 years ago; cite 2024-aug | Pro broadcast (thumbnail shows Orlando-Magic-style court logo) |
| 14 | [statslane/court-rvqsa](https://universe.roboflow.com/statslane/court-rvqsa) | 8,639 | "CC BY 4.0" | `court`; 1 version, 0 models | Updated 6 months ago; cite 2026-may | Indoor gym, elevated fixed camera, dense keypoint layout; sport not verifiable from thumbnails (surfaced in "basketball court keypoint" search; could be volleyball/multi-sport) |
| 15 | [statslanes-workspace/court-5mbb5](https://universe.roboflow.com/statslanes-workspace/court-5mbb5) | 3,000 | "CC BY 4.0" | `court` | Updated 7 months ago; cite 2026-mar | Same footage family as #14 |
| 16 | [nxt1/hs-basketball-court-nxt1](https://universe.roboflow.com/nxt1/hs-basketball-court-nxt1) | 2,280 | "CC BY 4.0" | `basketball` | Updated 2 hours ago on 2026-11-02; cite 2026-nov | **High-school gym, elevated fixed camera** ("Saint Francis" centre logo) |
| 17 | [ethan-liu/hs-basketball-court-keypoints](https://universe.roboflow.com/ethan-liu/hs-basketball-court-keypoints) | 861 | "CC BY 4.0" | `basketball`; 16 versions, 0 models | Updated 3 months ago; cite 2026-jul | **High-school gym, fixed elevated camera** |
| 18 | [ortinss-workspace-zrabf/paraiba-basketball-court-detec](https://universe.roboflow.com/ortinss-workspace-zrabf/paraiba-basketball-court-detec) | 638 | "CC BY 4.0" | `court`; yolo26n-pose; mAP50 97.4% | Updated 5 months ago; cite 2026-jun | Brazilian (Paraíba) league broadcast-style |
| 19 | [keltons-workspace-tjkou/cmu-basketball-court-keypoints](https://universe.roboflow.com/keltons-workspace-tjkou/cmu-basketball-court-keypoints) | 92 | "CC BY 4.0" | `basketball_court`; yolo26n-pose; 3 versions | Updated 5 months ago; cite 2026-may | **US COLLEGE (Carnegie Mellon, NCAA D-III)** — elevated fixed wide camera of a college gym, not TV broadcast |
| 20 | [roboticflows-workspace/homography-4k](https://universe.roboflow.com/roboticflows-workspace/homography-4k) | 681 | "CC BY 4.0" | `Homography` | Updated 6 months ago; cite 2026-may | Club/amateur gym, fixed elevated 4K cameras |
| 21 | [roboticflows-workspace/homography-wluvo](https://universe.roboflow.com/roboticflows-workspace/homography-wluvo) | 190 | "CC BY 4.0" | `Homography` | cite 2026-may | Same family as #20 |
| 22 | [roboticflows-workspace/homography-new](https://universe.roboflow.com/roboticflows-workspace/homography-new) | 71 | "CC BY 4.0" | `Homography` | cite 2026-may | Same family as #20 |
| 23 | [mater-dei-6qhed/court-keypoint-detection-vhku5](https://universe.roboflow.com/mater-dei-6qhed/court-keypoint-detection-vhku5) | 258 | "CC BY 4.0" | `Court`; 0 versions | Updated 2 months ago; cite 2026-sep | Workspace name suggests Mater Dei high school; not verified |
| 24 | [alexanders-workspace-rkbz9/keypoint-just-court](https://universe.roboflow.com/alexanders-workspace-rkbz9/keypoint-just-court) | 201 | "CC BY 4.0" | `court`; yolo26s-pose | cite 2026-may | Not inspected |
| 25 | [kaikai23s-workspace/court-wmy9y](https://universe.roboflow.com/kaikai23s-workspace/court-wmy9y) | 255 | "CC BY 4.0" | `court`; 4 versions, 0 models | Updated 3 years ago; cite 2024-feb | Not inspected |
| 26 | [frt-data-scientists/basketball-court-sides](https://universe.roboflow.com/frt-data-scientists/basketball-court-sides) | 1,400 | "BY-NC-SA 4.0" (only non-CC-BY keypoint set found) | 1 class `orientation` (court-side/orientation points, not a full landmark schema) | Updated 10 months ago; cite 2026-jan | Outdoor and indoor amateur courts |
| 27 | [val-aospz/court-keypoint](https://universe.roboflow.com/val-aospz/court-keypoint) | 99 | "CC BY 4.0" | `court_kpts`; 0 versions | Updated 3 years ago; cite 2024-apr | Not inspected |
| 28 | [jingyi/basketball-court-hlifr](https://universe.roboflow.com/jingyi/basketball-court-hlifr) | 70 | "CC BY 4.0" | `court`; roboflow-3-s-pose | cite 2024-aug | Not inspected |
| 29 | [playerreco/basketball-court-ketpoints](https://universe.roboflow.com/playerreco/basketball-court-ketpoints) and [-2](https://universe.roboflow.com/playerreco/basketball-court-ketpoints-2) | 70 each | "CC BY 4.0" | `court` | cite 2024-aug | Not inspected |
| 30 | [yolov8-hc4mh/basketball-court-fgokc](https://universe.roboflow.com/yolov8-hc4mh/basketball-court-fgokc) | 77 | "CC BY 4.0" | `ba l l`, `court`, `hoop` | cite 2026-feb | Not inspected |
| 31 | [abschauhan/basketball-court-gexl4](https://universe.roboflow.com/abschauhan/basketball-court-gexl4) | 64 | "CC BY 4.0" | `basketball-court-keypoints`; 8 versions, 4 models | cite 2025-sep | Not inspected |
| 32 | [basketball-detection-ova9c/basketball-court-keypoint](https://universe.roboflow.com/basketball-detection-ova9c/basketball-court-keypoint) | 35 | "CC BY 4.0" | `Court`; 7 versions, 3 models | cite 2025-may | Not inspected |
| 33 | [nba-model/court-gdp40](https://universe.roboflow.com/nba-model/court-gdp40) | 31 | "CC BY 4.0" | `left_court`, `right_court` | cite 2024-mar | NBA (by name) |
| 34 | [tickstrike/basketball-court-keypoints1](https://universe.roboflow.com/tickstrike/basketball-court-keypoints1) | 29 (page says 29; a search snippet said 92) | "CC BY 4.0" | `p1`, `p1-p2-` | Updated 2 years ago; cite 2024-jun | Not inspected |
| 35 | nba-dataset-annotation "Court Keypoint Detection" | 9 | — | — | workspace page lists 9 images ([source](https://universe.roboflow.com/nba-dataset-annotation)) | NBA (by name) |

Non-basketball projects that appear in basketball searches (exclude): `hotmintsauce/court-key-points` (572 imgs; classes court-left/right, net-left/right — **padel/tennis**, thumbnails confirm), `chussboi/court-aalvr` (335 imgs — padel), `sebastian-james-sampao/tigervision-homography` (462 imgs, class `football_pitch`), `ball-is-lyfe/court-s2r4x` (125, class `pitch`), `wrobbinz/court-qo45v` (61, `court`,`net`), `vip-volleyball-wgfn3/ptz-homography-keafj` (322, volleyball). Deleted/404 on 2026-11-02: `hackolite/basket_court_keypoint`, `basketball-formations/warriors_basketball_court_keyp`, `basketball-formations/bball_court_detect_obb`, `frt-data-scientists/basketball-court-5kpts-center`, `ayush-pyr1j/basketball`.

**B. Court segmentation / court-region projects**

| Project | Images | License | Classes | Date | Footage |
|---|---|---|---|---|---|
| [jonatan-beiruty/court-detection](https://universe.roboflow.com/jonatan-beiruty/court-detection) | 2,034 | "CC BY 4.0" | inst-seg: `basketball-court`, `three point line`; roboflow-3-n-seg, mAP50 98.9% | Updated 3 yrs ago; cite 2023-dec | Indoor FIBA-style/amateur court (thumbnail) |
| [zy-vevvi/basketball-court-3ud15](https://universe.roboflow.com/zy-vevvi/basketball-court-3ud15) | 1,996 | "CC BY 4.0" | inst-seg: `basketball-court`, `three point line` | cite 2023-dec | Likely same source as above |
| [zy-vevvi/court-segmentation](https://universe.roboflow.com/zy-vevvi/court-segmentation) | 1,294 | "CC BY 4.0" | inst-seg: `basketball-court`, `Center-circle`, `Paint`, `three point line`; yolov8s | cite 2024-apr; 3.54k views, 109 downloads | Amateur gym, elevated fixed camera |
| [zy-vevvi/nba_court_segmentation](https://universe.roboflow.com/zy-vevvi/nba_court_segmentation) | 0 (empty) | "MIT" | `basketball-court`, `paint`, `three-point-line` | cite 2024-jan | empty |
| [projects-aglqb/basketball-court](https://universe.roboflow.com/projects-aglqb/basketball-court) | 1,102 | "CC BY 4.0" | inst-seg: `player`, `referee`, `basketball`, `basketball_court`, `throwline` | Updated 4 yrs ago; cite 2023-jan | Not inspected |
| [hackolite/basket_seg](https://universe.roboflow.com/hackolite/basket_seg) | 658 | "Public Domain" | inst-seg: `basketball-court`, `paint`, `three-point-line` | cite 2024-may | Not inspected |
| [segmentation-datasets/basketball_court_segmentation](https://universe.roboflow.com/segmentation-datasets/basketball_court_segmentation) | 261 | "CC BY 4.0" | inst-seg: `court`, `three_second_area`, `two_point_area`; description: "court, 3pt line and free throw lines segmented. Couldn't find one on the internet so I decided to make my own" | cite 2024-jan | Not inspected |
| [nba-dataset-annotation/court-segmentation-new](https://universe.roboflow.com/nba-dataset-annotation/court-segmentation-new) | 188 | "CC BY 4.0" | `COURT` | cite 2024-feb | NBA (by name) |
| [mkurt/basketball-court-segmentation](https://universe.roboflow.com/mkurt/basketball-court-segmentation) | 182 | "CC BY 4.0" | `Court` | cite 2026-jan | Not inspected |
| [nbavision/court-detection-2](https://universe.roboflow.com/nbavision/court-detection-2) | 176 | "CC BY 4.0" | semantic-seg `Basketball-courts` | cite 2024-jan | NBA (by name) |
| [baskemtball/basketball-courts](https://universe.roboflow.com/baskemtball/basketball-courts) | 81 | "CC BY 4.0" | `Basketball-courts` | cite 2023-aug | — |
| [nba-dataset-annotation/court-segmentation-ythyh](https://universe.roboflow.com/nba-dataset-annotation/court-segmentation-ythyh) | 50 | "CC BY 4.0" | `Court` | cite 2023-dec | — |
| [liucheng-sun-d5e7y/basketball-court-zhyz0](https://universe.roboflow.com/liucheng-sun-d5e7y/basketball-court-zhyz0) | 50 | "CC BY 4.0" | 19 fine classes (backboard, basket-ground, court-left/right, free-throw-line-left/right, full-court, half-court-center/left/right, …) | cite 2023-aug | — |
| [court-detection/basketball-court-detection](https://universe.roboflow.com/court-detection/basketball-court-detection) | 20 | "CC BY 4.0" | `court` | cite 2024-dec | — |
| [vayvay/basketball_segment_detection](https://universe.roboflow.com/vayvay/basketball_segment_detection) | 0 | "CC BY 4.0" | `arc`,`court`,`paint`,`three-point` | — | empty |

Related object-detection sets with court-line boxes (not keypoints): [basketball-w376e/basketball-broadcast](https://universe.roboflow.com/basketball-w376e/basketball-broadcast) — 1,853 images, "CC BY 4.0", 11 classes incl. `3-Point Line`, `Court Paint`, `Free Throw Key`, `Half Court`, cite 2024-aug; [nba-model/court-obb](https://universe.roboflow.com/nba-model/court-obb) — 426 images, class `sideline`, cite 2024-feb.

**C. NCAA-footage projects on Universe WITHOUT court-geometry labels (flagged for completeness)**
- [ncaa-duke-player-detection/duke-ncaa-basketball-detection](https://universe.roboflow.com/ncaa-duke-player-detection/duke-ncaa-basketball-detection) — 560 images, "CC BY 4.0", object detection (`ball`, `duke_player`, `hoop`, `otherplayer`, `ref`, `zion`), 8 versions, cite 2025-mar; thumbnails are Duke TV-broadcast frames. **NCAA broadcast frames, player boxes only.**
- [college-basketball-computer-vision/basketball-court-view-detector](https://universe.roboflow.com/college-basketball-computer-vision/basketball-court-view-detector) — 611 images, "CC BY 4.0", classification (`court view` / `non court view`), cite 2025-jul; thumbnail is a college-arena broadcast frame. Same workspace has `basketball-ballhandler-efficiency` (71 images, OD) ([workspace page](https://universe.roboflow.com/college-basketball-computer-vision)). **NCAA broadcast frames, no geometry.**

**D. Roboflow blog/cookbook context**
- RF-DETR keypoint fine-tuning post uses `roboflow-jvuqo/basketball-court-detection-2` v19 and says the dataset "provides 33 landmarks mapped to specific locations on a basketball court"; it mentions NBA broadcast footage only as the unseen inference demo ([source](https://roboflow.com/blog/train-rf-detr-keypoint), Jun 26 2026). The RF-DETR docs cookbook uses the same project (v19, COCO-keypoint export) ([source](https://rfdetr.roboflow.com/latest/cookbooks/fine-tune_keypoints/)).
- 3-second-violation post (Jul 22 2025) uses dataset v13 and a YOLOv11 keypoint model; keypoints are matched "to a canonical court template via homography" but no coordinate table is published ([source](https://roboflow.com/blog/detect-3-second-violation-ai-basketball)).
- Download mechanism for all Universe sets: Roboflow web export or `roboflow` Python SDK/API key (COCO-keypoint, YOLOv8-pose etc.); the `README.dataset.txt` in exports carries "License: CC BY 4.0" ([example export](https://huggingface.co/datasets/koppolusameer/basketball-keypoint-detection-yolov8/raw/main/README.dataset.txt)).

### Inferences
- The 33-landmark schema of `basketball-court-detection-2` is numbered ("01".."41" with gaps), so any homography pipeline needs a hand-built index→court-coordinate map; three independent repos (Purgty, cs25m116, koppolusameer) confirm users reconstruct this themselves.
- `cocojumbopremium/homography-enrqp` (860) and `priver-one/basketball_courts` (1,120) are the only other NBA-broadcast keypoint sets above 500 images; both are undocumented and likely derived from or similar to Roboflow's set.
- For college broadcast training data there is effectively nothing on Universe; the closest substitutes are NBA broadcast sets (same 94×50 ft court, different 3-pt arc) or the CMU fixed-camera set.

### Gaps
- Per-keypoint names and the court-coordinate table for `basketball-court-detection-2` are not published anywhere I could reach; only the numbered ID list was recoverable.
- Universe search results are paginated and fuzzy; projects with no "basketball"/"court"/"homography" in title could be missed. Image counts for the search-only entries were not re-verified on their pages beyond what is tabled.
- Footage type for ~12 small sets marked "Not inspected" was not visually checked.
- Native image resolution is not exposed on Universe pages (only export preprocessing, e.g. 640×640 / 960×960 stretch for #1).

---

## Key question 2 — Academic datasets (DeepSportradar, KaliCalib, SportsFields-type sets, NCAA, TrackID3x3, Basketball-51, SpaceJam, Kaggle)

### Takeaway
The only open, research-grade basketball calibration ground truth is DeepSportradar-v1's camera-calibration task (728 image/calibration pairs from French LNB Pro A, Kaggle, non-commercial). EPFL SportCenter (50k amateur phone frames with homographies) and TrackID3x3 (3x3, fixed/drone cameras, manual court keypoints) are the other open options. The two academic basketball homography sets that actually use broadcast/NCAA footage — Sha et al.'s College Basketball (640 images) and Nie et al.'s SportsFields basketball split (489 images) — were never released; 3DMPB turned out not to be a court-calibration set.

### Cited Findings

**DeepSportradar-v1 camera calibration (a.k.a. "Basketball Instants Dataset")**
- 728 views, each an image + calibration; splits 480 train / 164 val / 84 test, plus a separate 84-image challenge set with secret parameters — [arXiv 2208.08190 HTML](https://arxiv.org/html/2208.08190v1); same split numbers in [KaliCalib](https://ar5iv.labs.arxiv.org/html/2209.07795).
- Resolution: test/challenge views 920×540 px (the challenge repo generates 960×540 with MULTIPLICATIVE_FACTOR 2); training views are crops of 480×270 × scale α; pixel density 40–120 px/m; source cameras 2–5 Mpx at 80–150 px/m — [arXiv HTML](https://arxiv.org/html/2208.08190v1); [challenge repo](https://github.com/DeepSportradar/camera-calibration-challenge).
- Label schema: 3×4 projection matrix P = K[R|T]; world origin at far-left court corner inner line edge, cm units, z down, zero distortion; submissions are JSON dicts with T, K, kc, R, C, P, Pinv, Kinv; evaluated by MSE (cm) of 6 projected points — [arXiv HTML](https://arxiv.org/html/2208.08190v1); [challenge repo](https://github.com/DeepSportradar/camera-calibration-challenge). No explicit court-keypoint or line labels; the baseline renders 20 court-line masks from the camera and uses line intersections — [challenge repo](https://github.com/DeepSportradar/camera-calibration-challenge).
- League/footage: 37 professional games, 15 arenas, French LNB Pro A, Keemotion/Synergy automated fixed cameras (not TV broadcast); FIBA courts; test arenas KS-FR-CAEN, -LIMOGES, -ROANNE; val arenas nantes, blois, fos — [Dataset Ninja](https://datasetninja.com/deep-sport-radar); [challenge repo](https://github.com/DeepSportradar/camera-calibration-challenge); [arXiv HTML](https://arxiv.org/html/2208.08190v1).
- License: Dataset Ninja lists "CC BY-NC-SA 4.0" for the DeepSportRadar release — [source](https://datasetninja.com/deep-sport-radar); the ISPGroup DeepSport page says "cc-by-nc-sa", "non-commercial research only" — [source](https://ispgroup.gitlab.io/code/deepsport); the gabriel-vanzandycke/deepsport GitHub project states CC BY-NC-ND 4.0 for the project (not necessarily the data) — [source](https://github.com/gabriel-vanzandycke/deepsport). The challenge README itself has no license clause — [source](https://github.com/DeepSportradar/camera-calibration-challenge/blob/main/README.md).
- Download: `kaggle datasets download deepsportradar/basketball-instants-dataset` (needs Kaggle API token) + `tools/download_dataset.py`; challenge set via `wget https://arena-data.keemotion.com/tmp/gva/challenge_set.zip`; baseline weights from same host — [challenge repo](https://github.com/DeepSportradar/camera-calibration-challenge). Underlying DeepSport set at [kaggle.com/datasets/gabrielvanzandycke/deepsport-dataset](https://www.kaggle.com/datasets/gabrielvanzandycke/deepsport-dataset).
- Dates: arXiv 17 Aug 2022; MMSports'22 (ACM MM, Lisbon, 10 Oct 2022); repo last pushed 2026-07-29 — [arXiv abs](https://arxiv.org/abs/2208.08190); [org listing via GitHub API, 2026-11-02].
- The calibration challenge ran only in 2022; the 2023 edition had instance segmentation, re-ID and cricket only, and no challenges are listed for 2024+ — [deepsportradar.github.io/challenge.html](https://deepsportradar.github.io/challenge.html). A "DeepSportradar-v2: A Multi-Sport Computer Vision Dataset for Sport Understandings" paper exists in MMSports'23 proceedings — [unpaywall record](https://unpaywall.org/10.1145%2F3606038); its contents could not be retrieved.
- Dataset Ninja's Supervisely export (1,456 images, 1752×2336 sample) covers only the instance-segmentation task, not calibration — [source](https://datasetninja.com/deep-sport-radar).

**KaliCalib (MMSports 2022 challenge winner)**
- Method paper: Maglo, Orcesi, Pham (CEA LIST); encoder-decoder predicting K = 91 perspective-aware sampled court keypoints at 960×540 input; test MSE 126.61 cm, challenge 140.14 cm vs baseline 592.48 cm — [ar5iv 2209.07795](https://ar5iv.labs.arxiv.org/html/2209.07795). (The repo README reports 107.78 mm test / 73.16 mm challenge for the released models — [winners repo README](https://raw.githubusercontent.com/DeepSportradar/2022-winners-camera-calibration-challenge/main/README.md).)
- Code: [github.com/CEA-LIST/KaliCalib](https://github.com/CEA-LIST/KaliCalib) and mirror [DeepSportradar/2022-winners-camera-calibration-challenge](https://github.com/DeepSportradar/2022-winners-camera-calibration-challenge); license CeCILL-2.1; ships `model_test.pth` and `model_challenge.pth` references; weights license not stated; no new data (uses the challenge set).

**EPFL SportCenter (Citraro et al. 2020, "Real-time camera pose estimation for sports fields")**
- Single-view basketball: 50,127 images in 28 sequences from two moving smartphones (Samsung A5, iPhone 6) filming an amateur match; each image has camera pose + homography to a court-centred world frame; intrinsics and court template provided; faces blurred; plus 4 volleyball sequences — [EPFL page](https://www.epfl.ch/labs/cvlab/data/sportcenter-dataset/).
- License: no SPDX string; "free of use for research purposes", cite requested; Google Drive download; MVA paper 2020-03-25 — [EPFL page](https://www.epfl.ch/labs/cvlab/data/sportcenter-dataset/). The Waterloo survey marks its public availability "?" ("reported as publicly available, but do not seem to actually be so") — [Waterloo paper PDF](https://openjournals.uwaterloo.ca/index.php/vsl/article/download/3542/4588) (Table 1).

**College Basketball dataset — Sha et al., CVPR 2020 (Stats Perform) — NCAA, NOT PUBLIC**
- "We create a dataset from 13 NCAA basketball games. We use 10 games for training and the remaining 3 for testing … each game was played in a unique venue … For each game, we selected 30-60 frames"; professional annotators clicked 4–6 point correspondences per image; 526 training and 114 testing images (1,052 with horizontal flips) — [CVPR 2020 PDF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Sha_End-to-End_Camera_Calibration_for_Broadcast_Videos_CVPR_2020_paper.pdf).
- Listed as "College Basketball | Basketball | Open: No | 640 images | Homographies" in the ProCC benchmark table — [arXiv 2404.09807 HTML](https://arxiv.org/html/2404.09807v1); Waterloo Table 1 lists it as 526 train / 114 test with no public link — [Waterloo PDF](https://openjournals.uwaterloo.ca/index.php/vsl/article/download/3542/4588).

**SportsFields — Nie, Chen, Hamid, WACV 2021 (Amazon) — NOT PUBLIC**
- 192 clips across 5 sports; basketball split: 21 videos, 359 train / 130 test images; broadcast footage; grid-keypoint + homography labels — [WACV 2021 PDF](https://openaccess.thecvf.com/content/WACV2021/papers/Nie_A_Robust_and_Efficient_Framework_for_Sports-Field_Registration_WACV_2021_paper.pdf) (Table 2). No release link in the paper; KaliCalib notes the MMSports set was "the first public dataset for basketball" ([source](https://ar5iv.labs.arxiv.org/html/2209.07795)).

**SFLNet — Tarashima (ACPR 2019 / JIP 2021) — NOT PUBLIC**
- Basketball field-localization set of 1,232 images, semi-automatically built from human point correspondences; Waterloo table shows no public availability — [Waterloo PDF](https://openjournals.uwaterloo.ca/index.php/vsl/article/download/3542/4588); [J-STAGE record](https://www.jstage.jst.go.jp/article/ipsjjip/29/0/29_328/_article).

**3DMPB (Huang et al., Pose2UV, IEEE TIP 2022)** — listed by ProCC as "Basketball | Open | 10k | Pinhole" ([source](https://arxiv.org/html/2404.09807v1)), but the dataset repo describes it as a multi-person outdoor-sport mesh/pose benchmark (13,665 images, 25,122 persons; boxes, 2D pose, SMPL, masks, camera parameters; Baidu/Google Drive; MIT for repo) with no court annotations and no sport named — [3DMPB-dataset repo](https://github.com/boycehbz/3DMPB-dataset). Not a court-registration dataset.

**TrackID3x3 (Yamada et al., MMSports'25, arXiv 2503.18282)**
- Subsets: Indoor (fixed camera, 1280×720, 7,531 frames), Outdoor (iPhone 13 fixed, 3840×2160, 143,276 frames), Drone (DJI Air 2S, 3840×2160, 127,119 frames); 6 players boxed; 10-keypoint poses on a subset — paper Table 2 (local pdftotext of [arXiv PDF](https://arxiv.org/pdf/2503.18282)).
- Court: "court localization by manually annotating keypoints on a single frame" per fixed camera; "annotated keypoints in the center of a 5-centimeter wide court line"; court size 9.50×15.05 m (Indoor) / 11.05×15.05 m (Outdoor); player boxes converted to court coordinates by homography — same PDF. University gym / university players (Japan), 3x3 rules — same PDF.
- License CC BY 4.0 (dataset), Apache-2.0 code, CC BY-NC 3.0 jersey-number subfolder; Google Drive download — [GitHub](https://github.com/open-starlab/TrackID3x3). The paper's Table 1 lists "DeepSport-Basketball 672 frames, broadcast camera, indoor" as the only other basketball set with pose.

**NCAA dataset (Ramanathan et al., CVPR 2016)** — 257 untrimmed NCAA games, 14,548 4-s clips with 11 action labels, ~9,000 frames with player boxes; no court/homography labels; a Toronto course mirror hosts a ~2.6 GB 7z — [course sheet](https://www.cs.toronto.edu/~urtasun/courses/CSC2541_Winter17/project_1.pdf); [survey](https://arxiv.org/pdf/2206.01038). **NCAA footage, but no calibration GT.**

**Basketball-51 (Shakya, Zhang, Zhou, CS&IT May 2021)** — clips from one live TV broadcast labelled 3-pt/2-pt/free-throw; no court labels; no download link found — [AIRCC abstract](https://aircconline.com/csit/abstract/v11n7/csit110712.html).

**SpaceJam (Francia)** — ~32,560 16-frame player clips with 2D joints, MIT, Google Drive; no court keypoints/homography — [GitHub](https://github.com/simonefrancia/SpaceJam).

**Other 2025–2026 basketball datasets checked (no court calibration GT):** BasketballBench (arXiv 2608.23435, NBA 2025-26 broadcast clips, CC BY 4.0, QA benchmark) — [source](https://arxiv.org/abs/2608.23435); PL-NBA (arXiv 2608.19646, 11k possession clips, events only) — [source](https://arxiv.org/abs/2608.19646); CourtSI/CourtSI-Bench (arXiv 2603.09896) covers badminton/tennis/table tennis only — [source](https://arxiv.org/abs/2603.09896); BasketLiDAR (arXiv 2508.15299) is MOT, available on request — [source](https://arxiv.org/abs/2508.15299); BASKET (2503.20781) is skill estimation. ProCC (2404.09807) benchmarks only soccer sets — [source](https://arxiv.org/abs/2404.09807); Fan et al. CVPRW 2025 NeRF calibration uses SoccerNetV3 and WC14 only (local pdftotext of the CVF PDF).

**Kaggle:** no basketball court keypoint/homography dataset other than the two DeepSport entries above surfaced in web search; `site:kaggle.com` queries returned only Roboflow pages ([search run 2026-11-02](https://universe.roboflow.com/nbavision/court-detection-2)).

### Inferences
- For a college-broadcast calibration model there is no open academic ground truth; the practical path is (a) DeepSportradar for pretraining a FIBA-court line/keypoint model, (b) Roboflow NBA sets for broadcast appearance, (c) self-annotation of NCAA frames (the Sha et al. recipe: 4–6 clicked correspondences on 30–60 frames per game, 13 games, gave 640 usable images).
- The repeated discrepancy "728 vs 672 frames" for DeepSport in TrackID3x3's table most likely reflects train+val+test (728) vs a pose-annotated subset (672); treat 728 as the calibration count.

### Gaps
- The exact license string on the Kaggle pages for `deepsportradar/basketball-instants-dataset` and `gabrielvanzandycke/deepsport-dataset` could not be rendered (Kaggle blocked both the fetcher and the browser tool); secondary sources disagree (CC BY-NC-SA 4.0 vs CC BY-NC-ND 4.0).
- Whether the Keemotion challenge_set.zip URL still resolves was not tested.
- DeepSportradar-v2 (MMSports'23) contents unknown.
- SportCenter's actual downloadability (Waterloo flagged "?") was not tested.

---

## Key question 3 — GitHub repos that ship or link court keypoint annotations or trained court models (and their licenses)

### Takeaway
No GitHub repo ships an original basketball court-keypoint dataset; every pipeline either pulls the Roboflow `basketball-court-detection-2` family through an API key or keeps its data private. A handful ship weights (Google-Drive/HF `.pt` files for 18- or 33-keypoint YOLO-pose models), mostly without a weights license, and the two research-grade repos (KaliCalib, DeepSportradar) are CeCILL-2.1 / Apache-2.0 code over non-commercial data.

### Cited Findings
- **cs25m116/Basketball-Player-Detection-Tracking-Identification** — keypoint detector finds "up to 33 court landmarks" pulled via Roboflow API; demo clips "Celtics vs Knicks, 2025 NBA Playoffs Game 1"; no dataset link, no license, no local keypoint weights — [repo](https://github.com/cs25m116/Basketball-Player-Detection-Tracking-Identification).
- **moorexavierb/CourtVision** — `court_keypoints.pt` pose model with "18 court landmarks (see court/geometry.py)", training notebook needs a Roboflow key; targets "professional basketball games (NBA, NCAA)" and "True NBA (28.65 × 15.24 m) or FIBA spec"; license and whether weights are included not stated — [repo](https://github.com/moorexavierb/CourtVision). (Claims NCAA applicability; no NCAA data shipped.)
- **mgulati3/CourtVision** — MIT; pose-model court landmarks for homography; "NBA game footage"; dataset not named — [repo](https://github.com/mgulati3/CourtVision).
- **abdullahtarek/basketball_analysis** — MIT; `court_keypoint_detector.pt` (YOLOv8) via Google Drive link; training notebook `basketball_court_keypoint_training.ipynb`; dataset not named — [repo](https://github.com/abdullahtarek/basketball_analysis). Downstream forks: Manishmaurya89, HanaFEKI (ENSTA 2024-25, no license shown) — [HanaFEKI](https://github.com/HanaFEKI/AI_BasketBall_Analysis_v1).
- **Perceptron04/HoopVision** (created 2026-10-07) — `court_keypoint_detection.pt` hosted on Hugging Face (not in repo); no license — [GitHub API + README, 2026-11-02](https://github.com/Perceptron04/HoopVision). **mosmar99/HoopVision** — manual 4-corner court definition + panorama stitching, no keypoint model — [wiki/search](https://github.com/mosmar99/HoopVision).
- **mingchen-analytics/nbacourt-vision-tracker** ("nbatrack", created 2026-10-21, no license) — PTZ camera model fitted to "a few landmark clicks on one keyframe", propagated by KLT; calibration quality gate; a synthetic broadcast simulator with exact GT; no dataset or weights — [repo](https://github.com/mingchen-analytics/nbacourt-vision-tracker).
- **ianmparker/basketball-perception** (created 2026-10-23, AGPL-3.0) — "18-keypoint court model" (median reprojection error 4.04 ft vs 9.5 ft baseline), trained on research-only footage and **not distributed**; only `yolov8n.pt`/`yolov8x-pose.pt` backbones downloadable; notes transfer fails to a fixed top-down high-school camera — [repo](https://github.com/ianmparker/basketball-perception).
- **Purgty/Basketball-Homography** (MIT, 2025-10) — YOLOv11m-pose "33 keypoints (left, center, right)"; README: "Map keypoints to labels (not provided in dataset)" i.e. the index→court map must be hand-made — [repo](https://github.com/Purgty/Basketball-Homography).
- **RoyFerguson0/basketball-analytics** (2026-07, no license) — court keypoint detector folder, landmarks "basket center, free-throw line, court boundaries"; dataset not named — [repo](https://github.com/RoyFerguson0/basketball-analytics).
- **darkAlert/sports-field-homography** (GPL-3.0, 2021-22) — UNet+STN homography regressor; checkpoint name `NCAA+v6-640x360-aug_unet-resnet34…CP_epoch8.pth` implies training on NCAA frames at 640×360; model auto-downloads; **no dataset released** — [repo](https://github.com/darkAlert/sports-field-homography). **Flag: NCAA-trained weights (GPL-3.0), data not public.**
- **nash1588/BasketballCourtTagger** (2021, no license) — manual GUI producing `court_pts`, `game_res`, `court_res`, `h_mat` JSON against NBA court bitmaps; the only open basketball homography annotation tool found — [repo](https://github.com/nash1588/BasketballCourtTagger).
- **oscardiaz1331/sportcal** (AGPL-3.0, 2026-10) — multi-sport keypoint+line calibration; no data or weights in repo; lists SoccerNet and tennis HF sets as sources; basketball not among trained sports — [repo](https://github.com/oscardiaz1331/sportcal).
- **windoll7777-beep/basketball-sam** (MIT, 2026-08) — anchor-flow court calibration for broadcast/fixed-cam; explicitly excludes CBA raw broadcasts and private labels from git — [repo](https://github.com/windoll7777-beep/basketball-sam).
- **GauravMohan1/Basketball_Homography** (26 stars, 2021, no license) — manual homography from single-view broadcast; no dataset — [repo](https://github.com/GauravMohan1/Basketball_Homography).
- **CEA-LIST/KaliCalib** — CeCILL-2.1 code; weights `model_test.pth`/`model_challenge.pth`; dataset = DeepSportradar challenge set — [repo](https://github.com/CEA-LIST/KaliCalib).
- **DeepSportradar org** repos: `camera-calibration-challenge` (no license file; last push 2026-07-29), `2022-winners-camera-calibration-challenge` (cecill-2.1), `ball-3d-localization-challenge`, `instance-segmentation-challenge` (apache-2.0), `player-reidentification-challenge` (apache-2.0) — [GitHub API listing, 2026-11-02](https://github.com/DeepSportradar).
- **stephanj/basketballVideoAnalysis** wiki — home-made youth-game footage, polygon court masks (~1,000 frames/game), 4-corner homography demo; no released dataset or license; last edited Jan 2020 — [wiki](https://github.com/stephanj/basketballVideoAnalysis/wiki).
- **Basket-Analytics/BasketTracking** (MIT) — `rectify_court.py` produces homographies/panoramas; no labelled data — [repo](https://github.com/Basket-Analytics/BasketTracking).
- GitHub REST search on 2026-11-02 for "basketball court keypoints/homography/calibration" returned ~45 repos; none besides those above describe shipping annotations (full list in search output: eli-newman/basketball-court-tracker (empty README), Wasim7x/BasketVision (MIT, Roboflow-trained, no weights), knaitebes93-blip/CourtVision (Apache-2.0, amateur clips), movinzhang0810 & jabir3936 3x3 pipelines, etc.).

### Inferences
- The "18-keypoint" repos (moorexavierb, ianmparker) and the "33-keypoint" repos (cs25m116, Purgty, koppolusameer) reflect two schema lineages: a custom 18-point layout vs Roboflow's 33-point project.
- darkAlert's `NCAA+v6` checkpoint is the only publicly downloadable model with evidence of NCAA training; its GPL-3.0 code license plus unknown data provenance limits reuse.

### Gaps
- Could not read `court/geometry.py` (moorexavierb) or ianmparker's label format to confirm the 18-point naming.
- Perceptron04/HoopVision's HF weights repo was not located/inspected.
- Most repos were created in 2025–2026 with zero stars; maintenance and correctness unverified.

---

## Key question 4 — Hugging Face Hub (datasets and models), plus other hubs

### Takeaway
Hugging Face has no original basketball court-geometry dataset; it has one re-export of the Roboflow 850-image set (MIT-tagged on HF, CC BY 4.0 in its Roboflow README) and four small court-keypoint model repos (ONNX/YOLO-pose) trained on that same Roboflow project. Nothing on Zenodo, IEEE DataPort, Figshare or Mendeley Data was found for basketball court keypoints.

### Cited Findings
- **Dataset `koppolusameer/basketball-keypoint-detection-yolov8`** (lastModified 2026-06-20): 850 images (train 610 / valid 124 / test 116), YOLOv8-pose format, `kpt_shape: [48, 3]`, 1 class `court`; exported from Roboflow workspace `sameer-prasad-koppolu/basketball-court-detection-2-btzpp` on June 20 2026; README.dataset.txt says "License: CC BY 4.0" while the HF card says `license: mit`; filenames are 2025 NBA playoff clips (Celtics-Knicks, Celtics-Magic, Cavaliers-Pacers, Cavaliers-Heat, Nuggets-Clippers) — [data.yaml](https://huggingface.co/datasets/koppolusameer/basketball-keypoint-detection-yolov8/raw/main/data.yaml); [README.dataset.txt](https://huggingface.co/datasets/koppolusameer/basketball-keypoint-detection-yolov8/raw/main/README.dataset.txt); [HF API](https://huggingface.co/api/datasets/koppolusameer/basketball-keypoint-detection-yolov8). **NBA broadcast, not college.**
- **Model `koppolusameer/yolo11n-basketball-court-keypoints`** (AGPL-3.0; 2026-06-20): yolo11n-pose fine-tuned on the above "(48-slot court skeleton)"; outputs pixel keypoints; "Converting to court coordinates (homography) is a separate downstream step requiring an index->court-location map" — [README](https://huggingface.co/koppolusameer/yolo11n-basketball-court-keypoints/raw/main/README.md).
- **Model `MTerryJack/Element-PitchCalib-Basketball-Court`** (lastModified 2026-02-06): ONNX yolov11m-pose, tags element_type:pitchcalib/object:court; metadata `source: https://universe.roboflow.com/roboflow-jvuqo/basketball-court-detection-2`; `keypoints_metadata.json` lists 33 keypoints named "01","02","04","05","07"–"17","19","21","23","25"–"35","37","38","40","41"; files include weights.onnx; no license, empty README — [HF API](https://huggingface.co/api/models/MTerryJack/Element-PitchCalib-Basketball-Court); [keypoints_metadata.json](https://huggingface.co/MTerryJack/Element-PitchCalib-Basketball-Court/raw/main/keypoints_metadata.json).
- **Model `AnzeZ/basketball-court-yolo11m-pose`** (2026-05-06): `best.pt` + training curves; args.yaml shows data = `basketball-court-detection-2-19/data.yaml` (Roboflow v19), 80 epochs, imgsz 640; no card/license — [args.yaml](https://huggingface.co/AnzeZ/basketball-court-yolo11m-pose/raw/main/args.yaml).
- **Model `Jinqi-T/Basketball-Calibration`** (2026-06-12): only `model_small_final.pth`, `model_tiny_final.pth`; no README/license/dataset info — [HF API](https://huggingface.co/api/models/Jinqi-T/Basketball-Calibration).
- HF Hub API searches (datasets: "basketball" 85 hits, "court", "homography", "calibration"; models: "basketball", "court", "pitchcalib", "court_keypoint", "nba-court") returned no other basketball court-geometry resource; court-keypoint datasets on HF are tennis only (`Gholamreza/tennis_court_keypoints_dataset`, `oweng/tennis_court_keypoints_dataset`); `Cracked-ANJ/CourtKeyNet` is badminton/tennis — [HF API run 2026-11-02](https://huggingface.co/api/datasets?search=basketball); [CourtKeyNet README](https://huggingface.co/Cracked-ANJ/CourtKeyNet/blob/main/README.md).
- Basketball HF datasets with other labels (no court GT): `UniqueData/basketball_tracking` (CC BY-NC-ND 4.0), `TrainingDataPro/basketball_tracking`, `muyu111/basketball` (SHOT: tracks/poses/gaze), `BestWJH/VRU_Basketball` (CC BY 4.0, 3D), `sumeetn/sportsmot-basketball-detection` — [HF API](https://huggingface.co/api/datasets?search=basketball).
- Zenodo / IEEE DataPort / Figshare / Mendeley: web search found no basketball court keypoint dataset (only DeepDarts on IEEE DataPort) — [search result](https://ieee-dataport.org/open-access/deepdarts-dataset/embed).
- Dataset Ninja hosts only the DeepSportRadar instance-segmentation export (CC BY-NC-SA 4.0) — [source](https://datasetninja.com/deep-sport-radar).

### Inferences
- All HF basketball court models trace to the same 850 NBA images; model-license tags (MIT/AGPL) are the uploader's, while the underlying images stay CC BY 4.0 per Roboflow and come from NBA broadcasts of unclear redistribution rights.

### Gaps
- `Jinqi-T/Basketball-Calibration` architecture/training data unknown.
- HF full-text search is name-based; a repo with an unrelated name could be missed.

---

## Consolidated NCAA / US-college flag list
1. Sha et al. 2020 "College Basketball" — 13 NCAA games, 640 images, homographies — **not public** ([CVPR PDF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Sha_End-to-End_Camera_Calibration_for_Broadcast_Videos_CVPR_2020_paper.pdf)).
2. `darkAlert/sports-field-homography` — checkpoint named `NCAA+v6-640x360…` — weights downloadable (GPL-3.0 code), data not public ([repo](https://github.com/darkAlert/sports-field-homography)).
3. `keltons-workspace-tjkou/cmu-basketball-court-keypoints` — 92 images, CC BY 4.0, CMU gym fixed camera (college, non-broadcast) ([Universe](https://universe.roboflow.com/keltons-workspace-tjkou/cmu-basketball-court-keypoints)).
4. `ncaa-duke-player-detection/duke-ncaa-basketball-detection` — 560 NCAA broadcast frames, player boxes only, CC BY 4.0 ([Universe](https://universe.roboflow.com/ncaa-duke-player-detection/duke-ncaa-basketball-detection)).
5. `college-basketball-computer-vision/basketball-court-view-detector` — 611 college broadcast frames, view classification only, CC BY 4.0 ([Universe](https://universe.roboflow.com/college-basketball-computer-vision/basketball-court-view-detector)).
6. NCAA action dataset (Ramanathan 2016) — 257 games, boxes/actions, no geometry ([course mirror](https://www.cs.toronto.edu/~urtasun/courses/CSC2541_Winter17/project_1.pdf)).
7. `moorexavierb/CourtVision` claims NBA/NCAA applicability but ships no NCAA data ([repo](https://github.com/moorexavierb/CourtVision)).
