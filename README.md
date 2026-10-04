# Autonomous Computer-Vision-Based Human-Following Robot

UE24CS352A Machine Learning mini-project.

A robot that follows a person using a single camera needs to solve two vision problems:

1. **Terrain classification:** which pixels are safe to drive on?
2. **Person detection and tracking:** where is the person to follow?

The two outputs are then combined into a drive command (`LEFT`, `RIGHT`, `FORWARD` or `STOP`).

The project is based on the Stanford report *"Golf Bag Carrier Robot Computer Vision"* (Gupta and Gloria), which used K-means, Felzenszwalb segmentation and patch features with softmax regression for terrain, and MobileNet-SSD for person tracking. We reproduce the terrain approach on a public dataset (RUGD) and build the person-following part on a pretrained detector.

**Team:** Sunidhi Shekar PES1UG24AM410 G Section and Samarth S Maladkar PES1UG24AM396 G Section.

**Status:** terrain classification and person following are done and tested (results below). The write-up and the slides are in progress, listed under **To do** at the end of this file.

---

## Repository layout

```
.
├── README.md
├── requirements.txt
├── src/
│   ├── terrain.py        # terrain classification: training, evaluation, predict_mask()
│   ├── check_data.py     # sanity check of dataset layout and colormap
│   ├── follow.py         # person detection, tracking and follow command
│   └── analyze_log.py    # summarises follow logs (FPS, found rate, command counts)
├── results/
│   ├── terrain_metrics.json   # accuracy, IoU, confusion matrices
│   ├── terrain_model.joblib   # trained model (used by follow.py)
│   ├── example_*.png          # photo | ground truth | prediction
│   └── follow_log_clip*.csv   # per-frame logs from the test clips
├── data/                 # NOT in the repo, see "Dataset" below
└── videos/               # NOT in the repo, your own test clips (git-ignored)

```

`results/demo*.mp4` and `yolov8n.pt` are also git-ignored.

---

## Setup (Windows, VS Code)

Requires Python 3.9 or newer and Git.

```
git clone <repository-url>
cd Autonomous-Computer-Vision-Based-Human-Following-Robot
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt

```

If PowerShell blocks the activation script, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once and activate again. On Mac or Linux, activate with `source venv/bin/activate`.

The install includes PyTorch (through `ultralytics`), so it can take 5-10 minutes. The first run of `follow.py` downloads the YOLOv8n weights (`yolov8n.pt`, about 6 MB), so it needs an internet connection.

Always run commands from the **project root**, because the scripts use relative paths such as `data/RUGD_sample` and `results/terrain_model.joblib`.

---

## Dataset

We use **RUGD (Robot Unstructured Ground Driving)**, a public dataset of video frames from a small ground robot driving on trails, parks, creeks and villages, with pixel-level labels for 24 classes.

- Official site: [http://rugd.vision](http://rugd.vision/) (Wigness et al., IROS 2019)
- The `data/` folder is git-ignored. Download the data yourself as below.
- Only needed to retrain the terrain model. `follow.py` just uses the saved `results/terrain_model.joblib`.

### Option 1: the sample (used for the results below)

1. Download **"Download Sample"** (19 MB) from the RUGD download page and unzip it.
2. Copy the inner folder into the project so that you have:

```
data/RUGD_sample/images/<name>.png
data/RUGD_sample/annotations/<name>.png
data/RUGD_sample/RUGD_annotation-colormap.txt

```

The sample has 26 labelled images from 18 sequences.

### Option 2: the full dataset (optional, larger)

Download **Raw Video Frames with Annotations** (5.3 GB) and **RGB Annotation Files** (58.7 MB), then unzip them into:

```
data/RUGD/RUGD_frames-with-annotations/<sequence>/<frame>.png
data/RUGD/RUGD_annotations/<sequence>/<frame>.png
data/RUGD/RUGD_annotation-colormap.txt

```

Copy the colormap file from the sample into `data/RUGD/`. Run the code with `data/RUGD` as the argument (see below). The full-data loader takes every 25th frame of each sequence.

### Class mapping

The 24 RUGD classes are grouped into three:


| Our class           | RUGD classes                                                       |
| ------------------- | ------------------------------------------------------------------ |
| **1 = traversable** | dirt, grass, asphalt, gravel, mulch, concrete, rockbed             |
| **2 = hazard**      | sand, water                                                        |
| **0 = other**       | everything else (trees, bushes, sky, people, buildings, void, ...) |


RUGD has no putting-green or tee-box class, so these cannot be detected.

---

## Running the terrain classifier

1. Check that the dataset and colormap are read correctly:

```
python src/check_data.py

```

Expected: `pairs found: 26 | sequences: 18`, an empty "names we expect but colormap lacks" list, and a colour-match fraction of about 1.0.

1. Train and evaluate:

```
python src/terrain.py                 # uses data/RUGD_sample
python src/terrain.py data/RUGD       # uses the full dataset

```

This writes `results/terrain_metrics.json`, `results/terrain_model.joblib` (the model with the best traversable IoU) and `results/example_*.png`. It takes a few minutes. `ConvergenceWarning` messages from scikit-learn are expected and do not stop the run.

### Using the trained model in code

```python
import joblib, cv2
from terrain import predict_mask

tm = joblib.load("results/terrain_model.joblib")           # {"model": ..., "r": ...}
rgb = cv2.cvtColor(cv2.imread("frame.png"), cv2.COLOR_BGR2RGB)
mask = predict_mask(tm["model"], tm["r"], rgb)             # 112 x 200 array: 0 other, 1 traversable, 2 hazard

```

---

## Terrain methods

Images are resized to 112 x 200, as in the Stanford report, to keep the methods lightweight.

- **K-means (baseline).** Pixels are clustered by RGB colour only (k = 8, unsupervised). Each cluster is then assigned the class that most of its training pixels have in the ground truth.
- **Patch features + softmax regression.** Each pixel is described by the colours of its r x r neighbourhood (3r² numbers), so the classifier sees local edges and gradients. A softmax (multinomial logistic regression) classifier predicts one of the three classes. We compare r = 1 (single pixel), 5 and 15.

**Evaluation.** The split is by whole **video sequence**, not by pixel or frame, so test images come from places the model never saw. We report pixel accuracy, a majority-class baseline (always guess "other"), and IoU of the traversable class.

---

## Terrain results

Trained on 18 images and tested on 8 images from 5 held-out sequences (RUGD sample).


| Method                             | Pixel accuracy | Traversable IoU |
| ---------------------------------- | -------------- | --------------- |
| Majority baseline (always "other") | 59.4%          | 0.000           |
| K-means, colour only (k = 8)       | 61.1%          | 0.393           |
| Softmax, r = 1 (single pixel)      | 68.2%          | 0.436           |
| Softmax, r = 5                     | 71.7%          | 0.486           |
| **Softmax, r = 15 (best, saved)**  | **74.2%**      | **0.529**       |


Larger neighbourhoods give steadily better results, which agrees with the Stanford report's argument for using local context. Colour alone (K-means) is barely above the baseline.

### Terrain limitations

- **The hazard class (sand, water) was not learned.** It makes up only 323 of about 403,000 training pixels in the sample, and the model never predicted it correctly on the test set. We only claim traversable-terrain detection.
- **Small test set** (8 images, one run), so small differences, for example r = 5 vs r = 15, may be noise.
- **Noisy masks.** A linear classifier on colour patches confuses foliage with grass, producing false positives above the horizon.
- The softmax optimiser reached its 300-iteration limit for r = 5 and r = 15, so those models may be slightly under-trained.
- No putting greens or tee boxes (not in RUGD).

---

## Person detection, tracking and follow command

`src/follow.py` finds a person in each video frame, tracks them, and turns their position and the terrain ahead into a drive command: `LEFT`, `RIGHT`, `FORWARD` or `STOP`.

### Method

1. **Detection.** YOLOv8n (pretrained on COCO, class 0 = person) detects people. No training was needed. This replaces the paper's MobileNet-SSD because it installs easily and has a built-in tracker.
2. **Tracking.** ByteTrack gives each person an ID that persists across frames.
3. **Locking the target.** Only the locked ID is followed. `--auto-lock` locks the person closest to the image centre in the first frame where a person appears. In a live run, press `L` instead.
4. **Steering.** `offset = (box_centre_x - frame_width / 2) / (frame_width / 2)`, from -1 to +1. Below -0.2 gives `LEFT`, above +0.2 gives `RIGHT`, otherwise `FORWARD` (the 0.2 dead zone avoids jitter).
5. **Distance.** If the target's box height is above 0.55 of the frame height, the target is close enough, so `STOP`.
6. **Terrain safety.** Every 5th frame the terrain model from `terrain.py` produces a 112 x 200 mask. If less than 50% of the bottom-centre strip (the ground right ahead) is traversable, `STOP`. If the area just below the target's feet is hazard, `STOP`.
7. **Lost target and re-lock.** If the locked ID is missing for 15 frames, the output is `STOP`. The tracker often gives a returning person a new ID, so the script then re-locks onto the person nearest to where the target was last seen. Detections with a box height under 15% of the frame are ignored for re-locking, so small background detections cannot take over the lock.

Decision order (safety first): no lock, then target lost, then unsafe ground, then hazard under target, then too close, then steer by offset, otherwise `FORWARD`.

### How to run

From the project root, with the virtual environment active:

```
python src/follow.py 0                                      # webcam
python src/follow.py videos/clip1_walk.mp4 --auto-lock      # video file
python src/follow.py videos/clip1_walk.mp4 --auto-lock --no-show

```

Keys: `L` = lock target, `Q` = quit.

Options: `--auto-lock` (lock on the first person seen), `--no-show` (no window, just save the outputs), `--out` and `--log` (change the output paths).

Outputs: `results/demo.mp4` (annotated video) and `results/follow_log.csv` (frame, target_found, command, reason, fps). Each run overwrites them, so copy or rename them between runs. To summarise logs:

```
python src/analyze_log.py results/follow_log_clip1.csv results/follow_log_clip2.csv results/follow_log_clip3.csv

```

Test clips go in `videos/`, which is git-ignored. Record your own (landscape, 30-60 sec).

### Thresholds (set by hand on the test clips)


| Constant           | Value | Meaning                                           |
| ------------------ | ----- | ------------------------------------------------- |
| `DEAD_ZONE`        | 0.2   | Offset below this goes straight                   |
| `TOO_CLOSE`        | 0.55  | Box height / frame height above this means `STOP` |
| `SAFE_FRACTION`    | 0.5   | Minimum traversable share of the strip ahead      |
| `LOST_FRAMES`      | 15    | Frames without the target before `STOP`           |
| `MIN_REACQ_HEIGHT` | 0.15  | Smallest box height allowed when re-locking       |
| `TERRAIN_EVERY`    | 5     | Run the terrain model every N frames              |


### Follow results

Three phone clips (1024 x 576, 30 fps) recorded in a residential courtyard, run on a laptop CPU.


| Clip                                              | Frames    | Average FPS | Target found |
| ------------------------------------------------- | --------- | ----------- | ------------ |
| 1. Walk toward, away from and across the camera   | 793       | 24.6        | 98.9%        |
| 2. A second person crosses in front of the target | 555       | 24.9        | 97.3%        |
| 3. Target leaves the frame and returns            | 585       | 25.3        | 68.9%        |
| **Overall**                                       | **1,933** | **24.9**    | **89.3%**    |


- **Clip 1:** `FORWARD` while the target is centred, `STOP` when they are close enough, `LEFT` or `RIGHT` on the sideways walk.
- **Clip 2:** the lock stayed on the target (no ID switches). Tracking was lost for 15 frames while the other person blocked the target, then the target was re-locked.
- **Clip 3:** the low found rate is expected, because the target was out of frame for about 167 frames and the output was `STOP` ("target lost"). The target was re-locked when they walked back in.

An earlier version of the re-lock rule locked onto a small background detection in clip 3 and stayed there. The minimum box height (`MIN_REACQ_HEIGHT`) fixed it.

### Follow limitations

- **Terrain safety was not demonstrated.** The terrain model was trained on RUGD trails and creeks (74.2% pixel accuracy, 8 test images). On our urban courtyard clips the mask is almost entirely "traversable", so the unsafe-ground rule never fired. The hazard class (sand, water) was not learned either.
- **Re-locking uses position.** With several people close to where the target left, it could lock onto the wrong one. The minimum box size only filters out small background detections.
- **Three short clips,** one location, one target. Thresholds were set by hand and not cross-validated.
- **Single camera,** so box height is a rough distance estimate only.
- Runs on video and webcam only, with no physical robot.

---

## References

1. Gupta, A. and Gloria, N. *Golf Bag Carrier Robot Computer Vision.* Stanford University (course project report).
2. Wigness, M., Eum, S., Rogers, J. G., Han, D. and Kwon, H. *A RUGD Dataset for Autonomous Navigation and Visual Perception in Unstructured Outdoor Environments.* IROS 2019.
3. Felzenszwalb, P. F. and Huttenlocher, D. P. *Efficient graph-based image segmentation.* IJCV 59, 2004.
4. Howard, A. G. et al. *MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications.* arXiv:1704.04861, 2017.

---

# To do

Reviews run **Mon Oct 5 to Fri Oct 9**. Final submission (repo and PDF write-up) is due **Sat Oct 10, 11:59 PM**.

## Done

- [x] Terrain classification (K-means baseline and patch features + softmax, evaluated on held-out sequences)
- [x] Person detection, tracking, lock, lost-target handling, re-lock and follow command (`src/follow.py`)
- [x] Test clips, FPS, target-found rate and ID-switch counts
- [x] README person-following section

## Remaining

- [ ] **Share the private repo with faculty and TAs** (Settings, then Collaborators). Only the repo owner can do this, and it is mandatory.
- [ ] **One-page PDF write-up:** problem statement, dataset details, approach, implementation overview, conclusions. The assignment text says both "one-page" and "two-page", so confirm with faculty.
- [ ] **Slide deck (about 8 slides):** problem, the paper, data, methods, terrain results with example images, follow demo with screenshots, limitations and future work.
- [ ] **Screenshots** from the demo videos: a success, a failure or limitation, the crossing, and the re-lock.
- [ ] **Live demo rehearsal.** Both members should be able to explain every part of the code for the Q&A.
- [ ] Find out the **review slot** and submit before the deadline.
- [ ] Keep committing regularly from both team members.

## Optional improvements

- [ ] **Terrain:** raise `max_iter` from 300 to 1000 in `src/terrain.py` to remove the convergence warning, then rerun.
- [ ] **Terrain:** run on the full RUGD dataset (`python src/terrain.py data/RUGD`) for a larger test set and more sand and water pixels. If done, recommit the new model and metrics, and update the results table.
- [ ] **Follow:** record a clip outdoors on grass or a dirt path to show the terrain safety rule firing.
- [ ] **Follow:** require a new person to be visible for a few frames before re-locking, to reduce wrong re-locks.
