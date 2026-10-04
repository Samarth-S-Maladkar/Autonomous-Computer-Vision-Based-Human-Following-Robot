# Autonomous Computer-Vision-Based Human-Following Robot

UE24CS352A Machine Learning mini-project.

A robot that follows a person using a single camera needs to solve two vision problems:

1. **Terrain classification:** which pixels are safe to drive on?
2. **Person detection and tracking:** where is the person to follow?

The two outputs are then combined into a drive command (`LEFT`, `RIGHT`, `FORWARD` or `STOP`).

The project is based on the Stanford report *"Golf Bag Carrier Robot Computer Vision"* (Gupta and Gloria), which used K-means, Felzenszwalb segmentation and patch features with softmax regression for terrain, and MobileNet-SSD for person tracking. We reproduce the terrain approach on a public dataset (RUGD) and build the person-following part on a pretrained detector.

**Team:** Sunidhi Shekar PES1UG24AM410 G Section and *(teammate name)*. 

**Status:** terrain classification is done (results below). Person following, the write-up and the slides are still in progress, listed under **To do** at the end of this file.

---

## Repository layout

```
.
├── README.md
├── requirements.txt
├── src/
│   ├── terrain.py        # terrain classification: training, evaluation, predict_mask()
│   ├── check_data.py     # sanity check of dataset layout and colormap
│   └── follow.py         # person following (not yet written)
├── results/
│   ├── terrain_metrics.json   # accuracy, IoU, confusion matrices
│   ├── terrain_model.joblib   # trained model (used by follow.py)
│   └── example_*.png          # photo | ground truth | prediction
└── data/                 # NOT in the repo, see "Dataset" below
```

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

The install includes PyTorch (through `ultralytics`), so it can take 5-10 minutes.

Always run commands from the **project root**, because the scripts use relative paths such as `data/RUGD_sample`.

---

## Dataset

We use **RUGD (Robot Unstructured Ground Driving)**, a public dataset of video frames from a small ground robot driving on trails, parks, creeks and villages, with pixel-level labels for 24 classes.

- Official site: <http://rugd.vision> (Wigness et al., IROS 2019)
- The `data/` folder is git-ignored. Download the data yourself as below.

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

| Our class | RUGD classes |
|---|---|
| **1 = traversable** | dirt, grass, asphalt, gravel, mulch, concrete, rockbed |
| **2 = hazard** | sand, water |
| **0 = other** | everything else (trees, bushes, sky, people, buildings, void, ...) |

RUGD has no putting-green or tee-box class, so these cannot be detected.

---

## Running the terrain classifier

1. Check that the dataset and colormap are read correctly:

```
python src/check_data.py
```

Expected: `pairs found: 26 | sequences: 18`, an empty "names we expect but colormap lacks" list, and a colour-match fraction of about 1.0.

2. Train and evaluate:

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

| Method | Pixel accuracy | Traversable IoU |
|---|---|---|
| Majority baseline (always "other") | 59.4% | 0.000 |
| K-means, colour only (k = 8) | 61.1% | 0.393 |
| Softmax, r = 1 (single pixel) | 68.2% | 0.436 |
| Softmax, r = 5 | 71.7% | 0.486 |
| **Softmax, r = 15 (best, saved)** | **74.2%** | **0.529** |

Larger neighbourhoods give steadily better results, which agrees with the Stanford report's argument for using local context. Colour alone (K-means) is barely above the baseline.

### Limitations

- **The hazard class (sand, water) was not learned.** It makes up only 323 of about 403,000 training pixels in the sample, and the model never predicted it correctly on the test set. We only claim traversable-terrain detection.
- **Small test set** (8 images, one run), so small differences, for example r = 5 vs r = 15, may be noise.
- **Noisy masks.** A linear classifier on colour patches confuses foliage with grass, producing false positives above the horizon.
- The softmax optimiser reached its 300-iteration limit for r = 5 and r = 15, so those models may be slightly under-trained.
- No putting greens or tee boxes (not in RUGD).

---

## References

1. Gupta, A. and Gloria, N. *Golf Bag Carrier Robot Computer Vision.* Stanford University (course project report).
2. Wigness, M., Eum, S., Rogers, J. G., Han, D. and Kwon, H. *A RUGD Dataset for Autonomous Navigation and Visual Perception in Unstructured Outdoor Environments.* IROS 2019.
3. Felzenszwalb, P. F. and Huttenlocher, D. P. *Efficient graph-based image segmentation.* IJCV 59, 2004.
4. Howard, A. G. et al. *MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications.* arXiv:1704.04861, 2017.

---

# To do

Reviews run **Mon Oct 5 to Fri Oct 9**. Final submission (repo and PDF write-up) is due **Sat Oct 10, 11:59 PM**.

## 1. Person following (`src/follow.py`)

This part depends only on `results/terrain_model.joblib` and `predict_mask()` in `src/terrain.py`. The RUGD dataset is not needed.

### Get set up
- [ ] `git pull`, then follow the **Setup** section above.
- [ ] Check that `results/terrain_model.joblib` exists.
- [ ] Commit under your own GitHub account (individual contribution is graded).

### Build it
- [ ] **Detect people** with a pretrained detector (`ultralytics` YOLOv8n, COCO class 0 = person). No training needed. Note in the write-up that this replaces the paper's MobileNet-SSD, and why (easier install, built-in tracker).
- [ ] **Track** with ByteTrack so each person keeps an ID across frames (`model.track(..., persist=True, tracker="bytetrack.yaml")`).
- [ ] **Lock the target:** press `L` to lock the ID of the person closest to the image centre. Only follow that ID.
- [ ] **Lost target:** if the locked ID is missing for N frames (start with 15), output `STOP`.
- [ ] **Steering:** `offset = (box_center_x - frame_width/2) / (frame_width/2)`, a value from -1 to +1. Use a dead zone (start with 0.2) to avoid jitter: below -0.2 gives `LEFT`, above +0.2 gives `RIGHT`.
- [ ] **Distance:** box height divided by frame height. Above about 0.55 means close enough, so `STOP`.
- [ ] **Terrain safety:** load the model with `joblib.load("results/terrain_model.joblib")` and call `predict_mask(model, r, rgb_frame)`. It returns a 112 x 200 array (0 = other, 1 = traversable, 2 = hazard). Input must be **RGB**, but OpenCV frames are BGR, so convert with `cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)`.
  - Look at the bottom-centre strip of the mask (the ground right ahead). If less than about 50% is traversable, output `STOP`.
  - Optionally stop if the ground just below the target's feet is hazard (see the terrain caveats below).
  - Run the terrain model only every few frames (for example every 5) to keep the frame rate up.
- [ ] **Decision order** (safety first): no lock or target lost, then unsafe ground, then hazard under target, then too close, then steer by offset, otherwise `FORWARD`.
- [ ] **Overlay and logging:** draw boxes (green for the target, grey for others), the terrain mask in a corner, and the command plus FPS on the frame. Save `results/demo.mp4` and `results/follow_log.csv` (frame, target found, command, reason, FPS).
- [ ] Run from a webcam (`python src/follow.py 0`) and from a video file.

### Test clips
- [ ] Record 30-60 seconds each, preferably outdoors on a path or park (matches the paper):
  1. One person walking toward, away from and across the camera.
  2. A second person crosses in front of the target (tests that the lock holds).
  3. The target leaves the frame and comes back (tests lost-target behaviour).
- [ ] Keep clips small and out of git (`results/*.mp4` is git-ignored).

### Numbers for the write-up
- [ ] **FPS** on a laptop, from `follow_log.csv`. The paper's concern is real-time speed on limited hardware.
- [ ] **Target-found rate:** fraction of frames where the locked person was found.
- [ ] **ID switches:** how many times the lock jumped to the wrong person, counted by watching the clip.
- [ ] Screenshots of one success and one failure (similar clothing, occlusion, or the target leaving the frame).

### Tune the thresholds
Starting values: dead zone 0.2, too-close 0.55, safe-ground fraction 0.5, terrain every 5 frames. Adjust them on the test clips and report the final values.

### Terrain model caveats
- **The hazard class (sand, water) was not learned.** The "hazard under the target" rule will almost never fire. The "unsafe ground ahead" rule, which uses the traversable class, is the one that works.
- **The mask is noisy.** It has false positives on foliage above the horizon and speckled edges, so the 50% threshold may need tuning. The model was trained on RUGD trails and creeks, so judge it on your own clips.
- Accuracy is 74.2% and traversable IoU is 0.53 on the RUGD sample (see the results table above), so don't expect precise ground maps.

## 2. Terrain (optional improvements)
- [ ] Raise `max_iter` from 300 to 1000 in `src/terrain.py` to remove the convergence warning, then rerun.
- [ ] Run on the full RUGD dataset (`python src/terrain.py data/RUGD`) for a larger test set and more sand and water pixels. If done, recommit the new model and metrics, and update the results table.

## 3. Repository
- [ ] Add a **Person detection, tracking and follow command** section to this README (method, how to run, keys `L` = lock and `Q` = quit, results, limitations), and update the **Status** line at the top.
- [ ] **Share the private repo with faculty and TAs** (Settings, then Collaborators). Only the repo owner can do this, and it is mandatory.
- [ ] Keep committing regularly from both team members.

## 4. Deliverables
- [ ] **One-page PDF write-up:** problem statement, dataset details, approach, implementation overview, conclusions. The assignment text says both "one-page" and "two-page", so confirm with faculty.
- [ ] **Slide deck (about 8 slides):** problem, the paper, data, methods, terrain results with example images, follow demo, limitations and future work.
- [ ] **Live demo rehearsal.** Both members should be able to explain every part of the code for the Q&A.
- [ ] Find out the **review slot** and submit before the deadline.
