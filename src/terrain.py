import glob, os, sys, json
import cv2, joblib, numpy as np
from skimage.util import view_as_windows
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix

H, W = 112, 200                          # small size, as in the paper
TRAV = {"dirt", "grass", "asphalt", "gravel", "mulch", "concrete", "rockbed"}
HAZ = {"sand", "water"}
IMG_EXT = (".png", ".jpg", ".jpeg")
rng = np.random.default_rng(0)


def norm(name):
    """'rock-bed' / 'Rock_Bed' -> 'rockbed' so names match our sets."""
    return name.lower().replace("-", "").replace("_", "").replace(" ", "")


def load_colormap(root):
    """Each line ends with 3 colour numbers; the name is the word just before them."""
    files = glob.glob(os.path.join(root, "*olormap*.txt"))
    if not files:
        raise FileNotFoundError(f"No colormap .txt found in {root}")
    cmap = {}
    for line in open(files[0]):
        t = line.split()
        if len(t) >= 4 and all(v.isdigit() for v in t[-3:]):
            cmap[norm(t[-4])] = tuple(int(v) for v in t[-3:])
    return cmap


def find_pairs(root):
    """Return [(frame_path, annotation_path, sequence_name)].
    Works for the sample layout (images/ + annotations/) and the full layout."""
    pairs = []
    imgs = sorted(p for p in glob.glob(os.path.join(root, "images", "*"))
                  if p.lower().endswith(IMG_EXT))
    if imgs:                                              # sample layout
        for p in imgs:
            stem = os.path.splitext(os.path.basename(p))[0]
            a = glob.glob(os.path.join(root, "annotations", stem + ".*"))
            if a:
                pairs.append((p, a[0], stem.rsplit("_", 1)[0]))
    else:                                                 # full layout
        fdir = os.path.join(root, "RUGD_frames-with-annotations")
        adir = os.path.join(root, "RUGD_annotations")
        for seq in sorted(os.listdir(fdir)):
            for p in sorted(glob.glob(os.path.join(fdir, seq, "*")))[::25]:
                a = os.path.join(adir, seq, os.path.basename(p))
                if os.path.exists(a):
                    pairs.append((p, a, seq))
    return pairs


def to_classes(ann_rgb, cmap):
    lab = np.zeros(ann_rgb.shape[:2], np.uint8)           # 0 = other
    for name, rgb in cmap.items():
        m = np.all(ann_rgb == rgb, axis=-1)
        if name in TRAV: lab[m] = 1
        elif name in HAZ: lab[m] = 2
    return lab


def load_pair(frame_path, ann_path, cmap):
    img, ann = cv2.imread(frame_path), cv2.imread(ann_path)
    if img is None or ann is None:
        raise IOError(f"Could not read {frame_path} or {ann_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)            # OpenCV loads BGR
    ann = cv2.cvtColor(ann, cv2.COLOR_BGR2RGB)
    lab = to_classes(ann, cmap)
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    lab = cv2.resize(lab, (W, H), interpolation=cv2.INTER_NEAREST)
    return img, lab


def patch_features(img, r):
    """Each pixel -> flattened r x r x 3 neighbourhood (r odd). r=1 is plain RGB."""
    if r == 1:
        return img.reshape(-1, 3).astype(np.float32) / 255
    p = r // 2
    padded = np.pad(img, ((p, p), (p, p), (0, 0)), mode="reflect")
    win = view_as_windows(padded, (r, r, 3))
    return win.reshape(H * W, -1).astype(np.float32) / 255


def split_data(root, cmap):
    pairs = find_pairs(root)
    seqs = sorted({p[2] for p in pairs})
    test_seqs = set(seqs[::4])                            # whole sequences held out
    train = [load_pair(f, a, cmap) for f, a, s in pairs if s not in test_seqs]
    test = [load_pair(f, a, cmap) for f, a, s in pairs if s in test_seqs]
    print(f"{len(seqs)} sequences | train images {len(train)} | test images {len(test)}")
    print("test sequences:", sorted(test_seqs))
    return train, test


def sample_xy(pairs, r, per_img):
    X, y = [], []
    for img, lab in pairs:
        f = patch_features(img, r)
        idx = rng.choice(len(f), per_img, replace=False)
        X.append(f[idx]); y.append(lab.reshape(-1)[idx])
    return np.concatenate(X), np.concatenate(y)


def score(y_true, y_pred):
    acc = float((y_true == y_pred).mean())
    base = float(np.bincount(y_true, minlength=3).max() / len(y_true))  # always guess commonest
    t, p = y_true == 1, y_pred == 1
    iou = float((t & p).sum() / max((t | p).sum(), 1))
    return {"pixel_acc": round(acc, 4), "majority_baseline": round(base, 4),
            "traversable_IoU": round(iou, 4)}


def predict_mask(model, r, img_rgb):
    small = cv2.resize(img_rgb, (W, H), interpolation=cv2.INTER_AREA)
    return model.predict(patch_features(small, r)).reshape(H, W)


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "data/RUGD_sample"
    cmap = load_colormap(root)
    print("classes in colormap:", sorted(cmap))
    train, test = split_data(root, cmap)
    per_img = max(800, 50000 // len(train))               # ~50k+ training pixels in total

    yt = np.concatenate([l.reshape(-1) for _, l in test])
    ytr = np.concatenate([l.reshape(-1) for _, l in train])
    print("train pixels per class [other, traversable, hazard]:", np.bincount(ytr, minlength=3))
    print("test  pixels per class [other, traversable, hazard]:", np.bincount(yt, minlength=3))
    Xt = np.concatenate([patch_features(i, 1) for i, _ in test])
    results = {}

    # ---- K-means baseline (RGB only) ----
    X, y = sample_xy(train, 1, per_img)
    km = MiniBatchKMeans(n_clusters=8, random_state=0, n_init=3).fit(X)
    cl = km.labels_
    cluster_class = np.array([np.bincount(y[cl == c], minlength=3).argmax()
                              if (cl == c).any() else 0 for c in range(8)])
    results["kmeans_k8"] = score(yt, cluster_class[km.predict(Xt)])
    print("K-means:", results["kmeans_k8"])

    # ---- Patch features + softmax regression, three window sizes ----
    best, best_iou = None, -1
    for r in (1, 5, 15):
        X, y = sample_xy(train, r, per_img)
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=300))
        model.fit(X, y)
        preds = np.concatenate([model.predict(patch_features(i, r)) for i, _ in test])
        res = score(yt, preds)
        res["confusion"] = confusion_matrix(yt, preds, labels=[0, 1, 2]).tolist()
        results[f"softmax_r{r}"] = res
        print(f"softmax r={r}:", res)
        if res["traversable_IoU"] > best_iou:
            best_iou, best = res["traversable_IoU"], (model, r)

    os.makedirs("results", exist_ok=True)
    json.dump(results, open("results/terrain_metrics.json", "w"), indent=2)
    joblib.dump({"model": best[0], "r": best[1]}, "results/terrain_model.joblib")
    print(f"Best window size: r={best[1]}")

    # ---- example pictures: image | ground truth | prediction ----
    colours = np.array([[60, 60, 60], [0, 200, 0], [230, 50, 50]], np.uint8)
    for k, (img, lab) in enumerate(test[:: max(1, len(test) // 5)][:5]):
        pred = predict_mask(best[0], best[1], img)
        row = np.hstack([img, colours[lab], colours[pred]])
        cv2.imwrite(f"results/example_{k}.png", cv2.cvtColor(row, cv2.COLOR_RGB2BGR))
    print("Saved metrics, model and example images to results/")