"""Person following: YOLOv8n + ByteTrack for the person, terrain model for safety.

Run from the project root:
    python src/follow.py 0                              # webcam
    python src/follow.py videos/clip1_walk.mp4          # video file
    python src/follow.py videos/clip1_walk.mp4 --auto-lock --no-show

Keys: L = lock target (person closest to image centre), Q = quit.
"""
import argparse
import csv
import os
import time

import cv2
import joblib
import numpy as np
from ultralytics import YOLO

from terrain import predict_mask

# Tunable thresholds (report the final values in the write-up)
DEAD_ZONE = 0.2        # |offset| below this -> go straight
TOO_CLOSE = 0.55       # box height / frame height above this -> STOP
SAFE_FRACTION = 0.5    # min traversable share of the strip ahead
LOST_FRAMES = 15       # frames without the target before STOP
MIN_REACQ_HEIGHT = 0.15  # ignore tiny, distant detections when re-acquiring (box height / frame height)
TERRAIN_EVERY = 5      # run terrain model every N frames


def pick_center_person(boxes, ids, width):
    """Return the track id of the person whose box centre is nearest the image centre."""
    best, best_d = None, 1e9
    for (x1, y1, x2, y2), tid in zip(boxes, ids):
        d = abs((x1 + x2) / 2 - width / 2)
        if d < best_d:
            best, best_d = int(tid), d
    return best


def ground_ahead_fraction(mask):
    """Share of traversable pixels in the bottom-centre strip of the 112x200 mask."""
    strip = mask[84:, 66:134]
    return float((strip == 1).mean())


def hazard_under_feet(mask, box, w, h):
    """True if the area just below the target's feet is mostly hazard (class 2)."""
    x1, y1, x2, y2 = box
    mh, mw = mask.shape
    cx = int(((x1 + x2) / 2) / w * mw)
    fy = int(min(y2 / h * mh, mh - 1))
    patch = mask[fy:min(fy + 4, mh), max(cx - 6, 0):min(cx + 6, mw)]
    return patch.size > 0 and float((patch == 2).mean()) > 0.5


def decide(locked, lost, safe, hazard, close, offset):
    """Safety-first decision order. Returns (command, reason)."""
    if not locked:
        return "STOP", "no target locked"
    if lost:
        return "STOP", "target lost"
    if not safe:
        return "STOP", "unsafe ground ahead"
    if hazard:
        return "STOP", "hazard under target"
    if close:
        return "STOP", "close enough"
    if offset < -DEAD_ZONE:
        return "LEFT", "target on left"
    if offset > DEAD_ZONE:
        return "RIGHT", "target on right"
    return "FORWARD", "target centred"


def overlay_mask(frame, mask):
    colors = np.array([[90, 90, 90], [0, 200, 0], [0, 0, 220]], dtype=np.uint8)  # BGR
    small = cv2.resize(colors[mask], (300, 168), interpolation=cv2.INTER_NEAREST)
    frame[10:178, 10:310] = small
    cv2.rectangle(frame, (10, 10), (310, 178), (255, 255, 255), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", help="0 for webcam, or path to a video file")
    ap.add_argument("--auto-lock", action="store_true", help="lock on first frame with a person")
    ap.add_argument("--no-show", action="store_true", help="do not open a window")
    ap.add_argument("--out", default="results/demo.mp4")
    ap.add_argument("--log", default="results/follow_log.csv")
    args = ap.parse_args()

    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open source: {args.source}")

    tm = joblib.load("results/terrain_model.joblib")
    det = YOLO("yolov8n.pt")  # downloads on first run (needs internet)

    os.makedirs("results", exist_ok=True)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"),
                             cap.get(cv2.CAP_PROP_FPS) or 20, (W, H))
    logf = open(args.log, "w", newline="")
    log = csv.writer(logf)
    log.writerow(["frame", "target_found", "command", "reason", "fps"])

    lock_id, missing, mask, fps, n = None, 0, np.zeros((112, 200), dtype=np.uint8), 0.0, 0
    last_cx = None  # last known x-centre of the target, used to re-acquire it
    safe = True

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.time()

        res = det.track(frame, persist=True, tracker="bytetrack.yaml",
                        classes=[0], verbose=False)[0]
        boxes, ids = [], []
        if res.boxes is not None and res.boxes.id is not None:
            boxes = res.boxes.xyxy.cpu().numpy()
            ids = res.boxes.id.cpu().numpy().astype(int)

        if lock_id is None and args.auto_lock and len(ids):
            lock_id = pick_center_person(boxes, ids, W)

        if n % TERRAIN_EVERY == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mask = predict_mask(tm["model"], tm["r"], rgb)
            safe = ground_ahead_fraction(mask) >= SAFE_FRACTION

        # Re-acquire: tracker gave the target a new ID after occlusion or leaving the frame.
        # Re-lock onto the person nearest to where the target was last seen.
        if (lock_id is not None and lock_id not in ids and missing >= LOST_FRAMES
                and len(ids) and last_cx is not None):
            heights = (boxes[:, 3] - boxes[:, 1]) / H
            ok_idx = np.where(heights >= MIN_REACQ_HEIGHT)[0]
            if len(ok_idx):
                centres = (boxes[ok_idx, 0] + boxes[ok_idx, 2]) / 2
                lock_id = int(ids[ok_idx[int(np.argmin(np.abs(centres - last_cx)))]])
                missing = 0

        target = None
        for b, tid in zip(boxes, ids):
            is_t = lock_id is not None and tid == lock_id
            if is_t:
                target = b
                last_cx = (b[0] + b[2]) / 2
            x1, y1, x2, y2 = map(int, b)
            color = (0, 255, 0) if is_t else (150, 150, 150)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID {tid}", (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        missing = 0 if target is not None else missing + 1
        lost = lock_id is not None and missing > LOST_FRAMES
        if target is not None:
            offset = ((target[0] + target[2]) / 2 - W / 2) / (W / 2)
            close = (target[3] - target[1]) / H > TOO_CLOSE
            hazard = hazard_under_feet(mask, target, W, H)
        else:
            offset, close, hazard = 0.0, False, False
        # while the target is briefly missing (<= LOST_FRAMES), keep the last decision cautious
        if lock_id is not None and target is None and not lost:
            cmd, reason = "STOP", "target not visible"
        else:
            cmd, reason = decide(lock_id is not None, lost, safe, hazard, close, offset)

        dt = time.time() - t0
        fps = 0.9 * fps + 0.1 / max(dt, 1e-6) if fps else 1 / max(dt, 1e-6)

        overlay_mask(frame, mask)
        cv2.putText(frame, f"{cmd} ({reason})", (10, H - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.putText(frame, f"FPS {fps:.1f}  lock {lock_id}", (10, H - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        writer.write(frame)
        log.writerow([n, int(target is not None), cmd, reason, f"{fps:.1f}"])
        n += 1

        if not args.no_show:
            cv2.imshow("follow", frame)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            if k == ord("l") and len(ids):
                lock_id = pick_center_person(boxes, ids, W)
                missing = 0

    cap.release()
    writer.release()
    logf.close()
    cv2.destroyAllWindows()
    print(f"Done. {n} frames. Saved {args.out} and {args.log}")


if __name__ == "__main__":
    main()