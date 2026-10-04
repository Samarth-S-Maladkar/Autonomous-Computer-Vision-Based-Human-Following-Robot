"""Summarise follow logs: average FPS, target-found rate, command counts.

Run from the project root:
    python src/analyze_log.py results/follow_log_clip1.csv results/follow_log_clip2.csv results/follow_log_clip3.csv
"""
import csv
import sys
from collections import Counter

for path in sys.argv[1:]:
    with open(path) as f:
        rows = list(csv.DictReader(f))
    n = len(rows)
    if n == 0:
        print(f"{path}: empty")
        continue
    fps = sum(float(r["fps"]) for r in rows) / n
    found = sum(int(r["target_found"]) for r in rows) / n
    cmds = Counter(r["command"] for r in rows)
    reasons = Counter(r["reason"] for r in rows)
    print(f"\n{path}")
    print(f"  frames: {n}")
    print(f"  average FPS: {fps:.1f}")
    print(f"  target-found rate: {found:.1%}")
    print(f"  commands: {dict(cmds)}")
    print(f"  reasons: {dict(reasons)}")
