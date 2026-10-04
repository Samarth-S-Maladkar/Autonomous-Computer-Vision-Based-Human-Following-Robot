import sys, cv2, numpy as np
from terrain import find_pairs, load_colormap, TRAV, HAZ

root = sys.argv[1] if len(sys.argv) > 1 else "data/RUGD_sample"
cmap = load_colormap(root)
print("colormap:", cmap)
pairs = find_pairs(root)
print("pairs found:", len(pairs), "| sequences:", len({p[2] for p in pairs}))
print("names we expect but colormap lacks:", sorted((TRAV | HAZ) - set(cmap)))

ann = cv2.cvtColor(cv2.imread(pairs[0][1]), cv2.COLOR_BGR2RGB)
known = np.zeros(ann.shape[:2], bool)
for rgb in cmap.values():
    known |= np.all(ann == rgb, axis=-1)
print("first annotation shape:", ann.shape)
print("fraction of pixels matching a colormap colour (should be ~1.0):", round(known.mean(), 3))