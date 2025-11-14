import os
import numpy as np
from glob import glob

ROOT = "/home/angelika/Desktop/Seoul/Intelligent Control/DATA_annotation/datasets/dataset6/labels"

def sort_polygon_clockwise(poly):
    poly = np.array(poly)
    cx = np.mean(poly[:, 0])
    cy = np.mean(poly[:, 1])

    angles = np.arctan2(poly[:, 1] - cy, poly[:, 0] - cx)
    return poly[np.argsort(angles)]

def fix_all_labels():
    txts = glob(os.path.join(ROOT, "**", "*.txt"), recursive=True)
    for path in txts:
        new_lines = []
        with open(path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 7:
                    continue

                cls, cx, cy, w, h = parts[:5]
                coords = list(map(float, parts[5:]))

                poly = []
                for i in range(0, len(coords), 2):
                    poly.append([coords[i], coords[i+1]])

                poly = sort_polygon_clockwise(poly)

                flat = " ".join(f"{x:.6f} {y:.6f}" for x, y in poly)
                new_lines.append(f"{cls} {cx} {cy} {w} {h} {flat}\n")

        with open(path, "w") as f:
            f.writelines(new_lines)

    print("✔ Все полигоны отсортированы!")

if __name__ == "__main__":
    fix_all_labels()
