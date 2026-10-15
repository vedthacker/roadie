"""Tiny SYNTHETIC stand-in for GTSRB (same folder layout / CSV format) used ONLY by tests and smoke runs.

It exists so the code can be exercised where the real dataset cannot be downloaded. Nothing produced
from it may ever be reported as a result of the project.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

HEADER = "Filename;Width;Height;Roi.X1;Roi.Y1;Roi.X2;Roi.Y2;ClassId"
COLORS = [(200, 30, 30), (30, 60, 200), (230, 200, 30), (240, 240, 240)]


def _sign(cls, size, rng, brightness):
    img = Image.new("RGB", (size, size), tuple(int(v) for v in rng.randint(60, 120, 3)))
    d = ImageDraw.Draw(img)
    m = size // 10
    box = [m, m, size - m, size - m]
    col = tuple(int(c * brightness) for c in COLORS[cls % 4])
    shape = (cls // 4) % 3
    if shape == 0:
        d.ellipse(box, fill=col)
    elif shape == 1:
        d.polygon([(size // 2, m), (m, size - m), (size - m, size - m)], fill=col)
    else:
        d.rectangle(box, fill=col)
    bars = 1 + cls % 3
    for b in range(bars):  # class-specific inner pattern -> gradients differ between classes
        x = size // 2 + (b - (bars - 1) / 2) * size // 6
        d.line([(x, size // 3), (x, 2 * size // 3)], fill=(10, 10, 10), width=max(1, size // 14))
    arr = np.asarray(img).astype(float) + rng.normal(0, 6, (size, size, 3))
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def make_synthetic_gtsrb(root: Path, n_classes=8, tracks=10, frames=6, n_test=160, seed=0) -> Path:
    """Create root/GTSRB/{Final_Training/Images, Final_Test/Images + GT-final_test.csv}."""
    rng = np.random.RandomState(seed)
    g = Path(root) / "GTSRB"
    for c in range(n_classes):
        cdir = g / "Final_Training" / "Images" / f"{c:05d}"
        cdir.mkdir(parents=True, exist_ok=True)
        rows = []
        n_tracks = tracks if c % 3 else tracks * 3  # deliberately imbalanced
        for t in range(n_tracks):
            base = int(rng.randint(28, 64))
            for f in range(frames):
                size = int(base * rng.uniform(0.9, 1.1))
                fn = f"{t:05d}_{f:05d}.ppm"
                _sign(c, size, rng, rng.uniform(0.6, 1.0)).save(cdir / fn)
                rows.append(f"{fn};{size};{size};{size // 10};{size // 10};{size - size // 10};{size - size // 10};{c}")
        (cdir / f"GT-{c:05d}.csv").write_text(HEADER + "\n" + "\n".join(rows) + "\n")
    tdir = g / "Final_Test" / "Images"
    tdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n_test):
        c = int(rng.randint(0, n_classes))
        size = int(rng.randint(28, 64))
        fn = f"{i:05d}.ppm"
        _sign(c, size, rng, rng.uniform(0.6, 1.0)).save(tdir / fn)
        rows.append(f"{fn};{size};{size};{size // 10};{size // 10};{size - size // 10};{size - size // 10};{c}")
    (g / "Final_Test" / "GT-final_test.csv").write_text(HEADER + "\n" + "\n".join(rows) + "\n")
    return g


if __name__ == "__main__":
    import sys
    make_synthetic_gtsrb(Path(sys.argv[1]))
