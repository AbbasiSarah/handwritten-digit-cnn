"""Recognise a handwritten digit in an image file (any size; dark or light background).

    python predict.py my_digit.png
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent / "src"))
from cnn2d import DigitCNN  # noqa: E402


def to_mnist(path):
    """Grey-scale, white digit on black, cropped and scaled to 20 px, centred in 28x28 by centre of mass."""
    a = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255
    if a.mean() > 0.5:          # dark ink on light paper -> invert
        a = 1 - a
    ys, xs = np.where(a > 0.2)
    if not len(xs):
        sys.exit("No digit found in the image.")
    a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = a.shape
    s = 20 / max(h, w)
    a = np.asarray(Image.fromarray((a * 255).astype(np.uint8)).resize((max(1, round(w * s)), max(1, round(h * s))),
                                                                     Image.LANCZOS), dtype=np.float32) / 255
    out = np.zeros((28, 28), np.float32)
    h, w = a.shape
    y0, x0 = (28 - h) // 2, (28 - w) // 2
    out[y0:y0 + h, x0:x0 + w] = a
    cy, cx = np.array(np.nonzero(out)).mean(1) if out.sum() else (14, 14)
    out = np.roll(out, (int(round(13.5 - cy)), int(round(13.5 - cx))), axis=(0, 1))
    return out[None, None]


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    net = DigitCNN().load(Path(__file__).parent / "models" / "digit_cnn.npz")
    p = net.predict_proba(to_mnist(sys.argv[1]))[0]
    print(f"Digit: {p.argmax()}  ({p.max():.1%} sure)")
    for d in np.argsort(p)[::-1][:3]:
        print(f"  {d}: {p[d]:.1%}")
