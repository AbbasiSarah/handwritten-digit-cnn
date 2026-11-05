"""A small 2-D convolutional neural network written from scratch in NumPy.

Architecture for 28x28 grey-scale digits::

    Conv 3x3, 1 -> 8   + ReLU + MaxPool 2x2     28x28 -> 14x14
    Conv 3x3, 8 -> 16  + ReLU + MaxPool 2x2     14x14 -> 7x7
    Flatten (784) -> Dense 64 + ReLU -> Dropout -> Dense 10 -> softmax

Every layer implements forward() and backward(); tests/test_cnn2d.py checks the
gradients numerically. Convolution uses the im2col trick so it runs as one
matrix multiplication.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

DATA = Path(__file__).resolve().parents[1] / "data"


# ------------------------------------------------------------------ data
def load_mnist(kind="train"):
    prefix = "train" if kind == "train" else "t10k"
    with gzip.open(DATA / f"{prefix}-images-idx3-ubyte.gz") as f:
        X = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 1, 28, 28)
    with gzip.open(DATA / f"{prefix}-labels-idx1-ubyte.gz") as f:
        y = np.frombuffer(f.read(), np.uint8, offset=8)
    return X.astype(np.float32) / 255.0, y.astype(np.int64)


# ---------------------------------------------------------------- layers
class Conv2D:
    """3x3 'same' convolution."""

    def __init__(self, cin, cout, rng, k=3):
        self.k = k
        self.W = rng.normal(0, np.sqrt(2 / (cin * k * k)), (cin * k * k, cout)).astype(np.float32)
        self.b = np.zeros(cout, np.float32)

    def forward(self, x, train=True):
        n, c, h, w = x.shape
        p = self.k // 2
        xp = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)))
        win = sliding_window_view(xp, (self.k, self.k), axis=(2, 3))      # n,c,h,w,k,k
        cols = win.transpose(0, 2, 3, 1, 4, 5).reshape(n * h * w, c * self.k * self.k)
        self.cache = (cols, x.shape)
        out = cols @ self.W + self.b                                       # n*h*w, cout
        return out.reshape(n, h, w, -1).transpose(0, 3, 1, 2)

    def backward(self, dout):
        cols, (n, c, h, w) = self.cache
        d = dout.transpose(0, 2, 3, 1).reshape(n * h * w, -1)
        self.dW, self.db = cols.T @ d, d.sum(0)
        dcols = (d @ self.W.T).reshape(n, h, w, c, self.k, self.k)
        p = self.k // 2
        dxp = np.zeros((n, c, h + 2 * p, w + 2 * p), np.float32)
        for i in range(self.k):
            for j in range(self.k):
                dxp[:, :, i:i + h, j:j + w] += dcols[:, :, :, :, i, j].transpose(0, 3, 1, 2)
        return dxp[:, :, p:p + h, p:p + w]


class ReLU:
    def forward(self, x, train=True):
        self.m = x > 0
        return x * self.m

    def backward(self, d):
        return d * self.m


class MaxPool2:
    def forward(self, x, train=True):
        n, c, h, w = x.shape
        r = x.reshape(n, c, h // 2, 2, w // 2, 2)
        out = r.max(axis=(3, 5))
        mask = r == out[:, :, :, None, :, None]
        self.mask = mask / mask.sum(axis=(3, 5), keepdims=True)
        self.shape = x.shape
        return out

    def backward(self, d):
        return (self.mask * d[:, :, :, None, :, None]).reshape(self.shape)


class Flatten:
    def forward(self, x, train=True):
        self.shape = x.shape
        return x.reshape(len(x), -1)

    def backward(self, d):
        return d.reshape(self.shape)


class Dense:
    def __init__(self, nin, nout, rng):
        self.W = rng.normal(0, np.sqrt(2 / nin), (nin, nout)).astype(np.float32)
        self.b = np.zeros(nout, np.float32)

    def forward(self, x, train=True):
        self.x = x
        return x @ self.W + self.b

    def backward(self, d):
        self.dW, self.db = self.x.T @ d, d.sum(0)
        return d @ self.W.T


class Dropout:
    def __init__(self, p, rng):
        self.p, self.rng = p, rng

    def forward(self, x, train=True):
        if not train:
            self.m = 1.0
            return x
        self.m = (self.rng.random(x.shape) > self.p).astype(np.float32) / (1 - self.p)
        return x * self.m

    def backward(self, d):
        return d * self.m


# ----------------------------------------------------------------- model
class DigitCNN:
    def __init__(self, seed=0, dropout=0.25):
        rng = np.random.default_rng(seed)
        self.rng = rng
        self.layers = [Conv2D(1, 8, rng), ReLU(), MaxPool2(),
                       Conv2D(8, 16, rng), ReLU(), MaxPool2(),
                       Flatten(), Dense(784, 64, rng), ReLU(), Dropout(dropout, rng), Dense(64, 10, rng)]

    @property
    def params(self):
        return [l for l in self.layers if hasattr(l, "W")]

    def logits(self, x, train=False):
        for l in self.layers:
            x = l.forward(x, train)
        return x

    def loss_and_grad(self, x, y):
        z = self.logits(x, train=True)
        z = z - z.max(1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(1, keepdims=True)
        loss = -np.log(p[np.arange(len(y)), y] + 1e-12).mean()
        d = p
        d[np.arange(len(y)), y] -= 1
        d /= len(y)
        for l in reversed(self.layers):
            d = l.backward(d)
        return loss

    def predict_proba(self, x, batch=1000):
        out = []
        for i in range(0, len(x), batch):
            z = self.logits(x[i:i + batch])
            z = np.exp(z - z.max(1, keepdims=True))
            out.append(z / z.sum(1, keepdims=True))
        return np.concatenate(out)

    def fit(self, X, y, epochs=4, batch=64, lr=1e-3, Xval=None, yval=None, augment=True, log=print):
        b1, b2, t = 0.9, 0.999, 0
        m = {id(l): [np.zeros_like(l.W), np.zeros_like(l.b)] for l in self.params}
        v = {id(l): [np.zeros_like(l.W), np.zeros_like(l.b)] for l in self.params}
        self.history = []
        for ep in range(epochs):
            idx, tot = self.rng.permutation(len(X)), 0.0
            for s in range(0, len(X), batch):
                b = idx[s:s + batch]
                xb = shift_augment(X[b], self.rng) if augment else X[b]
                tot += self.loss_and_grad(xb, y[b]) * len(b)
                t += 1
                for l in self.params:
                    for k, (p, g) in enumerate([(l.W, l.dW), (l.b, l.db)]):
                        mm, vv = m[id(l)][k], v[id(l)][k]
                        mm *= b1; mm += (1 - b1) * g
                        vv *= b2; vv += (1 - b2) * g * g
                        p -= lr * (mm / (1 - b1 ** t)) / (np.sqrt(vv / (1 - b2 ** t)) + 1e-8)
            lr *= 0.75  # step decay
            rec = {"epoch": ep + 1, "train_loss": tot / len(X)}
            if Xval is not None:
                rec["val_acc"] = float((self.predict_proba(Xval).argmax(1) == yval).mean())
            self.history.append(rec)
            log(rec)
        return self

    def save(self, path):
        np.savez(path, **{f"{i}_{k}": getattr(l, k) for i, l in enumerate(self.layers)
                          if hasattr(l, "W") for k in ("W", "b")})

    def load(self, path):
        w = np.load(path)
        for i, l in enumerate(self.layers):
            if hasattr(l, "W"):
                l.W, l.b = w[f"{i}_W"], w[f"{i}_b"]
        return self


def shift_augment(x, rng, max_shift=2):
    """Randomly shift each image by up to 2 pixels: cheap robustness to off-centre digits."""
    out = np.zeros_like(x)
    dx, dy = rng.integers(-max_shift, max_shift + 1, 2)
    h = x.shape[2]
    out[:, :, max(dy, 0):h + min(dy, 0), max(dx, 0):h + min(dx, 0)] = \
        x[:, :, max(-dy, 0):h + min(-dy, 0), max(-dx, 0):h + min(-dx, 0)]
    return out
