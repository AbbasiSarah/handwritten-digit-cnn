"""Numerical gradient check of the NumPy CNN (python -m pytest tests)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cnn2d import DigitCNN  # noqa: E402


def test_gradients():
    rng = np.random.default_rng(3)
    net = DigitCNN(seed=1, dropout=0.0)
    for l in net.params:
        l.W = l.W.astype(np.float64); l.b = l.b.astype(np.float64)
    x = rng.random((3, 1, 28, 28))
    y = np.array([1, 7, 3])

    def loss():
        z = net.logits(x)
        z = z - z.max(1, keepdims=True)
        p = np.exp(z) / np.exp(z).sum(1, keepdims=True)
        return -np.log(p[np.arange(3), y]).mean()

    net.loss_and_grad(x, y)
    grads = [(l, l.dW.copy(), l.db.copy()) for l in net.params]
    eps = 1e-5
    for l, dW, db in grads:
        for arr, g in [(l.W, dW), (l.b, db)]:
            flat, gflat = arr.reshape(-1), g.reshape(-1)
            for i in rng.choice(flat.size, 4, replace=False):
                old = flat[i]
                flat[i] = old + eps; up = loss()
                flat[i] = old - eps; dn = loss()
                flat[i] = old
                num = (up - dn) / (2 * eps)
                assert abs(num - gflat[i]) < 1e-5 + 1e-3 * abs(num), (type(l).__name__, num, gflat[i])


if __name__ == "__main__":
    test_gradients()
    print("gradient check passed")
