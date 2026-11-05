"""Train the NumPy CNN on MNIST, evaluate on the 10,000 test images, export the web demo weights.

    python train.py            # ~5 minutes on a laptop CPU
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))
from cnn2d import DigitCNN, load_mnist  # noqa: E402

FIG = Path("results/figures")
ACC = "#1565C0"


def main(epochs=8):
    FIG.mkdir(parents=True, exist_ok=True)
    X, y = load_mnist("train")
    Xt, yt = load_mnist("test")
    Xtr, ytr, Xval, yval = X[:55000], y[:55000], X[55000:], y[55000:]   # last 5k = validation

    net = DigitCNN(seed=0).fit(Xtr, ytr, epochs=epochs, Xval=Xval, yval=yval,
                               log=lambda r: print({k: round(float(v), 4) for k, v in r.items()}))
    Path("models").mkdir(exist_ok=True)
    net.save("models/digit_cnn.npz")

    p = net.predict_proba(Xt)
    pred = p.argmax(1)
    acc = float((pred == yt).mean())
    cm = np.zeros((10, 10), int)
    np.add.at(cm, (yt, pred), 1)
    per_class = {str(d): float(cm[d, d] / cm[d].sum()) for d in range(10)}
    confusions = sorted(((int(cm[i, j]), i, j) for i in range(10) for j in range(10) if i != j), reverse=True)[:5]
    n_params = sum(l.W.size + l.b.size for l in net.params)
    report = {"test_accuracy": acc, "test_errors": int((pred != yt).sum()), "n_test": len(yt),
              "parameters": int(n_params), "per_class_accuracy": per_class,
              "top_confusions": [{"true": i, "predicted": j, "count": c} for c, i, j in confusions],
              "history": [{k: float(v) for k, v in h.items()} for h in net.history]}
    Path("results/metrics.json").write_text(json.dumps(report, indent=2))
    print(f"TEST accuracy {acc:.4f} ({report['test_errors']} errors / 10,000), {n_params:,} parameters")

    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    # training curve
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ep = [h["epoch"] for h in net.history]
    ax.plot(ep, [100 * h["val_acc"] for h in net.history], "o-", color=ACC, lw=2)
    ax.set(xlabel="Epoch", ylabel="Validation accuracy (%)", title="Learning curve (5,000 held-out images)")
    fig.tight_layout(); fig.savefig(FIG / "learning_curve.png", dpi=150); plt.close(fig)
    # confusion matrix
    fig, ax = plt.subplots(figsize=(6, 5.4))
    off = cm.copy(); np.fill_diagonal(off, 0)
    ax.imshow(off, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        if i != j and v:
            ax.text(j, i, v, ha="center", va="center", fontsize=8, color="white" if v > off.max() / 2 else "black")
    ax.set(xticks=range(10), yticks=range(10), xlabel="Predicted", ylabel="True digit",
           title="Mistakes on the 10,000 test images (diagonal hidden)")
    fig.tight_layout(); fig.savefig(FIG / "confusion_matrix.png", dpi=150); plt.close(fig)
    # misclassified examples
    wrong = np.where(pred != yt)[0][:16]
    fig, axes = plt.subplots(2, 8, figsize=(10, 3))
    for ax, i in zip(axes.flat, wrong):
        ax.imshow(Xt[i, 0], cmap="gray_r"); ax.axis("off")
        ax.set_title(f"{yt[i]} → {pred[i]}", fontsize=10)
    fig.suptitle("Examples the CNN got wrong (true → predicted)")
    fig.tight_layout(); fig.savefig(FIG / "mistakes.png", dpi=150); plt.close(fig)
    # learned first-layer filters
    W = net.layers[0].W.reshape(1, 3, 3, 8)[0]
    fig, axes = plt.subplots(1, 8, figsize=(9, 1.6))
    for k, ax in enumerate(axes):
        ax.imshow(W[:, :, k], cmap="RdBu_r"); ax.axis("off")
    fig.suptitle("The 8 learned 3×3 filters of the first layer")
    fig.tight_layout(); fig.savefig(FIG / "filters.png", dpi=150); plt.close(fig)

    # weights for the browser demo
    js = {f"{i}_{k}": np.round(getattr(l, k).astype(float), 5).tolist()
          for i, l in enumerate(net.layers) if hasattr(l, "W") for k in ("W", "b")}
    Path("docs/weights.js").write_text("window.DIGIT_WEIGHTS=" + json.dumps(js, separators=(",", ":")) + ";\n")


if __name__ == "__main__":
    main()
