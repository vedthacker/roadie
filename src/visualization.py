"""All plots. Muted, colour-blind-safe palette; every figure answers an analytical question."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import config  # noqa: E402
from .hog_features import hog_visualization  # noqa: E402

# Okabe-Ito based categorical palette (fixed order: raw/baseline, KNN, tree, SVM, tuned SVM)
PALETTE = ["#6B7280", "#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00"]
INK = "#1F2937"
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 150, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": "#9CA3AF", "axes.labelcolor": INK,
                     "text.color": INK, "xtick.color": INK, "ytick.color": INK, "axes.grid": True,
                     "grid.color": "#E5E7EB", "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "font.size": 9})


def _tag(title: str) -> str:
    return title + ("" if config.RUN_MODE == "full_dataset" else "  [DEVELOPMENT SUBSET]")


def _save(fig, path):
    path = config.FIGURES_DIR / path if isinstance(path, str) else path
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_class_distribution(counts: dict, path="class_distribution.png"):
    ids = sorted(counts)
    vals = [counts[i] for i in ids]
    med = float(np.median(vals))
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.bar([str(i) for i in ids], vals, color=PALETTE[1], width=0.7)
    ax.axhline(med, color=PALETTE[6], lw=1.2, ls="--")
    ax.text(len(ids) - 0.5, med, f" median = {med:.0f}", color=INK, va="bottom", ha="right")
    ax.set_xlabel("ClassId"); ax.set_ylabel("Training images")
    ax.set_title(_tag(f"Class distribution (max/min = {max(vals) / min(vals):.1f}x)"), loc="left")
    return _save(fig, path)


def plot_sample_images(images, labels, path="sample_images.png", per_class=1, max_classes=43, seed=config.SEED):
    rng = np.random.RandomState(seed)
    classes = sorted(set(int(c) for c in labels))[:max_classes]
    cols = 9
    rows = int(np.ceil(len(classes) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.25, rows * 1.5))
    for ax in np.ravel(axes):
        ax.axis("off")
    for ax, c in zip(np.ravel(axes), classes):
        idx = np.where(labels == c)[0]
        ax.imshow(images[rng.choice(idx)], cmap="gray"); ax.set_title(str(c), fontsize=8)
    fig.suptitle(_tag("One preprocessed example per class (grayscale, ROI crop, 32x32)"), x=0.01, ha="left")
    return _save(fig, path)


def plot_image_dimensions(widths, heights, path="image_size_distribution.png"):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    axes[0].hist(widths, bins=40, color=PALETTE[1]); axes[0].set_xlabel("Original width (px)")
    axes[0].set_ylabel("Images")
    axes[1].scatter(widths, heights, s=3, alpha=.3, color=PALETTE[3])
    axes[1].set_xlabel("Width (px)"); axes[1].set_ylabel("Height (px)")
    fig.suptitle(_tag("Original image sizes (varies a lot -> fixed resize needed)"), x=0.01, ha="left")
    return _save(fig, path)


def plot_hog_visualization(gray_imgs, titles, path="hog_visualization.png"):
    n = len(gray_imgs)
    fig, axes = plt.subplots(2, n, figsize=(n * 2, 4.4))
    for i, (im, t) in enumerate(zip(gray_imgs, titles)):
        _, vis = hog_visualization(im)
        axes[0, i].imshow(im, cmap="gray"); axes[0, i].set_title(t, fontsize=7); axes[0, i].axis("off")
        axes[1, i].imshow(vis, cmap="magma"); axes[1, i].axis("off")
    axes[0, 0].text(-0.1, 0.5, "input", transform=axes[0, 0].transAxes, rotation=90, va="center", ha="right")
    axes[1, 0].text(-0.1, 0.5, "HOG", transform=axes[1, 0].transAxes, rotation=90, va="center", ha="right")
    fig.suptitle("HOG visualisation (orientation histograms per cell)", x=0.01, ha="left")
    return _save(fig, path)


def plot_raw_vs_hog(X_raw, X_hog, y, path="raw_vs_hog_features.png", n_classes=6, seed=config.SEED):
    """2-D PCA of a few classes under both representations (PCA fitted on the plotted training data only)."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    rng = np.random.RandomState(seed)
    classes = sorted(set(y.tolist()))[:n_classes]
    idx = np.concatenate([rng.choice(np.where(y == c)[0], min(150, (y == c).sum()), replace=False) for c in classes])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, X, name in zip(axes, (X_raw, X_hog), (f"Raw pixels ({X_raw.shape[1]}-d)", f"HOG ({X_hog.shape[1]}-d)")):
        Z = PCA(2, random_state=seed).fit_transform(StandardScaler().fit_transform(X[idx]))
        for k, c in enumerate(classes):
            m = y[idx] == c
            ax.scatter(Z[m, 0], Z[m, 1], s=8, alpha=.7, color=PALETTE[k % len(PALETTE)], label=f"class {c}")
        ax.set_title(name, loc="left"); ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    axes[1].legend(fontsize=7, frameon=False, loc="best")
    fig.suptitle(_tag("Do the representations separate classes? (PCA of the training subset)"), x=0.01, ha="left")
    return _save(fig, path)


def plot_model_comparison(df, metric_cols=("accuracy", "f1_macro"), path="model_performance_comparison.png",
                          title="Model comparison"):
    models = df["model"].tolist()
    x = np.arange(len(models))
    w = 0.8 / len(metric_cols)
    fig, ax = plt.subplots(figsize=(max(7, 1.5 * len(models)), 4))
    for k, m in enumerate(metric_cols):
        ax.bar(x + k * w - 0.4 + w / 2, df[m], w * 0.92, label=m, color=PALETTE[(k * 3 + 1) % len(PALETTE)])
        for xi, v in zip(x, df[m]):
            ax.text(xi + k * w - 0.4 + w / 2, v + 0.005, f"{v:.3f}", ha="center", fontsize=6.5)
    ax.set_xticks(x); ax.set_xticklabels([m.replace(" ", "\n", 1) for m in models], fontsize=8)
    ax.set_ylim(0, 1.08); ax.set_ylabel("Score"); ax.legend(frameon=False)
    ax.set_title(_tag(title), loc="left")
    return _save(fig, path)


def plot_confusion_matrix(cm, labels, path, title="Confusion matrix", normalize=True):
    mat = cm.astype(float)
    if normalize:
        mat = mat / np.maximum(mat.sum(axis=1, keepdims=True), 1)
    n = len(labels)
    fig, ax = plt.subplots(figsize=(8.5 if n > 20 else 5.5, 7.5 if n > 20 else 4.8))
    im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    fs = 5.5 if n > 20 else 8
    ax.set_xticklabels(labels, rotation=90, fontsize=fs); ax.set_yticklabels(labels, fontsize=fs)
    ax.set_xlabel("Predicted ClassId"); ax.set_ylabel("True ClassId"); ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.04, label="Row-normalised share" if normalize else "Count")
    ax.set_title(_tag(title), loc="left", fontsize=9)
    return _save(fig, path)


def plot_per_class_f1(table, path="per_class_f1.png", title="Per-class F1"):
    t = table.sort_values("f1")
    fig, ax = plt.subplots(figsize=(7, max(4, len(t) * 0.18)))
    ax.barh([str(c) for c in t["class_id"]], t["f1"], color=PALETTE[3])
    ax.set_xlim(0, 1.02); ax.set_xlabel("F1"); ax.set_ylabel("ClassId"); ax.tick_params(axis="y", labelsize=6)
    ax.set_title(_tag(title), loc="left")
    return _save(fig, path)


def plot_misclassified(images, y_true, y_pred, path="misclassified_examples.png", n=24, seed=config.SEED):
    wrong = np.where(y_true != y_pred)[0]
    if len(wrong) == 0:
        fig, ax = plt.subplots(figsize=(4, 1.5)); ax.axis("off"); ax.text(.5, .5, "No misclassifications", ha="center")
        return _save(fig, path)
    rng = np.random.RandomState(seed)
    pick = rng.choice(wrong, min(n, len(wrong)), replace=False)
    cols = 8
    rows = int(np.ceil(len(pick) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.8))
    for ax in np.ravel(axes):
        ax.axis("off")
    for ax, i in zip(np.ravel(axes), pick):
        ax.imshow(images[i], cmap="gray"); ax.set_title(f"true {y_true[i]}\npred {y_pred[i]}", fontsize=7)
    fig.suptitle(_tag(f"Random misclassified test examples ({len(wrong)} errors in total)"), x=0.01, ha="left")
    return _save(fig, path)


def plot_tuning_heatmap(cv_rows, path="grid_search_svm.png"):
    """Mean CV F1 for rbf SVM over (C, gamma)."""
    import pandas as pd
    df = pd.DataFrame(cv_rows)
    df = df[df["param_kernel"] == "rbf"] if "param_kernel" in df.columns else df
    if not {"param_C", "param_gamma"} <= set(df.columns):
        return None
    piv = df.pivot_table(index="param_C", columns="param_gamma", values="mean_test_score", aggfunc="max")
    piv = piv.reindex(sorted(piv.index, key=float))
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    im = ax.imshow(piv.values, cmap="Blues", aspect="auto")
    ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels(piv.columns); ax.set_yticks(range(piv.shape[0]))
    ax.set_yticklabels(piv.index); ax.set_xlabel("gamma"); ax.set_ylabel("C"); ax.grid(False)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"{piv.values[i, j]:.3f}", ha="center", va="center", fontsize=8,
                    color="white" if piv.values[i, j] > np.nanmean(piv.values) else INK)
    fig.colorbar(im, ax=ax, label="mean CV macro-F1")
    ax.set_title(_tag("SVM GridSearchCV (RBF)"), loc="left", fontsize=9)
    return _save(fig, path)


def plot_brightness_contrast(mean_intensity, contrast, path="eda_brightness_contrast.png"):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    axes[0].hist(mean_intensity, bins=50, color=PALETTE[2]); axes[0].set_xlabel("Mean grayscale intensity (0-255)")
    axes[0].set_ylabel("Images"); axes[0].axvline(40, color=INK, ls="--", lw=1)
    axes[1].hist(contrast, bins=50, color=PALETTE[1]); axes[1].set_xlabel("Contrast = std of grayscale intensity")
    axes[1].axvline(15, color=INK, ls="--", lw=1)
    fig.suptitle(_tag("Illumination and contrast vary strongly (dashed = 'dark' / 'low-contrast' thresholds)"), x=0.01, ha="left", fontsize=9)
    return _save(fig, path)


def plot_class_mean_images(images, labels, path="eda_class_mean_images.png"):
    classes = sorted(set(int(c) for c in labels))
    cols = 9
    rows = int(np.ceil(len(classes) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.2, rows * 1.4))
    for ax in np.ravel(axes):
        ax.axis("off")
    for ax, c in zip(np.ravel(axes), classes):
        ax.imshow(images[labels == c].mean(axis=0), cmap="gray", vmin=0, vmax=255); ax.set_title(str(c), fontsize=7)
    fig.suptitle(_tag("Class-mean images (training split)"), x=0.01, ha="left")
    return _save(fig, path)


def plot_similar_pairs(images, labels, pairs, path="eda_similar_classes.png"):
    n = len(pairs)
    fig, axes = plt.subplots(2, n, figsize=(n * 1.9, 4.4))
    axes = np.atleast_2d(axes).reshape(2, n)
    for k, pr in enumerate(pairs):
        for r, key in enumerate(("class_a", "class_b")):
            c = pr[key]
            axes[r, k].imshow(images[labels == c].mean(axis=0), cmap="gray", vmin=0, vmax=255)
            axes[r, k].set_title(f"{c}: {config.class_name(c)[:22]}", fontsize=6); axes[r, k].axis("off")
        axes[0, k].text(0.5, 1.28, f"cos={pr['cosine_similarity']:.3f}", transform=axes[0, k].transAxes, ha="center", fontsize=7)
    fig.suptitle(_tag("Most similar class pairs by mean-HOG cosine similarity"), x=0.01, ha="left")
    return _save(fig, path)


def plot_size_by_class(widths_by_class: dict, path="eda_size_by_class.png"):
    ids = sorted(widths_by_class)
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.boxplot([widths_by_class[i] for i in ids], tick_labels=[str(i) for i in ids], showfliers=False,
               boxprops=dict(color=PALETTE[1]), medianprops=dict(color=PALETTE[6]))
    ax.tick_params(axis="x", labelsize=6); ax.set_xlabel("ClassId"); ax.set_ylabel("Original image width (px)")
    ax.set_title(_tag("Original image width per class"), loc="left")
    return _save(fig, path)
