#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figura del espacio de embeddings para el manuscrito.

Proyecta a dos dimensiones los embeddings de ejercicio aprendidos por cada
modelo y los colorea por tipo de ejercicio, que es el criterio semantico
primario de la evaluacion. Un panel por modelo, misma proyeccion y mismos
hiperparametros para todos, de modo que los paneles sean comparables entre si.

    pip install umap-learn
    python fkg_experiments.py ... --save-emb emb/ --save-seeds 0
    python make_umap_figure.py --emb emb/ --nodes fitkg_v3_output/nodes.csv \
        --edges fitkg_v3_output/edges.csv --lam 0 --out figura_embeddings

Salidas: <out>.svg (vectorial, para el manuscrito) y <out>.png a 600 dpi.

Sobre el color: se colorean los tres tipos de ejercicio mas frecuentes y el
resto se agrupa en una categoria neutra. La paleta esta validada para vision
con deficiencia de color en el caso mas exigente, en el que cualquier par de
colores puede aparecer contiguo, que es justamente lo que ocurre en un
diagrama de dispersion.
"""

import argparse
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]     # azul, naranja, aqua
OTHER = "#a8a69e"
INK = "#0b0b0b"
INK2 = "#52514e"
SURFACE = "#fcfcfb"

ORDER = ["mf", "distmult", "gcn", "gcn_distmult", "rgcn"]
TITLES = {
    "mf": "(a) Free embeddings",
    "distmult": "(b) DistMult",
    "gcn": "(c) GCN",
    "gcn_distmult": "(d) GCN + DistMult",
    "rgcn": "(e) R-GCN",
}


COLOR_BY = {"type": ("hasType", "Exercise type"),
            "force": ("hasForce", "Force direction"),
            "mechanic": ("hasMechanic", "Mechanic"),
            "muscle": ("targets", "Primary muscle"),
            "intensity": ("hasIntensity", "Intensity level")}


def exercise_labels(nodes_csv, edges_csv, exercise_index, rel):
    """Etiqueta cada ejercicio por el objeto de la relacion indicada."""
    nodes = pd.read_csv(nodes_csv)
    edges = pd.read_csv(edges_csv)
    name = dict(zip(nodes.node_id, nodes.name))
    sub = edges[edges.rel == rel]
    lab = {}
    for s_, t_ in zip(sub.source, sub.target):     # si hay varios, se toma el primero
        lab.setdefault(int(s_), int(t_))
    return [name.get(lab.get(int(i)), "Unknown") for i in exercise_index]


def project(X, method, seed, n_neighbors, min_dist, metric):
    if method == "umap":
        try:
            import umap
        except ImportError:
            raise SystemExit("Falta umap-learn:  pip install umap-learn\n"
                             "O usa --method pca para una vista rapida.")
        return umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric=metric,
                         random_state=seed).fit_transform(X)
    if method == "tsne":
        from sklearn.manifold import TSNE
        return TSNE(n_components=2, random_state=seed, init="pca",
                    metric=metric).fit_transform(X)
    from sklearn.decomposition import PCA
    return PCA(n_components=2, random_state=seed).fit_transform(X)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--emb", default="emb")
    p.add_argument("--nodes", default="fitkg_v3_output/nodes.csv")
    p.add_argument("--edges", default="fitkg_v3_output/edges.csv")
    p.add_argument("--lam", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--method", choices=["umap", "tsne", "pca"], default="umap")
    p.add_argument("--n-neighbors", dest="nn", type=int, default=15)
    p.add_argument("--min-dist", dest="md", type=float, default=0.1)
    p.add_argument("--metric", default="cosine")
    p.add_argument("--color-by", dest="color_by", default="type",
                   choices=list(COLOR_BY),
                   help="atributo que define el color: type, force, mechanic, muscle o "
                        "intensity. Util para descubrir por que criterio organiza el "
                        "espacio cada modelo.")
    p.add_argument("--out", default="figura_embeddings")
    a = p.parse_args()

    import torch
    files = {}
    for f in glob.glob(os.path.join(a.emb, "emb_*.pt")):
        m = re.match(r"emb_(.+)_lam([\d.eE+-]+)_seed(\d+)\.pt", os.path.basename(f))
        if m and abs(float(m.group(2)) - a.lam) < 1e-9 and int(m.group(3)) == a.seed:
            files[m.group(1)] = f
    models = [m for m in ORDER if m in files]
    if not models:
        raise SystemExit(f"No hay embeddings en '{a.emb}' para lambda={a.lam} y "
                         f"semilla={a.seed}. Corre el runner con --save-emb.")
    print("modelos encontrados:", ", ".join(models))

    first = torch.load(files[models[0]], map_location="cpu", weights_only=False)
    ex_idx = first["exercise_index"].numpy()
    rel, legend_title = COLOR_BY[a.color_by]
    labels = np.array(exercise_labels(a.nodes, a.edges, ex_idx, rel))
    counts = pd.Series(labels).value_counts()
    top = list(counts.index[:3])
    groups = [(t, SERIES[i]) for i, t in enumerate(top)] + [("Other", OTHER)]
    print(f"coloreando por {legend_title} ({rel}):", ", ".join(f"{t} ({counts[t]})" for t in top),
          f"| Other ({counts.iloc[3:].sum()})")

    ncol = 3
    nrow = int(np.ceil((len(models) + 1) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(9.2, 3.1 * nrow), facecolor=SURFACE)
    axes = np.atleast_1d(axes).ravel()

    for ax, model in zip(axes, models):
        d = torch.load(files[model], map_location="cpu", weights_only=False)
        X = d["z"].numpy()[ex_idx]
        Y = project(X, a.method, a.seed, a.nn, a.md, a.metric)
        ax.set_facecolor(SURFACE)
        for name, color in groups:
            sel = (labels == name) if name != "Other" else ~np.isin(labels, top)
            ax.scatter(Y[sel, 0], Y[sel, 1], s=11, c=color, linewidths=.35,
                       edgecolors=SURFACE, alpha=.9 if name != "Other" else .55,
                       zorder=3 if name != "Other" else 2)
        ax.set_title(TITLES.get(model, model), fontsize=9.5, color=INK, pad=6)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#dedcd6"); s.set_linewidth(.8)

    leg = axes[len(models)]
    leg.set_facecolor(SURFACE); leg.axis("off")
    for i, (name, color) in enumerate(groups):
        y = .78 - i * .16
        leg.scatter([.08], [y], s=46, c=color, edgecolors=SURFACE, linewidths=.5,
                    transform=leg.transAxes, clip_on=False)
        n = int(counts[name]) if name != "Other" else int(counts.iloc[3:].sum())
        leg.text(.18, y, f"{name}  ({n})", transform=leg.transAxes, fontsize=9.5,
                 color=INK, va="center")
    leg.text(.08, .95, legend_title, transform=leg.transAxes, fontsize=9.5,
             color=INK2, va="center", weight="bold")
    for ax in axes[len(models) + 1:]:
        ax.axis("off"); ax.set_facecolor(SURFACE)

    fig.tight_layout(pad=1.2)
    fig.savefig(a.out + ".svg", format="svg", facecolor=SURFACE, bbox_inches="tight")
    fig.savefig(a.out + ".png", dpi=600, facecolor=SURFACE, bbox_inches="tight")
    print(f"\n{a.out}.svg y {a.out}.png escritos")
    print("Pie de figura sugerido:")
    print(f"  Two-dimensional projection of the learned exercise representations "
          f"({a.method.upper()}, n_neighbors = {a.nn}, min_dist = {a.md}, "
          f"{a.metric} metric, random seed {a.seed}), coloured by "
          f"{legend_title.lower()}. "
          f"All panels share the same projection parameters.")


if __name__ == "__main__":
    main()