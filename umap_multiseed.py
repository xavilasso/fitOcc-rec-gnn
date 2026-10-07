#!/usr/bin/env python3
"""
Figuras UMAP del espacio de representaciones, en calidad de publicacion.

Dos modos, elegidos automaticamente por el numero de semillas:

  * UNA semilla  -> rejilla de modelos, para el cuerpo del articulo
      python fig_umap.py --emb-dir emb_seeds --seeds 0 --out figura4.pdf

  * VARIAS       -> rejilla modelos x semillas, para el suplementario
      python fig_umap.py --emb-dir emb_seeds --seeds 0 1 2 --out figuraS3.pdf

Cada corrida escribe el PDF vectorial y, al lado, un PNG a 600 dpi (lo que MDPI
recomienda como minimo). Pasa --tiff si la revista pide TIFF.

Depende de: torch (solo para leer los .pt), numpy, pandas, scikit-learn,
umap-learn, matplotlib.
"""
import argparse, json, math, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import silhouette_score

# Fuentes incrustadas como TrueType: evita el texto en curvas y los problemas
# de maquetacion con PDF/EPS.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["svg.fonttype"] = "none"

LABEL = {"mf": "Free embeddings", "distmult": "DistMult", "transe": "TransE",
         "complex": "ComplEx", "gcn": "GCN", "gcn_distmult": "GCN + DistMult",
         "rgcn": "R-GCN"}
# dos tonos distinguibles tambien en escala de grises, gris para static y
# gris claro para los ejercicios sin atributo force declarado
COLORS = {"push": "#1f6feb", "pull": "#d4891f", "static": "#6e7781",
          "unspecified": "#d0d7de"}
LEVELS = ["push", "pull", "static", "unspecified"]


def force_labels(nodes_csv, exercise_index):
    """Direccion de fuerza por ejercicio, leida de attrs_json de nodes.csv."""
    df = pd.read_csv(nodes_csv)
    key = "node_id" if "node_id" in df.columns else df.columns[0]
    by_id = {int(r[key]): r for _, r in df.iterrows()}
    out = []
    for nid in exercise_index:
        attrs = by_id[int(nid)].get("attrs_json")
        f = None
        if isinstance(attrs, str) and attrs.strip().startswith("{"):
            try:
                f = json.loads(attrs).get("force")
            except json.JSONDecodeError:
                f = None
        out.append((f or "unspecified").lower())
    return np.array(out)


def load_emb(emb_dir, model, seed, lam=0.0):
    import torch
    path = os.path.join(emb_dir, f"emb_{model}_lam{lam:g}_seed{seed}.pt")
    if not os.path.exists(path):
        raise SystemExit(f"No existe {path}. Corriste --save-emb con --save-seeds {seed}?")
    d = torch.load(path, map_location="cpu")
    idx = d["exercise_index"].numpy()
    return d["z"].numpy()[idx], idx


def project(z, seed, n_neighbors, min_dist):
    import umap
    return umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                     metric="cosine", random_state=seed).fit_transform(z)


def silhouette_force(z, y):
    """Silhouette por direccion de fuerza en el espacio original, no en la proyeccion."""
    keep = y != "unspecified"
    if len(set(y[keep])) < 2:
        return float("nan")
    return float(silhouette_score(z[keep], y[keep], metric="cosine"))


def draw_panel(ax, emb, y, point_size, equal_aspect, rng):
    """Un panel. Los puntos se dibujan en orden aleatorio para que ninguna
    categoria quede sistematicamente encima de otra."""
    perm = rng.permutation(len(emb))
    ax.scatter(emb[perm, 0], emb[perm, 1], s=point_size, alpha=0.8,
               c=[COLORS[v] for v in y[perm]], linewidths=0)
    ax.set_xticks([]); ax.set_yticks([])
    if equal_aspect:
        ax.set_aspect("equal", adjustable="datalim")
    for s in ax.spines.values():
        s.set_linewidth(0.6); s.set_color("#8c959f")


def save(fig, out, dpi, tiff):
    base, ext = os.path.splitext(out)
    fig.savefig(out, bbox_inches="tight")            # vectorial si es .pdf/.svg/.eps
    written = [out]
    if ext.lower() != ".png":
        fig.savefig(base + ".png", dpi=dpi, bbox_inches="tight")
        written.append(base + ".png")
    if tiff:
        fig.savefig(base + ".tiff", dpi=dpi, bbox_inches="tight",
                    pil_kwargs={"compression": "tiff_lzw"})
        written.append(base + ".tiff")
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb-dir", required=True)
    ap.add_argument("--nodes", default="nodes.csv")
    ap.add_argument("--models", nargs="+",
                    default=["mf", "distmult", "transe", "complex",
                             "gcn", "gcn_distmult", "rgcn"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--lam", type=float, default=0.0)
    ap.add_argument("--n-neighbors", type=int, default=15)
    ap.add_argument("--min-dist", type=float, default=0.1)
    ap.add_argument("--ncol", type=int, default=4,
                    help="columnas de la rejilla cuando se grafica una sola semilla")
    ap.add_argument("--panel", type=float, default=2.6,
                    help="lado del panel en pulgadas")
    ap.add_argument("--point-size", type=float, default=None)
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument("--tiff", action="store_true")
    ap.add_argument("--no-equal-aspect", dest="equal_aspect", action="store_false")
    ap.add_argument("--no-panel-letters", dest="panel_letters", action="store_false")
    ap.add_argument("--out", default="fig_umap.pdf")
    ap.add_argument("--stability-csv", default=None)
    cfg = ap.parse_args()

    one_seed = len(cfg.seeds) == 1
    psize = cfg.point_size if cfg.point_size else (7.0 if one_seed else 4.0)
    rng = np.random.default_rng(0)

    if one_seed:
        n = len(cfg.models)
        ncol = min(cfg.ncol, n)
        nrow = math.ceil(n / ncol)
        slots = nrow * ncol
        legend_slot = slots > n          # si sobra una celda, la legenda va ahi
    else:
        nrow, ncol = len(cfg.models), len(cfg.seeds)
        legend_slot = False

    fig, axes = plt.subplots(nrow, ncol, figsize=(cfg.panel * ncol, cfg.panel * nrow),
                             squeeze=False, layout="constrained")
    rows = []
    letters = "abcdefghijklmnopqrstuvwxyz"

    if one_seed:
        seed = cfg.seeds[0]
        for k, model in enumerate(cfg.models):
            i, j = divmod(k, ncol)
            z, idx = load_emb(cfg.emb_dir, model, seed, cfg.lam)
            y = force_labels(cfg.nodes, idx)
            sil = silhouette_force(z, y)
            rows.append({"model": model, "seed": seed, "silhouette_force_64d": sil})
            ax = axes[i][j]
            draw_panel(ax, project(z, seed, cfg.n_neighbors, cfg.min_dist),
                       y, psize, cfg.equal_aspect, rng)
            pre = f"({letters[k]}) " if cfg.panel_letters else ""
            ax.set_title(f"{pre}{LABEL.get(model, model)}   s = {sil:.3f}", fontsize=9)
        for k in range(len(cfg.models), nrow * ncol):   # celdas sobrantes
            i, j = divmod(k, ncol)
            axes[i][j].axis("off")
    else:
        for i, model in enumerate(cfg.models):
            for j, seed in enumerate(cfg.seeds):
                z, idx = load_emb(cfg.emb_dir, model, seed, cfg.lam)
                y = force_labels(cfg.nodes, idx)
                sil = silhouette_force(z, y)
                rows.append({"model": model, "seed": seed, "silhouette_force_64d": sil})
                ax = axes[i][j]
                draw_panel(ax, project(z, seed, cfg.n_neighbors, cfg.min_dist),
                           y, psize, cfg.equal_aspect, rng)
                ax.set_title(f"seed {seed}   s = {sil:.3f}", fontsize=8)
                if j == 0:
                    ax.set_ylabel(LABEL.get(model, model), fontsize=9)

    handles = [Line2D([], [], marker="o", linestyle="none", markersize=5,
                      markerfacecolor=COLORS[lv], markeredgecolor="none", label=lv)
               for lv in LEVELS]
    if legend_slot:
        i, j = divmod(nrow * ncol - 1, ncol)
        axes[i][j].legend(handles=handles, loc="center", frameon=False,
                          fontsize=9, title="force direction",
                          title_fontsize=9, handletextpad=0.4)
    else:
        fig.legend(handles=handles, loc="outside lower center", ncol=len(LEVELS),
                   frameon=False, fontsize=9)

    for f in save(fig, cfg.out, cfg.dpi, cfg.tiff):
        print("escrito:", f)

    df = pd.DataFrame(rows)
    out_csv = cfg.stability_csv or (os.path.splitext(cfg.out)[0] + "_silhouette.csv")
    df.to_csv(out_csv, index=False)
    print("\nSilhouette por direccion de fuerza (espacio de 64 dimensiones):")
    for model in cfg.models:
        v = df[df.model == model]["silhouette_force_64d"].values
        if len(v) > 1:
            print("  %-15s %.3f ± %.3f   (%s)" % (
                LABEL.get(model, model), v.mean(), v.std(ddof=1),
                ", ".join("%.3f" % x for x in v)))
        else:
            print("  %-15s %.3f" % (LABEL.get(model, model), v[0]))
    print("\ntabla:", out_csv)


if __name__ == "__main__":
    main()