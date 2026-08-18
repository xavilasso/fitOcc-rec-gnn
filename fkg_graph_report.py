#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Caracterizacion estructural del Fitness Knowledge Graph.

Produce el analisis que el review pide para la Seccion 5.1 y que las tablas
actuales no contienen: distribucion de grado, componentes conexas, densidad por
tipo de relacion, cardinalidad de dominio y rango, y efecto hub.

    python fkg_graph_report.py --graph graph_data.pt --nodes nodes.csv \
        --rel-map edge_rel_mapping.json --out-prefix grafo
"""

import argparse
import json

import numpy as np
import pandas as pd
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--graph", default="graph_data.pt")
    p.add_argument("--nodes", default="nodes.csv")
    p.add_argument("--rel-map", dest="rel_map", default="edge_rel_mapping.json")
    p.add_argument("--exercise-label", dest="exercise_label", default="Exercise")
    p.add_argument("--out-prefix", dest="prefix", default="grafo")
    a = p.parse_args()

    try:
        data = torch.load(a.graph, weights_only=False)
    except TypeError:
        data = torch.load(a.graph)
    nodes = pd.read_csv(a.nodes)
    with open(a.rel_map) as f:
        rel2id = json.load(f)
    id2rel = {v: k for k, v in rel2id.items()}

    ei = data.edge_index.long().cpu()
    et = data.edge_type.long().cpu()
    n = int(ei.max()) + 1
    nr = int(et.max()) + 1
    types = nodes["type"].astype(str).tolist()

    # ---- aristas dirigidas vs aserciones unicas -------------------------
    lo = torch.minimum(ei[0], ei[1]).long()
    hi = torch.maximum(ei[0], ei[1]).long()
    key = (lo * nr + et) * n + hi
    uniq, first = np.unique(key.numpy(), return_index=True)
    keep = np.sort(first)
    ei_u, et_u = ei[:, keep], et[keep]

    print(f"nodos                       : {n}")
    print(f"aristas en el tensor        : {ei.size(1)}")
    print(f"aserciones unicas (sin espejo): {ei_u.size(1)}")
    print(f"factor de duplicacion       : {ei.size(1) / max(ei_u.size(1), 1):.2f}\n")

    # ---- por relacion ---------------------------------------------------
    rows = []
    for r in range(nr):
        m = et_u == r
        if not m.any():
            continue
        h = ei_u[0, m]
        t = ei_u[1, m]
        heads, tails = h.unique(), t.unique()
        e = int(m.sum())
        possible = len(heads) * len(tails)
        _, cnt_h = h.unique(return_counts=True)
        _, cnt_t = t.unique(return_counts=True)
        rows.append(dict(
            relation=id2rel.get(r, str(r)), edges=e,
            domain=len(heads), range=len(tails),
            density=e / possible if possible else np.nan,
            mean_out_degree=float(cnt_h.float().mean()),
            mean_in_degree=float(cnt_t.float().mean()),
            max_in_degree=int(cnt_t.max()),
        ))
    rel_df = pd.DataFrame(rows).sort_values("edges", ascending=False)
    print("=== Por tipo de relacion (aserciones unicas) ===")
    print(rel_df.to_markdown(index=False, floatfmt=".4f"))

    # ---- grado global y hubs -------------------------------------------
    deg = np.bincount(ei_u[0].numpy(), minlength=n) + np.bincount(ei_u[1].numpy(), minlength=n)
    nodes_deg = pd.DataFrame({"node": np.arange(n), "type": types[:n], "degree": deg})

    print("\n=== Distribucion de grado por tipo de nodo ===")
    g = nodes_deg.groupby("type")["degree"].agg(
        ["count", "mean", "std", "min", "median", "max"])
    print(g.to_markdown(floatfmt=".2f"))

    print("\n=== 15 nodos de mayor grado (efecto hub) ===")
    top = nodes_deg.sort_values("degree", ascending=False).head(15).copy()
    top["share_of_edges"] = top["degree"] / ei_u.size(1)
    print(top.to_markdown(index=False, floatfmt=".4f"))

    ex_mask = nodes_deg["type"] == a.exercise_label
    ex_deg = nodes_deg.loc[ex_mask, "degree"].values
    if len(ex_deg):
        qs = np.percentile(ex_deg, [0, 25, 50, 75, 90, 100])
        print(f"\ngrado de los nodos de ejercicio  min/Q1/mediana/Q3/P90/max: "
              f"{qs[0]:.0f} / {qs[1]:.0f} / {qs[2]:.0f} / {qs[3]:.0f} / {qs[4]:.0f} / {qs[5]:.0f}")

    # ---- componentes conexas -------------------------------------------
    A = coo_matrix((np.ones(ei_u.size(1)), (ei_u[0].numpy(), ei_u[1].numpy())), shape=(n, n))
    ncomp, labels = connected_components(A, directed=False)
    sizes = np.bincount(labels)
    print(f"\ncomponentes conexas (no dirigido): {ncomp}  "
          f"| mayor componente: {sizes.max()} nodos ({sizes.max() / n:.1%})")

    # ---- cobertura de isVariationOf ------------------------------------
    if "isVariationOf" in rel2id:
        m = et_u == rel2id["isVariationOf"]
        cov = len(torch.cat([ei_u[0, m], ei_u[1, m]]).unique())
        print(f"ejercicios tocados por isVariationOf: {cov} "
              f"({cov / max(int(ex_mask.sum()), 1):.1%} de los ejercicios)")

    rel_df.to_csv(f"{a.prefix}_por_relacion.csv", index=False)
    nodes_deg.to_csv(f"{a.prefix}_grados.csv", index=False)
    g.to_csv(f"{a.prefix}_grado_por_tipo.csv")
    print(f"\nArchivos escritos con prefijo '{a.prefix}_'")


if __name__ == "__main__":
    main()