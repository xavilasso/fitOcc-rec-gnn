#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reproduce de punta a punta los resultados del articulo a partir del grafo
publicado en KG-OccFitness.

    # todo (entrenamiento incluido: varias horas en CPU)
    python reproduce.py --kg-dir ../KG-OccFitness/fitkg_v3_output

    # solo tablas y figuras a partir de los CSV y embeddings ya versionados (minutos)
    python reproduce.py --kg-dir ../KG-OccFitness/fitkg_v3_output --skip-training

Antes de correr nada comprueba el SHA-256 de los archivos del grafo, de modo que
un tercero sepa si esta usando exactamente el mismo grafo que el articulo.
"""

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Grafo con el que se produjeron los resultados publicados (KG-OccFitness, v3).
EXPECTED_SHA256 = {
    "graph_data.pt": "04c810e46c9f8f895f2331700b3d15a2aedf08db67ef5cdf16c6840567542d0e",
    "nodes.csv": "f5c4462c6210a60c6be8958cdde8fd1e40debe398fd6077426ce3cd68a358f35",
    "edges.csv": "87dbd9d26ee992657c6c997590c90af11fb19c175428a921279ad4e05edee553",
    "edge_rel_mapping.json": "3fbd57bcfe2818d33298d7b7697c8a3c479fb64cacd213db202c15388709102f",
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run(args, cwd):
    print("\n$", " ".join(str(a) for a in args))
    subprocess.run([sys.executable, *map(str, args)], cwd=cwd, check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kg-dir", required=True, type=Path,
                   help="carpeta fitkg_v3_output/ del repositorio KG-OccFitness")
    p.add_argument("--out-dir", type=Path, default=HERE / "output")
    p.add_argument("--emb-dir", type=Path, default=HERE / "emb")
    p.add_argument("--skip-training", action="store_true",
                   help="no entrena: regenera tablas y figuras desde los CSV y "
                        "embeddings existentes")
    p.add_argument("--ignore-hash", action="store_true",
                   help="continuar aunque el grafo no coincida con el del articulo")
    a = p.parse_args()

    kg = a.kg_dir.resolve()
    out = a.out_dir.resolve()
    emb = a.emb_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    # ---- 0. el grafo es el del articulo? ---------------------------------
    bad = []
    for name, expected in EXPECTED_SHA256.items():
        f = kg / name
        if not f.exists():
            sys.exit(f"No existe {f}. Construye el grafo con build_fitkg_lite_v3.py "
                     f"(repositorio KG-OccFitness) o apunta --kg-dir a su salida.")
        got = sha256(f)
        print(f"{name:24s} {'OK' if got == expected else 'DISTINTO'}  {got[:16]}")
        if got != expected:
            bad.append(name)
    if bad and not a.ignore_hash:
        sys.exit(f"\nEl grafo no es el del articulo ({', '.join(bad)}). "
                 f"Usa --ignore-hash para continuar de todos modos.")

    graph = ["--graph", kg / "graph_data.pt", "--nodes", kg / "nodes.csv",
             "--rel-map", kg / "edge_rel_mapping.json"]
    runner = HERE / "fkg_experiments.py"
    models = ["mf", "distmult", "gcn", "gcn_distmult", "rgcn"]

    # ---- 1. entrenamiento --------------------------------------------------
    if not a.skip_training:
        # resultados principales: 5 modelos x 2 lambdas x 30 semillas
        run([runner, *graph, "--models", *models, "--lambdas", "0.0", "0.3",
             "--runs", "30", "--out", out / "results_raw.csv"], cwd=out)
        # control de fuga: hasType fuera del entrenamiento y de la propagacion
        run([runner, *graph, "--models", *models, "--lambdas", "0.0",
             "--runs", "30", "--exclude-relations", "hasType",
             "--out", out / "results_control_sin_hastype.csv"], cwd=out)
        # embeddings de la semilla 0 para la figura
        run([runner, *graph, "--models", *models, "--lambdas", "0.0",
             "--runs", "1", "--save-emb", emb, "--save-seeds", "0",
             "--out", out / "results_v3_emb.csv"], cwd=out)

    # ---- 2. tablas ---------------------------------------------------------
    run([HERE / "fkg_stats.py", out / "results_raw.csv", "--reference", "rgcn",
         "--lam", "0.3", "--prefix", "tabla"], cwd=out)
    run([HERE / "fkg_stats.py", out / "results_control_sin_hastype.csv",
         "--reference", "rgcn", "--lam", "0.0", "--prefix", "control"], cwd=out)
    run([HERE / "fkg_graph_report.py", *graph, "--out-prefix", "grafo"], cwd=out)

    # ---- 3. figuras --------------------------------------------------------
    for color_by, name in (("type", "figura_embeddings"), ("force", "fig_force"),
                           ("mechanic", "fig_mechanic"), ("muscle", "fig_muscle")):
        run([HERE / "make_umap_figure.py", "--emb", emb, "--nodes", kg / "nodes.csv",
             "--edges", kg / "edges.csv", "--lam", "0", "--color-by", color_by,
             "--out", out / name], cwd=out)

    print(f"\nListo. Todo en {out}")


if __name__ == "__main__":
    main()
