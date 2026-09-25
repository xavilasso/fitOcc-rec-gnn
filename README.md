# fitOcc-rec-gnn

Relational-learning experiments on the **Fitness Knowledge Graph (FKG)**. Five encoders are
trained on link prediction under one shared protocol, and the exercise representations they
learn are evaluated against two independent semantic ground truths: shared target muscle and
shared exercise type.

The graph itself is built in [KG-OccFitness](https://github.com/xavilasso/KG-OccFitness). This
repository accompanies:

> K. Avila, J. Lasso-Saldaña, J. Gálvez, O. Avalos. *Design and Construction of a Multi-Relational
> Fitness Knowledge Graph for Semantic Exercise Representation.* Mathematics, MDPI, 2026.
> DOI: `[PENDIENTE]`

The results reported in the article were produced with release `[PENDIENTE: tag]` of this
repository (Zenodo DOI `[PENDIENTE]`) on the graph of release `[PENDIENTE: tag]` of
KG-OccFitness.

---

## Models

| id | Encoder | Decoder | Label in the article |
|---|---|---|---|
| `mf` | free embeddings | dot product | Free embeddings (matrix factorization) |
| `distmult` | free embeddings | DistMult | DistMult (relation types, no propagation) |
| `gcn` | 2-layer GCN | dot product | GCN (topology, no relation types) |
| `gcn_distmult` | 2-layer GCN | DistMult | GCN + relation-aware decoder |
| `rgcn` | 2-layer R-GCN | DistMult | R-GCN (relation-aware propagation) |

A non-learned baseline (`jaccard`: Jaccard similarity of target-muscle sets) is always added
as the first row of every results file.

## Protocol

All models share the same protocol, so that differences between them can only come from the
encoder or the decoder.

- **Split.** Per relation, 70 / 10 / 20 train / validation / test, with the seed of the run.
  Relations with fewer than 10 assertions go entirely to train.
- **Message passing** uses train assertions only, plus their inverses as separate relation
  types (`--no-inverse` disables them).
- **Input.** GCN and R-GCN receive a learnable 64-d input embedding per node, the same capacity
  as the free-embedding models (`--raw-features` uses the one-hot type vector of
  `graph_data.pt` instead).
- **Negatives.** One per positive, corrupting head or tail within the range observed for that
  relation, filtered against the full graph.
- **Loss.** Binary cross-entropy on link prediction plus, when λ > 0, a pairwise margin loss
  (margin 0.3, 20 positives per anchor) that pulls together exercises sharing a target muscle.
  The pairs are built from train assertions only.
- **Training.** Adam, lr 1e-3, weight decay 1e-5, dropout 0.3, hidden and output dimension 64,
  up to 150 epochs, early stopping on validation AUC with patience 20. Test is evaluated once,
  on the best validation checkpoint.
- **Metrics.** AUC and AP on test link prediction, globally and per relation; Δ similarity,
  Retrieval@5 and silhouette of exercise embeddings against the muscle and type ground truths.
  The ground truths are always built from the **full** graph.
- **Seeds.** 30 runs, seeds 0–29. Seed *s* fixes the split, the initialisation and the negatives,
  and runs are deterministic on CPU: repeating a seed reproduces its metrics to within 1e-5.

All values above are the defaults of `fkg_experiments.py`; `python fkg_experiments.py --help`
lists every option.

---

## Installation

Tested with Python 3.11.9 on Windows 11, CPU only.

```bash
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu
pip install torch_scatter==2.1.2 torch_sparse==0.6.18 torch_cluster==1.6.3 torch_spline_conv==1.2.2 \
    -f https://data.pyg.org/whl/torch-2.3.0+cpu.html
pip install -r requirements.txt
```

Check the environment on a synthetic graph (about one minute, does not read any data):

```bash
python fkg_experiments.py --smoke          # writes smoke_results.csv; the numbers are meaningless
```

---

## Reproducing the article

Clone both repositories side by side:

```
parent/
├── KG-OccFitness/      contains fitkg_v3_output/
└── fitOcc-rec-gnn/
```

### One command

```bash
python reproduce.py --kg-dir ../KG-OccFitness/fitkg_v3_output                  # everything, several hours on CPU
python reproduce.py --kg-dir ../KG-OccFitness/fitkg_v3_output --skip-training  # tables and figures only, minutes
```

`reproduce.py` first checks the SHA-256 of the graph files against the graph used in the article
and stops if they differ. With `--skip-training` it rebuilds every table and figure from the
results and embeddings already versioned in `output/` and `emb/`.

### Step by step

The input of the experiments is the output of the graph builder in KG-OccFitness:
`graph_data.pt`, `nodes.csv` and `edge_rel_mapping.json`, all in `fitkg_v3_output/`. They are
**not** copied into this repository; pass their paths explicitly, since the defaults of the
scripts look for them in the working directory.

```bash
KG=../KG-OccFitness/fitkg_v3_output
GRAPH="--graph $KG/graph_data.pt --nodes $KG/nodes.csv --rel-map $KG/edge_rel_mapping.json"

# 1. main results: 5 models × λ ∈ {0, 0.3} × 30 seeds
python fkg_experiments.py $GRAPH --models mf distmult gcn gcn_distmult rgcn \
    --lambdas 0.0 0.3 --runs 30 --out output/results_raw.csv

# 2. leakage control: hasType removed from training and message passing
python fkg_experiments.py $GRAPH --models mf distmult gcn gcn_distmult rgcn \
    --lambdas 0.0 --runs 30 --exclude-relations hasType \
    --out output/results_control_sin_hastype.csv

# 3. embeddings of seed 0 for the figures
python fkg_experiments.py $GRAPH --models mf distmult gcn gcn_distmult rgcn \
    --lambdas 0.0 --runs 1 --save-emb emb --save-seeds 0 --out output/results_v3_emb.csv

# 4. tables (written to the working directory with the given prefix)
cd output
python ../fkg_stats.py results_raw.csv --reference rgcn --lam 0.3 --prefix tabla
python ../fkg_stats.py results_control_sin_hastype.csv --reference rgcn --lam 0.0 --prefix control
python ../fkg_graph_report.py --graph ../$KG/graph_data.pt --nodes ../$KG/nodes.csv \
    --rel-map ../$KG/edge_rel_mapping.json --out-prefix grafo
cd ..

# 5. figures
python make_umap_figure.py --emb emb --nodes $KG/nodes.csv --edges $KG/edges.csv \
    --lam 0 --color-by type --out output/figura_embeddings
# same with --color-by force | mechanic | muscle  →  output/fig_force, fig_mechanic, fig_muscle
```

`fkg_stats.py` summarises a single λ (`--lam`, default 0.3). The control file only contains
λ = 0, so it must be run with `--lam 0.0`; otherwise the summary table holds only the baseline.

### Relevant flags

| Script | Flag | Effect |
|---|---|---|
| `fkg_experiments.py` | `--models` | subset of `mf distmult gcn gcn_distmult rgcn` (default: all) |
| | `--lambdas` | weights of the functional margin loss (default `0.0 0.3`) |
| | `--runs` | number of seeds, 0 … runs−1 (default 30) |
| | `--exclude-relations` | drops those relations (and their inverses) from training and message passing; the semantic ground truth is still built from the full graph |
| | `--save-emb DIR` | saves `emb_<model>_lam<λ>_seed<s>.pt` with the full node embedding matrix |
| | `--save-seeds` | seeds whose embeddings are saved (default `0`; empty list = all) |
| | `--smoke` | synthetic graph, 2 seeds, 5 epochs, output `smoke_results.csv` |
| `fkg_stats.py` | `--reference`, `--lam`, `--prefix` | reference model of the paired tests, λ summarised, output prefix |
| `make_umap_figure.py` | `--color-by` | `type`, `force`, `mechanic`, `muscle` or `intensity` |
| | `--method` | `umap` (default), `tsne` or `pca` |

---

## Which file produces which result

| Article | File in `output/` | Produced by |
|---|---|---|
| `[PENDIENTE: Tabla N]` main comparison (mean ± sd over 30 seeds) | `tabla_principal.csv`, `tabla_principal.tex` | `fkg_stats.py results_raw.csv --lam 0.3` |
| `[PENDIENTE: Tabla N]` paired Wilcoxon vs R-GCN, Holm-corrected, Cohen's d | `tabla_significancia.csv` | idem |
| `[PENDIENTE: Tabla N]` effect of the functional loss (λ = 0.3 vs λ = 0) | `tabla_ablacion_lambda.csv` | idem |
| `[PENDIENTE: Tabla N]` AUC / AP per relation | `tabla_por_relacion.csv` | idem |
| `[PENDIENTE: Tabla N]` leakage control without `hasType` | `control_principal.csv`, `control_significancia.csv` | `fkg_stats.py results_control_sin_hastype.csv --lam 0.0` |
| `[PENDIENTE: Tabla N]` structural characterisation of the graph | `grafo_por_relacion.csv`, `grafo_grado_por_tipo.csv`, `grafo_grados.csv` | `fkg_graph_report.py` |
| `[PENDIENTE: Figura N]` embedding space coloured by exercise type | `figura_embeddings.svg/.png` | `make_umap_figure.py --color-by type` |
| `[PENDIENTE]` same coloured by force, mechanic, muscle | `fig_force.*`, `fig_mechanic.*`, `fig_muscle.*` | `make_umap_figure.py --color-by …` |

Raw per-seed results, one row per (model, λ, seed):

| File | Rows | Content |
|---|---:|---|
| `output/results_raw.csv` | 301 | baseline + 5 models × 2 λ × 30 seeds |
| `output/results_control_sin_hastype.csv` | 151 | baseline + 5 models × λ = 0 × 30 seeds, `hasType` excluded |
| `output/results_v3_emb.csv` | 6 | baseline + 5 models × λ = 0 × seed 0 (the run that saved `emb/`) |

`emb/` holds the five embedding files of that run, from which the figures are drawn. Rebuilding
the figures from them reproduces the published PNG byte for byte.

SHA-256 of the raw results:

```
f8685760ab246d49e564548fc59dcfb031546c9fd68a7e9bf85642d161742e13  output/results_raw.csv
ab4438430e50eb0caa6a8eef09848938a2593da0a362eb0ad986e20a6b6df825  output/results_control_sin_hastype.csv
5e0b6c35501461471bc707dfba9e2b00ccab02ffcabbd2e1e82e7dd0491b9ca6  output/results_v3_emb.csv
```

---

## Repository layout

```
fkg_experiments.py     unified runner: models, protocol, metrics
fkg_stats.py           summary, paired tests, λ ablation, per-relation table
fkg_graph_report.py    structural characterisation of the graph
make_umap_figure.py    2-D projection of the learned exercise embeddings
reproduce.py           runs the whole chain and checks the input graph
output/                raw results, tables and figures of the article
emb/                   embeddings of seed 0 used for the figures
requirements.txt
LICENSE
```

## Citation

```bibtex
@article{avila2026fkg,
  title   = {Design and Construction of a Multi-Relational Fitness Knowledge Graph
             for Semantic Exercise Representation},
  author  = {Avila, Karla and Lasso-Salda{\~n}a, Javier and G{\'a}lvez, Jorge and Avalos, Omar},
  journal = {Mathematics},
  year    = {2026},
  doi     = {[PENDIENTE]}
}
```

## Licence

MIT (`LICENSE`).
