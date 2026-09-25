# fitOcc-rec-gnn

Relational representation learning over the **Fitness Knowledge Graph (FKG)**. This repository
contains the experiments that validate the graph built in
[KG-OccFitness](https://github.com/xavilasso/KG-OccFitness), and reproduces every table and figure
of the results section of:

> K. Avila, J. Lasso-Saldaña, J. Gálvez, O. Avalos. *Design and Construction of a Multi-Relational
> Fitness Knowledge Graph for Semantic Exercise Representation.* Mathematics, MDPI. `[CONFIRMAR año/DOI]`

---

## What the experiments answer

The evaluation asks whether the semantic organization introduced during graph construction is
recoverable by a model, and separates two properties that turn out to behave differently:

- **Edge reconstruction** — how well a model recovers assertions that were removed from training.
- **Semantic organization** — whether exercises sharing properties that were never given as a
  training signal end up close together in the embedding space.

Five encoders are compared under an identical protocol so that the difference between them can only
come from the encoder or the decoder:

| Id | Encoder | Decoder | What it isolates |
|---|---|---|---|
| `mf` | free embeddings | dot | neither structure nor relation types |
| `distmult` | free embeddings | DistMult | relation types, no propagation |
| `gcn` | GCN | dot | propagation, no relation types |
| `gcn_distmult` | GCN | DistMult | propagation and scoring separated |
| `rgcn` | R-GCN | DistMult | relation types inside message passing |

A non-learned Jaccard baseline over target-muscle sets fixes the floor.

---

## Protocol

- Stratified 70 / 10 / 20 split **per relation**, so that sparse relations still reach the test set.
- Mirror assertions collapsed before splitting. The stored tensor holds each assertion in both
  directions with the same relation id; without collapsing, an assertion can land in train and its
  mirror in test, and since every decoder here is symmetric that test assertion is trivially
  scorable.
- Negative sampling restricted to the observed range of each relation and filtered against the true
  triples of the complete graph. The proportion of queries for which no valid negative could be
  drawn is reported per relation and stays below 0.03 %.
- Learnable input embeddings for the propagating encoders, so the comparison measures the encoder
  and not the richness of the input: the stored node features are a one-hot of the node type, which
  would give all 873 exercises an identical input vector.
- Early stopping on validation AUC, patience 20. The test partition is evaluated once, at the end.
- 30 seeds. Paired Wilcoxon signed-rank test with Holm correction and paired Cohen's *d*.

---

## Repository layout

```
fkg_experiments.py      unified runner: the five models, both objectives, all seeds
fkg_stats.py            paired tests, Holm correction, effect sizes, LaTeX export
fkg_graph_report.py     structural characterisation of the graph
make_umap_figure.py     two-dimensional projection of the learned embeddings
fitkg_output/           graph inputs consumed by the runner   [CONFIRMAR contenido]
output/                 result CSVs
emb/                    saved embeddings, for the UMAP figure
requirements.txt
```

### Inputs

The runner expects three files produced by the construction repository:

| File | Origin |
|---|---|
| `graph_data.pt` | PyTorch Geometric tensor: `x`, `edge_index`, `edge_type` |
| `nodes.csv` | `fitkg_v3_output/nodes.csv` of KG-OccFitness |
| `edge_rel_mapping.json` | relation name → relation id |

`[CONFIRMAR]` Document here how `graph_data.pt` and `edge_rel_mapping.json` are generated from
`fitkg_v3_output/`. This is the single step that someone reproducing the work cannot guess, and it
is the one the reviewers asked to be able to identify exactly.

---

## Reproducing the reported results

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# quick check on a synthetic graph, touches no data
python fkg_experiments.py --smoke

# main experiment: five models, both objectives, 30 seeds  (Tables 6, 8, 9)
python fkg_experiments.py --models mf distmult gcn gcn_distmult rgcn \
    --lambdas 0.0 0.3 --runs 30 --out output/results_raw.csv

# leakage control: hasType and its inverse removed from training and propagation  (Table 10)
python fkg_experiments.py --models mf distmult gcn gcn_distmult rgcn \
    --lambdas 0.0 --runs 30 --exclude-relations hasType \
    --out output/results_control_sin_hastype.csv

# paired tests, Holm correction, effect sizes  (Table 7)
python fkg_stats.py --raw output/results_raw.csv

# structural characterisation  (Tables 3 and 4)
python fkg_graph_report.py

# embeddings and projection  (Figure 5)
python fkg_experiments.py --models mf distmult gcn gcn_distmult rgcn \
    --lambdas 0.0 --runs 1 --save-emb emb/ --save-seeds 0
python make_umap_figure.py --emb emb/ --color-by force --out figura_embeddings
```

### `--exclude-relations`

Removes the listed relations from the training graph, the propagation graph and their inverses, and
**reindexes the remaining relations** so that the relation ids stay contiguous. Without the
reindexing the R-GCN would carry one dead `W_r` matrix and DistMult one dead relation vector, and
the parameter counts would no longer be comparable with the main run.

The ground truth of the semantic metrics is always built from the **complete** graph, which is what
makes this a control: the model never observes the relation, and is still measured on the attribute
that relation encodes.

Note that the global AUC of a control run is not comparable with the main run, because removing a
relation changes the composition of the test set. The comparable quantities are the semantic
metrics, which use the same ground truth in both conditions.

---

## Which file produces which table

| Paper | Produced by |
|---|---|
| Table 3, Table 4 | `fkg_graph_report.py` |
| Table 6 | `fkg_experiments.py` → `results_raw.csv`, λ = 0 |
| Table 7 | `fkg_stats.py` → `results_raw.csv` |
| Table 8 | `fkg_experiments.py`, per-relation column of `results_raw.csv` |
| Table 9 | `fkg_experiments.py` → `results_raw.csv`, λ = 0 against λ = 0.3 |
| Table 10 | `fkg_experiments.py --exclude-relations hasType` |
| Figure 5 | `make_umap_figure.py` |

---

## Requirements

Python 3.11, PyTorch, PyTorch Geometric, scikit-learn, pandas, scipy, umap-learn.
Exact versions in `requirements.txt`. `[CONFIRMAR]` versions of torch and torch-geometric, which
are the two that matter for reproducibility.

## Citation and licence

Same citation as the construction repository. Code released under the MIT Licence.
