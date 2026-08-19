#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Runner unificado de experimentos para el Fitness Knowledge Graph.

Corre TODOS los modelos con el mismo protocolo, de modo que las diferencias entre ellos solo puedan venir del encoder / decoder y no del entrenamiento.

Uso:

    # comprobacion rapida con grafo sintetico (no toca tus datos)
    python fkg_experiments.py --smoke

    # ablacion de circularidad: los cinco modelos con y sin regularizacion funcional
    python fkg_experiments.py --models mf distmult gcn gcn_distmult rgcn \
        --lambdas 0.0 0.3 --runs 30 --out results_raw.csv
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, silhouette_score
from sklearn.model_selection import train_test_split

try:
    from torch_geometric.nn import GCNConv, RGCNConv
except ImportError:  # solo para --smoke sin PyG instalado
    GCNConv = RGCNConv = None


# =====================================================================
# 1. Carga del grafo
# =====================================================================

def load_graph(args):
    """Devuelve edge_index, edge_type, x, tipos de nodo y el mapa de relaciones."""
    try:
        data = torch.load(args.graph, weights_only=False)
    except TypeError:  # torch < 2.0
        data = torch.load(args.graph)
    nodes_df = pd.read_csv(args.nodes)
    with open(args.rel_map) as f:
        rel2id = json.load(f)

    return dict(
        x=data.x.float(),
        edge_index=data.edge_index.long(),
        edge_type=data.edge_type.long(),
        node_type=nodes_df["type"].tolist(),
        rel2id=rel2id,
    )


def synthetic_graph(seed=0):
    """Grafo sintetico con la misma forma que el FKG (para --smoke)."""
    rng = np.random.default_rng(seed)
    counts = dict(Exercise=452, Muscle=25, Equipment=17, ExerciseType=10,
                  Intensity=3, Goal=2)
    node_type, offsets, pos = [], {}, 0
    for t, n in counts.items():
        offsets[t] = np.arange(pos, pos + n)
        node_type += [t] * n
        pos += n
    ex = offsets["Exercise"]

    rel2id = {"targets": 0, "recruits": 1, "requires": 2, "hasType": 3,
              "hasIntensity": 4, "supportsGoal": 5, "isVariationOf": 6}
    spec = [("targets", offsets["Muscle"], 837), ("recruits", offsets["Muscle"], 266),
            ("requires", offsets["Equipment"], 453), ("hasType", offsets["ExerciseType"], 453),
            ("hasIntensity", offsets["Intensity"], 453), ("supportsGoal", offsets["Goal"], 894),
            ("isVariationOf", ex, 62)]

    src, dst, typ = [], [], []
    for name, tails, n_edges in spec:
        seen = set()
        while len(seen) < n_edges:
            h = int(rng.choice(ex))
            t = int(rng.choice(tails))
            if h != t:
                seen.add((h, t))
        for h, t in seen:
            src.append(h); dst.append(t); typ.append(rel2id[name])

    n = pos
    return dict(
        x=torch.eye(n),
        edge_index=torch.tensor([src, dst], dtype=torch.long),
        edge_type=torch.tensor(typ, dtype=torch.long),
        node_type=node_type,
        rel2id=rel2id,
    )


# =====================================================================
# 2. Particion estratificada train / val / test
# =====================================================================

def split_edges(edge_index, edge_type, val_ratio, test_ratio, seed):
    parts = {k: {"src": [], "dst": [], "rel": []} for k in ("train", "val", "test")}

    for r in edge_type.unique().tolist():
        mask = edge_type == r
        e = edge_index[:, mask].t()                      # [E_r, 2]
        idx = np.arange(e.size(0))

        if e.size(0) < 10:                               # relacion demasiado pequeña
            chunks = {"train": idx, "val": np.array([], int), "test": np.array([], int)}
        else:
            tr, tmp = train_test_split(idx, test_size=val_ratio + test_ratio,
                                       random_state=seed)
            va, te = train_test_split(tmp, test_size=test_ratio / (val_ratio + test_ratio),
                                      random_state=seed)
            chunks = {"train": tr, "val": va, "test": te}

        for name, sel in chunks.items():
            if len(sel) == 0:
                continue
            parts[name]["src"].append(e[sel, 0])
            parts[name]["dst"].append(e[sel, 1])
            parts[name]["rel"].append(torch.full((len(sel),), r, dtype=torch.long))

    out = {}
    for name, d in parts.items():
        out[name] = (
            torch.stack([torch.cat(d["src"]), torch.cat(d["dst"])]),
            torch.cat(d["rel"]),
        )
    return out


def deduplicate_undirected(edge_index, edge_type):
    """Colapsa (u, r, v) y (v, r, u) en una sola asercion.

    El tensor guardado en graph_data.pt contiene cada arista en las dos
    direcciones con el MISMO id de relacion. Si no se colapsan antes de
    particionar, una arista puede caer en train y su espejo en test, y como
    todos los decoders usados aqui son simetricos, esa arista de test es
    trivialmente puntuable. Es fuga directa entre particiones.
    """
    n = int(edge_index.max()) + 1
    nr = int(edge_type.max()) + 1
    lo = torch.minimum(edge_index[0], edge_index[1]).long()
    hi = torch.maximum(edge_index[0], edge_index[1]).long()
    key = (lo * nr + edge_type.long()) * n + hi
    _, first = np.unique(key.cpu().numpy(), return_index=True)
    keep = torch.from_numpy(np.sort(first))
    return edge_index[:, keep], edge_type[keep], edge_index.size(1) - len(keep)


def add_inverse_relations(edge_index, edge_type, num_rel):
    """Duplica el grafo de propagacion con las relaciones inversas."""
    inv_index = edge_index.flip(0)
    inv_type = edge_type + num_rel
    return (torch.cat([edge_index, inv_index], dim=1),
            torch.cat([edge_type, inv_type]))


# =====================================================================
# 3. Muestreo negativo restringido por tipo y filtrado
# =====================================================================

class NegativeSampler:
    """Corrompe cabeza o cola dentro del rango observado de cada relacion y
    rechaza los triples que existen realmente en el grafo completo."""

    def __init__(self, edge_index, edge_type, num_nodes, num_rel):
        self.num_nodes = num_nodes
        self.num_rel = num_rel
        self.heads, self.tails = {}, {}
        for r in range(num_rel):
            m = edge_type == r
            if m.any():
                self.heads[r] = edge_index[0, m].unique()
                self.tails[r] = edge_index[1, m].unique()
        keys = self._key(edge_index[0], edge_type, edge_index[1])
        self.true_keys = torch.unique(keys)

    def _key(self, h, r, t):
        return h.long() * (self.num_rel * self.num_nodes) + r.long() * self.num_nodes + t.long()

    def sample(self, edge_index, edge_type, max_tries=25):
        E = edge_index.size(1)
        neg_h = edge_index[0].clone()
        neg_t = edge_index[1].clone()
        corrupt_tail = torch.rand(E) < 0.5
        pending = torch.ones(E, dtype=torch.bool)

        for _ in range(max_tries):
            if not pending.any():
                break
            for r in edge_type.unique().tolist():
                sel = pending & (edge_type == r)
                n = int(sel.sum())
                if n == 0:
                    continue
                hs, ts = self.heads[r], self.tails[r]
                ct = corrupt_tail[sel]
                cand_h = hs[torch.randint(len(hs), (n,))]
                cand_t = ts[torch.randint(len(ts), (n,))]
                neg_h[sel] = torch.where(ct, edge_index[0][sel], cand_h)
                neg_t[sel] = torch.where(ct, cand_t, edge_index[1][sel])
            pending = torch.isin(self._key(neg_h, edge_type, neg_t), self.true_keys)

        neg_index = torch.stack([neg_h, neg_t])
        return neg_index, pending          # pending == negativos imposibles

    def failure_by_relation(self, edge_type, failed):
        out = {}
        for r in edge_type.unique().tolist():
            m = edge_type == r
            out[int(r)] = float(failed[m].float().mean())
        return out


# =====================================================================
# 4. Modelos: encoder x decoder, con normalizacion comun
# =====================================================================

class Encoder(nn.Module):
    """kind: 'free' (embeddings libres, sin propagacion) | 'gcn' | 'rgcn'."""

    def __init__(self, kind, num_nodes, in_dim, hidden_dim, out_dim,
                 num_relations, num_bases, dropout, normalize,
                 learnable_input=True, input_dim=64):
        super().__init__()
        self.kind, self.normalize = kind, normalize
        self.drop = nn.Dropout(dropout)

        # Entrada aprendible para los encoders con propagacion.
        # El graph_data.pt de la v3 trae x = one-hot del tipo de nodo (8 dims),
        # asi que TODOS los ejercicios comparten el mismo vector de entrada.
        # Sin esto, mf y distmult parten de 64 dims aprendibles por nodo y gcn y
        # rgcn de un vector identico para 873 nodos: la comparacion mediria la
        # riqueza de la entrada, no el encoder.
        
        self.learnable_input = learnable_input and kind in ("gcn", "rgcn")
        eff_in = input_dim if self.learnable_input else in_dim
        if self.learnable_input:
            self.inp = nn.Embedding(num_nodes, input_dim)
            nn.init.xavier_uniform_(self.inp.weight)

        if kind == "free":
            self.emb = nn.Embedding(num_nodes, out_dim)
            nn.init.xavier_uniform_(self.emb.weight)
        elif kind == "gcn":
            self.conv1 = GCNConv(eff_in, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, out_dim)
        elif kind == "rgcn":
            self.conv1 = RGCNConv(eff_in, hidden_dim, num_relations=num_relations,
                                  num_bases=num_bases)
            self.conv2 = RGCNConv(hidden_dim, out_dim, num_relations=num_relations,
                                  num_bases=num_bases)
        else:
            raise ValueError(kind)

    def forward(self, x, edge_index, edge_type):
        if self.kind == "free":
            return F.normalize(self.emb.weight, p=2, dim=1) if self.normalize \
                else self.emb.weight
        feat = self.inp.weight if self.learnable_input else x
        if self.kind == "gcn":
            h = self.drop(F.relu(self.conv1(feat, edge_index)))
            z = self.conv2(h, edge_index)
        else:
            h = self.drop(F.relu(self.conv1(feat, edge_index, edge_type)))
            z = self.conv2(h, edge_index, edge_type)
        return F.normalize(z, p=2, dim=1) if self.normalize else z


class Model(nn.Module):
    def __init__(self, encoder_kind, decoder_kind, num_nodes, in_dim, hidden_dim,
                 out_dim, num_relations_mp, num_relations_score, num_bases,
                 dropout, normalize, learnable_input=True, input_dim=64):
        super().__init__()
        self.encoder = Encoder(encoder_kind, num_nodes, in_dim, hidden_dim, out_dim,
                               num_relations_mp, num_bases, dropout, normalize,
                               learnable_input, input_dim)
        self.decoder_kind = decoder_kind
        if decoder_kind == "distmult":
            self.rel_weights = nn.Parameter(torch.empty(num_relations_score, out_dim))
            nn.init.xavier_uniform_(self.rel_weights)

    def forward(self, x, edge_index, edge_type):
        return self.encoder(x, edge_index, edge_type)

    def decode(self, z, edge_index, edge_type):
        src, dst = edge_index
        if self.decoder_kind == "dot":
            return (z[src] * z[dst]).sum(dim=1)
        return (z[src] * self.rel_weights[edge_type] * z[dst]).sum(dim=1)


MODEL_ZOO = {
    # id                encoder   decoder      etiqueta para el paper
    "mf":            ("free",  "dot",      "Free embeddings (matrix factorization)"),
    "distmult":      ("free",  "distmult", "DistMult (relation types, no propagation)"),
    "gcn":           ("gcn",   "dot",      "GCN (topology, no relation types)"),
    "gcn_distmult":  ("gcn",   "distmult", "GCN + relation-aware decoder"),
    "rgcn":          ("rgcn",  "distmult", "R-GCN (relation-aware propagation)"),
}


# =====================================================================
# 5. Pares funcionales y perdida de margen por par
# =====================================================================

def exercise_nodes(G, cfg):
    """Indices de los nodos de ejercicio, segun la columna 'type' de nodes.csv."""
    idx = [i for i, t in enumerate(G["node_type"]) if str(t) == cfg.exercise_label]
    if not idx:
        raise SystemExit(
            f"Ningun nodo con type == '{cfg.exercise_label}'. Valores disponibles: "
            f"{sorted(set(map(str, G['node_type'])))}. Usa --exercise-label.")
    return idx


def incidence_matrix(edge_index, edge_type, rel_id, rows, cols):
    """Matriz binaria fila x columna para una relacion concreta."""
    row_pos = {n: i for i, n in enumerate(rows)}
    col_pos = {n: i for i, n in enumerate(cols)}
    M = np.zeros((len(rows), len(cols)), dtype=bool)
    m = edge_type == rel_id
    for h, t in zip(edge_index[0, m].tolist(), edge_index[1, m].tolist()):
        if h in row_pos and t in col_pos:
            M[row_pos[h], col_pos[t]] = True
    return M


def shared_mask(M):
    """True en (i,j) si i y j comparten al menos una columna. Diagonal en False."""
    S = (M.astype(np.int16) @ M.astype(np.int16).T) > 0
    np.fill_diagonal(S, False)
    return S


def sample_triplets(shared, rng, max_pos_per_anchor):
    """(ancla, positivo, negativo) por ejercicio, indices locales."""
    a, p, n = [], [], []
    n_ex = shared.shape[0]
    for i in range(n_ex):
        pos = np.flatnonzero(shared[i])
        neg = np.flatnonzero(~shared[i])
        neg = neg[neg != i]
        if len(pos) == 0 or len(neg) == 0:
            continue
        k = min(max_pos_per_anchor, len(pos))
        for j in rng.choice(pos, size=k, replace=False):
            a.append(i); p.append(int(j)); n.append(int(rng.choice(neg)))
    return np.array(a), np.array(p), np.array(n)


def margin_loss(z_ex, triplets, margin, mode="pairwise"):
    a, p, n = triplets
    if len(a) == 0:
        return torch.zeros((), device=z_ex.device)
    pos = (z_ex[a] * z_ex[p]).sum(dim=1)
    neg = (z_ex[a] * z_ex[n]).sum(dim=1)
    if mode == "batch":                       # comportamiento de los scripts originales
        return F.relu(margin - pos.mean() + neg.mean())
    return F.relu(margin - pos + neg).mean()


# =====================================================================
# 6. Metricas
# =====================================================================

def link_metrics(scores_pos, scores_neg):
    y = np.concatenate([np.ones(len(scores_pos)), np.zeros(len(scores_neg))])
    s = np.concatenate([scores_pos, scores_neg])
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    return roc_auc_score(y, s), average_precision_score(y, s)


def semantic_metrics(z_ex, related, k=5):
    """Delta similarity y Retrieval@k sobre similitud coseno."""
    zn = F.normalize(z_ex, p=2, dim=1)
    S = (zn @ zn.t()).cpu().numpy()
    np.fill_diagonal(S, -np.inf)

    rel = related
    unrel = ~related
    np.fill_diagonal(unrel, False)

    delta = float(S[rel].mean() - S[unrel].mean()) if rel.any() and unrel.any() else float("nan")

    topk = np.argsort(-S, axis=1)[:, :k]
    hits = np.take_along_axis(rel, topk, axis=1)
    retrieval = float(hits.mean())
    return delta, retrieval


def retrieval_from_similarity(S, related, k=5):
    S = S.copy().astype(float)
    np.fill_diagonal(S, -np.inf)
    topk = np.argsort(-S, axis=1)[:, :k]
    return float(np.take_along_axis(related, topk, axis=1).mean())


def jaccard_similarity(M):
    A = M.astype(np.float32)
    inter = A @ A.T
    sizes = A.sum(axis=1, keepdims=True)
    union = sizes + sizes.T - inter
    return np.divide(inter, np.maximum(union, 1e-9))


def safe_silhouette(z_ex, labels):
    labels = np.asarray(labels)
    keep = pd.Series(labels).groupby(labels).transform("size").values > 1
    if keep.sum() < 3 or len(np.unique(labels[keep])) < 2:
        return float("nan")
    return float(silhouette_score(
        F.normalize(z_ex, p=2, dim=1).cpu().numpy()[keep], labels[keep], metric="cosine"))


# =====================================================================
# 7. Un experimento = (modelo, lambda, semilla)
# =====================================================================

def run_once(cfg, G, model_id, lam, seed):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    device = cfg.device

    edge_index, edge_type, x = G["edge_index"], G["edge_type"], G["x"]
    num_nodes = x.size(0)
    num_rel = int(edge_type.max()) + 1

    parts = split_edges(edge_index, edge_type, cfg.val_ratio, cfg.test_ratio, seed)
    tr_idx, tr_typ = parts["train"]

    # grafo de propagacion: solo aristas de train, mas sus inversas
    if cfg.add_inverse:
        mp_idx, mp_typ = add_inverse_relations(tr_idx, tr_typ, num_rel)
        num_rel_mp = num_rel * 2
    else:
        mp_idx, mp_typ, num_rel_mp = tr_idx, tr_typ, num_rel

    sampler = NegativeSampler(edge_index, edge_type, num_nodes, num_rel)

    encoder_kind, decoder_kind, _ = MODEL_ZOO[model_id]
    model = Model(encoder_kind, decoder_kind, num_nodes, x.size(1), cfg.hidden_dim,
                  cfg.out_dim, num_rel_mp, num_rel, cfg.num_bases, cfg.dropout,
                  normalize=cfg.normalize, learnable_input=cfg.learnable_input,
                  input_dim=cfg.input_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    x_d, mp_idx_d, mp_typ_d = x.to(device), mp_idx.to(device), mp_typ.to(device)

    # ---- pares funcionales, construidos SOLO con aristas de train ----
    exercises = exercise_nodes(G, cfg)
    muscles = sorted({int(t) for t in tr_idx[1][tr_typ == G["rel2id"]["targets"]].tolist()})
    M_train = incidence_matrix(tr_idx, tr_typ, G["rel2id"]["targets"], exercises, muscles)
    shared_train = shared_mask(M_train)

    best_val, best_state, patience = -np.inf, None, 0
    for epoch in range(1, cfg.epochs + 1):
        model.train(); opt.zero_grad()
        z = model(x_d, mp_idx_d, mp_typ_d)

        neg_idx, _ = sampler.sample(tr_idx, tr_typ)
        pos_s = model.decode(z, tr_idx.to(device), tr_typ.to(device))
        neg_s = model.decode(z, neg_idx.to(device), tr_typ.to(device))
        link_loss = F.binary_cross_entropy_with_logits(
            torch.cat([pos_s, neg_s]),
            torch.cat([torch.ones_like(pos_s), torch.zeros_like(neg_s)]))

        loss = link_loss
        if lam > 0:
            trip = sample_triplets(shared_train, rng, cfg.pairs_per_anchor)
            loss = loss + lam * margin_loss(z[exercises], trip, cfg.margin, cfg.loss_mode)

        loss.backward(); opt.step()

        # ---- early stopping sobre VAL ----
        model.eval()
        with torch.no_grad():
            z = model(x_d, mp_idx_d, mp_typ_d)
            v_idx, v_typ = parts["val"]
            v_neg, _ = sampler.sample(v_idx, v_typ)
            auc, _ = link_metrics(
                model.decode(z, v_idx.to(device), v_typ.to(device)).cpu().numpy(),
                model.decode(z, v_neg.to(device), v_typ.to(device)).cpu().numpy())

        if auc > best_val:
            best_val, patience = auc, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= cfg.patience:
                break

    # ---- evaluacion final: UNA sola vez, sobre TEST ----
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        z = model(x_d, mp_idx_d, mp_typ_d)
        te_idx, te_typ = parts["test"]
        te_neg, failed = sampler.sample(te_idx, te_typ)
        pos_s = model.decode(z, te_idx.to(device), te_typ.to(device)).cpu().numpy()
        neg_s = model.decode(z, te_neg.to(device), te_typ.to(device)).cpu().numpy()

    auc, ap = link_metrics(pos_s, neg_s)

    id2rel = {v: k for k, v in G["rel2id"].items()}
    per_rel = {}
    for r in te_typ.unique().tolist():
        m = (te_typ == r).numpy()
        r_auc, r_ap = link_metrics(pos_s[m], neg_s[m])
        per_rel[id2rel.get(int(r), str(r))] = dict(
            auc=r_auc, ap=r_ap,
            neg_fail=float(failed[te_typ == r].float().mean()),
            n_test=int(m.sum()))

    # ---- guardar embeddings para la figura UMAP ----
    if cfg.save_emb and (not cfg.save_seeds or seed in cfg.save_seeds):
        import os
        os.makedirs(cfg.save_emb, exist_ok=True)
        fname = f"emb_{model_id}_lam{lam:g}_seed{seed}.pt"
        torch.save({"z": z.detach().cpu(),
                    "model": model_id, "lam": lam, "seed": seed,
                    "exercise_index": torch.tensor(exercises)},
                   os.path.join(cfg.save_emb, fname))

    # ---- metricas semanticas con dos ground truths independientes ----
    z_ex = z[exercises].detach().cpu()

    muscles_all = sorted({int(t) for t in
                          edge_index[1][edge_type == G["rel2id"]["targets"]].tolist()})
    M_all = incidence_matrix(edge_index, edge_type, G["rel2id"]["targets"], exercises, muscles_all)
    related_muscle = shared_mask(M_all)

    types_all = sorted({int(t) for t in
                        edge_index[1][edge_type == G["rel2id"]["hasType"]].tolist()})
    T_all = incidence_matrix(edge_index, edge_type, G["rel2id"]["hasType"], exercises, types_all)
    related_type = shared_mask(T_all)

    d_mus, r_mus = semantic_metrics(z_ex, related_muscle, cfg.topk)
    d_typ, r_typ = semantic_metrics(z_ex, related_type, cfg.topk)

    type_labels = np.array([np.argmax(row) if row.any() else -1 for row in T_all])

    return dict(
        model=model_id, label=MODEL_ZOO[model_id][2], lam=lam, seed=seed,
        params=sum(p.numel() for p in model.parameters()),
        val_auc=best_val, auc=auc, ap=ap,
        delta_muscle=d_mus, retrieval_muscle=r_mus,
        delta_type=d_typ, retrieval_type=r_typ,
        silhouette_type=safe_silhouette(z_ex, type_labels),
        per_relation=json.dumps(per_rel),
    )


def jaccard_baseline(G, cfg):
    """Baseline sin aprendizaje: similitud de Jaccard sobre conjuntos de musculos."""
    edge_index, edge_type = G["edge_index"], G["edge_type"]
    exercises = exercise_nodes(G, cfg)

    muscles = sorted({int(t) for t in edge_index[1][edge_type == G["rel2id"]["targets"]].tolist()})
    M = incidence_matrix(edge_index, edge_type, G["rel2id"]["targets"], exercises, muscles)
    types = sorted({int(t) for t in edge_index[1][edge_type == G["rel2id"]["hasType"]].tolist()})
    T = incidence_matrix(edge_index, edge_type, G["rel2id"]["hasType"], exercises, types)

    S = jaccard_similarity(M)
    return dict(
        model="jaccard", label="Jaccard on target-muscle sets (no learning)",
        lam=float("nan"), seed=0, params=0,
        val_auc=float("nan"), auc=float("nan"), ap=float("nan"),
        delta_muscle=float("nan"),
        retrieval_muscle=retrieval_from_similarity(S, shared_mask(M), cfg.topk),
        delta_type=float("nan"),
        retrieval_type=retrieval_from_similarity(S, shared_mask(T), cfg.topk),
        silhouette_type=float("nan"),
        per_relation="{}",
    )


# =====================================================================
# 8. main
# =====================================================================

def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--graph", default="graph_data.pt")
    ap_.add_argument("--nodes", default="nodes.csv")
    ap_.add_argument("--rel-map", dest="rel_map", default="edge_rel_mapping.json")
    ap_.add_argument("--out", default="results_raw.csv")
    ap_.add_argument("--models", nargs="+", default=list(MODEL_ZOO))
    ap_.add_argument("--lambdas", nargs="+", type=float, default=[0.0, 0.3])
    ap_.add_argument("--runs", type=int, default=30)
    ap_.add_argument("--epochs", type=int, default=150)
    ap_.add_argument("--patience", type=int, default=20)
    ap_.add_argument("--hidden-dim", dest="hidden_dim", type=int, default=64)
    ap_.add_argument("--out-dim", dest="out_dim", type=int, default=64)
    ap_.add_argument("--num-bases", dest="num_bases", type=int, default=None)
    ap_.add_argument("--dropout", type=float, default=0.3)
    ap_.add_argument("--lr", type=float, default=1e-3)
    ap_.add_argument("--weight-decay", dest="weight_decay", type=float, default=1e-5)
    ap_.add_argument("--margin", type=float, default=0.3)
    ap_.add_argument("--pairs-per-anchor", dest="pairs_per_anchor", type=int, default=20)
    ap_.add_argument("--loss-mode", dest="loss_mode", choices=["pairwise", "batch"],
                     default="pairwise")
    ap_.add_argument("--val-ratio", dest="val_ratio", type=float, default=0.1)
    ap_.add_argument("--test-ratio", dest="test_ratio", type=float, default=0.2)
    ap_.add_argument("--topk", type=int, default=5)
    ap_.add_argument("--exercise-label", dest="exercise_label", default="Exercise",
                     help="valor de la columna 'type' de nodes.csv para los ejercicios")
    ap_.add_argument("--no-normalize", dest="normalize", action="store_false")
    ap_.add_argument("--raw-features", dest="learnable_input", action="store_false",
                     help="usar data.x tal cual en gcn/rgcn en vez de embeddings "
                          "de entrada aprendibles (comparacion menos justa)")
    ap_.add_argument("--input-dim", dest="input_dim", type=int, default=64,
                     help="dimension de los embeddings de entrada aprendibles")
    ap_.add_argument("--save-emb", dest="save_emb", default=None,
                     help="carpeta donde guardar los embeddings aprendidos (un .pt por "
                          "modelo, lambda y semilla). Sin esto no se guarda nada.")
    ap_.add_argument("--save-seeds", dest="save_seeds", nargs="*", type=int, default=[0],
                     help="semillas cuyos embeddings se guardan; vacio guarda todas")
    ap_.add_argument("--no-inverse", dest="add_inverse", action="store_false")
    ap_.add_argument("--keep-duplicates", dest="dedup", action="store_false",
                     help="NO colapsar las aristas espejo (reproduce la fuga train/test)")
    ap_.add_argument("--smoke", action="store_true",
                     help="grafo sintetico, 2 semillas, 5 epochs")
    cfg = ap_.parse_args()
    cfg.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if cfg.smoke:
        G = synthetic_graph()
        cfg.runs, cfg.epochs, cfg.patience = 2, 5, 5
        cfg.out = "smoke_results.csv"
        print(">> MODO SMOKE: grafo sintetico, resultados sin valor cientifico")
    else:
        G = load_graph(cfg)

    if cfg.dedup:
        G["edge_index"], G["edge_type"], removed = deduplicate_undirected(
            G["edge_index"], G["edge_type"])
        print(f">> aristas espejo colapsadas: {removed} eliminadas, "
              f"{G['edge_index'].size(1)} aserciones unicas")

    print(f"nodos={G['x'].size(0)}  aristas={G['edge_index'].size(1)}  "
          f"relaciones={int(G['edge_type'].max()) + 1}  x.shape={tuple(G['x'].shape)}")
    print(f"device={cfg.device}  inversas={cfg.add_inverse}  normalize={cfg.normalize}\n")

    rows = [jaccard_baseline(G, cfg)]
    t0 = time.time()
    for model_id in cfg.models:
        for lam in cfg.lambdas:
            for seed in range(cfg.runs):
                r = run_once(cfg, G, model_id, lam, seed)
                rows.append(r)
                print(f"[{time.time() - t0:7.1f}s] {model_id:13s} lam={lam:<4} seed={seed:<3} "
                      f"AUC={r['auc']:.4f} AP={r['ap']:.4f} "
                      f"dMus={r['delta_muscle']:.4f} R@5mus={r['retrieval_muscle']:.4f} "
                      f"R@5typ={r['retrieval_type']:.4f}")
                pd.DataFrame(rows).to_csv(cfg.out, index=False)

    print(f"\nListo -> {cfg.out}")


if __name__ == "__main__":
    main()