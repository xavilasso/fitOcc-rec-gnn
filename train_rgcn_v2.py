# training_rgcn_v2.py
import torch
import torch.nn.functional as F
from torch_geometric.nn import to_hetero
from torch_geometric.utils import negative_sampling
from rgcn_model import RGCN
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
import pandas as pd
import json
import numpy as np
import random

# FUNCIONES PARA CONTEXTUALIZAR NUESTRO TRAINING
def split_edges(edge_index, edge_type, test_ratio=0.2, random_state=None):

    edge_index = edge_index.t()

    train_edges = []
    train_types = []
    test_edges = []
    test_types = []

    for rel in edge_type.unique():

        mask = edge_type == rel
        rel_edges = edge_index[mask]
        rel_types = edge_type[mask]

        idx = torch.arange(rel_edges.size(0))

        train_idx, test_idx = train_test_split(
            idx.numpy(),
            test_size=test_ratio,
            random_state=random_state
        )

        train_edges.append(rel_edges[train_idx])
        train_types.append(rel_types[train_idx])
        test_edges.append(rel_edges[test_idx])
        test_types.append(rel_types[test_idx])

    train_edge_index = torch.cat(train_edges).t().contiguous()
    train_edge_type = torch.cat(train_types)

    test_edge_index = torch.cat(test_edges).t().contiguous()
    test_edge_type = torch.cat(test_types)

    return train_edge_index, train_edge_type, test_edge_index, test_edge_type

def build_exercise_muscle_map(edge_index, edge_type, nodes_df, muscle_rel_id):
    exercise_nodes = nodes_df[nodes_df["type"] == "Exercise"].index.tolist()
    exercise_set = set(exercise_nodes)

    muscle_map = {idx: set() for idx in exercise_nodes}

    mask = edge_type == muscle_rel_id
    rel_edges = edge_index[:, mask]

    for i in range(rel_edges.size(1)):
        src = rel_edges[0, i].item()
        dst = rel_edges[1, i].item()

        if src in exercise_set:
            muscle_map[src].add(dst)

    return muscle_map

def build_functional_pairs(muscle_map, num_neg=2):
    pos_pairs = []
    neg_pairs = []

    exercises = list(muscle_map.keys())

    for ex in exercises:
        same = [
            other for other in exercises
            if other != ex and len(muscle_map[ex] & muscle_map[other]) > 0
        ]

        diff = [
            other for other in exercises
            if len(muscle_map[ex] & muscle_map[other]) == 0
        ]

        for s in same:
            pos_pairs.append((ex, s))

        for _ in range(min(num_neg, len(diff))):
            neg = diff[torch.randint(0, len(diff), (1,)).item()]
            neg_pairs.append((ex, neg))

    return pos_pairs, neg_pairs

def functional_loss(z, pos_pairs, neg_pairs, margin=0.3):

    if len(pos_pairs) == 0:
        return torch.tensor(0.0, device=z.device)

    pos_sim = []
    neg_sim = []

    for (i, j) in pos_pairs:
        pos_sim.append((z[i] * z[j]).sum())

    for (i, j) in neg_pairs:
        neg_sim.append((z[i] * z[j]).sum())

    pos_sim = torch.stack(pos_sim)
    neg_sim = torch.stack(neg_sim)

    loss = F.relu(margin - pos_sim.mean() + neg_sim.mean())

    return loss

def evaluate_link_prediction(model, z, edge_index, edge_type, num_nodes):

    neg_edge_index = negative_sampling(
        edge_index=edge_index,
        num_nodes=num_nodes,
        num_neg_samples=edge_index.size(1)
    )

    neg_edge_type = edge_type.clone()

    pos_scores = model.decode(z, edge_index, edge_type)
    neg_scores = model.decode(z, neg_edge_index, neg_edge_type)

    scores = torch.cat([pos_scores, neg_scores]).cpu().detach()
    labels = torch.cat([
        torch.ones_like(pos_scores),
        torch.zeros_like(neg_scores)
    ]).cpu()

    probs = torch.sigmoid(scores)

    auc = roc_auc_score(labels, probs)
    ap = average_precision_score(labels, probs)

    return auc, ap


# ============================================================
# MAIN - EXPERIMENT CONFIG
# ============================================================

NUM_RUNS = 30
EPOCHS = 150
PATIENCE = 20
LAMBDA_FUNC = 0.3

results = []

# INICIALIZAMOS
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Cargando grafo")

data = torch.load(r"C:\Users\javie\Desktop\Maestria\KG-OccFitness\fitkg_lite_output\graph_data.pt").to(device)
nodes_df = pd.read_csv(r"C:\Users\javie\Desktop\Maestria\KG-OccFitness\fitkg_lite_output\nodes.csv")

# Relations
with open(r"C:\Users\javie\Desktop\Maestria\KG-OccFitness\fitkg_lite_output\edge_rel_mapping.json") as f:
    rel2id = json.load(f)

x = data.x
edge_index = data.edge_index
edge_type = data.edge_type

num_nodes = x.size(0)
num_relations = int(edge_type.max().item()) + 1
in_dim = x.size(1)

print(f"Nodos: {num_nodes}")
print(f"Relaciones: {num_relations}")
muscle_rel_id = rel2id["targets"]   # <-- AJUSTAR según tu KG, en este caso lo hacemos en relación a qué ejercicio afecta a qué músculo


for run in range(NUM_RUNS):

    print(f"\n================ R-GCN RUN {run} ================")

    # --------------------------------------------------------
    # 1️⃣ Seeds
    # --------------------------------------------------------
    torch.manual_seed(run)
    np.random.seed(run)
    random.seed(run)
    torch.cuda.manual_seed_all(run)
    train_edge_index, train_edge_type, test_edge_index, test_edge_type = split_edges(
        edge_index, edge_type, random_state=run
    )

    # Importante: usar SOLO train edges para functional map
    muscle_map = build_exercise_muscle_map(
        train_edge_index, train_edge_type, nodes_df, muscle_rel_id
    )

    pos_pairs, neg_pairs = build_functional_pairs(muscle_map)

    model = RGCN(
        #num_nodes=num_nodes,
        in_dim=in_dim,
        hidden_dim=64,
        out_dim=64,
        num_relations=num_relations
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-5
    )

    #lambda_func = 0.3
    best_auc = 0
    besgt_ap = 0
    #patience = 20
    patience_counter = 0
    #epochs = 150


    for epoch in range(1, EPOCHS + 1):

        model.train()
        optimizer.zero_grad()

        z = model(x, train_edge_index, train_edge_type)

        # LINK LOSS
        neg_edge_index = negative_sampling(
            edge_index=train_edge_index,
            num_nodes=num_nodes,
            num_neg_samples=train_edge_index.size(1)
        )

        neg_edge_type = train_edge_type.clone()

        pos_scores = model.decode(z, train_edge_index, train_edge_type)
        neg_scores = model.decode(z, neg_edge_index, neg_edge_type)

        link_loss = F.binary_cross_entropy_with_logits(
            torch.cat([pos_scores, neg_scores]),
            torch.cat([
                torch.ones_like(pos_scores),
                torch.zeros_like(neg_scores)
            ]).float()
        )

        # FUNCTIONAL LOSS
        func_loss = functional_loss(z, pos_pairs, neg_pairs)

        total_loss = link_loss + LAMBDA_FUNC * func_loss

        total_loss.backward()
        optimizer.step()

        # VALIDATION
        model.eval()
        with torch.no_grad():
            z = model(x, train_edge_index, train_edge_type)
            auc, ap = evaluate_link_prediction(
                model, z, test_edge_index, test_edge_type, num_nodes
            )

        print(f"Epoch {epoch:03d} | "
            f"Loss: {total_loss.item():.4f} | "
            f"AUC: {auc:.4f}")
        
        if auc > best_auc:
            best_auc = auc
            best_ap = ap
            patience_counter = 0
            torch.save(model.state_dict(), f"best_rgcn_run{run}.pt")
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            break

    print(f"Run {run} Final AUC: {best_auc:.4f}")

    results.append({
        "run": run,
        "auc": best_auc,
        "ap": best_ap
    })
    
    # --------------------------------------------------------
    # Guardar embeddings del mejor modelo del run
    # --------------------------------------------------------
    model.load_state_dict(torch.load(f"best_rgcn_run{run}.pt"))
    model.eval()

    with torch.no_grad():
        z = model(x, train_edge_index, train_edge_type)

    torch.save(
        z.detach().cpu(),
        f"./fitkg_output/rgcn_emb/rgcn_embeddings_run{run}.pt"
    )

    print(f"Embeddings guardados: rgcn_embeddings_run{run}.pt")

        # if auc > best_auc:
        #     best_auc = auc
        #     patience_counter = 0
        #     torch.save(model.state_dict(), "best_rgcn.pt")
        # else:
        #     patience_counter += 1

        # if patience_counter >= PATIENCE:
        #     print("Early stopping triggered")
        #     break


# ============================================================
# FINAL EVALUATION
# ============================================================

# model.load_state_dict(torch.load("best_rgcn.pt"))
# model.eval()

# with torch.no_grad():
#     z = model(x, train_edge_index, train_edge_type)

# auc, ap = evaluate_link_prediction(
#     model, z, test_edge_index, test_edge_type, num_nodes
# )

# print("Final AUC:", auc)
# print("Final AP:", ap)

# torch.save(z.detach().cpu(), "rgcn_embeddings.pt")
# print("Embeddings guardados.")

# ============================================================
# RESULTS ANALYSIS
# ============================================================

results_df = pd.DataFrame(results)

mean_auc = results_df["auc"].mean()
std_auc = results_df["auc"].std()
min_auc = results_df["auc"].min()
max_auc = results_df["auc"].max()

mean_ap = results_df["ap"].mean()
std_ap = results_df["ap"].std()
min_ap = results_df["ap"].min()
max_ap = results_df["ap"].max()

print("\n================ FINAL RESULTS ================")

print("\nAUC Metrics:")
print(f"Mean ± Std : {mean_auc:.4f} ± {std_auc:.4f}")
print(f"Min        : {min_auc:.4f}")
print(f"Max        : {max_auc:.4f}")
print(f"Range      : {(max_auc - min_auc):.4f}")

print("\nAP Metrics:")
print(f"Mean ± Std : {mean_ap:.4f} ± {std_ap:.4f}")
print(f"Min        : {min_ap:.4f}")
print(f"Max        : {max_ap:.4f}")
print(f"Range      : {(max_ap - min_ap):.4f}")

results_df.to_csv("rgcn_experiment_results.csv", index=False)

print("\nResultados guardados en rgcn_experiment_results.csv")