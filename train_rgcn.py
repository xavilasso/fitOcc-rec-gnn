# training_rgcn.py
import torch
import torch.nn.functional as F
from torch_geometric.nn import to_hetero
from torch_geometric.utils import negative_sampling
from rgcn_model import RGCN
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

def split_edges(edge_index, edge_type, test_ratio=0.2, random_state=42):
    """
    Split estratificado por tipo de relación
    """
    edge_index = edge_index.t()  # [num_edges, 2]
    edge_type = edge_type

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


#Cargamos archivo tensor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Cargando grafo")

data = torch.load(r"C:\Users\javie\Desktop\Maestria\KG-OccFitness\fitkg_lite_output\graph_data.pt").to(device)

#Verificamos estructura correcta en el archivo pt (tensor)
#print(type(data))
#print(data)

x = data.x
edge_index = data.edge_index
edge_type = data.edge_type

num_nodes = x.size(0)
num_relations = int(edge_type.max().item()) + 1
in_dim = x.size(1)

print(f"Total nodos: {num_nodes}")
print(f"Dim entrada: {in_dim}, Relaciones: {num_relations}")

train_edge_index, train_edge_type, test_edge_index, test_edge_type = split_edges(
    edge_index, edge_type, test_ratio=0.2
)

print("Train edges:", train_edge_index.size(1))
print("Test edges:", test_edge_index.size(1))

# Modelo
model = RGCN(
    num_nodes=num_nodes,
    in_dim=in_dim,
    hidden_dim=64,
    out_dim=64,
    num_relations=num_relations
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

# Entrenamiento

lambda_func = 0.3
best_auc = 0
patience = 20
patience_counter = 0
epochs = 150

for epoch in range(1, epochs + 1):

    model.train()
    optimizer.zero_grad()

    z = model(x, train_edge_index, train_edge_type)

    # -------- LINK LOSS --------
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

    # -------- FUNCTIONAL LOSS --------
    func_loss = functional_loss(z, pos_pairs, neg_pairs)

    total_loss = link_loss + lambda_func * func_loss

    total_loss.backward()
    optimizer.step()

    # -------- VALIDATION --------
    model.eval()
    with torch.no_grad():
        z = model(x, train_edge_index, train_edge_type)
        auc, ap = evaluate_link_prediction(
            z, test_edge_index, test_edge_type, num_nodes
        )

    print(f"Epoch {epoch:03d} | "
          f"Loss: {total_loss.item():.4f} | "
          f"AUC: {auc:.4f}")

    # -------- EARLY STOPPING --------
    if auc > best_auc:
        best_auc = auc
        patience_counter = 0
        torch.save(model.state_dict(), "best_rgcn.pt")
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print("Early stopping triggered")
        break


# epochs = 150
# for epoch in range(1, epochs + 1):
#     optimizer.zero_grad()

#     #z = model(x, edge_index, edge_type)

#     # 80/20
#     z = model(x, train_edge_index, train_edge_type)

#     # Edge positive samples
#     #pos_edge_index = edge_index

#     # Con el 80/20
#     pos_edge_index = train_edge_index


#     # Negative samples
#     # neg_edge_index = negative_sampling(
#     #     edge_index=pos_edge_index,
#     #     num_nodes=num_nodes,
#     #     num_neg_samples=pos_edge_index.size(1)
#     # )

#     # Con el 80/20
#     neg_edge_index = negative_sampling(
#         edge_index=pos_edge_index,
#         num_nodes=num_nodes,
#         num_neg_samples=pos_edge_index.size(1)
#     )


#     # scores producto punto
#     #pos_scores = (z[pos_edge_index[0]] * z[pos_edge_index[1]]).sum(dim=1)
#     pos_scores = model.decode(z, pos_edge_index, pos_edge_type)

#     neg_scores = (z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=1)

#     # perdida BCE 
#     loss = F.binary_cross_entropy_with_logits(
#         torch.cat([pos_scores, neg_scores]),
#         torch.cat([
#             torch.ones_like(pos_scores),
#             torch.zeros_like(neg_scores)
#         ]).float()
#     )

#     loss.backward()
#     optimizer.step()

#     if epoch % 10 == 0 or epoch == 1:
#         acc = ((torch.sigmoid(torch.cat([pos_scores, neg_scores])) > 0.5)
#                == torch.cat([
#                    torch.ones_like(pos_scores),
#                    torch.zeros_like(neg_scores)
#                ]).bool()).float().mean().item()

#         print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | Acc: {acc:.4f}")

def evaluate_link_prediction(z, pos_edge_index, num_nodes):
    # negative samples
    neg_edge_index = negative_sampling(
        edge_index=pos_edge_index,
        num_nodes=num_nodes,
        num_neg_samples=pos_edge_index.size(1)
    )

    pos_scores = (z[pos_edge_index[0]] * z[pos_edge_index[1]]).sum(dim=1)
    neg_scores = (z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=1)

    scores = torch.cat([pos_scores, neg_scores]).cpu().detach()
    labels = torch.cat([
        torch.ones_like(pos_scores),
        torch.zeros_like(neg_scores)
    ]).cpu()

    probs = torch.sigmoid(scores)

    auc = roc_auc_score(labels, probs)
    ap = average_precision_score(labels, probs)

    return auc, ap

# model.eval()
# with torch.no_grad():
#     z = model(x, train_edge_index, train_edge_type)

# auc, ap = evaluate_link_prediction(z, test_edge_index, num_nodes)

# print(f"\nTest AUC: {auc:.4f}")
# print(f"Test AP: {ap:.4f}")

model.load_state_dict(torch.load("best_rgcn.pt"))
model.eval()

with torch.no_grad():
    z = model(x, train_edge_index, train_edge_type)

auc, ap = evaluate_link_prediction(
    z, test_edge_index, test_edge_type, num_nodes
)

print("Final AUC:", auc)
print("Final AP:", ap)

# Guardar embeddings
torch.save(z.detach().cpu(), "fitkg_output/rgcn_embeddings.pt")
print("Embeddings guardados en: fitkg_output/rgcn_embeddings.pt")
