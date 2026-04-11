# training_rgcn.py
import torch
import torch.nn.functional as F
from torch_geometric.nn import to_hetero
from torch_geometric.utils import negative_sampling
from gcn_model import GCN
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Cargando grafo")

data = torch.load(r"C:\Users\javie\Desktop\Maestria\KG-OccFitness\fitkg_lite_output\graph_data.pt").to(device)

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
model = GCN(
    in_dim=in_dim,
    hidden_dim=128,
    out_dim=128
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

# Entrenamiento
epochs = 150
for epoch in range(1, epochs + 1):
    optimizer.zero_grad()

    #z = model(x, edge_index)

    # 80/20
    z = model(x, train_edge_index)

    # Edge positive samples
    #pos_edge_index = edge_index

    # 80/20
    pos_edge_index = train_edge_index

    # Negative samples
    # neg_edge_index = negative_sampling(
    #     edge_index=pos_edge_index,
    #     num_nodes=num_nodes,
    #     num_neg_samples=pos_edge_index.size(1)
    # )

    # 80/20
    neg_edge_index = negative_sampling(
        edge_index=train_edge_index,
        num_nodes=num_nodes,
        num_neg_samples=train_edge_index.size(1)
    )

    # scores producto punto
    pos_scores = (z[pos_edge_index[0]] * z[pos_edge_index[1]]).sum(dim=1)
    neg_scores = (z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=1)

    # perdida BCE 
    loss = F.binary_cross_entropy_with_logits(
        torch.cat([pos_scores, neg_scores]),
        torch.cat([
            torch.ones_like(pos_scores),
            torch.zeros_like(neg_scores)
        ]).float()
    )

    loss.backward()
    optimizer.step()

    if epoch % 10 == 0 or epoch == 1:
        acc = ((torch.sigmoid(torch.cat([pos_scores, neg_scores])) > 0.5)
               == torch.cat([
                   torch.ones_like(pos_scores),
                   torch.zeros_like(neg_scores)
               ]).bool()).float().mean().item()

        print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | Acc: {acc:.4f}")

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

model.eval()
with torch.no_grad():
    z = model(x, train_edge_index)

auc, ap = evaluate_link_prediction(z, test_edge_index, num_nodes)

print(f"\nTest AUC: {auc:.4f}")
print(f"Test AP: {ap:.4f}")

# Guardar embeddings
torch.save(z.detach().cpu(), "fitkg_output/gcn_embeddings.pt")
print("Embeddings guardados en: fitkg_output/gcn_embeddings.pt")
