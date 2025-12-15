# training_rgcn.py
import torch
import torch.nn.functional as F
from torch_geometric.nn import to_hetero
from torch_geometric.utils import negative_sampling
from rgcn_model import RGCN

#Cargamos archivo tensor

#GRAPH_PT = "fitkg_output/graph_data_hetero.pt"
#GRAPH_PT = r"C:\Users\javie\Desktop\Maestria\KG-OccFitness\fitkg_lite_output\graph_data.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#data = torch.load(GRAPH_PT).to(device)

# Filtramos el grafo para quedarnos solo con Exercise como nodo principal
# (PyG requiere un grafo homogéneo para R-GCN)

# data_homo = data.to_homogeneous(edge_type_attr='relation')
# edge_index = data_homo.edge_index
# edge_type = data_homo.relation  # cada arista tiene tipo relacional codificado como entero

# x = data_homo.x
# num_nodes = x.size(0)
# num_relations = int(edge_type.max().item()) + 1
# in_dim = x.size(1)

print("Cargando grafo...")

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

# Modelo
model = RGCN(
    num_nodes=num_nodes,
    in_dim=in_dim,
    hidden_dim=128,
    out_dim=128,
    num_relations=num_relations
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

# Entrenamiento
epochs = 150
for epoch in range(1, epochs + 1):
    optimizer.zero_grad()

    z = model(x, edge_index, edge_type)

    # Edge positive samples
    pos_edge_index = edge_index

    # Negative samples
    neg_edge_index = negative_sampling(
        edge_index=pos_edge_index,
        num_nodes=num_nodes,
        num_neg_samples=pos_edge_index.size(1)
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

# Guardar embeddings
torch.save(z.detach().cpu(), "fitkg_output/rgcn_embeddings.pt")
print("Embeddings guardados en: fitkg_output/rgcn_embeddings.pt")
