import torch
import umap
import pandas as pd
import matplotlib.pyplot as plt

# load
emb = torch.load("fitkg_output/rgcn_embeddings.pt").numpy()
nodes = pd.read_csv(r"C:\Users\javie\Desktop\Maestria\KG-OccFitness\fitkg_lite_output\nodes.csv")

# reduce
reducer = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    n_components=2,
    random_state=42
)
emb_2d = reducer.fit_transform(emb)

# dataframe
df = pd.DataFrame({
    "x": emb_2d[:,0],
    "y": emb_2d[:,1],
    "type": nodes["type"],
    "name": nodes["name"]
})


#PLOTEO SOLO EJERCICIOS

# plt.figure(figsize=(10,8))

# df_ex = df[df["type"] == "Exercise"]

# plt.scatter(df_ex["x"], df_ex["y"], alpha=0.6, s=20)

# plt.title("Exercise embeddings (R-GCN + KG)")
# plt.xlabel("UMAP-1")
# plt.ylabel("UMAP-2")
# plt.show()


# PLOTEO GENERAL

for t in df["type"].unique():
    subset = df[df["type"] == t]
    plt.scatter(subset["x"], subset["y"], label=t, alpha=0.6, s=20)

plt.legend()
plt.title("Node embeddings (R-GCN + KG)")
plt.show()