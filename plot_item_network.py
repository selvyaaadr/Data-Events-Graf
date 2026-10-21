import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("training.csv")

views = df[df["event"] == "view"][
    ["visitorid", "itemid"]
].drop_duplicates()

# Ambil 100 item paling sering dilihat
top_items = (
    views["itemid"]
    .value_counts()
    .head(100)
    .index
)

sample = views[views["itemid"].isin(top_items)]

matrix = pd.crosstab(
    sample["itemid"],
    sample["visitorid"]
)

similarity = cosine_similarity(matrix)

G = nx.Graph()
items = matrix.index.tolist()

for item in items:
    G.add_node(item)

for i in range(len(items)):
    for j in range(i + 1, len(items)):
        if similarity[i, j] > 0:
            G.add_edge(
                items[i],
                items[j],
                weight=similarity[i, j]
            )

plt.figure(figsize=(12, 9))
pos = nx.spring_layout(G, seed=42)

nx.draw(
    G, pos,
    node_size=40,
    with_labels=False
)

plt.title("Proyeksi Item-Item berdasarkan Cosine Similarity")
plt.show()

print("Jumlah simpul:", G.number_of_nodes())
print("Jumlah sisi:", G.number_of_edges())