import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("training.csv")

# Ambil interaksi VIEW
views = df[df["event"] == "view"][
    ["visitorid", "itemid"]
].drop_duplicates()

# Pilih 100 user paling aktif untuk visualisasi
top_users = (
    views["visitorid"]
    .value_counts()
    .head(100)
    .index
)

sample = views[views["visitorid"].isin(top_users)]

matrix = pd.crosstab(
    sample["visitorid"],
    sample["itemid"]
)

similarity = cosine_similarity(matrix)

G = nx.Graph()
users = matrix.index.tolist()

for i, user in enumerate(users):
    G.add_node(user)

for i in range(len(users)):
    for j in range(i + 1, len(users)):
        if similarity[i, j] > 0:
            G.add_edge(
                users[i],
                users[j],
                weight=similarity[i, j]
            )

plt.figure(figsize=(12, 9))
pos = nx.spring_layout(G, seed=42)

nx.draw(
    G, pos,
    node_size=40,
    with_labels=False
)

plt.title("Proyeksi User-User berdasarkan Cosine Similarity")
plt.show()

print("Jumlah simpul:", G.number_of_nodes())
print("Jumlah sisi:", G.number_of_edges())