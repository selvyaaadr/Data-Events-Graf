
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# 1. Membaca dataset
df = pd.read_csv("training.csv")

print("Ukuran dataset:", df.shape)
print("Kolom dataset:", df.columns.tolist())

# 2. Mengambil interaksi VIEW
views = df[
    df["event"].astype(str).str.lower().eq("view")
][["visitorid", "itemid"]].drop_duplicates()

print("Jumlah interaksi VIEW unik:", len(views))

# 3. Memilih 100 item yang paling sering dilihat
top_items = views["itemid"].value_counts().head(100).index
sample = views[views["itemid"].isin(top_items)]

# 4. Membentuk matriks Item-User
matrix = pd.crosstab(
    sample["itemid"],
    sample["visitorid"]
)

print("Dimensi matriks Item-User:", matrix.shape)

# 5. Menghitung Cosine Similarity antar-item
similarity = cosine_similarity(matrix)

# 6. Membuat graf Item-Item
G = nx.Graph()

items = matrix.index.tolist()
G.add_nodes_from(items)

# Ambang similarity agar graf tidak terlalu padat
threshold = 0.10

for i in range(len(items)):
    for j in range(i + 1, len(items)):
        score = similarity[i, j]

        if score >= threshold:
            G.add_edge(
                items[i],
                items[j],
                weight=float(score)
            )

print("Jumlah simpul:", G.number_of_nodes())
print("Jumlah sisi:", G.number_of_edges())

# 7. Menampilkan graf
plt.figure(figsize=(12, 9))

pos = nx.spring_layout(G, seed=42)

nx.draw_networkx(
    G,
    pos,
    with_labels=False,
    node_size=80,
    width=0.8
)

plt.title("Graf Item-Item berdasarkan Cosine Similarity")
plt.axis("off")
plt.tight_layout()
plt.show()

# 8. Menampilkan 5 sisi dengan similarity tertinggi
edges = sorted(
    G.edges(data=True),
    key=lambda edge: edge[2]["weight"],
    reverse=True
)

print("\nTop 5 pasangan item paling mirip:")
for item1, item2, data in edges[:5]:
    print(
        f"Item {item1} - Item {item2}: "
        f"similarity = {data['weight']:.4f}"
    )
