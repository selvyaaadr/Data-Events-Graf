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

# 3. Memilih 100 pengunjung yang paling aktif melihat item
top_users = views["visitorid"].value_counts().head(100).index
sample = views[views["visitorid"].isin(top_users)]

# 4. Membentuk matriks User-Item
matrix = pd.crosstab(
    sample["visitorid"],
    sample["itemid"]
)

print("Dimensi matriks User-Item:", matrix.shape)

# 5. Menghitung Cosine Similarity antar-pengunjung
similarity = cosine_similarity(matrix)

# 6. Membuat graf User-User
G = nx.Graph()

users = matrix.index.tolist()
G.add_nodes_from(users)

# Ambang similarity
threshold = 0.10

for i in range(len(users)):
    for j in range(i + 1, len(users)):
        score = similarity[i, j]

        if score >= threshold:
            G.add_edge(
                users[i],
                users[j],
                weight=float(score)
            )

print("Jumlah simpul:", G.number_of_nodes())
print("Jumlah sisi:", G.number_of_edges())

# 7. Visualisasi graf
plt.figure(figsize=(12, 9))

pos = nx.spring_layout(G, seed=42)

nx.draw_networkx(
    G,
    pos,
    with_labels=False,
    node_size=80,
    width=0.8
)

plt.title("Graf User-User berdasarkan Cosine Similarity")
plt.axis("off")
plt.tight_layout()
plt.show()

# 8. Menampilkan 5 pasangan pengunjung paling mirip
edges = sorted(
    G.edges(data=True),
    key=lambda edge: edge[2]["weight"],
    reverse=True
)

print("\nTop 5 pasangan pengunjung paling mirip:")

for user1, user2, data in edges[:5]:
    print(
        f"User {user1} - User {user2}: "
        f"similarity = {data['weight']:.4f}"
    )