
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# ==================================================
# 1. MEMBACA DATA
# ==================================================
train = pd.read_csv("training.csv")
test = pd.read_csv("testing.csv")

# Memastikan kolom yang dibutuhkan tersedia
required = {"visitorid", "itemid"}
if not required.issubset(train.columns) or not required.issubset(test.columns):
    raise ValueError(
        "Pastikan training.csv dan testing.csv memiliki "
        "kolom visitorid dan itemid."
    )

# ==================================================
# 2. MEMBENTUK MATRIKS USER-ITEM
# ==================================================
# Interaksi yang sama dihitung satu kali
interactions = train[["visitorid", "itemid"]].drop_duplicates()

users = np.sort(interactions["visitorid"].unique())
items = np.sort(interactions["itemid"].unique())

user_to_idx = {u: i for i, u in enumerate(users)}
item_to_idx = {it: i for i, it in enumerate(items)}

rows = interactions["visitorid"].map(user_to_idx).to_numpy()
cols = interactions["itemid"].map(item_to_idx).to_numpy()
values = np.ones(len(interactions), dtype=np.float32)

user_item = csr_matrix(
    (values, (rows, cols)),
    shape=(len(users), len(items))
)

# ==================================================
# 3. MEMILIH USER SECARA ACAK DAN REPRODUCIBLE
# ==================================================
# User harus ada pada training dan testing
test_users = set(test["visitorid"].unique())
eligible_users = np.array([
    u for u in users if u in test_users
])

if len(eligible_users) == 0:
    raise ValueError(
        "Tidak ada user yang muncul pada training dan testing."
    )

RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)
target_user = rng.choice(eligible_users)

print("Random state:", RANDOM_STATE)
print("Jumlah kandidat user:", len(eligible_users))
print("User terpilih:", target_user)

# ==================================================
# 4. MENENTUKAN 5 NEIGHBOR TERDEKAT (K = 5)
# ==================================================
target_idx = user_to_idx[target_user]

# Meminta 6 kandidat: user itu sendiri + 5 tetangga
knn = NearestNeighbors(
    metric="cosine",
    algorithm="brute",
    n_neighbors=min(6, len(users))
)
knn.fit(user_item)

distances, indices = knn.kneighbors(
    user_item[target_idx],
    n_neighbors=min(6, len(users))
)

neighbors = []
for distance, idx in zip(distances[0], indices[0]):
    if idx != target_idx:
        neighbors.append((idx, 1 - distance))
    if len(neighbors) == 5:
        break

print("\n5 tetangga terdekat:")
for idx, similarity in neighbors:
    print(
        "Visitor:", users[idx],
        "| Cosine similarity:", round(float(similarity), 4)
    )

# ==================================================
# 5. MEMBUAT REKOMENDASI ITEM
# ==================================================
# Item yang sudah berinteraksi dengan target tidak direkomendasikan
seen_items = set(
    interactions.loc[
        interactions["visitorid"] == target_user, "itemid"
    ]
)

# Skor item = jumlah similarity tetangga yang berinteraksi dengannya
scores = {}

for neighbor_idx, similarity in neighbors:
    if similarity <= 0:
        continue

    neighbor_items = user_item.getrow(neighbor_idx).indices

    for item_idx in neighbor_items:
        item_id = items[item_idx]

        if item_id not in seen_items:
            scores[item_id] = (
                scores.get(item_id, 0.0) + similarity
            )

ranked_items = sorted(
    scores.items(),
    key=lambda x: (-x[1], x[0])
)

top5 = ranked_items[:5]

print("\nTop 5 rekomendasi:")
for rank, (item_id, score) in enumerate(top5, start=1):
    print(
        f"{rank}. Item {item_id} | Skor rekomendasi: {score:.4f}"
    )

print("\nTiga item rekomendasi teratas:")
for item_id, score in top5[:3]:
    print(f"Item {item_id} | Skor: {score:.4f}")

# ==================================================
# 6. EVALUASI MENGGUNAKAN DATA TESTING
# ==================================================
# Definisi relevan: item yang muncul dalam interaksi
# user terpilih pada testing.csv
actual_items = set(
    test.loc[test["visitorid"] == target_user, "itemid"]
)

recommended_items = [item for item, score in top5]
hits = [1 if item in actual_items else 0
        for item in recommended_items]

# Hit@5: 1 jika setidaknya satu rekomendasi cocok
hit_at_5 = int(sum(hits) > 0)

# NDCG@5 dengan relevansi biner
dcg = sum(
    rel / np.log2(rank + 1)
    for rank, rel in enumerate(hits, start=1)
)

ideal_hits = min(len(actual_items), 5)
idcg = sum(
    1 / np.log2(rank + 1)
    for rank in range(1, ideal_hits + 1)
)

ndcg_at_5 = dcg / idcg if idcg > 0 else 0.0

print("\nEvaluasi:")
print("Jumlah item relevan di testing:", len(actual_items))
print("Item aktual di testing:", sorted(actual_items))
print("Relevansi Top 5:", hits)
print("Hit@5:", hit_at_5)
print("NDCG@5:", round(ndcg_at_5, 4))
