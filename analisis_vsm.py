# membatasi visualisasi pada 100 User dan 100 Item paling aktif supaya tidak mencoba menggambar puluhan ribu simpul sekaligus. Matriks VSM awal tetap dibentuk dari seluruh interaksi view yang tersedia pada training.csv.

from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

# =========================================
# 1. MEMBACA DATA TRAINING
# =========================================
BASE_DIR = Path(__file__).resolve().parent
FILE = BASE_DIR / "training.csv"

df = pd.read_csv(FILE)

# Normalisasi nama kolom
df.columns = df.columns.str.lower().str.strip()

required = {"visitorid", "itemid", "event"}
if not required.issubset(df.columns):
    raise ValueError(
        f"Kolom wajib tidak lengkap. Kolom tersedia: {df.columns.tolist()}"
    )

# Gunakan event view untuk VSM awal
views = df[
    df["event"].astype(str).str.lower().str.strip() == "view"
].copy()

views = views.dropna(subset=["visitorid", "itemid"])
views["visitorid"] = pd.to_numeric(views["visitorid"], errors="coerce")
views["itemid"] = pd.to_numeric(views["itemid"], errors="coerce")
views = views.dropna(subset=["visitorid", "itemid"])

views["visitorid"] = views["visitorid"].astype("int64")
views["itemid"] = views["itemid"].astype("int64")

# Satu pasangan user-item dihitung sebagai satu interaksi
interaksi = views[["visitorid", "itemid"]].drop_duplicates()

# =========================================
# 2. MEMBENTUK MATRIKS VSM USER-ITEM
# =========================================
users = interaksi["visitorid"].unique()
items = interaksi["itemid"].unique()

user_map = {u: idx for idx, u in enumerate(users)}
item_map = {i: idx for idx, i in enumerate(items)}

baris = interaksi["visitorid"].map(user_map).to_numpy()
kolom = interaksi["itemid"].map(item_map).to_numpy()
nilai = np.ones(len(interaksi), dtype=np.float32)

X = csr_matrix(
    (nilai, (baris, kolom)),
    shape=(len(users), len(items))
)

print("\n=== KARAKTERISTIK VSM ===")
print("Jumlah baris interaksi unik:", len(interaksi))
print("Jumlah User:", len(users))
print("Jumlah Item:", len(items))
print("Shape matriks User-Item:", X.shape)
print("Jumlah elemen non-zero:", X.nnz)
print("Sparsity:", round(1 - X.nnz / (X.shape[0] * X.shape[1]), 6))

# =========================================
# 3. MEMBANDINGKAN UKURAN PROYEKSI
# =========================================
# A_user = X @ X.T
# A_item = X.T @ X
#
# Tidak menghitung kedua matriks secara penuh karena
# jumlah user dan item sangat besar.

print("\n=== DIMENSI PROYEKSI TEORITIS ===")
print("Proyeksi User-User:", (X.shape[0], X.shape[0]))
print("Proyeksi Item-Item:", (X.shape[1], X.shape[1]))

# =========================================
# 4. AMBIL SAMPEL UNTUK VISUALISASI
# =========================================
# Ambil user yang paling aktif agar graf sampel informatif.
jumlah_user_aktif = np.asarray(X.sum(axis=1)).ravel()
jumlah_item_aktif = np.asarray(X.sum(axis=0)).ravel()

N_USER = min(100, X.shape[0])
N_ITEM = min(100, X.shape[1])

user_idx = np.argsort(jumlah_user_aktif)[-N_USER:]
item_idx = np.argsort(jumlah_item_aktif)[-N_ITEM:]

X_user_sample = X[user_idx, :]
X_item_sample = X[:, item_idx]

# Normalisasi baris untuk cosine similarity
X_user_norm = normalize(X_user_sample, norm="l2", axis=1)
X_item_norm = normalize(X_item_sample.T, norm="l2", axis=1)

# Proyeksi hanya pada sampel terpilih
S_user = (X_user_norm @ X_user_norm.T).toarray()
S_item = (X_item_norm @ X_item_norm.T).toarray()

# =========================================
# 5. MEMBUAT GRAF PROYEKSI
# =========================================
def graph_from_similarity(sim, labels, threshold=0.10):
    G = nx.Graph()
    G.add_nodes_from(labels)

    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            score = sim[i, j]
            if score >= threshold:
                G.add_edge(
                    labels[i],
                    labels[j],
                    weight=float(score)
                )
    return G

user_labels = [int(users[i]) for i in user_idx]
item_labels = [int(items[i]) for i in item_idx]

G_user = graph_from_similarity(S_user, user_labels)
G_item = graph_from_similarity(S_item, item_labels)

print("\n=== HASIL PROYEKSI SAMPEL ===")
print("User-User | simpul:", G_user.number_of_nodes(),
      "| relasi:", G_user.number_of_edges())
print("Item-Item | simpul:", G_item.number_of_nodes(),
      "| relasi:", G_item.number_of_edges())

# =========================================
# 6. VISUALISASI GRAF SAMPEL
# =========================================
for G, judul, nama_file in [
    (G_user, "Proyeksi User-User (Sampel)", "graf_user_user.png"),
    (G_item, "Proyeksi Item-Item (Sampel)", "graf_item_item.png")
]:
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)

    nx.draw_networkx(
        G,
        pos,
        with_labels=False,
        node_size=35,
        width=0.5
    )

    plt.title(judul)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(BASE_DIR / nama_file, dpi=200)
    plt.show()

# =========================================
# 7. SIMILARITY TOP-5 UNTUK CONTOH SAMPEL
# =========================================
def top_similarities(sim, labels, target_position, top_n=5):
    scores = sim[target_position].copy()
    scores[target_position] = -1

    urutan = np.argsort(scores)[::-1][:top_n]

    return pd.DataFrame({
        "simpul_acuan": labels[target_position],
        "simpul_mirip": [labels[i] for i in urutan],
        "cosine_similarity": [scores[i] for i in urutan]
    })

if len(user_labels) > 1:
    hasil_user = top_similarities(
        S_user, user_labels, target_position=0
    )
    print("\nTop-5 User mirip dengan user acuan:")
    print(hasil_user.to_string(index=False))
    hasil_user.to_csv(
        BASE_DIR / "similarity_user_top5.csv", index=False
    )

if len(item_labels) > 1:
    hasil_item = top_similarities(
        S_item, item_labels, target_position=0
    )
    print("\nTop-5 Item mirip dengan item acuan:")
    print(hasil_item.to_string(index=False))
    hasil_item.to_csv(
        BASE_DIR / "similarity_item_top5.csv", index=False
    )

print("\nSelesai. File graf dan hasil similarity disimpan di folder proyek.")
