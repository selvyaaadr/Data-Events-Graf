
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

# Folder tempat script dan training.csv berada
FOLDER = Path(__file__).resolve().parent
FILE_DATA = FOLDER / "training.csv"

# Baca data
df = pd.read_csv(FILE_DATA)
df.columns = df.columns.str.strip().str.lower()

# Pastikan kolom yang dibutuhkan tersedia
kolom = {"visitorid", "itemid", "event"}
if not kolom.issubset(df.columns):
    raise ValueError(
        f"Kolom wajib {kolom}; kolom yang ditemukan: {list(df.columns)}"
    )

# Gunakan interaksi VIEW untuk membentuk VSM
df["event"] = df["event"].astype(str).str.strip().str.lower()
df = df[df["event"].isin(["view", "viewed"])].copy()

df["visitorid"] = pd.to_numeric(df["visitorid"], errors="coerce")
df["itemid"] = pd.to_numeric(df["itemid"], errors="coerce")
df = df.dropna(subset=["visitorid", "itemid"])
df = df.drop_duplicates(subset=["visitorid", "itemid"])

if df.empty:
    raise ValueError("Tidak ada interaksi VIEW yang valid di training.csv")

# Ambil 100 User dan 100 Item dengan interaksi terbanyak
top_users = df["visitorid"].value_counts().head(100).index
top_items = df["itemid"].value_counts().head(100).index

# Matriks User-Item untuk sampel User
du = df[df["visitorid"].isin(top_users)]
users = pd.Index(top_users)
items_u = pd.Index(sorted(du["itemid"].unique()))

user_matrix = csr_matrix(
    (
        np.ones(len(du)),
        (
            users.get_indexer(du["visitorid"]),
            items_u.get_indexer(du["itemid"])
        )
    ),
    shape=(len(users), len(items_u))
)

# Matriks Item-User untuk sampel Item
di = df[df["itemid"].isin(top_items)]
items = pd.Index(top_items)
users_i = pd.Index(sorted(di["visitorid"].unique()))

item_matrix = csr_matrix(
    (
        np.ones(len(di)),
        (
            items.get_indexer(di["itemid"]),
            users_i.get_indexer(di["visitorid"])
        )
    ),
    shape=(len(items), len(users_i))
)

def simpan_top5(matrix, ids, kolom_awal, kolom_akhir, nama_file):
    # Hitung similarity cosine pada sampel
    sim = cosine_similarity(matrix)
    hasil = []

    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            nilai = float(sim[i, j])
            if nilai > 0:
                hasil.append({
                    kolom_awal: int(ids[i]),
                    kolom_akhir: int(ids[j]),
                    "similarity": nilai
                })

    hasil = pd.DataFrame(
        hasil,
        columns=[kolom_awal, kolom_akhir, "similarity"]
    )

    hasil = hasil.sort_values(
        "similarity", ascending=False
    ).head(5)

    hasil.to_csv(FOLDER / nama_file, index=False)

    print(f"\n{nama_file}")
    if hasil.empty:
        print("Tidak ditemukan pasangan dengan similarity positif.")
    else:
        print(hasil.to_string(index=False))

    return hasil

# Simpan lima pasangan User-User dan Item-Item teratas
simpan_top5(
    user_matrix, users,
    "visitorid_1", "visitorid_2",
    "similarity_user_top5.csv"
)

simpan_top5(
    item_matrix, items,
    "itemid_1", "itemid_2",
    "similarity_item_top5.csv"
)

print("\nSelesai. File CSV disimpan di:", FOLDER)
