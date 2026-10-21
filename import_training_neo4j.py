
import pandas as pd
from pathlib import Path
from neo4j import GraphDatabase

# ==================================================
# 1. PENGATURAN
# ==================================================

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "training.csv"

URI = "bolt://localhost:7687"
USERNAME = "neo4j"

# GANTI dengan password database Neo4j milikmu.
# Jangan membagikan password ini kepada siapa pun.
PASSWORD = "ISI DENGAN PASWORD"

BATCH_SIZE = 100


# ==================================================
# 2. MEMBACA DATA TRAINING
# ==================================================

def baca_data():
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"File training.csv tidak ditemukan di:\n{CSV_PATH}\n\n"
            "Pastikan training.csv berada satu folder "
            "dengan import_training_neo4j.py."
        )

    df = pd.read_csv(CSV_PATH)

    kolom_wajib = [
        "timestamp",
        "visitorid",
        "event",
        "itemid",
        "transactionid"
    ]

    kolom_hilang = [
        kolom for kolom in kolom_wajib
        if kolom not in df.columns
    ]

    if kolom_hilang:
        raise ValueError(
            f"Kolom berikut tidak ditemukan di training.csv: "
            f"{kolom_hilang}\n"
            f"Kolom yang tersedia: {df.columns.tolist()}"
        )

    df = df[kolom_wajib].copy()

    # Pastikan kolom numerik terbaca dengan benar.
    for kolom in ["timestamp", "visitorid", "itemid"]:
        df[kolom] = pd.to_numeric(
            df[kolom], errors="raise"
        ).astype("int64")

    # Rapikan nama event.
    df["event"] = (
        df["event"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    # Event yang didukung oleh script.
    event_valid = ["view", "addtocart", "transaction"]

    # Hentikan proses jika ada event yang tidak dikenal.
    event_tidak_valid = sorted(
        set(df["event"].dropna().unique()) - set(event_valid)
    )

    if event_tidak_valid:
        raise ValueError(
            f"Ditemukan jenis event yang tidak dikenal: "
            f"{event_tidak_valid}"
        )

    if df["event"].isna().any():
        raise ValueError("Terdapat nilai event yang kosong.")

    # transactionid boleh kosong untuk event selain transaksi.
    # Jika ada nilainya, ubah menjadi integer.
    df["transactionid"] = pd.to_numeric(
        df["transactionid"], errors="raise"
    ).astype("Int64")

    print("=" * 55)
    print("DATA TRAINING BERHASIL DIBACA")
    print("=" * 55)
    print(f"Lokasi file : {CSV_PATH}")
    print(f"Jumlah baris: {len(df):,}")
    print("\nDistribusi event:")
    print(df["event"].value_counts().to_string())

    return df


# ==================================================
# 3. MEMBUAT STRUKTUR GRAF DI NEO4J
# ==================================================

QUERY_VIEW = """
UNWIND $batch AS row
MERGE (v:Visitor {visitorId: row.visitorid})
MERGE (i:Item {itemId: row.itemid})
MERGE (v)-[:VIEWED {timestamp: row.timestamp}]->(i)
"""

QUERY_CART = """
UNWIND $batch AS row
MERGE (v:Visitor {visitorId: row.visitorid})
MERGE (i:Item {itemId: row.itemid})
MERGE (v)-[:ADDED_TO_CART {timestamp: row.timestamp}]->(i)
"""

QUERY_TRANSACTION = """
UNWIND $batch AS row
MERGE (v:Visitor {visitorId: row.visitorid})
MERGE (i:Item {itemId: row.itemid})
MERGE (v)-[:BOUGHT {
    timestamp: row.timestamp,
    transactionId: row.transactionid
}]->(i)
"""


# ==================================================
# 4. IMPOR DATA KE NEO4J
# ==================================================

def impor_event(session, df, nama_event, query):
    data_event = df[df["event"] == nama_event]
    total = len(data_event)

    print(f"\nMengimpor event: {nama_event}")
    print(f"Jumlah baris: {total:,}")

    if total == 0:
        print("Tidak ada data untuk event ini.")
        return

    for mulai in range(0, total, BATCH_SIZE):
        bagian = data_event.iloc[
            mulai:mulai + BATCH_SIZE
        ]

        records = bagian.to_dict(orient="records")

        # Konversi tipe data agar sesuai dengan Neo4j.
        for record in records:
            record["timestamp"] = int(record["timestamp"])
            record["visitorid"] = int(record["visitorid"])
            record["itemid"] = int(record["itemid"])

            nilai_transaksi = record["transactionid"]

            if pd.isna(nilai_transaksi):
                record["transactionid"] = None
            else:
                record["transactionid"] = int(
                    nilai_transaksi
                )

        session.run(query, batch=records).consume()

        selesai = min(mulai + BATCH_SIZE, total)
        print(f"  Progres: {selesai:,}/{total:,}")

    print(f"Impor event {nama_event} selesai.")


# ==================================================
# 5. MENJALANKAN PROGRAM
# ==================================================

def main():
    if PASSWORD == "GANTI_DENGAN_PASSWORD_NEO4J":
        raise ValueError(
            "Password belum diatur.\n"
            "Buka bagian PENGATURAN dan ganti nilai PASSWORD "
            "dengan password database Neo4j milikmu."
        )

    # Baca dan validasi CSV sebelum terhubung ke database.
    df = baca_data()

    print("\nMenghubungkan ke Neo4j...")

    driver = GraphDatabase.driver(
        URI,
        auth=(USERNAME, PASSWORD)
    )

    try:
        driver.verify_connectivity()
        print("Koneksi Neo4j berhasil.")

        with driver.session() as session:
            impor_event(
                session, df, "view", QUERY_VIEW
            )

            impor_event(
                session, df, "addtocart", QUERY_CART
            )

            impor_event(
                session, df, "transaction", QUERY_TRANSACTION
            )

        print("\n" + "=" * 55)
        print("IMPOR TRAINING SELESAI")
        print("=" * 55)
        print("Selanjutnya, periksa data melalui Neo4j Browser.")

    finally:
        driver.close()


if __name__ == "__main__":
    main()
