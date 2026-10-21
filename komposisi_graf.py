from neo4j import GraphDatabase

URI = "bolt://localhost:7687"
AUTH = ("neo4j", "selvy2004")

driver = GraphDatabase.driver(URI, auth=AUTH)

query = """
MATCH (n)
RETURN labels(n) AS jenis_simpul,
       count(n) AS jumlah_simpul
ORDER BY jenis_simpul
"""

with driver.session() as session:
    hasil = session.run(query)

    for record in hasil:
        print(
            record["jenis_simpul"],
            record["jumlah_simpul"]
        )

driver.close()