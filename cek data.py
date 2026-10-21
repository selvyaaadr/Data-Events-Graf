
import pandas as pd

df = pd.read_csv("events.csv")

print("Ukuran dataset:", df.shape)
print("\nNama kolom:")
print(df.columns.tolist())

print("\nLima data pertama:")
print(df.head())

print("\nInformasi dataset:")
df.info()
