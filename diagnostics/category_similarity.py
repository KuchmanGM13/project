import pandas as pd
import numpy as np
from common.embeddings import embed_text
from sklearn.metrics.pairwise import cosine_similarity


CSV_PATH = "data/Resume.csv"

CATEGORIES = [
    "INFORMATION-TECHNOLOGY",
    "ENGINEERING",
    "FINANCE",
]


df = pd.read_csv(CSV_PATH)

df = df[df["Category"].isin(CATEGORIES)].copy()

df = df[
    df["Resume_str"].notna()
    & (df["Resume_str"].str.strip() != "")
].drop_duplicates(
    subset=["Resume_str"]
)

print("Number of resumes:", len(df))
print()
print("Resumes by category:")
print(df["Category"].value_counts())

embeddings = []

for i, text in enumerate(df["Resume_str"]):
    embedding = embed_text(text)
    embeddings.append(embedding)

    if (i + 1) % 25 == 0:
        print(f"Processed {i + 1}/{len(df)} resumes")

embeddings = np.array(embeddings)

print()
print("Embeddings shape:", embeddings.shape)


norms = np.linalg.norm(embeddings, axis=1)

print("Minimum norm:", norms.min())
print("Maximum norm:", norms.max())

rng = np.random.default_rng(42)

print()
print("Within-category similarity")
print("---------------------------")

for category in CATEGORIES:
    indices = np.where(df["Category"].values == category)[0]

    similarities = []

    for _ in range(300):
        i, j = rng.choice(indices, size=2, replace=False)

        similarity = np.dot(embeddings[i], embeddings[j])
        similarities.append(similarity)

    print(f"\n{category}")
    print(f"Pairs: {len(similarities)}")
    print(f"Mean: {np.mean(similarities):.4f}")
    print(f"Median: {np.median(similarities):.4f}")
    print(f"P10: {np.quantile(similarities, 0.10):.4f}")
    print(f"P90: {np.quantile(similarities, 0.90):.4f}")


print()
print("Between-category similarity")
print("----------------------------")

for category_a, category_b in [
    ("INFORMATION-TECHNOLOGY", "ENGINEERING"),
    ("INFORMATION-TECHNOLOGY", "FINANCE"),
    ("ENGINEERING", "FINANCE"),
]:

    indices_a = np.where(df["Category"].values == category_a)[0]
    indices_b = np.where(df["Category"].values == category_b)[0]

    similarities = []

    for _ in range(300):
        i = rng.choice(indices_a)
        j = rng.choice(indices_b)

        similarity = np.dot(embeddings[i], embeddings[j])
        similarities.append(similarity)

    print(f"\n{category_a} <-> {category_b}")
    print(f"Pairs: {len(similarities)}")
    print(f"Mean: {np.mean(similarities):.4f}")
    print(f"Median: {np.median(similarities):.4f}")
    print(f"P10: {np.quantile(similarities, 0.10):.4f}")
    print(f"P90: {np.quantile(similarities, 0.90):.4f}")