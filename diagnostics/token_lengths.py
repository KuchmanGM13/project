from sentence_transformers import SentenceTransformer
import pandas as pd

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

tokenizer = model.tokenizer

text = "Python developer with experience in FastAPI and PostgreSQL"

tokens = tokenizer.encode(
    text,
    add_special_tokens=True,
    truncation=False
)

print(tokens)
print("Number of tokens:", len(tokens))

df = pd.read_csv("data/Resume.csv")

categories = [
    "INFORMATION-TECHNOLOGY",
    "ENGINEERING",
    "FINANCE"
]

df = df[df["Category"].isin(categories)].copy()

df["token_length"] = df["Resume_str"].apply(
    lambda text: len(
        tokenizer.encode(
            text,
            add_special_tokens=True,
            truncation=False
        )
    )
)

print()
print("Number of resumes:", len(df))
print(df[["ID", "Category", "token_length"]].head(10))

print()
print("Resume token length statistics")
print("--------------------------------")

print("Min:", df["token_length"].min())
print("Median:", df["token_length"].median())
print("Mean:", df["token_length"].mean())
print("P75:", df["token_length"].quantile(0.75))
print("P90:", df["token_length"].quantile(0.90))
print("P95:", df["token_length"].quantile(0.95))
print("Max:", df["token_length"].max())

over_256 = (df["token_length"] > 256).mean() * 100
over_512 = (df["token_length"] > 512).mean() * 100

print()
print(f"> 256 tokens: {over_256:.2f}%")
print(f"> 512 tokens: {over_512:.2f}%")

print()
print("Statistics by category")
print("-----------------------")

print(
    df.groupby("Category")["token_length"].agg(
        ["count", "min", "median", "mean", "max"]
    )
)