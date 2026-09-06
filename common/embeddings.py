from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 224
CHUNK_OVERLAP = 32

model = SentenceTransformer(MODEL_NAME)
tokenizer = model.tokenizer


def split_into_chunks(text):
    pass
    

def embed_chunks(chunks):
    chunk_texts = [
        tokenizer.decode(chunk, skip_special_tokens=True)
        for chunk in chunks
    ]

    embeddings = model.encode(
        chunk_texts,
        normalize_embeddings=True
    )

    return embeddings


# TEST
text = "Python developer " * 1000

chunks = split_into_chunks(text)

embeddings = embed_chunks(chunks)

print("Number of chunks:", len(chunks))
print("Embedding shape:", embeddings.shape)