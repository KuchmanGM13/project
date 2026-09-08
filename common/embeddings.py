from sentence_transformers import SentenceTransformer
import numpy as np


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 224
CHUNK_OVERLAP = 32

model = SentenceTransformer(MODEL_NAME)
tokenizer = model.tokenizer


def split_into_chunks(text):
    token_ids = tokenizer.encode(
        text,
        add_special_tokens=False,
        truncation=False
    )

    chunks = []

    step = CHUNK_SIZE - CHUNK_OVERLAP

    for start in range(0, len(token_ids), step):
        chunk = token_ids[start:start + CHUNK_SIZE]
        chunks.append(chunk)

        if start + CHUNK_SIZE >= len(token_ids):
            break

    return chunks
    

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


def embed_text(text):
    chunks = split_into_chunks(text)

    embeddings = embed_chunks(chunks)

    mean_embedding = np.mean(embeddings, axis=0)

    normalized_embedding = mean_embedding / np.linalg.norm(mean_embedding)

    return normalized_embedding