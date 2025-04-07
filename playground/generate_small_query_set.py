import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import struct

# ==== Config ====
e5_model_name = "intfloat/e5-base"
output_query_file = "query_file.fbin"

# Example queries
queries = [
    "What is retrieval-augmented generation?",
    "Explain how neural networks learn.",
    "Best way to encode documents for semantic search.",
]

# ==== Load E5 model ====
tokenizer = AutoTokenizer.from_pretrained(e5_model_name)
model = AutoModel.from_pretrained(e5_model_name)
model.eval()

# ==== Encode queries ====
def encode(texts):
    batch_dict = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors='pt'
    )
    with torch.no_grad():
        outputs = model(**batch_dict)
        embeddings = outputs.last_hidden_state[:, 0]  # CLS token
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)  # cosine normalization
    return embeddings.cpu().numpy()

query_embeddings = encode(["query: " + q for q in queries])
num, dim = query_embeddings.shape

# ==== Save as DiskANN fbin format ====
with open(output_query_file, "wb") as f:
    f.write(struct.pack("I", num))  # number of queries
    f.write(struct.pack("I", dim))  # embedding dimension
    query_embeddings.astype(np.float32).tofile(f)

print(f"Saved {num} queries to {output_query_file} ({dim}-dim float32 vectors)")
