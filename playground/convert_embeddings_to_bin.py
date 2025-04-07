import numpy as np
import faiss
import os
import struct

# Paths
memmap_file = "indexes/emb_e5.memmap"
faiss_index_file = "indexes/e5_Flat.index"
output_file = "wiki18_vectors.bin"
corpus_file = "corpus/wiki18_100w.jsonl"

# Step 1: Load and inspect the memory-mapped embeddings
embeddings = np.memmap(memmap_file, dtype="float32", mode="r")
n_elements = embeddings.size
print(f"Total elements in {memmap_file}: {n_elements}")

# Step 2: Get dimensionality from FAISS index
index = faiss.read_index(faiss_index_file)
dim = index.d
n_vectors_faiss = index.ntotal
print(f"FAISS index: {n_vectors_faiss} vectors, {dim} dimensions")

# Step 3: Infer and reshape
n_vectors = n_elements // dim
if n_elements % dim != 0:
    raise ValueError(f"Embedding size ({n_elements}) not divisible by dim ({dim})")
embeddings = embeddings.reshape(n_vectors, dim)
print(f"Reshaped embeddings: {embeddings.shape}")

# Step 4: Validate against FAISS and corpus
if n_vectors != n_vectors_faiss:
    raise ValueError(f"Mismatch: {n_vectors} vectors in embeddings, {n_vectors_faiss} in FAISS index")

# Optional: Check corpus size
with open(corpus_file, "r") as f:
    corpus_lines = sum(1 for _ in f)
if n_vectors != corpus_lines:
    print(f"Warning: {n_vectors} vectors vs {corpus_lines} corpus lines")

# Step 5: Convert to DiskANN binary format
with open(output_file, "wb") as f:
    # Write header (2 uint32): number of vectors and dimensions
    f.write(struct.pack("I", n_vectors))
    f.write(struct.pack("I", dim))
    
    # Write vectors as float32
    embeddings.astype(np.float32).tofile(f)

print(f"Saved {output_file} ({os.path.getsize(output_file)} bytes)")
