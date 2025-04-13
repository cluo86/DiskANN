from datasets import load_dataset
import numpy as np
import struct
import torch
from transformers import AutoTokenizer, AutoModel
import argparse
from tqdm import tqdm

def save_vectors_to_bin(vectors, output_file):
    """
    Save vectors to DiskANN-compatible binary format.
    
    Args:
        vectors: numpy array of shape (num_vectors, num_dimensions)
        output_file: path to save the binary file
    """
    # Ensure vectors are float32 (required by DiskANN)
    vectors = vectors.astype(np.float32)
    
    num_vectors = vectors.shape[0]
    num_dimensions = vectors.shape[1]
    
    with open(output_file, 'wb') as f:
        # Write metadata as int32
        f.write(struct.pack('i', num_vectors))
        f.write(struct.pack('i', num_dimensions))
        
        # Write vector data
        f.write(vectors.tobytes())
    
    print(f"Saved {num_vectors} vectors with {num_dimensions} dimensions to {output_file}")

def pooling(pooler_output, last_hidden_state, attention_mask=None, pooling_method="mean"):
    if pooling_method == "mean":
        last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0) # type: ignore
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None] # type: ignore
    elif pooling_method == "cls":
        return last_hidden_state[:, 0]
    elif pooling_method == "pooler":
        return pooler_output
    else:
        raise NotImplementedError("Pooling method not implemented!")

def load_model(model_path: str, use_fp16: bool = False):
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    model.eval()
    model.cuda()
    if use_fp16:
        model = model.half()
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)

    return model, tokenizer

import datasets
from datasets import Dataset
def load_corpus(corpus_path: str) -> Dataset:
    corpus = datasets.load_dataset("json", data_files=corpus_path, split="train")
    return corpus # type: ignore

class Encoder:
    """
    Encoder class for encoding queries using a specified model.

    Attributes:
        model_path (str): The path to the model.
        pooling_method (str): The method used for pooling.
        max_length (int): The maximum length of the input sequences.
        use_fp16 (bool): Whether to use FP16 precision.

    Methods:
        encode(query_list: List[str], is_query=True) -> np.ndarray:
            Encodes a list of queries into embeddings.
    """

    def __init__(self, model_path, pooling_method, max_length, use_fp16):
        self.model_path = model_path
        self.pooling_method = pooling_method
        self.max_length = max_length
        self.use_fp16 = use_fp16

        self.model, self.tokenizer = load_model(model_path=model_path, use_fp16=use_fp16)

    @torch.inference_mode()
    def encode(self, query: str) -> np.ndarray:
        query = f"passage: {query}"

        inputs = self.tokenizer(
            query, max_length=self.max_length, padding=True, truncation=True, return_tensors="pt"
        )
        inputs = {k: v.cuda() for k, v in inputs.items()}

        if "T5" in type(self.model).__name__:
            # T5-based retrieval model
            decoder_input_ids = torch.zeros((inputs["input_ids"].shape[0], 1), dtype=torch.long).to(
                inputs["input_ids"].device
            )
            output = self.model(**inputs, decoder_input_ids=decoder_input_ids, return_dict=True)
            query_emb = output.last_hidden_state[:, 0, :]

        else:
            output = self.model(**inputs, return_dict=True)
            query_emb = pooling(
                output.pooler_output, output.last_hidden_state, inputs["attention_mask"], self.pooling_method
            )
        query_emb = torch.nn.functional.normalize(query_emb, dim=-1)
        query_emb = query_emb.detach().cpu().numpy()
        query_emb = query_emb.astype(np.float32, order="C")
        return query_emb
    
def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description='Generate query vectors from SQuAD dataset')
    parser.add_argument('--num_queries', type=int, default=100, 
                        help='Number of queries to generate embeddings for. Default: first 100 queries in dataset')
    parser.add_argument('--dataset', type=str, default="rajpurkar/squad", choices=["rajpurkar/squad", "Stanford/web_questions"],
                        help='Dataset to use. Default: rajpurkar/squad')
    parser.add_argument('--output', type=str, default=None,
                        help='Output filename. Default: {dataset_name}_query_vectors_{num_queries}.bin')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='Batch size for processing. Default: 10')
    args = parser.parse_args()
    
    # Create the encoder
    encoder = Encoder(
        model_path="intfloat/e5-base-v2",
        pooling_method="mean",
        max_length=180,
        use_fp16=True,
    )
    
    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    ds = load_dataset(args.dataset)
    
    # Get questions
    questions = ds["train"]["question"]
    total_questions = len(questions)
    
    # Determine number of queries to process
    num_queries = args.num_queries if args.num_queries is not None else total_questions
    num_queries = min(num_queries, total_questions)  # Ensure we don't exceed available questions
    
    print(f"Processing {num_queries} queries out of {total_questions} total questions")
    queries = questions[:num_queries]
    
    # Generate embeddings with tqdm progress bar
    print("Generating embeddings...")
    
    # Process in batches
    batch_size = args.batch_size
    embeddings_list = []
    
    # Use tqdm for progress bar
    for i in tqdm(range(0, len(queries), batch_size), desc="Encoding queries", unit="batch"):
        batch = queries[i:i+batch_size]
        batch_embeddings = [encoder.encode(query) for query in batch]
        embeddings_list.extend(batch_embeddings)
    
    embeddings = np.vstack(embeddings_list)
    
    # Determine output filename
    dataset_name = args.dataset.split("/")[-1]
    output_file = args.output if args.output else f"{dataset_name}_query_vectors_{num_queries}.bin"
    
    # Save to binary format
    save_vectors_to_bin(embeddings, output_file)
    
    print(f"Query vectors saved to '{output_file}'")
    print(f"You can now use this file with compute_groundtruth")

if __name__ == "__main__":
    main()