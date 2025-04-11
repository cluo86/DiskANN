import mmap
import os
from collections import defaultdict
import multiprocessing as mp
import numpy as np
import tqdm
import pickle

class Iteration:
    def __init__(self, iteration_num, time_us):
        self.iteration_num = iteration_num
        self.time_us = time_us
        self.result_ids = []  # IDs in the result set
        
        # Pipeline pool data
        self.pp_size = 0
        self.pp_ids = []
        
        # Stability metrics
        self.opt_num = 0
        self.stable_count = 0
        self.unstable_count = 0
        self.first_unstable_idx = 0
        self.balancer = 0
        self.max_num = 0
        self.prefetch_offset = 0
        self.next_opt = 0
        self.pp_size_final = 0
        
    def __str__(self):
        return f"Iteration {self.iteration_num}: time {self.time_us} us, PP size {self.pp_size}, stable count {self.stable_count}, unstable count {self.unstable_count}, balancer {self.balancer}, max_num {self.max_num}"

    def get_result_ids(self):
        return self.result_ids
    
class Query:
    def __init__(self, query_id):
        self.query_id = query_id
        self.iterations = []
        
    def add_iteration(self, iteration):
        self.iterations.append(iteration)
        
    def total_time(self):
        if not self.iterations:
            return 0
        return self.iterations[-1].time_us
    
    def __str__(self):
        return f"Query {self.query_id}: {len(self.iterations)} iterations, total time {self.total_time()} us"
    
    def iteration_count(self):
        return len(self.iterations)
    
    def get_iterations(self):
        return self.iterations
    
    
def create_query_index(filepath):
    """Create an index of query start positions in the file"""
    query_positions = []
    
    # Get file size for progress reporting
    file_size = os.path.getsize(filepath)
    
    with open(filepath, 'rb') as f:
        # Memory map the file for faster access
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        
        # Use byte search for faster matching
        iteration_prefix = b"Iteration 1 "
        
        pos = mm.find(iteration_prefix)
        with tqdm.tqdm(total=file_size, desc="Creating query index") as pbar:
            while pos != -1:
                query_positions.append(pos)
                # Update progress bar based on file position
                pbar.update(pos - pbar.n)
                
                # Find next occurrence after current position
                mm.seek(pos + 1)
                next_pos = mm.find(iteration_prefix)
                if next_pos == -1:
                    break
                pos = next_pos
                
        mm.close()
    
    print(f"Found {len(query_positions)} queries in the trace file")
    return query_positions

def process_query_chunk(args):
    """Process a chunk of queries from the file"""
    filepath, start_positions, chunk_id, num_chunks = args
    
    # Calculate range for this chunk
    chunk_size = len(start_positions) // num_chunks
    start_idx = chunk_id * chunk_size
    end_idx = start_idx + chunk_size if chunk_id < num_chunks - 1 else len(start_positions)
    
    # Process only positions in this chunk
    positions = start_positions[start_idx:end_idx]
    
    # Statistics to collect
    stats = {
        "iterations": [],
        "query_times": [],
        "pipeline_sizes": [],
        "stability": [],
        "unstability": [],
        "max_nums": [],
        "balancers": []
    }
    
    with open(filepath, 'r') as f:
        for i, pos in enumerate(positions):
            # Seek to the start position of this query
            f.seek(pos)
            
            # Process the query lines until next query or EOF
            iteration_count = 0
            last_time = 0
            
            while True:
                line = f.readline()
                if not line or (i < len(positions) - 1 and f.tell() >= positions[i+1]):
                    break
                    
                if line.startswith("Iteration"):
                    iteration_count += 1
                    
                    # Extract time from the line
                    if "(time=" in line:
                        time_str = line.split("(time=")[1].split(" us)")[0]
                        try:
                            last_time = int(time_str)
                        except ValueError:
                            continue
                    
                    # Extract PP size
                    parts = line.split("|")
                    if len(parts) > 1 and "Pipeline Pool" in parts[1]:
                        try:
                            pp_size = int(parts[1].split("(size=")[1].split(")")[0])
                            stats["pipeline_sizes"].append(pp_size)
                        except (ValueError, IndexError):
                            pass
                    
                    # Extract stability metrics
                    if len(parts) > 2:
                        metrics = parts[2]
                        try:
                            if "stable=" in metrics:
                                stable = int(metrics.split("stable=")[1].split(",")[0])
                                stats["stability"].append(stable)
                            
                            if "unstable=" in metrics:
                                unstable = int(metrics.split("unstable=")[1].split(",")[0])
                                stats["unstability"].append(unstable)
                            
                            if "max_num=" in metrics:
                                max_num = int(metrics.split("max_num=")[1].split(",")[0])
                                stats["max_nums"].append(max_num)
                            
                            if "balancer=" in metrics:
                                balancer = float(metrics.split("balancer=")[1].split(",")[0])
                                stats["balancers"].append(balancer)
                        except (ValueError, IndexError):
                            pass
            
            stats["iterations"].append(iteration_count)
            stats["query_times"].append(last_time)
    
    return stats


def extract_and_save_query(filepath, query_index, query_positions, output_dir):
    """Extract a single query and save to file for later analysis"""
    if query_index >= len(query_positions):
        print(f"Query index {query_index} out of range (max: {len(query_positions)-1})")
        return False
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    start_pos = query_positions[query_index]
    end_pos = query_positions[query_index + 1] if query_index < len(query_positions) - 1 else None
    
    query = Query(query_index)
    
    with open(filepath, 'r') as f:
        f.seek(start_pos)
        
        while True:
            pos = f.tell()
            if end_pos and pos >= end_pos:
                break
                
            line = f.readline()
            if not line:
                break
                
            if not line.startswith("Iteration"):
                continue
            
            # Parse iteration
            parts = line.split('|')
            
            # Basic iteration info
            part1 = parts[0].strip()
            iter_info = part1.split("(time=")
            iter_num = int(iter_info[0].replace("Iteration ", "").strip())
            time_part = iter_info[1].split(" us)")[0]
            time_us = int(time_part)
            
            iteration = Iteration(iter_num, time_us)
            
            # Result IDs - critical for recall calculation
            id_part = iter_info[1].split("us) ")[1].strip()
            iteration.result_ids = [int(id) for id in id_part.split()]
            
            # Extract pipeline pool data
            if len(parts) > 1 and "Pipeline Pool" in parts[1]:
                pp_part = parts[1].strip()
                iteration.pp_size = int(pp_part.split("(size=")[1].split(")")[0])
                if ":" in pp_part and iteration.pp_size > 0:
                    ids_part = pp_part.split(":")[1].strip()
                    if ids_part:
                        iteration.pp_ids = [int(id) for id in ids_part.split(", ")]
            
            # Extract stability metrics
            if len(parts) > 2:
                metrics = parts[2].strip()
                if "Opt=" in metrics:
                    iteration.opt_num = int(metrics.split("Opt=")[1].split()[0])
                if "stable=" in metrics:
                    iteration.stable_count = int(metrics.split("stable=")[1].split(",")[0])
                if "unstable=" in metrics:
                    iteration.unstable_count = int(metrics.split("unstable=")[1].split(",")[0])
                if "first_unstable_idx=" in metrics:
                    iteration.first_unstable_idx = int(metrics.split("first_unstable_idx=")[1].split(",")[0])
                if "balancer=" in metrics:
                    iteration.balancer = float(metrics.split("balancer=")[1].split(",")[0])
                if "max_num=" in metrics:
                    iteration.max_num = int(metrics.split("max_num=")[1].split(",")[0])
                if "prefetch_offset=" in metrics:
                    iteration.prefetch_offset = int(metrics.split("prefetch_offset=")[1].split(",")[0])
                if "next_opt=" in metrics:
                    iteration.next_opt = int(metrics.split("next_opt=")[1].split(",")[0])
                if "PPSIZE=" in metrics:
                    iteration.pp_size_final = int(metrics.split("PPSIZE=")[1].split(",")[0])
            
            query.add_iteration(iteration)
    
    # Save query to file
    output_file = f"{output_dir}/query_{query_index}.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump(query, f)
    
    print(f"Saved query {query_index} with {len(query.iterations)} iterations to {output_file}")
    return True


def extract_all_queries(trace_file, output_dir, num_processes=8):
    """Extract and save all queries from the trace file"""
    # First create the index
    query_positions = create_query_index(trace_file)
    
    # Save the index itself for future use
    index_file = f"{output_dir}/query_index.npy"
    os.makedirs(output_dir, exist_ok=True)
    np.save(index_file, np.array(query_positions))
    print(f"Saved query index to {index_file}")
    
    # Process queries in batches to avoid memory issues
    batch_size = 1000  # Adjust based on memory constraints
    total_queries = len(query_positions)
    
    for batch_start in range(0, total_queries, batch_size):
        batch_end = min(batch_start + batch_size, total_queries)
        print(f"Processing queries {batch_start} to {batch_end-1} (out of {total_queries})...")
        
        # Create process pool for this batch
        with mp.Pool(processes=num_processes) as pool:
            args = [(trace_file, i, query_positions, output_dir) for i in range(batch_start, batch_end)]
            list(tqdm.tqdm(pool.starmap(extract_and_save_query, args), 
                          total=len(args), 
                          desc=f"Batch {batch_start//batch_size + 1}"))
    
    print(f"All {total_queries} queries extracted and saved to {output_dir}")


def load_query(query_index, queries_dir):
    """Load a query from saved file"""
    file_path = f"{queries_dir}/query_{query_index}.pkl"
    
    if not os.path.exists(file_path):
        print(f"Query {query_index} data not found at {file_path}")
        return None
    
    with open(file_path, 'rb') as f:
        query = pickle.load(f)
    
    return query


import matplotlib.pyplot as plt
from utils import calculate_recall_from_gt_file

def calculate_recall_by_iteration(queries_dir, gt_file, max_iterations=128, k=10):
    """
    Calculate recall@k for each iteration across all queries.
    
    Args:
        queries_dir: Directory containing saved query pickle files
        gt_file: Ground truth file path for recall calculation
        max_iterations: Maximum iteration to calculate (default 128)
        k: k value for recall@k calculation
    
    Returns:
        Dictionary mapping iteration numbers to recall values
    """
    # Get all query files
    query_fns = [f for f in os.listdir(queries_dir) if f.startswith("query_") and f.endswith(".pkl")]    
    total_queries = len(query_fns)
    query_files = [f"query_{i}.pkl" for i in range(0, total_queries)]
    
    print(f"Found {total_queries} saved queries in {queries_dir}")
    
    # Load all queries (this can be optimized if memory is an issue)
    print("Loading queries...")
    queries = []
    for query_file in tqdm.tqdm(query_files):
        with open(os.path.join(queries_dir, query_file), 'rb') as f:
            query = pickle.load(f)
            queries.append(query)
    
    # Calculate recall for each iteration
    recalls = {}
    
    print(f"Calculating recall for iterations 1-{max_iterations}...")
    for iteration in tqdm.tqdm(range(1, max_iterations + 1)):
        # Collect result IDs for this iteration across all queries
        iteration_results = []
        
        for query in queries:
            # Find the specific iteration for this query
            matching_iterations = [i for i in query.iterations if i.iteration_num == iteration]
            
            # If this query has the current iteration, add its result IDs
            if matching_iterations:
                iteration_results.append(matching_iterations[0].result_ids)  # Take top-k results
        
        # Skip if no queries have this iteration
        if not iteration_results:
            print(f"No queries have iteration {iteration}, stopping.")
            break
        
        # Convert to numpy array
        # print(iteration_results)
        results_array = np.array(iteration_results)
        # print(results_array)
        # Calculate recall
        recall = calculate_recall_from_gt_file(K=k, ids=results_array, gt_file=gt_file)
        recalls[iteration] = recall
        print(f"Recall@{k} for iteration {iteration}: {recall}")
        
    return recalls

def save_recall_results(recalls, output_file):
    """Save recall results to a file and generate plot"""
    # Save to text file
    with open(output_file, 'w') as f:
        f.write("Iteration,Recall@10\n")
        for iteration, recall in recalls.items():
            f.write(f"{iteration},{recall:.6f}\n")
    
    # Generate plot
    plt.figure(figsize=(10, 6))
    iterations = list(recalls.keys())
    recall_values = list(recalls.values())
    
    plt.plot(iterations, recall_values, marker='o')
    plt.xlabel('Iteration')
    plt.ylabel('Recall@10')
    plt.title('Recall@10 by Iteration')
    plt.grid(True)
    
    # Save plot
    plot_file = output_file.replace('.csv', '.png')
    plt.savefig(plot_file)
    plt.close()
    
    print(f"Results saved to {output_file}")
    print(f"Plot saved to {plot_file}")


def calculate_latency_by_iteration(queries_dir, max_iterations=128):
    """
    Calculate average latency for each iteration across all queries.
    
    Args:
        queries_dir: Directory containing saved query pickle files
        max_iterations: Maximum iteration to calculate (default 128)
    
    Returns:
        Dictionary mapping iteration numbers to average latency values in microseconds
    """
    # Get all query files
    query_files = [f"query_{i}.pkl" for i in range(1, 87600)]
    total_queries = len(query_files)
    
    print(f"Found {total_queries} saved queries in {queries_dir}")
    
    # Initialize data structures to track latencies
    iteration_times = {i: [] for i in range(1, max_iterations + 1)}
    
    # Load and process queries one by one
    print("Processing queries to calculate latencies...")
    for query_file in tqdm.tqdm(query_files):
        try:
            with open(os.path.join(queries_dir, query_file), 'rb') as f:
                query = pickle.load(f)
                
                # Sort iterations by number to ensure correct order
                sorted_iterations = sorted(query.iterations, key=lambda x: x.iteration_num)
                
                for iter_obj in sorted_iterations:
                    if 1 <= iter_obj.iteration_num <= max_iterations:
                        iteration_times[iter_obj.iteration_num].append(iter_obj.time_us)
        except (FileNotFoundError, EOFError, pickle.PickleError) as e:
            # Skip problematic files
            continue
    
    # Calculate average latency for each iteration
    avg_latencies = {}
    for iteration, latencies in iteration_times.items():
        if latencies:  # Only calculate if we have data
            avg_latencies[iteration] = sum(latencies) / len(latencies)
    
    return avg_latencies

def save_latency_results(latencies, output_file):
    """Save latency results to a file and generate plot"""
    # Save to text file
    with open(output_file, 'w') as f:
        f.write("Iteration,AverageLatency_us\n")
        for iteration, latency in sorted(latencies.items()):
            f.write(f"{iteration},{latency:.2f}\n")
    
    # Generate plot
    plt.figure(figsize=(10, 6))
    iterations = sorted(latencies.keys())
    latency_values = [latencies[i] for i in iterations]
    
    plt.plot(iterations, latency_values, marker='o', color='red')
    plt.xlabel('Iteration')
    plt.ylabel('Average Latency (μs)')
    plt.title('Average Latency by Iteration')
    plt.grid(True)
    
    # Save plot
    plot_file = output_file.replace('.csv', '.png')
    plt.savefig(plot_file)
    plt.close()
    
    print(f"Latency results saved to {output_file}")
    print(f"Latency plot saved to {plot_file}")


import argparse

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Process DiskANN trace files for analysis')
    
    # Required arguments
    parser.add_argument('trace_file', type=str, help='Path to the trace file (required)')
    
    # Optional arguments with defaults
    parser.add_argument('--output_dir', type=str, default='saved_queries',
                        help='Directory to save processed queries (default: saved_queries)')
    parser.add_argument('--gt_file', type=str, default='gt_files/squad_gt10',
                        help='Ground truth file for recall calculation (default: gt_files/squad_gt10)')
    parser.add_argument('--max_iterations', type=int, default=128,
                        help='Maximum number of iterations to analyze (default: 128)')
    parser.add_argument('--num_processes', type=int, default=8,
                        help='Number of processes to use for parallel processing (default: 8)')
    parser.add_argument('--k', type=int, default=10,
                        help='k value for recall@k calculation (default: 10)')
    
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    
    # Check if queries are already extracted
    if not os.path.exists(args.output_dir) or len([f for f in os.listdir(args.output_dir) if f.startswith("query_")]) == 0:
        # First extract all queries
        extract_all_queries(args.trace_file, args.output_dir, num_processes=args.num_processes)
    
    # Calculate recall by iteration
    recalls = calculate_recall_by_iteration(args.output_dir, args.gt_file, 
                                          max_iterations=args.max_iterations, k=args.k)
    
    # Save results
    recall_output = f"recall_at{args.k}_by_iteration.csv"
    save_recall_results(recalls, recall_output)
    
    print("\nCalculating average latency per iteration...")
    latencies = calculate_latency_by_iteration(args.output_dir, max_iterations=args.max_iterations)
    
    # Save latency results
    latency_output = "latency_by_iteration.csv"
    save_latency_results(latencies, latency_output)
    
    # Print some sample latencies
    print("\nAverage latencies for first 10 iterations (microseconds):")
    for i in range(1, min(11, max(latencies.keys()) + 1)):
        if i in latencies:
            print(f"Iteration {i}: {latencies[i]:.2f} μs")
            
            