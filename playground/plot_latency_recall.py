import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

def plot_latency_recall(latency_file, recall_file, min_iter=2, max_iter=128):
    """
    Generate latency vs. recall plots and analysis from DiskANN iteration data.
    
    Args:
        latency_file: Path to latency CSV file
        recall_file: Path to recall CSV file
        min_iter: Minimum iteration to include (default: 2)
        max_iter: Maximum iteration to include (default: 128)
    """
    print(f"Analyzing data from:\n- {latency_file}\n- {recall_file}")
    
    # Extract configuration from filenames
    latency_config = os.path.basename(latency_file).replace("latency_by_iteration_", "").replace(".csv", "")
    recall_config = os.path.basename(recall_file).replace("recall_at", "").replace("_by_iteration_", "").replace(".csv", "")

    # Get L and W from latency config
    L = latency_config.split("_W")[0].replace("L", "")  # Extract L number after "L"
    W = latency_config.split("_W")[1]  # Extract W number

    # Get k value from recall config
    k = recall_config.split("_L")[0]  # Extract k value before "_L"
        
    # Set plot titles and output filenames based on config
    plot_title = f"DiskANN Search Analysis (L={L}, W={W}, k={k})"
    output_filename = f"recall{k}_latency_L{L}_W{W}"
    
    # Load the CSV files
    recall_df = pd.read_csv(recall_file)
    latency_df = pd.read_csv(latency_file)
    
    # Ensure iteration column is integer type
    recall_df['Iteration'] = recall_df['Iteration'].astype(int)
    latency_df['Iteration'] = latency_df['Iteration'].astype(int)
    
    # Filter data to include only iterations within specified range
    filtered_recall_df = recall_df[(recall_df['Iteration'] >= min_iter) & (recall_df['Iteration'] <= max_iter)]
    filtered_latency_df = latency_df[(latency_df['Iteration'] >= min_iter) & (latency_df['Iteration'] <= max_iter)]
    
    # Find the recall column name (can be Recall@k for different k values)
    recall_col = [col for col in recall_df.columns if col.startswith('Recall')][0]
    
    # Convert to dictionaries for easy lookup (with filtered data)
    recall_by_iter = dict(zip(filtered_recall_df['Iteration'], filtered_recall_df[recall_col]))
    latency_by_iter = dict(zip(filtered_latency_df['Iteration'], filtered_latency_df['AverageLatency_us']))
    
    # Determine the maximum recall for target selection (in percentage)
    max_recall = max(float(r) for r in recall_by_iter.values())
    
    # Target recall percentages - dynamically set based on data
    # All values in percentage (0-100 scale)
    target_max = min(90, max_recall)
    target_step = 10
    if target_max > 80:
        # Add more granular steps for high recall values
        target_recalls = list(range(10, 80, target_step)) + list(range(80, int(target_max) + 5, 5))
    else:
        target_recalls = list(range(10, int(target_max) + target_step, target_step))
    
    print(f"Analyzing performance at recall targets: {[f'{r:.1f}%' for r in target_recalls]}")
    
    results = []
    
    # Find the first iteration that achieves each target recall
    for target in target_recalls:
        found = False
        for iteration, recall in sorted(recall_by_iter.items(), key=lambda x: int(x[0])):
            if float(recall) >= target:
                if int(iteration) in latency_by_iter:  # Make sure we have latency data
                    results.append({
                        'Target Recall': target,  # Already percentage
                        'Achieved Recall': float(recall),  # Already percentage
                        'Iteration': int(iteration),
                        'Latency (μs)': latency_by_iter[int(iteration)]
                    })
                    found = True
                    break  # Move to next target
        
        if not found:
            print(f"Warning: Could not find iteration achieving {target:.1f}% recall")
    
    # Create a DataFrame from results
    results_df = pd.DataFrame(results)
    
    # Save results to CSV
    results_csv = f"{output_filename}_milestones.csv"
    results_df.to_csv(results_csv, index=False)
    print(f"Saved milestone data to {results_csv}")
    
    # Create a more detailed visualization with two subplots
    plt.figure(figsize=(12, 10))
    
    # Plot 1: Recall vs Iterations (log scale)
    plt.subplot(2, 1, 1)
    iterations = [int(i) for i in filtered_recall_df['Iteration']]
    recalls = [float(r) for r in filtered_recall_df[recall_col]]  # Already percentage
    plt.plot(iterations, recalls, 'b-', linewidth=2)
    plt.xscale('log', base=2)  # Set x-axis to logarithmic scale with base 2
    
    # Set explicit ticks for powers of 2 from min_iter to max_iter
    x_ticks = [2**i for i in range(int(np.log2(min_iter)), int(np.log2(max_iter))+1)]
    x_ticks = [x for x in x_ticks if min_iter <= x <= max_iter]
    plt.xticks(x_ticks, [str(x) for x in x_ticks])
    
    plt.xlabel('Iteration')
    plt.ylabel(f'Recall@{k} (%)')
    plt.title(f'Recall vs Iterations - L={L}, W={W}')
    plt.grid(True, which="both")  # Grid lines for both major and minor ticks
    
    # Highlight target recall points
    for target in target_recalls:
        for i, recall in enumerate(recalls):
            if recall >= target:
                plt.scatter(iterations[i], recall, color='red', zorder=5)
                plt.annotate(f"{target:.1f}%", 
                            (iterations[i], recall),
                            xytext=(5, 5),
                            textcoords='offset points')
                break
    
    # Plot 2: Latency vs Recall
    plt.subplot(2, 1, 2)
    plt.plot(results_df['Target Recall'], results_df['Latency (μs)'], 'ro-', linewidth=2)
    plt.xlabel(f'Recall@{k} (%)')
    plt.ylabel('Latency (μs)')
    plt.title(f'Latency vs Recall - L={L}, W={W}')
    plt.grid(True)
    
    # Add threshold labels
    for i, row in results_df.iterrows():
        plt.annotate(f"{row['Iteration']}",
                    (row['Target Recall'], row['Latency (μs)']),
                    xytext=(0, 10),
                    textcoords='offset points',
                    ha='center')
    
    plt.tight_layout()
    plt.savefig(f"{output_filename}_analysis.png", dpi=300)
    print(f"Saved analysis plot to {output_filename}_analysis.png")
    
    # Create a standalone latency vs recall plot
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['Target Recall'], results_df['Latency (μs)'], marker='o', linewidth=2)
    
    # Add iteration labels
    for i, row in results_df.iterrows():
        plt.annotate(f"{row['Iteration']}",
                    (row['Target Recall'], row['Latency (μs)']),
                    xytext=(0, 10),
                    textcoords='offset points',
                    ha='center')
    
    plt.xlabel(f'Recall@{k} (%)')
    plt.ylabel('Latency (μs)')
    plt.title(f'Latency Required for Different Recall Levels - L={L}, W={W}')
    plt.grid(True)
    plt.savefig(f"{output_filename}_curve.png", dpi=300)
    print(f"Saved latency-recall curve to {output_filename}_curve.png")
    
    # Display the results table
    print("\nPerformance milestones:")
    print(results_df.to_string(index=False))

def main():
    parser = argparse.ArgumentParser(description='Plot latency vs recall for DiskANN search results')
    parser.add_argument('latency_file', help='Path to latency by iteration CSV file')
    parser.add_argument('recall_file', help='Path to recall by iteration CSV file')
    parser.add_argument('--min-iter', type=int, default=2, help='Minimum iteration to include in analysis')
    parser.add_argument('--max-iter', type=int, default=128, help='Maximum iteration to include in analysis')
    
    args = parser.parse_args()
    plot_latency_recall(args.latency_file, args.recall_file, args.min_iter, args.max_iter)

if __name__ == "__main__":
    main()