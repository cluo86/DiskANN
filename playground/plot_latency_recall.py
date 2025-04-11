import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the CSV files
recall_df = pd.read_csv("recall_at10_by_iteration.csv")
latency_df = pd.read_csv("latency_by_iteration.csv")

# Filter data to include only iterations from 2 to 128
filtered_recall_df = recall_df[(recall_df['Iteration'] >= 2) & (recall_df['Iteration'] <= 128)]
filtered_latency_df = latency_df[(latency_df['Iteration'] >= 2) & (latency_df['Iteration'] <= 128)]

# Convert to dictionaries for easy lookup (with filtered data)
recall_by_iter = dict(zip(filtered_recall_df['Iteration'], filtered_recall_df['Recall@10']))
latency_by_iter = dict(zip(filtered_latency_df['Iteration'], filtered_latency_df['AverageLatency_us']))

# Target recall percentages
target_recalls = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9]  # 10%, 20%, ..., 85%

results = []


# Find the first iteration that achieves each target recall
for target in target_recalls:
    for iteration, recall in sorted(recall_by_iter.items(), key=lambda x: int(x[0])):
        if float(recall) >= target:
            if int(iteration) in latency_by_iter:  # Make sure we have latency data
                results.append({
                    'Target Recall': target * 100,  # Convert to percentage
                    'Achieved Recall': float(recall) * 100,
                    'Iteration': int(iteration),
                    'Latency (μs)': latency_by_iter[int(iteration)]
                })
            break  # Move to next target

# Create a table
results_df = pd.DataFrame(results)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(results_df['Target Recall'], results_df['Latency (μs)'], marker='o', linewidth=2)
plt.xlabel('Recall (%)')
plt.ylabel('Latency (μs)')
plt.title('Latency Required for Different Recall Levels')
plt.grid(True)
plt.savefig('latency_vs_recall.png')

# Create a more detailed visualization
plt.figure(figsize=(12, 8))

# Plot the full relationship between iteration count and recall
plt.subplot(2, 1, 1)
iterations = [int(i) for i in filtered_recall_df['Iteration']]
recalls = [float(r) * 100 for r in filtered_recall_df['Recall@10']]
plt.plot(iterations, recalls, 'b-')
plt.xscale('log', base=2)  # Set x-axis to logarithmic scale with base 2

# Set explicit ticks for powers of 2 from 2 to 128
x_ticks = [2, 4, 8, 16, 32, 64, 128]
plt.xticks(x_ticks, [str(x) for x in x_ticks])

plt.xlabel('Iteration (log₂ scale)')
plt.ylabel('Recall (%)')
plt.title('Recall vs Iterations (log₂ scale)')
plt.grid(True, which="both")  # Grid lines for both major and minor ticks

# Highlight target recall points
for target in target_recalls:
    for i, recall in enumerate(recalls):
        if recall >= target * 100:
            plt.scatter(iterations[i], recall, color='red', zorder=5)
            plt.annotate(f"{target*100:.0f}%", 
                        (iterations[i], recall),
                        xytext=(5, 5),
                        textcoords='offset points')
            break

# Plot latency vs recall as a second subplot
plt.subplot(2, 1, 2)
plt.plot(results_df['Target Recall'], results_df['Latency (μs)'], 'ro-', linewidth=2)
plt.xlabel('Recall (%)')
plt.ylabel('Latency (μs)')
plt.title('Latency vs Recall')
plt.grid(True)

plt.tight_layout()
plt.savefig('recall_latency_analysis.png')

# also print the results
print(results_df)