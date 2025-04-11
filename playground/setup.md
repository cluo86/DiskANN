# Setup

Refer to the [workflows/SSD_index.md](https://github.com/microsoft/DiskANN/blob/main/workflows/SSD_index.md) for building index.


## Running the disk index search
cd into build dir
run 
./apps/search_disk_index  --data_type float --dist_fn l2 --index_path_prefix data/sift/disk_index_sift_learn_R32_L50_A1.2 --query_file data/sift/10q.fbin  --gt_file data/sift/sift_query_learn_gt100 -K 10 -L 100 --result_path data/sift/res --num_nodes_to_cache 10000 -W 1 -T 1


## Notes

- Currently, the prefetch results are recorded to the local data strucure `pipeline_pool` and emitted to stdout. 
- Later, we can adapt it to use a callback function, api, file, or other methods to connect LLM with DiskANN.

## GT

./compute_groundtruth --data_type float --dist_fn l2 --bas
e_file /data/rso31/DiskANN_indexes/e5_diskann_0.6/ann_vectors.bin
 --query_file squad_query_vectors.bin --gt_file squad_gt100 --K 100

./compute_groundtruth --data_type float --dist_fn l2 --base_file /data/rso31/DiskANN_indexes/e5_diskann_0.6/ann_vectors.bin --query_file wq_query_vectors.bin --gt_file wq_gt100 --K 100

## how to run wiki_search.py

uv sync
uv build && uv pip install dist/*.whl

## collect trace

python wiki_search.py > trace
./clean_trace.sh trace # get output.trace
./rm_recall.sh output.trace # get output_cleaned.trace

python process_trace.py output_cleaned.trace # get latency & recall by iter

python plot_latency_recall.py