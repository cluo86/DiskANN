# Setup

Refer to the [workflows/SSD_index.md](https://github.com/microsoft/DiskANN/blob/main/workflows/SSD_index.md) for building index.

## Notes

- Currently, the prefetch results are recorded to the local data strucure `pipeline_pool` and emitted to stdout. 
- Later, we can adapt it to use a callback function, api, file, or other methods to connect LLM with DiskANN.

## setup python env

uv sync
uv build && uv pip install dist/*.whl

## get query bin

see squad.py to generate query bin, for example
python squad.py --num_queries 100 --output squad_query_vectors.bin

## compute GT

./compute_groundtruth --data_type float --dist_fn l2 --bas
e_file /data/rso31/DiskANN_indexes/e5_diskann_0.6/ann_vectors.bin
 --query_file squad_query_vectors.bin --gt_file squad_gt100 --K 100

./compute_groundtruth --data_type float --dist_fn l2 --base_file /data/rso31/DiskANN_indexes/e5_diskann_0.6/ann_vectors.bin --query_file wq_query_vectors.bin --gt_file wq_gt100 --K 100

## collect trace

- set DISKANN_WRITE_TRACE=1 if we want to save traces to file

../build/apps/search_disk_index --data_type float --dist_fn l2 --index_path_prefix /data/FlashRAG_data/indexes/e5_diskann_0.6/ann --query_file squad_query_vectors_400.bin --gt_file gt_files/squad_gt10 -K 10 -L 1024 --result_path res/ --num_nodes_to_cache 210153 -W 8

this will generate two csv files
latency_by_iteration_L{l_value}_W{beam_width}.csv
recall{k}L{l_value}_W{beam_width}.csv

trace file will be saved to the same dir if specified

## plot

see plot_latency_recall.py

example: 
python plot_latency_recall.py latency.csv recall.csv