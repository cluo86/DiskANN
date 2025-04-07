## Convert embeddings 

python convert_embeddings_to_bin.py

## Build index
./apps/build_disk_index --data_type float --data_path data/wiki18_vectors.bin --index_path_prefix data/wiki18_diskann_index --dist_fn l2 -M 64 -B 0.003

## Create query

python create_small_query