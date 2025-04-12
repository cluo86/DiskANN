// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <functional>
#ifdef _WINDOWS
#include <numeric>
#endif
#include <string>
#include <vector>

#include "distance.h"
#include "parameters.h"

namespace diskann
{

// chengqi: add this struct to collect query stats for each iteration
struct IterationDetails;


struct QueryStats
{
    float total_us = 0; // total time to process query in micros
    float io_us = 0;    // total time spent in IO
    float cpu_us = 0;   // total time spent in CPU

    unsigned n_4k = 0;         // # of 4kB reads
    unsigned n_8k = 0;         // # of 8kB reads
    unsigned n_12k = 0;        // # of 12kB reads
    unsigned n_ios = 0;        // total # of IOs issued
    unsigned read_size = 0;    // total # of bytes read
    unsigned n_cmps_saved = 0; // # cmps saved
    unsigned n_cmps = 0;       // # cmps
    unsigned n_cache_hits = 0; // # cache_hits
    unsigned n_hops = 0;       // # search hops

    bool collect_trace = false;         // Flag to enable trace collection
    std::vector<IterationDetails> iteration_stats;  // Per-iteration details
    
    // Optional query vector storage
    bool store_query = false;
    std::vector<float> query_vector;

};

struct IterationDetails
{
    uint32_t iteration_num = 0;         // Iteration number
    float time_us = 0;                  // Cumulative time at this iteration
    float iteration_time_us = 0;        // Time for just this iteration
    uint32_t beam_width = 0;            // Actual beam width used
    uint32_t ios_for_iteration = 0;     // Number of IOs in this iteration
    uint32_t cache_hits_for_iteration = 0; // Cache hits in this iteration
    
    // Result set details
    std::vector<uint32_t> result_ids;   // Top-k results at this iteration
    
    // Pipeline pool details
    uint32_t pp_size = 0;               // Size of pipeline pool
    std::vector<uint32_t> pp_ids;       // IDs in pipeline pool
    
    // Stability metrics
    uint32_t stable_count = 0;          // Stable nodes count
    uint32_t unstable_count = 0;        // Unstable nodes count
    uint32_t first_unstable_idx = 0;    // First unstable index
    float balancer = 0.0f;              // Balancer metric
    uint32_t max_num = 0;               // Max num metric
    uint32_t prefetch_offset = 0;       // Prefetch offset
    uint32_t next_opt = 0;              // Next opt value

};

template <typename T>
inline T get_percentile_stats(QueryStats *stats, uint64_t len, float percentile,
                              const std::function<T(const QueryStats &)> &member_fn)
{
    std::vector<T> vals(len);
    for (uint64_t i = 0; i < len; i++)
    {
        vals[i] = member_fn(stats[i]);
    }

    std::sort(vals.begin(), vals.end(), [](const T &left, const T &right) { return left < right; });

    auto retval = vals[(uint64_t)(percentile * len)];
    vals.clear();
    return retval;
}

template <typename T>
inline double get_mean_stats(QueryStats *stats, uint64_t len, const std::function<T(const QueryStats &)> &member_fn)
{
    double avg = 0;
    for (uint64_t i = 0; i < len; i++)
    {
        avg += (double)member_fn(stats[i]);
    }
    return avg / len;
}
} // namespace diskann
