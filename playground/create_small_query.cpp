#include <iostream>
#include <fstream>
#include <vector>
#include <stdexcept>
#include <string>

void create_small_query_bin(const std::string& input_file, const std::string& output_file, uint32_t num_queries_to_extract) {
    // Open input file
    std::ifstream in_file(input_file, std::ios::binary);
    if (!in_file) {
        throw std::runtime_error("Failed to open input file: " + input_file);
    }

    // Read header
    uint32_t num_queries, dimensions;
    in_file.read(reinterpret_cast<char*>(&num_queries), sizeof(uint32_t));
    in_file.read(reinterpret_cast<char*>(&dimensions), sizeof(uint32_t));
    if (!in_file) {
        throw std::runtime_error("Failed to read header from " + input_file);
    }

    std::cout << "Original: " << num_queries << " queries, " << dimensions << " dimensions\n";

    // Validate extraction size
    if (num_queries_to_extract > num_queries) {
        num_queries_to_extract = num_queries; // Cap to available queries
    }
    if (num_queries_to_extract < 1) {
        throw std::runtime_error("Number of queries to extract must be positive");
    }

    // Read the first num_queries_to_extract vectors
    std::vector<float> queries(num_queries_to_extract * dimensions);
    in_file.read(reinterpret_cast<char*>(queries.data()), num_queries_to_extract * dimensions * sizeof(float));
    if (!in_file && in_file.gcount() != (long int)(num_queries_to_extract * dimensions * sizeof(float))) {
        throw std::runtime_error("Failed to read " + std::to_string(num_queries_to_extract) + " queries");
    }
    in_file.close();

    // Open output file
    std::ofstream out_file(output_file, std::ios::binary);
    if (!out_file) {
        throw std::runtime_error("Failed to open output file: " + output_file);
    }

    // Write new header
    out_file.write(reinterpret_cast<const char*>(&num_queries_to_extract), sizeof(uint32_t));
    out_file.write(reinterpret_cast<const char*>(&dimensions), sizeof(uint32_t));

    // Write query data
    out_file.write(reinterpret_cast<const char*>(queries.data()), num_queries_to_extract * dimensions * sizeof(float));
    if (!out_file) {
        throw std::runtime_error("Failed to write to " + output_file);
    }
    out_file.close();

    std::cout << "Created " << output_file << " with " << num_queries_to_extract << " queries\n";
}

int main(int argc, char* argv[]) {
    try {
        // Check if we have the correct number of arguments
        if (argc != 4) {
            std::cerr << "Usage: " << argv[0] << " <input_file> <output_file> <num_queries_to_extract>" << std::endl;
            return 1;
        }

        std::string input_file = argv[1];
        std::string output_file = argv[2];
        uint32_t num_queries_to_extract;
        
        try {
            num_queries_to_extract = static_cast<uint32_t>(std::stoi(argv[3]));
        } catch (const std::exception& e) {
            std::cerr << "Error: Invalid number of queries specified" << std::endl;
            return 1;
        }

        create_small_query_bin(input_file, output_file, num_queries_to_extract);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}