#!/bin/bash

# Check if an input file is provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 <input_file> [output_file]"
    exit 1
fi

# Input file (first argument)
input_file="$1"

# Output file (second argument or default to input file with _output suffix)
if [ $# -gt 1 ]; then
    output_file="$2"
else
    output_file="${input_file%.trace}_output.trace"
fi

# Process the input file with awk and save the output to the specified output file
awk '/Loading the cache list into memory....done./ {found=1} found' "$input_file" > "$output_file"

# Print a message indicating where the output is saved
echo "Output saved to $output_file"
