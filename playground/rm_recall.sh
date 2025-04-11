#!/bin/bash

# Check arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <input_file> [output_file]"
    exit 1
fi

input_file="$1"

# Set output file, or use default
if [ $# -gt 1 ]; then
    output_file="$2"
else
    output_file="${input_file%.trace}_cleaned.trace"
fi

# Split file into:
#   1. Head: everything before the last 'Iteration' line (inclusive)
#   2. Tail: everything after that line
# Then clean the tail by removing lines that start with 'Query'

# Get the line number of the last 'Iteration'
iteration_line=$(grep -n '^Iteration' "$input_file" | tail -n 1 | cut -d: -f1)

if [ -z "$iteration_line" ]; then
    echo "No 'Iteration' line found. Output is identical to input."
    cp "$input_file" "$output_file"
    exit 0
fi

# Extract head (everything up to and including Iteration)
head -n "$iteration_line" "$input_file" > "$output_file"

# Extract tail (everything after Iteration), and clean 'Query' lines
tail -n +"$((iteration_line + 1))" "$input_file" | grep -v '^Query' >> "$output_file"

echo "Cleaned output saved to $output_file"
