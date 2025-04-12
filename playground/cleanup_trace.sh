#!/bin/bash
# filepath: /data/home/cluo86/DiskANN/playground/cleanup_trace.sh

# Check if an input file is provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 <input_file> [keep_original=false]"
    exit 1
fi

# Input file (first argument)
input_file="$1"

# Whether to keep the original file (default is false)
keep_original=${2:-false}

if [ ! -f "$input_file" ]; then
    echo "Error: Input file '$input_file' not found."
    exit 1
fi

# Create temp files
temp_file1=$(mktemp)
temp_file2=$(mktemp)
original_size=$(wc -c < "$input_file")

echo "Cleaning trace file '$input_file'..."

# Step 1: Extract content starting from "Loading the cache list..." line
echo "Step 1: Removing header content..."
awk '/Loading the cache list into memory....done./ {found=1} found' "$input_file" > "$temp_file1"

# Step 2: Remove 'Query' lines after the last 'Iteration' line
echo "Step 2: Removing trailing recall lines..."

# Get the line number of the last 'Iteration'
iteration_line=$(grep -n '^Iteration' "$temp_file1" | tail -n 1 | cut -d: -f1)

if [ -z "$iteration_line" ]; then
    echo "No 'Iteration' line found. Skipping step 2."
    cp "$temp_file1" "$temp_file2"
else
    # Extract head (everything up to and including Iteration)
    head -n "$iteration_line" "$temp_file1" > "$temp_file2"

    # Extract tail (everything after Iteration), and clean 'Query' lines
    tail -n +"$((iteration_line + 1))" "$temp_file1" | grep -v '^Query' >> "$temp_file2"
fi

# Create backup if requested
if [ "$keep_original" = "true" ]; then
    backup_file="${input_file}.bak"
    echo "Creating backup of original file as '$backup_file'"
    cp "$input_file" "$backup_file"
fi

# Replace original file with cleaned version
cp "$temp_file2" "$input_file"

# Clean up temp files
rm -f "$temp_file1" "$temp_file2"

# Calculate statistics
cleaned_size=$(wc -c < "$input_file")
reduction=$((original_size - cleaned_size))
percentage=$((reduction * 100 / original_size))
echo "File size reduced from $(numfmt --to=iec-i --suffix=B $original_size) to $(numfmt --to=iec-i --suffix=B $cleaned_size) ($percentage% reduction)"

echo "Trace cleanup complete: '$input_file'"