#!/bin/bash

# Assign command-line arguments to variables
results_folder=$1
sparse_folder=$2

# Ensure the results folder exists
mkdir -p "$results_folder"

# Base directory where the sparse results are located
base_dir="/home/nibio/mutable-outside-world/for_instance_no_outer_sparse_many_times/${sparse_folder}"

# Call the Python script for each results directory
for i in {10,25,50,75,100}; do
    python3 nibio_inference/merge_inference_results.py -i "${base_dir}/results_${i}" -o "${results_folder}/results_${i}.csv"
done

# Merge CSV files
{
  # Handle the first file separately to keep its header
  head -n 1 "${results_folder}/results_10.csv" &&

  # Loop through the files, tail -n +2 skips the header of each file
  for i in 100 75 50 25 10; do
    tail -n +2 "${results_folder}/results_${i}.csv"
  done
} > "${results_folder}/merged_results.csv"

