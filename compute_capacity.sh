#!/bin/bash

# Base directories
BASE_DIR=/home/nibio/data/timing_check

# Plot names
declare -a PLOTS=("austrian_plot" "english_plot" "for_instance_no_outer" "german_plot" "mls")

# Function to calculate folder sizes
calculate_folder_sizes() {
    local ORIGINAL_DIR=$1

    for PLOT in "${PLOTS[@]}"; do
        echo "Calculating size for ${PLOT}"
        du -sh ${ORIGINAL_DIR}/${PLOT}/ | awk '{print $1 " GB\t" $2}'
    done
}

# Run size calculation for a given dataset (e.g., sparse_1000 or sparse_500)
run_size_calculation() {
    local DATASET_NAME=$1

    echo "Calculating sizes for ${DATASET_NAME}..."

    local ORIGINAL_DIR=${BASE_DIR}/${DATASET_NAME}

    calculate_folder_sizes ${ORIGINAL_DIR}

    echo "Done calculating sizes for ${DATASET_NAME}"
}

# Execute the size calculations
run_size_calculation "sparse_10"
run_size_calculation "sparse_100"
run_size_calculation "sparse_500"
run_size_calculation "sparse_1000"
run_size_calculation "original_as_is"

# Additional datasets can be added as needed
