#!/bin/bash
export PYTHONPATH='/home/nibio/mutable-outside-world'

# Base directories
# BASE_DIR=~/data/test_data_agnostic_instanceSeg

BASE_DIR=/home/nibio/data/timing_check

# BASE_DIR=/home/nibio/mutable-outside-world/maciek_1_2_3  # Debugging

# Plot names
# declare -a PLOTS=("austrian_plot" "english_plot" "for_instance" "german_plot" "mls")

declare -a PLOTS=("austrian_plot" "english_plot" "for_instance_no_outer" "german_plot" "mls")

# declare -a PLOTS=("for_instance_no_outer")


# declare -a PLOTS=("b_0" "b_1")  # Debugging


# Function to run inference tests
run_inference_tests() {
    local ORIGINAL_DIR=$1
    local RESULTS_DIR=$2

    for PLOT in "${PLOTS[@]}"; do
        echo "Running paper test for ${PLOT}"
        start_time=$(date +%s.%N)  # Record start time
        bash run_inference.sh ${ORIGINAL_DIR}/${PLOT}/ ${RESULTS_DIR}/${PLOT}_out/
        end_time=$(date +%s.%N)  # Record end time
        execution_time=$(echo "$end_time - $start_time" | bc)  # Calculate execution time
        echo "Execution time for ${PLOT}: ${execution_time} seconds"
    done
}

# Function to run metrics
run_metrics() {
    local RESULTS_DIR=$1

    for PLOT in "${PLOTS[@]}"; do
        echo "Running metrics for ${PLOT}"
        python3 metrics/instance_segmentation_metrics_in_folder.py \
            --gt_las_folder_path ${RESULTS_DIR}/${PLOT}_out/final_results/ \
            --target_las_folder_path ${RESULTS_DIR}/${PLOT}_out/final_results/ \
            --output_folder_path ${RESULTS_DIR}/${PLOT}_out/metrics_out \
            --remove_ground --verbose
    done
}

# Run tests and metrics for a given dataset (e.g., sparse_1000 or sparse_500)
run_paper_tests() {
    local DATASET_NAME=$1

    echo "Running paper tests for ${DATASET_NAME}..."

    local ORIGINAL_DIR=${BASE_DIR}/${DATASET_NAME}
    local RESULTS_DIR=${BASE_DIR}/results_$(echo ${DATASET_NAME} | grep -oP '\d+')

    run_inference_tests ${ORIGINAL_DIR} ${RESULTS_DIR}
    # run_metrics ${RESULTS_DIR}

    echo "Done with paper tests for ${DATASET_NAME}"
}

# Execute the tests
run_paper_tests "sparse_10"
run_paper_tests "sparse_100"
run_paper_tests "sparse_500"
run_paper_tests "sparse_1000"
run_paper_tests "original_as_is"


# run_paper_tests "a_500"  # Debugging
# run_paper_tests "a_1000"  # Debugging

