#!/bin/bash
set -e

export PYTHONPATH='/home/nibio/mutable-outside-world'

# Provide path to the input and output directories and also information if to clean the output directory from command line
SOURCE_DIR="$1"
DEST_DIR="$2"
CLEAN_OUTPUT_DIR="$3"

# Set default values if not provided
: "${SOURCE_DIR:=/home/nibio/mutable-outside-world/data_for_test}"
: "${DEST_DIR:=/home/nibio/mutable-outside-world/data_for_test_results}"
: "${CLEAN_OUTPUT_DIR:=true}"

# If someone fails to provide this correctly then print the usage and exit
if [ -z "$SOURCE_DIR" ] || [ -z "$DEST_DIR" ] || [ -z "$CLEAN_OUTPUT_DIR" ]; then
    echo "Usage: run_inference.sh <path_to_input_dir> <path_to_output_dir> <clean_output_dir>"
    exit 1
fi

# Clear the output directory if requested
if [ "$CLEAN_OUTPUT_DIR" = "true" ]; then
    rm -rf "$DEST_DIR"/*
fi

# Check if local paths are provided, change them to absolute paths if so
if [[ "$SOURCE_DIR" != /* ]]; then
    SOURCE_DIR=$(pwd)/"$SOURCE_DIR"
fi

if [[ "$DEST_DIR" != /* ]]; then
    DEST_DIR=$(pwd)/"$DEST_DIR"
fi

# Print the input and output directories
echo "Input directory: $SOURCE_DIR"
echo "Output directory: $DEST_DIR"

# Rename the input files to the format that the inference script expects 
# Change '-' to '_' in the file names

# Copy the input files to the output directory to avoid changing the original files in an input_data directory
mkdir -p "$DEST_DIR/input_data"
cp -r "$SOURCE_DIR/"* "$DEST_DIR/input_data/"

python3 nibio_inference/fix_naming_of_input_files.py "$DEST_DIR/input_data"

# UTM normalization 
python3 nibio_inference/pipeline_utm2local_parallel.py -i "$DEST_DIR/input_data" -o "$DEST_DIR/utm2local"

# Update the eval.yaml file with the correct paths
cp /home/nibio/mutable-outside-world/conf/eval.yaml "$DEST_DIR"
python3 nibio_inference/modify_eval.py "$DEST_DIR/eval.yaml" "$DEST_DIR/utm2local" "$DEST_DIR"

# clear cache
python3 nibio_inference/clear_cache.py --eval_yaml "$DEST_DIR/eval.yaml"

# Run the inference script with the config file
python3 eval.py --config-name "$DEST_DIR/eval.yaml"

echo "Done with inference using the config file: $DEST_DIR/eval.yaml"

# Rename the output files result_0.ply , result_1.ply, ... to the original file names but with the prefix "inference_"
python3 /home/nibio/mutable-outside-world/nibio_inference/rename_result_files_instance.py "$DEST_DIR/eval.yaml" "$DEST_DIR"

# Rename segmentation files
python3 /home/nibio/mutable-outside-world/nibio_inference/rename_result_files_segmentation.py "$DEST_DIR/eval.yaml" "$DEST_DIR"

FINAL_DEST_DIR="$DEST_DIR/final_results"

# Run merge script
python3 /home/nibio/mutable-outside-world/nibio_inference/merge_pt_ss_is_in_folders.py -i "$DEST_DIR/utm2local" -s "$DEST_DIR" -o "$FINAL_DEST_DIR" -v

# remove numbers in the beginning of the file names

# Loop through all files in the specified folder
for file in "$FINAL_DEST_DIR"/*; do
    # Extract just the filename from the path
    filename=$(basename "$file")

    # Use parameter expansion to remove the initial number and underscore
    new_name=$(echo "$filename" | sed 's/^[0-9]*_//')

    # Construct the new file path
    new_file_path="$FINAL_DEST_DIR/$new_name"

    # Rename the file
    mv -n "$file" "$new_file_path"
done

# Avoid using ls to count files. This way, you can handle filenames with newlines or other problematic characters.
num_files=$(find "$FINAL_DEST_DIR" -maxdepth 1 -type f | wc -l)

echo "Number of files in the final results directory: $num_files"
