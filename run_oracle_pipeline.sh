#!/bin/bash
set -e

# Set DEBUG_MODE (change this to true or false as needed, DEBUG is for local testing, false is for running on the oracle)
# DEBUG_MODE=true
DEBUG_MODE=false

# Set the path (change this to the path taken from the config file)
# if [ "$DEBUG_MODE" = true ]; then
#     PATH_DATA='/home/nibio/mutable-outside-world' #TODO: change this to the path
# else
#     PATH_DATA='/home/datascience'
# fi

PATH_DATA='/home/datascience'

# Set the input and output folders in the oracle
ORACLE_IN_DATA_FOLDER="$PATH_DATA/docker_in_folder" # This is the folder where the input data is stored on the oracle
ORACLE_OUT_DATA_FOLDER="$PATH_DATA/docker_out_folder" # This is the folder where the output data is stored on the oracle

TMP_IN_DATA_FOLDER="$PATH_DATA/tmp_in_folder" # This is the folder where the input data is stored on the oracle temporarily
TMP_OUT_DATA_FOLDER="$PATH_DATA/tmp_out_folder" # This is the folder where the output data is stored on the oracle temporarily

# Set the input and output folders which mimic the bucket
DOCKER_IN_FOLDER='/home/nibio/mutable-outside-world/bucket_in_folder' # this just mimics the input bucket
DOCKER_OUT_FOLDER='/home/nibio/mutable-outside-world/bucket_out_folder' # this just mimics the output bucket

# function to read the input from the oracle
run_oracle_wrapper_input() {
    if [ "$DEBUG_MODE" = true ]; then
        # This is mapped in the docker run
        bucket_location=${DOCKER_IN_FOLDER}
    else
        # Get the input location from the environment variable
        bucket_location=${OBJ_INPUT_LOCATION}

        # Remap the input location
        bucket_location=${bucket_location//@axqlz2potslu/}
        bucket_location=${bucket_location//oci:\/\//\/mnt\/}
    fi

    # Create the input folder if it does not exist in the docker container
    mkdir -p "$ORACLE_IN_DATA_FOLDER"

    # Copy files from bucket_location to the input folder
    shopt -s nullglob # Enable nullglob to handle empty directories
    cp -r "$bucket_location"/* "$ORACLE_IN_DATA_FOLDER"

    #check if in ORACLE_IN_DATA_FOLDER there are .zip files
    for file in "$ORACLE_IN_DATA_FOLDER"/*.zip; do
        if [ -f "$file" ]; then
            # Unzip the file
            echo "Unzipping $file"
            unzip "$file" -d "$ORACLE_IN_DATA_FOLDER"
            # Remove the zip file
            rm "$file"
        fi
    done
}

# function to write the output to the oracle
run_oracle_wrapper_output() {
    if [ "$DEBUG_MODE" = true ]; then
        # This is mapped in the docker run
        bucket_location=${DOCKER_OUT_FOLDER}
    else
        # Get the output location from the environment variable
        bucket_location=${OBJ_OUTPUT_LOCATION}

        # Remap the output location
        bucket_location=${bucket_location//@axqlz2potslu/}
        bucket_location=${bucket_location//oci:\/\//\/mnt\/}
    fi

    # Create the output folder if it does not exist in the docker container
    mkdir -p "$bucket_location"

    # make a temporary folder to store the results
    mkdir -p "$PATH_DATA/results"

    # copy the results from the oracle output folder to the temporary folder
    cp -r "$ORACLE_OUT_DATA_FOLDER/final_results/"* "$PATH_DATA/results"

    # Find and zip only the files in the specified directory, excluding subfolders
    find "$PATH_DATA/results" -maxdepth 1 -type f -exec zip "$PATH_DATA/results.zip" {} +

    # copy the zipped results to the output location
    cp "$PATH_DATA/results.zip" "$bucket_location"

    # # Zip the output folder
    # zip "$ORACLE_OUT_DATA_FOLDER/results.zip" "$ORACLE_OUT_DATA_FOLDER/final_results"/*

    # # Copy the zipped folder to the output_location
    # cp "$ORACLE_OUT_DATA_FOLDER/results.zip" "$bucket_location"
}

### Main execution ###

# Run the input script
run_oracle_wrapper_input

# Create temporary folders if they do not exist
mkdir -p "$TMP_IN_DATA_FOLDER"
mkdir -p "$TMP_OUT_DATA_FOLDER"
mkdir -p "$ORACLE_OUT_DATA_FOLDER/final_results"

# Get a list of files in ORACLE_IN_DATA_FOLDER
# Save the current IFS
OLD_IFS="$IFS"
# Change IFS to handle only newline characters
IFS=$'\n'

files=($(find "$ORACLE_IN_DATA_FOLDER" -maxdepth 1 -type f))

IFS="$OLD_IFS"

# Process files in chunks of 10
for ((i=0; i<${#files[@]}; i+=10)); do
    # Copy up to 10 files to TMP_IN_DATA_FOLDER
    cp "${files[@]:i:10}" "$TMP_IN_DATA_FOLDER/"

    # Run the inference script on these files
    bash run_inference.sh "$TMP_IN_DATA_FOLDER" "$TMP_OUT_DATA_FOLDER"

    # Copy results from TMP_OUT_DATA_FOLDER to ORACLE_OUT_DATA_FOLDER
    cp -r "$TMP_OUT_DATA_FOLDER/final_results/"* "$ORACLE_OUT_DATA_FOLDER/final_results/"

    # Clear TMP_IN_DATA_FOLDER for the next batch
    rm -rf "$TMP_IN_DATA_FOLDER"/*
done

# Run the output script
run_oracle_wrapper_output

echo "Processing complete."
