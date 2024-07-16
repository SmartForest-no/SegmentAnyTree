#!/bin/bash

# Check if an argument is provided
# if [ "$#" -ne 1 ]; then
#     echo "Usage: $0 <folder A>"
#     exit 1
# fi

# Assign the folder A to a variable, if variable is empty then use default value of /home/nibio/mutable-outside-world/data_for_test
folderA="$1"
: "${folderA:=/home/nibio/mutable-outside-world/data_for_test}"


# Check if folder A exists
if [ ! -d "$folderA" ]; then
    echo "Folder A does not exist."
    exit 1
fi

# Create folder B and C
folderB="/home/nibio/mutable-outside-world/B_temp_folder"
folderC="/home/nibio/mutable-outside-world/data_for_test_results_final"
folderTemp="/home/nibio/mutable-outside-world/temp_folder"

# remove old folders
rm -rf "$folderB"
rm -rf "$folderC"
rm -rf "$folderTemp"

mkdir -p "$folderB"
mkdir -p "$folderC"
mkdir -p "$folderTemp"

# Function to copy and process files
process_files() {
    # Copy files to folder B
    cp "$@" "$folderB/"

    # Run inference script on folder B
    bash /home/nibio/mutable-outside-world/run_inference.sh "$folderB" "$folderTemp" 

    # Copy results to folder C
    cp "$folderTemp/final_results"/* "$folderC/"
}

# Get a list of files in folder A
files=($(find "$folderA" -maxdepth 1 -type f))

# Process files in chunks of 10
for ((i=0; i<${#files[@]}; i+=10)); do
    # check if folder B is empty if not remove all files
    if [ "$(ls -A "$folderB")" ]; then
        rm -rf "$folderB"/*
    fi
    process_files "${files[@]:i:10}"
done

echo "Processing complete."
