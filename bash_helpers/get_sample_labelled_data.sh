#!/bin/bash

# This script is used to get the labelled data from the sample data
SOURCE_DIR="/home/nibio/mutable-outside-world/sample_test_data/labelled/"
DEST_DIR="/home/nibio/mutable-outside-world/data_for_test"


# clear the destination directory
rm -rf $DEST_DIR/*

# Copy the files
cp -r $SOURCE_DIR/* $DEST_DIR

# print the files copied
echo "Files copied:"
ls $DEST_DIR

