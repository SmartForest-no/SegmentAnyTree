import os
import csv
import argparse
import pandas as pd

import shutil  # Legg til dette øverst i filen din

from nibio_inference.merge_inference_results import main as merge_main


def process_folders(list_of_folders, output_file, verbose):
    """Process the list of folders and merge metrics."""

    # Create a temporary folder to hold the merged metrics
    tmp_folder = 'tmp_merged_metrics'
    os.makedirs(tmp_folder, exist_ok=True)

    # Traverse directories
    for folder in list_of_folders:
        if verbose:
            print(f"Processing folder {folder}")
        # Remove trailing slash if it exists
        folder = folder.rstrip('/')
        # perform merge and generate a temporary file which is named after the folder
        temp_output_file = os.path.join(tmp_folder, os.path.basename(folder) + '.csv')
        merge_main(folder, temp_output_file)

    # read all the csv files from the temporary folder as pandas dataframes and merge them so that they are accoring to the order of the list_of_folders

    df = pd.DataFrame()
    for file in os.listdir(tmp_folder):
        if file.endswith('.csv'):
            df = df.append(pd.read_csv(os.path.join(tmp_folder, file)), ignore_index=True)

    # Delete the temporary folder
    shutil.rmtree(tmp_folder) 

    # Write to csv
    df.to_csv(output_file, index=False)


if __name__ == '__main__':
    list_of_folders = [
        "/home/nibio/data/test_data_agnostic_instanceSeg/results_/",
        "/home/nibio/data/test_data_agnostic_instanceSeg/results_1000/",
        "/home/nibio/data/test_data_agnostic_instanceSeg/results_500/",
        "/home/nibio/data/test_data_agnostic_instanceSeg/results_100/",
        "/home/nibio/data/test_data_agnostic_instanceSeg/results_10/"
    ]

    output_file = 'merged_metrics_all.csv'
    verbose = True

    process_folders(list_of_folders, output_file, verbose)
