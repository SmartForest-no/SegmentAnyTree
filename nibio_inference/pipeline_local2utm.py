import os
import json
import argparse
import jaklas

from tqdm import tqdm

from nibio_inference.pandas_to_ply import pandas_to_ply
from nibio_inference.pandas_to_las import pandas_to_las  # Assuming you have this function
from nibio_inference.ply_to_pandas import ply_to_pandas
from nibio_inference.las_to_pandas import las_to_pandas  # Assuming you have this function


def revert_ply_modification(input_file_path, min_values_path, output_file_path):
    # read the input file
    points_df = ply_to_pandas(input_file_path)

    with open(min_values_path, 'r') as f:
        min_values = json.load(f)
    
    min_x, min_y, min_z = min_values

    # add the min values back to x, y, z
    points_df['x'] = points_df['x'] + min_x
    points_df['y'] = points_df['y'] + min_y
    points_df['z'] = points_df['z'] + min_z

    # save the modified file
    pandas_to_ply(points_df, output_file_path=output_file_path)


def revert_las_modification(input_file_path, min_values_path, output_file_path):
    # read the input file
    points_df = las_to_pandas(input_file_path)

    with open(min_values_path, 'r') as f:
        min_values = json.load(f)
    
    min_x, min_y, min_z = min_values

    # add the min values back to x, y, z
    points_df['x'] = points_df['x'] + min_x
    points_df['y'] = points_df['y'] + min_y
    points_df['z'] = points_df['z'] + min_z

    # save the modified file
    jaklas.write(points_df, output_file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Revert the modification of ply and las files.')
    parser.add_argument('-i', '--input_folder', type=str, help='Path to the input folder containing modified files.')
    parser.add_argument('-m', '--min_values_folder', type=str, help='Path to the folder containing min_values.json files.')
    parser.add_argument('-o', '--output_folder', type=str, help='Path to the output folder to save reverted files.')

    args = parser.parse_args()

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    for filename in tqdm(os.listdir(args.input_folder), desc='Reverting files', unit='file'):
        input_file_path = os.path.join(args.input_folder, filename)
        min_values_path = os.path.join(args.min_values_folder, os.path.splitext(filename)[0] + "_min_values.json")
        output_file_name = os.path.splitext(filename)[0] + "_reverted.ply"
        output_file_path = os.path.join(args.output_folder, output_file_name)
        
        if filename.endswith(".ply"):
            revert_ply_modification(input_file_path, min_values_path, output_file_path)
        elif filename.endswith(".las"):
            revert_las_modification(input_file_path, min_values_path, output_file_path)

    print("Output files are saved in: " + args.output_folder)
