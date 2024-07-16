import argparse
import json
import os
from joblib import Parallel, delayed
from nibio_inference.las_to_pandas import las_to_pandas
from nibio_inference.pandas_to_ply import pandas_to_ply
from nibio_inference.ply_to_pandas import ply_to_pandas


def process_file(filename, input_folder, output_folder):
    input_file_path = os.path.join(input_folder, filename)
    base_filename = os.path.splitext(filename)[0]
    output_file_path = os.path.join(output_folder, f"{base_filename}_out.ply")
    json_file_path = os.path.join(output_folder, f"{base_filename}_out_min_values.json")

    if filename.endswith((".ply", ".las", ".laz")):
        modification_pipeline(input_file_path, output_file_path, json_file_path, filename.endswith(".ply"))


def modification_pipeline(input_file_path, output_file_path, json_file_path, is_ply):
    coord_names = ['x', 'y', 'z'] if is_ply else ['X', 'Y', 'Z']
    print(f"Processing in utm2local: {input_file_path}")

    points_df = ply_to_pandas(input_file_path) if is_ply else las_to_pandas(input_file_path)

    min_values = {name: points_df[name].min() for name in coord_names}
    for name in coord_names:
        points_df[name] -= min_values[name]

    min_values_list = [float(val) for val in min_values.values()]

    with open(json_file_path, 'w') as f:
        print(f"Saving min values to: {json_file_path}")
        json.dump(min_values_list, f)

    pandas_to_ply(points_df, csv_file_provided=False, output_file_path=output_file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process las or laz files and save results as ply files.')
    parser.add_argument('-i', '--input_folder', type=str, help='Path to the input folder containing ply files.')
    parser.add_argument('-o', '--output_folder', type=str, help='Path to the output folder to save las files.')

    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)

    filenames = os.listdir(args.input_folder)

    print(f"Processing {len(filenames)} files...")
    Parallel(n_jobs=4)(
        delayed(process_file)(filename, args.input_folder, args.output_folder) for filename in filenames
    )
    print(f"Output files are saved in: {args.output_folder}")
