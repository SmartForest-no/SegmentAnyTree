import argparse
import json
import os

from tqdm import tqdm

from nibio_inference.las_to_pandas import las_to_pandas
from nibio_inference.pandas_to_ply import pandas_to_ply
from nibio_inference.ply_to_pandas import ply_to_pandas


def ply_modification_pipeline(input_file_path, output_file_path):
    #TODO: to correct the x to X, y to Y, z to Z as in las files


    # read the input file
    points_df = ply_to_pandas(input_file_path, csv_file_path=None)

    ### steps to do #########

    # find the min value of x, y, z
    min_x = points_df['x'].min()
    min_y = points_df['y'].min()
    min_z = points_df['z'].min()

    # subtract the min value from x, y, z
    points_df['x'] = points_df['x'] - min_x
    points_df['y'] = points_df['y'] - min_y
    points_df['z'] = points_df['z'] - min_z

    # save the min values to a json file
    min_values = [min_x, min_y, min_z]

    min_values = [float(min_x), float(min_y), float(min_z)]  # Convert min_x, min_y, and min_z to float

    with open('min_values.json', 'w') as f:
        json.dump(min_values, f)

    ### end of steps to do ###

    # save the input file as output file
    pandas_to_ply(points_df, csv_file_provided=False, output_file_path=output_file_path)

    return None

def las_modification_pipeline(input_file_path, output_file_path, json_file_path=None):
    # read the input file
    points_df = las_to_pandas(input_file_path, csv_file_path=None)

    ### steps to do #########

  # find the min value of X, Y, Z
    min_X = points_df['X'].min()
    min_Y = points_df['Y'].min()
    min_Z = points_df['Z'].min()

    # subtract the min value from X, Y, Z
    points_df['X'] = points_df['X'] - min_X
    points_df['Y'] = points_df['Y'] - min_Y
    points_df['Z'] = points_df['Z'] - min_Z


    # save the min values to a json file
    min_values = [min_X, min_Y, min_Z]
    min_values = [float(min_X), float(min_Y), float(min_Z)]  # Convert min_X, min_Y, and min_Z to float

    with open(json_file_path, 'w') as f:
        json.dump(min_values, f)

    ### end of steps to do ###

    # save the input file as output file
    pandas_to_ply(points_df, csv_file_provided=False, output_file_path=output_file_path)

    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process las or laz files and save results as ply files.')
    parser.add_argument('-i', '--input_folder', type=str, help='Path to the input folder containing ply files.')
    parser.add_argument('-o', '--output_folder', type=str, help='Path to the output folder to save las files.')

    args = parser.parse_args()

    # Ensure output directory exists
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)


    # write how many files are processed
    print("Processing " + str(len(os.listdir(args.input_folder))) + " files...")
    # Loop through each file in the input directory with tqdm for a progress bar
    for filename in tqdm(os.listdir(args.input_folder), desc='Processing files', unit='file'): 
      
        input_file_path = os.path.join(args.input_folder, filename)
        output_file_name = os.path.splitext(filename)[0] + "_out.ply"
        output_file_path = os.path.join(args.output_folder, output_file_name)
            
        # run the pipeline
        if filename.endswith(".ply"):
            ply_modification_pipeline(input_file_path, output_file_path)
        elif filename.endswith(".las") or filename.endswith(".laz"):
            las_modification_pipeline(
                input_file_path, 
                output_file_path, 
                json_file_path=os.path.join(args.output_folder, os.path.splitext(filename)[0] + "_min_values.json")
            )

        # move the json file to the output folder and rename it after the output file core name
        # os.rename("min_values.json", os.path.join(args.output_folder, os.path.splitext(filename)[0] + "_out_min_values.json"))

    # write where the output files are saved
    print("Output files are saved in: " + args.output_folder)