import os
import jaklas
from nibio_inference.las_to_pandas import las_to_pandas
from nibio_inference.pandas_to_las import pandas_to_las
import argparse


def remove_outer_points_for_instance(las_file, output_file_path):
    """
    Remove points outside of the instance bounding box
    """
    df = las_to_pandas(las_file)

    # leave only the points which are not in the instance
    df = df[df['classification'] != 3]

    # print columns
    # print(df.columns)

    # use jaklas to save the dataframe to las file with a new name
    jaklas.write(df, output_file_path)
    
    # pandas_to_las(df, 
    #               csv_file_provided=False,
    #               output_file_path=output_file_path,
    #               verbose=False)


if __name__ == "__main__":
    # las_file = "/home/nibio/mutable-outside-world/test_sparsity/CULS_plot_2_annotated.las"
    
    parser = argparse.ArgumentParser(description='Remove points outside of the instance bounding box.')

    # read a folder name 
    parser.add_argument('-i', '--input_folder', type=str, help='Path to the input folder containing modified files.')
    parser.add_argument('-o', '--output_folder', type=str, help='Path to the output folder to save reverted files.')

    args = parser.parse_args()

    # read all the files in the folder
    las_files = os.listdir(args.input_folder)

    # check if there are laz or las files
    las_files = [f for f in las_files if f.endswith(".las") or f.endswith(".laz")]

    # loop over all the files
    for las_file in las_files:
        print("Processing file: {}".format(las_file))
        # remove the outer points
        remove_outer_points_for_instance(os.path.join(args.input_folder, las_file), os.path.join(args.output_folder, las_file))

    # print the number of files
    print("Processed {} files. Done.".format(len(las_files)))

