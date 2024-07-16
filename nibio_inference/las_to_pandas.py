import numpy as np
import pandas as pd
import laspy

# works with laspy 2.1.2

def las_to_pandas(las_file_path, csv_file_path=None):
    file_content = laspy.read(las_file_path)

    basic_dimensions = list(file_content.point_format.dimension_names)

    # List all the available basic parameters from the LAS file
    # basic_dimensions = ['X', 'Y', 'Z', 'intensity', 'return_num', 'num_returns',
    #                     'scan_direction_flag', 'edge_of_flight_line', 'classification',
    #                     'synthetic', 'key_point', 'withheld', 'scan_angle_rank',
    #                     'user_data', 'pt_src_id']
    
    # Filter only available dimensions
    available_dimensions = [dim for dim in basic_dimensions if hasattr(file_content, dim.lower())]

    # Put all basic dimensions into a numpy array
    basic_points = np.vstack([getattr(file_content, dim.lower()) for dim in available_dimensions]).T
    
    # Fetch any extra dimensions
    gt_extra_dimensions = list(file_content.point_format.extra_dimension_names)

    # get only the extra dimensions which are not already in the basic dimensions
    gt_extra_dimensions = list(set(gt_extra_dimensions) - set(available_dimensions))

    if gt_extra_dimensions:
        extra_points = np.vstack([getattr(file_content, dim) for dim in gt_extra_dimensions]).T
        # Combine basic and extra dimensions
        all_points = np.hstack((basic_points, extra_points))
        all_columns = available_dimensions + gt_extra_dimensions
    else:
        all_points = basic_points
        all_columns = available_dimensions

    # Create dataframe
    points_df = pd.DataFrame(all_points, columns=all_columns)

    # convert x, y, z to float
    # points_df['X'] = points_df['X'].astype(float)
    # points_df['Y'] = points_df['Y'].astype(float)
    # points_df['Z'] = points_df['Z'].astype(float)

    # Save pandas dataframe to csv
    if csv_file_path is not None:
        points_df.to_csv(csv_file_path, index=False, header=True, sep=',')

    return points_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Convert las or laz files to pandas dataframes.')
    parser.add_argument('-i', '--input_file', type=str, help='Path to the input file.')
    parser.add_argument('-o', '--output_file', type=str, help='Path to the output file.')

    args = parser.parse_args()

    las_to_pandas(args.input_file, args.output_file)
