import json
from nibio_inference.ply_to_pandas import ply_to_pandas
from nibio_inference.pandas_to_ply import pandas_to_ply


def bring_back_to_utm_coordinates(path_to_new_file, path_to_the_old_file):
    points_df = ply_to_pandas(path_to_new_file)

    min_values_path = path_to_the_old_file.replace('.ply', '_min_values.json')

    with open(min_values_path, 'r') as f:
        min_values = json.load(f)
    
    min_x, min_y, min_z = min_values

    # add the min values back to x, y, z
    points_df['x'] = points_df['x'] + min_x
    points_df['y'] = points_df['y'] + min_y
    points_df['z'] = points_df['z'] + min_z

    # save the modified file
    pandas_to_ply(points_df, output_file_path=path_to_new_file)