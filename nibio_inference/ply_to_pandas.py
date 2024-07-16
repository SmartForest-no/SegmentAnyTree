import numpy as np
import pandas as pd
from plyfile import PlyData

def ply_to_pandas(ply_file_path, csv_file_path=None):
    """
    Reads a PLY file and converts its vertex data to a numpy array, then saves it to a CSV file.

    Args:
        ply_file_path (str): The path to the PLY file to be read.
        csv_file_path (str, optional): The path to the CSV file to be saved. If not provided, 
                                       the output CSV will have the same name as the input PLY file.

    Returns:
        numpy.ndarray: A 2D array containing the vertex data from the PLY file.
    """
    # Load the PLY file
    ply_content = PlyData.read(ply_file_path)

    # Identify available elements in the PLY file
    available_elements = [elem.name for elem in ply_content.elements]

    # Determine the element that represents the point cloud data
    if 'vertex' in available_elements:
        point_element_name = 'vertex'
    elif 'point' in available_elements:
        point_element_name = 'point'
    else:
        print("Could not find a suitable element for point cloud data.")
        return

    # Extract data from the chosen element
    point_data = ply_content[point_element_name].data
    property_names = point_data.dtype.names
    data_arrays = [point_data[prop_name] for prop_name in property_names]
    all_points = np.vstack(data_arrays).T

    # Save the data to a CSV file
    if csv_file_path is not None:
        np.savetxt(csv_file_path, all_points, delimiter=',', header=','.join(property_names), comments='', fmt='%s')

    # put the data into a pandas dataframe
    points_df = pd.DataFrame(all_points, columns=property_names)

    return points_df

if __name__ == "__main__":
    # ply_path = "/home/nibio/mutable-outside-world/code/gitlab_fsct/instance_segmentation_classic/Binbin_data_paper/TUWIEN_test_test.ply"
    # csv_path = "/home/nibio/mutable-outside-world/code/gitlab_fsct/instance_segmentation_classic/Binbin_data_paper/TUWIEN_test_test.csv"
    import argparse
    parser = argparse.ArgumentParser(description='Convert a PLY file to a CSV file.')
    parser.add_argument('ply_path', type=str, help='Path to the PLY file to be read.')
    parser.add_argument('csv_path', type=str, help='Path to the CSV file to be saved.')
    args = parser.parse_args()
    ply_path = args.ply_path
    csv_path = args.csv_path

    ply_to_pandas(ply_path, csv_path)
