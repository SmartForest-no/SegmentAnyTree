import os
import pandas as pd
from tqdm import tqdm

from nibio_inference.ply_to_pandas import ply_to_pandas
from nibio_inference.las_to_pandas import las_to_pandas
from nibio_inference.pandas_to_ply import pandas_to_ply

def split_pointcloud(df, x_step, y_step, overlap=0.0, verbose=False):
    """
    Splits the point cloud DataFrame based on x and y step sizes with overlap.
    
    Args:
    - df (pd.DataFrame): Point cloud data with 'X', 'Y' and other columns.
    - x_step (float): Width of the grid cell along the X-axis.
    - y_step (float): Height of the grid cell along the Y-axis.
    - overlap (float): Overlap fraction. E.g., 0.2 means 20% overlap.
    
    Returns:
    - list of pd.DataFrame: A list of DataFrames, each representing a grid cell's worth of points.
    """
    
    # if overlap is not between 0 and 1, raise an error
    if overlap < 0 or overlap > 1:
        raise ValueError(f"Overlap must be between 0 and 1, but got {overlap}")
    
    # Determine the bounding box of the data
    # check if x or X are in the csv file and convert to X and Y and Z if needed
    if 'x' in df.columns:
        df = df.rename(columns={'x': 'X'})
    if 'y' in df.columns:
        df = df.rename(columns={'y': 'Y'})
    if 'z' in df.columns:
        df = df.rename(columns={'z': 'Z'})

    x_min = df['X'].min()
    x_max = df['X'].max()
    y_min = df['Y'].min()
    y_max = df['Y'].max()

    # Create empty list to store chunks
    chunks = []

    # Compute overlap step
    x_overlap_step = x_step * overlap
    y_overlap_step = y_step * overlap

    # Split the data based on the grid defined by x_step, y_step, and overlap
    x_current = x_min
    while x_current < x_max:
        y_current = y_min
        while y_current < y_max:
            # Filter the points that fall within the current grid cell, including overlap
            chunk = df[(df['X'] >= x_current) & (df['X'] < x_current + x_step) &
                       (df['Y'] >= y_current) & (df['Y'] < y_current + y_step)]
            
            # Append the chunk to our list
            chunks.append(chunk)
            
            # Move to the next grid cell along Y-axis, accounting for overlap
            y_current += (y_step - y_overlap_step)
        
        # Move to the next grid cell along X-axis, accounting for overlap
        x_current += (x_step - x_overlap_step)

    return chunks

# Example usage:
# chunks = split_pointcloud(df, x_step=10, y_step=10, overlap=0.2)



# input_file = '/home/nibio/mutable-outside-world/data_for_test/small_1.ply'
# output_dir = '/home/nibio/mutable-outside-world/split_test_out'

# df = ply_to_pandas(input_file)
# chunks = split_pointcloud(df, x_step=10, y_step=10, overlap=0.2)

# for i, chunk in enumerate(chunks):
#     chunk.to_csv(os.path.join(output_dir, f'chunk_{i}.csv'), index=False)


# if main

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Split point clouds in chunks based on x, y steps and overlap.")
    parser.add_argument('-i', '--input_dir', type=str, required=True, help="Input directory containing point cloud files.")
    parser.add_argument('-o', '--output_dir', type=str, required=True, help="Output directory to save chunked point cloud files.")
    parser.add_argument('--x_step', type=float, required=False, default=10.0, help="Width of the grid cell along the X-axis.")
    parser.add_argument('--y_step', type=float, required=False,  default=10.0, help="Height of the grid cell along the Y-axis.")
    parser.add_argument('--overlap', type=float, default=0.0, help="Overlap fraction (e.g., 0.2 means 20% overlap). Default is 0.0. Must be between 0 and 1.")

    args = parser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    x_step = args.x_step
    y_step = args.y_step
    overlap = args.overlap

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Process each file in the directory
    for file_name in tqdm(os.listdir(input_dir)):
        input_file = os.path.join(input_dir, file_name)

        if input_file.endswith('.laz') or input_file.endswith('.las'):
            df = las_to_pandas(input_file)
        elif input_file.endswith('.ply'):
            df = ply_to_pandas(input_file)
        else:
            print(f"Skipping unsupported file format: {input_file}")
            continue

        # Split point cloud
        chunks = split_pointcloud(df, x_step=x_step, y_step=y_step, overlap=overlap)

        # Save chunks to output directory as ply files
        for i, chunk in enumerate(chunks):
            output_file = os.path.join(output_dir, f'{os.path.splitext(file_name)[0]}_chunk_{i}.ply')
            pandas_to_ply(chunk, csv_file_provided=False, output_file_path=output_file)