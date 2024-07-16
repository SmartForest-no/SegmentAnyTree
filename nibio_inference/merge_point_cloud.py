import os
import argparse
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from joblib import Parallel, delayed

from nibio_inference.ply_to_pandas import ply_to_pandas
from nibio_inference.las_to_pandas import las_to_pandas
from nibio_inference.pandas_to_ply import pandas_to_ply


def compute_centroid(points):
    centroid = points[['x', 'y', 'z']].mean()
    return tuple(int(x * 1000 + 0.5) / 1000 for x in centroid)


def process_chunk(chunk, prediction='preds'):
    centroid = compute_centroid(chunk)
    unique_values = chunk[prediction].unique()

    x_min, x_max = chunk['x'].min(), chunk['x'].max()
    y_min, y_max = chunk['y'].min(), chunk['y'].max()
    border_limit = 0.05

    chunk['x_border'] = ((chunk['x'] <= x_min + border_limit) | (chunk['x'] >= x_max - border_limit)).astype(int)
    chunk['y_border'] = ((chunk['y'] <= y_min + border_limit) | (chunk['y'] >= y_max - border_limit)).astype(int)

    border_trees = chunk[(chunk['x_border'] == 1) | (chunk['y_border'] == 1)]
    unique_values_border_trees = border_trees[prediction].unique()

    chunk['border_tree'] = chunk[prediction].isin(unique_values_border_trees).astype(int)
    chunk['x_centroid'], chunk['y_centroid'], chunk['z_centroid'] = 0, 0, 0

    border_trees_centroid = []
    for value in unique_values_border_trees:
        centroid_value = compute_centroid(chunk[chunk[prediction] == value])
        border_trees_centroid.append(centroid_value)
        chunk.loc[chunk[prediction] == value, ['x_centroid', 'y_centroid', 'z_centroid']] = centroid_value

    return centroid, unique_values, len(unique_values), border_trees_centroid, unique_values_border_trees


def merge_pointclouds(chunk_name, chunks_list, prediction='preds'):
    results = Parallel(n_jobs=-1, prefer="threads")(
        delayed(process_chunk)(chunk) for chunk in tqdm(chunks_list, desc="Computing centroids")
    )

    results_df = pd.DataFrame(
        results,
        columns=["centroid", "unique_values", "number_of_unique_values", "border_trees_centroid", "unique_values_border_trees"]
    )

    start_value = 0
    for index, row in results_df.iterrows():
        chunk = chunks_list[index]
        mapping = {old_val: new_val for old_val, new_val in zip(row['unique_values'], range(start_value, start_value + row['number_of_unique_values']))}
        chunk[prediction] = chunk[prediction].map(mapping)
        start_value += row['number_of_unique_values']

    # results_df.to_csv(f"{chunk_name}_results.csv")
    concatenated =  pd.concat(chunks_list, ignore_index=True)

    # compute  border tree centroid for concatenated point cloud using simular method as for chunks
    concatenated['x_border_global'] = ((concatenated['x'] <= concatenated['x'].min() + 0.05) | (concatenated['x'] >= concatenated['x'].max() - 0.05)).astype(int)
    concatenated['y_border_global'] = ((concatenated['y'] <= concatenated['y'].min() + 0.05) | (concatenated['y'] >= concatenated['y'].max() - 0.05)).astype(int)
    
    # find for border trees in concatenated point cloud
    border_trees_global = concatenated[(concatenated['x_border_global'] == 1) | (concatenated['y_border_global'] == 1)]
    unique_values_border_trees_global = border_trees_global[prediction].unique()

    # find centroid for border trees in concatenated point cloud
    concatenated['border_tree_global'] = concatenated[prediction].isin(unique_values_border_trees_global).astype(int)
    concatenated['x_centroid_global'], concatenated['y_centroid_global'], concatenated['z_centroid_global'] = 0, 0, 0

    # make all the trees which are local border trees and global border trees the same time not border trees in concatenated point cloud
    concatenated.loc[concatenated['border_tree_global'] == 1, 'border_tree'] = 0

    # ESSENTIAL UP 
    concatenated.sort_values(by=['x', 'y', 'z'], inplace=True)

    # get a list of min and max values for x and y in chunks as a tuple
    x_min_max = [(chunk['x'].min(), chunk['x'].max()) for chunk in chunks_list]
    y_min_max = [(chunk['y'].min(), chunk['y'].max()) for chunk in chunks_list]

    # print (x_min_max) and print (y_min_max)
    print(x_min_max)
    print(y_min_max)

    # get all centroids of the border trees in concatenated point cloud and group them by the prediction value and taking 
    # x_centroid, y_centroid and z_centroid mean values
    all_centroids = concatenated[concatenated['border_tree'] == 1].groupby(prediction).first()[['x_centroid', 'y_centroid', 'z_centroid']]

    print(f"Number of centroids: {len(all_centroids)}")

    # group  centroids by chunks by checking if the centroid is in the range of min and max values of x and y of the chunks
    # and put them into a dictionary
    centroids_by_chunk = defaultdict(list)

    for chunk, x_item, y_item in zip(chunks_list, x_min_max, y_min_max):
        for centroid in all_centroids.values:
            if x_item[0] <= centroid[0] <= x_item[1] and y_item[0] <= centroid[1] <= y_item[1]:
                centroids_by_chunk[(x_item, y_item)].append(centroid)
                # Remove the centroid from all_centroids to ensure it's not counted again
                all_centroids = all_centroids.drop(all_centroids[(all_centroids['x_centroid'] == centroid[0]) & (all_centroids['y_centroid'] == centroid[1])].index)


    print(f"Number of centroids: {len(centroids_by_chunk)}")
    # print(centroids_by_chunk)



    return concatenated


def main():
    parser = argparse.ArgumentParser(description="Merge point cloud chunks into a single point cloud.")
    parser.add_argument('-i', '--input_dir', type=str, required=True, help="Input directory containing chunked point cloud files.")
    parser.add_argument('-o', '--output_dir', type=str, required=True, help="Output directory to save merged point clouds.")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    grouped_chunks = defaultdict(list)
    for file_name in tqdm(os.listdir(args.input_dir)):
        input_file = os.path.join(args.input_dir, file_name)
        if input_file.endswith(('.laz', '.las')):
            df = las_to_pandas(input_file)
        elif input_file.endswith('.ply'):
            df = ply_to_pandas(input_file)
        else:
            print(f"Skipping unsupported file format: {input_file}")
            continue

        stem = "_".join(file_name.split("_")[:-1]).replace("_chunk", "")
        grouped_chunks[stem].append(df)

    for stem, chunks in grouped_chunks.items():
        merged_df = merge_pointclouds(stem, chunks)
        output_file = os.path.join(args.output_dir, f"{stem}.ply")
        pandas_to_ply(merged_df, csv_file_provided=False, output_file_path=output_file)


if __name__ == '__main__':
    main()
