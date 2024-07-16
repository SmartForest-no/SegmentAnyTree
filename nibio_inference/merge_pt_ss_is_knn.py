import argparse
import json
import sys
import jaklas
import laspy

import concurrent.futures
import numpy as np

from scipy.spatial import KDTree

import pandas as pd

import dask.dataframe as dd  # dask==2021.8.1

from nibio_inference.ply_to_pandas import ply_to_pandas
from nibio_inference.las_to_pandas import las_to_pandas
from nibio_inference.pandas_to_ply import pandas_to_ply
from nibio_inference.pandas_to_las import pandas_to_las


class MergePtSsIsKnn(object):
    def __init__(self, 
                 point_cloud, 
                 semantic_segmentation, 
                 instance_segmentation, 
                 output_file_path,
                 verbose=False
                 ):
        
        self.point_cloud = point_cloud
        self.semantic_segmentation = semantic_segmentation
        self.instance_segmentation = instance_segmentation
        self.output_file_path = output_file_path
        self.verbose = verbose


    def parallel_join(self, df1_chunk, df2, on_columns, how_type):
        return df1_chunk.merge(df2, on=on_columns, how=how_type)

    def main_parallel_join(self, df1, df2, on_columns=['x', 'y', 'z'], how_type='outer', n_workers=4):
        # Split df1 into chunks
        chunks = np.array_split(df1, n_workers)
        
        results = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
            for chunk in chunks:
                results.append(executor.submit(self.parallel_join, chunk, df2, on_columns, how_type))
        
        # Concatenate results
        return pd.concat([r.result() for r in results])


    def merge(self):

        if self.verbose:
            print('Merging point cloud, semantic segmentation and instance segmentation.')

       # Read and preprocess data
        def preprocess_data(file_path):
            df = ply_to_pandas(file_path)
            df.rename(columns={'X': 'x', 'Y': 'y', 'Z': 'z'}, inplace=True)
            df.sort_values(by=['x', 'y', 'z'], inplace=True)
            return df

        point_cloud_df = preprocess_data(self.point_cloud)
        semantic_segmentation_df = preprocess_data(self.semantic_segmentation)
        instance_segmentation_df = preprocess_data(self.instance_segmentation)

        # Create a KDTree using the points from df_semantic_segmentation
        tree_semantic = KDTree(semantic_segmentation_df[['x', 'y', 'z']])

        # For each point in df_point_cloud, find the closest point in df_semantic_segmentation
        _, indices_semantic = tree_semantic.query(point_cloud_df[['x', 'y', 'z']])

        # Fetch the corresponding rows from df_semantic_segmentation
        df_semantic_segmentation_matched = semantic_segmentation_df.iloc[indices_semantic]

        # Reset index for the new dataframe
        df_semantic_segmentation_matched.reset_index(drop=True, inplace=True)

        # save the matched semantic segmentation to a file for debugging with an appropriate name
        df_semantic_segmentation_matched.to_csv(self.output_file_path.replace('.laz', '_semantic_segmentation_matched.csv'), index=False)

        # do for instance segmentation
        # Create a KDTree using the points from df_instance_segmentation
        tree_instance = KDTree(instance_segmentation_df[['x', 'y', 'z']])

        # For each point in df_point_cloud, find the closest point in df_instance_segmentation
        _, indices_instance = tree_instance.query(point_cloud_df[['x', 'y', 'z']])

        # Fetch the corresponding rows from df_instance_segmentation
        df_instance_segmentation_matched = instance_segmentation_df.iloc[indices_instance]

        # Reset index for the new dataframe
        df_instance_segmentation_matched.reset_index(drop=True, inplace=True)

        # save the matched instance segmentation to a file for debugging with an appropriate name
        df_instance_segmentation_matched.to_csv(self.output_file_path.replace('.laz', '_instance_segmentation_matched.csv'), index=False)

        # Rename columns for semantic and instance segmentations
        df_semantic_segmentation_matched.columns = [f'{col}_semantic_segmentation' for col in semantic_segmentation_df.columns]
        df_instance_segmentation_matched.columns = [f'{col}_instance_segmentation' for col in instance_segmentation_df.columns]

        # Merge point cloud
        # create and empty data frame
        merged_df = pd.DataFrame()
        # add all the columns from point_cloud_df to the merged data frame
        merged_df = pd.concat([merged_df, point_cloud_df], axis=1)

        # add all the columns from df_semantic_segmentation_matched to the merged data frame except the x, y and z columns
        merged_df = pd.concat([merged_df, df_semantic_segmentation_matched.drop(columns=['x', 'y', 'z'])], axis=1)

        # # add all the columns from df_instance_segmentation_matched to the merged data frame except the x, y and z columns
        # merged_df = pd.concat([merged_df, df_instance_segmentation_matched.drop(columns=['x', 'y', 'z'])], axis=1)


        # remove the following columns from the merged data frame : x_instance_segmentation, y_instance_segmentation, z_instance_segmentation
        merged_df.drop(columns=['x_instance_segmentation', 'y_instance_segmentation', 'z_instance_segmentation'], inplace=True)

        # remove the following columns from the merged data frame : x_semantic_segmentation, y_semantic_segmentation, z_semantic_segmentation
        merged_df.drop(columns=['x_semantic_segmentation', 'y_semantic_segmentation', 'z_semantic_segmentation'], inplace=True)

        # Post-process merged data
        min_values_path = self.point_cloud.replace('.ply', '_min_values.json')
        with open(min_values_path, 'r') as f:
            min_values = json.load(f)
        
        min_x, min_y, min_z = min_values
        merged_df['x'] = merged_df['x'].astype(float) + min_x
        merged_df['y'] = merged_df['y'].astype(float) + min_y
        merged_df['z'] = merged_df['z'].astype(float) + min_z
        return merged_df
    
    def save(self, merged_df):
        if 'return_num' in merged_df:
            if self.verbose:
                print('Clipping return_num to 7 for file: {}'.format(self.output_file_path))
            merged_df['return_num'] = merged_df['return_num'].clip(upper=7)

            if self.verbose:
                print('Clipping done for return_num.')


        if 'num_returns' in merged_df:
            if self.verbose:
                print('Clipping num_returns to 7 for file: {}'.format(self.output_file_path))
            merged_df['num_returns'] = merged_df['num_returns'].clip(upper=7)

            if self.verbose:
                print('Clipping done for num_returns.')

        # cols_with_value = []

        # for col in merged_df.columns:
        #     if 8.0 in merged_df[col].unique():
        #         cols_with_value.append(col)

        # print("Columns containing the value 8.0:", cols_with_value)


        # save the merged data frame to a file
        # merged_df.to_csv(self.output_file_path, index=False)

        # save the merged data frame to a file
        # pandas_to_ply(
        #     merged_df,
        #     csv_file_provided=False,
        #     output_file_path=self.output_file_path
        #     )
        
        # pandas_to_las(
        #     merged_df,
        #     csv_file_provided=False,
        #     output_file_path=self.output_file_path
        #     )
        
    

        # save the merged data frame to a file using jaklas as a .las file
        jaklas.write(merged_df, self.output_file_path, point_format=2, scale=(0.001, 0.001, 0.001))

        # convert to laz
        las = laspy.read(self.output_file_path)
        las.write(self.output_file_path.replace('.las', '.laz'), do_compress=True)


    def run(self):
        if self.verbose:
            print('point_cloud: {}'.format(self.point_cloud))
            print('semantic_segmentation: {}'.format(self.semantic_segmentation))
            print('instance_segmentation: {}'.format(self.instance_segmentation))
            
        merged_df = self.merge()
        if self.output_file_path is not None:
            self.save(merged_df)

        if self.verbose:
            print('Done for:')
            print('output_file_path: {}'.format(self.output_file_path))
        
        return merged_df
    
    
    def __call__(self):
        return self.run()
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Merge point cloud, semantic segmentation and instance segmentation.')
    parser.add_argument('-pc', '--point_cloud', help='Path to the point cloud file.')
    parser.add_argument('-ss', '--semantic_segmentation', help='Path to the semantic segmentation file.')
    parser.add_argument('-is', '--instance_segmentation', help='Path to the instance segmentation file.')
    parser.add_argument('-o', '--output_file_path', help='Path to the output file.')
    parser.add_argument('-v', '--verbose', action='store_true', help='Print verbose output.')
    
    # generate help message if no arguments are provided
    if len(sys.argv)==1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = vars(parser.parse_args())

    # get the arguments
    POINT_CLOUD = args['point_cloud']
    SEMANTIC_SEGMENTATION = args['semantic_segmentation']
    INSTANCE_SEGMENTATION = args['instance_segmentation']
    OUTPUT_FILE_PATH = args['output_file_path']
    VERBOSE = args['verbose']



    # run the merge
    merge_pt_ss_is = MergePtSsIsKnn(
        point_cloud=POINT_CLOUD,
        semantic_segmentation=SEMANTIC_SEGMENTATION,
        instance_segmentation=INSTANCE_SEGMENTATION,
        output_file_path=OUTPUT_FILE_PATH,
        verbose=VERBOSE
        )()
    


    

    


