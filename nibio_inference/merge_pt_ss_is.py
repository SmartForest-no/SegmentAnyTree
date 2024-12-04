import argparse
import json
import sys
import jaklas
import laspy

import concurrent.futures
import numpy as np

import pandas as pd

import dask.dataframe as dd  # dask==2021.8.1

from nibio_inference.ply_to_pandas import ply_to_pandas
from nibio_inference.las_to_pandas import las_to_pandas
from nibio_inference.pandas_to_ply import pandas_to_ply
from nibio_inference.pandas_to_las import pandas_to_las


class MergePtSsIs(object):
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
            df['xyz_index'] = df['x'].astype(str) + "_" + df['y'].astype(str) + "_" + df['z'].astype(str)
            df.set_index('xyz_index', inplace=True)
            return df

        point_cloud_df = preprocess_data(self.point_cloud)
        semantic_segmentation_df = preprocess_data(self.semantic_segmentation)
        instance_segmentation_df = preprocess_data(self.instance_segmentation)

        # save to csv files for debugging
        # point_cloud_df.to_csv(self.point_cloud.replace('.ply', '.csv'))
        # semantic_segmentation_df.to_csv(self.semantic_segmentation.replace('.ply', '.csv'))
        # instance_segmentation_df.to_csv(self.instance_segmentation.replace('.ply', '.csv'))

        # Rename columns for semantic and instance segmentations
        semantic_segmentation_df.columns = [f'{col}_semantic_segmentation' for col in semantic_segmentation_df.columns]
        instance_segmentation_df.columns = [f'{col}_instance_segmentation' for col in instance_segmentation_df.columns]

        # Convert to Dask DataFrames for parallel processing
        point_cloud_dd = dd.from_pandas(point_cloud_df, npartitions=48)
        semantic_segmentation_dd = dd.from_pandas(semantic_segmentation_df, npartitions=48)
        instance_segmentation_dd = dd.from_pandas(instance_segmentation_df, npartitions=48)

        # Merge using Dask
        merged_dd = point_cloud_dd.join(semantic_segmentation_dd, how='outer')
        merged_dd = merged_dd.join(instance_segmentation_dd, how='outer')

        # Convert back to pandas DataFrame
        merged_df = merged_dd.compute()

        # remove the following columns from the merged data frame : x_instance_segmentation, y_instance_segmentation, z_instance_segmentation
        merged_df.drop(columns=['x_instance_segmentation', 'y_instance_segmentation', 'z_instance_segmentation'], inplace=True)

        # remove the following columns from the merged data frame : x_semantic_segmentation, y_semantic_segmentation, z_semantic_segmentation
        merged_df.drop(columns=['x_semantic_segmentation', 'y_semantic_segmentation', 'z_semantic_segmentation'], inplace=True)

        # remove the following colum 'gt_semantic_segmentation' from the merged data frame
        # merged_df.drop(columns=['gt_semantic_segmentation'], inplace=True)

        # where in column 'preds_instance_segmentation' there is no value, replace it with the value from column 'preds_semantic_segmentation'
        # merged_df['preds_instance_segmentation'] = merged_df['preds_instance_segmentation'].fillna(merged_df['preds_semantic_segmentation'])

        # rename column 'preds_semantic_segmentation' to 'predSemantic'
        merged_df.rename(columns={'preds_semantic_segmentation': 'PredSemantic'}, inplace=True)

        # rename column 'preds_instance_segmentation' to 'predInstance'
        merged_df.rename(columns={'preds_instance_segmentation': 'PredInstance'}, inplace=True)


        # Post-process merged data
        min_values_path = self.point_cloud.replace('.ply', '_min_values.json')
        with open(min_values_path, 'r') as f:
            min_values = json.load(f)
        
        min_x, min_y, min_z = min_values
        merged_df['x'] = merged_df['x'].astype(float) + min_x
        merged_df['y'] = merged_df['y'].astype(float) + min_y
        merged_df['z'] = merged_df['z'].astype(float) + min_z

        # add 1 to PredInstance column
        merged_df['PredInstance'] = merged_df['PredInstance'] + 1

        # Assign NaNs in PredInstance to 0. This is because the instance segmentation
        # may not have been able to assign an instance ID to every point, so after the
        # outer join, it is possible to have missing values in PredInstance. This causes
        # an issue when saving the data to a .las file, as we want to cast the column to
        # an unsigned integer type, which does not support NaNs.
        merged_df['PredInstance'] = merged_df['PredInstance'].fillna(0)

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
        
        pandas_to_las(
            merged_df,
            csv_file_provided=False,
            output_file_path=self.output_file_path,
            do_compress=True,
            verbose=self.verbose
            )
        
        # save the merged data frame to a file using jaklas as a .las file
        # jaklas.write(merged_df, self.output_file_path, point_format=3, scale=(0.001, 0.001, 0.001))

        # # convert to laz
        # las = laspy.read(self.output_file_path)
        # las.write(self.output_file_path.replace('.las', '.laz'), do_compress=True)


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
    merge_pt_ss_is = MergePtSsIs(
        point_cloud=POINT_CLOUD,
        semantic_segmentation=SEMANTIC_SEGMENTATION,
        instance_segmentation=INSTANCE_SEGMENTATION,
        output_file_path=OUTPUT_FILE_PATH,
        verbose=VERBOSE
        )()
    


    

    


