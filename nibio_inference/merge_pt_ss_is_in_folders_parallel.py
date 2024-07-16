import argparse
import glob
import os
import sys
from typing import Any

from tqdm import tqdm
from joblib import Parallel, delayed

from nibio_inference.merge_pt_ss_is import MergePtSsIs


class MergePtSsIsInFolders(object):
    def __init__(self, 
                 input_data_folder_path,
                 segmented_data_folder_path,
                 output_data_folder_path,
                 verbose=False
                    ):
        self.input_data_folder_path = input_data_folder_path
        self.segmented_data_folder_path = segmented_data_folder_path
        self.output_data_folder_path = output_data_folder_path
        self.verbose = verbose

        # create the output_data_folder_path if it does not exist
        if not os.path.exists(self.output_data_folder_path):
            os.makedirs(self.output_data_folder_path)

    def merge_file(self, item):
        MergePtSsIs(
            point_cloud=item[0],
            semantic_segmentation=item[1],
            instance_segmentation=item[2],
            output_file_path=os.path.join(self.output_data_folder_path, os.path.basename(item[0]).split('.')[0] + '.laz'),
            verbose=self.verbose
        )()

    def merge(self):
        # find all the ply files in the input_data_folder_path
        input_data_files = glob.glob(os.path.join(self.input_data_folder_path, '*.ply'))

        # find all the ply files in the segmented_data_folder_path
        segmented_data_files = glob.glob(os.path.join(self.segmented_data_folder_path, '*.ply'))

        # combine the input_data_folder_path and the core names of the files in the input_data_folder_path
        input_data = {os.path.basename(file).split('.')[0]: file for file in input_data_files}

        # find all the _semantic_segmentation.ply files in the segmented_data_folder_path
        semantic_segmentation_files = [file for file in segmented_data_files if '_semantic_segmentation.ply' in file]

        # combine the segmented_data_folder_path and the core names of the files in the segmented_data_folder_path and remove the _semantic_segmentation.ply suffix
        semantic_segmentation = {os.path.basename(file).split('.')[0].replace('_semantic_segmentation', ''): file for file in semantic_segmentation_files}

        # find all the _instance_segmentation.ply files in the segmented_data_folder_path
        instance_segmentation_files = [file for file in segmented_data_files if '_instance_segmentation.ply' in file]

        # combine the segmented_data_folder_path and the core names of the files in the segmented_data_folder_path
        instance_segmentation = {os.path.basename(file).split('.')[0].replace('_instance_segmentation', ''): file for file in instance_segmentation_files}

        list_of_matched_tupeles = []
        for item in input_data.keys(): 
            if item in semantic_segmentation.keys() and item in instance_segmentation.keys():
                list_of_matched_tupeles.append((input_data[item], semantic_segmentation[item], instance_segmentation[item]))

        # Use joblib's Parallel and delayed to run in parallel
        Parallel(n_jobs=-1)(delayed(self.merge_file)(item) for item in tqdm(list_of_matched_tupeles))

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        if self.verbose:
            print('Merging point cloud, semantic segmentation and instance segmentation.')
        return self.merge()


if __name__ == '__main__':
    # use argparse to get the input and output file paths
    parser = argparse.ArgumentParser(description='Merge point cloud, semantic segmentation and instance segmentation.')
    parser.add_argument('-i', '--input_data_folder_path', type=str, help='Path to the input data folder.')
    parser.add_argument('-s', '--segmented_data_folder_path', type=str, help='Path to the segmented data folder.')
    parser.add_argument('-o', '--output_data_folder_path', type=str, help='Path to the output data folder.')
    parser.add_argument('-v', '--verbose', action='store_true', help='Print verbose output.')

    # generate help message if no arguments are provided
    if len(sys.argv)==1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = vars(parser.parse_args())

    # get the arguments
    INPUT_DATA_FOLDER_PATH = args['input_data_folder_path']
    SEGMENTED_DATA_FOLDER_PATH = args['segmented_data_folder_path']
    OUTPUT_DATA_FOLDER_PATH = args['output_data_folder_path']
    VERBOSE = args['verbose']

    # run the merge
    merge_pt_ss_is_in_folders = MergePtSsIsInFolders(
        input_data_folder_path=INPUT_DATA_FOLDER_PATH,
        segmented_data_folder_path=SEGMENTED_DATA_FOLDER_PATH,
        output_data_folder_path=OUTPUT_DATA_FOLDER_PATH,
        verbose=VERBOSE
    )()
