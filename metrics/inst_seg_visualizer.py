import argparse
import glob
import os
import laspy
import numpy as np
import pandas as pd
from tqdm import tqdm

class InstSegVisualizer:
    GT_LABEL_NAME = 'treeID'
    TARGET_LABEL_NAME = 'preds_instance_segmentation'
    CSV_GT_LABEL_NAME = 'gt_label(dominant_label)'
    CSV_TARGET_LABEL_NAME = 'pred_label'

    def __init__(self, folder_with_metrics, folder_with_point_pt, verbose) -> None:
        self.folder_with_metrics = folder_with_metrics
        self.folder_with_point_pt = folder_with_point_pt
        self.verbose = verbose

    def get_las_file_paths(self, folder_path):
        las_file_paths = glob.glob(folder_path + '/*.las', recursive=False)
        # account also for .laz files
        las_file_paths += glob.glob(folder_path + '/*.laz', recursive=False)
        las_file_paths.sort()
        return las_file_paths

    def get_csv_file_paths(self, folder_path):
        csv_file_paths = glob.glob(folder_path + '/*.csv', recursive=False)
        csv_file_paths.sort()
        return csv_file_paths

    def match_las_and_csv_files(self, las_file_paths, csv_file_paths):
        matched_paths = []
        for las_file_path, csv_file_path in zip(las_file_paths, csv_file_paths):
            las_file_name = las_file_path.split('/')[-1]
            csv_file_name = csv_file_path.split('/')[-1]
            if las_file_name.split('.')[0] == csv_file_name.split('.')[0]:
                matched_paths.append((las_file_path, csv_file_path))
            else:
                raise Exception('The las file name and the csv file name do not match')
        return matched_paths

    def get_gt_and_pred_labels_from_csv_for_single_file(self, csv_file_path):
        df = pd.read_csv(csv_file_path)
        gt_labels = df[self.CSV_GT_LABEL_NAME].values
        pred_labels = df[self.CSV_TARGET_LABEL_NAME].values
        # map to int
        gt_labels = gt_labels.astype(int)
        pred_labels = pred_labels.astype(int)
        # create a list of tuples
        matched_labels = list(zip(gt_labels, pred_labels))
        return matched_labels

    def extract_overlapping_pc_in_single_file(self, las_file_path, matched_labels):
        # get the las file name
        las_file_name = las_file_path.split('/')[-1]

        # create if does not exist the folder with metrics named after the las file
        if not os.path.exists(self.folder_with_metrics + '/' + las_file_name.split('.')[0]):
            os.makedirs(self.folder_with_metrics + '/' + las_file_name.split('.')[0])

        for gt_label, pred_label in tqdm(matched_labels):
             # read the las file
            las_file = laspy.read(las_file_path)

            # get GT_LABEL_NAME points
            gt_points = las_file.points[las_file.points[self.GT_LABEL_NAME] == gt_label]
            # create a new array of gt points with the same shape as the red array and fill it with 255
            gt_points['red'] = np.ones(gt_points['red'].shape) * 255
            gt_points['green'] = np.ones(gt_points['green'].shape) * 0
            gt_points['blue'] = np.ones(gt_points['blue'].shape) * 0

            # get TARGET_LABEL_NAME points
            pred_points = las_file.points[las_file.points[self.TARGET_LABEL_NAME] == pred_label]

            # create a new array of pred points with the same shape as the blue array and fill it with 255
            pred_points['red'] = np.ones(pred_points['red'].shape) * 0
            pred_points['green'] = np.ones(pred_points['green'].shape) * 0
            pred_points['blue'] = np.ones(pred_points['blue'].shape) * 255
 
            # dump both arrays to las to the folder with metrics named after the las file
            las_file.points = gt_points
            las_file.write(self.folder_with_metrics + '/' + las_file_name.split('.')[0] + '/' + las_file_name.split('.')[0] + '_gt_' + str(gt_label) + '.las')
            las_file.points = pred_points
            las_file.write(self.folder_with_metrics + '/' + las_file_name.split('.')[0] + '/' + las_file_name.split('.')[0] + '_pred_' + str(pred_label) + '.las')

            # merge  using pdal and os.system and save the merged file in the folder with metrics named after the las file
            # os.system(
            #     'pdal merge ' + self.folder_with_metrics + '/' + 
            #     las_file_name.split('.')[0] + '/' + las_file_name.split('.')[0] + 
            #     '_gt_' + str(gt_label) + '.las ' + self.folder_with_metrics + '/' + 
            #     las_file_name.split('.')[0] + '/' + las_file_name.split('.')[0] + 
            #     '_pred_' + str(pred_label) + '.las ' + self.folder_with_metrics + '/' + 
            #     las_file_name.split('.')[0] + '/' + las_file_name.split('.')[0] + 
            #     '_gt_' + str(gt_label) + '_pred_' + str(pred_label) + '.las'
            #     )

    def extract_overlapping_pc_of_all_files(self, matched_files):
        for las_file_path, csv_file_path in matched_files:
            if self.verbose:
                print('Processing file: ', las_file_path)
            matched_labels = self.get_gt_and_pred_labels_from_csv_for_single_file(csv_file_path)
            self.extract_overlapping_pc_in_single_file(las_file_path, matched_labels)

    def main(self):
        las_file_paths = self.get_las_file_paths(self.folder_with_point_pt)
        csv_file_paths = self.get_csv_file_paths(self.folder_with_metrics)
        matched_files = self.match_las_and_csv_files(las_file_paths, csv_file_paths)
        self.extract_overlapping_pc_of_all_files(matched_files)

        if self.verbose:
            # print the number of las files
            print('Number of las files: ', len(las_file_paths))
            # print where the files were saved
            print('The files were saved in: ', self.folder_with_metrics)
      
if __name__ == '__main__':
    # use argparse to get the folder with metrics
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_with_metrics', type=str, default='metrics')
    parser.add_argument('--folder_with_point_pt', type=str, default='point_cloud')
    parser.add_argument('--verbose', help="Print more information.", action="store_true")
    args = parser.parse_args()
   
    InstSegVisualizer(
        args.folder_with_metrics, 
        args.folder_with_point_pt,
        args.verbose).main()
 
