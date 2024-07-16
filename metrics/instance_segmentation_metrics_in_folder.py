import csv
import glob
import os
import argparse
from joblib import Parallel, delayed
import laspy
import jaklas

from metrics.instance_segmentation_metrics import InstanceSegmentationMetrics
from metrics.attach_labels_to_las_file_pred2gt import AttachLabelsToLasFilePred2Gt

from nibio_inference.ply_to_pandas import ply_to_pandas
from nibio_inference.pandas_to_las import pandas_to_las

class InstanceSegmentationMetricsInFolder():
    GT_LABEL_NAME = 'treeID' # 'StemID' or 'treeID' 
    TARGET_LABEL_NAME = 'preds_instance_segmentation' # 'instance_nr'

    def __init__(
        self,
        gt_las_folder_path,
        target_las_folder_path,
        output_folder_path=None, # if None, output will be saved in this folder
        remove_ground=False,
        verbose=False
    ):
        self.gt_las_folder_path = gt_las_folder_path
        self.target_las_folder_path = target_las_folder_path
        self.output_folder_path = output_folder_path
        # create output folder if not exists
        if self.output_folder_path is not None:
            if not os.path.exists(self.output_folder_path):
                os.makedirs(self.output_folder_path)
        self.remove_ground = remove_ground
        self.verbose = verbose

    def main(self):
        # if there are ply files convert them to las files using ply_to_pandas and then pandas_to_las
        # get all ply files in the gt_las_folder_path
        gt_ply_file_paths = glob.glob(self.gt_las_folder_path + '/*.ply', recursive=False)

        # iterate over the ply files
        for gt_ply_file_path in gt_ply_file_paths:
            # read the ply file
            points_df = ply_to_pandas(gt_ply_file_path)

            # save pandas dataframe to csv file
            points_df.to_csv(gt_ply_file_path.replace('.ply', '.csv'), index=False)

            # save the ply file as las file using jaklas
            jaklas.write(points_df, gt_ply_file_path.replace('.ply', '.las'), point_format=2, scale=(0.001, 0.001, 0.001))

        # get all las files in the gt_las_folder_path
        gt_las_file_paths = glob.glob(self.gt_las_folder_path + '/*.las', recursive=False)
        # account also for laz files
        gt_las_file_paths += glob.glob(self.gt_las_folder_path + '/*.laz', recursive=False)
     
        gt_las_file_paths.sort()

        # get all ply files in the target_las_folder_path
        target_ply_file_paths = glob.glob(self.target_las_folder_path + '/*.ply', recursive=False)

        # iterate over the ply files
        for target_ply_file_path in target_ply_file_paths:
            # read the ply file
            points_df = ply_to_pandas(target_ply_file_path)
            # save the ply file as las file using jaklas
            jaklas.write(points_df, target_ply_file_path.replace('.ply', '.las'), point_format=2, scale=(0.001, 0.001, 0.001))

        # get all las files in the target_las_folder_path
        target_las_file_paths = glob.glob(self.target_las_folder_path + '/*.las', recursive=False)
        # account also for laz files
        target_las_file_paths += glob.glob(self.target_las_folder_path + '/*.laz', recursive=False)
        target_las_file_paths.sort()

        # check that the number of las files in the gt_las_folder_path and target_las_folder_path are the same
        if len(gt_las_file_paths) != len(target_las_file_paths):
            # print names of the folders
            print('gt_las_folder_path: ' + self.gt_las_folder_path)
            print('target_las_folder_path: ' + self.target_las_folder_path)
            print('Number of files in gt_las_folder_path: ' + str(len(gt_las_file_paths)))
            print('Number of files in target_las_folder_path: ' + str(len(target_las_file_paths)))
            raise Exception('The number of las files in the gt_las_folder_path and target_las_folder_path are not the same')

        # iterate over the las files
        for gt_las_file_path, target_las_file_path in zip(gt_las_file_paths, target_las_file_paths):

            # read the las file check if las file is not empty
            gt_las_file = laspy.read(gt_las_file_path)
            if len(gt_las_file.points) == 0:
                # remove the las file and the corresponding target_las_file_path from the list of paths
                gt_las_file_paths.remove(gt_las_file_path)
                target_las_file_paths.remove(target_las_file_path)
                if self.verbose:
                    print('Removed empty las file from the list: ' + gt_las_file_path)

            # check if las file is not empty
            target_las_file = laspy.read(target_las_file_path)
            if len(target_las_file.points) == 0:
                # remove the las file and the corresponding target_las_file_path from the list of paths
                gt_las_file_paths.remove(gt_las_file_path)
                target_las_file_paths.remove(target_las_file_path)
                if self.verbose:
                    print('Removed empty las file from the list: ' + target_las_file_path)

        # match the core name in the gt_las_file_path and target_las_file_path and make tuples of the matched paths
        matched_paths = []
        for gt_las_file_path, target_las_file_path in zip(gt_las_file_paths, target_las_file_paths):

            # print what files are being matched and processed
            if self.verbose:
                print('Matching: ' + gt_las_file_path + ' and ' + target_las_file_path)

            # get the core name of the gt_las_file_path
            gt_las_file_core_name = os.path.basename(gt_las_file_path).split('.')[0]
            # get the core name of the target_las_file_path
            target_las_file_core_name = os.path.basename(target_las_file_path).split('.')[0]

            # check that the core name of the gt_las_file_path and target_las_file_path are the same
            if gt_las_file_core_name in target_las_file_core_name:
                # make a tuple of the matched paths
                matched_paths.append((gt_las_file_path, target_las_file_path)) 

        # check if all are matched if not raise an exception

        print('matched_paths: ',  matched_paths)
        # check this by looking into matched_paths and the number of files in gt_las_file_paths and target_las_file_paths
        if len(matched_paths) != len(gt_las_file_paths) or len(matched_paths) != len(target_las_file_paths):
            # print names of the folders
            print('gt_las_folder_path: ' + self.gt_las_folder_path)
            print('target_las_folder_path: ' + self.target_las_folder_path)
            print('Number of files in gt_las_folder_path: ' + str(len(gt_las_file_paths)))
            print('Number of files in target_las_folder_path: ' + str(len(target_las_file_paths)))
            print('Number of matched files: ' + str(len(matched_paths)))
            raise Exception('Not all files are matched')

        # run the instance segmentation metrics for each matched las file
        metric_dict_list = []
        f1_scores_weighted_list = []

        parallel_output = Parallel(n_jobs=-1, verbose=0)(
            delayed(self.compute_metrics)(gt_las_file_path, target_las_file_path) for gt_las_file_path, target_las_file_path in matched_paths
        )

        # extract the metric_dict_list and f1_scores_weighted_list from the paralle_output
        for metric_dict_mean,  f1_score_weighted in parallel_output:
            metric_dict_list.append(metric_dict_mean) 
            f1_scores_weighted_list.append(f1_score_weighted)

        # this is serial version of the above code
        # for gt_las_file_path, target_las_file_path in matched_paths:
        #     metric_dict, f1_score_weighted = self.compute_metrics(gt_las_file_path, target_las_file_path)
        #     metric_dict_list.append(metric_dict)
        #     f1_scores_weighted_list.append(f1_score_weighted)

        # calculate the mean f1 score of weighted f1 scores
        # mean_f1_score = sum(f1_scores_weighted_list) / len(f1_scores_weighted_list)

        # use a dirty hack to compute the mean of the metrics (accounts for broken point clouds caused by phils code)

        mean_f1_score = sum(f1_scores_weighted_list) / len(gt_las_file_paths)
        # mean_f1_score = sum(f1_scores_weighted_list) / 16

        print('numer of files: ' + str(len(gt_las_file_paths)))


        # calculate the mean metrics for all the elements in the metric_dict_list
        # create a mean_metrics dictionary and initialize it with zeros
        mean_metrics = {}
        for metric_dict in metric_dict_list:
            for key, value in metric_dict.items():
                mean_metrics[key] = 0

        for metric_dict in metric_dict_list:
            for key, value in metric_dict.items():
                mean_metrics[key] += value 

        # devide the mean_metrics by the number of metric_dict_list
        for key, value in mean_metrics.items():
            mean_metrics[key] = value / len(metric_dict_list)
   
        if self.output_folder_path is not None:
            # create the output folder path
            save_to_csv_path = os.path.join(self.output_folder_path, 'summary_metrics_all_plots.csv')
            # save the mean metrics to a csv file
            with open(save_to_csv_path, 'w') as csv_file:
                writer = csv.writer(csv_file)
                for key, value in mean_metrics.items():
                    # round the value to 3 decimal places
                    value = round(value, 3)
                    writer.writerow([key, value])
                    
        if self.verbose:
            print('Mean F1 Score: {}'.format(mean_f1_score))
            # print the mean metrics
            print('Mean Metrics: {}'.format(mean_metrics))

        return mean_f1_score

    def compute_metrics(self, gt_las_file_path, target_las_file_path):
        # get the core name of the gt_las_file_path

        gt_las_file_core_name = os.path.basename(gt_las_file_path).split('.')[0]

    
        if self.output_folder_path is not None:
            # create the output folder path
            save_to_csv_path = os.path.join(self.output_folder_path, gt_las_file_core_name + '.csv')
            # attach the labels to the target las file

                  # attach labels to the las file
            # AttachLabelsToLasFilePred2Gt(
            #         gt_las_file_path,
            #         target_las_file_path,
            #         update_las_file_path = os.path.join(self.output_folder_path, gt_las_file_core_name + '.las'),
            #         gt_label_name=self.GT_LABEL_NAME,
            #         target_label_name=self.TARGET_LABEL_NAME,
            #         verbose=self.verbose
            #     ).main()

        else:
            save_to_csv_path = None

        # run the instance segmentation metrics
        instance_segmentation_metrics = InstanceSegmentationMetrics(
            gt_las_file_path,
            target_las_file_path,
            remove_ground=self.remove_ground,
            csv_file_name=save_to_csv_path,
            verbose=self.verbose
        )
        metric_dict, metric_dict_weighted_by_tree_hight, metric_dict_mean = instance_segmentation_metrics.main()
        f1_score_weighted = metric_dict_mean['f1_score']

        return metric_dict_mean, f1_score_weighted

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_las_folder_path', type=str, required=True)
    parser.add_argument('--target_las_folder_path', type=str, required=True)
    parser.add_argument('--output_folder_path', type=str, required=False, default=None)
    parser.add_argument('--remove_ground', action='store_true', help="Do not take into account the ground (class 0).", default=False)
    parser.add_argument('--verbose', action='store_true', help="Print information about the process")
    args = parser.parse_args()

    # run the instance segmentation metrics in folder
    instance_segmentation_metrics_in_folder = InstanceSegmentationMetricsInFolder(
        args.gt_las_folder_path,
        args.target_las_folder_path,
        args.output_folder_path,
        args.remove_ground,
        verbose=args.verbose
    )

    mean_f1_score = instance_segmentation_metrics_in_folder.main()