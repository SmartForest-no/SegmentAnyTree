import argparse
import json
import os
from joblib import Parallel, delayed
import laspy
from matplotlib import pyplot as plt
import numpy as np
from sklearn.neighbors import KDTree
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score, precision_score, recall_score


class MetricSemSeg:
    """
    This function returns the metrics for semantic segmentation. It takes in the ground truth folder and the predicted folder.
    It returns the following metrics:
    1. f1 score
    2. precision score
    3. recall score
    4. confusion matrix

    The input point clouds should be in the las format.
    """
    def __init__(
        self, 
        gt_folder,
        pred_folder,
        adjust_gt_data=False,
        plot_confusion_matrix=False, 
        verbose=False
        ):
        self.gt_folder = gt_folder
        self.pred_folder = pred_folder
        self.adjust_gt_data = adjust_gt_data # if True, the ground truth data will be adjusted to match the predicted data
        # this means removing data from 0 class which are considered as not important and should not be considered in the metrics
        self.plot_confusion_matrix = plot_confusion_matrix
        self.verbose = verbose

    def get_file_name_list(self):

        # get list of the original point clouds
        file_name_list_original = []
        for file in os.listdir(self.gt_folder):
            if file.endswith(".las"):
                file_name_list_original.append(os.path.join(self.gt_folder, file))

        # sort the list in place
        file_name_list_original.sort()

        # get the files which have the same core name as the original point clouds
        file_name_list_original_core = []
        for file_name in file_name_list_original:
            file_name_list_original_core.append(os.path.basename(file_name).split('.')[0])

        # get list of the predicted point clouds
        file_name_list_predicted = []
        for file in os.listdir(self.pred_folder):
            # get files which have the same core name as the original point clouds and suffix of '.segmented'
            if os.path.basename(file).split('.')[0] in file_name_list_original_core and os.path.basename(file).split('.')[1] == 'segmented':
                print("processing file {}".format(file))
                # check is the file is a las file
                if file.endswith(".las"):
                    file_name_list_predicted.append(os.path.join(self.pred_folder, file))
                # check if the file is a ply file and there is no las file with the same core name
                elif file.endswith(".ply") and (os.path.basename(file).split('.')[0] + '.segmented.las' not in os.listdir(self.pred_folder)):
                    # if not convert it to las file
                    file_name = os.path.join(self.pred_folder, file)
                    file_name_las = os.path.basename(file).split('.')[0] + '.segmented.las'
                    file_name_las = os.path.join(self.pred_folder, file_name_las)
                    if self.verbose:
                        print("converting {} to las file".format(file))
                    os.system("pdal translate {} {} --writers.las.dataformat_id=3 --writers.las.extra_dims=all".format(file_name, file_name_las))
                    file_name_list_predicted.append(file_name_las)

        # remove double files
        file_name_list_predicted = list(set(file_name_list_predicted))

        # sort the list in place
        file_name_list_predicted.sort()

        # check if the number of files in the two folders is the same
        if len(file_name_list_original) != len(file_name_list_predicted):
            raise ValueError("The two folders do not have the same number of files")

        # zip the two lists
        file_name_list = list(zip(file_name_list_original, file_name_list_predicted))

        return file_name_list

    def get_labels_from_point_file(self, file_name, predicted=False):
        """
        This function returns the labels of a point cloud file.
        """
        point_cloud = laspy.read(file_name)
        if self.adjust_gt_data and not predicted:
            if self.verbose:
                print("adjusting ground truth data")
            labels = point_cloud.label
            labels = labels[labels != 0]
            labels = labels - 1
        else:
            # if min label is not 0, then subtract 1 from all labels
            if np.min(point_cloud.label) != 0:
                labels = point_cloud.label - 1
            else:
                labels = point_cloud.label

        # convert to int
        labels = labels.astype(int)

        return labels

    def get_xyz_from_point_file(self, file_name, predicted=False):
        """
        This function returns the xyz coordinates of a point cloud file.
        """
        point_cloud = laspy.read(file_name)
        if self.adjust_gt_data and not predicted:
            if self.verbose:
                print("adjusting ground truth data")
            # find all the points xyz with label 0
            labels = point_cloud.label
            xyz = np.vstack((
                point_cloud.x[labels !=0], 
                point_cloud.y[labels !=0], 
                point_cloud.z[labels !=0]
                )).transpose()

        else:
            xyz = np.vstack((point_cloud.x, point_cloud.y, point_cloud.z)).transpose()

        return xyz

    def get_metrics_for_single_file(self, file_name_original, file_name_predicted):
        """
        This function returns the confusion matrix for a single point cloud and also the precision, recall and f1 score.
        """

        # get labels
        labels_predicted = self.get_labels_from_point_file(file_name_predicted, predicted=True)
        labels_original = self.get_labels_from_point_file(file_name_original)

        # find and print the ranges of the labels
        if self.verbose:
            print("labels_predicted range: ", np.min(labels_predicted), np.max(labels_predicted))
            print("labels_original range: ", np.min(labels_original), np.max(labels_original))

        # print shape of labels
        if self.verbose:
            print("labels_predicted.shape: ", labels_predicted.shape)
            print("labels_original.shape: ", labels_original.shape)


        # get points
        xyz_predicted = self.get_xyz_from_point_file(file_name_predicted, predicted=True)
        xyz_original = self.get_xyz_from_point_file(file_name_original)


        # find the closest point in the original point cloud for each point in the predicted point cloud using the euclidean distance using knn

        # nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(xyz_original)
        # distances, indices = nbrs.kneighbors(xyz_predicted)

        tree = KDTree(xyz_original, leaf_size=50, metric='euclidean')       
        # query the tree for Y     
        indices = tree.query(xyz_predicted, k=1, return_distance=False)   

        # get the labels of the closest points
        labels_original_closest = labels_original[indices]

        # get the confusion matrix
        conf_matrix = np.round(confusion_matrix(labels_original_closest, labels_predicted, normalize='true'), decimals=2)

        # if conf_matrix.shape[0] == 3 add diagonal elements to make it 4x4 at dimension 2
        if conf_matrix.shape[0] == 3:
            if self.verbose:
                print("conf_matrix.shape[0] == 3 expanding to 4x4 at dimension 2")
            conf_matrix = np.insert(conf_matrix, 2, 0, axis=1)
            conf_matrix = np.insert(conf_matrix, 2, 0, axis=0)

        # print the confusion matrix shape
        if self.verbose:
            print("conf_matrix.shape: ", conf_matrix.shape)

        # get the class names
        if conf_matrix.shape[0] == 3:
            class_names = ['terrain', 'vegetation', 'stem']
        elif conf_matrix.shape[0] == 4:
            class_names = ['terrain', 'vegetation', 'CWD', 'stem']

        # get picture of the confusion matrix
        if self.plot_confusion_matrix:
 
            disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=class_names)
            disp.plot()
            plt.savefig(file_name_original + '_confusion_matrix.png')

        # compute precision, recall and f1 score using sklearn.metrics 
        precision = precision_score(labels_original_closest, labels_predicted, average='weighted')
        precision = np.round(precision, decimals=3)
        recall = recall_score(labels_original_closest, labels_predicted, average='weighted')
        recall = np.round(recall, decimals=3)
        f1 = f1_score(labels_original_closest, labels_predicted, average='weighted')
        f1 = np.round(f1, decimals=3)

        # compute precision, recall and f1 per class per class_name
        precision_per_class = {}
        recall_per_class = {}
        f1_per_class = {}
        for name in class_names:
            precision_per_class[name] = precision_score(labels_original_closest, labels_predicted, labels=[class_names.index(name)], average='weighted')
            precision_per_class[name] = np.round(precision_per_class[name], decimals=3)
            recall_per_class[name] = recall_score(labels_original_closest, labels_predicted, labels=[class_names.index(name)], average='weighted')
            recall_per_class[name] = np.round(recall_per_class[name], decimals=3)
            f1_per_class[name] = f1_score(labels_original_closest, labels_predicted, labels=[class_names.index(name)], average='weighted')
            f1_per_class[name] = np.round(f1_per_class[name], decimals=3)
        
        # put all the results in a dictionary
        results = {
            'confusion_matrix': conf_matrix,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'precision_per_class': precision_per_class,
            'recall_per_class': recall_per_class,
            'f1_per_class': f1_per_class
        }

        return results

    def get_metrics_for_all_files(self):
        """
        This function returns the confusion matrix for all point clouds and also the precision, recall and f1 score.
        """
        file_name_list = self.get_file_name_list()

        # create empty lists for the results
        conf_matrix_list = []
        precision_list = []
        recall_list = []
        f1_list = []
        precision_per_class_list = []
        recall_per_class_list = []
        f1_per_class_list = []

        # loop over all files
        # results = []
        # for file_name_original, file_name_predicted in tqdm(file_name_list):
        #     if self.verbose:
        #         print("file_name_original: ", file_name_original)
        #         print("file_name_predicted: ", file_name_predicted)

        #     results.append(self.get_metrics_for_single_file(file_name_original, file_name_predicted))

        # parallelize the computation
        results = Parallel(n_jobs=-1, verbose=0)(
            delayed(self.get_metrics_for_single_file)(file_name_original, file_name_predicted) for file_name_original, file_name_predicted in file_name_list
        )

        # extract the results from the dictionary
        for results in results:
            conf_matrix_list.append(results['confusion_matrix'])
            precision_list.append(results['precision'])
            recall_list.append(results['recall'])
            f1_list.append(results['f1'])
            precision_per_class_list.append(results['precision_per_class'])
            recall_per_class_list.append(results['recall_per_class'])
            f1_per_class_list.append(results['f1_per_class'])

        # compute the mean of the confusion matrix
        conf_matrix_mean = np.mean(conf_matrix_list, axis=0)

        # two decimal places
        conf_matrix_mean = np.round(conf_matrix_mean, decimals=2)

        if conf_matrix_mean.shape[0] == 3:
            class_names = ['terrain', 'vegetation', 'stem']
        elif conf_matrix_mean.shape[0] == 4:
            class_names = ['terrain', 'vegetation', 'CWD', 'stem']

        # save the confusion matrix
        if self.plot_confusion_matrix:

            disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_mean, display_labels=class_names)
            disp.plot()
            plt.savefig('confusion_matrix_mean.png')

        # compute the mean of the precision, recall and f1 score
        precision_mean = np.mean(precision_list)
        recall_mean = np.mean(recall_list)
        f1_mean = np.mean(f1_list)

        # two decimal places
        precision_mean = np.round(precision_mean, decimals=2)
        recall_mean = np.round(recall_mean, decimals=2)
        f1_mean = np.round(f1_mean, decimals=2)

        # compute the mean of the precision, recall and f1 score per class
        # compute separately for items which have 3 and 4 classes
        # create a separate list for item with 3 classes defined class_names
        precision_per_class_list_3 = {}
        recall_per_class_list_3 = {}
        f1_per_class_list_3 = {}
        for name in class_names:
            precision_per_class_list_3[name] = []
            recall_per_class_list_3[name] = []
            f1_per_class_list_3[name] = []

        # create a separate list for item with 4 classes
        precision_per_class_list_4 = {}
        recall_per_class_list_4 = {}
        f1_per_class_list_4 = {}
        for name in class_names:
            precision_per_class_list_4[name] = []
            recall_per_class_list_4[name] = []
            f1_per_class_list_4[name] = []

        # loop over all items
        for precision_per_class, recall_per_class, f1_per_class in zip(precision_per_class_list, recall_per_class_list, f1_per_class_list):
            # check if the item has 3 or 4 classes
            if len(precision_per_class) == 3:
                for name in class_names:
                    precision_per_class_list_3[name].append(precision_per_class[name])
                    recall_per_class_list_3[name].append(recall_per_class[name])
                    f1_per_class_list_3[name].append(f1_per_class[name])
            elif len(precision_per_class) == 4:
                for name in class_names:
                    precision_per_class_list_4[name].append(precision_per_class[name])
                    recall_per_class_list_4[name].append(recall_per_class[name])
                    f1_per_class_list_4[name].append(f1_per_class[name])
            else:
                raise ValueError("The number of classes is not 3 or 4.")

        # compute the mean of the precision, recall and f1 score per class for items with 3 classes
        precision_per_class_list_mean_3 = {} 
        recall_per_class_list_mean_3 = {}
        f1_per_class_list_mean_3 = {}
        for name in class_names:
            precision_per_class_list_mean_3[name] = np.mean(precision_per_class_list_3[name])
            recall_per_class_list_mean_3[name] = np.mean(recall_per_class_list_3[name])
            f1_per_class_list_mean_3[name] = np.mean(f1_per_class_list_3[name])

        # compute the mean of the precision, recall and f1 score per class for items with 4 classes
        precision_per_class_list_mean_4 = {}
        recall_per_class_list_mean_4 = {}
        f1_per_class_list_mean_4 = {}
        for name in class_names:
            precision_per_class_list_mean_4[name] = np.mean(precision_per_class_list_4[name])
            recall_per_class_list_mean_4[name] = np.mean(recall_per_class_list_4[name])
            f1_per_class_list_mean_4[name] = np.mean(f1_per_class_list_4[name])

        # compute the mean of the precision, recall and f1 score per class
        precision_per_class_mean = {}
        recall_per_class_mean = {}
        f1_per_class_mean = {}

        # check if nan values are present in the list of mean values, if yes, replace them with 0
        for name in class_names:
            if np.isnan(precision_per_class_list_mean_3[name]):
                precision_per_class_list_mean_3[name] = 0
            if np.isnan(recall_per_class_list_mean_3[name]):
                recall_per_class_list_mean_3[name] = 0
            if np.isnan(f1_per_class_list_mean_3[name]):
                f1_per_class_list_mean_3[name] = 0
            if np.isnan(precision_per_class_list_mean_4[name]):
                precision_per_class_list_mean_4[name] = 0
            if np.isnan(recall_per_class_list_mean_4[name]):
                recall_per_class_list_mean_4[name] = 0
            if np.isnan(f1_per_class_list_mean_4[name]):
                f1_per_class_list_mean_4[name] = 0

        for name in class_names:
            # if both items are different from 0, compute the mean
            if precision_per_class_list_mean_3[name] != 0 and precision_per_class_list_mean_4[name] != 0:
                precision_per_class_mean[name] = (precision_per_class_list_mean_3[name] + precision_per_class_list_mean_4[name]) / 2
            # if one of the items is 0, take the other one
            elif precision_per_class_list_mean_3[name] == 0:
                precision_per_class_mean[name] = precision_per_class_list_mean_4[name]
            elif precision_per_class_list_mean_4[name] == 0:
                precision_per_class_mean[name] = precision_per_class_list_mean_3[name]
            # if both items are 0, set the mean to 0
            elif precision_per_class_list_mean_3[name] == 0 and precision_per_class_list_mean_4[name] == 0:
                precision_per_class_mean[name] = 0

            # if both items are different from 0, compute the mean
            if recall_per_class_list_mean_3[name] != 0 and recall_per_class_list_mean_4[name] != 0:
                recall_per_class_mean[name] = (recall_per_class_list_mean_3[name] + recall_per_class_list_mean_4[name]) / 2
            # if one of the items is 0, take the other one
            elif recall_per_class_list_mean_3[name] == 0:
                recall_per_class_mean[name] = recall_per_class_list_mean_4[name]
            elif recall_per_class_list_mean_4[name] == 0:
                recall_per_class_mean[name] = recall_per_class_list_mean_3[name]
            # if both items are 0, set the mean to 0
            elif recall_per_class_list_mean_3[name] == 0 and recall_per_class_list_mean_4[name] == 0:
                recall_per_class_mean[name] = 0

            # if both items are different from 0, compute the mean
            if f1_per_class_list_mean_3[name] != 0 and f1_per_class_list_mean_4[name] != 0:
                f1_per_class_mean[name] = (f1_per_class_list_mean_3[name] + f1_per_class_list_mean_4[name]) / 2
            # if one of the items is 0, take the other one
            elif f1_per_class_list_mean_3[name] == 0:
                f1_per_class_mean[name] = f1_per_class_list_mean_4[name]
            elif f1_per_class_list_mean_4[name] == 0:
                f1_per_class_mean[name] = f1_per_class_list_mean_3[name]
            # if both items are 0, set the mean to 0
            elif f1_per_class_list_mean_3[name] == 0 and f1_per_class_list_mean_4[name] == 0:
                f1_per_class_mean[name] = 0


        # reduct the number of decimal places to 2
        for name in class_names:
            precision_per_class_mean[name] = round(precision_per_class_mean[name], 2)
            recall_per_class_mean[name] = round(recall_per_class_mean[name], 2)
            f1_per_class_mean[name] = round(f1_per_class_mean[name], 2)

        # put all the results in a dictionary
        results = {
            'confusion_matrix': conf_matrix_mean,
            'precision': precision_mean,
            'recall': recall_mean,
            'f1': f1_mean,
            'precision_per_class': precision_per_class_mean,
            'recall_per_class': recall_per_class_mean,
            'f1_per_class': f1_per_class_mean
        }

        if self.verbose:
            # save results to a json file in self.gt_folder
            with open(os.path.join(self.gt_folder, 'results.json'), 'w') as f:
                # convert the numpy arrays to lists
                results['confusion_matrix'] = results['confusion_matrix'].tolist()
                json.dump(results, f, indent=4)
          

            print("The results are saved to the file: ", os.path.join(self.gt_folder, 'results.json'))

        return results
      
    def main(self):
        # get the metrics for all files
        results = self.get_metrics_for_all_files()
        print("results: ", results)

        return results

if __name__ == '__main__':
    # use argparse to parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_original', type=str, default='data/original', help='path to the original point clouds directory')
    parser.add_argument('--path_predicted', type=str, default='data/predicted', help='path to the predicted point clouds directory')
    parser.add_argument('--adjust_gt_data', action='store_true', default=False,  help='adjust the ground truth data')
    parser.add_argument('--plot_confusion_matrix', action='store_true', default=False,  help='plot the confusion matrix')
    parser.add_argument('--verbose', action='store_true', default=False,  help='print more information')
    args = parser.parse_args()

    # create an instance of the class
    metrics = MetricSemSeg(
        args.path_original, 
        args.path_predicted, 
        args.adjust_gt_data,
        args.plot_confusion_matrix, 
        args.verbose
        )

    # get the metrics
    results = metrics.main()







