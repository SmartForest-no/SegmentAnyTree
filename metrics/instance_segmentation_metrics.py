import argparse
import os
import laspy
import logging
import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree

logging.basicConfig(level=logging.INFO)

class InstanceSegmentationMetrics:
    GT_LABEL_NAME = 'treeID'  #GT_LABEL_NAME = 'StemID', 'treeID'
    # TARGET_LABEL_NAME = 'preds_instance_segmentation' # 'instance_nr'
    TARGET_LABEL_NAME = 'PredInstance' # 'instance_nr'

    def __init__(
        self, 
        input_file_path, 
        instance_segmented_file_path, 
        remove_ground = False,
        csv_file_name=None,
        verbose=False
        ) -> None:

        self.input_file_path = input_file_path
        self.instance_segmented_file_path = instance_segmented_file_path
        self.remove_ground = remove_ground
        self.csv_file_name = csv_file_name
        self.verbose = verbose
        # read and prepare input las file and instance segmented las file
        self.input_las = laspy.read(self.input_file_path)

        self.instance_segmented_las = laspy.read(self.instance_segmented_file_path)

        self.skip_flag = self.check_if_labels_exist(
            X_label=self.GT_LABEL_NAME,
            Y_label=self.TARGET_LABEL_NAME
            )

        if not self.skip_flag:
            # get labels from input las file
            self.X_labels = self.input_las[self.GT_LABEL_NAME].astype(int) 

            # get labels from instance segmented las file
            self.Y_labels = self.instance_segmented_las[self.TARGET_LABEL_NAME].astype(int) 

            self.Y_labels += 1

            # make all points with label < 0 to 0
            self.Y_labels[self.Y_labels < 0] = 0
            # if self.remove_ground:
            #     # the labeling starts from 0, so we need to remove the ground
            #     self.Y_labels += 1
    
            # do knn mapping
            self.dict_Y = self.do_knn_mapping()
        else:
            logging.info('Skipping the file: {}'.format(self.input_file_path))


    def check_scale(self, las_file):
        # Access header information
        x_dimension_units = las_file.header.x_scale
        y_dimension_units = las_file.header.y_scale
        z_dimension_units = las_file.header.z_scale

        # Print units for X, Y, and Z dimensions
        print(f"Units for X: {x_dimension_units}")
        print(f"Units for Y: {y_dimension_units}")
        print(f"Units for Z: {z_dimension_units}")



    def check_if_labels_exist(self, X_label='treeID', Y_label='instance_nr'):
        # check if the labels exist in the las files
        skip_flag = False

        if X_label not in self.input_las.header.point_format.dimension_names:
            skip_flag = True
        if Y_label not in self.instance_segmented_las.header.point_format.dimension_names:
            skip_flag = True
        
        return skip_flag

    def do_knn_mapping(self):
        x = self.input_las.x
        y = self.input_las.y
        z = self.input_las.z
        X = np.vstack((x, y, z)).T

        # X = self.input_las.xyz
        y_x = self.instance_segmented_las.x
        y_y = self.instance_segmented_las.y
        y_z = self.instance_segmented_las.z
        Y = np.vstack((y_x, y_y, y_z)).T

        # Y = self.instance_segmented_las.xyz
        X_labels = self.X_labels
        Y_labels = self.Y_labels

        # create a KDTree for X
        tree = KDTree(X, leaf_size=50, metric='euclidean')       
        # query the tree for Y     
        ind = tree.query(Y, k=1, return_distance=False)   

        # get labels for ind
        ind_labels_Y = X_labels[ind]
        # reshape to 1D
        ind_labels_Y = ind_labels_Y.reshape(-1) # labels from X matched to Y

        # get all the indices in X which were matched to Y
        residual_ind = np.delete(np.arange(X.shape[0]), ind.reshape(-1)) # indices of X which were not matched to Y

        # create a dictionary which contains Y, Y_labels and ind_labels_Y
        dict_Y = {
            'X': X, # X is the input las file
            'Y': Y, # Y is the instance segmented las file
            'Y_labels': Y_labels,  # Y_labels is the instance segmented las file
            'ind_labels_Y': ind_labels_Y, # ind_labels_Y is the labels from X matched to Y (new gt labels) 
            'ind': ind, # ind is the indices of X which were matched to Y
            'residual_ind': residual_ind # residual_ind is the indices of X which were not matched to Y
            }
        
        return dict_Y

    def get_dominant_lables_sorted(self):
        # get unique labels from Y_labels
        Y_unique_labels = np.unique(self.Y_labels)

        print('Y_unique_labels in the procedure: ', len(Y_unique_labels))
    
        dominant_labels = {}
        for label in Y_unique_labels:
            # get the indices of Y_labels == label
            ind_Y_labels = np.where(self.Y_labels == label)[0]
            # get the ind_labels_Y for these indices
            ind_labels_Y = self.dict_Y['ind_labels_Y'][ind_Y_labels]
            # get the unique ind_labels_Y
            unique_ind_labels_Y = np.unique(ind_labels_Y)
            # print the number of points for each unique ind_labels_Y
            tmp_dict = {}
            for unique_ind_label_Y in unique_ind_labels_Y:
                # get the indices of ind_labels_Y == unique_ind_label_Y
                ind_ind_labels_Y = np.where(ind_labels_Y == unique_ind_label_Y)[0]
                # put the number of points to the tmp_dict
                tmp_dict[str(unique_ind_label_Y)] = ind_ind_labels_Y.shape[0]
        
            # put the dominant label to the dominant_labels
            dominant_labels[str(label)] = tmp_dict

        # sort dominant_labels by the number of points
        dominant_labels_sorted = {}
        for key, value in dominant_labels.items():
            dominant_labels_sorted[key] = {k: v for k, v in sorted(value.items(), key=lambda item: item[1], reverse=True)}

        # iterate over the dominant_labels_sorted and sort it based on the first value of sub-dictionary
        dominant_labels_sorted = {
            k: v for k, v in sorted(dominant_labels_sorted.items(), key=lambda item: list(item[1].values())[0], reverse=True)}
        
        print('dominant_labels_sorted: ', len(dominant_labels_sorted))

        return dominant_labels_sorted

    def extract_from_sub_dict(self, target_dict, label):
        new_dict = {}

        for key_outer, value_outer in target_dict.items():
            tmp_dict = {}
            
            for item_inner in value_outer.keys():
                if item_inner == label:
                    tmp_dict[item_inner] = value_outer[item_inner]
            new_dict[key_outer] = (tmp_dict)
        return new_dict


    # define a function that finds class in input_file with the most points
    def find_dominant_classes_in_gt(self, input_file):
        # get the unique labels
        unique_labels = np.unique(input_file[self.GT_LABEL_NAME]).astype(int)
        # create a dictionary
        tmp_dict = {}
        for label in unique_labels:
            # get the indices of input_file.treeID == label
            ind_label = np.where(input_file[self.GT_LABEL_NAME] == label)[0]
            # put the number of points to the tmp_dict
            tmp_dict[str(label)] = ind_label.shape[0]
        # sort tmp_dict by the number of points
        tmp_dict_sorted = {k: v for k, v in sorted(tmp_dict.items(), key=lambda item: item[1], reverse=True)}

        # remove key 0 from tmp_dict_sorted
        if self.remove_ground:
            tmp_dict_sorted.pop('0', None)

        return tmp_dict_sorted.keys()

    def get_the_dominant_label(self, dominant_labels_sorted):
        # get the dominant label
        # iterate over the dominant_labels_sorted and sort it based on the first value of sub-dictionary 
        # if sub-dictionary is empty, remove the key from the dictionary

        for key, value in dominant_labels_sorted.copy().items():
            if not value:
                del dominant_labels_sorted[key]

        dominant_labels_sorted = {
            k: v for k, v in sorted(dominant_labels_sorted.items(), key=lambda item: list(item[1].values())[0], reverse=True)}

        dominant_label_key = list(dominant_labels_sorted.keys())[0]
        dominant_label_value = list(dominant_labels_sorted.values())[0]
        dominant_label = list(dominant_label_value.keys())[0]
       
        # get dominant_label_key and dominant_label for which dominant_label_value.values() has the highest value
        for key, value in dominant_labels_sorted.items():
            for item in value.keys():
                if value[item] > dominant_label_value[dominant_label]:
                    dominant_label_key = key
                    dominant_label = item
                    dominant_label_value = value

        return dominant_label_key, dominant_label


    def remove_dominant_label(self, dominant_labels_sorted, dominant_label_key, dominant_label):
        # remove the dominant_label_key from dominant_labels_sorted
        dominant_labels_sorted.pop(dominant_label_key)
        # remove the dominant_label from the sub-dictionary of dominant_labels_sorted
        for key, value in dominant_labels_sorted.items():
            if dominant_label in value:
                value.pop(dominant_label)

        return dominant_labels_sorted


    def iterate_over_pc(self):
        label_mapping_dict = {}

        dominant_labels_sorted = self.get_dominant_lables_sorted()
        gt_classes_to_iterate = self.find_dominant_classes_in_gt(self.input_las)

        for gt_class in gt_classes_to_iterate:
            # Check if all dominant labels are exhausted
            if not any(dominant_labels_sorted.values()):
                break

            # If only one dominant label remains
            if len(dominant_labels_sorted) == 1:
                dominant_label_key, dominant_label = self.get_the_dominant_label(dominant_labels_sorted)
                label_mapping_dict[dominant_label_key] = dominant_label
                break

            # Extract dominant labels corresponding to the current gt_class
 
            extracted = self.extract_from_sub_dict(dominant_labels_sorted, gt_class)

            # If extracted labels are empty, continue to the next iteration
            if not any(extracted.values()):
                continue

            dominant_label_key, dominant_label = self.get_the_dominant_label(extracted)

            self.remove_dominant_label(dominant_labels_sorted, dominant_label_key, dominant_label)
            label_mapping_dict[dominant_label_key] = dominant_label

        # Convert keys and values to integers
        label_mapping_dict = {int(k): int(v) for k, v in label_mapping_dict.items()}

        return label_mapping_dict

    def compute_metrics(self):
        # get the label_mapping_dict
        metric_dict = {}

        if not self.skip_flag:
            label_mapping_dict = self.iterate_over_pc()

            for label in list(label_mapping_dict.keys()):
                # get the indices of Y_labels == label
                ind_Y_labels_label = np.where(self.Y_labels == label)[0]

                # get the ind_labels_Y for these indices
                ind_labels_Y = self.dict_Y['ind_labels_Y'][ind_Y_labels_label]

                # get the dominant label for this label
                dominant_label = label_mapping_dict[label]

                # get the indices of ind_labels_Y == dominant_label
                ind_dominant_label = np.where(ind_labels_Y == dominant_label)[0]

                ## true positive is the number of points for dominant_label
                true_positive = ind_dominant_label.shape[0]

                ## points which are within the relabelled pred but are not dominant_label
                false_positive = ind_Y_labels_label.shape[0] - true_positive

                ## false negative is the number of points which are not in Y but are in X
                false_negative = np.where(self.X_labels[self.dict_Y['residual_ind']] == dominant_label)[0].shape[0]

                ## true negative 
                true_negative = self.dict_Y['X'].shape[0] - false_negative - true_positive - false_positive

                # sum all the true_positive, false_positive, false_negative, true_negative
                sum_all = true_positive + false_positive + false_negative + true_negative

                # get precision
                precision = true_positive / (true_positive + false_positive)
                # get recall
                recall = true_positive / (true_positive + false_negative)
                # get f1 score
                f1_score = 2 * (precision * recall) / (precision + recall)
                # get IoU
                IoU = true_positive / (true_positive + false_positive + false_negative)


                Z_values = np.array(self.input_las.z)[self.X_labels == dominant_label]
                hight_of_tree_gt = (Z_values.max() - Z_values.min())
                # find hight of the tree in the prediction
                Z_values = np.array(self.instance_segmented_las.z)[self.Y_labels == label]
                hight_of_tree_pred = (Z_values.max() - Z_values.min()) 

                # get abs resiudal of the hight of the tree in the prediction
                residual_hight_of_tree_pred = (hight_of_tree_gt - hight_of_tree_pred)

                rmse_hight = np.square(residual_hight_of_tree_pred)

                # create tmp dict
                tmp_dict = {
                'pred_label': label,
                'gt_label(dominant_label)': dominant_label,
                'high_of_tree_gt': hight_of_tree_gt,
                'high_of_tree_pred': hight_of_tree_pred,
                'residual_hight(gt_minus_pred)': residual_hight_of_tree_pred,
                'rmse_hight': rmse_hight,
                'sum_all': sum_all,
                'true_positive': true_positive, 
                'false_positive': false_positive, 
                'false_negative': false_negative, 
                'true_negative': true_negative,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'IoU': IoU,
                }

                metric_dict[str(label)] = tmp_dict
            
        # list of interesting metrics 
        interesting_parameters = ['precision', 'recall', 'f1_score', 'IoU', 'residual_hight(gt_minus_pred)', 'rmse_hight']

        # weight the metrics by tree hight
        metric_dict_weighted_by_tree_hight = {}
        # itialize the metric_dict_weighted_by_tree_hight
        for parameter in interesting_parameters:
            metric_dict_weighted_by_tree_hight[parameter] = 0

        # do this if there is at least one label
        if metric_dict:
            for label in metric_dict.keys():
                for parameter in interesting_parameters:
                    metric_dict_weighted_by_tree_hight[parameter] += metric_dict[label]['high_of_tree_gt'] * metric_dict[label][parameter]
            # divide by the sum of the hights of the trees
            for parameter in interesting_parameters:
                metric_dict_weighted_by_tree_hight[parameter] /= sum([metric_dict[label]['high_of_tree_gt'] for label in metric_dict.keys()])
                if parameter == 'rmse_hight':
                    # compute sqrt of the residual hight (we are computing RMSE)
                    metric_dict_weighted_by_tree_hight[parameter] = metric_dict_weighted_by_tree_hight[parameter] ** 0.5

        # compute the mean of the metrics
        metric_dict_mean = {}
        for parameter in interesting_parameters:
            metric_dict_mean[parameter] = 0

        # do this if there is at least one label
        if metric_dict:
            for key, value in metric_dict.items():
                for parameter in interesting_parameters:
                    metric_dict_mean[parameter] += value[parameter]

            for parameter in interesting_parameters:
                # metric_dict_mean[parameter] = metric_dict_mean[parameter] /  len(np.unique(self.input_las[self.GT_LABEL_NAME]))
                metric_dict_mean[parameter] = metric_dict_mean[parameter] /  len(metric_dict) # local metric

                if parameter == 'rmse_hight':
                    # compute sqrt of the residual hight (we are computing RMSE)
                    metric_dict_mean[parameter] = metric_dict_mean[parameter] ** 0.5

        # compute tree level metrics
        if metric_dict:
            # get the number of trees in the ground truth
            gt_trees = np.unique(self.input_las[self.GT_LABEL_NAME])

            # remove 0 from gt_trees
            gt_trees = gt_trees[gt_trees != 0]

            # get the number of trees that are predicted correctly
            trees_predicted = np.unique([metric_dict[key]['gt_label(dominant_label)'] for key in metric_dict.keys()])

            trees_predicted_all = [metric_dict[key]['gt_label(dominant_label)'] for key in metric_dict.keys()]

            # iterate over metric_dict and get the number of trees that are predicted correctly with IoU > 0.5
            trees_correctly_predicted_IoU = np.unique([metric_dict[key]['gt_label(dominant_label)'] for key in metric_dict.keys() if metric_dict[key]['IoU'] > 0.5])

            # convert to set
            gt_trees = set(gt_trees)
            trees_predicted = set(trees_predicted)
            trees_correctly_predicted_IoU = set(trees_correctly_predicted_IoU)


            tree_level_metric = {
                'true_positve (detection rate)': len(trees_correctly_predicted_IoU) / len(gt_trees), 
                'false_positve (commission)': len(trees_predicted - trees_correctly_predicted_IoU) / len(trees_predicted), 
                'false_negative (omissions)': len(gt_trees - trees_correctly_predicted_IoU) / len(gt_trees), 
                'gt': len(gt_trees)}

            # add tree level metrics to the metric_dict_mean
            metric_dict_mean.update(tree_level_metric)

            if self.verbose:
                print('Tree level metrics:')    
                print(f'Trees in the ground truth: {gt_trees}')
                print(f'Trees correctly predicted: {trees_predicted}')
                print(f'Trees correctly predicted with IoU > 0.5: {trees_correctly_predicted_IoU}')

                print(tree_level_metric)

            

        return metric_dict, metric_dict_weighted_by_tree_hight, metric_dict_mean

    def print_metrics(self, metric_dict):
        for key, value in metric_dict.items():
            print(f'Label: {key}')
            for key2, value2 in value.items():
                print(f'{key2}: {value2}')
            print('')

    def save_to_csv_file(self, metric_dict):
        df = pd.DataFrame(metric_dict).T
        # save to csv file and show the index
        df.to_csv(self.csv_file_name, index=True, header=True)


    def main(self):
        metric_dict, metric_dict_weighted_by_tree_hight, metric_dict_mean  = self.compute_metrics()

        if self.verbose:
            f1_weighted_by_tree_hight = metric_dict_weighted_by_tree_hight['f1_score']
            print(f'f1_score_weighted: {f1_weighted_by_tree_hight}')
            for key, value in metric_dict.items():
                print(f'Label: {key}')
                for key2, value2 in value.items():
                    print(f'{key2}: {value2}')
                print('')
       
        if self.csv_file_name is not None:
            self.save_to_csv_file(metric_dict)
            if self.verbose:
                print(f'Metrics saved to {self.csv_file_name}')

        return metric_dict, metric_dict_weighted_by_tree_hight, metric_dict_mean


# main
if __name__ == '__main__':
    # do argparse input_file_path, instance_segmented_file_path, verbose
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file_path', type=str, required=True)
    parser.add_argument('--instance_segmented_file_path', type=str, required=True)
    parser.add_argument('--remove_ground', action='store_true', help="Do not take into account the ground (class 0).", default=False)
    parser.add_argument('--csv_file_name', type=str, help="Name of the csv file to save the metrics to", default=None)
    parser.add_argument('--verbose', action='store_true', help="Print information about the process", default=False)

    args = parser.parse_args()

    # create instance of the class InstanceSegmentationMetrics
    instance_segmentation_metrics = InstanceSegmentationMetrics(
        args.input_file_path, 
        args.instance_segmented_file_path, 
        args.remove_ground,
        args.csv_file_name,
        args.verbose
        )
    
    # compute metrics
    metric_dict, _, _ = instance_segmentation_metrics.main()