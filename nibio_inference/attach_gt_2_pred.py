import laspy
import argparse
import pandas as pd
import numpy as np
from sklearn.neighbors import KDTree

import logging
logging.basicConfig(level=logging.INFO)

class AttachLabelsToLasFileGt2Pred():
    def __init__(
        self, 
        gt_las_file_path, 
        target_las_file_path, 
        update_las_file_path,
        gt_label_name='gt_label',
        target_label_name='target_label',
        verbose=False
         ):

        self.gt_las_file_path = gt_las_file_path
        self.target_las_file_path = target_las_file_path
        self.update_las_file_path = update_las_file_path
        self.gt_label_name = gt_label_name
        self.target_label_name = target_label_name
        self.verbose = verbose
     

    def attach_labels(self):
        # read las file
        gt_las = laspy.read(self.gt_las_file_path)
        target_las = laspy.read(self.target_las_file_path)

        # read x, y, z and gt_label from gt las file to variables and set a size of sample_size
        gt = gt_las.xyz
        target = target_las.xyz
        gt_label = gt_las[self.gt_label_name]

        # create a tree from target las file
        tree = KDTree(gt, leaf_size=50, metric='euclidean')
        # find the nearest neighbor for each point in target las file
        ind = tree.query(target, k=1, return_distance=False)

        # map target labels to gt labels
        target_labels = gt_label[ind]

        # get point format of target las file
        point_format = target_las.point_format
        # get header of target las file
        header = target_las.header

        # create a new las file with the same header as target las file
        new_header = laspy.LasHeader(point_format=point_format.id, version=header.version)
        # add gt_label and target_label extra dimensions to the new las file
         # get extra dimensions from target las file
        target_extra_dimensions = list(target_las.point_format.extra_dimension_names)

        # add extra dimensions to new las file
        for item in target_extra_dimensions:
            new_header.add_extra_dim(laspy.ExtraBytesParams(name=item, type=np.int32))
            
        # add gt_label and target_label extra dimensions to the new las file
        new_header.add_extra_dim(laspy.ExtraBytesParams(name=self.target_label_name, type=np.int32))

        new_las = laspy.LasData(new_header)

        # copy x, y, z, gt_label and target_label from target las file to the new las file
        new_las.x = target_las.x
        new_las.y = target_las.y
        new_las.z = target_las.z

        # copy contents of extra dimensions from target las file to the new las file
        for item in target_extra_dimensions:
            new_las[item] = target_las[item]

        new_las[self.target_label_name] = target_labels.T[0]
        
        # write the new las file
        new_las.write(self.update_las_file_path)

    def main(self):
        self.attach_labels(
        )
        if self.verbose:
            # write a report using logging
            logging.info('gt_las_file_path: {}'.format(self.gt_las_file_path))
            logging.info('target_las_file_path: {}'.format(self.target_las_file_path))
            logging.info('update_las_file_path: {}'.format(self.update_las_file_path))
            logging.info('gt_label_name: {}'.format(self.gt_label_name))
            logging.info('target_label_name: {}'.format(self.target_label_name))

            # print the size of the las files
            gt_las = laspy.read(self.gt_las_file_path)
            target_las = laspy.read(self.target_las_file_path)
            gt_las_size = gt_las.x.shape[0]
            target_las_size = target_las.x.shape[0]
            logging.info('gt_las_size: {}'.format(gt_las_size))
            logging.info('target_las_size: {}'.format(target_las_size))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Attach labels to las file')
    parser.add_argument('--gt_las_file_path', type=str, required=True)
    parser.add_argument('--target_las_file_path', type=str, required=True)
    parser.add_argument('--update_las_file_path', type=str, required=True)
    parser.add_argument('--gt_label_name', type=str, default='gt_label')
    parser.add_argument('--target_label_name', type=str, default='target_label')
    parser.add_argument('--verbose', action='store_true', help="Print information about the process")
    args = parser.parse_args()

    # create an instance of AttachLabelsToLasFile class
    attach_labels_to_las_file = AttachLabelsToLasFileGt2Pred(
        gt_las_file_path=args.gt_las_file_path,
        target_las_file_path=args.target_las_file_path,
        update_las_file_path=args.update_las_file_path,
        gt_label_name=args.gt_label_name,
        target_label_name=args.target_label_name,
        verbose=args.verbose

        )

    # call main function
    attach_labels_to_las_file.main()
