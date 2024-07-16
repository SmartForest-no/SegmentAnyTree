import pandas as pd
from nibio_inference.las_to_pandas import las_to_pandas



class Metric(object):
    gt_label = 'treeID'
    preds_label = 'preds_instance_segmentation'

    def __init__(self, laz_file_path, verbose=False):
        self.df = las_to_pandas(laz_file_path)
        self.verbose = verbose


    def confusion_matrix(self):
        # Create a confusion matrix
        confusion = pd.crosstab(self.df[self.gt_label], self.df[self.preds_label])
        # save the confusion matrix to a csv file
        confusion.to_csv('confusion_matrix.csv')

        return confusion


    def __call__(self):
        self.confusion_matrix()

if __name__ ==  "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file', type=str, required=True, help="Input file.")
    parser.add_argument('-v', '--verbose', action='store_true', help="Print verbose output.")
    args = parser.parse_args()


    metric = Metric(args.input_file, verbose=args.verbose)
    metric()
    print(metric.confusion_matrix())
