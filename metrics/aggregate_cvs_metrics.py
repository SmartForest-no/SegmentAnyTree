import os
import pandas as pd

class AggregateCsvMetrics(object):
    def __init__(self, folder_path, verbose=False):
        self.folder_path = folder_path
        self.verbose = verbose

    def get_list_of_csv_files(self):
        # read all csv files in a folder and return a list of csv files and full path
        csv_files = []
        for file in os.listdir(self.folder_path):
            if file.endswith(".csv"):
                csv_files.append(os.path.join(self.folder_path, file))

        return csv_files
    
    def compute_average_for_single_csv(self, csv_file):
        # compute the average for a single csv file
        # read the csv file
        df = pd.read_csv(csv_file)

        # create a dictionary with a key as the column name 
        params = {}

        # print csv name of column names
        if self.verbose:
            print(df.columns)


        # compute how many rows are in the csv file in column IoU > 0.5
        num = df[df['IoU'] > 0.5].shape[0]

        # get number of rows in the csv file
        total = df.shape[0]

        # compute the average
        detection_rate = num / total

        # add the average to the dictionary
        params['Detection Rate'] = detection_rate

        # compute average of 'f1_score'
        f1_score = df['f1_score'].mean()

        # add the average to the dictionary
        params['f1_score'] = f1_score

        if self.verbose:
            print(params)

        return params
        

    def compute_average_for_all_csv(self):
        # compute the average for all csv files
        list_of_csvs = self.get_list_of_csv_files()

        for csv_file in list_of_csvs:
            # skip "summary_metrics_all_plots.csv"
            if csv_file.endswith("summary_metrics_all_plots.csv"):
                continue
            if self.verbose:
                print(f"Processing {csv_file}")
            self.compute_average_for_single_csv(csv_file)

    def __call__(self):
        self.get_list_of_csv_files()


if __name__ ==  "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_folder', type=str, required=True, help="Input folder.")
    parser.add_argument('-v', '--verbose', action='store_true', help="Print verbose output.")
    args = parser.parse_args()


    aggregate_csv_metrics = AggregateCsvMetrics(args.input_folder, verbose=args.verbose)
    aggregate_csv_metrics()
    print(aggregate_csv_metrics.compute_average_for_all_csv())