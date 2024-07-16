import os
import pandas as pd
from nibio_inference.las_to_pandas import las_to_pandas
import numpy as np


class Viz():
    """
    Class for visualization
    """
    PRED_LABEL = 'pred_label'
    GT_LABEL = 'gt_label(dominant_label)'

    def __init__(
            self,
            cvs_path: str,
            gt_path: str,
            pred_path: str,
            output_path: str,
            verbose: bool = False
    ):
        self.cvs_path = cvs_path
        self.gt_path = gt_path
        self.pred_path = pred_path
        self.output_path = output_path
        # create output_path if not exist
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        self.verbose = verbose


    def _get_df(self, path: str) -> pd.DataFrame:
        """
        Get dataframe from path
        """
        df = las_to_pandas(path)
        return df
    
    def _get_gt_df(self) -> pd.DataFrame:
        """
        Get ground truth dataframe
        """
        return self._get_df(self.gt_path)
    
    def _get_pred_df(self) -> pd.DataFrame:
        """
        Get prediction dataframe
        """
        return self._get_df(self.pred_path)
    
    def _get_cvs_df(self) -> dict:
        """
        Get cvs dataframe
        """
        csv_df = pd.read_csv(self.cvs_path)

        print("DataFrame columns:", csv_df.columns)
        print("GT_LABEL:", self.GT_LABEL)

        print("Type of GT_LABEL:", type(self.GT_LABEL))
        print("Type of DataFrame column name:", type(csv_df.columns[2]))  # Assuming 'gt_label(dominant_label)' is the third column

        # print name of columns
        csv_df.columns = csv_df.columns.str.strip()

        if self.verbose:
            print(csv_df.columns)

        # get PRED_LABEL and GT_LABEL and match them into a dict where key is GT_LABEL and value is PRED_LABEL
        csv_df = csv_df[[self.GT_LABEL, self.PRED_LABEL]]
        csv_df = csv_df.set_index(self.GT_LABEL)
        csv_dict = csv_df.to_dict()[self.PRED_LABEL]
        # reduce pred by 1
        csv_dict = {int(k): int(v) - 1 for k, v in csv_dict.items()}
        return csv_dict
    

    def main(self):
        """
        Main function for visualization
        """
        gt_df = self._get_gt_df()
        pred_df = self._get_pred_df()
        csv_dict = self._get_cvs_df()

        # iterate through csv_dict
        for gt_label in csv_dict:
            pred_label = csv_dict[gt_label]
            gt_df_filtered = gt_df[gt_df['treeID'] == gt_label]
            
            # get only X, Y, Z, and mark them as RGB  blue
            gt_df_xyz = gt_df_filtered[['X', 'Y', 'Z']].copy()
            gt_df_xyz['R'] = 0
            gt_df_xyz['G'] = 0
            gt_df_xyz['B'] = 255
            pred_df_filtered = pred_df[pred_df['preds_instance_segmentation'] == pred_label]
            # get only X, Y, Z, and mark them as RGB  red
            pred_df_xyz = pred_df_filtered[['X', 'Y', 'Z']].copy()
            pred_df_xyz['R'] = 255
            pred_df_xyz['G'] = 0
            pred_df_xyz['B'] = 0

            # Ignore points that do not exist in both gt and pred dataframes
          
            # save as csv with name of gt_label and pred_label combined with _ as filename in output_path
            df = pd.concat([gt_df_xyz, pred_df_xyz])
            df.to_csv(f'{self.output_path}/{gt_label}_{pred_label}.csv')

            # Find outliers based on false positives and false negatives
            # outliers are points that exist in gt but not in pred or vice versa
            # outliers are marked as RGB red
            # find points which are gt_df_xyz but not pred_df_xyz and vice versa
            # drop first column which is index column from df

            df_only_xyz = df[['X', 'Y', 'Z']].copy()    

           # mark as outliers the points which are not duplicated
            outliers_df = df_only_xyz.drop_duplicates(keep=False)


            print("df_only_xyz:", df_only_xyz.head())

            # Save outliers to CSV
            outliers_df['R'] = 255
            outliers_df['G'] = 0
            outliers_df['B'] = 0
            outliers_df.to_csv(f'{self.output_path}/{gt_label}_{pred_label}_outliers.csv')

            if self.verbose:
                num_outliers = len(outliers_df)
                print(f'gt_label: {gt_label}, pred_label: {pred_label}, num_outliers: {num_outliers}')

          

if __name__ == '__main__':
    viz = Viz(
        cvs_path='/home/nibio/mutable-outside-world/for_instance_no_outer_sparse_many_times/sparse_0/results_100/culs_out/metrics_out/6309_CULS_plot_2_annotated_out.csv',
        gt_path='/home/nibio/mutable-outside-world/for_instance_no_outer_sparse_many_times/sparse_0/results_100/culs_out/input_data/6309_CULS_plot_2_annotated.las',
        pred_path='/home/nibio/mutable-outside-world/for_instance_no_outer_sparse_many_times/sparse_0/results_100/culs_out/final_results/6309_CULS_plot_2_annotated_out.laz',
        output_path='visualization/output',
        verbose=True
    )
    viz.main()