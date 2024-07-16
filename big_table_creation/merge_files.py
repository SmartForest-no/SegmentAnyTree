import os
import pandas as pd
import argparse

def merge_files(folder_name):
    subfolders = [f.path for f in os.scandir(folder_name) if f.is_dir()]
    dataframes = {}

    for subfolder in subfolders:
        subfolder_name = subfolder.split('/')[-1]
        df = pd.read_csv(os.path.join(subfolder, 'metrics_out', 'summary_metrics_all_plots.csv'), header=None, index_col=0)
        df = df.transpose()
        dataframes[subfolder_name] = df

    # Create a list for each metric
    detection = []
    omission = []
    commission = []
    rmse = []
    f1 = []

    # show all the column names in the dataframes
    for key in dataframes.keys():
        print(key, dataframes[key].columns)

    for key in ['CULS', 'NIBIO', 'RMIT', 'SCION', 'TUWIEN']:
        if 'true_positve (detection rate)' in dataframes[key].columns:
            detection.append(dataframes[key]['true_positve (detection rate)'].iloc[0])
        else:
            detection.append(None)

        if 'false_negative (omissions)' in dataframes[key].columns:
            omission.append(dataframes[key]['false_negative (omissions)'].iloc[0])
        else:
            omission.append(None)

        if 'false_positve (commission)' in dataframes[key].columns:
            commission.append(dataframes[key]['false_positve (commission)'].iloc[0])
        else:
            commission.append(None)

        if 'rmse_hight' in dataframes[key].columns:
            rmse.append(dataframes[key]['rmse_hight'].iloc[0])
        else:
            rmse.append(None)

        if 'f1_score' in dataframes[key].columns:
            f1.append(dataframes[key]['f1_score'].iloc[0])
        else:
            f1.append(None)

    # Create the merged dataframe
    merged = pd.DataFrame({
        'detection': detection,
        'omission': omission,
        'commission': commission,
        'rmse': rmse,
        'f1': f1
    }, index=['CULS', 'NIBIO', 'RMIT', 'SCION', 'TUWIEN'])

    print(merged)
    # cerate the folder if it does not exist
    if not os.path.exists(os.path.join(folder_name, 'metrics_out')):
        os.makedirs(os.path.join(folder_name, 'metrics_out'))

    merged.to_csv(os.path.join(folder_name, 'metrics_out', 'summary_metrics_all_plots.csv'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--folder_name', type=str, required=True)
    args = parser.parse_args()
    merge_files(args.folder_name)
