import os
import csv
import argparse

def process_file(filepath, folder_name, results):
    """Process the csv file and update results."""
    with open(filepath, 'r') as file:
        reader = csv.reader(file, delimiter=',')
        metrics = {}
        for rows in reader:
            try:
                metrics[rows[0]] = float(rows[1])
            except IndexError:
                print(f"Skipped problematic line in {filepath}: {rows}")
            except ValueError:
                print(f"Could not convert value to float in {filepath}: {rows[1]}")
        results[folder_name] = {
            'completeness': metrics.get('true_positve (detection rate)', 0),
            'Omission': metrics.get('false_negative (omissions)', 0),
            'commission': metrics.get('false_positve (commission)', 0),
            'RMSE': metrics.get('rmse_hight', 0),
            'F1': metrics.get('f1_score', 0)
        }

def main(input_folder, output_file="merged_metrics.csv"):
    results = {}  # Dictionary to hold results. Folder name as key.

    # Traverse directories
    for root, dirs, files in os.walk(input_folder):
        if os.path.basename(root) == 'metrics_out':
            csv_file_path = os.path.join(root, 'summary_metrics_all_plots.csv')
            if os.path.exists(csv_file_path):
                folder_name = os.path.basename(os.path.dirname(root))
                process_file(csv_file_path, folder_name, results)           

    # Write to csv
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['Folder', 'completeness', 'Omission', 'commission', 'RMSE', 'F1']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for folder, metrics in results.items():
            row = {'Folder': folder}
            row.update(metrics)
            writer.writerow(row)

    print(f"Data has been merged and saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge metrics from subfolders.")
    parser.add_argument('-i', '--input_folder', type=str, help="Path to the input folder containing subfolders to process.")
    parser.add_argument('-o', '--output', type=str, default="merged_metrics.csv", help="Name of the output CSV file.")
    args = parser.parse_args()

    main(args.input_folder, args.output)
