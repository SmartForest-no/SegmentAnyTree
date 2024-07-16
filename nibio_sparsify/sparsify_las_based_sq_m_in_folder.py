import argparse
import logging
import os
from multiprocessing import Pool, cpu_count

import laspy
import pandas as pd  # Import pandas
from tqdm import tqdm

from nibio_sparsify.sparsify_las_based_sq_m import SparsifyLasBasedSqM

class SparsifyLasBasedSqMInFolder(SparsifyLasBasedSqM):
    def __init__(self, input_folder, output_folder=None, target_density=10, verbose=False):
        super().__init__(input_file=None, output_folder=output_folder, target_density=target_density, verbose=verbose)
        self.directory_with_point_clouds = input_folder
        self.report = []

    def process_single_file(self, point_cloud_path):
        relative_path = os.path.relpath(point_cloud_path, self.directory_with_point_clouds)
        output_path = os.path.join(self.output_folder, relative_path)
        output_dir = os.path.dirname(output_path)

        try:
            os.makedirs(output_dir)
        except FileExistsError:
            pass  # Directory already exists, no need to create it again

        self.input_file = point_cloud_path
        self.output_file = output_path
        self.process()

        # Return a dictionary containing details about the sparsification process
        return {
            "input_file": self.input_file,
            "original_density": self.density,
            "target_density": self.target_density,
            "new_density": self.new_point_cloud_density
        }

    def reduce_point_clouds(self):
        point_cloud_paths = []

        for root, dirs, files in os.walk(self.directory_with_point_clouds):
            for file in files:
                if file.endswith(".las"):
                    point_cloud_paths.append(os.path.join(root, file))

        self.logger.info(f"Found {len(point_cloud_paths)} point clouds.")

        # Use multiprocessing to process the files in parallel and collect the results
        with Pool(processes=cpu_count()) as pool:
            results = list(tqdm(pool.imap(self.process_single_file, point_cloud_paths), total=len(point_cloud_paths)))

        # Convert the list of dictionaries to a pandas DataFrame
        df = pd.DataFrame(results)

        # Save the DataFrame to a CSV file
        df.to_csv("sparsification_report.csv", index=False)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_folder", help="The folder with .las files to sparsify.")
    parser.add_argument("-o", "--output_folder", default=None, help="The folder where the sparse point clouds will be saved.")
    parser.add_argument("-d", "--target_density", help="The target density in points per square meter.", default=10, type=int)
    parser.add_argument("-v", "--verbose", action="store_true", help="Print information about the process")

    args = parser.parse_args()

    sparsifier = SparsifyLasBasedSqMInFolder(
        args.input_folder,
        args.output_folder,
        args.target_density,
        args.verbose
    )
    sparsifier.reduce_point_clouds()
