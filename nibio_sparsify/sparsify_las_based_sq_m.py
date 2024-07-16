import argparse
import os
import numpy as np
import random
import laspy
from scipy.spatial import ConvexHull
import logging
from tqdm import tqdm


class SparsifyLasBasedSqM:
    def __init__(self, input_file, output_folder=None, target_density=10, verbose=False):
        self.input_file = input_file
        if output_folder is None:
            self.output_folder = os.path.dirname(input_file)
        else:
            self.output_folder = output_folder
        self.target_density = target_density
        self.density = None
        self.new_point_cloud_density = None
        self.verbose = verbose
        self.output_file = None  # New attribute

        # Initialize logging
        self.logger = logging.getLogger(__name__)
        if self.verbose:
            self.logger.setLevel(logging.INFO)
        else:
            self.logger.setLevel(logging.WARNING)

        self.logger.info(f"Initialized with input file: {self.input_file}, target density: {self.target_density}")

    def calculate_density_convex_hull(self, las):
        # check if las.X or las.x is used
        if hasattr(las, 'x') and las.x is not None:
            # las.x is used
            points_3D = np.vstack((las.x, las.y, las.z)).transpose()
        else:
            # las.X is used
            points_3D = np.vstack((las.X, las.Y, las.Z)).transpose()

        points_2D = points_3D[:, :2]
        hull = ConvexHull(points_2D)
        area = hull.volume
        point_count = len(points_2D)
        density = point_count / area
        return density

    def sparsify(self, point_cloud):
        self.density = self.calculate_density_convex_hull(point_cloud)
        self.logger.info(f"Point cloud density: {self.density} points per square meter.")

        # Check if the point cloud density is already less than or equal to the target density
        if self.density <= self.target_density:
            self.logger.info(f"Point cloud is already sparser than the target density. Skipping sparsification.")
            return point_cloud.points  # Return the original points without sparsification

        x = point_cloud.x
        keep_count = int(len(x) * (self.target_density / self.density))
        sampled_indices = random.sample(range(len(x)), keep_count)
        filtered_point_cloud = point_cloud.points[sampled_indices]
        self.new_point_cloud_density = self.calculate_density_convex_hull(filtered_point_cloud)
        self.logger.info(f"Reduced point cloud size from {len(x)} to {len(filtered_point_cloud)} points.")
        self.logger.info(f"Reduced point cloud by {(1 - len(filtered_point_cloud) / len(x)) * 100}%.")
        self.logger.info(f"New point cloud density: {self.new_point_cloud_density} points per square meter.")
        return filtered_point_cloud

    def process(self):
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        inFile = laspy.read(self.input_file)
        filtered_points = self.sparsify(inFile)
        self.logger.info("Creating output laspy object")
        outFile = laspy.create(point_format=inFile.point_format, file_version=inFile.header.version)
        outFile.header = inFile.header
        outFile.points = filtered_points
        if self.output_file is None:  # If output_file is not set, use the default naming
            self.output_file = os.path.join(self.output_folder, os.path.basename(self.input_file).replace(".las", f"_sparse_{self.target_density}.las"))
        outFile.write(self.output_file)


class SparsifyLasBasedSqMInFolder(SparsifyLasBasedSqM):
    def __init__(self, input_folder, output_folder=None, target_density=10, verbose=False):
        super().__init__(input_file=None, output_folder=output_folder, target_density=target_density, verbose=verbose)
        self.directory_with_point_clouds = input_folder
        self.report = None

    def reduce_point_clouds(self):
        point_cloud_paths = []

        for root, dirs, files in os.walk(self.directory_with_point_clouds):
            for file in files:
                if file.endswith(".las"):
                    point_cloud_paths.append(os.path.join(root, file))

        self.logger.info(f"Found {len(point_cloud_paths)} point clouds.")

        for point_cloud_path in tqdm(point_cloud_paths):
            relative_path = os.path.relpath(point_cloud_path, self.directory_with_point_clouds)
            output_path = os.path.join(self.output_folder, relative_path)
            output_dir = os.path.dirname(output_path)

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            self.input_file = point_cloud_path
            self.output_file = output_path
            self.process()


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
