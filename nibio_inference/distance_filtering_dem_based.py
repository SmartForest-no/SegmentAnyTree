import argparse

import laspy
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.spatial import KDTree

from nibio_inference.pandas_to_las import pandas_to_las
from nibio_inference.ply_to_pandas import ply_to_pandas


class DistanceFilteringDemBased(object):
    # MLS data : 1 - ground, 2 - vegetation, 3 - CWD, 4 - trunk
    GROUND_CLASS = 0
    TARGET_CLASS = 1 

    def __init__(self, distance, input_file_path, output_file_path,  verbose=False):
        # compute distance filtering based on DEM
        # remove points which are smaller than distance from the DEM
        # low vegetation - 0.01 m

        self.distance = distance
        self.point_cloud_file = input_file_path
        self.output_file_path = output_file_path
        self.verbose = verbose

    def read_las_and_put_to_pandas(self, input_file):
        # Read the file
        points_df = ply_to_pandas(input_file)

        # get points which belong to the ground class
        ground_points = points_df[points_df['label'] == self.GROUND_CLASS]

        # get points which belong to the target class
        target_points = points_df[points_df['label'] == self.TARGET_CLASS]

        # get just x, y, z columns from ground_points
        ground_points = ground_points[['x', 'y', 'z']]

        # get just x, y, z columns from target_points
        target_points = target_points[['x', 'y', 'z']]

        return points_df, ground_points, target_points

    def compute_dem_for_ground(self, ground_points):
        # DEM - Digital Elevation Model

        # create digital elevation model (DEM) from label 1 points
        x = ground_points['x']
        y = ground_points['y']
        z = ground_points['z']

        # create a grid of points
        xi = np.linspace(x.min(), x.max(), 1000)
        yi = np.linspace(y.min(), y.max(), 1000)
        xi, yi = np.meshgrid(xi, yi)

        # interpolate
        zi = griddata((x, y), z, (xi, yi), method='linear')

        # fill the NaN values and missing and numertical values  with 0
        zi = np.nan_to_num(zi, copy=False)

        # replace non-numerical values with 0
        zi[zi == np.inf] = 0

        # reduce precision of the DEM to 2 decimal places
        zi = np.around(zi, decimals=2)

        # put DEM into pandas dataframe 
        dem = pd.DataFrame({'x': xi.flatten(), 'y': yi.flatten(), 'z': zi.flatten(), })

        return dem

    def compute_distance_between_dem_and_target(self, dem, target_points):
        original_target_points = target_points.copy()
        target_points = target_points.values

        # Create the KD-tree using the DEM data
        dem_tree = KDTree(dem[:, :2])

        # Compute distances for all target points
        _, id = dem_tree.query(target_points[:, :2], workers=-1)

        # get distance in z direction between target points and dem
        original_target_points['z_distance'] = original_target_points['z'] - dem[id, 2]

        # add the distance in z direction to the original dataframe as the column 'z_distance'
        original_target_points['z_distance'] = original_target_points['z_distance'].values

        return original_target_points


    def filter_points(self, target_points_with_distances):
        # filter points
        target_points_with_distances = target_points_with_distances[target_points_with_distances['z_distance'] < self.distance]

        return target_points_with_distances
  

    def update_las_file(self, points_df, target_points_with_distances):
        # remove all the points which belong to the target class and are in the target_points_with_distances dataframe
        points_df = points_df[~points_df.isin(target_points_with_distances)].dropna()

        return points_df

    def save_las_file(self, points_df):
        # save pandas dataframe to csv
        pandas_to_las.pandas_to_las(points_df, self.output_file_path, csv_file_provided=False, verbose=self.verbose)

    def run(self):
        if self.verbose:
            print('distance: {}'.format(self.distance))
            print('point_cloud_file: {}'.format(self.point_cloud_file))
            print('output_las_file_path: {}'.format(self.output_file_path))
        points_df, ground_points, target_points = self.read_las_and_put_to_pandas(input_file=self.point_cloud_file) 

        dem = self.compute_dem_for_ground(ground_points=ground_points)

        target_points_with_distances = self.compute_distance_between_dem_and_target(dem=dem.values, target_points=target_points)

        filtered_points = self.filter_points(target_points_with_distances=target_points_with_distances)

        points_df = self.update_las_file(points_df=points_df, target_points_with_distances=filtered_points)

        self.save_las_file(points_df=points_df)

    def __call__(self):
        self.run()


if __name__ == '__main__':
    # parse the arguments
    parser = argparse.ArgumentParser(description='Distance filtering based on DEM')
    parser.add_argument('-d', '--distance', help='Distance in meters e.g. 0.5', default=0.5, required=False, type=float)
    parser.add_argument('-i', '--input', help='Input file', required=True)
    parser.add_argument('-o', '--output', help='Output file', required=True)
    parser.add_argument('-v', '--verbose', action='store_true', help="Print information about the process")    
    
    args = vars(parser.parse_args())

    # get the arguments
    DISTANCE = args['distance']
    INPUT_LAS_FILE = args['input']
    OUTPUT_LAS_FILE = args['output']
    VERBOSE = args['verbose']

    # run the distance filtering
    DistanceFilteringDemBased(
        distance=DISTANCE, 
        input_file_path=INPUT_LAS_FILE, 
        output_file_path=OUTPUT_LAS_FILE, 
        verbose=VERBOSE
        )
    ()
    
   


