import laspy
import pandas as pd

# works with laspy 2.1.2 (the other versions are not tested)

def pandas_to_las(csv, csv_file_provided=False, output_file_path=None, do_compress=False, verbose=False):
    """
    Convert a pandas DataFrame to a .las file.

    Parameters
    ----------
    csv : pandas DataFrame
        The DataFrame to be converted to .las file.
        But if the csv_file_provided argument is true,
        the csv argument is considered as the path to the .csv file.
    las_file_path : str
        The path to the .las file to be created.
    csv_file_provided : str, optional
        The path to the .csv file to be converted to .las file.
        If None, the csv argument is used instead.
        The default is None.
    """
    # Check if the csv_file_provided argument is provided

    if csv_file_provided:
        df = pd.read_csv(csv, sep=',')
    else:
        df = csv

    standard_columns_with_data_types = {
    'X': 'int32',
    'Y': 'int32',
    'Z': 'int32',
    'intensity': 'uint16',
    'return_number': 'uint8',
    'number_of_returns': 'uint8',
    'synthetic': 'uint8',
    'key_point': 'uint8',
    'withheld': 'uint8',
    'overlap': 'uint8',  # Data type not specified in the provided formats
    'scanner_channel': 'uint8',  # Data type not specified in the provided formats
    'scan_direction_flag': 'uint8',
    'edge_of_flight_line': 'uint8',
    'classification': 'uint8',
    'user_data': 'uint8',
    'scan_angle': 'uint16',  # Data type not specified in the provided formats
    # 'scan_angle_rank': 'int8',
    'point_source_id': 'uint16',
    'gps_time': 'float64',
    'red': 'uint16',  # Data type not specified in the provided formats
    'green': 'uint16',  # Data type not specified in the provided formats
    'blue': 'uint16'   # Data type not specified in the provided formats
    }

    extended_columns_with_data_types = {
        'Amplitude': 'float64',
        'Pulse_width': 'float64',
        'Reflectance': 'float64',
        'Deviation': 'int32',
        'PredSemantic' : 'uint8',
        'PredInstance' : 'uint16',
    }

    # Standardize column names to match LAS format
    df.rename(columns={'x': 'X', 'y': 'Y', 'z': 'Z'}, inplace=True)

    # Calculate scales and offsets for your point data
    scale = [0.001, 0.001, 0.001]  # Example scale factors
    offset = [df['X'].min(), df['Y'].min(), df['Z'].min()]  # Minimum values as offsets

    # Create a new .las file with correct header information
    las_header = laspy.LasHeader(point_format=6, version="1.4")
    las_header.scale = scale
    las_header.offset = offset

    # Bounds
    min_bounds = offset  # already calculated as minimums
    max_bounds = [df['X'].max(), df['Y'].max(), df['Z'].max()]  # Maximum values
    las_header.min = min_bounds
    las_header.max = max_bounds

    # if there is scan_angle_rank column map it to scan_angle
    if 'scan_angle_rank' in df.columns:
        df.rename(columns={'scan_angle_rank': 'scan_angle'}, inplace=True)

    # check if the columns in the dataframe match the standard columns and make a list of the columns that match
    standard_columns = list(las_header.point_format.dimension_names)
    columns_which_match = [column for column in standard_columns if column in df.columns]

    # remove X, Y and Z from the list
    columns_which_match.remove('X')
    columns_which_match.remove('Y')
    columns_which_match.remove('Z')

    # get extra columns as columns which don't match
    extra_columns = [column for column in df.columns if column not in standard_columns]

    # check if the extra columns exist in the extended_columns_with_data_types dictionary
    # for those which do not exist, take the data type from data frame
    for column in extra_columns:
        if column not in extended_columns_with_data_types.keys():
            extended_columns_with_data_types[column] = df[column].dtype

    # add extra columns to the las file with the correct data types
    for column in extra_columns:
        las_header.add_extra_dim(laspy.ExtraBytesParams(name=column, type=extended_columns_with_data_types[column]))

    # create a new las file with the correct header information
    las_file = laspy.LasData(las_header)

    # Assigning the scaled and offset data
    las_file.X = (df['X'] - offset[0]) / scale[0]
    las_file.Y = (df['Y'] - offset[1]) / scale[1]
    las_file.Z = (df['Z'] - offset[2]) / scale[2]

    # add standard columns to the las file with the correct data types
    for column in columns_which_match:
        las_file[column] = df[column].astype(standard_columns_with_data_types[column])

    # add extra columns to the las file with the correct data types
    for column in extra_columns:
        las_file[column] = df[column].astype(extended_columns_with_data_types[column])

    # Write the file to disk
    if do_compress:
        output_file_path = output_file_path.replace('.las', '.laz')
        las_file.write(output_file_path, do_compress=True)
    else:
        las_file.write(output_file_path, do_compress=False)

    if verbose:
        if csv_file_provided:
            print('The input file was is {}'.format(csv))
        
        if do_compress:
            print('File saved as: {}'.format(output_file_path.replace('.las', '.laz')))
        else:
            print('File saved as: {}'.format(output_file_path))



