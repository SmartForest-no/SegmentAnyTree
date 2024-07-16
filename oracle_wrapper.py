import os
import shutil

PATH_DATA = '/home/datascience' # TODO: change this to the path taken from the config file
DEBUG_MODE = False

# read the configuration file
def run_oracle_wrapper():

    if DEBUG_MODE:
        # this is mapped in the docker run
        data_location = "/home/data_bucket"

    else:
        # get the input and output locations from the environment variables
        data_location = os.environ['OBJ_INPUT_LOCATION']

        # remap the input and output locations
        data_location = data_location.replace("@axqlz2potslu", "").replace("oci://", "/mnt/")

    # create the output folder if it does not exist
    os.makedirs(PATH_DATA, exist_ok=True)

    # copy files from input_location to the input folder
    shutil.copytree(data_location, os.path.join(PATH_DATA, 'data'))

if __name__ == '__main__':
    run_oracle_wrapper()