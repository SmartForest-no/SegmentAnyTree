import laspy
import numpy as np
from plyfile import PlyData, PlyElement
from pathlib import Path
import csv
import random

#This file converts the las files into ply files and does some preprocessing along the way (namely it can remove points
#belonging to certain classification labels and it does a specific mapping from classification to semantic segmentation).
#It also creates a train - validation - test split from a train - test split given in data_split_metadata.csv

def print_las_info(las_file_path):
    print("las info of file: " + str(las_file_path))
    las = laspy.read(las_file_path)
    print(".las data:", las)

    for dimension in las.point_format.dimensions:
        print("Dimension name: ", dimension.name)
        print("Dimension type: ", dimension.dtype)
        print("Unique values: ", np.unique(las[dimension.name]))


def get_coord(value, scale):
    return value*scale


def las_to_ply(las_file_path, ply_file_path, remove_ground=False, remove_lowveg=False, remove_outpoints=False):
    """
    conversion from .las to .ply data type
    :param las_file_path (str): path to .las file
    :param ply_file_path (str): path where .ply file should be saved without incorporating folder name that depends on what points we have removed
    :param remove_ground (bool): whether ground points should be removed
    :param remove_lowveg (bool): whether low vegetation points should be removed
    :param remove_outpoints (bool): whether outpoints should be removed
    :return: (str) path where .ply file has actually been saved with incorporating folder name that depends on what points we have removed
    """
    las = laspy.read(las_file_path)

    scale_x, scale_y, scale_z = las.header.scale
    #we ignore the offset given in las.header.offset for each dimension, as relative position between las data files doesn't matter
    X, Y, Z = get_coord(las.X, scale_x), get_coord(las.Y, scale_y), get_coord(las.Z, scale_z)

    len = las.X.size

    #remove points from certain classification labels
    print("{} percent of the points are ground points.".format((np.count_nonzero(las.classification==2)/len)))
    print("{} percent of the points are low vegetation points.".format((np.count_nonzero(las.classification == 1) / len)))
    print("{} percent of the points are outpoints.".format((np.count_nonzero(las.classification == 3) / len)))

    foldername_addition = ""
    points_to_keep = np.full(len, True)
    if (not remove_ground) and (not remove_lowveg) and (not remove_outpoints):
        foldername_addition += "_full"
    if remove_ground: #ground removed
        points_to_keep = np.logical_and(points_to_keep, las.classification!=2)
        foldername_addition += "_groundrm"
    if remove_lowveg: #low vegetation removed
        points_to_keep = np.logical_and(points_to_keep, las.classification!=1)
        foldername_addition += "_lowvegrm"
    if remove_outpoints: #outpoints removed
        points_to_keep = np.logical_and(points_to_keep, las.classification!=3)
        foldername_addition += "_outpointsrm"

    classification_new = las.classification[points_to_keep] #classification without points that should be removed
    len_new = classification_new.shape[0]
    data_struct = np.zeros(len_new, dtype=np.dtype([('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('intensity', 'f4'), ('semantic_seg', 'f4'), ('treeID', 'f4')]))
    data_struct['x'] = X.astype('f4')[points_to_keep]
    data_struct['y'] = Y.astype('f4')[points_to_keep]
    data_struct['z'] = Z.astype('f4')[points_to_keep]
    data_struct['intensity'] = las.intensity.astype('f4')[points_to_keep]
    data_struct['treeID'] = las.treeID.astype('f4')[points_to_keep]
    #data_struct['treeSP'] = las.treeSP.astype('f4')[points_to_keep]

    #adds the string foldername_addition to the data folder name to make different data folders depending on what points we have removed
    path_to_region = ply_file_path.parents[4].joinpath(ply_file_path.parts[-5]+foldername_addition).joinpath(*ply_file_path.parts[-4:-1])
    if not path_to_region.is_dir():
        path_to_region.mkdir(parents=True, exist_ok=True)
    ply_file_path_datanamechange = path_to_region.joinpath(ply_file_path.name)

    #mapping from classification to semantic segmentation labels (0: unclassified, 1: non-tree, 2: tree)
    sem_seg = np.full(classification_new.shape, 20.0, dtype='f4')
    sem_seg[classification_new==0] = 0.0 #unclassified -> unclassified
    sem_seg[classification_new==1] = 1.0 #lowveg -> non-tree
    if remove_lowveg and sem_seg[classification_new==1].any():
        print("removing low vegetation points not successful")
    sem_seg[classification_new==2] = 1.0 #ground -> non-tree
    if remove_ground and sem_seg[classification_new==2].any():
        print("removing ground points not successful")
    sem_seg[classification_new==3] = 0.0 #outpoints (trees outside of the measured/annotated plots) -> unclassified
    if remove_outpoints and sem_seg[classification_new==3].any():
        print("removing outpoints points not successful")
    sem_seg[classification_new==4] = 2.0 #stem -> tree
    sem_seg[classification_new==5] = 2.0 #live-branches -> tree
    sem_seg[classification_new==6] = 2.0 #branches -> tree
    data_struct['semantic_seg'] = sem_seg

    if (sem_seg==20.0).any():
        print("conversion not successful")

    el = PlyElement.describe(data_struct, 'vertex', comments=['Created manually from las files.'])
    PlyData([el], byte_order='<').write(ply_file_path_datanamechange)
    return ply_file_path_datanamechange

def train_val_test_split(train_test_split_path):
    """
    create a train - validation - test split from a train - test split by declaring some (original) train files as validation files
    :param train_test_split_path (str): path to data_split_metadata.csv which provides train test split
    :return:
            splitlist (list): list of all data files' relative paths, i.e. [CULS/plot_1_annotated.las, ...],
            forest_region_list (list): in which subfolder/forest region (e.g. CULS) the datafiles are,
            split_list (list): #whether the datafiels are used as train, validation or test file

    """
    csv_file = open(train_test_split_path)
    csv_reader = csv.reader(csv_file, delimiter=',')

    rel_path_list = [] #list of all data files' relative paths, i.e. [CULS/plot_1_annotated.las, ...]
    forest_region_list = [] #in which subfolder/forest region (e.g. CULS) the datafiles are
    split_list = [] #whether the datafiels are used as train, val or test file

    num_train=0
    num_test=0
    line_count = 0
    for row in csv_reader:#for each datafile
        if line_count != 0:
            if row[2]=="train":
                num_train += 1
            elif row[2]=="test":
                num_test += 1
            else:
                print("Problem: split is neither train nor test.")
        line_count += 1

    # sample randomly, but fixed (fixed because we have fixed random.seed(42) in the beginning of __main__)
    train_val_split = random.sample(range(num_train), int(0.25*num_train))
    train_val_counter = 0
    csv_file = open(train_test_split_path)
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count != 0:
            rel_path_list.append(row[0])
            forest_region_list.append(row[1])
            if row[2] == "test":
                split_list.append("test")
            elif row[2]=="train":
                if train_val_counter in train_val_split:
                    split_list.append("val")
                else:
                    split_list.append("train")
                train_val_counter +=1
            else:
                print("Problem: split is neither train nor test.")
        line_count += 1

    return rel_path_list, forest_region_list, split_list


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    random.seed(42) #set seed so that validation set gets chosen randomly, but fixed from within the files annotated as
    #"train" by Stefano's train test split in data_split_metadata.csv

    #las_to_ply('/local/home/vfrawa/Documents/data/NIBIO2/plot16_annotated.las', '/local/home/vfrawa/Documents/data/NIBIO2/plot16_annotated_noground_nolowveg_nooutp.ply', True, True, True)

    #TO ADAPT: path to las data folder (data from the different regions (CULS, etc.) and data_split_metadata.csv must be in this folder)
    las_data_basepath = Path('/local/home/vfrawa/Documents/data')
    train_test_split_path = str(las_data_basepath) + '/data_split_metadata.csv'
    rel_path_list, forest_region_list, split_list = train_val_test_split(train_test_split_path) #creates train-val-test split from train-test split
    #TO ADAPT: path where the code folder "OutdoorPanopticSeg_V2" is located
    code_basepath = '/local/home/vfrawa/Documents/code'
    codes_data_basepath = Path(code_basepath + '/OutdoorPanopticSeg_V2/data/treeinsfused/raw') #this is where the ply files should be located so that the code accesses them
    #TO ADAPT: choose whether points labelled as ground, low vegetation and outpoints should be removed entirely or not
    remove_ground = False
    remove_lowveg = False
    remove_outpoints = False

    testpath_list = []
    for i in range(len(rel_path_list)): #per .las data file
        las_file_path = las_data_basepath.joinpath(rel_path_list[i])
        print(str(las_file_path))
        # print_las_info(las_file_path)
        ply_file_path = codes_data_basepath.joinpath(las_file_path.parts[-2]).joinpath(forest_region_list[i] + "_" + las_file_path.stem + "_" + split_list[i] + ".ply")
        ply_file_path_datanamechange = las_to_ply(las_file_path, ply_file_path, remove_ground, remove_lowveg, remove_outpoints)
        if split_list[i]=="test":
            testpath_list.append(str(ply_file_path_datanamechange))

    print(testpath_list) #list of paths of all files used as test files -> can be used for fold in conf/eval.yaml

