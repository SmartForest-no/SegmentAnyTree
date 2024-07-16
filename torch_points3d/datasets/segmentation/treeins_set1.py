import os
import os.path as osp
from itertools import repeat, product
import numpy as np
import h5py
import torch
import random
import glob
from plyfile import PlyData, PlyElement
from torch_geometric.data import InMemoryDataset, Data, extract_zip, Dataset
from torch_geometric.data.dataset import files_exist
from torch_geometric.data import DataLoader
import torch_geometric.transforms as T
import logging
from sklearn.neighbors import NearestNeighbors, KDTree
from tqdm.auto import tqdm as tq
import csv
import pandas as pd
import pickle
import gdown
import shutil
# PLY reader
from torch_points3d.modules.KPConv.plyutils import read_ply
from torch_points3d.datasets.samplers import BalancedRandomSampler
import torch_points3d.core.data_transform as cT
from torch_points3d.datasets.base_dataset import BaseDataset

DIR = os.path.dirname(os.path.realpath(__file__))
log = logging.getLogger(__name__)

# @Treeins: a lot of code copied from torch_points3d/datasets/segmentation/npm3d.py, for changes see @Treeins


#@Treeins: two semantic segmentation classes: non-tree and tree
Treeins_NUM_CLASSES = 5
INV_OBJECT_LABEL = {
    0: "low_vegetation",
    1: "ground",
    2: "stem_points",
    3: "live_branches",
    4: "branches",
}

OBJECT_COLOR = np.asarray(
    [
        [233, 229, 107],  # 'low_vegetation' .-> .yellow
        [95, 156, 196],  # 'ground' .-> . blue
        [179, 116, 81],  # 'stem_points'  ->  brown
        [241, 149, 131],  # 'live_branches'  ->  salmon
        [81, 163, 148],  # 'branches'  ->  bluegreen
        [0, 0, 0],  # unlabelled .->. black
    ]
)


OBJECT_LABEL = {name: i for i, name in INV_OBJECT_LABEL.items()}

################################### UTILS #######################################
def object_name_to_label(object_class):
    """convert from object name in NPPM3D to an int"""
    object_label = OBJECT_LABEL.get(object_class, OBJECT_LABEL["unclassified"])
    return object_label

def read_treeins_format(train_file, add_input_features=[], label_out=True, verbose=False, debug=False):
    """extract data from a treeins file"""
    raw_path = train_file
    data = read_ply(raw_path)
    # xyz = np.vstack((data['x'], data['y'], data['z'])).astype(np.float32).T

    # check if data['X'] exist if does exist extract data from data['X'] instead of data['x']
    if 'X' in data.dtype.names:
        xyz = np.vstack((data['X'], data['Y'], data['Z'])).astype(np.float32).T
    else:
        xyz = np.vstack((data['x'], data['y'], data['z'])).astype(np.float32).T

    if not label_out:
        return xyz
    # # @Treeins: read in semantic segmentation labels (0: unclassified, 1: non-tree, 2: tree) and change them to
    # # (-1: unclassified, 0, non-tree, 1: tree) because unclassified must have label -1
    # semantic_labels = data['semantic_seg'].astype(np.int64)-1
    # #semantic_labels = data['scalar_semantic_label'].astype(np.int64)  #3d forest test
    # # @Treeins: The attribute treeID tells us to which tree a point belongs, hence we use it as instance labels
    # instance_labels = data['treeID'].astype(np.int64)+1
    # #instance_labels = data['scalar_instance_label'].astype(np.int64)  #3d forest test

    # set all semantic labels to 0 if ther are no semantic labels in the data
    if 'semantic_seg' in data.dtype.names:
        semantic_labels = data['semantic_seg'].astype(np.int64)-1
    else:
        semantic_labels = np.zeros((xyz.shape[0],), dtype=np.int64)
    
    # @Treeins: The attribute treeID tells us to which tree a point belongs, hence we use it as instance labels
    # instance_labels = data['treeID'].astype(np.int64)+1

    # set all instance labels to 0 if ther are no instance labels in the data
    if 'treeID' in data.dtype.names:
        instance_labels = data['treeID'].astype(np.int64)+1
    else:
        instance_labels = np.zeros((xyz.shape[0],), dtype=np.int64)


    if add_input_features == [] or add_input_features ==None:
        return (
            torch.from_numpy(xyz),
            torch.from_numpy(semantic_labels),
            torch.from_numpy(instance_labels),
        )
    else:
        Sum = data['Sum'].astype(np.float32)
        Omnivariance = data['Omnivariance'].astype(np.float32)
        Eigenentropy = data['Eigenentropy'].astype(np.float32)
        Anisotropy = data['Anisotropy'].astype(np.float32)
        planarity = data['planarity'].astype(np.float32)
        linearity = data['linearity'].astype(np.float32)
        Surface_var = data['Surface_var'].astype(np.float32)
        scattering = data['scattering'].astype(np.float32)
        verticality = data['verticality'].astype(np.float32)
        Verticality2 = data['Verticality2'].astype(np.float32)
        moment1_1 = data['moment1_1'].astype(np.float32)
        moment1_2 = data['moment1_2'].astype(np.float32)
        moment2_1 = data['moment2_1'].astype(np.float32)
        moment2_2 = data['moment2_2'].astype(np.float32)
        intensity = data['intensity'].astype(np.float32)
        return_num = data['return_num'].astype(np.float32)
        scan_angle_rank = data['scan_angle_rank'].astype(np.float32)
        
        #print(np.unique(instance_labels))
        return (
            torch.from_numpy(xyz),
            torch.from_numpy(semantic_labels),
            torch.from_numpy(instance_labels),
            torch.from_numpy(Sum),
            torch.from_numpy(Omnivariance),
            torch.from_numpy(Eigenentropy),
            torch.from_numpy(Anisotropy),
            torch.from_numpy(planarity),
            torch.from_numpy(linearity),
            torch.from_numpy(Surface_var),
            torch.from_numpy(scattering),
            torch.from_numpy(verticality),
            torch.from_numpy(Verticality2),
            torch.from_numpy(moment1_1),
            torch.from_numpy(moment1_2),
            torch.from_numpy(moment2_1),
            torch.from_numpy(moment2_2),
            torch.from_numpy(intensity),
            torch.from_numpy(return_num),
            torch.from_numpy(scan_angle_rank)
        )
    


def to_ply(pos, label, file):
    assert len(label.shape) == 1
    assert pos.shape[0] == label.shape[0]
    pos = np.asarray(pos)
    colors = OBJECT_COLOR[np.asarray(label)]
    ply_array = np.ones(
        pos.shape[0], dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    )
    ply_array["x"] = pos[:, 0]
    ply_array["y"] = pos[:, 1]
    ply_array["z"] = pos[:, 2]
    ply_array["red"] = colors[:, 0]
    ply_array["green"] = colors[:, 1]
    ply_array["blue"] = colors[:, 2]
    el = PlyElement.describe(ply_array, "Treeins")
    PlyData([el], byte_order=">").write(file)
    
def to_eval_ply(pos, pre_label, gt, file):
    assert len(pre_label.shape) == 1
    assert len(gt.shape) == 1
    assert pos.shape[0] == pre_label.shape[0]
    assert pos.shape[0] == gt.shape[0]
    pos = np.asarray(pos)
    ply_array = np.ones(
        pos.shape[0], dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("preds", "u16"), ("gt", "u16")]
    )
    ply_array["x"] = pos[:, 0]
    ply_array["y"] = pos[:, 1]
    ply_array["z"] = pos[:, 2]
    ply_array["preds"] = np.asarray(pre_label)
    ply_array["gt"] = np.asarray(gt)
    PlyData.write(file)
    
def to_ins_ply(pos, label, file):
    assert len(label.shape) == 1
    assert pos.shape[0] == label.shape[0]
    pos = np.asarray(pos)
    max_instance = np.max(np.asarray(label)).astype(np.int32)+1
    rd_colors = np.random.randint(255, size=(max_instance,3), dtype=np.uint8)
    colors = rd_colors[np.asarray(label)]
    ply_array = np.ones(
        pos.shape[0], dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    )
    ply_array["x"] = pos[:, 0]
    ply_array["y"] = pos[:, 1]
    ply_array["z"] = pos[:, 2]
    ply_array["red"] = colors[:, 0]
    ply_array["green"] = colors[:, 1]
    ply_array["blue"] = colors[:, 2]
    PlyData.write(file)



################################### Used for fused NPM3D radius sphere ###################################


class TreeinsOriginalFused(InMemoryDataset):
    """ Original Treeins dataset. Each area is loaded individually and can be processed using a pre_collate transform.
    This transform can be used for example to fuse the area into a single space and split it into 
    spheres or smaller regions. If no fusion is applied, each element in the dataset is a single area by default.

    Parameters
    ----------
    root: str
        path to the directory where the data will be saved
    forest regions: list of str
        @Treeins: specifies from which forest region(s) data files should be used for training and validation, [] means taking data files from all forest regions
    test_area: list
        @Treeins: during training/running train.py: [] means taking all specified test files (i.e. all files with name ending in "test" for testing, otherwise list of ints indexing into which of these specified test files to use
        @Treeins: during evaluation/running eval.py: paths to files to test model on
    split: str
        can be one of train, trainval, val or test
    pre_collate_transform:
        Transforms to be applied before the data is assembled into samples (apply fusing here for example)
    keep_instance: bool
        set to True if you wish to keep instance data
    pre_transform
    transform
    pre_filter
    """

    num_classes = Treeins_NUM_CLASSES

    def __init__(
        self,
        root,
        grid_size,
        forest_regions=[], #@Treeins
        add_input_features=[],
        test_area=[],
        split="train",
        transform=None,
        pre_transform=None,
        pre_collate_transform=None,
        pre_filter=None,
        keep_instance=False,
        verbose=False,
        debug=False,
    ):
        self.transform = transform
        self.pre_collate_transform = pre_collate_transform
        self.forest_regions = forest_regions
        self.add_input_features = add_input_features
        self.test_area = test_area
        self.keep_instance = keep_instance
        self.verbose = verbose
        self.debug = debug
        self._split = split
        self.grid_size = grid_size
     
        super(TreeinsOriginalFused, self).__init__(root, transform, pre_transform, pre_filter)
        # @Treeins: case for training/when running train.py
        if len(self.test_area)==0 or isinstance(self.test_area[0], int):
            
            if split == "train":
                path = self.processed_paths[0]
            elif split == "val":
                path = self.processed_paths[1]
            elif split == "test":
                path = self.processed_paths[2]
            elif split == "trainval":
                path = self.processed_paths[3]
            else:
                raise ValueError((f"Split {split} found, but expected either " "train, val, trainval or test"))
            self._load_data(path)

            if split == "test":
                # @Treeins: load all files specified in the test_area list
                if self.test_area == []:
                    "PROBLEM: In branch split==test even though test_area was not initialized yet"
                self.raw_test_data = [torch.load(self.raw_areas_paths[test_area_i]) for test_area_i in self.test_area]

        # @Treeins: case for evaluation/when running eval.py
        else:
            # @Treeins: process all files at the paths given in test_area list for evaluation
            self.process_test(test_area)
            path = self.processed_path
            self._load_data(path)
            self.raw_test_data = [torch.load(raw_area_path) for raw_area_path in self.raw_areas_paths]

    @property
    def center_labels(self):
        if hasattr(self.data, "center_label"):
            return self.data.center_label
        else:
            return None

    @property
    def raw_file_names(self):
        """returns list of paths to the .ply raw data files we use"""
        if self.forest_regions == [] or self.forest_regions ==None:  # @Treeins: get all data file names in folder self.raw_dir
            return glob.glob(self.raw_dir + '/**/*.ply', recursive=True)
        else: #@Treeins: get data file names coming from the forest region(s) in self.forest_regions within the folder self.raw_dir
            raw_files_list = []
            for region in self.forest_regions:
                raw_files_list += glob.glob(self.raw_dir + "/" + region + "/*.ply", recursive=False)
            return raw_files_list

    @property
    def processed_dir(self):
        """returns path to the directory which contains the processed data files,
               e.g. path/to/project/OutdoorPanopticSeg_V2/data/treeinsfused/processed_0.2"""
        processed_dir_prefix = 'processed_' + str(self.grid_size) #add grid size to the processed directory's name
        if self.forest_regions != [] and self.forest_regions != None: #@Treeins: add forest regions to the processed directory's name (if not all forest regions are used)
            processed_dir_prefix += "_" + str(self.forest_regions)

        # @Treeins: case for training/when running train.py
        if len(self.test_area)==0 or isinstance(self.test_area[0], int):
            return osp.join(self.root, processed_dir_prefix)
        # @Treeins: case for evaluation/when running eval.py
        else:
            return osp.join(self.root, processed_dir_prefix+'_test')

    @property
    def pre_processed_path(self):
        pre_processed_file_names = "preprocessed.pt"
        return os.path.join(self.processed_dir, pre_processed_file_names)

    @property
    def raw_areas_paths(self):
        """returns list of paths to .pt files saved in self.processed_dir and created from the .ply raw data files"""
        # @Treeins: case for training/when running train.py
        if len(self.test_area)==0 or isinstance(self.test_area[0], int):
            if not hasattr(self, "num_datafiles"):
                self.num_datafiles = len(self.raw_file_names)
            return [os.path.join(self.processed_dir, "raw_area_%i.pt" % i) for i in range(self.num_datafiles)]
        # @Treeins: case for evaluation/when running eval.py
        else:
            return [os.path.join(self.processed_dir, 'raw_area_'+os.path.split(f)[-1].split('.')[0]+'.pt') for f in self.test_area]


    @property
    def processed_file_names(self):
        """return list of paths to all kinds of files in the processed directory"""
        # @Treeins: case for training/when running train.py
        if len(self.test_area)==0 or isinstance(self.test_area[0], int):
            return (
            ["{}.pt".format(s) for s in ["train", "val", "test", "trainval"]]
            + self.raw_areas_paths
            + [self.pre_processed_path]
        )
        # @Treeins: case for evaluation/when running eval.py
        else:
            return ['processed_'+os.path.split(f)[-1].split('.')[0]+'.pt' for f in self.test_area]

    @property
    def raw_test_data(self):
        return self._raw_test_data

    @raw_test_data.setter
    def raw_test_data(self, value):
        self._raw_test_data = value

    @property
    def num_features(self):
        feats = self[0].x
        if feats is not None:
            return feats.shape[-1]
        return 0
    #def download(self):
    #    super().download()

    def process(self):
        """Takes the given .ply files, processes them and saves the newly created files in self.processed_dir.
        This method is used during training/running train.py."""

        if not os.path.exists(self.pre_processed_path):# @Treeins: if we haven't already processed the raw .ply data files in a previous run with the same grid_size and forest_regions
            input_ply_files = self.raw_file_names
            #list of forest names
            list_forest_name = os.listdir(self.raw_dir)
            # Gather data per area
            data_list = [[] for _ in range(len(input_ply_files))] #@Treeins: list of lists which each contains one .ply data file
            for area_num, file_path in enumerate(input_ply_files):
                area_name = os.path.split(file_path)[-1]
                #get forest region name
                a = os.path.dirname(file_path) #file path
                forest_name = os.path.basename(a) #file forder name
                if self.add_input_features == [] or self.add_input_features ==None:
                    xyz, semantic_labels, instance_labels = read_treeins_format(
                    file_path, self.add_input_features, label_out=True, verbose=self.verbose, debug=self.debug
                )
                else:
                    xyz, semantic_labels, instance_labels, Sum, Omnivariance,Eigenentropy, Anisotropy, planarity, linearity, Surface_var, scattering, verticality, Verticality2, moment1_1, moment1_2, moment2_1, moment2_2, intensity, return_num, scan_angle_rank= read_treeins_format(
                    file_path, self.add_input_features, label_out=True, verbose=self.verbose, debug=self.debug
                )

                data = Data(pos=xyz, y=semantic_labels)
                data.forest_name = 0 #list_forest_name.index(forest_name)
                data.validation_set = False
                data.test_set = False
                #@Treeins: list of lists which each contains one .ply data file
                if area_name[-7:-4]=="val":
                    data.validation_set = True
                #@Treeins:  if "test" at end of area_name, i.e. at end of .ply file name, we put data file into test set
                elif area_name[-8:-4]=="test":
                    data.test_set = True
                    self.test_area.append(area_num)

                if self.keep_instance:
                    data.instance_labels = instance_labels
                    
                if self.add_input_features != [] and self.add_input_features !=None:
                    data.Sum = Sum
                    data.Omnivariance = Omnivariance
                    data.Eigenentropy = Eigenentropy
                    data.Anisotropy = Anisotropy
                    data.planarity = planarity
                    data.linearity = linearity
                    data.Surface_var = Surface_var
                    data.scattering = scattering
                    data.verticality = verticality
                    data.Verticality2 = Verticality2
                    data.moment1_1 = moment1_1
                    data.moment1_2 = moment1_2
                    data.moment2_1 = moment2_1
                    data.moment2_2 = moment2_2
                    data.intensity = intensity
                    data.return_num = return_num
                    data.scan_angle_rank = scan_angle_rank

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                print("area_num:")
                print(area_num)
                print("data:")  #Data(pos=[30033430, 3], validation_set=False, y=[30033430])
                print(data)
                data_list[area_num].append(data)
            print("data_list")
            print(data_list)
            raw_areas = cT.PointCloudFusion()(data_list)
            print("raw_areas") # [Batch(instance_labels=[590811], pos=[590811, 3], test_set=[1], validation_set=[1], y=[590811]), Batch(instance_labels=[869420], pos=[869420, 3], test_set=[1], validation_set=[1], y=[869420]), Batch(instance_labels=[797789], pos=[797789, 3], test_set=[1], validation_set=[1], y=[797789]), Batch(instance_labels=[643246], pos=[643246, 3], test_set=[1], validation_set=[1], y=[643246]), Batch(instance_labels=[826224], pos=[826224, 3], test_set=[1], validation_set=[1], y=[826224]), Batch(instance_labels=[689970], pos=[689970, 3], test_set=[1], validation_set=[1], y=[689970]), Batch(instance_labels=[790505], pos=[790505, 3], test_set=[1], validation_set=[1], y=[790505]), Batch(instance_labels=[747886], pos=[747886, 3], test_set=[1], validation_set=[1], y=[747886]), Batch(instance_labels=[774428], pos=[774428, 3], test_set=[1], validation_set=[1], y=[774428]), Batch(instance_labels=[657704], pos=[657704, 3], test_set=[1], validation_set=[1], y=[657704]), Batch(instance_labels=[712460], pos=[712460, 3], test_set=[1], validation_set=[1], y=[712460]), Batch(instance_labels=[902134], pos=[902134, 3], test_set=[1], validation_set=[1], y=[902134]), Batch(instance_labels=[548212], pos=[548212, 3], test_set=[1], validation_set=[1], y=[548212]), Batch(instance_labels=[361483], pos=[361483, 3], test_set=[1], validation_set=[1], y=[361483]), Batch(instance_labels=[768537], pos=[768537, 3], test_set=[1], validation_set=[1], y=[768537]), Batch(instance_labels=[619576], pos=[619576, 3], test_set=[1], validation_set=[1], y=[619576]), Batch(instance_labels=[600336], pos=[600336, 3], test_set=[1], validation_set=[1], y=[600336]), Batch(instance_labels=[569716], pos=[569716, 3], test_set=[1], validation_set=[1], y=[569716]), Batch(instance_labels=[825402], pos=[825402, 3], test_set=[1], validation_set=[1], y=[825402]), Batch(instance_labels=[636120], pos=[636120, 3], test_set=[1], validation_set=[1], y=[636120]), Batch(instance_labels=[929771], pos=[929771, 3], test_set=[1], validation_set=[1], y=[929771]), Batch(instance_labels=[710364], pos=[710364, 3], test_set=[1], validation_set=[1], y=[710364]), Batch(instance_labels=[747211], pos=[747211, 3], test_set=[1], validation_set=[1], y=[747211]), Batch(instance_labels=[822989], pos=[822989, 3], test_set=[1], validation_set=[1], y=[822989]), Batch(instance_labels=[488489], pos=[488489, 3], test_set=[1], validation_set=[1], y=[488489]), Batch(instance_labels=[739967], pos=[739967, 3], test_set=[1], validation_set=[1], y=[739967]), Batch(instance_labels=[816221], pos=[816221, 3], test_set=[1], validation_set=[1], y=[816221]), Batch(instance_labels=[684834], pos=[684834, 3], test_set=[1], validation_set=[1], y=[684834]), Batch(instance_labels=[603103], pos=[603103, 3], test_set=[1], validation_set=[1], y=[603103]), Batch(instance_labels=[832181], pos=[832181, 3], test_set=[1], validation_set=[1], y=[832181]), Batch(instance_labels=[727537], pos=[727537, 3], test_set=[1], validation_set=[1], y=[727537]), Batch(instance_labels=[646622], pos=[646622, 3], test_set=[1], validation_set=[1], y=[646622]), Batch(instance_labels=[657490], pos=[657490, 3], test_set=[1], validation_set=[1], y=[657490]), Batch(instance_labels=[583478], pos=[583478, 3], test_set=[1], validation_set=[1], y=[583478]), Batch(instance_labels=[877991], pos=[877991, 3], test_set=[1], validation_set=[1], y=[877991]), Batch(instance_labels=[733789], pos=[733789, 3], test_set=[1], validation_set=[1], y=[733789]), Batch(instance_labels=[794052], pos=[794052, 3], test_set=[1], validation_set=[1], y=[794052]), Batch(instance_labels=[625610], pos=[625610, 3], test_set=[1], validation_set=[1], y=[625610]), Batch(instance_labels=[818370], pos=[818370, 3], test_set=[1], validation_set=[1], y=[818370]), Batch(instance_labels=[627210], pos=[627210, 3], test_set=[1], validation_set=[1], y=[627210]), Batch(instance_labels=[579792], pos=[579792, 3], test_set=[1], validation_set=[1], y=[579792]), Batch(instance_labels=[797122], pos=[797122, 3], test_set=[1], validation_set=[1], y=[797122]), Batch(instance_labels=[856790], pos=[856790, 3], test_set=[1], validation_set=[1], y=[856790]), Batch(instance_labels=[397940], pos=[397940, 3], test_set=[1], validation_set=[1], y=[397940]), Batch(instance_labels=[698928], pos=[698928, 3], test_set=[1], validation_set=[1], y=[698928]), Batch(instance_labels=[771875], pos=[771875, 3], test_set=[1], validation_set=[1], y=[771875]), Batch(instance_labels=[423204], pos=[423204, 3], test_set=[1], validation_set=[1], y=[423204]), Batch(instance_labels=[571117], pos=[571117, 3], test_set=[1], validation_set=[1], y=[571117]), Batch(instance_labels=[842309], pos=[842309, 3], test_set=[1], validation_set=[1], y=[842309]), Batch(instance_labels=[813269], pos=[813269, 3], test_set=[1], validation_set=[1], y=[813269]), Batch(instance_labels=[3084916], pos=[3084916, 3], test_set=[1], validation_set=[1], y=[3084916]), Batch(instance_labels=[3946098], pos=[3946098, 3], test_set=[1], validation_set=[1], y=[3946098]), Batch(instance_labels=[1816672], pos=[1816672, 3], test_set=[1], validation_set=[1], y=[1816672]), Batch(instance_labels=[2280049], pos=[2280049, 3], test_set=[1], validation_set=[1], y=[2280049]), Batch(instance_labels=[7568844], pos=[7568844, 3], test_set=[1], validation_set=[1], y=[7568844]), Batch(instance_labels=[2977537], pos=[2977537, 3], test_set=[1], validation_set=[1], y=[2977537]), Batch(instance_labels=[1884678], pos=[1884678, 3], test_set=[1], validation_set=[1], y=[1884678]), Batch(instance_labels=[3589254], pos=[3589254, 3], test_set=[1], validation_set=[1], y=[3589254]), Batch(instance_labels=[3184258], pos=[3184258, 3], test_set=[1], validation_set=[1], y=[3184258]), Batch(instance_labels=[3311297], pos=[3311297, 3], test_set=[1], validation_set=[1], y=[3311297]), Batch(instance_labels=[4943818], pos=[4943818, 3], test_set=[1], validation_set=[1], y=[4943818]), Batch(instance_labels=[7866181], pos=[7866181, 3], test_set=[1], validation_set=[1], y=[7866181]), Batch(instance_labels=[6374998], pos=[6374998, 3], test_set=[1], validation_set=[1], y=[6374998]), Batch(instance_labels=[5460307], pos=[5460307, 3], test_set=[1], validation_set=[1], y=[5460307]), Batch(instance_labels=[5302422], pos=[5302422, 3], test_set=[1], validation_set=[1], y=[5302422]), Batch(instance_labels=[6163377], pos=[6163377, 3], test_set=[1], validation_set=[1], y=[6163377]), Batch(instance_labels=[5269942], pos=[5269942, 3], test_set=[1], validation_set=[1], y=[5269942]), Batch(instance_labels=[5027570], pos=[5027570, 3], test_set=[1], validation_set=[1], y=[5027570]), Batch(instance_labels=[6705567], pos=[6705567, 3], test_set=[1], validation_set=[1], y=[6705567]), Batch(instance_labels=[4066375], pos=[4066375, 3], test_set=[1], validation_set=[1], y=[4066375]), Batch(instance_labels=[4551358], pos=[4551358, 3], test_set=[1], validation_set=[1], y=[4551358]), Batch(instance_labels=[5223631], pos=[5223631, 3], test_set=[1], validation_set=[1], y=[5223631]), Batch(instance_labels=[5846863], pos=[5846863, 3], test_set=[1], validation_set=[1], y=[5846863]), Batch(instance_labels=[6890118], pos=[6890118, 3], test_set=[1], validation_set=[1], y=[6890118]), Batch(instance_labels=[5195391], pos=[5195391, 3], test_set=[1], validation_set=[1], y=[5195391]), Batch(instance_labels=[5000698], pos=[5000698, 3], test_set=[1], validation_set=[1], y=[5000698]), Batch(instance_labels=[7061905], pos=[7061905, 3], test_set=[1], validation_set=[1], y=[7061905]), Batch(instance_labels=[5762467], pos=[5762467, 3], test_set=[1], validation_set=[1], y=[5762467]), Batch(instance_labels=[6915118], pos=[6915118, 3], test_set=[1], validation_set=[1], y=[6915118]), Batch(instance_labels=[6366607], pos=[6366607, 3], test_set=[1], validation_set=[1], y=[6366607]), Batch(instance_labels=[1483208], pos=[1483208, 3], test_set=[1], validation_set=[1], y=[1483208]), Batch(instance_labels=[357435], pos=[357435, 3], test_set=[1], validation_set=[1], y=[357435])]
            print(raw_areas)
            for i, area in enumerate(raw_areas):
                torch.save(area, self.raw_areas_paths[i])


            for area_datas in data_list:
                # Apply pre_transform
                if self.pre_transform is not None:
                    area_datas = self.pre_transform(area_datas)
            torch.save(data_list, self.pre_processed_path)
        # if we already processed the raw .ply data files in a previous run with the same grid_size and forest_regions, we can simply load the processed data
        else:
            data_list = torch.load(self.pre_processed_path)

        if self.debug:
            return

        train_data_list = []
        val_data_list = []
        trainval_data_list = []
        test_data_list = []
        #list is a list containing one single data file path
        for list in data_list:
            # data is one single file path
            for data in list:
                validation_set = data.validation_set
                del data.validation_set
                test_set = data.test_set
                del data.test_set
                if validation_set:
                    val_data_list.append(data)
                elif test_set:
                    test_data_list.append(data)
                else:
                    train_data_list.append(data)
        trainval_data_list = val_data_list + train_data_list

        print("train_data_list:")
        print(train_data_list)
        print("test_data_list:")
        print(test_data_list)
        print("val_data_list:")
        print(val_data_list)
        print("trainval_data_list:")
        print(trainval_data_list)
        if self.pre_collate_transform:
            log.info("pre_collate_transform ...")
            log.info(self.pre_collate_transform)
            train_data_list = self.pre_collate_transform(train_data_list)
            val_data_list = self.pre_collate_transform(val_data_list)
            test_data_list = self.pre_collate_transform(test_data_list)
            trainval_data_list = self.pre_collate_transform(trainval_data_list)

        self._save_data(train_data_list, val_data_list, test_data_list, trainval_data_list)

    def process_test(self, test_area):
        """Takes the .ply files specified in data:fold: [...] in the file conf/eval.yaml as test files, processes them and saves the newly created files in self.processed_dir.
        This method is used during evaluation/running eval.py.
        @Treeins: Method is extended so that we can evaluate on more than one test file."""

        self.processed_path = osp.join(self.processed_dir,'processed_test.pt')

        #if not os.path.exists(self.processed_dir):
        #    os.mkdir(self.processed_dir)

        test_data_list = []
        for i, file_path in enumerate(test_area): #for each .ply test data file's path
            area_name = os.path.split(file_path)[-1] #e.g. SCION_plot_31_annotated_test.ply
            processed_area_path = osp.join(self.processed_dir, self.processed_file_names[i])
            if not os.path.exists(processed_area_path): #if .pt file corresponding to .ply test file at file_path hasn't been created and saved in self.processed_dir yet
                
                if self.add_input_features == [] or self.add_input_features ==None:
                    xyz, semantic_labels, instance_labels = read_treeins_format(
                    file_path, self.add_input_features, label_out=True, verbose=self.verbose, debug=self.debug
                )
                else:
                    xyz, semantic_labels, instance_labels, Sum, Omnivariance,Eigenentropy, Anisotropy, planarity, linearity, Surface_var, scattering, verticality, Verticality2, moment1_1, moment1_2, moment2_1, moment2_2, intensity, return_num, scan_angle_rank= read_treeins_format(
                    file_path, self.add_input_features, label_out=True, verbose=self.verbose, debug=self.debug
                )
                #xyz, semantic_labels, instance_labels = read_treeins_format(
                #    file_path, label_out=True, verbose=self.verbose, debug=self.debug
                #)
                data = Data(pos=xyz, y=semantic_labels)
                if self.keep_instance:
                    data.instance_labels = instance_labels
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.add_input_features != [] and self.add_input_features !=None:
                    data.Sum = Sum
                    data.Omnivariance = Omnivariance
                    data.Eigenentropy = Eigenentropy
                    data.Anisotropy = Anisotropy
                    data.planarity = planarity
                    data.linearity = linearity
                    data.Surface_var = Surface_var
                    data.scattering = scattering
                    data.verticality = verticality
                    data.Verticality2 = Verticality2
                    data.moment1_1 = moment1_1
                    data.moment1_2 = moment1_2
                    data.moment2_1 = moment2_1
                    data.moment2_2 = moment2_2
                    data.intensity = intensity
                    data.return_num = return_num
                    data.scan_angle_rank = scan_angle_rank
                print("area_name:")
                print(area_name)
                print("data:")  #Data(pos=[30033430, 3], validation_set=False, y=[30033430])
                print(data)
                # @Treeins: important to not just append data, but [data], so that after the cT.PointCloudFusion() transformation below, we still separate the single data files
                test_data_list.append([data])
                torch.save(data, processed_area_path)
            else: #if .pt file corresponding to .ply test file at file_path has already been created and saved in self.processed_dir
                data = torch.load(processed_area_path)
                # @Treeins: important to not just append data, but [data], so that after the cT.PointCloudFusion() transformation below, we still separate the single data files
                test_data_list.append([data])

        # test_data_list is a list of lists where each such list contains one single test data file's data -> [[Data (...)],[Data(...)], ...]
        raw_areas = cT.PointCloudFusion()(test_data_list)
        for i, area in enumerate(raw_areas):#@Treeins: for each batch in raw_areas (where one batch comes exactly from one test data file)
            torch.save(area, self.raw_areas_paths[i])

        if self.debug:
            return

        print("test_data_list:")
        print(test_data_list)

        # @Treeins: take the test data file's data out of the inner list:
        # test_data_list is of format [[Data(...)], [Data(...)], ...] vs. new_test_data_list is of format [Data(...), Data(...), ...]
        new_test_data_list = [listelem[0] for listelem in test_data_list]
        test_data_list = new_test_data_list
        if self.pre_collate_transform:
            log.info("pre_collate_transform ...")
            log.info(self.pre_collate_transform)
            test_data_list = self.pre_collate_transform(test_data_list)
        torch.save(test_data_list, self.processed_path)


    def _save_data(self, train_data_list, val_data_list, test_data_list, trainval_data_list):
        torch.save(self.collate(train_data_list), self.processed_paths[0])
        torch.save(self.collate(val_data_list), self.processed_paths[1])
        torch.save(self.collate(test_data_list), self.processed_paths[2])
        torch.save(self.collate(trainval_data_list), self.processed_paths[3])

    def _load_data(self, path):
        self.data, self.slices = torch.load(path)


class TreeinsSphere(TreeinsOriginalFused):
    """ Small variation of TreeinsOriginalFused that allows random sampling of spheres
    within an Area during training and validation. Spheres have a radius of 8m. If sample_per_epoch is not specified, spheres
    are taken on a 0.16m grid.

    Parameters
    ----------
    root: str
        path to the directory where the data will be saved
    test_area: int
        number between 1 and 4 that denotes the area used for testing
    train: bool
        Is this a train split or not
    pre_collate_transform:
        Transforms to be applied before the data is assembled into samples (apply fusing here for example)
    keep_instance: bool
        set to True if you wish to keep instance data
    sample_per_epoch
        Number of spheres that are randomly sampled at each epoch (-1 for fixed grid)
    radius
        radius of each sphere
    pre_transform
    transform
    pre_filter
    """

    def __init__(self, root, sample_per_epoch=100, radius=8, grid_size=0.12, *args, **kwargs):
        self._sample_per_epoch = sample_per_epoch
        self._radius = radius
        self._grid_sphere_sampling = cT.GridSampling3D(size=grid_size, mode="last")
        super().__init__(root, grid_size, *args, **kwargs)

    def __len__(self):
        if self._sample_per_epoch > 0:
            return self._sample_per_epoch
        else:
            return len(self._test_spheres)

    def len(self):
        return len(self)

    def get(self, idx):
        if self._sample_per_epoch > 0:
            return self._get_random()
        else:
            return self._test_spheres[idx].clone()

    def process(self):  # We have to include this method, otherwise the parent class skips processing
        # @Treeins: case for training/when running train.py
        if len(self.test_area)==0 or isinstance(self.test_area[0], int):
            super().process()
        # @Treeins: case for evaluation/when running eval.py
        else:
            super().process_test(self.test_area)

    def download(self):  # We have to include this method, otherwise the parent class skips download
        super().download()

    def _get_random(self):
        # Random spheres biased towards getting more low frequency classes
        chosen_label = np.random.choice(self._labels, p=self._label_counts)
        valid_centres = self._centres_for_sampling[self._centres_for_sampling[:, 4] == chosen_label]
        centre_idx = int(random.random() * (valid_centres.shape[0] - 1))
        centre = valid_centres[centre_idx]
        area_data = self._datas[centre[3].int()]
        sphere_sampler = cT.SphereSampling(self._radius, centre[:3], align_origin=False)
        return sphere_sampler(area_data)

    def _save_data(self, train_data_list, val_data_list, test_data_list, trainval_data_list):
        torch.save(train_data_list, self.processed_paths[0])
        torch.save(val_data_list, self.processed_paths[1])
        torch.save(test_data_list, self.processed_paths[2])
        torch.save(trainval_data_list, self.processed_paths[3])

    def _load_data(self, path):
        self._datas = torch.load(path)
        if not isinstance(self._datas, list):
            self._datas = [self._datas]
        if self._sample_per_epoch > 0:
            self._centres_for_sampling = []
            #print(self._datas)
            for i, data in enumerate(self._datas):
                assert not hasattr(
                    data, cT.SphereSampling.KDTREE_KEY
                )  # Just to make we don't have some out of date data in there
                low_res = self._grid_sphere_sampling(data.clone())
                centres = torch.empty((low_res.pos.shape[0], 5), dtype=torch.float)
                centres[:, :3] = low_res.pos
                centres[:, 3] = i
                centres[:, 4] = low_res.y
                self._centres_for_sampling.append(centres)
                tree = KDTree(np.asarray(data.pos), leaf_size=10)
                setattr(data, cT.SphereSampling.KDTREE_KEY, tree)

            self._centres_for_sampling = torch.cat(self._centres_for_sampling, 0)
            uni, uni_counts = np.unique(np.asarray(self._centres_for_sampling[:, -1]), return_counts=True)
            uni_counts = np.sqrt(uni_counts.mean() / uni_counts)
            self._label_counts = uni_counts / np.sum(uni_counts)
            self._labels = uni
        else:
            grid_sampler = cT.GridSphereSampling(self._radius, self._radius, center=False)
            self._test_spheres = grid_sampler(self._datas)


class TreeinsCylinder(TreeinsSphere):
    def _get_random(self):
        # Random spheres biased towards getting more low frequency classes
        while True:
            while True:
                chosen_region = np.random.choice(self._region_labels, p=self._region_label_counts)
                chosen_label = np.random.choice(self._labels, p=self._label_counts)
                #valid_centres = self._centres_for_sampling[self._centres_for_sampling[:, 4] == chosen_label]
                valid_centres = self._centres_for_sampling[self._centres_for_sampling[:, 5] == chosen_region]
                valid_centres = valid_centres[valid_centres[:, 4] == chosen_label]
                if valid_centres.shape[0]>0:
                    break
            centre_idx = int(random.random() * (valid_centres.shape[0] - 1))
            centre = valid_centres[centre_idx]
            area_data = self._datas[centre[3].int()]
            cylinder_sampler = cT.CylinderSampling(self._radius, centre[:3], align_origin=False)
            cylinder_area = cylinder_sampler(area_data) #@Treeins
            if (torch.any(cylinder_area.y==2) or torch.any(cylinder_area.y==3) or torch.any(cylinder_area.y==4)).item(): #@Treeins: ensure that cylinder_area contains at least one point labelled as tree
                return cylinder_area

    def _load_data(self, path):
        self._datas = torch.load(path)
        if not isinstance(self._datas, list):
            self._datas = [self._datas]
        if self._sample_per_epoch > 0:
            self._centres_for_sampling = []
            for i, data in enumerate(self._datas):
                assert not hasattr(
                    data, cT.CylinderSampling.KDTREE_KEY
                )  # Just to make we don't have some out of date data in there
                low_res = self._grid_sphere_sampling(data.clone())
                centres = torch.empty((low_res.pos.shape[0], 6), dtype=torch.float)
                centres[:, :3] = low_res.pos
                centres[:, 3] = i
                centres[:, 4] = low_res.y
                centres[:, 5] = low_res.forest_name
                self._centres_for_sampling.append(centres)
                tree = KDTree(np.asarray(data.pos[:, :-1]), leaf_size=10)
                setattr(data, cT.CylinderSampling.KDTREE_KEY, tree)

            self._centres_for_sampling = torch.cat(self._centres_for_sampling, 0)
            uni, uni_counts = np.unique(np.asarray(self._centres_for_sampling[:, -2]), return_counts=True)
            idx_valid = uni != -1
            uni = uni[idx_valid]
            uni_counts = uni_counts[idx_valid]
            uni_counts = np.sqrt(uni_counts.mean() / uni_counts)
            self._label_counts = uni_counts / np.sum(uni_counts)
            self._labels = uni
            #calculate weights for forest region
            uni, uni_counts = np.unique(np.asarray(self._centres_for_sampling[:, -1]), return_counts=True)
            uni_counts = np.sqrt(uni_counts.mean() / uni_counts)
            self._region_label_counts = uni_counts / np.sum(uni_counts)
            self._region_labels = uni
        else:
            grid_sampler = cT.GridCylinderSampling(self._radius, self._radius, center=False)
            self._test_spheres = []
            self._num_spheres = []
            for i, data in enumerate(self._datas): #for each test data file's data
                test_spheres = grid_sampler(data)
                # @Treeins: self._test_spheres is list of grid-cylinder-sampled data from all test data
                # -> data from different data files in the same list without separation
                # -> to recover separation again in the end, we need self._num_spheres
                self._test_spheres = self._test_spheres + test_spheres
                # @Treeins: saves how many cylinders were sampled from each test data file respectively
                # -> like this, we can recover which cylinder comes from which test data file
                self._num_spheres = self._num_spheres + [len(test_spheres)]


class TreeinsFusedDataset(BaseDataset):
    """ Wrapper around NPM3DSphere that creates train and test datasets.

    Parameters
    ----------
    dataset_opt: omegaconf.DictConfig
        Config dictionary that should contain

            - dataroot
            - fold: test_area parameter
            - pre_collate_transform
            - train_transforms
            - test_transforms
    """

    INV_OBJECT_LABEL = INV_OBJECT_LABEL

    def __init__(self, dataset_opt):
        super().__init__(dataset_opt)

        sampling_format = dataset_opt.get("sampling_format", "sphere")
        dataset_cls = TreeinsCylinder if sampling_format == "cylinder" else TreeinsSphere

        self.train_dataset = dataset_cls(
            self._data_path,
            sample_per_epoch=3000,
            test_area=self.dataset_opt.fold,
            split="train",
            pre_collate_transform=self.pre_collate_transform,
            transform=self.train_transform,
        )

        self.val_dataset = dataset_cls(
            self._data_path,
            sample_per_epoch=-1,
            test_area=self.dataset_opt.fold,
            split="val",
            pre_collate_transform=self.pre_collate_transform,
            transform=self.val_transform,
        )
        self.test_dataset = dataset_cls(
            self._data_path,
            sample_per_epoch=-1,
            test_area=self.dataset_opt.fold,
            split="test",
            pre_collate_transform=self.pre_collate_transform,
            transform=self.test_transform,
        )

        if dataset_opt.class_weight_method:
            self.add_weights(class_weight_method=dataset_opt.class_weight_method)

    @property
    def test_data(self):
        return self.test_dataset[0].raw_test_data
        
    @property
    def test_data_spheres(self):
        return self.test_dataset[0]._test_spheres

    @property
    def test_data_num_spheres(self):
        return self.test_dataset[0]._num_spheres

    @staticmethod
    def to_ply(pos, label, file):
        """ Allows to save NPM3D predictions to disk using NPM3D color scheme

        Parameters
        ----------
        pos : torch.Tensor
            tensor that contains the positions of the points
        label : torch.Tensor
            predicted label
        file : string
            Save location
        """
        to_ply(pos, label, file)

    def get_tracker(self, wandb_log: bool, tensorboard_log: bool):
        """Factory method for the tracker

        Arguments:
            wandb_log - Log using weight and biases
            tensorboard_log - Log using tensorboard
        Returns:
            [BaseTracker] -- tracker
        """
        #from torch_points3d.metrics.s3dis_tracker import S3DISTracker
        #return S3DISTracker(self, wandb_log=wandb_log, use_tensorboard=tensorboard_log)
        from torch_points3d.metrics.segmentation_tracker import SegmentationTracker
        return SegmentationTracker(self, wandb_log=wandb_log, use_tensorboard=tensorboard_log)
