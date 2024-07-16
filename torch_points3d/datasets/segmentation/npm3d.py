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

NPM3D_NUM_CLASSES = 9

FILE_NAMES=["Lille1_1","Lille1_2","Lille2","Paris"]


INV_OBJECT_LABEL = {
    #0: "unclassified",
    0: "ground",
    1: "buildings",
    2: "poles",
    3: "bollards",
    4: "trash_cans",
    5: "barriers",
    6: "pedestrians",
    7: "cars",
    8: "natural",
}


OBJECT_COLOR = np.asarray(
    [
        #[233, 229, 107],  # 'unclassified' .-> .yellow
        [95, 156, 196],  # 'ground' .-> . blue
        [179, 116, 81],  # 'buildings'  ->  brown
        [241, 149, 131],  # 'poles'  ->  salmon
        [81, 163, 148],  # 'bollards'  ->  bluegreen
        [77, 174, 84],  # 'trash_cans'  ->  bright green
        [108, 135, 75],  # 'barriers'   ->  dark green
        [41, 49, 101],  # 'pedestrians'  ->  darkblue
        [79, 79, 76],  # 'cars'  ->  dark grey
        [223, 52, 52],  # 'natural'  ->  red
        [0, 0, 0],  # unlabelled .->. black
    ]
)

OBJECT_LABEL = {name: i for i, name in INV_OBJECT_LABEL.items()}

ALL_FILE_NAMES = ['Lille1_1_train', 'Lille1_1_test', 'Lille1_1_val', 'Lille1_2_train', 'Lille1_2_test', 'Lille1_2_val', 'Lille2_train', 'Lille2_test', 'Lille2_val', 'Paris_train', 'Paris_test', 'Paris_val']

################################### UTILS #######################################
def object_name_to_label(object_class):
    """convert from object name in NPPM3D to an int"""
    object_label = OBJECT_LABEL.get(object_class, OBJECT_LABEL["unclassified"])
    return object_label

def read_npm3d_format(train_file, label_out=True, verbose=False, debug=False):
    """extract data from a room folder"""

    #room_type = room_name.split("_")[0]
    #room_label = ROOM_TYPES[room_type]
    raw_path = train_file
    data = read_ply(raw_path)
    xyz = np.vstack((data['x'], data['y'], data['z'])).astype(np.float32).T
    if not label_out:
        return xyz
    semantic_labels = data['scalar_class'].astype(np.int64)-1
    instance_labels = data['scalar_label'].astype(np.int64)+1
    #print(np.unique(instance_labels))
    return (
        torch.from_numpy(xyz),
        torch.from_numpy(semantic_labels),
        torch.from_numpy(instance_labels),
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
    el = PlyElement.describe(ply_array, "NPM3D")
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


class NPM3DOriginalFused(InMemoryDataset):
    """ Original NPM3D dataset. Each area is loaded individually and can be processed using a pre_collate transform. 
    This transform can be used for example to fuse the area into a single space and split it into 
    spheres or smaller regions. If no fusion is applied, each element in the dataset is a single area by default.

    Parameters
    ----------
    root: str
        path to the directory where the data will be saved
    test_area: int
        number between 1 and 4 that denotes the area used for testing
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

    #form_url = (
    #    ""
    #)
    #download_url = ""
    #zip_name = ""
    num_classes = NPM3D_NUM_CLASSES

    def __init__(
        self,
        root,
        grid_size,
        test_area=4,
        split="train",
        transform=None,
        pre_transform=None,
        pre_collate_transform=None,
        pre_filter=None,
        keep_instance=False,
        verbose=False,
        debug=False,
    ):
        #assert test_area >= 1 and test_area <= 4
        if isinstance(test_area, int):
            assert test_area >= 1 and test_area <= 4
        else:
            assert len(test_area) == 1
            self.area_name = os.path.split(test_area[0])[-1].split('.')[0]
        self.transform = transform
        self.pre_collate_transform = pre_collate_transform
        self.test_area = test_area
        self.keep_instance = keep_instance
        self.verbose = verbose
        self.debug = debug
        self._split = split
        self.grid_size = grid_size
     
        super(NPM3DOriginalFused, self).__init__(root, transform, pre_transform, pre_filter)
        if isinstance(test_area, int):
            
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
                self.raw_test_data = torch.load(self.raw_areas_paths[test_area - 1])
        else:
            #self.process_test(test_area)
            path = self.processed_paths[0]
            self._load_data(path)
            self.raw_test_data = torch.load(self.raw_areas_paths[0])

    @property
    def center_labels(self):
        if hasattr(self.data, "center_label"):
            return self.data.center_label
        else:
            return None

    @property
    def raw_file_names(self):
        return [osp.join(self.raw_dir, f+'.ply') for f in ALL_FILE_NAMES]

    @property
    def processed_dir(self):
        #return osp.join(self.root,'processed_'+str(self.grid_size)+'_'+str(self.test_area))
        if isinstance(self.test_area, int):
            return osp.join(self.root,'processed_'+str(self.grid_size)+'_'+str(self.test_area))
        else:
            return osp.join(self.root,'processed_'+str(self.grid_size)+'_'+str(self.area_name))

    @property
    def pre_processed_path(self):
        pre_processed_file_names = "preprocessed.pt"
        return os.path.join(self.processed_dir, pre_processed_file_names)

    @property
    def raw_areas_paths(self):
        #return [os.path.join(self.processed_dir, "raw_area_%i.pt" % i) for i in range(4)]
        if isinstance(self.test_area, int):
            return [os.path.join(self.processed_dir, "raw_area_%i.pt" % i) for i in range(8)]
        else:
            return [os.path.join(self.processed_dir, 'raw_area_'+self.area_name+'.pt')]

    @property
    def processed_file_names(self):
        test_area = self.test_area
        #return (
        #    ["{}_{}.pt".format(s, test_area) for s in ["train", "val", "test", "trainval"]]
        #    + self.raw_areas_paths
        #    + [self.pre_processed_path]
        #)
        if isinstance(test_area, int):
            return (
                ["{}_{}.pt".format(s, test_area) for s in ["train", "val", "test", "trainval"]]
                + self.raw_areas_paths
                + [self.pre_processed_path]
            )
        else:
            return (['processed_'+self.area_name+'.pt'])

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
        if not os.path.exists(self.pre_processed_path):
        
            input_ply_files =[osp.join(self.raw_dir, f+'.ply') for f in ALL_FILE_NAMES]

            # Gather data per area
            data_list = [[] for _ in range(8)]
            area_num_true = 0
            for area_num, file_path in enumerate(input_ply_files):
                
                area_name = os.path.split(file_path)[-1]
                #find train/val/test files for current test area
                is_train_data = False
                is_validation_data = False
                is_test_data = False
                if "val" in area_name: #area_num == self.test_area-1:
                    is_validation_data = True
                elif FILE_NAMES[self.test_area-1] in area_name and "test" in area_name:
                    is_test_data = True
                elif FILE_NAMES[self.test_area-1] not in area_name and "train" in area_name:
                    is_train_data = True
                 
                #if current file does not belong to train/val/test files, then pass     
                if not (is_train_data or is_test_data or is_validation_data):
                    continue
                
                xyz, semantic_labels, instance_labels = read_npm3d_format(
                    file_path, label_out=True, verbose=self.verbose, debug=self.debug
                )

                data = Data(pos=xyz, y=semantic_labels)
                data.validation_set = False
                data.test_set = False
                if "val" in area_name: #area_num == self.test_area-1:
                    data.validation_set = True
                #elif FILE_NAMES[self.test_area-1] in area_name:
                elif "test" in area_name:
                    data.test_set = True
                    
                if self.keep_instance:
                    data.instance_labels = instance_labels

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                print("area_num:")
                print(area_num_true)
                print("data:")  #Data(pos=[30033430, 3], validation_set=False, y=[30033430])
                print(data)
                data_list[area_num_true].append(data)
                area_num_true += 1
            print("data_list")
            print(data_list)
            raw_areas = cT.PointCloudFusion()(data_list)
            print("raw_areas")
            print(raw_areas)
            for i, area in enumerate(raw_areas):
                torch.save(area, self.raw_areas_paths[i])

            for area_datas in data_list:
                # Apply pre_transform
                if self.pre_transform is not None:
                    #for data in area_datas:
                    area_datas = self.pre_transform(area_datas)
            torch.save(data_list, self.pre_processed_path)
        else:
            data_list = torch.load(self.pre_processed_path)

        if self.debug:
            return

        train_data_list = []
        val_data_list = []
        trainval_data_list = []
        test_data_list = []
        for i in range(8):
            #if i != self.test_area - 1:
            #train_data_list[i] = []
            #val_data_list[i] = []
            for data in data_list[i]:
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
        #test_data_list = data_list[self.test_area - 1]

        #train_data_list = list(train_data_list.values())
        #val_data_list = list(val_data_list.values())
        #trainval_data_list = list(trainval_data_list.values())
        #test_data_list = data_list[self.test_area - 1]

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

        #preprocess_dir = osp.join(self.root,'processed_'+str(self.grid_size))
        #self.processed_path = osp.join(preprocess_dir,'processed.pt')

        #if not os.path.exists(preprocess_dir):
        #    os.mkdir(preprocess_dir)
        test_data_list = []
        #self.raw_path = []
        for i, file_path in enumerate(test_area):
            area_name = os.path.split(file_path)[-1]
            if not os.path.exists(self.pre_processed_path):
                xyz, semantic_labels, instance_labels = read_npm3d_format(
                    file_path, label_out=True, verbose=self.verbose, debug=self.debug
                )
                data = Data(pos=xyz, y=semantic_labels)
                if self.keep_instance:
                    data.instance_labels = instance_labels
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                print("area_name:")
                print(area_name)
                print("data:")  #Data(pos=[30033430, 3], validation_set=False, y=[30033430])
                print(data)
                test_data_list.append(data)
                # if self.pre_transform is not None:
                #     for data in test_data_list:
                #         data = self.pre_transform(data)
                #torch.save(data, pre_processed_path)
                torch.save(data, self.pre_processed_path)

                
            else:
                #data = torch.load(pre_processed_path)
                data = torch.load(self.pre_processed_path)
                test_data_list.append(data)

        raw_areas = cT.PointCloudFusion()(test_data_list)
        #torch.save(raw_areas, self.raw_path[0])
        torch.save(raw_areas, self.raw_areas_paths[0])
            
        if self.debug:
            return

        print("test_data_list:")
        print(test_data_list)
        if self.pre_collate_transform:
            log.info("pre_collate_transform ...")
            log.info(self.pre_collate_transform)
            test_data_list = self.pre_collate_transform(test_data_list)
        #torch.save(test_data_list, self.processed_path)
        torch.save(test_data_list, self.processed_paths[0])


    def _save_data(self, train_data_list, val_data_list, test_data_list, trainval_data_list):
        torch.save(self.collate(train_data_list), self.processed_paths[0])
        torch.save(self.collate(val_data_list), self.processed_paths[1])
        torch.save(self.collate(test_data_list), self.processed_paths[2])
        torch.save(self.collate(trainval_data_list), self.processed_paths[3])

    def _load_data(self, path):
        self.data, self.slices = torch.load(path)


class NPM3DSphere(NPM3DOriginalFused):
    """ Small variation of NPM3DOriginalFused that allows random sampling of spheres 
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
        #super().process()
        if isinstance(self.test_area, int):
            super().process()
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


class NPM3DCylinder(NPM3DSphere):
    def _get_random(self):
        # Random spheres biased towards getting more low frequency classes
        chosen_label = np.random.choice(self._labels, p=self._label_counts)
        valid_centres = self._centres_for_sampling[self._centres_for_sampling[:, 4] == chosen_label]
        centre_idx = int(random.random() * (valid_centres.shape[0] - 1))
        centre = valid_centres[centre_idx]
        area_data = self._datas[centre[3].int()]
        cylinder_sampler = cT.CylinderSampling(self._radius, centre[:3], align_origin=False)
        return cylinder_sampler(area_data)

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
                centres = torch.empty((low_res.pos.shape[0], 5), dtype=torch.float)
                centres[:, :3] = low_res.pos
                centres[:, 3] = i
                centres[:, 4] = low_res.y
                self._centres_for_sampling.append(centres)
                tree = KDTree(np.asarray(data.pos[:, :-1]), leaf_size=10)
                setattr(data, cT.CylinderSampling.KDTREE_KEY, tree)

            self._centres_for_sampling = torch.cat(self._centres_for_sampling, 0)
            uni, uni_counts = np.unique(np.asarray(self._centres_for_sampling[:, -1]), return_counts=True)
            uni_counts = np.sqrt(uni_counts.mean() / uni_counts)
            self._label_counts = uni_counts / np.sum(uni_counts)
            self._labels = uni
        else:
            grid_sampler = cT.GridCylinderSampling(self._radius, self._radius, center=False)
            self._test_spheres = grid_sampler(self._datas)


class NPM3DFusedDataset(BaseDataset):
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
        dataset_cls = NPM3DCylinder if sampling_format == "cylinder" else NPM3DSphere

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
