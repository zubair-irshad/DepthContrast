# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import logging
import os

import numpy as np

# import sys
# sys.path.append('/home/zubairirshad/DepthContrast')
from datasets.transforms.augment3d import get_transform3d

# try:
#     ### Default uses minkowski engine
#     from datasets.transforms.voxelizer import Voxelizer
#     from datasets.transforms import transforms
# except:
#     pass
    
# try:
#     try:
#         from spconv.utils import VoxelGeneratorV2 as VoxelGenerator
#     except:
#         from spconv.utils import VoxelGenerator
# except:
#     pass

from torch.utils.data import Dataset
import torch

### Waymo lidar range
#POINT_RANGE = np.array([  0. , -75. ,  -3. ,  75.0,  75. ,   3. ], dtype=np.float32)
# POINT_RANGE = np.array([0, -40, -3, 70.4, 40, 1], dtype=np.float32)#np.array([  0. , -75. ,  -3. ,  75.0,  75. ,   3. ], dtype=np.float32) ### KITTI

class DepthContrastDataset_NeRFMAE(Dataset):
    """Base Self Supervised Learning Dataset Class."""

    def __init__(self, cfg):
        self.split = "train" ### Default is training
        self.label_objs = []
        self.data_paths = []
        self.label_paths = []
        self.cfg = cfg
        self.batchsize_per_replica = cfg["BATCHSIZE_PER_REPLICA"]
        self.label_sources = []#cfg["LABEL_SOURCES"]
        self.dataset_names = cfg["DATASET_NAMES"]
        self.label_type = cfg["LABEL_TYPE"]
        self.AUGMENT_COORDS_TO_FEATS = False #optional
        self._labels_init = False
        # self._get_data_files("train")

        self.data_paths = str(self.cfg["DATA_PATHS"][0])
        print(self.data_paths)
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        logging.info(f"Rank: {local_rank} Data files:\n{self.data_paths}")

        self.features_path = os.path.join(self.data_paths, "features")

        split_path = os.path.join(self.data_paths, "front3d_split.npz")

        with np.load(split_path) as split:
            self.data_objs = split["train_scenes"]
            # self.test_scenes = split["test_scenes"]
            # self.val_scenes = split["val_scenes"]
            
        # self.data_objs = np.load(self.data_paths[0]) ### Only load the first one for now

    # def _get_data_files(self, split):
    #     local_rank = int(os.environ.get("LOCAL_RANK", 0))

    #     self.data_paths = self.cfg["DATA_PATHS"]
    #     self.label_paths = []
        
    #     logging.info(f"Rank: {local_rank} Data files:\n{self.data_paths}")
    #     logging.info(f"Rank: {local_rank} Label files:\n{self.label_paths}")

    # def _augment_coords_to_feats(self, coords, feats, labels=None):
    #     # Center x,y
    #     coords_center = coords.mean(0, keepdims=True)
    #     coords_center[0, 2] = 0
    #     norm_coords = coords - coords_center
    #     feats = np.concatenate((feats, norm_coords), 1)
    #     return coords, feats, labels


    def construct_grid(self, res):
        res_x, res_y, res_z = res
        x = torch.linspace(0, res_x, res_x)
        y = torch.linspace(0, res_y, res_y)
        z = torch.linspace(0, res_z, res_z)

        scale = torch.tensor(res).max()
        x /= scale
        y /= scale
        z /= scale

        # Shift by 0.5 voxel
        x += 0.5 * (1.0 / scale)
        y += 0.5 * (1.0 / scale)
        z += 0.5 * (1.0 / scale)

        grid = []
        for i in range(res_z):
            for j in range(res_y):
                for k in range(res_x):
                    grid.append([x[k], y[j], z[i]])

        return torch.tensor(grid)

    @staticmethod
    def density_to_alpha(density):
        return np.clip(1.0 - np.exp(-np.exp(density) / 100.0), 0.0, 1.0)
    
    def load_data(self, idx):
        is_success = True
        point_path = self.data_objs[idx]

        scene_features_path = os.path.join(self.features_path, point_path + ".npz")
        # try:

        with np.load(scene_features_path) as features:
            rgbsigma = features["rgbsigma"]
            alpha = self.density_to_alpha(rgbsigma[..., -1])
            rgbsigma[..., -1] = alpha
            res = features["resolution"]

        rgbsigma = np.transpose(rgbsigma, (2, 1, 0, 3)).reshape(-1, 4)
        alpha = rgbsigma[:, -1]
        mask = alpha > 0.01
        
        grid = self.construct_grid(res)

        point = grid[mask, :]

        rgbsigma = rgbsigma[mask, :]

        point = np.concatenate([point, rgbsigma], 1)

        # point = np.load(point_path)
        ### Add height
        floor_height = np.percentile(point[:,2],0.99)
        height = point[:,2] - floor_height
        point = np.concatenate([point, np.expand_dims(height, 1)],1)
        # except Exception as e:
        #     logging.warn(
        #         f"Couldn't load: {self.point_dataset[idx]}. Exception: \n{e}"
        #     )
        #     point = np.zeros([50000, 7])
        #     is_success = False
        return point, is_success

    def __getitem__(self, idx):

        cfg = self.cfg
        # TODO: this doesn't yet handle the case where the length of datasets
        # could be different.
        # if cfg["DATA_TYPE"] == "point_vox":
        #     item = {"data": [], "data_valid": [], "data_moco": [], "vox": [], "vox_moco": []}

        #     data, valid = self.load_data(idx)
        #     item["data"].append(data)
        #     item["data_moco"].append(np.copy(data))
        #     item["vox"].append(np.copy(data))
        #     item["vox_moco"].append(np.copy(data))
        #     item["data_valid"].append(1 if valid else -1)
        # else:
        item = {"data": [], "data_moco": [], "data_valid": [], "data_idx": []}
        
        data, valid = self.load_data(idx)

        print("data shape", data.shape)
        item["data"].append(data)
        item["data_moco"].append(np.copy(data))
        item["data_valid"].append(1 if valid else -1)

        ### Make copies for moco setting
        item["label"] = []
        item["label"].append(idx)

        ### Apply the transformation here
        # if (cfg["DATA_TYPE"] == "point_vox"):
        #     tempitem = {"data": item["data"]}
        #     tempdata = get_transform3d(tempitem, cfg["POINT_TRANSFORMS"])
        #     item["data"] = tempdata["data"]

        #     tempitem = {"data": item["data_moco"]}
        #     tempdata = get_transform3d(tempitem, cfg["POINT_TRANSFORMS"])
        #     item["data_moco"] = tempdata["data"]

        #     tempitem = {"data": item["vox"]}
        #     tempdata = get_transform3d(tempitem, cfg["POINT_TRANSFORMS"], vox=True)
        #     coords = tempdata["data"][0][:,:3]
        #     feats = tempdata["data"][0][:,3:6]*255.0#np.ones(coords.shape)*255.0
        #     labels = np.zeros(coords.shape[0]).astype(np.int32)
        #     item["vox"] = [self.toVox(coords, feats, labels)]
            
        #     tempitem = {"data": item["vox_moco"]}
        #     tempdata = get_transform3d(tempitem, cfg["POINT_TRANSFORMS"], vox=True)
        #     coords = tempdata["data"][0][:,:3]
        #     feats = tempdata["data"][0][:,3:6]*255.0#np.ones(coords.shape)*255.0
        #     labels = np.zeros(coords.shape[0]).astype(np.int32)                    
        #     item["vox_moco"] = [self.toVox(coords, feats, labels)]               
        # else:
        tempitem = {"data": item["data"]}
        tempdata = get_transform3d(tempitem, cfg["POINT_TRANSFORMS"], vox=cfg["VOX"])
        # if cfg["VOX"]:
        #     coords = tempdata["data"][0][:,:3]
        #     feats = tempdata["data"][0][:,3:6]*255.0
        #     labels = np.zeros(coords.shape[0]).astype(np.int32)
        #     item["data"] = [self.toVox(coords, feats, labels)]
        # else:
        item["data"] = tempdata["data"]

        tempitem = {"data": item["data_moco"]}                
        tempdata = get_transform3d(tempitem, cfg["POINT_TRANSFORMS"], vox=cfg["VOX"])
        # if cfg["VOX"]:
        #     coords = tempdata["data"][0][:,:3]
        #     feats = tempdata["data"][0][:,3:6]*255.0#np.ones(coords.shape)*255.0
        #     labels = np.zeros(coords.shape[0]).astype(np.int32)                    
        #     item["data_moco"] = [self.toVox(coords, feats, labels)]
        # else:
        item["data_moco"] = tempdata["data"]

        return item

    def __len__(self):
        return len(self.data_objs)

    def get_available_splits(self, dataset_config):
        return [key for key in dataset_config if key.lower() in ["train", "test"]]

    def num_samples(self, source_idx=0):
        return len(self.data_objs)

    def get_batchsize_per_replica(self):
        # this searches for batchsize_per_replica in self and then in self.dataset
        return getattr(self, "batchsize_per_replica", 1)

    def get_global_batchsize(self):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
        else:
            world_size = 1
        return self.get_batchsize_per_replica() * world_size



import argparse
parser = argparse.ArgumentParser(description='PyTorch Self Supervised Training in 3D')

parser.add_argument('cfg', help='model directory')
parser.add_argument('--quiet', action='store_true')

parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://localhost:15475', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--ngpus', default=8, type=int,
                    help='number of GPUs to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

import yaml

if __name__ == "__main__":

    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.cfg))

    dataset = DepthContrastDataset_NeRFMAE(cfg['dataset'])

    #Now read the dataset 

    # dataset = DepthContrastDataset_NeRFMAE(cfg['dataset'])
    data = dataset[0]
    print(data.keys())


    #Visualize pointclouds

    import open3d as o3d

    print("data", data['data'][0].shape)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data['data'][0][:,:3])
    pcd.colors = o3d.utility.Vector3dVector(data['data'][0][:,3:6])
    o3d.visualization.draw_geometries([pcd])



    
    #check dataloader

