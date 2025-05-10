import random
import time
import yaml
import os
import einops
from matplotlib import pyplot as plt
import torch
import torch.multiprocessing as mp
from tqdm import tqdm
from scipy.spatial.transform import Rotation
import numpy as np

from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.loss_utils import l1_loss, ssim
from utils.logging_utils import Log
from utils.multiprocessing_utils import clone_obj
from utils.pose_utils import update_pose
from utils.slam_utils import get_loss_mapping, get_loss_mapping_rgb, get_loss_tracking
# from gaussian_optimizer_utils.util_gau import load_ply
from utils.camera_utils import Camera 
from utils.dataset import TUMDataset, ReplicaDataset
from utils.config_utils import load_config
from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2, getWorld2View2
from munch import munchify
from gaussian_splatting.scene.gaussian_model import GaussianModel
from mpl_toolkits.mplot3d import Axes3D

import evo
import numpy as np
import torch
from evo.core import metrics, trajectory
from evo.core.metrics import PoseRelation, Unit
from evo.core.trajectory import PosePath3D, PoseTrajectory3D
from evo.tools import plot
from evo.tools.plot import PlotMode
from evo.tools.settings import SETTINGS

from utils.eval_utils import evaluate_evo
import tkinter as tk
from tkinter import filedialog
import pickle

PATH = "/home/curdin/master_thesis/outputs/"
# DATE = "25_04_07/"
DATE = "25_05_07/"
# DATE = "25_05_07_11kf_ro2/"
# DATE = "25_05_08/"
DATE = "25_05_08_room0/"
GAUSSIANS_FILENAME = "gaussians.ply"

FOLDERNAME = "2025-05-07_12-30_0_it" + "/"
# FOLDERNAME = "2025-05-08_11-18_replica_office1_0_it_window4" + "/"
# FOLDERNAME = "2025-05-07_23-23_30_it_window5" + "/"
folders = [PATH+DATE+f for f in os.listdir(PATH+DATE)]


device = "cuda"
gaussians_filepath = PATH + DATE + FOLDERNAME + GAUSSIANS_FILENAME
FOLDERPATH = PATH + DATE + FOLDERNAME

dataset_path = "/home/curdin/repos/MonoGS/datasets/replica/"
# dataset_name = "office_2/"
dataset_name = "room_0/"
config_path = os.path.join("/home/curdin/repos/MonoGS/configs/rgbd/replica/office2.yaml")
# config_path = os.path.join("/home/curdin/repos/MonoGS/configs/rgbd/replica/room0.yaml")

# with open(config_path, "r") as yml:
#     config = yaml.safe_load(yml)
# config = load_config(config_path)
# model_params = munchify(config["model_params"])
# dataset = ReplicaDataset(args=model_params, config=config, path=dataset_path + dataset_name)
# gt_poses = dataset.poses

def load_dataset(used_dataset):
    dataset_path = os.path.join("/home/curdin/repos/MonoGS/", used_dataset)
    print(f"Loading dataset from {dataset_path}")
    dataset_config_name = used_dataset.split("/")[-1]
    dataset_config_path = os.path.join(f"/home/curdin/repos/MonoGS/configs/rgbd/replica/{dataset_config_name}.yaml")
    dataset_config = load_config(dataset_config_path)
    model_params = munchify(dataset_config["model_params"])
    dataset = ReplicaDataset(args=model_params, config=dataset_config, path=dataset_path)
    return dataset

def load_splatt3rSLAM_config(config_path):
    with open(os.path.join(config_path, "base.yaml"), "r") as yml:
        config = yaml.safe_load(yml)
    return config

# filepath = PATH + DATE + FILENAME
def select_folder(initial_dir = PATH):
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    folder_path = filedialog.askdirectory(initialdir=initial_dir, title="Select a Folder")
    if folder_path:
        print(f"Selected folder: {folder_path}")
    else:
        print("No folder selected.")
    return folder_path

def eval_trajectory(folder, optimized_poses=False):
    txt_file_path = os.path.join(folder, "keyframe_poses.txt")
    if os.path.exists(txt_file_path):
        with open(txt_file_path, "r") as file:
            lines = file.readlines()
            timestamps = [line.split()[0] for line in lines]
            numbers = [
                [float(num) for num in line.split() if num.lstrip('-').replace('.', '', 1).isdigit()]
                for line in lines
            ]
    else:
        print(f"Files not found: {txt_file_path}")
        return
    keyframe_ids = [int(num[0]) for num in numbers]

    config = load_splatt3rSLAM_config(folder)
    dataset = load_dataset(config["used_dataset"])
    gt_poses = dataset.poses

    keyframe_gt_poses = [np.linalg.inv(gt_poses[i*4]) for i in keyframe_ids]
    frame_poses = []
    for number in numbers:
        frame_pose = np.eye(4)
        frame_pose[:3, :3] = Rotation.from_quat(number[4:]).as_matrix()
        frame_pose[:3, 3] = number[1:4]
        frame_poses.append(frame_pose)

    optimized_poses_filepath = os.path.join(folder, "optimized_poses.pkl")
    if os.path.exists(optimized_poses_filepath):
        with open(optimized_poses_filepath, "rb") as file:
            optimized_poses = pickle.load(file)
    else:
        print(f"File not found: {optimized_poses_filepath}")
        return    
    print(optimized_poses)
    # poses_cw_opt = [np.eye(4).astype(np.float32)]
    poses_cw_opt = []
    for (key, opt_pose) in optimized_poses.items():
        print(key)
        T_cw_opt = torch.linalg.inv(opt_pose).cpu().numpy()
        print(T_cw_opt)
        poses_cw_opt.append(T_cw_opt)
        print(len(poses_cw_opt))
    print(poses_cw_opt)

    def calculate_ate(estimated_poses, ground_truth_poses):
        errors = []
        for est_pose, gt_pose in zip(estimated_poses, ground_truth_poses):
            # est_translation = np.array(est_pose[:3, 3])
            est_translation = est_pose
            # gt_translation = np.array(gt_pose[:3, 3])
            gt_translation = gt_pose
            error = np.linalg.norm(est_translation - gt_translation)
            errors.append(error)
        ate = np.mean(errors)
        return ate
    # print(frame_poses)

    print("Estimated Poses: ---------------------------------------------")
    print(poses_cw_opt)
    def umeyama(X, Y):
        """
        Estimates the Sim(3) transformation between `X` and `Y` point sets.

        Estimates c, R and t such as c * R @ X + t ~ Y.

        Parameters
        ----------
        X : numpy.array
            (m, n) shaped numpy array. m is the dimension of the points,
            n is the number of points in the point set.
        Y : numpy.array
            (m, n) shaped numpy array. Indexes should be consistent with `X`.
            That is, Y[:, i] must be the point corresponding to X[:, i].
        
        Returns
        -------
        c : float
            Scale factor.
        R : numpy.array
            (3, 3) shaped rotation matrix.
        t : numpy.array
            (3, 1) shaped translation vector.
        """
        mu_x = X.mean(axis=1).reshape(-1, 1)
        mu_y = Y.mean(axis=1).reshape(-1, 1)
        var_x = np.square(X - mu_x).sum(axis=0).mean()
        cov_xy = ((Y - mu_y) @ (X - mu_x).T) / X.shape[1]
        U, D, VH = np.linalg.svd(cov_xy)
        S = np.eye(X.shape[0])
        if np.linalg.det(U) * np.linalg.det(VH) < 0:
            S[-1, -1] = -1
        c = np.trace(np.diag(D) @ S) / var_x
        R = U @ S @ VH
        t = mu_y - c * R @ mu_x
        return c, R, t
    
    ape_stats = evaluate_evo(keyframe_gt_poses, frame_poses, "/home/curdin/master_thesis/outputs/", label="test", monocular=True, ape_only=True)

    print("Optimized Poses: ---------------------------------------------")

    print(len(poses_cw_opt))
    print(len(keyframe_gt_poses))
    opt_ape_stats = evaluate_evo(keyframe_gt_poses, poses_cw_opt, "/home/curdin/master_thesis/outputs/", label="test", monocular=True, ape_only=True)

    frame_poses = np.array(frame_poses)
    keyframe_positions = frame_poses[:, :3, 3]
    keyframe_gt_poses = np.array(keyframe_gt_poses)
    gt_keyframe_positions = keyframe_gt_poses[:, :3, 3]

# DATE = "25_05_09_opt_pose/"
DATE = "25_05_09_opt_pose_end/"

folder = select_folder(initial_dir=PATH+DATE)
eval_trajectory(folder)
