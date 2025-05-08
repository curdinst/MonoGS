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
dataset_name = "office_2/"
config_path = os.path.join("/home/curdin/repos/MonoGS/configs/rgbd/replica/office2.yaml")
with open(config_path, "r") as yml:
    config = yaml.safe_load(yml)
config = load_config(config_path)
model_params = munchify(config["model_params"])
dataset = ReplicaDataset(args=model_params, config=config, path=dataset_path + dataset_name)
gt_poses = dataset.poses
# filepath = PATH + DATE + FILENAME



def compare_render_results():
    folders = [PATH+DATE+f for f in os.listdir(PATH+DATE)]
    results_mean_dict = {}
    results_dict = {}
    print("Folders in PATH:")
    folders = sorted(folders)
    for folder in folders:
        
        print(folder)
        txt_file = os.path.join(folder, "rendering_results.txt")
        if os.path.exists(txt_file):
            with open(txt_file, "r") as file:
                lines = file.readlines()
        else:
            print(f"File not found: {txt_file}")
            continue
        if folder.split("/")[-1][1] == "_":
            test_settings = folder.split("/")[-1][19:]
        else:
            test_settings = folder.split("/")[-1][17:]
        results_dict[test_settings + "_ssim"] = {}
        results_dict[test_settings + "_l1"] = {}
        frame_id = 0
        for line in lines:
            if "ssim mean" in line:
                ssim_val = float(line.split(":")[1].strip())
                results_mean_dict[test_settings] = {"ssim_mean": ssim_val}
            elif "l1 mean" in line:
                l1_loss_val = float(line.split(":")[1].strip())
                results_mean_dict[test_settings]["l1_mean"] = l1_loss_val
            elif  "ssim" in line:
                ssim_val = float(line.split(":")[1].strip())
                frame_nr = int(line.split(":")[0].split("_")[-1])
                results_dict[test_settings + "_ssim"][frame_nr] = ssim_val
                frame_id += 1
            elif "l1" in line:
                l1_loss_val = float(line.split(":")[1].strip())
                frame_nr = int(line.split(":")[0].split("_")[-1])
                results_dict[test_settings + "_l1"][frame_nr] = l1_loss_val
    
    print("Results Dictionary:")
    for key, value in results_mean_dict.items():
        print(f"Iterations: {key}, SSIM: {value['ssim_mean']}, L1 Loss: {value['l1_mean']}")
    for key, value in results_dict.items():
        print(f"Iterations: {key}, value: {value}")

    # num_subplots = int(len(results_dict.keys())/2)
    # i = 1
    # for test_settings in results_dict.keys():
    #     if "_ssim" in test_settings:
    #         frame_ids = list(results_dict[test_settings].keys())
    #         ssim_values = list(results_dict[test_settings].values())
    #         plt.subplot(1, num_subplots, i)
    #         i += 1
    #         plt.bar(frame_ids, ssim_values, label=test_settings)
    #         plt.title(test_settings)
    #         plt.ylim(0.6, 1)
    #         plt.xlabel("Frame ID")
    #         plt.ylabel("SSIM Value")
    # for test_settings in results_dict.keys():
    #     if "_l1" in test_settings:
    #         frame_ids = list(results_dict[test_settings].keys())
    #         l1_values = list(results_dict[test_settings].values())
    #         plt.subplot(1, num_subplots, i)
    #         i += 1
    #         plt.bar(frame_ids, l1_values, label=test_settings)
    #         plt.title(test_settings)
            
    # plt.suptitle("SSIM Values for Replica Office 2")
    test_name = None
    for test_settings in results_dict.keys():
        if "_ssim" in test_settings:
            print(f"Test settings: {test_settings}")
            label = test_settings.split("_")[2] + "_" + test_settings.split("_")[3] + "_" + test_settings.split("_")[4]
            frame_ids = list(results_dict[test_settings].keys())
            ssim_values = list(results_dict[test_settings].values())
            plt.scatter(frame_ids, ssim_values, label=label)
            plt.ylim(0.4, 1.01)
            plt.xlabel("Frame ID")
            plt.ylabel("SSIM Values")
            test_name = test_settings[:15]
    plt.legend()
    plt.title(f"SSIM Values for {test_name}")
    plt.show()