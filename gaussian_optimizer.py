import random
import time
import yaml
import os
import einops
from matplotlib import pyplot as plt
import torch
import torch.multiprocessing as mp
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.loss_utils import l1_loss, ssim
from utils.logging_utils import Log
from utils.multiprocessing_utils import clone_obj
from utils.pose_utils import update_pose
from utils.slam_utils import get_loss_mapping, get_loss_mapping_rgb, get_loss_tracking
# from gaussian_optimizer_utils.util_gau import load_ply
from utils.camera_utils import Camera 
from utils.dataset import TUMDataset
from utils.config_utils import load_config
from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2, getWorld2View2
from munch import munchify
from gaussian_splatting.scene.gaussian_model import GaussianModel

PATH = "/home/curdin/master_thesis/outputs/"
# DATE = "25_04_07/"
DATE = "25_04_15/"
# FILENAME = "rgbd_dataset_freiburg1_desk_2025-04-07_16-48-53gaussmap.ply"
# FILENAME = "rgbd_dataset_freiburg1_desk_2025-04-07_21-21-46_all_gaussmap.ply"
# FILENAME = "rgbd_dataset_freiburg1_desk_2025-04-07_17-25-26gaussmap_abol_rot.ply"
# FILENAME = "rgbd_dataset_freiburg1_desk_2025-04-07_17-35-24gaussmap_abol_scale.ply"
# FILENAME = "rgbd_dataset_freiburg1_desk_2025-04-08_17-24-48gaussmap.ply"
# FILENAME = "rgbd_dataset_freiburg1_desk_2025-04-08_17-29-00_abolrotw_gaussmap.ply"

# FILENAME = "rgbd_dataset_freiburg1_desk_2025-04-14_12-13-04gaussmap_save_first_gaussians.ply"
# FILENAME = "rgbd_dataset_freiburg1_desk_2025-04-14_14-01-11_wa.ply"
FILENAME = "rgbd_dataset_freiburg1_desk_2025-04-15_13-06-17_wa.ply"


TEST_NAME = FILENAME[56:-4]
device = "cuda"
folder_path = None
filepath = PATH + "" + "gaussians1.ply"
filepath = "/home/curdin/Downloads/gaussians.ply"
# filepath = PATH + DATE + FILENAME

# gaussians = load_ply(filepath)

tum_path = "/home/curdin/repos/MonoGS/datasets/tum/rgbd_dataset_freiburg1_desk/"
config_path = os.path.join("/home/curdin/repos/MonoGS/configs/mono/tum/fr1_desk.yaml")
with open(config_path, "r") as yml:
    config = yaml.safe_load(yml)
config = load_config(config_path)
model_params = munchify(config["model_params"])
dataset = TUMDataset(model_params, tum_path, config=config)

gaussian_model = GaussianModel(sh_degree=0)
GaussianModel.load_ply(gaussian_model, filepath)

def print_gaussians():
        print(f" GAussian model ==================================================0")
        print(f"activ: {gaussian_model.active_sh_degree }")
        print(f"max_s: {gaussian_model.max_sh_degree}")
        print(f"_xyz : {gaussian_model._xyz}")
        print(f"_feat: {gaussian_model._features_dc}")
        print(f"_feat: {gaussian_model._features_rest}")
        print(f"_scal: {gaussian_model._scaling}")
        print(f"_rota: {gaussian_model._rotation}")
        print(f"_opac: {gaussian_model._opacity}")
        print(f"max_r: {gaussian_model.max_radii2D }")
        print(f"xyz_g: {gaussian_model.xyz_gradient_accum}")
        print(f"uniqu: {gaussian_model.unique_kfIDs }")
        print(f"n_obs: {gaussian_model.n_obs}")
        print(f"optim: {gaussian_model.optimizer}")
        print(f"scali: {gaussian_model.scaling_activation}")
        print(f"scali: {gaussian_model.scaling_inverse_activation}")
        print(f"covar: {gaussian_model.covariance_activation}")
        print(f"opaci: {gaussian_model.opacity_activation}")
        print(f"inver: {gaussian_model.inverse_opacity_activation}")
        print(f"rotat: {gaussian_model.rotation_activation}")
        print(f"confi: {gaussian_model.config}")
        print(f"ply_i: {gaussian_model.ply_input}")
        print(f"isotr: {gaussian_model.isotropic}")
        print(f" shapes   ++++++++++++++++++ shapes ++++++++++++++++++++ shapes")
        print(f"_xyz : {gaussian_model._xyz.shape}")
        print(f"_feat: {gaussian_model._features_dc.shape}")
        print(f"_feat: {gaussian_model._features_rest.shape}")
        print(f"_scal: {gaussian_model._scaling.shape}")
        print(f"_rota: {gaussian_model._rotation.shape}")
        print(f"_opac: {gaussian_model._opacity.shape}")
        print(f"max_r: {gaussian_model.max_radii2D .shape}")
        print(f"xyz_g: {gaussian_model.xyz_gradient_accum.shape}")
        print(f"uniqu: {gaussian_model.unique_kfIDs .shape}")
        print(f"n_obs: {gaussian_model.n_obs.shape}")
        print(f" -------------------------------------------------------------")

print_gaussians()
# exit()
# print(f"xyz: {gaussian_model._xyz.shape}")
# print(f"feature dc: {gaussian_model._features_dc.shape}")
# print(f"feature rest: {gaussian_model._features_rest.shape}")
# print(f"opacity: {gaussian_model._opacity.shape}")
# print(f"scaling: {gaussian_model._scaling.shape}")
# print(f"rotation: {gaussian_model._rotation.shape}")
# print(f"")
# gaussian_model._xyz = gaussian_model._xyz[0:1000]
# opt_params = munchify(config["opt_params"])
# gaussian_model.training_setup(opt_params)

USE_MASK = True

# Load a text file and extract the numbers
txt_file_path = PATH + DATE + "rgbd_dataset_freiburg1_desk.txt"
txt_file_path_dated = PATH + DATE + FILENAME[:47] + ".txt"

KEYFRAME_IDX = 5

# Ensure the file exists
if os.path.exists(txt_file_path):
    with open(txt_file_path, "r") as file:
        lines = file.readlines()
        timestamps = [line.split()[0] for line in lines]
        numbers = [
            [float(num) for num in line.split() if num.lstrip('-').replace('.', '', 1).isdigit()]
            for line in lines
        ]
        # print("Extracted numbers:", numbers)
elif os.path.exists(txt_file_path_dated):
    with open(txt_file_path_dated, "r") as file:
        lines = file.readlines()
        timestamps = [line.split()[1] for line in lines]
        numbers = []
        for line in lines:
            line_nums = [float(num) for num in line.split() if num.lstrip('-').replace('.', '', 1).isdigit()]
            numbers.append(line_nums[1:])
            # print(line_nums[1:-1])
        # numbers = [
        #     [(i, float(num)) for num in enumerate(line.split()) if num.lstrip('-').replace('.', '', 1).isdigit() and num != "0"]
        #     for line in lines
        # ]
        # print("Extracted numbers:", numbers)
else:
    print(f"Files not found: {txt_file_path} or {txt_file_path_dated}")
poses = torch.tensor(numbers, dtype=torch.float32, device=device)
rotations = [R.from_quat(num[4:]).as_matrix() for num in numbers]
rotations = torch.tensor(rotations, dtype=torch.float32, device=device)
rotations = rotations.reshape(-1, 3, 3)
rotations = rotations.transpose(dim0=1, dim1=2)
translations = poses[:, 1:4]
for i in range(translations.shape[0]):
    translations[i] = -rotations[i] @ translations[i]
print(timestamps)

frame_poses_path = PATH + DATE + FILENAME[:47] + "_all_poses.txt"
if os.path.exists(frame_poses_path):
    with open(frame_poses_path, "r") as file:
        lines = file.readlines()
        frame_timestamps = [line.split()[1] for line in lines]
        frame_numbers = []
        for line in lines:
            line_nums = [float(num) for num in line.split(" ")]
            frame_numbers.append(line_nums[1:-1])
            # if line_nums[0] == 91:
            #     print(line)
            #     print(line.split(" "))
            #     print(line_nums)
            # print(f"len numbers: {len(line_nums)} {line_nums[0]}")
            # print(line_nums[1:-1])
        # numbers = [
        #     [(i, float(num)) for num in enumerate(line.split()) if num.lstrip('-').replace('.', '', 1).isdigit() and num != "0"]
        #     for line in lines
        # ]
        # print("Extracted numbers:", frame_numbers)
else:
    print(f"Files not found: {frame_poses_path}")
frame_poses = torch.tensor(frame_numbers, dtype=torch.float32, device=device)
frame_rotations = [R.from_quat(num[4:]).as_matrix() for num in frame_numbers]
frame_rotations = torch.tensor(frame_rotations, dtype=torch.float32, device=device)
frame_rotations = frame_rotations.reshape(-1, 3, 3)
frame_rotations = frame_rotations.transpose(dim0=1, dim1=2)
frame_translations = frame_poses[:, 1:4]
for i in range(frame_translations.shape[0]):
    frame_translations[i] = -frame_rotations[i] @ frame_translations[i]
# print(frame_timestamps)

gt_pose_path = tum_path + "groundtruth.txt"
gt_translations = {}
gt_rotations = {}
gt_TF = {}
if os.path.exists(gt_pose_path):
    with open(gt_pose_path, "r") as file:
        lines = file.readlines()
        gt_timestamps = [line.split()[0] for line in lines]
        # gt_numbers = [
        #     [float(num) for num in line.split() if num.lstrip('-').replace('.', '', 1).isdigit()]
        #     for line in lines
        # ]
        i = 0
        for line in lines[3:]:
            linevals = [float(num) for num in line.split() if num.lstrip('-').replace('.', '', 1).isdigit()]
            stamp = float(line.split()[0])
            if i == 0:
                T_init = torch.eye(4)
                T_init[0:3, 3] = torch.tensor(linevals[1:4])
                T_init[0:3, 0:3] = torch.from_numpy(R.from_quat(linevals[4:]).as_matrix())
                i += 1
            else:
                T = torch.eye(4)
                T[0:3, 3] = torch.tensor(linevals[1:4])
                T[0:3, 0:3] = torch.from_numpy(R.from_quat(linevals[4:]).as_matrix())
                gt_TF[stamp] = torch.linalg.inv(T_init) @ T
                gt_translations[stamp] = gt_TF[stamp][0:3, 3]
                gt_rotations[stamp] = gt_TF[stamp][0:3, 0:3]
else:
    print(f"Ground truth file not found: {gt_pose_path}")

mask_path = PATH + DATE + f"masks_{FILENAME[:-4]}/"
mask_files = [f for f in os.listdir(mask_path) if f.endswith('.pt')]
masks = {}
for mask_file in mask_files:
    mask_path_full = os.path.join(mask_path, mask_file)
    mask = torch.load(mask_path_full).to(device=device)
    mask = einops.rearrange(mask, "(h w) ->h w", h=dataset.height, w=dataset.width)
    masks[int(mask_file[:-3])] = mask

i = 0
for key in sorted(masks.keys()):
    print(f"mask key: {key}")
    masks[i] = masks.pop(key)
    i += 1
print(f"Loaded {len(masks)} mask files.")
# 1/0
print(masks.keys())
# print(masks["123.pt"].shape)

import numpy as np
from scipy.spatial.transform import Rotation as R
from datetime import datetime
from torchvision.transforms import functional as F

def interpolate_pose(T_i, T_j, alpha):
    """
    Interpolates pose between T_i and T_j using SLERP and linear translation interpolation.

    Args:
        T_i: np.array (4x4) - start pose
        T_j: np.array (4x4) - end pose
        alpha: float - interpolation factor between 0 and 1

    Returns:
        T_f: np.array (4x4) - interpolated pose
    """
    # Extract rotations and translations
    R_i = T_i[:3, :3]
    t_i = T_i[:3, 3]

    R_j = T_j[:3, :3]
    t_j = T_j[:3, 3]

    # Convert to quaternions
    q_i = R.from_matrix(R_i)
    q_j = R.from_matrix(R_j)

    # SLERP interpolation of rotation
    q_interp = R.slerp(0, 1, [q_i, q_j])(alpha)
    R_f = q_interp.as_matrix()

    # Linear interpolation of translation
    t_f = (1 - alpha) * t_i + alpha * t_j

    # Construct interpolated pose matrix
    T_f = np.eye(4)
    T_f[:3, :3] = R_f
    T_f[:3, 3] = t_f

    return T_f


# i = 0
# for ts in timestamps:
#     print(f"Translation {ts}: {gt_translations[ts]} - {translations[i]}")
#     i += 1
#     print(f"Rotation {ts}: {gt_rotations[ts]} - {rotations[i]}")
                




gt_imgs = []
for i in range(len(timestamps)):
    gt_imgs.append(plt.imread("/home/curdin/repos/MonoGS/datasets/tum/rgbd_dataset_freiburg1_desk/rgb/" + str(timestamps[i]) + ".png"))
gt_frame_imgs = []
for j in range(len(frame_timestamps)):
    gt_frame_imgs.append(plt.imread("/home/curdin/repos/MonoGS/datasets/tum/rgbd_dataset_freiburg1_desk/rgb/" + str(frame_timestamps[j]) + ".png"))
print(gt_imgs[0].shape)
# Downsample the gt_imgs by a factor of 1.25
downsample_factor = 1.25
gt_imgs = [
    F.resize(torch.from_numpy(img).permute(2, 0, 1), 
             size=(int(img.shape[0] / downsample_factor), int(img.shape[1] / downsample_factor))
    ).permute(1, 2, 0).numpy()
    for img in gt_imgs
]
# plt.imshow(gt_imgs[0])
# plt.show()

projection_matrix = getProjectionMatrix2(
            znear=0.01,
            zfar=100.0,
            fx=dataset.fx,
            fy=dataset.fy,
            cx=dataset.cx,
            cy=dataset.cy,
            W=dataset.width,
            H=dataset.height,
        ).transpose(0, 1)
projection_matrix = projection_matrix.to(device=device)


def render_camera(keyframe_idx, is_keyframe = True):
    viewpoint = Camera.init_from_dataset(
                        dataset, idx=keyframe_idx, projection_matrix=projection_matrix
                    )
    if is_keyframe:
        viewpoint.T = translations[keyframe_idx]
        viewpoint.R = rotations[keyframe_idx]
    else:
        viewpoint.T = frame_translations[keyframe_idx]
        viewpoint.R = frame_rotations[keyframe_idx]
    pipeline_params = munchify(config["pipeline_params"])

    background = torch.tensor([0, 0, 0], dtype=torch.float32, device=device)

    render_pkg = render(
                        viewpoint, gaussian_model, pipeline_params, background
                    )
    # print(render_pkg)
    image = render_pkg["render"]
    gt_image = gt_imgs[keyframe_idx]
    gt_image = torch.from_numpy(gt_image)
    gt_image_rearranged = einops.rearrange(gt_image, "h w c -> c h w").to(device=device)
    del render_pkg
    del viewpoint
    return (
        image.detach().cpu(),
        # viewspace_point_tensor,
        # visibility_filter,
        # radii,
        # depth,
        # opacity,
        # n_touched,
        gt_image.detach().cpu(),
        gt_image_rearranged.detach().cpu(),
    )
# image, viewspace_point_tensor, visibility_filter, radii, depth, opacity, n_touched, gt_image, gt_image_rearranged = render_camera(7)

def plot_img(images1, images2, gt_images, num_it, keyframe_window, validation_frames, val_renders_init, val_renders_end, use_keyframes = True, folder_path=None):
    # img_list = []
    title_txt_size = 10
    num_views = len(keyframe_window) + len(validation_frames)
    num_kw_views = len(keyframe_window)
    
    fig = plt.figure(figsize=(20, 20))
    plt.suptitle(f"Render optimized with {num_it} iterations on Views {keyframe_window}")
    plt.axis("off")
    renders = {}
    render_titles = {}
    
    i = 0
    for key in keyframe_window:
        img = einops.rearrange(images1[key].cpu().detach().numpy(), "c h w -> h w c")
        # plt.subplot(num_views, 3, 3*i+1)
        plt.subplot(3, num_views, i+1)
        plt.title(f"Initial View "+str(key)+f"\n SSIM: {images1[str(key)+'SSIM']} \n L1: {images1[str(key)+'L1']}", fontsize=title_txt_size)
        renders["Init_View_"+str(key)] = img
        render_titles["Init_View_"+str(key)] = f"Initial View "+str(key)+f"\n SSIM: {images1[str(key)+'SSIM']} \n L1: {images1[str(key)+'L1']}"
        plt.imshow(img)
        plt.axis("off")
        img = einops.rearrange(images2[key].cpu().detach().numpy(), "c h w -> h w c")
        # plt.subplot(num_views, 3, 3*i+2)
        plt.subplot(3, num_views, num_views+i+1)
        plt.title(f"Optimized View "+str(key)+f"\n SSIM: {images2[str(key)+'SSIM']} \n L1: {images2[str(key)+'L1']}", fontsize=title_txt_size)
        renders["Opt_View_"+str(key)] = img
        render_titles["Opt_View_"+str(key)] = f"Optimized View "+str(key)+f"\n SSIM: {images2[str(key)+'SSIM']} \n L1: {images2[str(key)+'L1']}"
        plt.imshow(img)
        plt.axis("off")
        img = gt_images[key] if use_keyframes else gt_frame_imgs[key]
        # plt.subplot(num_views, 3, 3*i+3)
        plt.subplot(3, num_views, num_views*2+i+1)
        plt.title("GT View" + str(key), fontsize=title_txt_size)
        renders["GT_View_"+str(key)] = img
        render_titles["GT_View_"+str(key)] = "GT View " + str(key)
        plt.imshow(img)
        plt.axis("off")
        i += 1

    i = num_kw_views
    for key in validation_frames:
        img = einops.rearrange(val_renders_init[key].cpu().detach().numpy(), "c h w -> h w c")
        # plt.subplot(num_views, 3, 3*i+1)
        plt.subplot(3, num_views, i+1)
        plt.title(f"Initial validation View "+str(key)+f"\n SSIM: {val_renders_init[str(key)+'SSIM']} \n L1: {val_renders_init[str(key)+'L1']}", fontsize=title_txt_size)
        renders["Init_val_View_"+str(key)] = img
        render_titles["Init_val_View_"+str(key)] = f"Initial validation View "+str(key)+f"\n SSIM: {val_renders_init[str(key)+'SSIM']} \n L1: {val_renders_init[str(key)+'L1']}"
        plt.imshow(img)
        plt.axis("off")
        img = einops.rearrange(val_renders_end[key].cpu().detach().numpy(), "c h w -> h w c")
        # plt.subplot(num_views, 3, 3*i+2)
        plt.subplot(3, num_views, num_views+i+1)
        plt.title(f"Optimized validation View "+str(key)+f"\n SSIM: {val_renders_end[str(key)+'SSIM']} \n L1: {val_renders_end[str(key)+'L1']}", fontsize=title_txt_size)
        renders["Opt_val_View_"+str(key)] = img
        render_titles["Opt_val_View_"+str(key)] = f"Optimized validation View "+str(key)+f"\n SSIM: {val_renders_end[str(key)+'SSIM']} \n L1: {val_renders_end[str(key)+'L1']}"
        plt.imshow(img)
        plt.axis("off")
        img = gt_images[key] if use_keyframes else gt_frame_imgs[key]
        # plt.subplot(num_views, 3, 3*i+3)
        plt.subplot(3, num_views, num_views*2+i+1)
        plt.title("GT_View_" + str(key), fontsize=title_txt_size)
        renders["GT_View_"+str(key)] = img
        render_titles["GT_View_"+str(key)] = "GT View " + str(key)
        plt.imshow(img)
        plt.axis("off")
        i += 1
    plt.show()
    if folder_path is not None:
        fig.savefig(os.path.join(folder_path, f"rendered_images.png"))
        print(f"")

    img_folder = os.path.join(folder_path, "renders")
    os.mkdir(img_folder)
    for key, image in renders.items():
        fig = plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.title(render_titles[key])
        plt.axis("off")
        fig.savefig(os.path.join(img_folder, f"{key}.png"))

def save_metrics(images1, images2, gt_images, num_it, keyframe_window, validation_frames, val_renders_init, val_renders_end, use_keyframes = True, folder_path=None):

    if folder_path is not None:
        metrics_file_path = os.path.join(folder_path, "metrics.txt")
        with open(metrics_file_path, "w") as f:
            f.write(f"Metrics after {num_it} iterations:\n\n")
            f.write("Keyframe Window Metrics:\n")
            f.write(f"Frame : metric:  Initial - Optimized\n")
            for key in keyframe_window:
                f.write(f"{key} : SSIM: {images1[str(key)+'SSIM']} - {images2[str(key)+'SSIM']}\n")
                f.write(f"{key} : L1:   {images1[str(key)+'L1']} - {images2[str(key)+'L1']}\n")
            f.write("Validation Frame Metrics:\n")
            f.write(f"Frame : metric:  Initial - Optimized\n")
            for key in validation_frames:
                f.write(f"{key} : SSIM: {val_renders_init[str(key)+'SSIM']} - {val_renders_end[str(key)+'SSIM']}\n")
                f.write(f"{key} : L1:   {val_renders_init[str(key)+'L1']} - {val_renders_end[str(key)+'L1']}\n")
        print(f"Metrics saved to {metrics_file_path}")
    for key in keyframe_window:
        images1[str(key)+'SSIM']
        images1[str(key)+'L1']
        images2[str(key)+'SSIM']
        images2[str(key)+'L1']

 

def optimize_gaussians(keyframe_window=[0,1,2], iterations=1, lambda_dssim = 0.2, validation_frames = [], densify_prune=False, use_keyframes = False, folder_path=None):
    print(f"Optimizing gaussians for window {keyframe_window} on {iterations} iterations")
    densify_grad_threshold = 0.0002
    gaussian_th = 0.7
    gaussian_extent = 6.0
    size_threshold = 20

    renders_init = {}
    renders_end = {}
    val_renders_init = {}
    val_renders_end = {}

    if use_keyframes:
        viewpoint_stack = {}
        for cam_idx in keyframe_window:
            viewpoint = Camera.init_from_dataset(
                dataset, idx=cam_idx, projection_matrix=projection_matrix
            )
            viewpoint.T = translations[cam_idx]
            viewpoint.R = rotations[cam_idx]
            viewpoint.original_image = torch.from_numpy(einops.rearrange(gt_imgs[cam_idx], "h w c -> c h w")).to(device=device)
            viewpoint_stack[cam_idx] = viewpoint
    else:
        frame_viewpoint_stack = {}
        for cam_idx in range(len(frame_timestamps)):
            viewpoint = Camera.init_from_dataset(
                dataset, idx=cam_idx, projection_matrix=projection_matrix
            )
            viewpoint.T = frame_translations[cam_idx]
            viewpoint.R = frame_rotations[cam_idx]
            viewpoint.original_image = torch.from_numpy(einops.rearrange(gt_frame_imgs[cam_idx], "h w c -> c h w")).to(device=device)
            frame_viewpoint_stack[cam_idx] = viewpoint
    

    for cam_idx in validation_frames:
        if cam_idx in keyframe_window:
            print("image already in keyframe window")
            continue
        image, gt_image, gt_image_rearranged = render_camera(cam_idx, use_keyframes)
        ssim_val = ssim(image, gt_image_rearranged)
        l1_loss_val = l1_loss(image, gt_image_rearranged)
        print("Optimized gaussians view", cam_idx)
        print("SSIM: ", ssim_val.item())
        print("L1 Loss: ", l1_loss_val.item())
        val_renders_init[cam_idx] = image.clone()
        val_renders_init[str(cam_idx)+"SSIM"] = round(ssim_val.item(), 3)
        val_renders_init[str(cam_idx)+"L1"] = round(l1_loss_val.item(), 3)

    pipeline_params = munchify(config["pipeline_params"])

    iteration_count = 0
    gaussian_update_every = 150
    gaussian_update_offset = 40
    for i in range(iterations):
        iteration_count += 1
        loss_mapping = 0

        for cam_idx in keyframe_window:
            # viewpoint = Camera.init_from_dataset(
            #                     dataset, idx=keyframe_idx, projection_matrix=projection_matrix
            #                 )

            # viewpoint.T = translations[keyframe_idx]
            # viewpoint.R = rotations[keyframe_idx]
            # viewpoint.original_image = torch.from_numpy(einops.rearrange(gt_imgs[keyframe_idx], "h w c -> c h w")).to(device=device)

            background = torch.tensor([0, 0, 0], dtype=torch.float32, device=device)
            if use_keyframes:
                viewpoint = viewpoint_stack[cam_idx]
            else:
                viewpoint = frame_viewpoint_stack[cam_idx]
            render_pkg = render(viewpoint, gaussian_model, pipeline_params, background)
            image = render_pkg["render"]*masks[cam_idx] if USE_MASK else render_pkg["render"]
            print(image.shape)
            print(f"image original shape: {render_pkg['render'].shape}")
            # loss_mapping += get_loss_mapping_rgb(config, image, None, viewpoint)
            l1_loss_val = l1_loss(image, viewpoint.original_image*masks[cam_idx]) if USE_MASK else l1_loss(image, viewpoint.original_image)
            ssim_val = ssim(image, viewpoint.original_image*masks[cam_idx]) if USE_MASK else ssim(image, viewpoint.original_image)
            # ssim_val = ssim(image, viewpoint.original_image)
            loss = (1 - lambda_dssim) * l1_loss_val + lambda_dssim * (1 - ssim_val)
            loss_mapping = l1_loss_val
            # loss_mapping = ssim(image, viewpoint.original_image)
            if i == 0:
                renders_init[cam_idx] = image.clone()
                renders_init[str(cam_idx)+"SSIM"] = round(ssim_val.item(),3)
                renders_init[str(cam_idx)+"L1"] = round(l1_loss_val.item(),3)
            if i == iterations - 1:
                renders_end[cam_idx] = image.clone()
                renders_end[str(cam_idx)+"SSIM"] = round(ssim_val.item(),3)
                renders_end[str(cam_idx)+"L1"] = round(l1_loss_val.item(),3)
        # print("SSIM: ", ssim(image, viewpoint.original_image))
        # if i == 0:
        #     first_render = image.clone()
        
        # scaling = gaussian_model.get_scaling
        # isotropic_loss = torch.abs(scaling - scaling.mean(dim=1).view(-1, 1))
        # loss_mapping += 10 * isotropic_loss.mean()
            print("loss mapping: ", loss_mapping.item())
            loss_mapping.backward()
            with torch.no_grad():

                update_gaussian = (
                    densify_prune and 
                    iteration_count % gaussian_update_every == gaussian_update_offset
                )
                if update_gaussian:
                    print("densifying and pruning")
                    print(f"num Gaussians: {gaussian_model._xyz.shape[0]}")
                    gaussian_model.densify_and_prune(
                        densify_grad_threshold,
                        gaussian_th,
                        gaussian_extent,
                        size_threshold,
                    )
                    print(f"num Gaussians after: {gaussian_model._xyz.shape[0]}")
                # if update_gaussian:
                #     gaussian_model.densify_and_prune(
                #         opt_params.densify_grad_threshold,
                #         gaussian_th,
                #         gaussian_extent,
                #         size_threshold,
                #     )
                #     gaussian_split = True

                ## Opacity reset
                # if (iteration_count % gaussian_reset) == 0 and (
                #     not update_gaussian
                # ):
                #     Log("Resetting the opacity of non-visible Gaussians")
                #     gaussian_model.reset_opacity_nonvisible(visibility_filter_acm)
                #     gaussian_split = True

                gaussian_model.optimizer.step()
                gaussian_model.optimizer.zero_grad(set_to_none=True)
                gaussian_model.update_learning_rate(iteration_count)
                loss_mapping = 0

    for cam_idx in validation_frames:
        if cam_idx in keyframe_window:
            print("image already in keyframe window")
            continue
        image, gt_image, gt_image_rearranged = render_camera(cam_idx)
        ssim_val = ssim(image, gt_image_rearranged)
        l1_loss_val = l1_loss(image, gt_image_rearranged)
        print("Optimized gaussians view", cam_idx)
        print("SSIM: ", ssim_val.item())
        print("L1 Loss: ", l1_loss_val.item())
        val_renders_end[cam_idx] = image.clone()
        val_renders_end[str(cam_idx)+"SSIM"] = round(ssim_val.item(), 3)
        val_renders_end[str(cam_idx)+"L1"] = round(l1_loss_val.item(),3)
        
        # val_renders_end[cam_idx] = renders_end[cam_idx].clone()
    # loss_init.backward()
    plot_img(renders_init, renders_end, gt_imgs, iterations, keyframe_window, validation_frames, val_renders_init, val_renders_end, use_keyframes, folder_path=folder_path)
    save_metrics(renders_init, renders_end, gt_imgs, iterations, keyframe_window, validation_frames, val_renders_init, val_renders_end, use_keyframes, folder_path=folder_path)
    return

def tracking(cur_frame_idx, viewpoint, tracking_itr_num=30):
    prev_idx = cur_frame_idx - 1
    prev = Camera.init_from_dataset(
        dataset, idx=prev_idx, projection_matrix=projection_matrix
    )
    prev.T = translations[prev_idx]
    prev.R = rotations[prev_idx]
    prev.original_image = torch.from_numpy(einops.rearrange(gt_imgs[prev_idx], "h w c -> c h w")).to(device=device)
    prev.update_RT(prev.R, prev.T)

    opt_params = []
    opt_params.append(
        {
            "params": [viewpoint.cam_rot_delta],
            "lr": config["Training"]["lr"]["cam_rot_delta"],
            "name": "rot_{}".format(viewpoint.uid),
        }
    )
    opt_params.append(
        {
            "params": [viewpoint.cam_trans_delta],
            "lr": config["Training"]["lr"]["cam_trans_delta"],
            "name": "trans_{}".format(viewpoint.uid),
        }
    )
    opt_params.append(
        {
            "params": [viewpoint.exposure_a],
            "lr": 0.01,
            "name": "exposure_a_{}".format(viewpoint.uid),
        }
    )
    opt_params.append(
        {
            "params": [viewpoint.exposure_b],
            "lr": 0.01,
            "name": "exposure_b_{}".format(viewpoint.uid),
        }
    )

    pose_optimizer = torch.optim.Adam(opt_params)
    pipeline_params = munchify(config["pipeline_params"])
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device=device)

    for tracking_itr in range(tracking_itr_num):
        render_pkg = render(
            viewpoint, gaussian_model, pipeline_params, background
        )
        image, depth, opacity = (
            render_pkg["render"],
            render_pkg["depth"],
            render_pkg["opacity"],
        )
        pose_optimizer.zero_grad()
        loss_tracking = get_loss_tracking(
            config, image, depth, opacity, viewpoint
        )
        print(f"Tracking loss: {loss_tracking.item()}")
        loss_tracking.backward()

        with torch.no_grad():
            pose_optimizer.step()
            converged = update_pose(viewpoint)

        if converged:
            print("Converged")

            break

    # self.median_depth = get_median_depth(depth, opacity)
    plt.imshow(einops.rearrange(viewpoint.original_image.detach().cpu().numpy(), "c h w -> h w c")*0.5 + einops.rearrange(image.detach().cpu().numpy(), "c h w -> h w c")*0.5)
    plt.show()
    return render_pkg



# current_frame_idx = 1
# viewpoint = Camera.init_from_dataset(
#                     dataset, idx=current_frame_idx, projection_matrix=projection_matrix
#                 )

# image, gt_image, gt_image_rearranged = render_camera(1)
# plt.imshow(gt_image*0.5 + einops.rearrange(image, "c h w -> h w c")*0.5)
# plt.show()

# viewpoint.T = translations[current_frame_idx]
# viewpoint.R = rotations[current_frame_idx]
# viewpoint.original_image = torch.from_numpy(einops.rearrange(gt_imgs[current_frame_idx], "h w c -> c h w")).to(device=device)
# viewpoint.compute_grad_mask(config)
# tracking(current_frame_idx, viewpoint, tracking_itr_num=120)

# image, gt_image, gt_image_rearranged = render_camera(1)





# print(f"translation 01 {torch.linalg.norm(translations[0] - translations[1])}")

# optimize_gaussians(list(range(11)), 30, validation_frames = [])
# optimize_gaussians([0,10,20,30,40], 20, validation_frames = [], use_keyframes=False)

def run_test(init_lr=0.5, iterations=30, keyframe_window=[0,1,2], validation_frames=[], use_keyframes=False):
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_path = os.path.join(PATH, DATE, f"Gauss_opt_{current_time}")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder created at: {folder_path}")
    else:
        print(f"Folder already exists at: {folder_path}")
    densify_prune = False

    opt_params = {
        "iterations": 30000,
        "position_lr_init": 0.0016,
        "position_lr_final": 0.0000016,
        "position_lr_delay_mult": 0.01,
        "position_lr_max_steps": 30000,
        "feature_lr": 0.0025,
        "opacity_lr": 0.05,
        "scaling_lr": 0.001,
        "rotation_lr": 0.001,
        "percent_dense": 0.01,
        "lambda_dssim": 0.2,
        "densification_interval": 100,
        "opacity_reset_interval": 3000,
        "densify_from_iter": 500,
        "densify_until_iter": 15000,
        "densify_grad_threshold": 0.0002,
    }
    # Save all parameters to a text file
    params_file_path = os.path.join(folder_path, "params.txt")
    with open(params_file_path, "w") as f:
        f.write(f"Initial Learning Rate: {init_lr}\n")
        f.write(f"Iterations: {iterations}\n")
        f.write(f"Keyframe Window: {keyframe_window}\n")
        f.write(f"Validation Frames: {validation_frames}\n")
        f.write(f"Use Keyframes: {use_keyframes}\n")
        f.write(f"Config Path: {config_path}\n")
        f.write(f"Dataset Path: {tum_path}\n")
        f.write(f"Gaussian File Path: {filepath}\n")
        f.write(f"Densify and Prone: {densify_prune}\n")
        f.write(f"Use Mask: {USE_MASK}\n")
        f.write("Optimization Parameters:\n")
        f.write(f"  Iterations: {opt_params['iterations']}\n")
        f.write(f"  Position LR Init: {opt_params['position_lr_init']}\n")
        f.write(f"  Position LR Final: {opt_params['position_lr_final']}\n")
        f.write(f"  Position LR Delay Mult: {opt_params['position_lr_delay_mult']}\n")
        f.write(f"  Position LR Max Steps: {opt_params['position_lr_max_steps']}\n")
        f.write(f"  Feature LR: {opt_params['feature_lr']}\n")
        f.write(f"  Opacity LR: {opt_params['opacity_lr']}\n")
        f.write(f"  Scaling LR: {opt_params['scaling_lr']}\n")
        f.write(f"  Rotation LR: {opt_params['rotation_lr']}\n")
        f.write(f"  Percent Dense: {opt_params['percent_dense']}\n")
        f.write(f"  Lambda DSSIM: {opt_params['lambda_dssim']}\n")
        f.write(f"  Densification Interval: {opt_params['densification_interval']}\n")
        f.write(f"  Opacity Reset Interval: {opt_params['opacity_reset_interval']}\n")
        f.write(f"  Densify From Iter: {opt_params['densify_from_iter']}\n")
        f.write(f"  Densify Until Iter: {opt_params['densify_until_iter']}\n")
        f.write(f"  Densify Grad Threshold: {opt_params['densify_grad_threshold']}\n")
    print(f"Parameters saved to {params_file_path}")

    gaussian_model.init_lr(init_lr)



    opt_params = munchify(opt_params)
    gaussian_model.training_setup(opt_params)
    optimize_gaussians(keyframe_window=keyframe_window, iterations=iterations, validation_frames=validation_frames, densify_prune=False, use_keyframes=use_keyframes, folder_path=folder_path)

    GaussianModel.save_ply(
        gaussian_model, folder_path + "/" + FILENAME[:-4] + f"_optimized.ply"
    )



keyframe_window = list(range(11))
keyframe_window.remove(5)
keyframe_window.remove(9)
validation_window = [5, 9]
# run_test(init_lr=0.05, iterations=40, keyframe_window=keyframe_window, validation_frames=validation_window, use_keyframes=True)

# img = render_camera(0)
# print(f"img shape: {img[0].shape}")

# for i in range(0, 10, 2):
#     window = [i, i+1]
#     random_idx = random.randint(0, 11)
#     while random_idx in window:
#         random_idx = random.randint(0, 11)
#     window.append(random_idx)
#     print("Optimizing gaussians for window", window)
#     optimize_gaussians(window, 30, validation_frames = [])


# GaussianModel.save_ply(
#     gaussian_model, PATH + DATE + "optimized_gaussians/" + FILENAME[:-4] + f"_optimized.ply"
# )


# image, gt_image, gt_image_rearranged = render_camera(0)
# plt.imshow(gt_image*0.5 + einops.rearrange(image, "c h w -> h w c")*0.5)
# plt.show()



def eval_renders():
    losses = {
    "ssim": [],
    "l1_loss": [],
    }
    test_name = "first_gaussians_mask"
    N = len(timestamps)
    N = 2
    plt.figure(figsize=(20, 20)) 
    for idx in range(N):
        print("Rendering image", idx)
        image, gt_image, gt_image_rearranged = render_camera(idx)
        ssim_val = ssim(image, gt_image_rearranged)
        l1_loss_val = l1_loss(image, gt_image_rearranged)
        losses["ssim"].append(ssim_val.item())
        losses["l1_loss"].append(l1_loss_val.item())
        # del image
        # del gt_image
        # del gt_image_rearranged
        image_to_show = einops.rearrange(image.detach().cpu().numpy(), "c h w -> h w c")
        plt.subplot(N, 2, idx * 2 + 1)
        plt.imshow(image_to_show)
        plt.title(f"Rendered Image {idx}")
        plt.axis("off")
        plt.subplot(N, 2, idx * 2 + 2)
        plt.imshow(gt_image)
        plt.title(f"GT Image {idx}")
        plt.axis("off")
    
    plt.show()
    plt.savefig(os.path.join(PATH, DATE, f"{test_name}.png"))

    losses["ssim_mean"] = sum(losses["ssim"]) / len(losses["ssim"])
    losses["l1_loss_mean"] = sum(losses["l1_loss"]) / len(losses["l1_loss"])
    # Save losses as a text file
    losses_file_path = os.path.join(PATH, DATE, f"losses_{test_name}.txt")
    with open(losses_file_path, "w") as f:
        f.write("SSIM Losses:\n")
        f.write("\n".join(map(str, losses["ssim"])) + "\n")
        f.write("\nL1 Losses:\n")
        f.write("\n".join(map(str, losses["l1_loss"])) + "\n")
        f.write(f"\nMean SSIM Loss: {losses['ssim_mean']}\n")
        f.write(f"\nMean L1 Loss: {losses['l1_loss_mean']}\n")
    print(f"Losses saved to {losses_file_path}")
    # print(f"ssim: {ssim(image, gt_image_rearranged)}")
    # print(f"l1_loss: {l1_loss(image, gt_image_rearranged)} ")

# eval_renders()

viewpoint = Camera.init_from_dataset(
                        dataset, idx=0, projection_matrix=projection_matrix
                    )

viewpoint.T = torch.zeros(3, dtype=torch.float32, device=device)
viewpoint.R = torch.eye(3, dtype=torch.float32, device=device)
pipeline_params = munchify(config["pipeline_params"])

background = torch.tensor([0, 0, 0], dtype=torch.float32, device=device)

render_pkg = render(
                    viewpoint, gaussian_model, pipeline_params, background
                )
# print(render_pkg)
# image = render_pkg["render"]
# import cv2
# gt_img = cv2.imread("/home/curdin/1305031452.791720.png")
# gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
# gt_img = cv2.resize(gt_img, (512, 384), interpolation=cv2.INTER_AREA)
# gt_image = torch.from_numpy(gt_img)
# print(gt_image.shape)
# gt_image_rearranged = einops.rearrange(gt_image, "h w c -> c h w").type(torch.float32).to(device=device)/255.0
# print(f"image shape: {image.shape}")
# ssim_val = ssim(image, gt_image_rearranged)
# print("SSIM: ", ssim_val.item())
# l1_loss_val = l1_loss(image, gt_image_rearranged)
# print("L1 Loss: ", l1_loss_val.item())

# plt.subplot(1, 2, 1)
# plt.title(f"Splatt3r ply: SSIM: {round(ssim_val.item(), 3)} L1: {round(l1_loss_val.item(), 3)}")
# plt.imshow(einops.rearrange(image.detach().cpu().numpy(), "c h w -> h w c"))
# plt.subplot(1, 2, 2)
# plt.title("GT Image")
# plt.imshow(gt_image.detach().cpu().numpy())
# plt.show()