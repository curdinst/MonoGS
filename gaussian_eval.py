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
import numpy as np

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
from mpl_toolkits.mplot3d import Axes3D

PATH = "/home/curdin/master_thesis/outputs/"
# DATE = "25_04_07/"
DATE = "25_05_07/"
FILENAME = "gaussians.ply"

FOLDERNAME = "2025-05-07_12-30_0_it" + "/"

TEST_NAME = FILENAME[56:-4]
device = "cuda"
filepath = PATH + DATE + FOLDERNAME + FILENAME
folder_path = PATH + DATE + FOLDERNAME
# filepath = PATH + DATE + FILENAME

tum_path = "/home/curdin/repos/MonoGS/datasets/tum/rgbd_dataset_freiburg1_desk/"
config_path = os.path.join("/home/curdin/repos/MonoGS/configs/mono/tum/fr1_desk.yaml")
with open(config_path, "r") as yml:
    config = yaml.safe_load(yml)
config = load_config(config_path)
model_params = munchify(config["model_params"])
dataset = TUMDataset(model_params, tum_path, config=config)

gaussians = GaussianModel(sh_degree=0)
GaussianModel.load_ply(gaussians, filepath)

keyframe_poses = []

txt_file_path = os.path.join(folder_path, "keyframe_poses.txt")
if os.path.exists(txt_file_path):
    with open(txt_file_path, "r") as file:
        lines = file.readlines()
        timestamps = [line.split()[0] for line in lines]
        numbers = [
            [float(num) for num in line.split() if num.lstrip('-').replace('.', '', 1).isdigit()]
            for line in lines
        ]
        # print("Extracted numbers:", numbers)
# elif os.path.exists(txt_file_path_dated):
#     with open(txt_file_path_dated, "r") as file:
#         lines = file.readlines()
#         timestamps = [line.split()[1] for line in lines]
#         numbers = []
#         for line in lines:
#             line_nums = [float(num) for num in line.split() if num.lstrip('-').replace('.', '', 1).isdigit()]
#             numbers.append(line_nums[1:])
#             # print(line_nums[1:-1])
#         # numbers = [
#         #     [(i, float(num)) for num in enumerate(line.split()) if num.lstrip('-').replace('.', '', 1).isdigit() and num != "0"]
#         #     for line in lines
#         # ]
#         # print("Extracted numbers:", numbers)
else:
    print(f"Files not found: {txt_file_path} or {txt_file_path}")
print(numbers)
poses = torch.tensor(numbers, dtype=torch.float32, device=device)
r_wc = [R.from_quat(num[4:]).as_matrix().tolist() for num in numbers]
r_wc = torch.tensor(r_wc, dtype=torch.float32, device=device)
r_wc = r_wc.reshape(-1, 3, 3)
r_cw = r_wc.transpose(dim0=1, dim1=2)
t_wc = poses[:, 2:5]
t_cw = poses[:, 2:5].clone()
for i in range(r_wc.shape[0]):
    t_cw[i] = -r_cw[i] @ t_wc[i]
print(timestamps)

t_w_wc = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=torch.float32, device=device)

def print_gaussians():
        print(f" GAussian model ==================================================0")
        print(f"activ: {gaussians.active_sh_degree }")
        print(f"max_s: {gaussians.max_sh_degree}")
        print(f"_xyz : {gaussians._xyz}")
        print(f"_feat: {gaussians._features_dc}")
        print(f"_feat: {gaussians._features_rest}")
        print(f"_scal: {gaussians._scaling}")
        print(f"_rota: {gaussians._rotation}")
        print(f"_opac: {gaussians._opacity}")
        print(f"max_r: {gaussians.max_radii2D }")
        print(f"xyz_g: {gaussians.xyz_gradient_accum}")
        print(f"uniqu: {gaussians.unique_kfIDs }")
        print(f"n_obs: {gaussians.n_obs}")
        print(f"optim: {gaussians.optimizer}")
        print(f"scali: {gaussians.scaling_activation}")
        print(f"scali: {gaussians.scaling_inverse_activation}")
        print(f"covar: {gaussians.covariance_activation}")
        print(f"opaci: {gaussians.opacity_activation}")
        print(f"inver: {gaussians.inverse_opacity_activation}")
        print(f"rotat: {gaussians.rotation_activation}")
        print(f"confi: {gaussians.config}")
        print(f"ply_i: {gaussians.ply_input}")
        print(f"isotr: {gaussians.isotropic}")
        print(f" shapes   ++++++++++++++++++ shapes ++++++++++++++++++++ shapes")
        print(f"_xyz : {gaussians._xyz.shape}")
        print(f"_feat: {gaussians._features_dc.shape}")
        print(f"_feat: {gaussians._features_rest.shape}")
        print(f"_scal: {gaussians._scaling.shape}")
        print(f"_rota: {gaussians._rotation.shape}")
        print(f"_opac: {gaussians._opacity.shape}")
        print(f"max_r: {gaussians.max_radii2D .shape}")
        print(f"xyz_g: {gaussians.xyz_gradient_accum.shape}")
        print(f"uniqu: {gaussians.unique_kfIDs .shape}")
        print(f"n_obs: {gaussians.n_obs.shape}")
        print(f" -------------------------------------------------------------")

gt_imgs = []

# 1305031466.527538 -0.892814576625824 0.5856775641441345 -0.36823898553848267 0.11222653090953827 0.10041153430938721 0.05415824428200722 0.9871119856834412



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


def render_camera(keyframe_idx):
    viewpoint = Camera.init_from_dataset(
                        dataset, idx=keyframe_idx, projection_matrix=projection_matrix
                    )
    viewpoint.T = translations[keyframe_idx]
    viewpoint.R = rotations[keyframe_idx]
    pipeline_params = munchify(config["pipeline_params"])

    background = torch.tensor([0, 0, 0], dtype=torch.float32, device=device)

    render_pkg = render(
                        viewpoint, gaussians, pipeline_params, background
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


print(r_wc, t_wc)

def plot_cameras(r_wc, t_wc):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot camera positions
    ax.scatter(t_wc[:, 0].cpu().numpy(), t_wc[:, 1].cpu().numpy(), t_wc[:, 2].cpu().numpy(), c='r', label='Camera Positions')
    

    # Draw camera frustums
    scale = 0.1  # Scale for the frustum size
    
    
    
    for i in range(len(t_wc)):
        origin = t_wc[i].cpu().numpy()
        x_dir = r_wc[i, :, 0].cpu().numpy() * scale
        y_dir = r_wc[i, :, 1].cpu().numpy() * scale
        z_dir = r_wc[i, :, 2].cpu().numpy() * scale
        # Draw coordinate axes in RGB colors
        ax.quiver(origin[0], origin[1], origin[2], x_dir[0], x_dir[1], x_dir[2], color='r', length=scale, normalize=True, label='X-axis' if i == 0 else "")
        ax.quiver(origin[0], origin[1], origin[2], y_dir[0], y_dir[1], y_dir[2], color='g', length=scale, normalize=True, label='Y-axis' if i == 0 else "")
        ax.quiver(origin[0], origin[1], origin[2], z_dir[0], z_dir[1], z_dir[2], color='b', length=scale, normalize=True, label='Z-axis' if i == 0 else "")
        # Define frustum corners
        top_left = origin + x_dir - y_dir + z_dir
        top_right = origin - x_dir - y_dir + z_dir
        bottom_left = origin + x_dir + y_dir + z_dir
        bottom_right = origin - x_dir + y_dir + z_dir
        
        # Draw frustum edges
        if i == 0:  # Draw the first camera in red
            ax.plot([origin[0], top_left[0]], [origin[1], top_left[1]], [origin[2], top_left[2]], 'r-')
            ax.plot([origin[0], top_right[0]], [origin[1], top_right[1]], [origin[2], top_right[2]], 'r-')
            ax.plot([origin[0], bottom_left[0]], [origin[1], bottom_left[1]], [origin[2], bottom_left[2]], 'r-')
            ax.plot([origin[0], bottom_right[0]], [origin[1], bottom_right[1]], [origin[2], bottom_right[2]], 'r-')
            
            ax.plot([top_left[0], top_right[0]], [top_left[1], top_right[1]], [top_left[2], top_right[2]], 'r-')
            ax.plot([top_left[0], bottom_left[0]], [top_left[1], bottom_left[1]], [top_left[2], bottom_left[2]], 'r-')
            ax.plot([bottom_left[0], bottom_right[0]], [bottom_left[1], bottom_right[1]], [bottom_left[2], bottom_right[2]], 'r-')
            ax.plot([top_right[0], bottom_right[0]], [top_right[1], bottom_right[1]], [top_right[2], bottom_right[2]], 'r-')
        else:
            ax.plot([origin[0], top_left[0]], [origin[1], top_left[1]], [origin[2], top_left[2]], 'k-')
            ax.plot([origin[0], top_right[0]], [origin[1], top_right[1]], [origin[2], top_right[2]], 'k-')
            ax.plot([origin[0], bottom_left[0]], [origin[1], bottom_left[1]], [origin[2], bottom_left[2]], 'k-')
            ax.plot([origin[0], bottom_right[0]], [origin[1], bottom_right[1]], [origin[2], bottom_right[2]], 'k-')
            
            ax.plot([top_left[0], top_right[0]], [top_left[1], top_right[1]], [top_left[2], top_right[2]], 'k-')
            ax.plot([top_left[0], bottom_left[0]], [top_left[1], bottom_left[1]], [top_left[2], bottom_left[2]], 'k-')
            ax.plot([bottom_left[0], bottom_right[0]], [bottom_left[1], bottom_right[1]], [bottom_left[2], bottom_right[2]], 'k-')
            ax.plot([top_right[0], bottom_right[0]], [top_right[1], bottom_right[1]], [top_right[2], bottom_right[2]], 'k-')
        # ax.plot([origin[0], top_left[0]], [origin[1], top_left[1]], [origin[2], top_left[2]], 'k-')
        # ax.plot([origin[0], top_right[0]], [origin[1], top_right[1]], [origin[2], top_right[2]], 'k-')
        # ax.plot([origin[0], bottom_left[0]], [origin[1], bottom_left[1]], [origin[2], bottom_left[2]], 'k-')
        # ax.plot([origin[0], bottom_right[0]], [origin[1], bottom_right[1]], [origin[2], bottom_right[2]], 'k-')
        
        # ax.plot([top_left[0], top_right[0]], [top_left[1], top_right[1]], [top_left[2], top_right[2]], 'k-')
        # ax.plot([top_left[0], bottom_left[0]], [top_left[1], bottom_left[1]], [top_left[2], bottom_left[2]], 'k-')
        # ax.plot([bottom_left[0], bottom_right[0]], [bottom_left[1], bottom_right[1]], [bottom_left[2], bottom_right[2]], 'k-')
        # ax.plot([top_right[0], bottom_right[0]], [top_right[1], bottom_right[1]], [top_right[2], bottom_right[2]], 'k-')
    
    # Align axes: x-axis horizontal, y-axis downwards
    # ax.view_init(elev=90, azim=0)
    ax.view_init(elev=30, azim=-60)  # Default-like view
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Camera Positions and Orientations')
    ax.legend()
    plt.show()
    
# plot_cameras(t_w_wc[None, ...] @ r_wc, (t_w_wc[None, ...] @ t_wc[..., None])[..., 0])

# eval_renders()

# viewpoint = Camera.init_from_dataset(
#                         dataset, idx=0, projection_matrix=projection_matrix
#                     )

# viewpoint.T = torch.zeros(3, dtype=torch.float32, device=device)
# viewpoint.R = torch.eye(3, dtype=torch.float32, device=device)
# pipeline_params = munchify(config["pipeline_params"])

# background = torch.tensor([0, 0, 0], dtype=torch.float32, device=device)

# render_pkg = render(
#                     viewpoint, gaussians, pipeline_params, background
#                 )
