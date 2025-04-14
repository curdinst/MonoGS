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
DATE = "25_04_14/"
# FILENAME = "rgbd_dataset_freiburg1_desk_2025-04-07_16-48-53gaussmap.ply"
# FILENAME = "rgbd_dataset_freiburg1_desk_2025-04-07_21-21-46_all_gaussmap.ply"
# FILENAME = "rgbd_dataset_freiburg1_desk_2025-04-07_17-25-26gaussmap_abol_rot.ply"
# FILENAME = "rgbd_dataset_freiburg1_desk_2025-04-07_17-35-24gaussmap_abol_scale.ply"
# FILENAME = "rgbd_dataset_freiburg1_desk_2025-04-08_17-24-48gaussmap.ply"
# FILENAME = "rgbd_dataset_freiburg1_desk_2025-04-08_17-29-00_abolrotw_gaussmap.ply"

# FILENAME = "rgbd_dataset_freiburg1_desk_2025-04-14_12-13-04gaussmap_save_first_gaussians.ply"
FILENAME = "rgbd_dataset_freiburg1_desk_2025-04-14_14-01-11_wa.ply"

TEST_NAME = FILENAME[56:-4]
device = "cuda"

filepath = PATH + DATE + FILENAME
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
gaussian_model.init_lr(0.01)
opt_params = munchify(config["opt_params"])
gaussian_model.training_setup(opt_params)



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


import numpy as np
from scipy.spatial.transform import Rotation as R

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
    gt_frame_imgs.append(plt.imread("/home/curdin/repos/MonoGS/datasets/tum/rgbd_dataset_freiburg1_desk/rgb/" + str(timestamps[i]) + ".png"))
print(gt_imgs[0].shape)


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
                        viewpoint, gaussian_model, pipeline_params, background
                    )
    # print(render_pkg)
    (
        image,
    ) = (
        render_pkg["render"],
    )
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

def plot_img(images1, images2, gt_images, num_it, keyframe_window, validation_frames, val_renders_init, val_renders_end):
    # img_list = []
    title_txt_size = 10
    num_views = len(keyframe_window) + len(validation_frames)
    num_kw_views = len(keyframe_window)
    plt.figure()
    plt.suptitle(f"Render optimized with {num_it} iterations on Views {keyframe_window}")
    plt.axis("off")
    i = 0
    for key in keyframe_window:
        img = einops.rearrange(images1[key].cpu().detach().numpy(), "c h w -> h w c")
        # plt.subplot(num_views, 3, 3*i+1)
        plt.subplot(3, num_views, i+1)
        plt.title(f"Initial View "+str(key)+f"\n SSIM: {images1[str(key)+'SSIM']} \n L1: {images1[str(key)+'L1']}", fontsize=title_txt_size)
        plt.imshow(img)
        plt.axis("off")
        img = einops.rearrange(images2[key].cpu().detach().numpy(), "c h w -> h w c")
        # plt.subplot(num_views, 3, 3*i+2)
        plt.subplot(3, num_views, num_views+i+1)
        plt.title(f"Optimized View "+str(key)+f"\n SSIM: {images2[str(key)+'SSIM']} \n L1: {images2[str(key)+'L1']}", fontsize=title_txt_size)
        plt.imshow(img)
        plt.axis("off")
        img = gt_images[key]
        # plt.subplot(num_views, 3, 3*i+3)
        plt.subplot(3, num_views, num_views*2+i+1)
        plt.title("GT View" + str(key), fontsize=title_txt_size)
        plt.imshow(img)
        plt.axis("off")
        i += 1

    i = num_kw_views
    for key in validation_frames:
        img = einops.rearrange(val_renders_init[key].cpu().detach().numpy(), "c h w -> h w c")
        # plt.subplot(num_views, 3, 3*i+1)
        plt.subplot(3, num_views, i+1)
        plt.title(f"Initial validation View "+str(key)+f"\n SSIM: {val_renders_init[str(key)+'SSIM']} \n L1: {val_renders_init[str(key)+'L1']}", fontsize=title_txt_size)
        plt.imshow(img)
        plt.axis("off")
        img = einops.rearrange(val_renders_end[key].cpu().detach().numpy(), "c h w -> h w c")
        # plt.subplot(num_views, 3, 3*i+2)
        plt.subplot(3, num_views, num_views+i+1)
        plt.title(f"Optimized validation View "+str(key)+f"\n SSIM: {val_renders_end[str(key)+'SSIM']} \n L1: {val_renders_end[str(key)+'L1']}", fontsize=title_txt_size)
        plt.imshow(img)
        plt.axis("off")
        img = gt_images[key]
        # plt.subplot(num_views, 3, 3*i+3)
        plt.subplot(3, num_views, num_views*2+i+1)
        plt.title("GT validation View" + str(key), fontsize=title_txt_size)
        plt.imshow(img)
        plt.axis("off")
        i += 1
    plt.show()
    # for key, image in img_dict.items():


    # image1_to_show = einops.rearrange(image1.cpu().detach().numpy(), "c h w -> h w c")
    # image2_to_show = einops.rearrange(image2.cpu().detach().numpy(), "c h w -> h w c")
    # gt_image_to_show = einops.rearrange(gt_image.cpu().detach().numpy(), "c h w -> h w c")

    # plt.subplot(2, num_views+1, )


    # plt.subplot(131)
    # plt.title("Rendered Image 1")
    # plt.imshow(image1_to_show)
    # plt.subplot(132)
    # plt.title("Rendered Image 2")
    # plt.imshow(image2_to_show)
    # plt.subplot(133)
    # plt.title("GT Image")
    # plt.imshow(gt_image_to_show)
    # plt.show()

def optimize_gaussians(keyframe_window=[0,1,2], iterations=1, lambda_dssim = 0.2, validation_frames = []):
    print(f"Optimizing gaussians for window {keyframe_window} on {iterations} iterations")
    densify_grad_threshold = 0.0002
    gaussian_th = 0.7
    gaussian_extent = 6.0
    size_threshold = 20

    renders_init = {}
    renders_end = {}
    val_renders_init = {}
    val_renders_end = {}

    viewpoint_stack = {}
    for cam_idx in keyframe_window:
        viewpoint = Camera.init_from_dataset(
            dataset, idx=cam_idx, projection_matrix=projection_matrix
        )
        viewpoint.T = translations[cam_idx]
        viewpoint.R = rotations[cam_idx]
        viewpoint.original_image = torch.from_numpy(einops.rearrange(gt_imgs[cam_idx], "h w c -> c h w")).to(device=device)
        viewpoint_stack[cam_idx] = viewpoint
    

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
            viewpoint = viewpoint_stack[cam_idx]
            render_pkg = render(viewpoint, gaussian_model, pipeline_params, background)
            (
                image,
                viewspace_point_tensor,
                visibility_filter,
                radii,
                depth,
                opacity,
                n_touched,
            ) = (
                render_pkg["render"],
                render_pkg["viewspace_points"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
                render_pkg["depth"],
                render_pkg["opacity"],
                render_pkg["n_touched"],
            )

            # loss_mapping += get_loss_mapping_rgb(config, image, None, viewpoint)
            l1_loss_val = l1_loss(image, viewpoint.original_image)
            ssim_val = ssim(image, viewpoint.original_image)
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
                    iteration_count % gaussian_update_every
                    == gaussian_update_offset
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
    plot_img(renders_init, renders_end, gt_imgs, iterations, keyframe_window, validation_frames, val_renders_init, val_renders_end)
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

optimize_gaussians(list(range(11)), 30, validation_frames = [])


# for i in range(0, 10, 2):
#     window = [i, i+1]
#     random_idx = random.randint(0, 11)
#     while random_idx in window:
#         random_idx = random.randint(0, 11)
#     window.append(random_idx)
#     print("Optimizing gaussians for window", window)
#     optimize_gaussians(window, 30, validation_frames = [])


GaussianModel.save_ply(
    gaussian_model, PATH + DATE + "optimized_gaussians/" + FILENAME[:-4] + f"_optimized.ply"
)


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