import numpy as np
import torch
from plyfile import PlyData
from dataclasses import dataclass
from plyfile import PlyData, PlyElement
from jaxtyping import Float, UInt8
from io import BytesIO
from torch import Tensor
import torchvision.transforms as tf
from PIL import Image


@dataclass
class GaussianData:
    xyz: np.ndarray
    rot: np.ndarray
    scale: np.ndarray
    opacity: np.ndarray
    sh: np.ndarray
    def flat(self) -> np.ndarray:
        ret = np.concatenate([self.xyz, self.rot, self.scale, self.opacity, self.sh], axis=-1)
        return np.ascontiguousarray(ret)
    
    def __len__(self):
        return len(self.xyz)
    
    @property 
    def sh_dim(self):
        return self.sh.shape[-1]


def naive_gaussian():
    gau_xyz = np.array([
        0, 0, 0,
        1, 0, 0,
        0, 1, 0,
        0, 0, 1,
    ]).astype(np.float32).reshape(-1, 3)
    gau_rot = np.array([
        1, 0, 0, 0,
        1, 0, 0, 0,
        1, 0, 0, 0,
        1, 0, 0, 0
    ]).astype(np.float32).reshape(-1, 4)
    gau_s = np.array([
        0.03, 0.03, 0.03,
        0.2, 0.03, 0.03,
        0.03, 0.2, 0.03,
        0.03, 0.03, 0.2
    ]).astype(np.float32).reshape(-1, 3)
    gau_c = np.array([
        1, 0, 1, 
        1, 0, 0, 
        0, 1, 0, 
        0, 0, 1, 
    ]).astype(np.float32).reshape(-1, 3)
    gau_c = (gau_c - 0.5) / 0.28209
    gau_a = np.array([
        1, 1, 1, 1
    ]).astype(np.float32).reshape(-1, 1)
    return GaussianData(
        gau_xyz,
        gau_rot,
        gau_s,
        gau_a,
        gau_c
    )


def load_ply(path):
    max_sh_degree = 0
    plydata = PlyData.read(path)
    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])),  axis=1)
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    # print(plydata.elements[0].properties)
    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
    # print(len(extra_f_names), max_sh_degree)
    assert len(extra_f_names)==3 * (max_sh_degree + 1) ** 2 - 3
    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
    # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
    features_extra = features_extra.reshape((features_extra.shape[0], 3, (max_sh_degree + 1) ** 2 - 1))
    features_extra = np.transpose(features_extra, [0, 2, 1])

    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

    # pass activate function
    xyz = xyz.astype(np.float32)
    rots = rots / np.linalg.norm(rots, axis=-1, keepdims=True)
    rots = rots.astype(np.float32)
    scales = np.exp(scales)
    scales = scales.astype(np.float32)
    opacities = 1/(1 + np.exp(- opacities))  # sigmoid
    opacities = opacities.astype(np.float32)
    shs = np.concatenate([features_dc.reshape(-1, 3), 
                        features_extra.reshape(len(features_dc), -1)], axis=-1).astype(np.float32)
    shs = shs.astype(np.float32)
    return GaussianData(xyz, rots, scales, opacities, shs)

def save_ply(path, gaussian_data):
    xyz = gaussian_data.xyz.astype(np.float32)
    rots = gaussian_data.rot.astype(np.float32)
    scales = np.log(gaussian_data.scale).astype(np.float32)  # Inverse of exp()
    opacities = -np.log(1 / gaussian_data.opacity - 1).astype(np.float32)  # Inverse of sigmoid
    shs = gaussian_data.sh.astype(np.float32)
    
    num_points = xyz.shape[0]
    max_sh_degree = int(np.sqrt(shs.shape[1] // 3) - 1)
    
    # Prepare structured array for PLY format
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('opacity', 'f4')]
    dtype += [(f'f_dc_{i}', 'f4') for i in range(3)]
    dtype += [(f'f_rest_{i}', 'f4') for i in range(3 * (max_sh_degree + 1) ** 2 - 3)]
    dtype += [(f'scale_{i}', 'f4') for i in range(scales.shape[1])]
    dtype += [(f'rot_{i}', 'f4') for i in range(rots.shape[1])]
    
    ply_array = np.empty(num_points, dtype=dtype)
    ply_array['x'], ply_array['y'], ply_array['z'] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    ply_array['opacity'] = opacities.flatten()
    
    ply_array['f_dc_0'] = shs[:, 0]
    ply_array['f_dc_1'] = shs[:, 1]
    ply_array['f_dc_2'] = shs[:, 2]
    
    for i in range(3 * (max_sh_degree + 1) ** 2 - 3):
        ply_array[f'f_rest_{i}'] = shs[:, 3 + i]
    
    for i in range(scales.shape[1]):
        ply_array[f'scale_{i}'] = scales[:, i]
    
    for i in range(rots.shape[1]):
        ply_array[f'rot_{i}'] = rots[:, i]
    
    ply_element = PlyElement.describe(ply_array, 'vertex')
    PlyData([ply_element]).write(path)
    print("Saved ply file to ", path)

to_tensor = tf.ToTensor()
def convert_images(
        images
    ) -> Float[Tensor, "batch 3 height width"]:
        torch_images = []
        for image in images:
            image = Image.open(BytesIO(image.numpy().tobytes()))
            torch_images.append(to_tensor(image))
        return torch.stack(torch_images)
# def save_ply(path, gaussian_data: GaussianData):
#     vertex = np.array([
#         (gaussian_data.xyz[i, 0], gaussian_data.xyz[i, 1], gaussian_data.xyz[i, 2],
#             gaussian_data.opacity[i, 0],
#             gaussian_data.sh[i, 0], gaussian_data.sh[i, 1], gaussian_data.sh[i, 2],
#             gaussian_data.scale[i, 0], gaussian_data.scale[i, 1], gaussian_data.scale[i, 2],
#             gaussian_data.rot[i, 0], gaussian_data.rot[i, 1], gaussian_data.rot[i, 2], gaussian_data.rot[i, 3])
#         for i in range(len(gaussian_data.xyz))
#     ], dtype=[
#         ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
#         ('opacity', 'f4'),
#         ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
#         ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
#         ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4')
#     ])

#     elements = [PlyElement.describe(vertex, 'vertex')]
#     PlyData(elements).write(path)

if __name__ == "__main__":
    gs = load_ply("C:\\Users\\MSI_NB\\Downloads\\viewers\\models\\train\\point_cloud\\iteration_7000\\point_cloud.ply")
    a = gs.flat()
    print(a.shape)
