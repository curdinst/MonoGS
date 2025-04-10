# Install submodules

## Laptop

conda create -n gauss2 python=3.7.13
conda activate gauss2
conda install nvidia/label/cuda-11.8.0::cuda-toolkit

which nvcc

conda install python=3.10
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1  pytorch-cuda=11.8 -c pytorch -c nvidia

conda install gxx_linux-64 (conda install gxx_linux-64=10 didnt work)

pip install submodules/simple-knn/
pip install submodules/diff-gaussian-rasterization/

pip install matplotlib
pip install rich
pip install trimesh
pip install munch
pip install open3d
pip install plyfile
pip install opencv-python
pip install jaxtyping
pip install wandb
pip install glfw
pip install imgviz
pip install OpenGL
pip install pyOpenGL
pip install glm


### try2

conda create -n MonoGS1 python=3.10

with channel: conda install -c nvidia/label/cuda-11.6.2 cuda-toolkit (didnt work always installs wrong nvcc version)

conda install nvidia/label/cuda-11.8.0::cuda-nvcc
pip install submodules/simple-knn/
pip install submodules/diff-gaussian-rasterization/

pip install matplotlib
pip install rich
pip install trimesh
pip install munch
pip install open3d
pip install plyfile
pip install opencv-python
pip install jaxtyping
pip install wandb
pip install glfw
pip install imgviz
pip install OpenGL
pip install pyOpenGL
pip install glm
