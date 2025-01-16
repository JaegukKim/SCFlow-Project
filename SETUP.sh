# Environment Setup

## Instructions

### Run the following commands to create a conda environment with required dependencies.

# create conda env
conda create -n SCFlow python=3.9 -y
conda activate SCFlow

## install pytorch
#conda install -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia

## install pytorch3d
conda install -y -c fvcore -c iopath -c conda-forge fvcore iopath
conda install jupyter -y
pip install scikit-image matplotlib plotly webdataset
pip install black usort flake8 flake8-bugbear flake8-comprehensions
conda install pytorch3d -c pytorch3d -y

## misc installations
pip install gdown timm tqdm omegaconf hydra-core accelerate plotly lpips simple-gpu-scheduler kornia opencv-python
pip install tensorboard tensorboardX orjson webdataset pytz pypng imageio scikit-image scikit-learn learn2learn trimesh
pip install pyopengl vispy

## install bop_toolkit
pip install cython
git clone https://github.com/thodan/bop_toolkit.git
#cd bop_toolkit
#pip install -r requirements.txt -e .        ## remove imageio line in requirements.txt
#cd ..
conda install -c conda-forge transformers

##SCFlow
pip install einops
