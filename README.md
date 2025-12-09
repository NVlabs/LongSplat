<!-- PROJECT LOGO -->

<p align="center">
  <h1 align="center">[ICCV 2025] LongSplat: Robust Unposed 3D Gaussian Splatting for Casual Long Videos</h1>
  <p align="center">
    <a href="https://linjohnss.github.io/"><strong>Chin-Yang Lin</strong></a>
    ·
    <a href="https://sunset1995.github.io/"><strong>Cheng Sun</strong></a>
    ·
    <a href="https://fuenyang1127.github.io/"><strong>Fu-En Yang</strong></a>
    <br>
    <a href="https://minhungchen.netlify.app/"><strong>Min-Hung Chen</strong></a>
    .
    <a href="https://sites.google.com/site/yylinweb/"><strong>Yen-Yu Lin</strong></a>
    ·
    <a href="https://yulunalexliu.github.io/"><strong>Yu-Lun Liu</strong></a>
  </p>
  <h3 align="center"><a href="https://linjohnss.github.io/longsplat">Project Page</a> | <a href="http://arxiv.org/abs/2508.14041">Paper</a> | <a href="https://drive.google.com/drive/folders/1b-QgNzWVpDaYkHQf3ycKORd7s5WekjGa?usp=drive_link">Evaluation Results</a></h3>
  <div align="center"></div>
</p>

<p align="center">
  <a href="">
    <img src="./assets/teaser.gif" alt="Logo" width="90%">
  </a>
</p>

## Overview
LongSplat is an unposed 3D Gaussian Splatting framework for robust reconstruction from casually captured long videos. Featuring incremental joint optimization, pose estimation with MASt3R, and adaptive octree anchoring, LongSplat achieves high-quality novel view synthesis from free viewpoints while remaining memory-efficient and scalable.

<p align="center">
  <a href="">
    <img src="./assets/teaser.png" alt="Logo" width="100%">
  </a>
</p>

## News

⚠️ **2025.12.10**: Fixed training and eval pipeline for reproduction results.

🎉 **2025.08.27**: Release converter to 3DGS format for compatibility with general 3DGS viewers. See [Convert to 3DGS Format](#convert-to-3dgs-format) section.

🎉 **2025.08.20**: Code release!



## Installation

1. Clone LongSplat.
```bash
git clone --recursive https://github.com/NVlabs/LongSplat.git
cd LongSplat
```

2. Create the environment
```bash
conda create -n longsplat python=3.10.13 cmake=3.14.0 -y
conda activate longsplat
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia  # use the correct version of cuda for your system
pip install -r requirements.txt
pip install submodules/simple-knn
pip install submodules/diff-gaussian-rasterization
pip install submodules/fused-ssim
```

3. Optional but highly suggested, compile the cuda kernels for RoPE (as in CroCo v2).
```bash
# DUST3R relies on RoPE positional embeddings for which you can compile some cuda kernels for faster runtime.
cd submodules/mast3r/dust3r/croco/models/curope/
python setup.py build_ext --inplace
cd ../../../../../../
```

## Dataset Preparation
DATAROOT is `./data` by default. Please first make data folder by `mkdir data`.


### Free Dataset
Download Free dataset from [Dropbox](https://www.dropbox.com/sh/jmfao2c4dp9usji/AAC7Ydj6rrrhy1-VvlAVjyE_a?dl=0), and save it into the `./data/free` folder.

### Hike Dataset
Download Hike dataset from [Google Drive](https://drive.google.com/drive/folders/1kGY-VijIbXNsNb7ghEywi1fvkH4BaIEz?usp=share_link), and save it into the `./data/hike` folder.

### Tanks and Temples

Download the data preprocessed by [Nope-NeRF](https://github.com/ActiveVisionLab/nope-nerf/?tab=readme-ov-file#Data) as below, and the data is saved into the `./data/tanks` folder.
```bash
wget https://www.robots.ox.ac.uk/~wenjing/Tanks.zip
```

<!-- ### CO3D
Download our preprocessed [data](https://ucsdcloud-my.sharepoint.com/:u:/g/personal/yafu_ucsd_edu/EftJV9Xpn0hNjmOiGKZuzyIBW5j6hAVEGhewc8aUcFShEA?e=x1aXVx), and put it saved into the `./data/co3d` folder. -->

## Run

### Training and Evaluation
The training scripts include both training, rendering, and evaluation steps:

Each `.sh` script runs three main Python scripts in sequence:
- `train.py`: Trains the LongSplat model
- `render.py`: Renders the trained model to generate novel views
- `metrics.py`: Evaluates the rendering quality and computes metrics

```bash
# For Free dataset
bash scripts/train_free.sh

# For Hike dataset
bash scripts/train_hike.sh

# For Tanks and Temples dataset
bash scripts/train_tnt.sh
```

### Run on your own video

**Input video suggestions:**
LongSplat can handle varied resolutions and fps. We suggest subsampling to around 10 fps and resizing longer videos to 512 px width for faster training. For best results, avoid videos with dynamic objects and ensure sufficient camera overlap between frames to avoid non-overlapping sparse input.

* To run LongSplat on your own video, you need first to convert your video to frames and save them to `./data/$CUSTOM_DATA/images/`

* Before running the script, you need to modify the `scene=` parameter in `scripts/train_custom.sh` to point to your custom data directory. For example, change `scene='./data/IMG_4190'` to `scene='./data/YOUR_CUSTOM_DATA'`.

* **Adjust training parameters based on video length and resolution:** For longer videos, you may need to increase the `--post_iter` parameter to ensure sufficient optimization. Consider adjusting this value proportionally to your video length.

* Run the following command:
```bash
bash scripts/train_custom.sh
```

### Convert to 3DGS Format

LongSplat uses an anchor + MLP structure for efficient reconstruction. We provide a conversion script to transform LongSplat results into the standard 3DGS format, which outputs a `point_cloud.ply` file that can be used with general 3DGS viewers.

**Note**: Converting to 3DGS format will change both quality and model size. We recommend applying pre-pruning to reduce the model size before conversion.

```bash
# Convert LongSplat output to 3DGS format
python convert_3dgs.py -m PATH_TO_TRAINED_MODEL --prune_ratio 0.6
```


This demonstrates the converted result visualized on [Three.js 3D Gaussian Splatting viewer](https://projects.markkellogg.org/threejs/demo_gaussian_splats_3d.php?art=1&cu=0,1,0&cp=0,1,0&cla=1,0,0&aa=false&2d=false&sh=0).

<p align="center">
  <a href="">
    <img src="./assets/demo_3dgs.gif" alt="Demo Video" width="90%">
  </a>
</p>

## Acknowledgement
Our render is built upon [3DGS](https://github.com/graphdeco-inria/gaussian-splatting). The data processing and visualization codes are partially borrowed from [Scaffold-GS](https://github.com/city-super/Scaffold-GS). We thank all the authors for their great repos.

## Citation

Please consider starring ⭐ this repo and citing our paper 📝 if you find it useful.
```
@inproceedings{lin2025longsplat,
  title={LongSplat: Robust Unposed 3D Gaussian Splatting for Casual Long Videos},
  author={Chin-Yang Lin and Cheng Sun and Fu-En Yang and Min-Hung Chen and Yen-Yu Lin and Yu-Lun Liu},
  booktitle={ICCV},
  year={2025}
}
```
