
# HOOD: Hierarchical Graphs for Generalized Modelling of Clothing Dynamics

### <img align=center src=./static/icons/project.png width='32'/> [Project](https://dolorousrtur.github.io/hood/) &ensp; <img align=center src=./static/icons/paper.png width='24'/> [Paper](https://arxiv.org/abs/2212.07242) &ensp;  

This is a repository with training and inference code for the paper [**"HOOD: Hierarchical Graphs for Generalized Modelling of Clothing Dynamics"**](https://arxiv.org/abs/2212.07242) (CVPR2023).

*Latest update: 30.09.2023, added notebook and config for running inference with any mesh sequence or SMPL pose sequence from a garment mesh in arbitrary pose*

## Installation

### Linux 系统安装

#### 使用 conda 环境文件安装
We provide a conda environment file `hood.yml` to install all the dependencies. 
You can create and activate the environment with the following commands:

```bash
conda env create -f hood.yml
conda activate hood
```

If you want to build the environment from scratch, here are the necessary commands: 
<details>
  <summary>Build enviroment from scratch (Linux)</summary>

```bash
# Create and activate a new environment
conda create -n hood python=3.9 -y
conda activate hood

# install pytorch (see https://pytorch.org/)
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia -y

# install pytorch_geometric (see https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)
conda install pyg -c pyg -y

# install pytorch3d (see https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md)
conda install -c fvcore -c iopath -c conda-forge fvcore iopath -y
conda install -c bottler nvidiacub -y
conda install pytorch3d -c pytorch3d -y


# install auxiliary packages with conda
conda install -c conda-forge munch pandas tqdm omegaconf matplotlib einops ffmpeg -y

# install more auxiliary packages with pip
pip install smplx aitviewer chumpy huepy

# create a new kernel for jupyter notebook
conda install ipykernel -y; python -m ipykernel install --user --name hood --display-name "hood"
```
</details>

---

### Windows 系统安装

由于原始的 `hood.yml` 是基于 Linux 系统配置的，Windows 用户需要按照以下步骤手动安装环境。

**已验证的版本配置**（推荐）：
| 组件 | 版本 |
|-----|------|
| Python | 3.10.x |
| PyTorch | 2.4.0+cu118 |
| CUDA | 11.8 |
| torch-geometric | 2.3.0 |
| pytorch3d | 0.7.8 |
| NumPy | 1.23.5 |
| moderngl-window | 2.4.6 (使用 pyglet 后端) |

<details>
  <summary>Windows 安装步骤（点击展开）</summary>

#### 步骤 1：创建 conda 环境

```powershell
# 创建并激活新环境（推荐 Python 3.10）
conda create -n hood python=3.10 -y
conda activate hood
```

#### 步骤 2：安装 PyTorch 2.4.0 + CUDA 11.8（已验证版本）

```powershell
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118
```

#### 步骤 3：安装 PyTorch Geometric 2.3.0 和扩展包

```powershell
# 安装 PyG 2.3.0（必须是此版本，新版本有 API 不兼容问题）
pip install torch_geometric==2.3.0

# 安装扩展包（版本需匹配 PyTorch 2.4.0 + CUDA 11.8）
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu118.html
```

#### 步骤 4：安装 PyTorch3D

PyTorch3D 在 Windows 上没有官方预编译包，推荐使用第三方预编译的 wheel 文件安装。

**下载预编译 wheel 文件**：
- 访问: https://github.com/facebookresearch/pytorch3d/releases 或搜索 "pytorch3d windows wheel"
- 下载与你的 PyTorch/CUDA/Python 版本匹配的 wheel 文件
- 对于 **PyTorch 2.4.0 + CUDA 11.8 + Python 3.10**，下载文件名类似：
  `pytorch3d-0.7.8+5043d15pt2.4.0cu118-cp310-cp310-win_amd64.whl`

```powershell
# 安装依赖
pip install fvcore iopath

# 安装下载的 wheel 文件（将路径替换为你的实际下载路径）
pip install pytorch3d-0.7.8+5043d15pt2.4.0cu118-cp310-cp310-win_amd64.whl
```

> **备选方案**：如果找不到预编译 wheel，可以从源码编译（需要 Visual Studio Build Tools）：
> ```powershell
> pip install "git+https://github.com/facebookresearch/pytorch3d.git"
> ```

#### 步骤 5：安装 NumPy 1.23.5（重要！）

```powershell
# 必须使用此版本，新版本与 chumpy 不兼容
pip install numpy==1.23.5
```

#### 步骤 6：安装基础 conda 包

```powershell
conda install -c conda-forge munch pandas tqdm omegaconf matplotlib einops ffmpeg pyyaml -y
```

#### 步骤 7：安装 pip 依赖

```powershell
pip install -r requirements.txt
```

#### 步骤 8：安装 chumpy（特殊处理）

```powershell
# 使用 --no-build-isolation 避免构建问题
pip install chumpy --no-build-isolation
```

#### 步骤 9：安装 Jupyter（可选，用于运行 notebook）

```powershell
pip install notebook
python -m ipykernel install --user --name hood --display-name "hood"
```

#### 步骤 10：设置环境变量

在 Windows 中设置环境变量：

**方法 A：临时设置（每次启动终端需要重新设置）**
```powershell
$env:HOOD_DATA = "E:\Projects4\HOOD\\hood_data"
$env:HOOD_PROJECT = "E:\Projects4\HOOD\"
```

**方法 B：永久设置**
1. 按 `Win + R`，输入 `sysdm.cpl`，按回车
2. 点击"高级"选项卡 -> "环境变量"
3. 在"用户变量"中点击"新建"
4. 添加 `HOOD_DATA` 和 `HOOD_PROJECT` 变量

</details>

<details>
  <summary>Windows 一键安装脚本（复制到 PowerShell 执行）</summary>

```powershell
# 激活环境
conda activate hood

# 1. 安装 PyTorch 2.4.0 + CUDA 11.8
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118

# 2. 安装 PyG 2.3.0 和扩展包
pip install torch_geometric==2.3.0
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu118.html

# 3. 安装 PyTorch3D（使用预编译 wheel 文件，需提前下载）
pip install fvcore iopath
# 将下面的文件名替换为你下载的 wheel 文件路径
pip install pytorch3d-0.7.8+5043d15pt2.4.0cu118-cp310-cp310-win_amd64.whl

# 4. 安装 NumPy 1.23.5
pip install numpy==1.23.5

# 5. 安装基础包
conda install -c conda-forge munch pandas tqdm omegaconf matplotlib einops ffmpeg pyyaml -y

# 6. 安装其他依赖
pip install -r requirements.txt

# 7. 安装 aitviewer 可视化依赖（使用 pyglet 后端）
pip install moderngl-window==2.4.6 pyglet

# 8. 安装 chumpy
pip install chumpy --no-build-isolation

# 9. 安装 Jupyter
pip install notebook
python -m ipykernel install --user --name hood --display-name "hood"
```

</details>

<details>
  <summary>Windows 常见问题</summary>

#### Q1: `ImportError: cannot import name 'bool' from 'numpy'`
这是 NumPy 版本过新导致的。解决方案：
```powershell
pip install numpy==1.23.5
```

#### Q2: `AttributeError: 'Inspector' object has no attribute 'inspect'`
这是 PyG 版本过新导致的 API 不兼容。解决方案：
```powershell
pip install torch_geometric==2.3.0
```

#### Q3: `DLL load failed while importing _C`
PyTorch3D 或 PyG 扩展包版本与 PyTorch 不匹配。确保使用正确的版本组合。

#### Q4: PyTorch3D 安装失败
- 推荐使用预编译 wheel 文件安装，下载地址：https://github.com/facebookresearch/pytorch3d/releases
- 确保下载的 wheel 文件与你的 PyTorch/CUDA/Python 版本匹配
- 如果从源码编译，需要先安装 Visual Studio Build Tools 和 C++ 编译器
- 源码安装命令: `pip install "git+https://github.com/facebookresearch/pytorch3d.git"`

#### Q5: chumpy 安装失败
```powershell
pip install chumpy --no-build-isolation
# 或从 GitHub 安装
pip install git+https://github.com/mattloper/chumpy.git
```

#### Q6: scikit-image 依赖冲突
```powershell
pip install scikit-image==0.21.0
```

#### Q7: aitviewer 显示问题 / `'PyQt5Window' object has no attribute '_ctx'`
Windows 上 aitviewer 需要使用 pyglet 后端而不是 PyQt5 后端。解决方案：
```powershell
pip install moderngl-window==2.4.6 pyglet
```

如果仍有 OpenGL 相关问题：
```powershell
pip install PyOpenGL PyOpenGL_accelerate
```

</details>

### Download data
#### HOOD data
Download the auxiliary data for HOOD using this [link](https://drive.google.com/file/d/1RdA4L6Fy50VsKZ8k7ySp5ps5YtWoHSgs/view?usp=sharing).
Unpack it anywhere you want and set the `HOOD_DATA` environmental variable to the path of the unpacked folder.
Also, set the `HOOD_PROJECT` environmental variable to the path you cloned this repository to:

```bash
export HOOD_DATA=/path/to/hood_data
export HOOD_PROJECT=/path/to/this/repository
```

#### SMPL models
Download the SMPL models using this [link](https://smpl.is.tue.mpg.de/). Unpack them into the `$HOOD_DATA/aux_data/smpl` folder.

In the end your `$HOOD_DATA` folder should look like this:
```
$HOOD_DATA
    |-- aux_data
        |-- datasplits // directory with csv data splits used for training the model
        |-- smpl // directory with smpl models
            |-- SMPL_NEUTRAL.pkl
            |-- SMPL_FEMALE.pkl
            |-- SMPL_MALE.pkl
        |-- garment_meshes // folder with .obj meshes for garments used in HOOD
        |-- garments_dict.pkl // dictionary with garmentmeshes and their auxilliary data used for training and inference
        |-- smpl_aux.pkl // dictionary with indices of SMPL vertices that correspond to hands, used to disable hands during inference to avoid body self-intersections
    |-- trained_models // directory with trained HOOD models
        |-- cvpr_submission.pth // model used in the CVPR paper
        |-- postcvpr.pth // model trained with refactored code with several bug fixes after the CVPR submission
        |-- fine15.pth // baseline model without denoted as "Fine15" in the paper (15 message-passing steps, no long-range edges)
        |-- fine48.pth // baseline model without denoted as "Fine48" in the paper (48 message-passing steps, no long-range edges)
```

## Inference
The jupyter notebook [Inference.ipynb](Inference.ipynb) contains an example of how to run inference of a trained HOOD model given a garment and a pose sequence.

It also has examples of such use-cases as adding a new garment from an .obj file and converting sequences from [AMASS](https://amass.is.tue.mpg.de/) and [VTO](https://github.com/isantesteban/vto-dataset) datasets to the format used in HOOD.

To run inference starting from arbitrary garment pose and arbitrary mesh sequence refer to the [InferenceFromMeshSequence.ipynb](InferenceFromMeshSequence.ipynb) notebook.  

## Training
To train a new HOOD model from scratch, you need to first download the [VTO](https://github.com/isantesteban/vto-dataset) dataset and convert it to our format.

You can find the instructions on how to do that and the commands used to start the training in the [Training.ipynb](Training.ipynb) notebook.

## Validation Sequences
You can download the sequences used for validation (Table 1 in the main paper and Tables 1 and 2 in the Supplementary) 
using [this link](https://drive.google.com/file/d/1jFkDWPZW2HwYsYqcXAC3hX0NlumBnqT3/view?usp=sharing)

You can find instructions on how to generate validation sequences and compute metrics over them in the [ValidationSequences.ipynb](ValidationSequences.ipynb) notebook.



## Repository structure
See the [RepoIntro.md](RepoIntro.md) for more details on the repository structure.



## Mesh Mode Training

Mesh mode allows training HOOD with pre-computed mesh sequences (body mesh + cloth mesh) instead of SMPL pose sequences. This is useful when you have mesh animation data from other sources.

### Data Format

Mesh mode requires paired body and cloth mesh sequences in `.pkl` format:

- **Body sequence**: Contains per-frame body mesh vertices `(N_frames, N_body_verts, 3)`
- **Cloth sequence**: Contains per-frame cloth mesh vertices `(N_frames, N_cloth_verts, 3)`
- **Cloth template**: OBJ file defining the cloth mesh topology (faces)

### Configuration

Edit `configs/mesh.yaml` to configure your training:

#### Single Sequence Mode
```yaml
dataloader:
  dataset:
    mesh:
      body_sequence_path: 'path/to/body_sequence.pkl'
      cloth_sequence_path: 'path/to/cloth_sequence.pkl'
      cloth_template_path: 'path/to/cloth_template.obj'
      
      n_coarse_levels: 3
      lookup_steps: 5
      noise_scale: 3e-3  # Position noise for data augmentation
      wholeseq: false    # Set to true for validation mode
```

#### Multi-Sequence Mode (with datasplit CSV)
```yaml
dataloader:
  dataset:
    mesh:
      data_root: 'vto_dataset_mesh'  # relative to $HOOD_DATA
      split_path: 'datasplits/mesh_train.csv'  # relative to $HOOD_DATA/aux_data
      
      n_coarse_levels: 3
      lookup_steps: 5
      noise_scale: 3e-3
      wholeseq: false
```

**Datasplit CSV format:**
```csv
id,garment,length
tshirt_shape00_01_01,tshirt,300
tshirt_shape00_01_02,tshirt,250
dress_shape01_02_01,dress,400
```

Where:
- `id`: Sequence identifier (used to construct file paths)
- `garment`: Garment name (used for template lookup)
- `length`: Number of frames in sequence

File paths are constructed as:
- Body: `{data_root}/body_sequence/{id}.pkl`
- Cloth: `{data_root}/{garment}_sequence/{id}.pkl`

### Run Training

```bash
python train_mesh.py
```

### Data Augmentation Limitations

Since mesh mode uses pre-computed mesh sequences (not SMPL parameters), the following data augmentation methods are **NOT supported**:
- Body shape variation
- Body pose perturbation
- Garment global rotation

**Supported augmentation:**
- Position noise injection (controlled by `noise_scale` parameter)

---

## Interactive Animation Viewer

A GUI-based animation viewer for visualizing mesh sequences and inference results.

### Features

- Import multiple animation files simultaneously (pkl, hdf5, obj)
- Play/pause animation with synchronized playback
- Translate and adjust individual animations
- Support for static meshes (displayed as first frame)

### Supported File Formats

| Format | Description |
|--------|-------------|
| `.pkl` | Mesh sequence, SMPL data, or inference output |
| `.h5/.hdf5` | Mesh sequence or SMPL data |
| `.obj` | Static mesh (displayed as single frame) |

### Usage

```bash
# Run from project root
python -m Tests.viewer.InteractiveAnimationViewer

# Or set PYTHONPATH first
$env:PYTHONPATH = "E:\Projects4\HOOD"  # PowerShell
python Tests/viewer/InteractiveAnimationViewer.py
```

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Space` | Play/Pause animation |
| `.` | Next frame |
| `,` | Previous frame |
| `Ctrl+I` | Import files |
| `Esc` | Quit |

### GUI Operations

1. **Import Animations**: `File -> Import Animations` or `Ctrl+I`
2. **Manage Animations**: Use the Animation Manager panel to:
   - Toggle visibility with checkboxes
   - Select animation to edit properties
   - Remove selected animation
3. **Translate Animation**: Drag X/Y/Z sliders in the panel
4. **Change Color**: Use the color picker for selected animation

---

## Citation
If you use this repository in your paper, please cite:
```
@inproceedings{grigorev2023hood,
  title={Hood: Hierarchical graphs for generalized modelling of clothing dynamics},
  author={Grigorev, Artur and Black, Michael J and Hilliges, Otmar},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={16965--16974},
  year={2023}
}
```
