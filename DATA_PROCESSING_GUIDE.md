# HOOD 数据处理指南

本文档全面概述了 HOOD 项目中的数据处理流程，包括数据集结构、数据格式以及从原始数据到模型输入的完整工作流程。

## 目录

1. [数据集架构概览](#1-数据集架构概览)
2. [数据输入格式](#2-数据输入格式)
3. [HeteroData 数据结构](#3-heterodata-数据结构)
4. [数据处理流程](#4-数据处理流程)
5. [代码结构](#5-代码结构)
6. [训练数据使用](#6-训练数据使用)
7. [完整数据加载流程](#7-完整数据加载流程)
8. [数据预处理](#8-数据预处理)
9. [数据缓存与存储机制](#9-数据缓存与存储机制)
10. [Mesh格式数据支持](#10-mesh格式数据支持)

---

## 1. 数据集架构概览

数据加载系统采用分层设计，包含以下组件：

```
create() ──► Loader ──► Dataset
               │
               ├── SequenceLoader (loads SMPL parameter sequences)
               ├── GarmentBuilder (builds garment mesh)
               └── BodyBuilder    (builds body mesh)
```

### 核心类

1. **Dataset 类** (datasets/postcvpr.py 第 748-805 行)
   - **训练模式** (wholeseq=False)：按帧索引加载数据，每个样本是单帧
   - **验证模式** (wholeseq=True)：加载整个序列

2. **关键组件**
   - `Config`：存储所有配置参数
   - `SequenceLoader`：从磁盘加载并预处理 SMPL 序列
   - `GarmentBuilder`：构建服装网格数据
   - `BodyBuilder`：构建身体（障碍物）网格数据
   - `VertexBuilder`：从 SMPL 参数构建顶点序列
   - `NoiseMaker`：添加高斯噪声用于数据增强

---

## 2. 数据输入格式

### SMPL 序列文件 (.pkl)

```python
{
    'body_pose': np.array,      # [Nx69] SMPL 姿态参数
    'global_orient': np.array,  # [Nx3] 全局朝向
    'transl': np.array,         # [Nx3] 全局位移
    'betas': np.array           # [10] 体型参数
}
```

### 服装字典 (garments_dict.pkl)

```python
{
    'garment_name': {
        'rest_pos': np.array,    # [Vx3] 标准姿态下顶点位置
        'faces': np.array,       # [Fx3] 面片索引
        'node_type': np.array,   # [Vx1] 顶点类型 (0=普通, 3=固定)
        'lbs': dict,             # 线性蒙皮权重和混合形状
        'center': list,          # 粗糙边的中心节点
        'coarse_edges': dict     # 预计算的粗糙边
    }
}
```

### 数据分割文件 (datasplit.csv)

```csv
id,length,garment
sequence_001,300,tshirt
sequence_002,250,dress
```

---

## 3. HeteroData 数据结构

HeteroData 是 PyTorch Geometric 库中的核心数据结构，用于在 HOOD 项目中表示异构图。它以统一的格式存储所有服装和身体网格数据。

### 什么是异构图？

异构图是指包含多种类型节点和多种类型边的图：

**在 HOOD 中：**
- **节点类型**：`cloth`（服装）、`obstacle`（身体）
- **边类型**：`mesh_edge`（网格边）、`coarse_edge0/1...`（粗糙边）

### 基本用法

```python
from torch_geometric.data import HeteroData

sample = HeteroData()

# 添加节点属性（两种等价方式）
sample['cloth'].pos = torch.tensor(...)
sample['cloth']['pos'] = torch.tensor(...)

# 添加边数据 (源类型, 边类型, 目标类型)
sample['cloth', 'mesh_edge', 'cloth'].edge_index = edges  # [2, E]

# 存储标量属性
sample['sequence_name'] = 'motion_001'
sample['garment_name'] = 'tshirt'
```

### HOOD 中的 HeteroData 结构

```
HeteroData(
    # ========== 服装节点 ==========
    cloth={
        pos:          [V, 3] or [V, N, 3],   # 当前帧顶点位置
        prev_pos:     [V, 3] or [V, N, 3],   # 前一帧顶点位置
        target_pos:   [V, 3] or [V, N, 3],   # 下一帧位置 (Ground Truth)
        lookup:       [V, L, 3],              # 未来 L 帧位置 (仅训练)
        rest_pos:     [V, 3],                 # 标准姿态位置
        vertex_type:  [V, 1],                 # 0=普通, 3=固定
        vertex_level: [V, 1],                 # 粗糙层级
        faces_batch:  [3, F],                 # 面片索引
    },
    
    # ========== 障碍物（身体）节点 ==========
    obstacle={
        pos:          [V_body, 3] or [V_body, N, 3],
        prev_pos:     ...,
        target_pos:   ...,
        lookup:       ...,
        vertex_type:  [V_body, 1],   # 1=普通障碍物, 2=手部 (推理时忽略)
        vertex_level: [V_body, 1],   # 始终为 0
        faces_batch:  [3, F_body],
    },
    
    # ========== 边 ==========
    (cloth, mesh_edge, cloth)]={
        edge_index: [2, E]           # 网格边
    },
    (cloth, coarse_edge0, cloth)]={
        edge_index: [2, E0]          # 第 0 层粗糙边
    },
    (cloth, coarse_edge1, cloth)]={
        edge_index: [2, E1]          # 第 1 层粗糙边 (如果有)
    },
    
    # ========== 元数据 ==========
    sequence_name='motion_001',
    garment_name='tshirt'
)
```

### HeteroData 的优势
| 特性 | 描述 |
|---------|-------------|
| 类型安全 | 不同类型的节点/边数据相互隔离 |
| 灵活扩展 | 易于添加新的节点/边类型 |
| 自动批处理 | 自动合并多个样本 |
| GPU 友好 | 所有张量可以一起移动到 GPU |
| 消息传递 | 直接支持 PyG MessagePassing 层 |

---

## 4. 数据处理流程

### 4.1 序列预处理 (SequenceLoader.process_sequence)

### 4.2 服装网格构建 (GarmentBuilder.build)

生成的数据结构：

| 属性 | 维度 | 描述 |
|-----------|------------|-------------|
| prev_pos | [Vx3] 或 [VxNx3] | 前一帧顶点位置 |
| pos | [Vx3] 或 [VxNx3] | 当前帧顶点位置 |
| target_pos | [Vx3] 或 [VxNx3] | 下一帧顶点位置 (GT) |
| lookup | [VxLx3] | 未来 L 帧位置 (用于训练) |
| rest_pos | [Vx3] | 标准姿态位置 |
| vertex_type | [Vx1] | 顶点类型 |
| mesh_edge | [2xE] | 网格边 |
| coarse_edge{i} | [2xEi] | 多层级粗糙边 |

### 4.3 身体网格构建 (BodyBuilder.build)

| 属性 | 维度 | 描述 |
|-----------|------------|-------------|
| prev_pos / pos / target_pos | [Vx3] | SMPL 身体顶点 |
| faces_batch | [3xF] | 身体面片 |
| vertex_type | [Vx1] | 1=普通障碍物, 2=手部 (推理时忽略) |

---

## 5. 代码结构

### 类关系概览

```
Config (stores config parameters)
  │
  ├─→ SequenceLoader (loads SMPL sequences)
  │    └─→ Loader
  │         └─→ Dataset
  │
  ├─→ VertexBuilder (builds vertices from SMPL)
  │    ├─→ GarmentBuilder
  │    │    └─→ Loader
  │    │         └─→ Dataset
  │    │
  │    └─→ BodyBuilder
  │         └─→ Loader
  │              └─→ Dataset
  │
  └─→ NoiseMaker (adds noise)
       └─→ GarmentBuilder
            └─→ Loader
                 └─→ Dataset
```

### 类职责

| 类 | 职责 | 依赖 | 被使用于 |
|-------|---------------|--------------|---------|
| Config | 存储所有配置参数 | 无 | 所有其他类 |
| VertexBuilder | 从 SMPL 参数构建顶点序列 | Config | GarmentBuilder, BodyBuilder |
| NoiseMaker | 为训练数据添加高斯噪声 | Config | GarmentBuilder |
| GarmentBuilder | 构建完整的服装 HeteroData | VertexBuilder, NoiseMaker, Config | Loader |
| BodyBuilder | 构建完整的身体 HeteroData | VertexBuilder, Config | Loader |
| SequenceLoader | 从磁盘加载并预处理 SMPL 序列 | Config | Loader |
| Loader | 组合所有 Builder 生成完整样本 | SequenceLoader, GarmentBuilder, BodyBuilder | Dataset |
| Dataset | PyTorch Dataset 接口，管理数据索引 | Loader | 训练/验证代码 |

### 数据流向

```
原始数据 (.pkl 文件)
     │
     ▼
SequenceLoader.load_sequence()
     │
     ├─→ GarmentBuilder.build()
     │    ├─→ VertexBuilder.add_verts()      # 添加 pos, prev_pos, target_pos
     │    ├─→ add_vertex_type()              # 添加顶点类型
     │    ├─→ NoiseMaker.add_noise()         # 添加训练噪声
     │    ├─→ add_restpos()                  # 添加静息位置
     │    ├─→ add_faces_and_edges()          # 添加面片和边
     │    ├─→ add_coarse()                   # 添加粗糙边
     │    └─→ add_button_edges()             # 添加纽扣边 (可选)
     │
     └─→ BodyBuilder.build()
          ├─→ VertexBuilder.add_verts()      # 添加 pos, prev_pos, target_pos
          ├─→ add_vertex_type()              # 添加顶点类型
          ├─→ add_faces()                    # 添加面片
          └─→ add_vertex_level()             # 添加层级标签
     │
     ▼
HeteroData
```

### 类层次结构 (组合关系)

```
Dataset
 └── Loader
      ├── SequenceLoader          # 加载 SMPL 参数
      ├── GarmentBuilder          # 构建服装
      │    ├── VertexBuilder      # 构建顶点
      │    └── NoiseMaker         # 添加噪声
      └── BodyBuilder             # 构建身体
           └── VertexBuilder      # 构建顶点 (共享类，独立实例)
```

### 关键方法调用链

当调用 `dataset[i]` 时：

```
Dataset.__getitem__(item)
    │
    ├── _find_idx(item)                     # 将全局索引转换为 (序列名, 帧索引, 服装名)
    │
    └── Loader.load_sample(fname, idx, garment_name, betas_id)
            │
            ├── SequenceLoader.load_sequence(fname)
            │       ├── 从磁盘读取 .pkl
            │       └── process_sequence()           # 预处理
            │
            ├── GarmentBuilder.build(sample, sequence, idx, garment_name)
            │       ├── VertexBuilder.add_verts()    # 添加 pos, prev_pos, target_pos
            │       ├── add_vertex_type()            # 添加顶点类型
            │       ├── NoiseMaker.add_noise()       # 添加训练噪声
            │       ├── add_restpos()                # 添加静息位置
            │       ├── add_faces_and_edges()        # 添加面片和边
            │       ├── add_coarse()                 # 添加粗糙边
            │       └── add_button_edges()           # 添加纽扣边 (可选)
            │
            └── BodyBuilder.build(sample, sequence, idx)
                    ├── VertexBuilder.add_verts()    # 添加 pos, prev_pos, target_pos
                    ├── add_vertex_type()            # 添加顶点类型
                    ├── add_faces()                  # 添加面片
                    └── add_vertex_level()           # 添加层级标签
```

---

## 6. 训练数据使用

### 数据集结构

从配置文件 `postcvpr.yaml` 可以看到：
```yaml
split_path: 'datasplits/train.csv'
```

训练数据通过一个 CSV 文件 (train.csv) 来定义，这个 CSV 文件包含多个动作序列的信息。

### 训练数据组织

```
├── train.csv (包含多个序列的索引)
│   ├── 序列 1: id, length, garment
│   ├── 序列 2: id, length, garment
│   ├── ...
│   └── 序列 952: id, length, garment
│
训练过程:
├── 将所有序列的帧展平为一个索引池
├── 每个训练步骤随机采样一个帧窗口（来自某个序列）
└── 通过多个 epoch 遍历所有序列的所有帧
```

### 训练帧选择机制

**train.py 会随机选择中间帧开始训练，而不是从第一帧开始。**

#### 帧选择流程

```mermaid
flowchart TD
    A[数据集初始化] --> B[计算每个序列可用帧数<br>_lens = N - 7]
    B --> C[展平所有帧为全局索引池<br>all_lens]
    C --> D[DataLoader 随机采样全局索引]
    D --> E[_find_idx 映射到<br>序列名 + 帧索引 idx]
    E --> F[load_sample 加载<br>idx 到 idx+lookup_steps 帧]
```

#### 关键代码

**`Dataset.__getitem__`** (datasets/postcvpr.py):

```python
def __getitem__(self, item: int) -> HeteroData:
    if self.wholeseq:
        # 验证模式：加载整个序列
        fname = self.datasplit.id[item]
        idx = 0
    else:
        # 训练模式：通过全局索引找到具体的 序列 + 帧位置
        fname, idx, garment_name = self._find_idx(item)
    
    sample = self.loader.load_sample(fname, idx, garment_name, betas_id=betas_id)
```

#### 具体过程

1. 假设序列有 100 帧，`lookup_steps=5`，则 `_lens = 100 - 7 = 93`（可用起始帧：0~92）
2. DataLoader 随机打乱索引池
3. 每个训练样本从 **随机帧位置 `idx`** 开始，加载 `idx` 到 `idx + lookup_steps + 2` 帧

### 衣服初始化机制

**动画帧的衣服是基于当前帧的 SMPL 姿态参数，通过 LBS（线性混合蒙皮）将静态模板衣服变形到该姿态，而不是从静态 T-pose 开始。**

#### 初始化流程

```mermaid
flowchart LR
    A[静态模板衣服<br>rest_pos] --> B[LBS 蒙皮]
    B --> C[当前帧姿态的衣服<br>posed_garment]
    
    subgraph LBS["LBS 蒙皮过程"]
        D[形状混合<br>shapedirs × betas]
        E[姿态混合<br>posedirs × pose]
        F[关节蒙皮变换<br>lbs_weights × joints]
    end
```

#### 关键代码

**`GarmentBuilder.make_cloth_verts`** (datasets/postcvpr.py):

```python
def make_cloth_verts(self, body_pose, global_orient, transl, betas, garment_name):
    # 使用 GarmentSMPL 模型，基于当前帧的 SMPL 参数生成衣服顶点
    garment_smpl_model = self.garment_smpl_model_dict[garment_name]
    full_pose = torch.cat([global_orient, body_pose], dim=1)
    
    with torch.no_grad():
        # 通过 LBS 将静态衣服变形到当前姿态
        vertices = garment_smpl_model.make_vertices(betas=betas, full_pose=full_pose, transl=transl)
```

#### 训练时的帧处理

| rollout step | `prev_pos` | `pos` | `target_pos` | 说明 |
|--------------|------------|-------|--------------|------|
| `idx=0` (第一帧) | `pos` (50%概率) | LBS(frame_idx) | LBS(frame_idx) | 初始帧，target=pos（静止） |
| `idx=1` (第二帧) | `pos` | `pos` | lookup[0] | 速度=0 |
| `idx≥2` | 上一帧的 `pos` | 上一帧的 `pred_pos` | lookup[idx-1] | 正常自回归 |

#### 设计优势

| 设计 | 优势 |
|------|------|
| 随机起始帧 | 增加训练多样性，模型能学习从任意姿态开始仿真 |
| LBS 初始化 | 衣服初始位置已经贴合人体，减少仿真初期的不稳定性 |

---

## 7. 完整数据加载流程

### 阶段一：初始化时构建 Dataset (train.py 第 14 行)

```
train.py:14
    │
    └── create_modules(modules, config)
            │
            └── create_dataloader_module(modules, config)   # arguments.py:207
                    │
                    ├── create_module(modules['dataset'], ...)  # 创建 Dataset 对象
                    │       │
                    │       └── datasets.postcvpr.create(mcfg)  # 调用 Dataset 模块的 create 函数
                    │               │
                    │               ├── create_loader(mcfg)     # 创建 Loader
                    │               │       ├── load_garments_dict()  # 加载服装字典
                    │               │       ├── smplx.SMPL()          # 加载 SMPL 模型
                    │               │       └── make_garment_smpl_dict()
                    │               │
                    │               ├── pd.read_csv(split_path)  # 读取数据分割表
                    │               │
                    │               └── Dataset(loader, datasplit)  # 创建 Dataset 实例
                    │
                    └── DataloaderModule(dataset, config)  # 包装成 DataloaderModule
```

### 阶段二：每个 Epoch 开始时创建 DataLoader (train.py 第 39 行)

```
train.py:39
    │
    └── dataloader_m.create_dataloader()
            │
            └── pyg.loader.DataLoader(self.dataset, ...)  # 创建 PyG DataLoader
```

### 阶段三：训练循环中按需加载数据

```
train.py:40-41  →  runner.run_epoch(... dataloader ...)
    │
    └── for batch in dataloader:  # 迭代时触发 Dataset.__getitem__
            │
            └── Dataset.__getitem__(item)         # datasets/postcvpr.py:781
                    │
                    └── Loader.load_sample(fname, idx, garment_name, betas_id)
                            │
                            ├── SequenceLoader.load_sequence(fname)  # 从磁盘加载 .pkl
                            │       └── pickle.load(filepath)
                            │
                            ├── GarmentBuilder.build(...)  # 构建服装 HeteroData
                            │
                            └── BodyBuilder.build(...)     # 构建身体 HeteroData
```

### 完整流程图

```mermaid
sequenceDiagram
    participant Main
    participant Dataset
    participant Loader
    participant SequenceLoader
    participant GarmentBuilder
    participant BodyBuilder
    participant Disk

    Main->>Dataset: dataset[i]
    Dataset->>Dataset: _find_idx(item)
    Dataset->>Loader: load_sample(fname, idx, garment)
    
    Loader->>SequenceLoader: load_sequence(fname)
    SequenceLoader->>Disk: read .pkl file
    Disk-->>SequenceLoader: SMPL parameters
    SequenceLoader-->>Loader: processed sequence
    
    par Build Garment
        Loader->>GarmentBuilder: build(sample, sequence, idx, garment_name)
        GarmentBuilder->>GarmentBuilder: VertexBuilder.add_verts()
        GarmentBuilder->>GarmentBuilder: add_vertex_type()
        GarmentBuilder->>GarmentBuilder: NoiseMaker.add_noise()
        GarmentBuilder->>GarmentBuilder: add_restpos()
        GarmentBuilder->>GarmentBuilder: add_faces_and_edges()
        GarmentBuilder->>GarmentBuilder: add_coarse()
        GarmentBuilder-->>Loader: garment data
    end
    
    par Build Body
        Loader->>BodyBuilder: build(sample, sequence, idx)
        BodyBuilder->>BodyBuilder: VertexBuilder.add_verts()
        BodyBuilder->>BodyBuilder: add_vertex_type()
        BodyBuilder->>BodyBuilder: add_faces()
        BodyBuilder->>BodyBuilder: add_vertex_level()
        BodyBuilder-->>Loader: body data
    end
    
    Loader-->>Dataset: HeteroData sample
    Dataset-->>Main: sample
```

---

## 8. 数据预处理

本文训练采用的数据集来自于 VTO 数据集。因此在开始训练前需要将 VTO 数据格式转成 HOOD 需要的数据格式。

### 格式转换

```
VTO Dataset (.pkl)                    HOOD 格式 (.pkl)
┌─────────────────────┐             ┌─────────────────────┐
│ {                   │             │ {                   │
│   'translation'     │  ──────►    │   'transl'          │  [Nx3]
│   'pose'            │  ──────►    │   'body_pose'       │  [Nx69]
│   'shape'           │  ──────►    │   'global_orient'   │  [Nx3]
│   ...               │             │   'betas'           │  [10]
│ }                   │             │ }                   │
└─────────────────────┘             └─────────────────────┘
```

### 转换过程
```python
# 输入路径：VTO 数据集中的 tshirt 模拟数据
simulations_path = Path(VTO_DATASET_PATH) / 'tshirt' / 'simulations'

# 输出路径：HOOD 数据目录下的 vto_dataset/smpl_parameters
out_root = Path(DEFAULTS.vto_root) / 'smpl_parameters'

# 遍历所有 952 个模拟序列
for simulation_path in tqdm(list(simulations_path.iterdir())):
    out_path = out_root / simulation_path.name
    convert_vto_to_pkl(simulation_path, out_path)
```

### convert_vto_to_pkl 函数

(data_making.py 第 192-221 行)

```python
def convert_vto_to_pkl(vto_seq_path, out_path, ...):
    vto_dict = pickle_load(vto_seq_path)

    out_dict = dict()
    out_dict['transl'] = vto_dict['translation'][start:]        # 提取位移
    out_dict['body_pose'] = vto_dict['pose'][start:, 3:72]      # 提取身体姿态(去掉前 3 维全局旋转)
    out_dict['global_orient'] = vto_dict['pose'][start:, :3]    # 提取全局朝向(前 3 维)
    out_dict['betas'] = vto_dict['shape'][0]                    # 提取体型参数

    # 可选：添加插值帧
    out_dict = make_interpolated_dict(out_dict, ...)
    
    pickle_dump(out_dict, out_path)  # 保存为新格式
```

### 字段映射

| VTO 原始格式 | HOOD 所需格式 |
|---------------------|----------------------|
| pose [N, 72] (全部姿态) | body_pose [N, 69] + global_orient [N, 3] |
| translation | transl |
| shape [N, 10] | betas [10] (仅第一帧) |

---

## 9. 数据缓存与存储机制

### 数据存储策略

在训练过程中，SMPL序列的预处理和mesh转换是**实时进行的**，数据存储在内存中直接发送给GPU，**没有显式地将转换后的mesh序列数据保存到电脑本地**。

### 数据处理流程

```
1. 磁盘存储的原始数据：
   - .pkl 文件存储 SMPL 参数 (body_pose, global_orient, transl, betas)
   - 每次训练需要时从磁盘读取

2. 实时处理过程（每次调用 Dataset.__getitem__）：
   - 读取原始 SMPL 参数
   - 使用 SMPL 模型实时转换成 mesh
   - 构建 HeteroData 对象
   - 数据保持在内存中

3. 数据流向：
   磁盘 (.pkl 文件，SMPL 参数)
      ↓
   内存 (实时转换为 mesh，构建 HeteroData)
      ↓
   GPU (DataLoader 批处理并移动到设备)
```

### 代码实现

#### 从磁盘加载原始参数

```python
# datasets/postcvpr.py:691-699
def load_sequence(self, fname: str, betas_id: int=None) -> dict:
    filepath = os.path.join(self.data_path, fname + '.pkl')
    with open(filepath, 'rb') as f:
        sequence = pickle.load(f)  # 加载原始 SMPL 参数
    sequence = self.process_sequence(sequence)
    return sequence  # 返回参数，不是 mesh
```

#### 实时构建完整样本

```python
# datasets/postcvpr.py:732-744
def load_sample(self, fname: str, idx: int, garment_name: str, betas_id: int) -> HeteroData:
    sequence = self.sequence_loader.load_sequence(fname, betas_id=betas_id)
    sample = HeteroData()
    sample = self.garment_builder.build(sample, sequence, idx, garment_name)
    sample = self.body_builder.build(sample, sequence, idx)
    return sample  # 直接返回，不保存
```

### 并行加载配置

从配置文件 `postcvpr.yaml`：

```yaml
dataloader:
  num_workers: 8      # 8 个 worker 并行加载数据
  batch_size: 1
```

`num_workers: 8` 表示使用 8 个 worker 进程并行加载数据，每个 worker 都会：
1. 从磁盘读取 .pkl 文件（SMPL 参数）
2. 实时转换为 mesh
3. 将数据放入内存队列供主进程使用

### 唯一的内存缓存

唯一的小缓存是在 `GarmentBuilder.add_coarse()` 方法中：

```python
# 缓存计算好的粗糙边，避免重复计算
if center in garment_dict['coarse_edges']:
    coarse_edges_dict = garment_dict['coarse_edges'][center]
else:
    coarse_edges_dict = make_coarse_edges(faces, center, n_levels=self.mcfg.n_coarse_levels)
    garment_dict['coarse_edges'][center] = coarse_edges_dict
```

这只是缓存小的边索引数据，不是 mesh 序列本身。

### 设计优缺点

| 优点 | 缺点 |
|------|------|
| 每次训练可以动态应用数据增强（如 `random_betas`） | 每次都需要重新计算 mesh，有计算开销 |
| 不需要额外的磁盘空间存储预处理的 mesh | 训练启动时间可能较长 |
| 数据格式变更时不需要重新预处理 | 重复计算相同的序列参数 |

---

## 关键配置参数 (Config 类)

| 参数 | 默认值 | 描述 |
|-----------|---------------|-------------|
| noise_scale | 3e-3 | 训练噪声强度 |
| lookup_steps | 5 | 预测未来帧数 |
| n_coarse_levels | 1 | 粗糙边层级数 |
| pinned_verts | False | 是否使用固定顶点 |
| separate_arms | False | 是否分离手臂 |
| wholeseq | False | 是否加载整序列 |

---

## 数据增强策略

主要增强方式：

1. **位置噪声**：noise_scale=3e-3 的高斯噪声
2. **随机体型**：random_betas 选项
3. **静息姿态缩放**：restpos_scale_min/max 参数
4. **随机粗糙边中心**：每次从图的多个中心点中随机选择

---

## 设计理念

这套数据系统的设计核心思想是：将 SMPL 姿态序列转换为服装和身体的网格顶点序列，同时通过多层级的图结构（网格边 + 粗糙边）来支持图神经网络的消息传递。

这种方法能够实现：
- 通过序列数据建模时间动态
- 通过图边建立空间关系
- 通过层级粗糙边提取多尺度特征
- 通过可配置的噪声和变换实现灵活的数据增强

---

## 10. Mesh格式数据支持

HOOD不仅支持SMPL参数格式，还支持直接使用Mesh格式的数据进行训练和推理。

### 支持的场景

| 功能 | SMPL格式 | Mesh格式 |
|------|---------|---------|
| 训练 | ✅ 支持 | ✅ 支持 |
| 推理 | ✅ 支持 | ✅ 支持 |

### 数据格式要求

将FBX或其他格式的mesh动画序列转换为.pkl文件，格式如下：

```python
{
    'verts': np.array,  # [N, V, 3] - N帧，每帧V个顶点的位置
    'faces': np.array,  # [F, 3]    - 网格面片索引（三角面片）
}
```

**关键要求：**
- `verts` 维度必须为 [N, V, 3]，所有帧顶点数必须一致
- `faces` 是静态的，描述mesh拓扑结构，只需要一份
- 坐标系：右手坐标系，单位为米

### 配置方法

使用 `from_any_pose` 模块，在配置文件中设置 `pose_sequence_type: 'mesh'`：

```yaml
dataloader:
  dataset:
    from_any_pose:
      pose_sequence_type: "mesh"           # 设置为mesh模式
      pose_sequence_path: 'your_data.pkl'   # mesh序列文件
      garment_template_path: 'your_garment.pkl'  # 服装模板，支持.pkl或.obj格式

runner:
  from_any_pose:
    # 训练或推理参数
```

**服装模板格式支持：**
- `.pkl` 文件：预处理后的服装数据，加载速度快
- `.obj` 文件：标准OBJ网格格式，首次加载会自动转换为内部格式

**obj格式自动转换：**
当使用 `.obj` 格式时，系统会自动：
1. 加载obj文件中的顶点和面片数据
2. 构建多层级粗糙边（coarse edges）用于碰撞检测
3. 转换为内部数据结构

注意：首次使用 `.obj` 格式时加载时间较长，建议对于重复训练的服装先转换为 `.pkl` 格式。

### 与SMPL模式的对比

| 特性 | SMPL模式 | Mesh模式 |
|------|---------|---------|
| **输入格式** | SMPL参数 [N, 69+3+3+10] | Mesh顶点 [N, V, 3] |
| **数据来源** | SMPL模型计算 | 直接使用mesh数据 |
| **优势** | 可调整体型、姿态，参数化控制 | 无信息损失，直接使用原始数据 |
| **适用场景** | 需要参数化控制 | 有完整的mesh动画序列 |
| **配置模块** | `postcvpr` 或 `from_any_pose` | `from_any_pose` |

### 数据处理流程

```mermaid
graph LR
    A[FBX/Mesh文件] --> B[转换工具]
    B --> C[.pkl文件<br/>{verts, faces}]
    C --> D[Dataset<br/pose_sequence_type: mesh]
    D --> E[BareMeshBodyBuilder]
    E --> F[HeteroData]
    F --> G[GPU训练/推理]
```

### FBX转换提示

将虚幻引擎或其他来源的FBX文件转换为HOOD格式时需要注意：

- 坐标系转换：虚幻引擎使用左手系，需转换为右手系
- 单位统一：确保坐标单位为米
- 帧率匹配：根据需要调整采样帧率
- 手部处理：如果需要区分手部，可在 `obstacle_dict_file` 中标注

### 模块选择

| 需求 | 推荐模块 | 说明 |
|------|---------|------|
| 大规模训练（多个序列） | `postcvpr` | 仅支持SMPL格式 |
| 单序列训练/推理 | `from_any_pose` | 支持SMPL和Mesh两种格式 |
| 自定义FBX数据 | `from_any_pose` + mesh格式 | 无需转换为SMPL |

---