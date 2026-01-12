"""
AnimationViewer.py
动画序列数据加载和处理模块

支持的文件格式：
1. mesh_sequence.pkl (mesh序列): 包含 'verts' [N, V, 3] 和 'faces' [F, 3]
2. 推理输出 pkl: 包含 'pred', 'cloth_faces', 'obstacle', 'obstacle_faces'
3. SMPL pose_sequence.pkl: 包含 'body_pose' [N, 69], 'global_orient' [N, 3], 
   'transl' [N, 3], 'betas' [10,]
4. OBJ 文件: 静态网格文件
5. HDF5 文件 (.h5/.hdf5): 高效存储的动画序列格式
   - Mesh 格式: /vertices [N,V,3], /faces [F,3], /obstacle_vertices, /obstacle_faces
   - SMPL 格式: /body_pose, /global_orient, /transl, /betas

核心组件：
- SequenceData: 统一的序列数据容器
- SequenceLoader: 序列加载器抽象基类
  - MeshSequenceLoader: Mesh 格式加载器
  - SMPLSequenceLoader: SMPL 格式加载器
  - OBJLoader: OBJ 静态网格加载器
  - HDF5SequenceLoader: HDF5 格式加载器
- SequenceLoaderFactory: 加载器工厂
"""

import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

import numpy as np
import torch
import smplx
import trimesh

# ============ 路径配置 ============
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.common import pickle_load


# ============ 常量定义 ============

class Paths:
    """路径常量"""
    SMPL_MODELS = PROJECT_ROOT / "hood_data" / "aux_data" / "smpl"
    DEFAULT_DATA = PROJECT_ROOT / "hood_data" / "fromanypose"
    
    SMPL_MODEL_FILES = {
        "male": "SMPL_MALE.pkl",
        "female": "SMPL_FEMALE.pkl",
        "neutral": "SMPL_NEUTRAL.pkl"
    }


class SequenceType:
    """序列类型常量"""
    MESH = "mesh"
    SMPL = "smpl"
    INFERENCE = "inference"
    OBJ = "obj"
    HDF5 = "hdf5"


# ============ 配置类 ============

@dataclass
class ViewerConfig:
    """可视化配置"""
    fps: int = 30
    mesh_color: Tuple[float, ...] = (0.3, 0.6, 0.8, 1.0)
    obstacle_color: Tuple[float, ...] = (0.4, 0.4, 0.4, 1.0)
    camera_distance: float = 3.0
    backface_culling: bool = False
    place_on_floor: bool = False


@dataclass
class SMPLConfig:
    """SMPL 模型配置"""
    gender: str = "neutral"
    
    @property
    def model_path(self) -> Path:
        filename = Paths.SMPL_MODEL_FILES.get(self.gender.lower(), "SMPL_NEUTRAL.pkl")
        return Paths.SMPL_MODELS / filename


# ============ 数据容器 ============

@dataclass
class SequenceData:
    """
    统一的序列数据容器
    
    所有加载器都将数据转换为此格式，实现数据格式的统一
    支持两种渲染模式：
    - mesh 模式：使用 faces（三角面）渲染
    - skeleton 模式：使用 edges（边）渲染
    """
    vertices: np.ndarray  # [N, V, 3] 顶点位置序列
    faces: np.ndarray     # [F, 3] 面索引（mesh模式）或 None
    
    # 可选的障碍物数据
    obstacle_vertices: Optional[np.ndarray] = None  # [N, V, 3]
    obstacle_faces: Optional[np.ndarray] = None     # [F, 3]
    
    # 骨骼/边数据（skeleton模式）
    edges: Optional[np.ndarray] = None  # [E, 2] 边索引
    bone_names: Optional[List[str]] = None  # [V,] 骨骼名称
    dt: Optional[np.ndarray] = None  # [N,] 每帧时间间隔
    
    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def n_frames(self) -> int:
        return self.vertices.shape[0]
    
    @property
    def n_vertices(self) -> int:
        return self.vertices.shape[1]
    
    @property
    def has_obstacle(self) -> bool:
        return self.obstacle_vertices is not None and self.obstacle_faces is not None
    
    @property
    def is_static(self) -> bool:
        """是否为静态网格（单帧）"""
        return self.n_frames == 1
    
    @property
    def is_skeleton(self) -> bool:
        """是否为骨骼模式（使用edges而非faces）"""
        return self.edges is not None and (self.faces is None or len(self.faces) == 0)
    
    @property
    def has_edges(self) -> bool:
        """是否有边数据"""
        return self.edges is not None and len(self.edges) > 0
    
    def repeat_to_frames(self, n_frames: int) -> 'SequenceData':
        """将静态网格重复为指定帧数的序列"""
        if self.n_frames == n_frames:
            return self
        
        vertices = np.repeat(self.vertices, n_frames, axis=0)
        
        obstacle_vertices = None
        if self.obstacle_vertices is not None:
            obstacle_vertices = np.repeat(self.obstacle_vertices, n_frames, axis=0)
        
        return SequenceData(
            vertices=vertices,
            faces=self.faces,
            obstacle_vertices=obstacle_vertices,
            obstacle_faces=self.obstacle_faces,
            edges=self.edges,
            bone_names=self.bone_names,
            dt=self.dt,
            metadata={**self.metadata, 'repeated_from': self.n_frames, 'repeated_to': n_frames}
        )
    
    def summary(self) -> str:
        """返回数据摘要信息"""
        if self.is_skeleton:
            type_str = "骨骼序列" if not self.is_static else "静态骨骼"
        else:
            type_str = "静态网格" if self.is_static else "动画序列"
        
        lines = [
            f"类型: {type_str}",
            f"帧数: {self.n_frames}",
            f"顶点数: {self.n_vertices}",
        ]
        
        if self.faces is not None and len(self.faces) > 0:
            lines.append(f"面数: {self.faces.shape[0]}")
        
        if self.has_edges:
            lines.append(f"边数: {self.edges.shape[0]}")
        
        if self.has_obstacle:
            lines.append(f"障碍物顶点: {self.obstacle_vertices.shape}")
            lines.append(f"障碍物面数: {self.obstacle_faces.shape[0]}")
        
        return "\n".join(lines)


# ============ 工具函数 ============

def adjust_color(color: Tuple[float, ...]) -> Tuple[float, ...]:
    """
    调整颜色使其更柔和美观
    
    :param color: RGBA 颜色元组
    :return: 调整后的颜色元组
    """
    color_arr = np.array(color, dtype=np.float32)
    color_arr[..., :3] = color_arr[..., :3] / max(color_arr[..., :3].max(), 1e-6)
    color_arr[..., :3] = color_arr[..., :3] * 0.3 + 0.3
    return tuple(color_arr)


# ============ 序列加载器 ============

class SequenceLoader(ABC):
    """序列加载器抽象基类"""
    
    @abstractmethod
    def load(self, path: str) -> SequenceData:
        """加载序列文件并返回统一格式的数据"""
        pass
    
    @staticmethod
    @abstractmethod
    def can_load(data: dict) -> bool:
        """检查给定的数据字典是否可以被此加载器处理"""
        pass


class MeshSequenceLoader(SequenceLoader):
    """
    Mesh 序列加载器
    
    支持格式：
    - mesh_sequence: {'verts': [N, V, 3], 'faces': [F, 3]}
    - 推理输出: {'pred': [N, V, 3], 'cloth_faces': [F, 3], ...}
    - 新 mesh_sequence: {'vertices': [N, V, 3], 'faces': [F, 3], ...}
    """
    
    def load(self, path: str) -> SequenceData:
        data = pickle_load(path)
        
        if not self.can_load(data):
            raise ValueError(f"无法加载为 Mesh 格式: {list(data.keys())}")
        
        # 解析顶点和面
        if 'verts' in data:
            vertices = np.array(data['verts'])
            faces = np.array(data['faces'])
        elif 'vertices' in data:
            vertices = np.array(data['vertices'])
            faces = np.array(data['faces'])
        else:  # 'pred' in data
            vertices = np.array(data['pred'])
            faces = np.array(data['cloth_faces'])
            if len(faces.shape) == 3:
                faces = faces[0]
        
        # 解析障碍物数据
        obstacle_vertices = None
        obstacle_faces = None
        if 'obstacle' in data and 'obstacle_faces' in data:
            obstacle_vertices = np.array(data['obstacle'])
            obstacle_faces = np.array(data['obstacle_faces'])
            if len(obstacle_faces.shape) == 3:
                obstacle_faces = obstacle_faces[0]
        
        return SequenceData(
            vertices=vertices,
            faces=faces,
            obstacle_vertices=obstacle_vertices,
            obstacle_faces=obstacle_faces,
            metadata={'source_format': 'mesh', 'path': path}
        )
    
    @staticmethod
    def can_load(data: dict) -> bool:
        return 'verts' in data or 'pred' in data or 'vertices' in data


class SMPLSequenceLoader(SequenceLoader):
    """
    SMPL 序列加载器
    
    支持格式：
    - SMPL 参数: {'body_pose': [N, 69], 'global_orient': [N, 3], 
                  'transl': [N, 3], 'betas': [10,]}
    """
    
    REQUIRED_KEYS = {'body_pose', 'global_orient', 'transl', 'betas'}
    
    def __init__(self, config: SMPLConfig = None):
        self.config = config or SMPLConfig()
        self._smpl_model = None
    
    @property
    def smpl_model(self) -> smplx.SMPL:
        """懒加载 SMPL 模型"""
        if self._smpl_model is None:
            model_path = self.config.model_path
            if not model_path.exists():
                raise FileNotFoundError(f"SMPL 模型不存在: {model_path}")
            print(f"加载 SMPL 模型: {model_path.name}")
            self._smpl_model = smplx.SMPL(str(model_path))
        return self._smpl_model
    
    def load(self, path: str) -> SequenceData:
        data = pickle_load(path)
        
        if not self.can_load(data):
            raise ValueError(
                f"不是有效的 SMPL 格式。需要: {self.REQUIRED_KEYS}，实际: {list(data.keys())}"
            )
        
        body_pose = np.array(data['body_pose'])
        global_orient = np.array(data['global_orient'])
        transl = np.array(data['transl'])
        betas = np.array(data['betas'])
        
        vertices = self._smpl_to_vertices(body_pose, global_orient, transl, betas)
        faces = self.smpl_model.faces.astype(np.int64)
        
        return SequenceData(
            vertices=vertices,
            faces=faces,
            metadata={
                'source_format': 'smpl',
                'path': path,
                'gender': self.config.gender,
                'n_frames': body_pose.shape[0]
            }
        )
    
    def _smpl_to_vertices(
        self,
        body_pose: np.ndarray,
        global_orient: np.ndarray,
        transl: np.ndarray,
        betas: np.ndarray
    ) -> np.ndarray:
        """将 SMPL 参数转换为顶点位置"""
        N = body_pose.shape[0]
        
        if len(betas.shape) == 1:
            betas = np.tile(betas, (N, 1))
        
        input_dict = {
            'body_pose': torch.FloatTensor(body_pose),
            'global_orient': torch.FloatTensor(global_orient),
            'transl': torch.FloatTensor(transl),
            'betas': torch.FloatTensor(betas)
        }
        
        with torch.no_grad():
            output = self.smpl_model(**input_dict)
        
        return output.vertices.numpy().astype(np.float32)
    
    @staticmethod
    def can_load(data: dict) -> bool:
        return SMPLSequenceLoader.REQUIRED_KEYS.issubset(set(data.keys()))


class OBJLoader:
    """OBJ 静态网格加载器"""
    
    def load(self, path: str) -> SequenceData:
        if not path.lower().endswith('.obj'):
            raise ValueError(f"不是 OBJ 文件: {path}")
        
        if not Path(path).exists():
            raise FileNotFoundError(f"OBJ 文件不存在: {path}")
        
        print(f"加载 OBJ 文件: {path}")
        mesh = trimesh.load(path, process=False)
        
        vertices = np.array(mesh.vertices, dtype=np.float32)
        faces = np.array(mesh.faces, dtype=np.int64)
        vertices = vertices[np.newaxis, ...]
        
        return SequenceData(
            vertices=vertices,
            faces=faces,
            metadata={
                'source_format': 'obj',
                'path': path,
                'filename': Path(path).name
            }
        )
    
    @staticmethod
    def can_load_file(path: str) -> bool:
        return path.lower().endswith('.obj')


class HDF5SequenceLoader(SequenceLoader):
    """
    HDF5 序列加载器
    
    支持格式：
    - Mesh HDF5: /vertices [N, V, 3], /faces [F, 3], /obstacle_vertices (可选), /obstacle_faces (可选)
    - SMPL HDF5: /body_pose [N, 69], /global_orient [N, 3], /transl [N, 3], /betas [10,]
    - Skeleton HDF5: /vertices [N, V, 3], /edges [E, 2], /bone_names (可选), /dt (可选)
    """
    
    HDF5_EXTENSIONS = {'.h5', '.hdf5'}
    MESH_FORMAT = "mesh_sequence_hdf5"
    SMPL_FORMAT = "smpl_sequence_hdf5"
    SKELETON_FORMAT = "skeleton_sequence"
    
    def __init__(self, smpl_config: SMPLConfig = None):
        self.smpl_config = smpl_config or SMPLConfig()
        self._smpl_model = None
        self._h5py = None
    
    def _ensure_h5py(self):
        if self._h5py is None:
            try:
                import h5py
                self._h5py = h5py
            except ImportError:
                raise ImportError(
                    "未安装 h5py 库，无法加载 HDF5 文件。\n"
                    "请运行: pip install h5py"
                )
    
    @property
    def smpl_model(self) -> smplx.SMPL:
        if self._smpl_model is None:
            model_path = self.smpl_config.model_path
            if not model_path.exists():
                raise FileNotFoundError(f"SMPL 模型不存在: {model_path}")
            print(f"加载 SMPL 模型: {model_path.name}")
            self._smpl_model = smplx.SMPL(str(model_path))
        return self._smpl_model
    
    def load(self, path: str) -> SequenceData:
        self._ensure_h5py()
        
        if not self.can_load_file(path):
            raise ValueError(f"不是 HDF5 文件: {path}")
        
        if not Path(path).exists():
            raise FileNotFoundError(f"HDF5 文件不存在: {path}")
        
        print(f"加载 HDF5 文件: {path}")
        
        with self._h5py.File(path, 'r') as f:
            hdf5_format = f.attrs.get('format', None)
            if isinstance(hdf5_format, bytes):
                hdf5_format = hdf5_format.decode('utf-8')
            
            if hdf5_format == self.SKELETON_FORMAT:
                return self._load_skeleton_hdf5(f, path)
            elif hdf5_format == self.SMPL_FORMAT:
                return self._load_smpl_hdf5(f, path)
            elif hdf5_format == self.MESH_FORMAT:
                return self._load_mesh_hdf5(f, path)
            else:
                # 自动检测格式
                if 'vertices' in f and 'edges' in f and 'faces' not in f:
                    return self._load_skeleton_hdf5(f, path)
                elif 'vertices' in f and 'faces' in f:
                    return self._load_mesh_hdf5(f, path)
                elif 'body_pose' in f:
                    return self._load_smpl_hdf5(f, path)
                else:
                    raise ValueError(
                        f"无法识别的 HDF5 格式。\n"
                        f"数据集: {list(f.keys())}\n"
                        f"属性: {dict(f.attrs)}"
                    )
    
    def _load_mesh_hdf5(self, f, path: str) -> SequenceData:
        vertices = np.array(f['vertices'], dtype=np.float32)
        faces = np.array(f['faces'], dtype=np.int64)
        
        if len(vertices.shape) == 2:
            vertices = vertices[np.newaxis, ...]
        
        obstacle_vertices = None
        obstacle_faces = None
        
        if 'obstacle_vertices' in f:
            obstacle_vertices = np.array(f['obstacle_vertices'], dtype=np.float32)
            if len(obstacle_vertices.shape) == 2:
                obstacle_vertices = obstacle_vertices[np.newaxis, ...]
        
        if 'obstacle_faces' in f:
            obstacle_faces = np.array(f['obstacle_faces'], dtype=np.int64)
        
        metadata = {
            'source_format': 'hdf5_mesh',
            'path': path,
            'filename': Path(path).name
        }
        
        for key, value in f.attrs.items():
            metadata[f'hdf5_{key}'] = value
        
        return SequenceData(
            vertices=vertices,
            faces=faces,
            obstacle_vertices=obstacle_vertices,
            obstacle_faces=obstacle_faces,
            metadata=metadata
        )
    
    def _load_skeleton_hdf5(self, f, path: str) -> SequenceData:
        """加载骨骼序列 HDF5 文件"""
        vertices = np.array(f['vertices'], dtype=np.float32)
        edges = np.array(f['edges'], dtype=np.int64)
        
        # 确保 vertices 是 [N, V, 3] 格式
        if len(vertices.shape) == 2:
            vertices = vertices[np.newaxis, ...]
        
        # 读取可选字段
        bone_names = None
        if 'bone_names' in f:
            bone_names_raw = f['bone_names'][:]
            bone_names = [
                name.decode('utf-8') if isinstance(name, bytes) else name 
                for name in bone_names_raw
            ]
        
        dt = None
        if 'dt' in f:
            dt = np.array(f['dt'], dtype=np.float32)
        
        metadata = {
            'source_format': 'hdf5_skeleton',
            'path': path,
            'filename': Path(path).name
        }
        
        for key, value in f.attrs.items():
            attr_value = value.decode('utf-8') if isinstance(value, bytes) else value
            metadata[f'hdf5_{key}'] = attr_value
        
        return SequenceData(
            vertices=vertices,
            faces=np.array([], dtype=np.int64),  # 空的 faces
            edges=edges,
            bone_names=bone_names,
            dt=dt,
            metadata=metadata
        )
    
    def _load_smpl_hdf5(self, f, path: str) -> SequenceData:
        body_pose = np.array(f['body_pose'], dtype=np.float32)
        global_orient = np.array(f['global_orient'], dtype=np.float32)
        transl = np.array(f['transl'], dtype=np.float32)
        betas = np.array(f['betas'], dtype=np.float32)
        
        vertices = self._smpl_to_vertices(body_pose, global_orient, transl, betas)
        faces = self.smpl_model.faces.astype(np.int64)
        
        metadata = {
            'source_format': 'hdf5_smpl',
            'path': path,
            'filename': Path(path).name,
            'gender': self.smpl_config.gender,
            'n_frames': body_pose.shape[0]
        }
        
        for key, value in f.attrs.items():
            metadata[f'hdf5_{key}'] = value
        
        return SequenceData(
            vertices=vertices,
            faces=faces,
            metadata=metadata
        )
    
    def _smpl_to_vertices(
        self,
        body_pose: np.ndarray,
        global_orient: np.ndarray,
        transl: np.ndarray,
        betas: np.ndarray
    ) -> np.ndarray:
        N = body_pose.shape[0]
        
        if len(betas.shape) == 1:
            betas = np.tile(betas, (N, 1))
        
        input_dict = {
            'body_pose': torch.FloatTensor(body_pose),
            'global_orient': torch.FloatTensor(global_orient),
            'transl': torch.FloatTensor(transl),
            'betas': torch.FloatTensor(betas)
        }
        
        with torch.no_grad():
            output = self.smpl_model(**input_dict)
        
        return output.vertices.numpy().astype(np.float32)
    
    @staticmethod
    def can_load(data: dict) -> bool:
        return False
    
    @staticmethod
    def can_load_file(path: str) -> bool:
        ext = Path(path).suffix.lower()
        return ext in HDF5SequenceLoader.HDF5_EXTENSIONS


# ============ 加载器工厂 ============

class SequenceLoaderFactory:
    """
    序列加载器工厂
    
    根据文件内容或扩展名自动选择合适的加载器
    """
    
    @classmethod
    def create(cls, sequence_type: str, **kwargs):
        """创建指定类型的加载器"""
        if sequence_type == SequenceType.SMPL:
            gender = kwargs.get('gender', 'neutral')
            return SMPLSequenceLoader(SMPLConfig(gender=gender))
        elif sequence_type == SequenceType.OBJ:
            return OBJLoader()
        elif sequence_type == SequenceType.HDF5:
            gender = kwargs.get('gender', 'neutral')
            return HDF5SequenceLoader(SMPLConfig(gender=gender))
        return MeshSequenceLoader()
    
    @classmethod
    def auto_detect(cls, path: str, **kwargs):
        """
        自动检测文件格式并返回合适的加载器
        
        :return: (加载器实例, 检测到的类型)
        """
        if OBJLoader.can_load_file(path):
            return OBJLoader(), SequenceType.OBJ
        
        if HDF5SequenceLoader.can_load_file(path):
            gender = kwargs.get('gender', 'neutral')
            return HDF5SequenceLoader(SMPLConfig(gender=gender)), SequenceType.HDF5
        
        data = pickle_load(path)
        
        if SMPLSequenceLoader.can_load(data):
            gender = kwargs.get('gender', 'neutral')
            return SMPLSequenceLoader(SMPLConfig(gender=gender)), SequenceType.SMPL
        
        if MeshSequenceLoader.can_load(data):
            return MeshSequenceLoader(), SequenceType.MESH
        
        raise ValueError(
            f"无法识别的文件格式。\n"
            f"支持: OBJ, HDF5, SMPL ({SMPLSequenceLoader.REQUIRED_KEYS}) 或 Mesh (verts/pred)\n"
            f"实际: {list(data.keys())}"
        )
    
    @classmethod
    def load_file(cls, path: str, **kwargs) -> SequenceData:
        """便捷方法：自动检测并加载文件"""
        loader, detected_type = cls.auto_detect(path, **kwargs)
        print(f"检测到格式: {detected_type}")
        return loader.load(path)
