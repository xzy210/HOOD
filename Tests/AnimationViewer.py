"""
AnimationViewer.py
使用 aitviewer 可视化动画序列和静态网格

支持的文件格式：
1. mesh_sequence.pkl (mesh序列): 包含 'verts' [N, V, 3] 和 'faces' [F, 3]
2. 推理输出 pkl: 包含 'pred', 'cloth_faces', 'obstacle', 'obstacle_faces'
3. SMPL pose_sequence.pkl: 包含 'body_pose' [N, 69], 'global_orient' [N, 3], 
   'transl' [N, 3], 'betas' [10,]
4. OBJ 文件: 静态网格文件

设计结构：
- SequenceData: 统一的序列数据容器
- SequenceLoader: 序列加载器抽象基类
  - MeshSequenceLoader: Mesh 格式加载器
  - SMPLSequenceLoader: SMPL 格式加载器
  - OBJLoader: OBJ 静态网格加载器
- AnimationViewer: 可视化器类
"""

import os
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
# 设置项目路径（必须在导入 aitviewer 之前）
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Windows DLL 加载顺序问题修复：
# meshes.py 中 pxr 库需要完整的 OpenGL 环境才能加载 _tf.pyd
# 直接导入 Meshes 时，只会部分初始化 moderngl_window（仅 VAO 类）
# 而 HeadlessRenderer -> Viewer 会完整导入 moderngl_window，初始化整个窗口系统
# 因此必须先导入 HeadlessRenderer，确保 OpenGL 环境就绪后再导入 Meshes
from aitviewer.headless import HeadlessRenderer  # noqa: F401 - 必须保留此导入
from aitviewer.renderables.meshes import Meshes
from aitviewer.scene.camera import PinholeCamera
from aitviewer.utils import path as aitviewer_path
from aitviewer.viewer import Viewer
from matplotlib import pyplot as plt

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
    INFERENCE = "inference"  # 推理输出格式
    OBJ = "obj"  # 静态 OBJ 网格


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
    """
    vertices: np.ndarray  # [N, V, 3] 顶点位置序列
    faces: np.ndarray     # [F, 3] 面索引
    
    # 可选的障碍物数据
    obstacle_vertices: Optional[np.ndarray] = None  # [N, V, 3]
    obstacle_faces: Optional[np.ndarray] = None     # [F, 3]
    
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
    
    def repeat_to_frames(self, n_frames: int) -> 'SequenceData':
        """
        将静态网格重复为指定帧数的序列
        
        :param n_frames: 目标帧数
        :return: 新的 SequenceData 对象
        """
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
            metadata={**self.metadata, 'repeated_from': self.n_frames, 'repeated_to': n_frames}
        )
    
    def summary(self) -> str:
        """返回数据摘要信息"""
        type_str = "静态网格" if self.is_static else "动画序列"
        lines = [
            f"类型: {type_str}",
            f"帧数: {self.n_frames}",
            f"顶点数: {self.n_vertices}",
            f"面数: {self.faces.shape[0]}",
        ]
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


def place_on_floor(vertices: np.ndarray, x_shift: float = 0, z_shift: float = 0) -> np.ndarray:
    """
    将网格放置到地面上（y=0），并进行水平位置偏移
    
    :param vertices: [N, V, 3] 或 [V, 3] 顶点位置
    :param x_shift: x轴偏移
    :param z_shift: z轴偏移
    :return: 调整后的顶点位置
    """
    vertices = vertices.copy()
    min_y = vertices[..., 1].min()
    vertices[..., 1] -= min_y
    vertices[..., 0] += x_shift
    vertices[..., 2] += z_shift
    return vertices


# ============ 序列加载器 ============

class SequenceLoader(ABC):
    """
    序列加载器抽象基类
    
    定义了加载器的统一接口，所有具体加载器都必须实现这些方法
    """
    
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
    """
    
    def load(self, path: str) -> SequenceData:
        data = pickle_load(path)
        
        if not self.can_load(data):
            raise ValueError(f"无法加载为 Mesh 格式: {list(data.keys())}")
        
        # 解析顶点和面
        if 'verts' in data:
            vertices = np.array(data['verts'])
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
        return 'verts' in data or 'pred' in data


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
        
        # 提取 SMPL 参数
        body_pose = np.array(data['body_pose'])
        global_orient = np.array(data['global_orient'])
        transl = np.array(data['transl'])
        betas = np.array(data['betas'])
        
        # 转换为顶点
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
        
        # 扩展 betas 维度
        if len(betas.shape) == 1:
            betas = np.tile(betas, (N, 1))
        
        # 转换为 tensor 并推理
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
    """
    OBJ 静态网格加载器
    
    加载 .obj 文件并转换为单帧的 SequenceData
    """
    
    def load(self, path: str) -> SequenceData:
        """
        加载 OBJ 文件
        
        :param path: OBJ 文件路径
        :return: 单帧的 SequenceData
        """
        if not path.lower().endswith('.obj'):
            raise ValueError(f"不是 OBJ 文件: {path}")
        
        if not Path(path).exists():
            raise FileNotFoundError(f"OBJ 文件不存在: {path}")
        
        print(f"加载 OBJ 文件: {path}")
        mesh = trimesh.load(path, process=False)
        
        # 获取顶点和面
        vertices = np.array(mesh.vertices, dtype=np.float32)
        faces = np.array(mesh.faces, dtype=np.int64)
        
        # 转换为序列格式 [1, V, 3]
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
        """检查文件是否为 OBJ 格式"""
        return path.lower().endswith('.obj')


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
        return MeshSequenceLoader()
    
    @classmethod
    def auto_detect(cls, path: str, **kwargs):
        """
        自动检测文件格式并返回合适的加载器
        
        :return: (加载器实例, 检测到的类型)
        """
        # 先检查文件扩展名
        if OBJLoader.can_load_file(path):
            return OBJLoader(), SequenceType.OBJ
        
        # pkl 文件需要读取内容判断
        data = pickle_load(path)
        
        # 按优先级检测
        if SMPLSequenceLoader.can_load(data):
            gender = kwargs.get('gender', 'neutral')
            return SMPLSequenceLoader(SMPLConfig(gender=gender)), SequenceType.SMPL
        
        if MeshSequenceLoader.can_load(data):
            return MeshSequenceLoader(), SequenceType.MESH
        
        raise ValueError(
            f"无法识别的文件格式。\n"
            f"支持: OBJ, SMPL ({SMPLSequenceLoader.REQUIRED_KEYS}) 或 Mesh (verts/pred)\n"
            f"实际: {list(data.keys())}"
        )
    
    @classmethod
    def load_file(cls, path: str, **kwargs) -> SequenceData:
        """
        便捷方法：自动检测并加载文件
        
        :param path: 文件路径
        :return: SequenceData
        """
        loader, detected_type = cls.auto_detect(path, **kwargs)
        print(f"检测到格式: {detected_type}")
        return loader.load(path)


# ============ 可视化器 ============

class AnimationViewer:
    """
    动画可视化器
    
    负责将 SequenceData 渲染到 aitviewer 中
    """
    
    def __init__(self, config: ViewerConfig = None):
        self.config = config or ViewerConfig()
        self._viewer = None
    
    def view(self, data: SequenceData, name: str = "sequence") -> None:
        """
        可视化单个序列
        
        :param data: 序列数据
        :param name: 序列名称（显示在 viewer 中）
        """
        print(f"序列信息:\n{data.summary()}")
        
        # 创建 viewer
        self._viewer = Viewer()
        self._viewer.playback_fps = self.config.fps
        
        # 创建并添加 mesh 对象
        meshes = self._create_meshes(data, name)
        for mesh in meshes:
            self._viewer.scene.add(mesh)
        
        # 设置相机
        self._setup_camera(meshes[0])
        
        print(f"启动 Viewer... (帧率: {self.config.fps} FPS)")
        self._viewer.run()
    
    def view_multiple(
        self, 
        data_list: List[SequenceData], 
        names: List[str] = None,
        spacing: float = 2.0,
        sync_frames: bool = True
    ) -> None:
        """
        并排可视化多个序列
        
        :param data_list: 序列数据列表
        :param names: 序列名称列表
        :param spacing: 序列间距
        :param sync_frames: 是否同步帧数（将静态网格重复到最大帧数）
        """
        if names is None:
            names = [f"seq_{i}" for i in range(len(data_list))]
        
        # 同步帧数：将静态网格重复到最大帧数
        if sync_frames:
            max_frames = max(d.n_frames for d in data_list)
            data_list = [d.repeat_to_frames(max_frames) for d in data_list]
        
        self._viewer = Viewer()
        self._viewer.playback_fps = self.config.fps
        
        cmap = plt.get_cmap('gist_rainbow')
        all_meshes = []
        
        for i, (data, name) in enumerate(zip(data_list, names)):
            # 为每个序列设置不同颜色
            color = cmap(i / max(len(data_list) - 1, 1))
            config = ViewerConfig(
                fps=self.config.fps,
                mesh_color=color,
                obstacle_color=self.config.obstacle_color,
                camera_distance=self.config.camera_distance,
                backface_culling=self.config.backface_culling,
                place_on_floor=self.config.place_on_floor
            )
            
            meshes = self._create_meshes(data, name, config)
            
            # 水平偏移
            x_shift = i * spacing
            for mesh in meshes:
                mesh.vertices[..., 0] += x_shift
            
            all_meshes.extend(meshes)
        
        # 添加到场景
        for mesh in all_meshes:
            self._viewer.scene.add(mesh)
        
        # 设置相机（调整距离以适应多个序列）
        adjusted_distance = self.config.camera_distance + len(data_list) * spacing / 2
        self._setup_camera(all_meshes[0], adjusted_distance)
        
        print(f"启动 Viewer... (帧率: {self.config.fps} FPS, 序列数: {len(data_list)})")
        self._viewer.run()
    
    def view_with_overlay(
        self,
        sequence_data: SequenceData,
        static_data: SequenceData,
        sequence_name: str = "sequence",
        static_name: str = "static"
    ) -> None:
        """
        叠加显示动画序列和静态网格（在同一位置）
        
        :param sequence_data: 动画序列数据
        :param static_data: 静态网格数据
        :param sequence_name: 序列名称
        :param static_name: 静态网格名称
        """
        # 将静态网格重复到与序列相同的帧数
        static_repeated = static_data.repeat_to_frames(sequence_data.n_frames)
        
        self._viewer = Viewer()
        self._viewer.playback_fps = self.config.fps
        
        # 创建序列网格
        seq_meshes = self._create_meshes(sequence_data, sequence_name)
        
        # 创建静态网格（使用不同颜色）
        static_config = ViewerConfig(
            fps=self.config.fps,
            mesh_color=self.config.obstacle_color,  # 使用障碍物颜色
            camera_distance=self.config.camera_distance,
            backface_culling=self.config.backface_culling,
            place_on_floor=self.config.place_on_floor
        )
        static_meshes = self._create_meshes(static_repeated, static_name, static_config)
        
        # 添加到场景
        for mesh in seq_meshes + static_meshes:
            self._viewer.scene.add(mesh)
        
        # 设置相机
        self._setup_camera(seq_meshes[0])
        
        print(f"启动 Viewer... (帧率: {self.config.fps} FPS)")
        print(f"  - 序列: {sequence_name} ({sequence_data.n_frames} 帧)")
        print(f"  - 静态: {static_name}")
        self._viewer.run()
    
    def _create_meshes(
        self, 
        data: SequenceData, 
        name: str,
        config: ViewerConfig = None
    ) -> List[Meshes]:
        """从序列数据创建 Meshes 对象列表"""
        config = config or self.config
        meshes = []
        
        # 处理顶点位置
        vertices = data.vertices
        if config.place_on_floor:
            vertices = place_on_floor(vertices)
        
        # 主网格
        color = adjust_color(config.mesh_color)
        main_mesh = Meshes(vertices, data.faces, name=f"{name}_main", color=color)
        main_mesh.backface_culling = config.backface_culling
        meshes.append(main_mesh)
        
        # 障碍物网格
        if data.has_obstacle:
            obs_vertices = data.obstacle_vertices
            if config.place_on_floor:
                obs_vertices = place_on_floor(obs_vertices)
            
            obs_color = adjust_color(config.obstacle_color)
            obs_mesh = Meshes(
                obs_vertices, 
                data.obstacle_faces, 
                name=f"{name}_obstacle", 
                color=obs_color
            )
            obs_mesh.backface_culling = config.backface_culling
            meshes.append(obs_mesh)
        
        return meshes
    
    def _setup_camera(self, target_mesh: Meshes, distance: float = None) -> None:
        """设置跟随目标网格的相机"""
        distance = distance or self.config.camera_distance
        
        positions, targets = aitviewer_path.lock_to_node(target_mesh, [0, 0, distance])
        camera = PinholeCamera(
            positions,
            targets,
            self._viewer.window_size[0],
            self._viewer.window_size[1],
            viewer=self._viewer
        )
        
        self._viewer.scene.add(camera)
        self._viewer.set_temp_camera(camera)


# ============ 便捷函数 ============

def view_sequence(
    path: str,
    sequence_type: str = "auto",
    gender: str = "neutral",
    config: ViewerConfig = None
) -> None:
    """
    便捷函数：加载并可视化序列文件
    
    :param path: pkl 文件路径
    :param sequence_type: 序列类型 ("auto", "mesh", "smpl")
    :param gender: SMPL 性别 (仅 SMPL 格式有效)
    :param config: 可视化配置
    """
    print(f"加载文件: {path}")
    
    # 获取加载器
    if sequence_type == "auto":
        loader, detected_type = SequenceLoaderFactory.auto_detect(path, gender=gender)
        print(f"检测到格式: {detected_type}")
    else:
        loader = SequenceLoaderFactory.create(sequence_type, gender=gender)
    
    # 加载数据
    data = loader.load(path)
    
    # 可视化
    viewer = AnimationViewer(config)
    viewer.view(data)


def view_multiple(
    paths: List[str],
    sequence_type: str = "auto",
    gender: str = "neutral",
    config: ViewerConfig = None,
    spacing: float = 2.0
) -> None:
    """
    便捷函数：并排可视化多个序列文件
    """
    data_list = []
    names = []
    
    for i, path in enumerate(paths):
        print(f"加载文件 {i+1}/{len(paths)}: {path}")
        
        if sequence_type == "auto":
            loader, _ = SequenceLoaderFactory.auto_detect(path, gender=gender)
        else:
            loader = SequenceLoaderFactory.create(sequence_type, gender=gender)
        
        data_list.append(loader.load(path))
        names.append(Path(path).stem)
    
    viewer = AnimationViewer(config)
    viewer.view_multiple(data_list, names, spacing)


def view_obj(
    obj_path: str,
    config: ViewerConfig = None
) -> None:
    """
    便捷函数：可视化单个 OBJ 文件
    
    :param obj_path: OBJ 文件路径
    :param config: 可视化配置
    """
    print(f"加载 OBJ 文件: {obj_path}")
    loader = OBJLoader()
    data = loader.load(obj_path)
    
    viewer = AnimationViewer(config)
    viewer.view(data, name=Path(obj_path).stem)


def view_obj_with_sequence(
    obj_path: str,
    sequence_path: str,
    mode: str = "overlay",
    sequence_type: str = "auto",
    gender: str = "neutral",
    config: ViewerConfig = None,
    spacing: float = 2.0
) -> None:
    """
    便捷函数：同时可视化 OBJ 文件和动画序列
    
    :param obj_path: OBJ 文件路径
    :param sequence_path: 动画序列 pkl 文件路径
    :param mode: 显示模式 "overlay"(叠加) 或 "side_by_side"(并排)
    :param sequence_type: 序列类型
    :param gender: SMPL 性别
    :param config: 可视化配置
    :param spacing: 并排模式下的间距
    """
    # 加载 OBJ
    print(f"加载 OBJ 文件: {obj_path}")
    obj_loader = OBJLoader()
    obj_data = obj_loader.load(obj_path)
    
    # 加载序列
    print(f"加载序列文件: {sequence_path}")
    if sequence_type == "auto":
        seq_loader, detected_type = SequenceLoaderFactory.auto_detect(sequence_path, gender=gender)
        print(f"检测到序列格式: {detected_type}")
    else:
        seq_loader = SequenceLoaderFactory.create(sequence_type, gender=gender)
    seq_data = seq_loader.load(sequence_path)
    
    viewer = AnimationViewer(config)
    
    if mode == "overlay":
        # 叠加显示
        viewer.view_with_overlay(
            sequence_data=seq_data,
            static_data=obj_data,
            sequence_name=Path(sequence_path).stem,
            static_name=Path(obj_path).stem
        )
    else:
        # 并排显示
        viewer.view_multiple(
            data_list=[obj_data, seq_data],
            names=[Path(obj_path).stem, Path(sequence_path).stem],
            spacing=spacing
        )


# ============ 命令行接口 ============

def parse_args():
    """解析命令行参数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='使用 aitviewer 可视化动画序列和静态网格（支持 pkl/obj 格式）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 自动检测格式（pkl 或 obj）
  python AnimationViewer.py path/to/file.pkl
  python AnimationViewer.py path/to/mesh.obj
  
  # 指定为 SMPL 格式
  python AnimationViewer.py path/to/pose_sequence.pkl --type smpl
  
  # 指定 SMPL 性别
  python AnimationViewer.py path/to/pose_sequence.pkl --type smpl --gender female
  
  # 同时显示 OBJ 和动画序列（叠加模式）
  python AnimationViewer.py garment.obj --with-sequence body_motion.pkl
  
  # 同时显示 OBJ 和动画序列（并排模式）
  python AnimationViewer.py garment.obj --with-sequence body_motion.pkl --mode side_by_side
"""
    )
    
    parser.add_argument(
        'file_path',
        type=str,
        nargs='?',
        default=None,
        help='文件路径 (pkl 或 obj)，不提供则使用默认文件'
    )
    parser.add_argument(
        '--type', '-t',
        type=str,
        choices=['auto', 'mesh', 'smpl', 'obj'],
        default='auto',
        help='文件类型 (默认: auto)'
    )
    parser.add_argument(
        '--gender', '-g',
        type=str,
        choices=['male', 'female', 'neutral'],
        default='neutral',
        help='SMPL 模型性别 (默认: neutral)'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=30,
        help='播放帧率 (默认: 30)'
    )
    parser.add_argument(
        '--distance',
        type=float,
        default=3.0,
        help='相机距离 (默认: 3.0)'
    )
    parser.add_argument(
        '--with-sequence', '-w',
        type=str,
        default=None,
        metavar='PKL_PATH',
        help='同时显示的动画序列 pkl 文件路径'
    )
    parser.add_argument(
        '--mode', '-m',
        type=str,
        choices=['overlay', 'side_by_side'],
        default='overlay',
        help='OBJ 与序列的显示模式: overlay(叠加) 或 side_by_side(并排) (默认: overlay)'
    )
    parser.add_argument(
        '--spacing',
        type=float,
        default=2.0,
        help='并排模式下的间距 (默认: 2.0)'
    )
    
    return parser.parse_args()


def get_default_path(file_type: str) -> Path:
    """获取默认的文件路径"""
    if file_type == 'smpl':
        return Paths.DEFAULT_DATA / 'pose_sequence.pkl'
    elif file_type == 'obj':
        return Paths.DEFAULT_DATA / 'tshirt.obj'
    return Paths.DEFAULT_DATA / 'mesh_sequence.pkl'


def main():
    """主入口函数"""
    args = parse_args()
    
    # 创建配置
    config = ViewerConfig(
        fps=args.fps,
        camera_distance=args.distance
    )
    
    # 确定主文件路径
    if args.file_path is None:
        default_path = get_default_path(args.type)
        if not default_path.exists():
            print(f"错误: 默认文件不存在: {default_path}")
            print("请提供文件路径作为参数")
            sys.exit(1)
        file_path = str(default_path)
        print(f"使用默认文件: {file_path}")
    else:
        file_path = args.file_path
    
    # 判断是否为 OBJ + 序列组合模式
    if args.with_sequence:
        # OBJ + 序列组合可视化
        if not file_path.lower().endswith('.obj'):
            print("警告: --with-sequence 参数通常与 OBJ 文件一起使用")
        
        view_obj_with_sequence(
            obj_path=file_path,
            sequence_path=args.with_sequence,
            mode=args.mode,
            sequence_type='auto',
            gender=args.gender,
            config=config,
            spacing=args.spacing
        )
    elif file_path.lower().endswith('.obj') or args.type == 'obj':
        # 单独的 OBJ 文件
        view_obj(file_path, config)
    else:
        # pkl 序列文件
        view_sequence(
            path=file_path,
            sequence_type=args.type,
            gender=args.gender,
            config=config
        )


if __name__ == '__main__':
    main()
