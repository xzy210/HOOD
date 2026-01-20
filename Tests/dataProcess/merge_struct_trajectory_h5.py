"""
合并Struct和Trajectory H5文件的数据处理脚本

该脚本读取一个struct h5文件和一个trajectory h5文件，输出多种格式的动画文件：
1. 按动画节点分割的骨骼动画文件
2. 全身骨骼位置动画文件（基于RefSkeleton/parent_indices构建edge）
3. 蒙皮网格动画文件（基于MeshSkinData计算）

用法:
    python merge_struct_trajectory_h5.py --struct <struct_file> --trajectory <trajectory_file> --output <output_file>

示例（合并输出单个文件）:
    python merge_struct_trajectory_h5.py \
        --struct ../../hood_data/EcoData/SK_TrainingSkirt_Struct_20260112_170103.hdf5 \
        --trajectory ../../hood_data/EcoData/SK_TrainingSkirt_Trajectory_20260112_170103.hdf5 \
        --output ../../hood_data/EcoData/SK_TrainingSkirt_Merged.hdf5

示例（按动画节点分割输出多个文件，包含全身骨骼和蒙皮网格）:
    python merge_struct_trajectory_h5.py \
        --struct ../../hood_data/EcoData/SK_TrainingSkirt_Struct_20260114_161713.hdf5 \
        --trajectory ../../hood_data/EcoData/SK_TrainingSkirt_Trajectory_20260114_161713.hdf5 \
        --output-dir ../../hood_data/EcoData/split_output \
        --split-by-node

坐标系转换:
    --coord-system 参数用于指定源数据的坐标系，自动转换到 OpenGL 坐标系（Y轴向上）
    - ue: UE引擎坐标系 (X前, Y右, Z上) -> 转换: (X, Y, Z) -> (X, Z, -Y)
    - unity: Unity坐标系 (X右, Y上, Z前，左手) -> 转换: (X, Y, Z) -> (X, Y, -Z)
    - none: 不进行坐标转换（默认）

输出文件格式:
    骨骼序列 (skeleton_sequence):
        - bone_names: [V,] 骨骼名称
        - dt: [N,] 每帧的时间间隔
        - edges: [E, 2] 边连接索引
        - vertices: [N, V, 3] 顶点位置序列
        
    网格序列 (mesh_sequence):
        - dt: [N,] 每帧的时间间隔
        - vertices: [N, V, 3] 顶点位置序列
        - faces: [F, 3] 三角面索引

Surfaces 数据处理:
    对于 Struct 文件中的 /Surfaces 组，每个 Surface 包含：
        - anim_node_uid: AnimNode 的 GUID
        - faces: [N, 3] 三角形，每个三角形记录3个骨骼名称
        - root_bone_names: [M] 根骨骼名称列表
    
    输出文件：
        - surface_<NodeClass>_<uid>.hdf5: Surface mesh 动画序列
        - surface_<NodeClass>_<uid>_ref.obj: 参考姿态静态网格（基于 ref_bone_poses_cs）
        - surface_<NodeClass>_<uid>_roots.txt: 根顶点索引列表
"""

import h5py
import numpy as np
import argparse
from pathlib import Path


def convert_coordinates(vertices: np.ndarray, coord_system: str) -> np.ndarray:
    """
    将顶点坐标从指定坐标系转换到 OpenGL 坐标系（Y轴向上，右手坐标系）
    
    Args:
        vertices: 顶点数组，形状为 [N, V, 3]
        coord_system: 源坐标系类型
            - 'ue': UE引擎 (X前, Y右, Z上，左手，cm) -> OpenGL (X, Z, -Y，m)
            - 'unity': Unity (X右, Y上, Z前，左手) -> OpenGL (X, Y, -Z)
            - 'none': 不转换
    
    Returns:
        转换后的顶点数组
    """
    if coord_system == 'none':
        return vertices
    
    converted = np.zeros_like(vertices)
    
    if coord_system == 'ue':
        # UE: X前, Y右, Z上 (左手坐标系，单位: cm)
        # OpenGL: X右, Y上, Z后 (右手坐标系，单位: m)
        # 转换: (X, Y, Z)_UE -> (X, Z, -Y)_OpenGL，并从 cm 转换为 m
        scale = 0.01  # cm -> m
        converted[..., 0] = vertices[..., 0] * scale   # X -> X
        converted[..., 1] = vertices[..., 2] * scale   # Z(上) -> Y(上)
        converted[..., 2] = -vertices[..., 1] * scale  # Y(右) -> -Z(前)
        print("  坐标转换: UE (X,Y,Z) cm -> OpenGL (X,Z,-Y) m")
        print("  单位转换: cm -> m (×0.01)")
    
    elif coord_system == 'unity':
        # Unity: X右, Y上, Z前 (左手坐标系)
        # OpenGL: X右, Y上, Z后 (右手坐标系)
        # 转换: 只需翻转Z轴
        converted[..., 0] = vertices[..., 0]   # X -> X
        converted[..., 1] = vertices[..., 1]   # Y -> Y
        converted[..., 2] = -vertices[..., 2]  # Z -> -Z
        print("  坐标转换: Unity (X,Y,Z) -> OpenGL (X,Y,-Z)")
    
    else:
        raise ValueError(f"未知的坐标系类型: {coord_system}")
    
    return converted


def _decode_string_array(arr: np.ndarray) -> list:
    """将字符串数组从bytes解码为str列表"""
    if arr.dtype == object:
        return [name.decode('utf-8') if isinstance(name, bytes) else name for name in arr]
    return list(arr)


def _decode_string_array_2d(arr: np.ndarray) -> list:
    """将二维字符串数组从bytes解码为str列表的列表"""
    result = []
    for row in arr:
        if isinstance(row, (list, np.ndarray)):
            decoded_row = []
            for item in row:
                if isinstance(item, bytes):
                    decoded_row.append(item.decode('utf-8'))
                else:
                    decoded_row.append(str(item) if not isinstance(item, str) else item)
            result.append(decoded_row)
        else:
            # 单个元素
            if isinstance(row, bytes):
                result.append(row.decode('utf-8'))
            else:
                result.append(str(row) if not isinstance(row, str) else row)
    return result


def load_struct_data(struct_file: str) -> dict:
    """
    从struct h5文件中加载数据
    
    Args:
        struct_file: struct h5文件路径
        
    Returns:
        包含bone_names, root_bone_names, end_bone_names等字段的字典
    """
    with h5py.File(struct_file, 'r') as f:
        # 读取 DynamicJoints 数据
        bone_names = _decode_string_array(f['DynamicJoints/bone_names'][:])
        
        # 读取 DynamicJoints 的 bone_indices（用于从全骨骼轨迹中提取位置）
        joint_bone_indices = None
        if 'DynamicJoints/bone_indices' in f:
            joint_bone_indices = np.array(f['DynamicJoints/bone_indices'][:], dtype=np.int32)
        
        # 读取root_bone_names和end_bone_names用于构建edges
        root_bone_names = _decode_string_array(f['Constraints/root_bone_names'][:])
        end_bone_names = _decode_string_array(f['Constraints/end_bone_names'][:])
        
        result = {
            'bone_names': bone_names,
            'joint_bone_indices': joint_bone_indices,
            'root_bone_names': root_bone_names,
            'end_bone_names': end_bone_names
        }
        
        # 读取动画节点相关数据（如果存在）
        if 'AnimNodes/anim_node_uids' in f:
            result['anim_node_uids'] = _decode_string_array(f['AnimNodes/anim_node_uids'][:])
            result['node_classes'] = _decode_string_array(f['AnimNodes/node_classes'][:])
            result['joint_node_uids'] = _decode_string_array(f['DynamicJoints/anim_node_uids'][:])
            result['constraint_node_uids'] = _decode_string_array(f['Constraints/anim_node_uids'][:])
        
        # 读取 RefSkeleton 数据（如果存在）
        if 'RefSkeleton/bone_names' in f:
            result['ref_skeleton'] = {
                'bone_names': _decode_string_array(f['RefSkeleton/bone_names'][:]),
                'parent_indices': np.array(f['RefSkeleton/parent_indices'][:], dtype=np.int32),
                'ref_bone_poses': np.array(f['RefSkeleton/ref_bone_poses'][:], dtype=np.float32),
            }
            if 'RefSkeleton/ref_bone_poses_cs' in f:
                result['ref_skeleton']['ref_bone_poses_cs'] = np.array(
                    f['RefSkeleton/ref_bone_poses_cs'][:], dtype=np.float32)
        
        # 读取 MeshSkinData 数据（如果存在）
        if 'MeshSkinData/ref_vertices' in f:
            result['mesh_skin_data'] = {
                'ref_vertices': np.array(f['MeshSkinData/ref_vertices'][:], dtype=np.float32),
                'bone_indices': np.array(f['MeshSkinData/bone_indices'][:], dtype=np.int32),
                'bone_weights': np.array(f['MeshSkinData/bone_weights'][:], dtype=np.float32),
                'triangles': np.array(f['MeshSkinData/triangles'][:], dtype=np.int32),
                'inv_bind_matrices': np.array(f['MeshSkinData/inv_bind_matrices'][:], dtype=np.float32),
            }
        
        # 读取 BodyRefSkeleton 数据（如果存在）
        if 'BodyRefSkeleton/bone_names' in f:
            result['body_ref_skeleton'] = {
                'bone_names': _decode_string_array(f['BodyRefSkeleton/bone_names'][:]),
                'parent_indices': np.array(f['BodyRefSkeleton/parent_indices'][:], dtype=np.int32),
                'ref_bone_poses': np.array(f['BodyRefSkeleton/ref_bone_poses'][:], dtype=np.float32),
            }
            if 'BodyRefSkeleton/ref_bone_poses_cs' in f:
                result['body_ref_skeleton']['ref_bone_poses_cs'] = np.array(
                    f['BodyRefSkeleton/ref_bone_poses_cs'][:], dtype=np.float32)
        
        # 读取 BodyMeshSkinData 数据（如果存在）
        if 'BodyMeshSkinData/ref_vertices' in f:
            result['body_mesh_skin_data'] = {
                'ref_vertices': np.array(f['BodyMeshSkinData/ref_vertices'][:], dtype=np.float32),
                'bone_indices': np.array(f['BodyMeshSkinData/bone_indices'][:], dtype=np.int32),
                'bone_weights': np.array(f['BodyMeshSkinData/bone_weights'][:], dtype=np.float32),
                'triangles': np.array(f['BodyMeshSkinData/triangles'][:], dtype=np.int32),
                'inv_bind_matrices': np.array(f['BodyMeshSkinData/inv_bind_matrices'][:], dtype=np.float32),
            }
        
        # 读取 Surfaces 数据（如果存在）
        if 'Surfaces' in f:
            surfaces = []
            surfaces_group = f['Surfaces']
            # 遍历所有 Surface 子组（按数字索引）
            surface_keys = sorted([k for k in surfaces_group.keys() if k.isdigit()], key=int)
            for key in surface_keys:
                surface_group = surfaces_group[key]
                surface_data = {
                    'anim_node_uid': surface_group['anim_node_uid'][()].decode('utf-8') 
                        if isinstance(surface_group['anim_node_uid'][()], bytes) 
                        else str(surface_group['anim_node_uid'][()]),
                    # faces 可能不存在（ribbon 类型节点没有三角面）
                    'faces': _decode_string_array_2d(surface_group['faces'][:]) if 'faces' in surface_group else [],
                    'root_bone_names': _decode_string_array(surface_group['root_bone_names'][:]),
                }
                surfaces.append(surface_data)
            result['surfaces'] = surfaces
    
    return result


def load_trajectory_data(trajectory_file: str, num_ref_bones: int = None, 
                         num_body_ref_bones: int = None) -> dict:
    """
    从trajectory h5文件中加载数据
    
    Args:
        trajectory_file: trajectory h5文件路径
        num_ref_bones: RefSkeleton 的骨骼数量（可选，用于验证）
        num_body_ref_bones: BodyRefSkeleton 的骨骼数量（可选，用于验证）
        
    Returns:
        包含 dt, positions, rotations, body_positions, body_rotations 的字典
    """
    with h5py.File(trajectory_file, 'r') as f:
        dt = np.array(f['DeltaTime'][:], dtype=np.float32)
        positions_flat = np.array(f['Positions'][:], dtype=np.float32)
        
        rotations_flat = None
        if 'Rotations' in f:
            rotations_flat = np.array(f['Rotations'][:], dtype=np.float32)
        
        scales_flat = None
        if 'Scales' in f:
            scales_flat = np.array(f['Scales'][:], dtype=np.float32)
        
        # 读取 Body 轨迹数据（如果存在）
        body_positions_flat = None
        body_rotations_flat = None
        body_scales_flat = None
        if 'BodyPositions' in f:
            body_positions_flat = np.array(f['BodyPositions'][:], dtype=np.float32)
        if 'BodyRotations' in f:
            body_rotations_flat = np.array(f['BodyRotations'][:], dtype=np.float32)
        if 'BodyScales' in f:
            body_scales_flat = np.array(f['BodyScales'][:], dtype=np.float32)
    
    n_frames = positions_flat.shape[0]
    n_pos_values = positions_flat.shape[1]
    
    # 推断骨骼数量
    if n_pos_values % 3 != 0:
        raise ValueError(f"Positions 数据 ({n_pos_values}) 无法整除3")
    
    num_bones = n_pos_values // 3
    
    if num_ref_bones is not None and num_bones != num_ref_bones:
        print(f"警告: 轨迹骨骼数 ({num_bones}) 与 RefSkeleton 骨骼数 ({num_ref_bones}) 不匹配")
    
    # 重塑为 [N, B, 3] 和 [N, B, 4]
    positions = positions_flat.reshape(n_frames, num_bones, 3)
    
    rotations = None
    if rotations_flat is not None:
        if rotations_flat.shape[1] == num_bones * 4:
            rotations = rotations_flat.reshape(n_frames, num_bones, 4)
        else:
            print(f"警告: Rotations 数据维度 ({rotations_flat.shape[1]}) 与预期 ({num_bones * 4}) 不匹配")
    
    scales = None
    if scales_flat is not None:
        if scales_flat.shape[1] == num_bones * 3:
            scales = scales_flat.reshape(n_frames, num_bones, 3)
        else:
            print(f"警告: Scales 数据维度 ({scales_flat.shape[1]}) 与预期 ({num_bones * 3}) 不匹配")
    
    result = {
        'dt': dt,
        'positions': positions,  # [N, B, 3] Part 全骨骼位置
        'rotations': rotations,  # [N, B, 4] Part 全骨骼旋转（四元数）
        'scales': scales,        # [N, B, 3] Part 全骨骼缩放
        'num_bones': num_bones
    }
    
    # 处理 Body 轨迹数据
    if body_positions_flat is not None:
        n_body_pos_values = body_positions_flat.shape[1]
        if n_body_pos_values % 3 != 0:
            print(f"警告: BodyPositions 数据 ({n_body_pos_values}) 无法整除3")
        else:
            num_body_bones = n_body_pos_values // 3
            if num_body_ref_bones is not None and num_body_bones != num_body_ref_bones:
                print(f"警告: Body 轨迹骨骼数 ({num_body_bones}) 与 BodyRefSkeleton 骨骼数 ({num_body_ref_bones}) 不匹配")
            
            result['body_positions'] = body_positions_flat.reshape(n_frames, num_body_bones, 3)
            result['num_body_bones'] = num_body_bones
            
            if body_rotations_flat is not None:
                if body_rotations_flat.shape[1] == num_body_bones * 4:
                    result['body_rotations'] = body_rotations_flat.reshape(n_frames, num_body_bones, 4)
                else:
                    print(f"警告: BodyRotations 数据维度 ({body_rotations_flat.shape[1]}) 与预期 ({num_body_bones * 4}) 不匹配")
            
            if body_scales_flat is not None:
                if body_scales_flat.shape[1] == num_body_bones * 3:
                    result['body_scales'] = body_scales_flat.reshape(n_frames, num_body_bones, 3)
                else:
                    print(f"警告: BodyScales 数据维度 ({body_scales_flat.shape[1]}) 与预期 ({num_body_bones * 3}) 不匹配")
    
    return result


def build_skeleton_edges_from_parents(parent_indices: np.ndarray) -> np.ndarray:
    """
    基于 parent_indices 构建骨骼层级的边
    
    Args:
        parent_indices: 父骨骼索引数组，根骨骼的父索引为 -1
        
    Returns:
        edges 数组，shape 为 (num_edges, 2)，每行为 [parent_idx, child_idx]
    """
    edges = []
    for child_idx, parent_idx in enumerate(parent_indices):
        if parent_idx >= 0:  # 跳过根骨骼
            edges.append([parent_idx, child_idx])
    return np.array(edges, dtype=np.int32) if edges else np.array([], dtype=np.int32).reshape(0, 2)


def quat_to_rotation_matrix_row_major(quat: np.ndarray) -> np.ndarray:
    """
    将四元数转换为 3x3 旋转矩阵（UE行向量右乘约定）
    
    UE的四元数格式为 (X, Y, Z, W)
    UE使用行向量右乘: v' = v * R
    
    Args:
        quat: 四元数 [x, y, z, w] 或 [..., 4]
        
    Returns:
        旋转矩阵 [3, 3] 或 [..., 3, 3]，行向量约定
    """
    # 支持批量处理
    original_shape = quat.shape[:-1]
    quat = quat.reshape(-1, 4)
    
    x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    
    # 归一化
    norm = np.sqrt(x*x + y*y + z*z + w*w + 1e-12)
    x, y, z, w = x/norm, y/norm, z/norm, w/norm
    
    # 构建旋转矩阵（UE行向量约定，是列向量约定的转置）
    n = quat.shape[0]
    R = np.zeros((n, 3, 3), dtype=np.float32)
    
    # UE的行向量约定旋转矩阵
    # 对于 v' = v * R，R 是列向量约定的转置
    R[:, 0, 0] = 1 - 2*(y*y + z*z)
    R[:, 0, 1] = 2*(x*y + z*w)  # 注意符号与列向量约定相反
    R[:, 0, 2] = 2*(x*z - y*w)
    R[:, 1, 0] = 2*(x*y - z*w)
    R[:, 1, 1] = 1 - 2*(x*x + z*z)
    R[:, 1, 2] = 2*(y*z + x*w)
    R[:, 2, 0] = 2*(x*z + y*w)
    R[:, 2, 1] = 2*(y*z - x*w)
    R[:, 2, 2] = 1 - 2*(x*x + y*y)
    
    return R.reshape(original_shape + (3, 3))


def compute_skinned_mesh(ref_vertices: np.ndarray, bone_indices: np.ndarray, 
                         bone_weights: np.ndarray, inv_bind_matrices: np.ndarray,
                         positions: np.ndarray, rotations: np.ndarray,
                         scales: np.ndarray = None) -> np.ndarray:
    """
    计算蒙皮网格的顶点位置（UE行向量右乘约定）
    
    UE蒙皮公式: v' = v * inv_bind * bone_transform
    其中：
    - v 是行向量 [1, 4]
    - inv_bind 是逆绑定矩阵 [4, 4]，将顶点从参考姿态变换到骨骼本地空间
    - bone_transform 是骨骼变换矩阵 [4, 4]，将顶点从骨骼本地空间变换到组件空间
    
    UE矩阵格式（行向量约定，带缩放）:
    [R00*Sx R01*Sx R02*Sx 0]
    [R10*Sy R11*Sy R12*Sy 0]
    [R20*Sz R21*Sz R22*Sz 0]
    [Tx     Ty     Tz     1]
    
    Args:
        ref_vertices: [V, 3] 参考姿态顶点位置
        bone_indices: [V, K] 每顶点的骨骼索引
        bone_weights: [V, K] 每顶点的骨骼权重
        inv_bind_matrices: [B, 16] 逆绑定矩阵（行主序 4x4，UE格式）
        positions: [N, B, 3] 每帧每骨骼的位置（组件空间）
        rotations: [N, B, 4] 每帧每骨骼的旋转（四元数 xyzw）
        scales: [N, B, 3] 每帧每骨骼的缩放（可选）
        
    Returns:
        [N, V, 3] 每帧的蒙皮顶点位置
    """
    N = positions.shape[0]  # 帧数
    B = positions.shape[1]  # 骨骼数
    V = ref_vertices.shape[0]  # 顶点数
    K = bone_indices.shape[1]  # 每顶点最大骨骼影响数
    
    # 重塑逆绑定矩阵为 [B, 4, 4]（UE行主序存储）
    inv_bind = inv_bind_matrices.reshape(B, 4, 4)
    
    # 构建每帧的骨骼变换矩阵 [N, B, 4, 4]（UE行向量约定）
    # UE格式: 旋转在左上3x3，平移在最后一行的前3列
    bone_transforms = np.zeros((N, B, 4, 4), dtype=np.float32)
    
    # 旋转部分（行向量约定）
    rot_matrices = quat_to_rotation_matrix_row_major(rotations)  # [N, B, 3, 3]
    
    # 应用缩放到旋转矩阵（按行缩放）
    # UE格式: 第i行乘以第i个缩放分量
    if scales is not None:
        # scales: [N, B, 3] -> 扩展为 [N, B, 3, 1] 用于广播
        scale_factors = scales[:, :, :, np.newaxis]  # [N, B, 3, 1]
        rot_matrices = rot_matrices * scale_factors  # [N, B, 3, 3]
    
    bone_transforms[:, :, :3, :3] = rot_matrices
    
    # 平移部分（UE行向量约定：平移在第4行）
    bone_transforms[:, :, 3, :3] = positions
    
    # 齐次坐标对角元素
    bone_transforms[:, :, 3, 3] = 1.0
    
    # 计算蒙皮矩阵: skinning_matrix = inv_bind @ bone_transform
    # 注意：UE的公式是 v' = v * inv_bind * bone_transform
    # 所以 skinning_matrix = inv_bind @ bone_transform
    skinning_matrices = np.einsum('bij,nbjk->nbik', inv_bind, bone_transforms)
    
    # 将参考顶点转换为齐次坐标 [V, 4]（行向量）
    ref_h = np.hstack([ref_vertices, np.ones((V, 1), dtype=np.float32)])
    
    # 蒙皮计算
    skinned = np.zeros((N, V, 3), dtype=np.float32)
    
    for k in range(K):
        bi = bone_indices[:, k]  # [V,] 骨骼索引
        w = bone_weights[:, k:k+1]  # [V, 1] 权重
        
        # 获取每个顶点对应的蒙皮矩阵 [N, V, 4, 4]
        transforms_k = skinning_matrices[:, bi, :, :]  # [N, V, 4, 4]
        
        # 变换顶点（行向量右乘）: [V, 4] @ [N, V, 4, 4] -> [N, V, 4]
        # einsum: 'vi,nvij->nvj' 表示 v[i] * M[i,j] -> result[j]
        transformed = np.einsum('vi,nvij->nvj', ref_h, transforms_k)
        
        # 加权累加
        skinned += w[np.newaxis, :, :] * transformed[:, :, :3]
    
    return skinned


def build_edges(bone_names: list, root_bone_names: list, end_bone_names: list, 
                constraint_mask: list = None) -> np.ndarray:
    """
    构建edges数组
    
    根据bone_names排序，将root_bone_names和end_bone_names配对构成边，
    然后将骨骼名称转换为排序后的索引
    
    Args:
        bone_names: 骨骼名称列表
        root_bone_names: 根骨骼名称列表
        end_bone_names: 末端骨骼名称列表
        constraint_mask: 可选的布尔掩码列表，指定哪些约束要包含
        
    Returns:
        edges数组，shape为(num_edges, 2)
    """
    # 对bone_names进行排序并创建名称到索引的映射
    sorted_bone_names = sorted(bone_names)
    name_to_index = {name: idx for idx, name in enumerate(sorted_bone_names)}
    
    # 构建edges
    edges = []
    skipped_count = 0
    
    for i, (root_name, end_name) in enumerate(zip(root_bone_names, end_bone_names)):
        # 如果有掩码，检查是否包含该约束
        if constraint_mask is not None and not constraint_mask[i]:
            continue
            
        # 检查两个骨骼名称是否都在bone_names中
        if root_name in name_to_index and end_name in name_to_index:
            root_idx = name_to_index[root_name]
            end_idx = name_to_index[end_name]
            edges.append([root_idx, end_idx])
        else:
            skipped_count += 1
    
    if skipped_count > 0:
        print(f"警告: 跳过了 {skipped_count} 条边（骨骼名称不在 bone_names 中）")
    
    return np.array(edges, dtype=np.int32) if edges else np.array([], dtype=np.int32).reshape(0, 2)


def extract_joint_positions(struct_data: dict, trajectory_data: dict) -> np.ndarray:
    """
    从全骨骼轨迹中提取 DynamicJoints 的位置
    
    Args:
        struct_data: load_struct_data 返回的结构数据
        trajectory_data: load_trajectory_data 返回的轨迹数据
        
    Returns:
        [N, num_joints, 3] DynamicJoints 的位置数组
    """
    positions = trajectory_data['positions']  # [N, B, 3]
    joint_bone_indices = struct_data.get('joint_bone_indices')
    
    if joint_bone_indices is not None:
        # 使用 bone_indices 从全骨骼轨迹中提取
        return positions[:, joint_bone_indices, :]
    else:
        # 兼容旧格式：假设 positions 就是 joints 的位置
        num_joints = len(struct_data['bone_names'])
        if positions.shape[1] == num_joints:
            return positions
        else:
            raise ValueError(
                f"无法提取 DynamicJoints 位置：缺少 bone_indices，且轨迹骨骼数 ({positions.shape[1]}) "
                f"与 DynamicJoints 数 ({num_joints}) 不匹配"
            )


def split_data_by_node(struct_data: dict, joint_positions: np.ndarray) -> dict:
    """
    按动画节点 uid 分割数据
    
    Args:
        struct_data: load_struct_data 返回的结构数据
        joint_positions: [N, num_joints, 3] DynamicJoints 的位置数组
        
    Returns:
        字典，key 为 (uid, node_class)，value 为该节点的数据字典
    """
    if 'anim_node_uids' not in struct_data:
        raise ValueError("Struct 文件中没有 AnimNodes 信息，无法按节点分割")
    
    anim_node_uids = struct_data['anim_node_uids']
    node_classes = struct_data['node_classes']
    joint_node_uids = struct_data['joint_node_uids']
    constraint_node_uids = struct_data['constraint_node_uids']
    
    all_bone_names = struct_data['bone_names']
    
    result = {}
    
    for uid, node_class in zip(anim_node_uids, node_classes):
        # 找出属于该 uid 的骨骼在 DynamicJoints 中的索引
        joint_indices = [i for i, juid in enumerate(joint_node_uids) if juid == uid]
        
        if not joint_indices:
            print(f"  警告: 节点 {node_class} (uid={uid}) 没有骨骼，跳过")
            continue
        
        # 提取该节点的骨骼名称
        node_bone_names = [all_bone_names[i] for i in joint_indices]
        
        # 创建约束掩码：只包含属于该节点的约束
        constraint_mask = [cuid == uid for cuid in constraint_node_uids]
        
        # 提取该节点的顶点数据
        node_vertices = joint_positions[:, joint_indices, :]
        
        result[(uid, node_class)] = {
            'bone_names': node_bone_names,
            'joint_indices': joint_indices,
            'vertices': node_vertices,
            'constraint_mask': constraint_mask
        }
    
    return result


def save_skeleton_h5(output_file: str, bone_names: list, dt: np.ndarray, 
                     edges: np.ndarray, vertices: np.ndarray, 
                     coord_system: str = 'none', node_class: str = None):
    """
    保存骨骼序列到 H5 文件
    
    Args:
        output_file: 输出文件路径
        bone_names: 骨骼名称列表（已排序）
        dt: 时间间隔数组
        edges: 边数组
        vertices: 顶点数组 [N, V, 3]
        coord_system: 源坐标系类型
        node_class: 节点类名（可选）
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(output_file, 'w') as f:
        # 设置格式标记（便于 Viewer 识别）
        f.attrs['format'] = 'skeleton_sequence'
        f.attrs['source_coord_system'] = coord_system
        if node_class:
            f.attrs['node_class'] = node_class
        
        # 保存bone_names（使用特殊编码存储字符串）
        dt_str = h5py.special_dtype(vlen=str)
        f.create_dataset('bone_names', data=bone_names, dtype=dt_str)
        
        # 保存dt
        f.create_dataset('dt', data=dt)
        
        # 保存edges
        f.create_dataset('edges', data=edges)
        
        # 保存vertices [N, V, 3]
        f.create_dataset('vertices', data=vertices)


def save_mesh_h5(output_file: str, dt: np.ndarray, vertices: np.ndarray, 
                 faces: np.ndarray, coord_system: str = 'none'):
    """
    保存网格动画到 H5 文件
    
    Args:
        output_file: 输出文件路径
        dt: 时间间隔数组
        vertices: 顶点数组 [N, V, 3]
        faces: 三角面索引 [F, 3]
        coord_system: 源坐标系类型
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(output_file, 'w') as f:
        # 设置格式标记（与 InteractiveAnimationViewer 兼容）
        f.attrs['format'] = 'mesh_sequence_hdf5'
        f.attrs['source_coord_system'] = coord_system
        
        # 保存dt
        f.create_dataset('dt', data=dt)
        
        # 保存vertices [N, V, 3]
        f.create_dataset('vertices', data=vertices)
        
        # 保存faces [F, 3]
        f.create_dataset('faces', data=faces)


def save_mesh_obj(output_file: str, vertices: np.ndarray, faces: np.ndarray,
                  coord_system: str = 'none'):
    """
    保存网格到 OBJ 文件（参考姿态）
    
    Args:
        output_file: 输出文件路径
        vertices: 顶点数组 [V, 3]
        faces: 三角面索引 [F, 3]（索引从0开始）
        coord_system: 源坐标系类型，用于坐标转换
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 坐标转换（如需要）
    if coord_system != 'none':
        # 将 [V, 3] 扩展为 [1, V, 3] 以复用 convert_coordinates
        vertices_3d = vertices[np.newaxis, :, :]
        vertices_3d = convert_coordinates(vertices_3d, coord_system)
        vertices = vertices_3d[0]
    
    with open(output_file, 'w') as f:
        # 写入文件头注释
        f.write(f"# OBJ file generated by merge_struct_trajectory_h5.py\n")
        f.write(f"# Vertices: {len(vertices)}, Faces: {len(faces)}\n")
        f.write(f"# Coordinate system: {coord_system}\n\n")
        
        # 写入顶点
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        
        f.write("\n")
        
        # 写入三角面（OBJ 索引从 1 开始）
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


def process_surface_data(surface: dict, ref_skeleton: dict) -> dict:
    """
    处理单个 Surface 数据，将骨骼名称转换为顶点索引
    
    Args:
        surface: Surface 数据字典，包含 faces（骨骼名称）和 root_bone_names
        ref_skeleton: RefSkeleton 数据字典，包含 bone_names 和 ref_bone_poses_cs
        
    Returns:
        处理后的数据字典，包含：
        - bone_names: 该 Surface 使用的唯一骨骼名称列表（作为顶点）
        - bone_indices: 骨骼在 RefSkeleton 中的索引
        - faces: [F, 3] 三角面索引（相对于 bone_names）
        - root_vertex_indices: root_bone_names 对应的顶点索引列表
    """
    faces_bone_names = surface['faces']  # [N, 3] 骨骼名称数组
    root_bone_names = surface['root_bone_names']
    
    ref_bone_names = ref_skeleton['bone_names']
    ref_name_to_idx = {name: idx for idx, name in enumerate(ref_bone_names)}
    
    # 收集所有唯一的骨骼名称
    unique_bones = []
    bone_set = set()
    for face in faces_bone_names:
        for bone_name in face:
            if bone_name not in bone_set:
                bone_set.add(bone_name)
                unique_bones.append(bone_name)
    
    # 创建骨骼名称到顶点索引的映射
    bone_to_vertex_idx = {name: idx for idx, name in enumerate(unique_bones)}
    
    # 获取每个骨骼在 RefSkeleton 中的索引
    bone_indices = []
    for bone_name in unique_bones:
        if bone_name in ref_name_to_idx:
            bone_indices.append(ref_name_to_idx[bone_name])
        else:
            print(f"  警告: 骨骼 '{bone_name}' 不在 RefSkeleton 中")
            bone_indices.append(-1)
    bone_indices = np.array(bone_indices, dtype=np.int32)
    
    # 将 faces 从骨骼名称转换为顶点索引
    faces = []
    for face in faces_bone_names:
        face_indices = [bone_to_vertex_idx[name] for name in face]
        faces.append(face_indices)
    faces = np.array(faces, dtype=np.int32)
    
    # 将 root_bone_names 转换为顶点索引
    root_vertex_indices = []
    for root_name in root_bone_names:
        if root_name in bone_to_vertex_idx:
            root_vertex_indices.append(bone_to_vertex_idx[root_name])
        else:
            print(f"  警告: 根骨骼 '{root_name}' 不在该 Surface 的顶点中")
    
    return {
        'bone_names': unique_bones,
        'bone_indices': bone_indices,
        'faces': faces,
        'root_vertex_indices': root_vertex_indices
    }


def save_root_vertices_txt(output_file: str, root_vertex_indices: list, bone_names: list = None):
    """
    保存 root 顶点索引到 txt 文件
    
    Args:
        output_file: 输出文件路径
        root_vertex_indices: root 顶点索引列表
        bone_names: 可选，骨骼名称列表（用于注释）
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        f.write(f"# Root vertex indices\n")
        f.write(f"# Total: {len(root_vertex_indices)}\n")
        if bone_names:
            f.write(f"# Format: vertex_index bone_name\n\n")
            for idx in root_vertex_indices:
                if 0 <= idx < len(bone_names):
                    f.write(f"{idx} {bone_names[idx]}\n")
                else:
                    f.write(f"{idx}\n")
        else:
            f.write(f"# Format: vertex_index\n\n")
            for idx in root_vertex_indices:
                f.write(f"{idx}\n")


def merge_and_save(struct_file: str, trajectory_file: str, output_file: str, coord_system: str = 'none'):
    """
    合并struct和trajectory数据并保存到新的h5文件（仅 DynamicJoints）
    
    Args:
        struct_file: struct h5文件路径
        trajectory_file: trajectory h5文件路径
        output_file: 输出h5文件路径
        coord_system: 源坐标系类型 ('ue', 'unity', 'none')
    """
    print(f"正在读取struct文件: {struct_file}")
    struct_data = load_struct_data(struct_file)
    
    num_ref_bones = None
    if 'ref_skeleton' in struct_data:
        num_ref_bones = len(struct_data['ref_skeleton']['bone_names'])
    
    print(f"正在读取trajectory文件: {trajectory_file}")
    trajectory_data = load_trajectory_data(trajectory_file, num_ref_bones)
    
    # 提取 DynamicJoints 位置
    joint_positions = extract_joint_positions(struct_data, trajectory_data)
    
    # 对bone_names排序
    bone_names = struct_data['bone_names']
    sorted_bone_names = sorted(bone_names)
    sort_indices = [bone_names.index(name) for name in sorted_bone_names]
    
    # 重新排列顶点
    vertices = joint_positions[:, sort_indices, :]
    
    # 坐标系转换
    if coord_system != 'none':
        print(f"正在进行坐标系转换 ({coord_system} -> OpenGL)...")
        vertices = convert_coordinates(vertices, coord_system)
    
    print("正在构建edges...")
    edges = build_edges(
        bone_names,
        struct_data['root_bone_names'],
        struct_data['end_bone_names']
    )
    
    print(f"正在保存到: {output_file}")
    save_skeleton_h5(output_file, sorted_bone_names, trajectory_data['dt'], 
                     edges, vertices, coord_system)
    
    print("=" * 50)
    print("合并完成！输出文件内容：")
    print("=" * 50)
    print(f"  format: skeleton_sequence")
    print(f"  source_coord_system: {coord_system}")
    print(f"  bone_names: shape=({len(sorted_bone_names)},)")
    print(f"  dt: shape={trajectory_data['dt'].shape}")
    print(f"  edges: shape={edges.shape}")
    print(f"  vertices: shape={vertices.shape}")
    print("=" * 50)


def merge_and_save_split(struct_file: str, trajectory_file: str, output_dir: str, 
                         coord_system: str = 'none', prefix: str = ''):
    """
    按动画节点分割并保存到多个h5文件，同时输出全身骨骼和蒙皮网格动画
    
    Args:
        struct_file: struct h5文件路径
        trajectory_file: trajectory h5文件路径
        output_dir: 输出目录路径
        coord_system: 源坐标系类型 ('ue', 'unity', 'none')
        prefix: 输出文件名前缀
    """
    print(f"正在读取struct文件: {struct_file}")
    struct_data = load_struct_data(struct_file)
    
    # 获取 RefSkeleton 信息（Part）
    num_ref_bones = None
    has_ref_skeleton = 'ref_skeleton' in struct_data
    has_mesh_skin = 'mesh_skin_data' in struct_data
    
    if has_ref_skeleton:
        num_ref_bones = len(struct_data['ref_skeleton']['bone_names'])
        print(f"  RefSkeleton 骨骼数: {num_ref_bones}")
    
    # 获取 BodyRefSkeleton 信息
    num_body_ref_bones = None
    has_body_ref_skeleton = 'body_ref_skeleton' in struct_data
    has_body_mesh_skin = 'body_mesh_skin_data' in struct_data
    
    if has_body_ref_skeleton:
        num_body_ref_bones = len(struct_data['body_ref_skeleton']['bone_names'])
        print(f"  BodyRefSkeleton 骨骼数: {num_body_ref_bones}")
    
    num_dynamic_joints = len(struct_data['bone_names'])
    print(f"  DynamicJoints 数量: {num_dynamic_joints}")
    
    if 'anim_node_uids' in struct_data:
        print(f"  动画节点数量: {len(struct_data['anim_node_uids'])}")
    
    if has_mesh_skin:
        mesh_skin = struct_data['mesh_skin_data']
        print(f"  Part 网格顶点数: {mesh_skin['ref_vertices'].shape[0]}")
        print(f"  Part 网格三角面数: {len(mesh_skin['triangles']) // 3}")
    
    if has_body_mesh_skin:
        body_mesh_skin = struct_data['body_mesh_skin_data']
        print(f"  Body 网格顶点数: {body_mesh_skin['ref_vertices'].shape[0]}")
        print(f"  Body 网格三角面数: {len(body_mesh_skin['triangles']) // 3}")
    
    if 'surfaces' in struct_data:
        print(f"  Surfaces 数量: {len(struct_data['surfaces'])}")
        for i, surface in enumerate(struct_data['surfaces']):
            uid_short = surface['anim_node_uid'][:6] if len(surface['anim_node_uid']) >= 6 else surface['anim_node_uid']
            num_faces = len(surface['faces'])
            num_roots = len(surface['root_bone_names'])
            print(f"    Surface {i}: uid={uid_short}, faces={num_faces}, roots={num_roots}")
    
    print(f"\n正在读取trajectory文件: {trajectory_file}")
    trajectory_data = load_trajectory_data(trajectory_file, num_ref_bones, num_body_ref_bones)
    print(f"  帧数: {trajectory_data['positions'].shape[0]}")
    print(f"  Part 轨迹骨骼数: {trajectory_data['num_bones']}")
    if 'body_positions' in trajectory_data:
        print(f"  Body 轨迹骨骼数: {trajectory_data['num_body_bones']}")
    
    # 提取 DynamicJoints 位置（用于按节点分割）
    joint_positions = extract_joint_positions(struct_data, trajectory_data)
    
    # 坐标系转换（对全骨骼位置数据）
    positions = trajectory_data['positions']
    if coord_system != 'none':
        print(f"\n正在进行坐标系转换 ({coord_system} -> OpenGL)...")
        positions = convert_coordinates(positions, coord_system)
        joint_positions = convert_coordinates(joint_positions, coord_system)
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    dt = trajectory_data['dt']
    output_count = 0
    
    print(f"\n正在保存到目录: {output_dir}")
    print("=" * 60)
    
    # ============================================
    # 1. 输出按动画节点分割的骨骼动画文件
    # ============================================
    if 'anim_node_uids' in struct_data:
        print("\n[1] 按动画节点分割骨骼动画...")
        split_data = split_data_by_node(struct_data, joint_positions)
        
        for (uid, node_class), node_data in split_data.items():
            # 构建输出文件名
            safe_class_name = node_class.replace('/', '_').replace('\\', '_').replace('::', '_')
            uid_short = uid[:6] if len(uid) >= 6 else uid
            if prefix:
                filename = f"{prefix}_node_{safe_class_name}_{uid_short}.hdf5"
            else:
                filename = f"node_{safe_class_name}_{uid_short}.hdf5"
            output_file = output_path / filename
            
            # 对该节点的骨骼名称排序
            node_bone_names = node_data['bone_names']
            sorted_bone_names = sorted(node_bone_names)
            sort_indices = [node_bone_names.index(name) for name in sorted_bone_names]
            
            # 重新排列顶点数据
            node_vertices = node_data['vertices'][:, sort_indices, :]
            
            # 构建该节点的 edges
            edges = build_edges(
                node_bone_names,
                struct_data['root_bone_names'],
                struct_data['end_bone_names'],
                constraint_mask=node_data['constraint_mask']
            )
            
            # 保存
            save_skeleton_h5(str(output_file), sorted_bone_names, dt,
                           edges, node_vertices, coord_system, node_class)
            output_count += 1
            
            print(f"    {filename}")
            print(f"      骨骼: {len(sorted_bone_names)}, 边: {len(edges)}, 形状: {node_vertices.shape}")
    
    # ============================================
    # 2. 输出全身骨骼位置动画文件
    # ============================================
    if has_ref_skeleton:
        print("\n[2] 输出全身骨骼位置动画...")
        ref_skeleton = struct_data['ref_skeleton']
        ref_bone_names = ref_skeleton['bone_names']
        parent_indices = ref_skeleton['parent_indices']
        
        # 基于 parent_indices 构建边
        skeleton_edges = build_skeleton_edges_from_parents(parent_indices)
        
        # 全骨骼位置
        skeleton_vertices = positions  # [N, B, 3]
        
        # 输出文件名
        if prefix:
            filename = f"{prefix}_full_skeleton.hdf5"
        else:
            filename = "full_skeleton.hdf5"
        output_file = output_path / filename
        
        save_skeleton_h5(str(output_file), ref_bone_names, dt,
                        skeleton_edges, skeleton_vertices, coord_system, "RefSkeleton")
        output_count += 1
        
        print(f"    {filename}")
        print(f"      骨骼: {len(ref_bone_names)}, 边: {len(skeleton_edges)}, 形状: {skeleton_vertices.shape}")
    
    # ============================================
    # 3. 输出蒙皮网格动画文件
    # ============================================
    if has_mesh_skin and trajectory_data['rotations'] is not None:
        print("\n[3] 计算并输出蒙皮网格动画...")
        mesh_skin = struct_data['mesh_skin_data']
        
        # 使用原始（未转换坐标的）位置、旋转和缩放进行蒙皮计算
        raw_positions = trajectory_data['positions']  # 原始组件空间数据
        raw_rotations = trajectory_data['rotations']
        raw_scales = trajectory_data.get('scales')  # 可选的缩放数据
        
        # 计算蒙皮后的顶点
        skinned_vertices = compute_skinned_mesh(
            mesh_skin['ref_vertices'],
            mesh_skin['bone_indices'],
            mesh_skin['bone_weights'],
            mesh_skin['inv_bind_matrices'],
            raw_positions,
            raw_rotations,
            raw_scales
        )
        
        # 坐标系转换
        if coord_system != 'none':
            skinned_vertices = convert_coordinates(skinned_vertices, coord_system)
        
        # 处理三角面索引（原始为扁平数组，需要重塑）
        triangles = mesh_skin['triangles']
        if triangles.ndim == 1:
            faces = triangles.reshape(-1, 3)
        else:
            faces = triangles
        
        # 输出文件名
        if prefix:
            filename = f"{prefix}_skinned_mesh.hdf5"
        else:
            filename = "skinned_mesh.hdf5"
        output_file = output_path / filename
        
        save_mesh_h5(str(output_file), dt, skinned_vertices, faces, coord_system)
        output_count += 1
        
        print(f"    {filename}")
        print(f"      顶点: {skinned_vertices.shape[1]}, 面: {len(faces)}, 形状: {skinned_vertices.shape}")
    elif has_mesh_skin:
        print("\n[3] 跳过 Part 蒙皮网格（缺少旋转数据）")
    
    # ============================================
    # 4. 输出 Body 蒙皮网格动画文件
    # ============================================
    if has_body_mesh_skin and 'body_rotations' in trajectory_data:
        print("\n[4] 计算并输出 Body 蒙皮网格动画...")
        body_mesh_skin = struct_data['body_mesh_skin_data']
        
        # 使用原始（未转换坐标的）位置、旋转和缩放进行蒙皮计算
        body_raw_positions = trajectory_data['body_positions']
        body_raw_rotations = trajectory_data['body_rotations']
        body_raw_scales = trajectory_data.get('body_scales')  # 可选的缩放数据
        
        # 计算蒙皮后的顶点
        body_skinned_vertices = compute_skinned_mesh(
            body_mesh_skin['ref_vertices'],
            body_mesh_skin['bone_indices'],
            body_mesh_skin['bone_weights'],
            body_mesh_skin['inv_bind_matrices'],
            body_raw_positions,
            body_raw_rotations,
            body_raw_scales
        )
        
        # 坐标系转换
        if coord_system != 'none':
            body_skinned_vertices = convert_coordinates(body_skinned_vertices, coord_system)
        
        # 处理三角面索引（原始为扁平数组，需要重塑）
        body_triangles = body_mesh_skin['triangles']
        if body_triangles.ndim == 1:
            body_faces = body_triangles.reshape(-1, 3)
        else:
            body_faces = body_triangles
        
        # 输出文件名
        if prefix:
            filename = f"{prefix}_body_skinned_mesh.hdf5"
        else:
            filename = "body_skinned_mesh.hdf5"
        output_file = output_path / filename
        
        save_mesh_h5(str(output_file), dt, body_skinned_vertices, body_faces, coord_system)
        output_count += 1
        
        print(f"    {filename}")
        print(f"      顶点: {body_skinned_vertices.shape[1]}, 面: {len(body_faces)}, 形状: {body_skinned_vertices.shape}")
    elif has_body_mesh_skin:
        print("\n[4] 跳过 Body 蒙皮网格（缺少旋转数据）")
    
    # ============================================
    # 5. 输出参考姿态网格 OBJ 文件
    # ============================================
    print("\n[5] 导出参考姿态网格 OBJ 文件...")
    
    # 5.1 Part Mesh 参考姿态
    if has_mesh_skin:
        mesh_skin = struct_data['mesh_skin_data']
        ref_vertices = mesh_skin['ref_vertices']
        
        # 处理三角面索引
        triangles = mesh_skin['triangles']
        if triangles.ndim == 1:
            faces = triangles.reshape(-1, 3)
        else:
            faces = triangles
        
        # 输出文件名
        if prefix:
            filename = f"{prefix}_ref_mesh.obj"
        else:
            filename = "ref_mesh.obj"
        output_file = output_path / filename
        
        save_mesh_obj(str(output_file), ref_vertices, faces, coord_system)
        output_count += 1
        
        print(f"    {filename}")
        print(f"      顶点: {len(ref_vertices)}, 面: {len(faces)}")
    
    # 5.2 Body Mesh 参考姿态
    if has_body_mesh_skin:
        body_mesh_skin = struct_data['body_mesh_skin_data']
        body_ref_vertices = body_mesh_skin['ref_vertices']
        
        # 处理三角面索引
        body_triangles = body_mesh_skin['triangles']
        if body_triangles.ndim == 1:
            body_faces = body_triangles.reshape(-1, 3)
        else:
            body_faces = body_triangles
        
        # 输出文件名
        if prefix:
            filename = f"{prefix}_body_ref_mesh.obj"
        else:
            filename = "body_ref_mesh.obj"
        output_file = output_path / filename
        
        save_mesh_obj(str(output_file), body_ref_vertices, body_faces, coord_system)
        output_count += 1
        
        print(f"    {filename}")
        print(f"      顶点: {len(body_ref_vertices)}, 面: {len(body_faces)}")
    
    # ============================================
    # 6. 输出 Surface mesh 动画和静态 OBJ
    # ============================================
    has_surfaces = 'surfaces' in struct_data and len(struct_data['surfaces']) > 0
    
    if has_surfaces and has_ref_skeleton:
        print("\n[6] 输出 Surface mesh 数据...")
        ref_skeleton = struct_data['ref_skeleton']
        ref_bone_poses_cs = ref_skeleton.get('ref_bone_poses_cs')
        
        for i, surface in enumerate(struct_data['surfaces']):
            uid = surface['anim_node_uid']
            uid_short = uid[:6] if len(uid) >= 6 else uid
            
            # 查找对应的 node_class
            node_class = "Surface"
            if 'anim_node_uids' in struct_data:
                for j, anim_uid in enumerate(struct_data['anim_node_uids']):
                    if anim_uid == uid:
                        node_class = struct_data['node_classes'][j]
                        break
            
            safe_class_name = node_class.replace('/', '_').replace('\\', '_').replace('::', '_')
            
            # 检查是否有 faces 数据（ribbon 类型可能没有）
            if len(surface['faces']) == 0:
                print(f"    跳过 Surface {i} ({node_class}, uid={uid_short}): 无三角面数据")
                continue
            
            # 处理 Surface 数据
            processed = process_surface_data(surface, ref_skeleton)
            
            if len(processed['faces']) == 0:
                print(f"    跳过 Surface {i} ({node_class}, uid={uid_short}): 处理后无有效三角面")
                continue
            
            bone_names = processed['bone_names']
            bone_indices = processed['bone_indices']
            faces = processed['faces']
            root_vertex_indices = processed['root_vertex_indices']
            
            print(f"    Surface {i}: {node_class} (uid={uid_short})")
            print(f"      顶点: {len(bone_names)}, 面: {len(faces)}, 根顶点: {len(root_vertex_indices)}")
            
            # 6.1 输出动画 mesh (H5)
            # 从全骨骼轨迹中提取该 Surface 使用的骨骼位置
            valid_bone_mask = bone_indices >= 0
            if not np.all(valid_bone_mask):
                print(f"      警告: 有 {np.sum(~valid_bone_mask)} 个骨骼不在 RefSkeleton 中，跳过动画输出")
            else:
                # 提取顶点位置序列 [N, V, 3]
                surface_vertices = positions[:, bone_indices, :]
                
                # 输出文件名
                if prefix:
                    filename = f"{prefix}_surface_{safe_class_name}_{uid_short}.hdf5"
                else:
                    filename = f"surface_{safe_class_name}_{uid_short}.hdf5"
                output_file = output_path / filename
                
                save_mesh_h5(str(output_file), dt, surface_vertices, faces, coord_system)
                output_count += 1
                print(f"      动画: {filename}")
            
            # 6.2 输出静态 OBJ（基于 ref_bone_poses_cs）
            if ref_bone_poses_cs is not None and np.all(valid_bone_mask):
                # ref_bone_poses_cs 的形状应该是 [B, 3] 或 [B, 4, 4]
                if ref_bone_poses_cs.ndim == 2 and ref_bone_poses_cs.shape[1] == 3:
                    # [B, 3] 格式：直接是位置
                    ref_vertices = ref_bone_poses_cs[bone_indices]
                elif ref_bone_poses_cs.ndim == 2 and ref_bone_poses_cs.shape[1] >= 3:
                    # 可能是 [B, 7] 或其他格式，取前3列作为位置
                    ref_vertices = ref_bone_poses_cs[bone_indices, :3]
                else:
                    print(f"      警告: ref_bone_poses_cs 格式不支持: {ref_bone_poses_cs.shape}")
                    ref_vertices = None
                
                if ref_vertices is not None:
                    # 输出文件名
                    if prefix:
                        filename = f"{prefix}_surface_{safe_class_name}_{uid_short}_ref.obj"
                    else:
                        filename = f"surface_{safe_class_name}_{uid_short}_ref.obj"
                    output_file = output_path / filename
                    
                    save_mesh_obj(str(output_file), ref_vertices, faces, coord_system)
                    output_count += 1
                    print(f"      静态OBJ: {filename}")
            elif ref_bone_poses_cs is None:
                print(f"      跳过静态OBJ: 缺少 ref_bone_poses_cs 数据")
            
            # 6.3 输出 root vertices txt
            if len(root_vertex_indices) > 0:
                if prefix:
                    filename = f"{prefix}_surface_{safe_class_name}_{uid_short}_roots.txt"
                else:
                    filename = f"surface_{safe_class_name}_{uid_short}_roots.txt"
                output_file = output_path / filename
                
                save_root_vertices_txt(str(output_file), root_vertex_indices, bone_names)
                output_count += 1
                print(f"      根顶点: {filename}")
    elif has_surfaces and not has_ref_skeleton:
        print("\n[6] 跳过 Surface mesh（缺少 RefSkeleton 数据）")
    
    print("\n" + "=" * 60)
    print(f"处理完成！共输出 {output_count} 个文件到: {output_dir}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='合并Struct和Trajectory H5文件')
    parser.add_argument('--struct', type=str, required=True,
                        help='Struct H5文件路径')
    parser.add_argument('--trajectory', type=str, required=True,
                        help='Trajectory H5文件路径')
    parser.add_argument('--output', type=str, default=None,
                        help='输出H5文件路径（与 --split-by-node 互斥）')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='输出目录路径（与 --split-by-node 配合使用）')
    parser.add_argument('--split-by-node', action='store_true',
                        help='按动画节点分割输出多个文件')
    parser.add_argument('--prefix', type=str, default='',
                        help='输出文件名前缀（与 --split-by-node 配合使用）')
    parser.add_argument('--coord-system', type=str, default='none',
                        choices=['none', 'ue', 'unity'],
                        help='源数据坐标系类型，用于转换到OpenGL坐标系 (默认: none)')
    
    args = parser.parse_args()
    
    if args.split_by_node:
        # 分割模式
        if not args.output_dir:
            parser.error("--split-by-node 需要配合 --output-dir 使用")
        merge_and_save_split(args.struct, args.trajectory, args.output_dir, 
                            args.coord_system, args.prefix)
    else:
        # 普通合并模式
        if not args.output:
            parser.error("请指定 --output 或使用 --split-by-node 模式")
        merge_and_save(args.struct, args.trajectory, args.output, args.coord_system)


if __name__ == '__main__':
    main()

