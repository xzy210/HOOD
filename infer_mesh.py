#!/usr/bin/env python3
"""Mesh模式推理脚本"""

import os
import argparse
import time
import numpy as np
import torch
from omegaconf import OmegaConf
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader as PyGDataLoader

from utils.common import pickle_load, move2device, triangles_to_edges
from utils.mesh_creation import obj2template, add_coarse_edges
from utils.coarse import make_coarse_edges
from utils.arguments import load_module

# ============ 默认参数配置 ============
DEFAULT_PATHS = {
    'checkpoint': 'hood_data/experiments/20251203_120152/checkpoints/step_0000060000.pth',
    # 'body': 'hood_data/EcoData/split_output/body_skinned_mesh.pkl',
    'body': '',
    'cloth': 'hood_data/EcoData/split_output/surface_LayeredClothing_A0F407.pkl',
    'template': 'hood_data/EcoData/split_output/surface_LayeredClothing_A0F407_ref.obj',
    'output': 'hood_data/temp/output_eco_data.pkl',
}

# ============ Pinned Vertices 配置 ============
# 设置需要固定的顶点索引列表，这些顶点的位置会跟随 cloth 序列中的位置
# 设为 None 或空列表 [] 表示不使用 pinned vertices
# DEFAULT_PINNED_INDICES = None  # 或 [] 或 [0, 1, 2, ...]
DEFAULT_PINNED_INDICES = [0,1,12,18,24,30,36,42,48,54,60,66]

# ============ Mesh Face 翻转配置 ============
# 如果 mesh 的 face 顶点顺序是反的（导致法线方向错误），设为 True 进行翻转
# 翻转会将 face [v0, v1, v2] 变为 [v0, v2, v1]，从而反转法线方向
# Body: 正确的法线应该指向 body 外部（远离身体中心）
# Cloth: 正确的法线应该指向布料外侧（通常是正面朝外）
FLIP_BODY_FACES = False
FLIP_CLOTH_FACES = False


# ============ 模型配置 ============
CONFIG = OmegaConf.create({
    'device': 'cuda:0',
    'n_coarse_levels': 3,
    'runner': {
        'postcvpr': {
            'device': 'cuda:0',
            'push_eps': 2e-3,
            'initial_ts': 1/3,
            'regular_ts': 1/30,
            'overwrite_pos_every_step': False,
            'material': {
                # min/max 用于归一化计算（即使使用 override 也需要）
                'density_min': 0.0434,
                'density_max': 0.7,
                'lame_mu_min': 15909,
                'lame_mu_max': 63636,
                'lame_lambda_min': 3535.4,
                'lame_lambda_max': 93333.7,
                'bending_coeff_min': 6.37e-08,
                'bending_coeff_max': 0.00131,
                'bending_multiplier': 1.0,
                # override: 推理时使用固定值
                'density_override': 0.20022,
                'lame_mu_override': 23600.0,
                'lame_lambda_override': 44400,
                'bending_coeff_override': 3.96e-05,
            }
        }
    },
    'model': {
        'postcvpr': {
            'core_model': 'postcvpr',
            'architecture': "f,c0|f,c0|f,c0|d:c0,c1|c0,c1|c0,c1|d:c1|c1|c1|u:c0,c1|c0,c1|c0,c1|u:f,c0|f,c0|f,c0",
            'collision_radius': 3e-2,
            'n_coarse_levels': 3
        }
    }
})


def load_pinned_indices(pinned_arg):
    """加载 pinned indices
    
    Args:
        pinned_arg: 可以是以下格式之一：
            - 逗号分隔的索引字符串: "0,1,2,3"
            - 文件路径 (.txt 每行一个索引, .pkl 包含列表, .npy numpy数组)
            - None 或空字符串
    
    Returns:
        list of int 或 None
    """
    if not pinned_arg:
        return None
    
    # 如果是文件路径
    if os.path.isfile(pinned_arg):
        ext = os.path.splitext(pinned_arg)[1].lower()
        if ext == '.txt':
            # 文本文件：每行一个索引，或逗号/空格分隔
            with open(pinned_arg, 'r') as f:
                content = f.read()
            # 支持多种分隔符
            indices = []
            for line in content.strip().split('\n'):
                for part in line.replace(',', ' ').split():
                    if part.strip().isdigit():
                        indices.append(int(part.strip()))
            return indices
        elif ext == '.pkl':
            data = pickle_load(pinned_arg)
            if isinstance(data, dict) and 'pinned_indices' in data:
                return list(data['pinned_indices'])
            elif isinstance(data, (list, np.ndarray)):
                return list(data)
            else:
                raise ValueError(f"无法从 {pinned_arg} 解析 pinned indices")
        elif ext == '.npy':
            return list(np.load(pinned_arg))
        else:
            raise ValueError(f"不支持的文件格式: {ext}")
    
    # 否则视为逗号分隔的索引字符串
    try:
        indices = [int(x.strip()) for x in pinned_arg.split(',') if x.strip()]
        return indices
    except ValueError:
        raise ValueError(f"无法解析 pinned indices: {pinned_arg}")


def build_sample(body_path, cloth_path, template_path, n_coarse=3, pinned_indices=None, 
                 flip_body_faces=False, flip_cloth_faces=False):
    """构建推理样本（包含coarse edge构建）
    
    Args:
        body_path: 身体序列 pkl 路径，如果为 None 或数据为空，将使用虚拟 body（不参与碰撞）
        cloth_path: 布料序列 pkl 路径  
        template_path: 布料模板路径 (.obj 或 .pkl)
        n_coarse: coarse edge 层级数
        pinned_indices: pinned 顶点索引列表，这些顶点的位置会跟随 cloth 序列中的位置
                        例如: [0, 1, 2, 100, 101] 表示索引为 0,1,2,100,101 的顶点是 pinned
        flip_body_faces: 是否翻转 body mesh 的 face 顶点顺序以反转法线方向
                         如果碰撞检测异常（布料穿透身体），可能需要设为 True
        flip_cloth_faces: 是否翻转 cloth mesh 的 face 顶点顺序以反转法线方向
                          通常影响弯曲能量计算（相邻面法线夹角）
    
    Returns:
        HeteroData sample
        
    Note:
        使用 pinned_indices 时，pinned 顶点的位置会在每一步从 cloth 序列的 target_pos 中读取。
        因此 cloth 序列数据必须包含所有帧的顶点位置（包括 pinned 顶点的期望位置）。
        
        关于 face 翻转:
        - Face 顶点顺序决定法线方向（右手定则）
        - Body: 正确的法线应该指向 body 外部
        - Cloth: 正确的法线应该指向布料正面（外侧）
        - 如果 face 是 [v0, v1, v2]，翻转后变为 [v0, v2, v1]
        
    Example:
        # 通过 Blender 获取 pinned 顶点索引：
        # 1. 在 Blender 中打开 .obj 文件
        # 2. 进入编辑模式，选择要固定的顶点
        # 3. 在 Scripting 标签中运行:
        #    import bpy, bmesh
        #    obj = bpy.context.active_object
        #    bm = bmesh.from_edit_mesh(obj.data)
        #    selected = [v.index for v in bm.verts if v.select]
        #    print(selected)
    """
    cloth = pickle_load(cloth_path)
    template = obj2template(template_path) if template_path.endswith('.obj') else pickle_load(template_path)
    cloth_verts = torch.FloatTensor(cloth['vertices']).permute(1, 0, 2)
    n_frames = cloth_verts.shape[1]
    
    sample = HeteroData()
    
    # Body - 检查是否为空
    body_is_empty = False
    if body_path is None:
        body_is_empty = True
    else:
        body = pickle_load(body_path)
        if body is None or 'vertices' not in body or len(body['vertices']) == 0:
            body_is_empty = True
    
    if body_is_empty:
        # 创建虚拟的单点 body（放在很远的地方，不会影响布料）
        print("[Body] 数据为空，使用虚拟 body（远距离单点，不参与碰撞）")
        dummy_pos = torch.zeros(1, n_frames, 3)
        dummy_pos[..., :] = 1000.0  # 放在 (1000, 1000, 1000) 很远的地方
        sample['obstacle'].prev_pos = dummy_pos
        sample['obstacle'].pos = dummy_pos
        sample['obstacle'].target_pos = dummy_pos
        # 创建一个退化的三角形（三个顶点都是同一个点）
        sample['obstacle'].faces_batch = torch.zeros((3, 1), dtype=torch.long)
        sample['obstacle'].vertex_type = torch.ones(1, 1).long()
        sample['obstacle'].vertex_level = torch.zeros(1, 1).long()
    else:
        body_verts = torch.FloatTensor(body['vertices']).permute(1, 0, 2)
        sample['obstacle'].prev_pos = body_verts
        sample['obstacle'].pos = body_verts
        sample['obstacle'].target_pos = body_verts
        
        # 处理 body faces（可选翻转）
        body_faces = np.array(body['faces'])
        if flip_body_faces:
            # 翻转 face 顶点顺序: [v0, v1, v2] -> [v0, v2, v1]
            # 这会反转法线方向
            body_faces = body_faces[:, [0, 2, 1]]
            print("[Body Faces] 已翻转 face 顶点顺序以反转法线方向")
        sample['obstacle'].faces_batch = torch.LongTensor(body_faces).T
        
        sample['obstacle'].vertex_type = torch.ones(body_verts.shape[0], 1).long()
        sample['obstacle'].vertex_level = torch.zeros(body_verts.shape[0], 1).long()
    
    # Cloth
    sample['cloth'].prev_pos = cloth_verts
    sample['cloth'].pos = cloth_verts
    sample['cloth'].target_pos = cloth_verts
    sample['cloth'].rest_pos = torch.FloatTensor(template.get('rest_pos', template['vertices']))
    
    # 处理 cloth faces（可选翻转）
    cloth_faces = np.array(template['faces'])
    if flip_cloth_faces:
        # 翻转 face 顶点顺序: [v0, v1, v2] -> [v0, v2, v1]
        # 这会反转法线方向
        cloth_faces = cloth_faces[:, [0, 2, 1]]
        print("[Cloth Faces] 已翻转 face 顶点顺序以反转法线方向")
    sample['cloth'].faces_batch = torch.LongTensor(cloth_faces).T
    
    # Vertex type: 0=NORMAL, 3=HANDLE(pinned)
    n_cloth_verts = cloth_verts.shape[0]
    vertex_type = torch.zeros(n_cloth_verts, 1, dtype=torch.long)
    
    if pinned_indices is not None and len(pinned_indices) > 0:
        # 根据提供的索引列表设置 pinned vertices
        pinned_indices = np.array(pinned_indices, dtype=np.int64)
        # 验证索引范围
        if pinned_indices.max() >= n_cloth_verts or pinned_indices.min() < 0:
            raise ValueError(f"pinned_indices 超出范围 [0, {n_cloth_verts-1}]")
        vertex_type[pinned_indices] = 3  # NodeType.HANDLE
        print(f"[Pinned Verts] 已设置 {len(pinned_indices)} 个 pinned 顶点 (vertex_type=3)")
    
    sample['cloth'].vertex_type = vertex_type
    
    # Mesh edges (使用可能翻转后的 faces)
    faces = torch.LongTensor(cloth_faces)
    sample['cloth', 'mesh_edge', 'cloth'].edge_index = triangles_to_edges(faces.unsqueeze(0))
    
    # Coarse edges (多层级边用于GNN消息传递)
    if n_coarse > 0:
        if 'center' not in template:
            template = add_coarse_edges(template, n_coarse)
        center = np.random.choice(template['center'])
        coarse_dict = template.get('coarse_edges', {}).get(center) or \
                      make_coarse_edges(template['faces'], center, n_levels=n_coarse)
        
        vertex_level = np.zeros((n_cloth_verts, 1), dtype=np.int64)
        for i in range(n_coarse):
            e = coarse_dict[i]
            # 确保边数组是正确的 2D 形状 [N, 2]
            e = np.array(e, dtype=np.int64)
            if e.ndim == 1:
                # 如果是 1D 数组，可能是空的或需要 reshape
                if len(e) == 0:
                    e = np.zeros((0, 2), dtype=np.int64)
                else:
                    e = e.reshape(-1, 2)
            if len(e) > 0:
                e = np.concatenate([e, e[:, [1, 0]]], axis=0)
                sample['cloth', f'coarse_edge{i}', 'cloth'].edge_index = torch.tensor(e.T)
                vertex_level[np.unique(e)] = i + 1
            else:
                # 空边集
                sample['cloth', f'coarse_edge{i}', 'cloth'].edge_index = torch.zeros((2, 0), dtype=torch.long)
        sample['cloth'].vertex_level = torch.tensor(vertex_level)
    else:
        sample['cloth'].vertex_level = torch.zeros(n_cloth_verts, 1).long()
    
    # 添加 garment_name (cloth_obj.set_batch 需要)
    sample.garment_name = os.path.basename(template_path).split('.')[0]
    
    return sample


def main():
    parser = argparse.ArgumentParser(
        description='Mesh模式HOOD推理',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Pinned Vertices 使用说明:
  --pinned-indices 参数支持以下格式:
  
  1. 直接指定索引（逗号分隔）:
     --pinned-indices "0,1,2,100,101"
  
  2. 文本文件（每行一个索引，或逗号/空格分隔）:
     --pinned-indices pinned.txt
  
  3. Pickle 文件（包含 list 或 {'pinned_indices': list}）:
     --pinned-indices pinned.pkl
  
  4. NumPy 文件:
     --pinned-indices pinned.npy

获取 pinned 顶点索引的方法（使用 Blender）:
  1. 在 Blender 中打开 .obj 文件
  2. 进入编辑模式 (Tab)，选择要固定的顶点
  3. 在 Scripting 标签中运行:
     import bpy, bmesh
     obj = bpy.context.active_object
     bm = bmesh.from_edit_mesh(obj.data)
     selected = [v.index for v in bm.verts if v.select]
     print(selected)
''')
    parser.add_argument('--checkpoint', '-c', default=DEFAULT_PATHS['checkpoint'])
    parser.add_argument('--body', '-b', default=DEFAULT_PATHS['body'],
                        help='身体序列 pkl 路径，设为 "none" 或 "" 表示不使用 body（纯布料模拟）')
    parser.add_argument('--cloth', '-l', default=DEFAULT_PATHS['cloth'])
    parser.add_argument('--template', '-t', default=DEFAULT_PATHS['template'])
    parser.add_argument('--output', '-o', default=DEFAULT_PATHS['output'])
    parser.add_argument('--device', '-d', default='cuda:0')
    parser.add_argument('--steps', '-n', type=int, default=-1)
    parser.add_argument('--pinned-indices', '-p', default=None,
                        help='pinned 顶点索引: 逗号分隔的索引 或 文件路径 (.txt/.pkl/.npy)')
    parser.add_argument('--flip-body-faces', action='store_true', default=None,
                        help='翻转 body mesh 的 face 顶点顺序以反转法线方向 (用于修复碰撞检测问题)')
    parser.add_argument('--flip-cloth-faces', action='store_true', default=None,
                        help='翻转 cloth mesh 的 face 顶点顺序以反转法线方向')
    args = parser.parse_args()
    
    device = args.device if torch.cuda.is_available() else 'cpu'
    CONFIG.device = device
    CONFIG.runner.postcvpr.device = device
    
    # 加载 pinned indices（命令行参数优先，否则使用文件内默认值）
    if args.pinned_indices:
        pinned_indices = load_pinned_indices(args.pinned_indices)
    else:
        pinned_indices = DEFAULT_PINNED_INDICES
    
    # 确定是否翻转 faces（命令行参数优先，否则使用文件内默认值）
    flip_body_faces = args.flip_body_faces if args.flip_body_faces is not None else FLIP_BODY_FACES
    flip_cloth_faces = args.flip_cloth_faces if args.flip_cloth_faces is not None else FLIP_CLOTH_FACES
    
    # 处理 body 参数（支持设为 'none' 或空来禁用 body）
    body_path = args.body
    if not body_path or body_path.lower() in ('none', 'null'):
        body_path = None
    
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Body: {body_path if body_path else '(无 body，纯布料模拟)'}")
    print(f"Cloth: {args.cloth}")
    print(f"Template: {args.template}")
    if pinned_indices:
        print(f"Pinned Indices: {len(pinned_indices)} 个顶点")
    if flip_body_faces:
        print(f"Flip Body Faces: 已启用")
    if flip_cloth_faces:
        print(f"Flip Cloth Faces: 已启用")
    
    # 创建模型和Runner
    model = load_module('models', CONFIG.model).create(CONFIG.model.postcvpr)
    from runners.postcvpr import Runner
    runner = Runner(model, {}, CONFIG.runner.postcvpr)
    runner.to(device)
    
    # 加载权重
    sd = torch.load(args.checkpoint, map_location=device, weights_only=False)
    runner.load_state_dict(sd.get('training_module', sd))
    runner.eval()
    
    # 构建样本并推理
    sample = build_sample(body_path, args.cloth, args.template, 
                          CONFIG.n_coarse_levels, pinned_indices=pinned_indices,
                          flip_body_faces=flip_body_faces, flip_cloth_faces=flip_cloth_faces)
    batch = next(iter(PyGDataLoader([sample], batch_size=1)))
    batch = move2device(batch, device)
    
    print("推理中...")
    t0 = time.time()
    with torch.no_grad():
        result = runner.valid_rollout(batch, n_steps=args.steps, bare=True)
    dt = time.time() - t0
    
    import pickle
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, 'wb') as f:
        pickle.dump(dict(result), f)
    n = result['pred'].shape[0]
    print(f"完成! {n}帧, {dt:.1f}s, {n/dt:.1f}fps -> {args.output}")


if __name__ == '__main__':
    main()
