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
    'body': 'Tests/data/vto_dataset_mesh/body_sequence/tshirt_shape00_01_01.pkl',
    'cloth': 'Tests/data/vto_dataset_mesh/tshirt_sequence/tshirt_shape00_01_01.pkl',
    'template': 'Tests/data/tshirt.obj',
    'output': 'output.pkl',
}

# ============ 模型配置 ============
CONFIG = OmegaConf.create({
    'device': 'cuda:0',
    'n_coarse_levels': 3,
    'runner': {
        'mesh': {
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


def build_sample(body_path, cloth_path, template_path, n_coarse=3):
    """构建推理样本（包含coarse edge构建）"""
    body = pickle_load(body_path)
    cloth = pickle_load(cloth_path)
    template = obj2template(template_path) if template_path.endswith('.obj') else pickle_load(template_path)
    
    body_verts = torch.FloatTensor(body['vertices']).permute(1, 0, 2)
    cloth_verts = torch.FloatTensor(cloth['vertices']).permute(1, 0, 2)
    
    sample = HeteroData()
    
    # Body
    sample['obstacle'].prev_pos = body_verts
    sample['obstacle'].pos = body_verts
    sample['obstacle'].target_pos = body_verts
    sample['obstacle'].faces_batch = torch.LongTensor(body['faces']).T
    sample['obstacle'].vertex_type = torch.ones(body_verts.shape[0], 1).long()
    sample['obstacle'].vertex_level = torch.zeros(body_verts.shape[0], 1).long()
    
    # Cloth
    sample['cloth'].prev_pos = cloth_verts
    sample['cloth'].pos = cloth_verts
    sample['cloth'].target_pos = cloth_verts
    sample['cloth'].rest_pos = torch.FloatTensor(template.get('rest_pos', template['vertices']))
    sample['cloth'].faces_batch = torch.LongTensor(template['faces']).T
    sample['cloth'].vertex_type = torch.zeros(cloth_verts.shape[0], 1).long()
    
    # Mesh edges
    faces = torch.LongTensor(template['faces'])
    sample['cloth', 'mesh_edge', 'cloth'].edge_index = triangles_to_edges(faces.unsqueeze(0))
    
    # Coarse edges (多层级边用于GNN消息传递)
    if n_coarse > 0:
        if 'center' not in template:
            template = add_coarse_edges(template, n_coarse)
        center = np.random.choice(template['center'])
        coarse_dict = template.get('coarse_edges', {}).get(center) or \
                      make_coarse_edges(template['faces'], center, n_levels=n_coarse)
        
        vertex_level = np.zeros((cloth_verts.shape[0], 1), dtype=np.int64)
        for i in range(n_coarse):
            e = coarse_dict[i].astype(np.int64)
            e = np.concatenate([e, e[:, [1, 0]]], axis=0)
            sample['cloth', f'coarse_edge{i}', 'cloth'].edge_index = torch.tensor(e.T)
            vertex_level[np.unique(e)] = i + 1
        sample['cloth'].vertex_level = torch.tensor(vertex_level)
    else:
        sample['cloth'].vertex_level = torch.zeros(cloth_verts.shape[0], 1).long()
    
    # 添加 garment_name (cloth_obj.set_batch 需要)
    sample.garment_name = os.path.basename(template_path).split('.')[0]
    
    return sample


def main():
    parser = argparse.ArgumentParser(description='Mesh模式HOOD推理')
    parser.add_argument('--checkpoint', '-c', default=DEFAULT_PATHS['checkpoint'])
    parser.add_argument('--body', '-b', default=DEFAULT_PATHS['body'])
    parser.add_argument('--cloth', '-l', default=DEFAULT_PATHS['cloth'])
    parser.add_argument('--template', '-t', default=DEFAULT_PATHS['template'])
    parser.add_argument('--output', '-o', default=DEFAULT_PATHS['output'])
    parser.add_argument('--device', '-d', default='cuda:0')
    parser.add_argument('--steps', '-n', type=int, default=-1)
    args = parser.parse_args()
    
    device = args.device if torch.cuda.is_available() else 'cpu'
    CONFIG.device = device
    CONFIG.runner.mesh.device = device
    
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Body: {args.body}")
    print(f"Cloth: {args.cloth}")
    print(f"Template: {args.template}")
    
    # 创建模型和Runner
    model = load_module('models', CONFIG.model).create(CONFIG.model.postcvpr)
    from runners.mesh import Runner
    runner = Runner(model, {}, CONFIG.runner.mesh)
    runner.to(device)
    
    # 加载权重
    sd = torch.load(args.checkpoint, map_location=device, weights_only=False)
    runner.load_state_dict(sd.get('training_module', sd))
    runner.eval()
    
    # 构建样本并推理
    sample = build_sample(args.body, args.cloth, args.template, CONFIG.n_coarse_levels)
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
