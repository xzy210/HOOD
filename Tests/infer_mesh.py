#!/usr/bin/env python3
"""
Inference script for HOOD using Mesh format animation sequences.

This script runs inference using mesh format pose sequences instead of SMPL parameters.

Required inputs:
1. Body mesh sequence file (.pkl): Contains body animation vertices and faces
   Format: {'vertices': np.ndarray (N, V_body, 3), 'faces': np.ndarray (F, 3), 'num_frames': int}
   
2. Garment template file (.obj or .pkl): Static garment mesh for rest_pos/canonical position
   This defines the garment's topology and rest shape.
   
3. Garment initial position file (.obj or .pkl): Garment position at frame 0
   This should be aligned with the first frame of the body sequence.

Output:
    A .pkl file containing predicted cloth animation sequence.
"""

import os
import sys
import argparse
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from pathlib import Path
from omegaconf import OmegaConf
from torch_geometric.data import HeteroData

from utils.common import pickle_dump, pickle_load, move2device, triangles_to_edges
from utils.mesh_creation import obj2template
from utils.coarse import make_coarse_edges
from utils.defaults import DEFAULTS


def load_mesh_sequence(path: str) -> dict:
    """
    Load a mesh sequence from .pkl file.
    
    Expected format:
        {'vertices': np.ndarray (N, V, 3), 'faces': np.ndarray (F, 3), 'num_frames': int}
    
    Also supports legacy format:
        {'verts': np.ndarray (N, V, 3), 'faces': np.ndarray (F, 3)}
    """
    data = pickle_load(path)
    
    # Handle different key names
    if 'vertices' in data:
        vertices = data['vertices']
    elif 'verts' in data:
        vertices = data['verts']
    else:
        raise ValueError(f"Mesh sequence must contain 'vertices' or 'verts' key")
    
    faces = data['faces']
    num_frames = data.get('num_frames', vertices.shape[0])
    
    return {
        'vertices': vertices,
        'faces': faces,
        'num_frames': num_frames
    }


def load_garment(path: str) -> dict:
    """
    Load a garment from .obj or .pkl file.
    
    Returns:
        dict with 'vertices' and 'faces' keys
    """
    if path.endswith('.obj'):
        garment_dict = obj2template(path)
    else:
        garment_dict = pickle_load(path)
    
    return garment_dict


def build_inference_sample(
    body_sequence_path: str,
    garment_template_path: str,
    garment_init_path: str,
    n_coarse_levels: int = 3
) -> HeteroData:
    """
    Build a HeteroData sample for inference.
    
    Args:
        body_sequence_path: Path to body mesh sequence .pkl file
        garment_template_path: Path to garment template (.obj or .pkl) for rest_pos
        garment_init_path: Path to garment initial position (.obj or .pkl) for frame 0
        n_coarse_levels: Number of coarse levels for mesh hierarchy
        
    Returns:
        HeteroData sample ready for inference
    """
    sample = HeteroData()
    
    # ============ Load body sequence ============
    print(f"Loading body sequence: {body_sequence_path}")
    body_data = load_mesh_sequence(body_sequence_path)
    body_vertices = body_data['vertices']  # (N, V_body, 3)
    body_faces = body_data['faces']  # (F_body, 3)
    n_frames = body_data['num_frames']
    print(f"  -> {n_frames} frames, {body_vertices.shape[1]} vertices")
    
    # Add body vertices [V, N, 3]
    body_verts_tensor = torch.FloatTensor(body_vertices).permute(1, 0, 2)
    sample['obstacle'].prev_pos = body_verts_tensor
    sample['obstacle'].pos = body_verts_tensor
    sample['obstacle'].target_pos = body_verts_tensor
    
    # Add body faces
    sample['obstacle'].faces_batch = torch.LongTensor(body_faces.astype(np.int64)).T
    
    # Add body vertex type (all regular: 1)
    n_body_verts = body_verts_tensor.shape[0]
    sample['obstacle'].vertex_type = torch.ones(n_body_verts, 1).long()
    sample['obstacle'].vertex_level = torch.zeros(n_body_verts, 1).long()
    
    # ============ Load garment template (rest_pos) ============
    print(f"Loading garment template: {garment_template_path}")
    garment_template = load_garment(garment_template_path)
    rest_pos = torch.FloatTensor(garment_template['vertices'])
    print(f"  -> {rest_pos.shape[0]} vertices (rest position)")
    
    # ============ Load garment initial position ============
    print(f"Loading garment initial position: {garment_init_path}")
    garment_init = load_garment(garment_init_path)
    init_pos = torch.FloatTensor(garment_init['vertices'])
    print(f"  -> {init_pos.shape[0]} vertices (initial position)")
    
    # Verify vertex counts match
    if rest_pos.shape[0] != init_pos.shape[0]:
        raise ValueError(
            f"Garment template has {rest_pos.shape[0]} vertices, "
            f"but initial position has {init_pos.shape[0]} vertices. "
            f"They must have the same topology."
        )
    
    n_cloth_verts = rest_pos.shape[0]
    
    # ============ Set cloth positions ============
    # Expand init_pos to match the sequence length: [V, 3] -> [V, N, 3]
    cloth_pos_expanded = init_pos.unsqueeze(1).expand(-1, n_frames, -1)
    
    sample['cloth'].prev_pos = cloth_pos_expanded.clone()
    sample['cloth'].pos = cloth_pos_expanded.clone()
    sample['cloth'].target_pos = cloth_pos_expanded.clone()
    sample['cloth'].rest_pos = rest_pos
    
    # ============ Add cloth faces and edges ============
    cloth_faces = torch.LongTensor(garment_template['faces'])
    sample['cloth'].faces_batch = cloth_faces.T
    edges = triangles_to_edges(cloth_faces.unsqueeze(0))
    sample['cloth', 'mesh_edge', 'cloth'].edge_index = edges
    
    # Add cloth vertex type (all normal: 0)
    sample['cloth'].vertex_type = torch.zeros(n_cloth_verts, 1).long()
    
    # ============ Add coarse edges for cloth ============
    if n_coarse_levels > 0:
        faces_np = garment_template['faces']
        
        # Get or compute center nodes
        if 'center' not in garment_template:
            from utils.mesh_creation import add_coarse_edges
            garment_template = add_coarse_edges(garment_template, n_coarse_levels)
        
        center_nodes = garment_template['center']
        center = np.random.choice(center_nodes)
        
        # Get or compute coarse edges
        if 'coarse_edges' not in garment_template:
            garment_template['coarse_edges'] = {}
        
        if center in garment_template['coarse_edges']:
            coarse_edges_dict = garment_template['coarse_edges'][center]
        else:
            coarse_edges_dict = make_coarse_edges(faces_np, center, n_levels=n_coarse_levels)
            garment_template['coarse_edges'][center] = coarse_edges_dict
        
        # Add coarse edges to sample
        vertex_level = np.zeros((n_cloth_verts, 1)).astype(np.int64)
        for i in range(n_coarse_levels):
            key = f'coarse_edge{i}'
            edges_coarse = coarse_edges_dict[i].astype(np.int64)
            edges_coarse = np.concatenate([edges_coarse, edges_coarse[:, [1, 0]]], axis=0)
            sample['cloth', key, 'cloth'].edge_index = torch.tensor(edges_coarse.T)
            
            # Update vertex level
            nodes_unique = np.unique(edges_coarse.reshape(-1))
            vertex_level[nodes_unique] = i + 1
        
        sample['cloth'].vertex_level = torch.tensor(vertex_level)
    else:
        sample['cloth'].vertex_level = torch.zeros(n_cloth_verts, 1).long()
    
    # Add metadata
    sample['sequence_name'] = os.path.basename(body_sequence_path)
    sample['garment_name'] = os.path.basename(garment_template_path)
    
    return sample


def resolve_path(path_str: str, project_root: str) -> str:
    """
    Resolve path: if relative, make it relative to project root.
    
    Args:
        path_str: Path string (can be relative or absolute)
        project_root: Project root directory
        
    Returns:
        Absolute path
    """
    if path_str is None:
        return None
    if os.path.isabs(path_str):
        return path_str
    return os.path.join(project_root, path_str)


def main():
    # Setup environment
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    torch.set_num_threads(1)
    
    # Get script directory and project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Default config file path (in same directory as script)
    default_config = os.path.join(script_dir, 'infer_mesh_config.yaml')
    
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Run inference with HOOD using Mesh format animation sequence',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 直接运行（使用默认配置文件 Tests/infer_mesh_config.yaml）
  python Tests/infer_mesh.py

  # 命令行参数会覆盖配置文件中的设置
  python Tests/infer_mesh.py \\
      --body_sequence Tests/data/body_sequence.pkl \\
      --output Tests/data/output.pkl

  # 指定其他配置文件
  python Tests/infer_mesh.py \\
      --config path/to/other_config.yaml
        """
    )
    parser.add_argument(
        '--config',
        type=str,
        default=default_config,
        help=f'Path to the config file (default: {default_config})'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to model checkpoint (.pth file), overrides config'
    )
    parser.add_argument(
        '--body_sequence',
        type=str,
        default=None,
        help='Path to body mesh sequence .pkl file, overrides config'
    )
    parser.add_argument(
        '--garment_template',
        type=str,
        default=None,
        help='Path to garment template file (.obj or .pkl) for rest_pos, overrides config'
    )
    parser.add_argument(
        '--garment_init',
        type=str,
        default=None,
        help='Path to garment initial position file (.obj or .pkl) for frame 0, overrides config'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path for the result .pkl file, overrides config'
    )
    parser.add_argument(
        '--n_steps',
        type=int,
        default=None,
        help='Number of steps to simulate (-1 for full sequence), overrides config'
    )
    parser.add_argument(
        '--n_coarse_levels',
        type=int,
        default=None,
        help='Number of coarse levels for mesh hierarchy, overrides config'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to run inference on, overrides config'
    )
    parser.add_argument(
        '--bare',
        action='store_true',
        default=None,
        help='Skip loss computation for faster inference, overrides config'
    )
    args = parser.parse_args()
    
    # ============ Load config file first ============
    config_path = os.path.abspath(args.config)
    if not os.path.exists(config_path):
        # Try relative to project root
        config_path = resolve_path(args.config, project_root)
    
    if not os.path.exists(config_path):
        print(f"\n[ERROR] Config file not found: {config_path}")
        sys.exit(1)
    
    config = OmegaConf.load(config_path)
    
    # ============ Merge config and command line args ============
    # Command line args override config file settings
    
    # Get inference config section (if exists)
    infer_cfg = config.get('inference', {})
    
    # Body sequence: command line > config
    if args.body_sequence is not None:
        body_seq_path = os.path.abspath(args.body_sequence)
    elif infer_cfg.get('body_sequence'):
        body_seq_path = resolve_path(infer_cfg.body_sequence, project_root)
    else:
        print("\n[ERROR] body_sequence is required. Set it in config or via --body_sequence")
        sys.exit(1)
    
    # Garment template: command line > config
    if args.garment_template is not None:
        garment_template_path = os.path.abspath(args.garment_template)
    elif infer_cfg.get('garment_template'):
        garment_template_path = resolve_path(infer_cfg.garment_template, project_root)
    else:
        print("\n[ERROR] garment_template is required. Set it in config or via --garment_template")
        sys.exit(1)
    
    # Garment init: command line > config
    if args.garment_init is not None:
        garment_init_path = os.path.abspath(args.garment_init)
    elif infer_cfg.get('garment_init'):
        garment_init_path = resolve_path(infer_cfg.garment_init, project_root)
    else:
        print("\n[ERROR] garment_init is required. Set it in config or via --garment_init")
        sys.exit(1)
    
    # Checkpoint: command line > config
    if args.checkpoint is not None:
        checkpoint_path = os.path.abspath(args.checkpoint)
    elif infer_cfg.get('checkpoint'):
        checkpoint_path = resolve_path(infer_cfg.checkpoint, project_root)
    else:
        print("\n[ERROR] checkpoint is required. Set it in config or via --checkpoint")
        sys.exit(1)
    
    # Output: command line > config > auto-generate
    if args.output is not None:
        output_path = os.path.abspath(args.output)
    elif infer_cfg.get('output'):
        output_path = resolve_path(infer_cfg.output, project_root)
    else:
        # Auto-generate output path
        output_dir = os.path.dirname(body_seq_path)
        body_name = os.path.splitext(os.path.basename(body_seq_path))[0]
        garment_name = os.path.splitext(os.path.basename(garment_template_path))[0]
        output_path = os.path.join(output_dir, f"output_{body_name}_{garment_name}.pkl")
    
    # n_steps: command line > config > default
    if args.n_steps is not None:
        n_steps = args.n_steps
    elif infer_cfg.get('n_steps') is not None:
        n_steps = infer_cfg.n_steps
    else:
        n_steps = -1
    
    # n_coarse_levels: command line > config > default
    if args.n_coarse_levels is not None:
        n_coarse_levels = args.n_coarse_levels
    elif config.get('n_coarse_levels') is not None:
        n_coarse_levels = config.n_coarse_levels
    else:
        n_coarse_levels = 3
    
    # device: command line > config > default
    if args.device is not None:
        device = args.device
    elif config.get('device'):
        device = config.device
    else:
        device = 'cuda:0'
    
    # bare: command line > config > default
    if args.bare is not None and args.bare:
        bare = True
    elif infer_cfg.get('bare') is not None:
        bare = infer_cfg.bare
    else:
        bare = False
    
    # ============ Verify input files exist ============
    print("=" * 60)
    print("HOOD Mesh Inference")
    print("=" * 60)
    print(f"\nConfig: {config_path}")
    
    # Check body sequence
    if not os.path.exists(body_seq_path):
        print(f"\n[ERROR] Body sequence file not found: {body_seq_path}")
        sys.exit(1)
    print(f"Body sequence: {body_seq_path}")
    
    # Check garment template
    if not os.path.exists(garment_template_path):
        print(f"\n[ERROR] Garment template file not found: {garment_template_path}")
        sys.exit(1)
    print(f"Garment template: {garment_template_path}")
    
    # Check garment init
    if not os.path.exists(garment_init_path):
        print(f"\n[ERROR] Garment initial position file not found: {garment_init_path}")
        sys.exit(1)
    print(f"Garment init pos: {garment_init_path}")
    
    # Check checkpoint
    if not os.path.exists(checkpoint_path):
        print(f"\n[ERROR] Checkpoint file not found: {checkpoint_path}")
        sys.exit(1)
    print(f"Checkpoint: {checkpoint_path}")
    
    print(f"Output: {output_path}")
    print(f"\nDevice: {device}")
    print(f"Coarse levels: {n_coarse_levels}")
    print(f"Steps: {n_steps if n_steps > 0 else 'all'}")
    print(f"Bare mode: {bare}")
    
    # ============ Load configuration and create modules ============
    print("\n" + "-" * 60)
    print("Loading configuration and creating modules...")
    
    from utils.arguments import load_module
    
    # Override device in config
    config.device = device
    
    # Load model module
    model_module = load_module('models', config.model)
    model_name = list(config.model.keys())[0]
    model_cfg = config.model[model_name]
    model = model_module.create(model_cfg)
    
    # Load criterions
    criterion_dict = {}
    if hasattr(config, 'criterions'):
        for criterion_name in config.criterions:
            criterion_module = load_module('criterions', config.criterions, criterion_name)
            criterion_dict[criterion_name] = criterion_module.create(config.criterions[criterion_name])
    
    # Create runner
    runner_cfg = config.runner.from_any_pose
    
    from runners.from_any_pose import Runner
    runner = Runner(model, criterion_dict, runner_cfg)
    runner.to(device)
    
    # ============ Load checkpoint ============
    print("\n" + "-" * 60)
    print("Loading checkpoint...")
    
    state_dict = torch.load(checkpoint_path, map_location=device)
    
    if 'training_module' in state_dict:
        runner.load_state_dict(state_dict['training_module'])
    else:
        runner.load_state_dict(state_dict)
    
    print(f"Checkpoint loaded: {checkpoint_path}")
    
    # Set model to eval mode
    runner.eval()
    
    # ============ Build inference sample ============
    print("\n" + "-" * 60)
    print("Building inference sample...")
    
    sample = build_inference_sample(
        body_sequence_path=body_seq_path,
        garment_template_path=garment_template_path,
        garment_init_path=garment_init_path,
        n_coarse_levels=n_coarse_levels
    )
    
    # Create a batch from the sample
    from torch_geometric.loader import DataLoader as PyGDataLoader
    dataloader = PyGDataLoader([sample], batch_size=1, shuffle=False)
    batch = next(iter(dataloader))
    
    # Move to device
    batch = move2device(batch, device)
    
    # ============ Run inference ============
    print("\n" + "-" * 60)
    print("Running inference...")
    
    start_time = time.time()
    
    with torch.no_grad():
        trajectories_dict = runner.valid_rollout(
            batch, 
            n_steps=n_steps, 
            bare=bare,
            record_time=True
        )
    
    total_time = time.time() - start_time
    
    # ============ Save results ============
    print("\n" + "-" * 60)
    print("Saving results...")
    
    # Create output directory if needed
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Save trajectory
    pickle_dump(dict(trajectories_dict), output_path)
    
    # Print summary
    n_predicted_frames = trajectories_dict['pred'].shape[0]
    print("\n" + "=" * 60)
    print("Inference completed!")
    print("=" * 60)
    print(f"\nPredicted frames: {n_predicted_frames}")
    print(f"Total time: {total_time:.2f}s")
    print(f"FPS: {n_predicted_frames / total_time:.2f}")
    print(f"\nOutput saved to: {output_path}")
    
    # Print output format info
    print("\nOutput format:")
    print("  'pred': np.ndarray (N, V_cloth, 3) - Predicted cloth positions")
    print("  'obstacle': np.ndarray (N, V_body, 3) - Body positions")
    print("  'cloth_faces': np.ndarray (F_cloth, 3) - Cloth faces")
    print("  'obstacle_faces': np.ndarray (F_body, 3) - Body faces")
    if 'metrics' in trajectories_dict:
        print("  'metrics': dict - Per-frame loss values")


if __name__ == '__main__':
    main()

