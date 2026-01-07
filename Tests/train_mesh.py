#!/usr/bin/env python3
"""
Training script for HOOD using Mesh format animation sequences.

This script trains the network using dual mesh sequence mode:
body mesh sequence + tshirt mesh sequence + static garment template

Output checkpoints and logs are saved in the Tests folder.
"""

import os
import sys
import argparse

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from omegaconf import OmegaConf

from utils.arguments import create_modules, load_module

# Import custom mesh training utilities
from Tests.mesh_training_utils import (
    DualMeshDatasetWrapper, MeshRunner, 
    mesh_run_epoch, create_tensorboard_writer
)


def main():
    # Setup environment
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    torch.set_num_threads(1)
    
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Train HOOD with Dual Mesh format animation sequence'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='Tests/train_mesh_config.yaml',
        help='Path to the config file (default: Tests/train_mesh_config.yaml)'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to checkpoint to resume training'
    )
    parser.add_argument(
        '--garment',
        type=str,
        default=None,
        help='Path to garment template file (.pkl or .obj, overrides config)'
    )
    # Arguments for dual mesh sequence mode
    parser.add_argument(
        '--body_sequence',
        type=str,
        default=None,
        help='Path to body mesh sequence .pkl file'
    )
    parser.add_argument(
        '--tshirt_sequence',
        type=str,
        default=None,
        help='Path to tshirt mesh sequence .pkl file'
    )
    parser.add_argument(
        '--tshirt_template',
        type=str,
        default=None,
        help='Path to static tshirt .obj file for rest_pos'
    )
    parser.add_argument(
        '--log_every',
        type=int,
        default=100,
        help='Print loss log every N steps (default: 100)'
    )
    parser.add_argument(
        '--no_tensorboard',
        action='store_true',
        help='Disable TensorBoard logging'
    )
    args = parser.parse_args()
    
    # Make config path absolute
    config_path = os.path.abspath(args.config)
    
    # Check if config file exists
    if not os.path.exists(config_path):
        print(f"\n[ERROR] Config file not found: {config_path}")
        sys.exit(1)
    
    # Load configuration
    print("=" * 60)
    print("HOOD Mesh Training")
    print("Mode: Dual Mesh Sequence (Body + Tshirt)")
    print("=" * 60)
    print(f"\nLoading config from: {args.config}")
    
    # Create default config structure
    from utils.arguments import struct_fix
    
    default_config = OmegaConf.structured({
        'config': config_path,
        'device': 'cuda:0',
        'dataloader': {'num_workers': 0, 'batch_size': 1, 'pyg_data': True},
        'experiment': {
            'name': 'mesh_training',
            'save_checkpoint_every': 100000,
            'n_epochs': 200,
            'checkpoint_path': None,
            'max_iter': None
        },
        'detect_anomaly': False,
        'step_start': 0
    })
    
    # Load configuration file
    config = OmegaConf.load(config_path)
    
    # Merge with default config
    config = OmegaConf.merge(default_config, config)
    
    # Set struct to False to allow modifications
    struct_fix(config)
    
    # Load modules from config
    modules = {}
    modules['model'] = load_module('models', config.model)
    
    # Load criterions
    modules['criterions'] = {}
    conf_criterions = config.criterions
    for criterion_name in conf_criterions:
        criterion_module = load_module('criterions', config.criterions, criterion_name)
        modules['criterions'][criterion_name] = criterion_module
    
    # Override checkpoint path if provided
    if args.checkpoint is not None:
        config.experiment.checkpoint_path = os.path.abspath(args.checkpoint)
        print(f"Checkpoint: {config.experiment.checkpoint_path}")
    
    # ============ Dual Mesh Sequence Mode ============
    # Get project root directory (parent of Tests folder)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    def resolve_path(path_str):
        """Resolve path: if relative, make it relative to project root"""
        if path_str is None:
            return None
        if os.path.isabs(path_str):
            return path_str
        return os.path.join(project_root, path_str)
    
    # Priority: command line args > config file
    if args.body_sequence:
        body_sequence_path = os.path.abspath(args.body_sequence)
    else:
        body_sequence_path = resolve_path(config.dataloader.dataset.from_any_pose.body_sequence_path)
    
    if args.tshirt_sequence:
        tshirt_sequence_path = os.path.abspath(args.tshirt_sequence)
    else:
        tshirt_sequence_path = resolve_path(config.dataloader.dataset.from_any_pose.tshirt_sequence_path)
    
    # Use tshirt_template if provided, otherwise try garment arg or config
    if args.tshirt_template:
        garment_template_path = os.path.abspath(args.tshirt_template)
    elif args.garment:
        garment_template_path = os.path.abspath(args.garment)
    else:
        garment_template_path = resolve_path(config.dataloader.dataset.from_any_pose.tshirt_template_path)
    
    print(f"\nBody sequence: {body_sequence_path}")
    print(f"Tshirt sequence: {tshirt_sequence_path}")
    print(f"Garment template: {garment_template_path}")
    print(f"Output directory: {config.experiment.output_dir}")
    
    # Verify files exist
    for path, name in [(body_sequence_path, "Body sequence"), 
                       (tshirt_sequence_path, "Tshirt sequence"),
                       (garment_template_path, "Garment template")]:
        if not os.path.exists(path):
            print(f"\n[ERROR] {name} file not found: {path}")
            sys.exit(1)
    
    print("\nCreating training modules...")
    
    # Create training module for dual mode
    from torch_geometric.loader import DataLoader as PyGDataLoader
    
    # Create model - need to get the nested config (e.g., config.model.postcvpr)
    model_name = list(config.model.keys())[0]  # Get first key, e.g., 'postcvpr'
    model_cfg = config.model[model_name]
    model = modules['model'].create(model_cfg)
    
    # Create runner (we need mcfg from config)
    runner_cfg = config.runner.from_any_pose
    
    # Create criterion dict
    criterion_dict = {}
    for name, criterion_module in modules['criterions'].items():
        criterion_dict[name] = criterion_module.create(config.criterions[name])
    
    # Create MeshRunner
    training_module = MeshRunner(model, criterion_dict, runner_cfg)
    training_module.to(config.device)
    
    # Initialize cloth_obj
    from utils.cloth_and_material import ClothMatAug
    training_module.cloth_obj = ClothMatAug(None, always_overwrite_mass=True)
    
    # Create optimizer and scheduler
    from torch.optim import Adam
    from torch.optim.lr_scheduler import LambdaLR
    
    # Default optimizer config
    lr = 1e-4
    decay_rate = 0.1
    decay_steps = 200000
    decay_min = 0.0
    step_start = 0
    
    # Override with config if available
    if hasattr(runner_cfg, 'optimizer'):
        opt_cfg = runner_cfg.optimizer
        lr = getattr(opt_cfg, 'lr', lr)
        decay_rate = getattr(opt_cfg, 'decay_rate', decay_rate)
        decay_steps = getattr(opt_cfg, 'decay_steps', decay_steps)
        decay_min = getattr(opt_cfg, 'decay_min', decay_min)
        step_start = getattr(opt_cfg, 'step_start', step_start)
    
    optimizer = Adam(training_module.parameters(), lr=lr)
    
    def sched_fun(step):
        decay = decay_rate ** (step // decay_steps) + 1e-2
        decay = max(decay, decay_min)
        return decay
    
    scheduler = LambdaLR(optimizer, sched_fun)
    scheduler.last_epoch = step_start
    
    aux_modules = {
        'optimizer': optimizer,
        'scheduler': scheduler
    }
    
    # Get n_coarse_levels from config
    n_coarse_levels = config.dataloader.dataset.from_any_pose.get('n_coarse_levels', 4)
    
    # Create DualMeshDatasetWrapper
    print("Creating DualMeshDatasetWrapper for body + tshirt training...")
    dual_dataset = DualMeshDatasetWrapper(
        body_sequence_path=body_sequence_path,
        tshirt_sequence_path=tshirt_sequence_path,
        garment_template_path=garment_template_path,
        n_coarse_levels=n_coarse_levels,
        lookup_steps=5
    )
    
    # Create DataLoader
    class DataLoaderManager:
        def __init__(self, dataset, batch_size=1, num_workers=0):
            self.dataset = dataset
            self.batch_size = batch_size
            self.num_workers = num_workers
        
        def create_dataloader(self):
            return PyGDataLoader(
                self.dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers
            )
    
    dataloader_m = DataLoaderManager(dual_dataset, batch_size=1, num_workers=0)
    
    # Load checkpoint if provided
    if config.experiment.checkpoint_path is not None and os.path.exists(config.experiment.checkpoint_path):
        sd = torch.load(config.experiment.checkpoint_path)
        
        if 'training_module' in sd:
            training_module.load_state_dict(sd['training_module'])
            
            for k, v in aux_modules.items():
                if k in sd:
                    print(f'{k} LOADED!')
                    v.load_state_dict(sd[k])
        else:
            training_module.load_state_dict(sd)
        print('LOADED:', config.experiment.checkpoint_path)
    
    if config.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)
    
    # Create output directory
    output_dir = os.path.abspath(config.experiment.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Create TensorBoard writer
    writer = None
    if not args.no_tensorboard:
        writer = create_tensorboard_writer(output_dir, config.experiment.name)
    
    # Training loop
    global_step = config.step_start
    
    torch.manual_seed(57)
    np.random.seed(57)
    
    print("\n" + "=" * 60)
    print("Starting Training")
    print(f"Log every: {args.log_every} steps")
    print(f"TensorBoard: {'Enabled' if writer else 'Disabled'}")
    print("=" * 60)
    
    try:
        for i in range(config.experiment.n_epochs):
            print(f"\nEpoch {i + 1}/{config.experiment.n_epochs}")
            print("-" * 60)
            
            dataloader = dataloader_m.create_dataloader()
            global_step = mesh_run_epoch(
                training_module, aux_modules, dataloader, i, config,
                global_step=global_step, writer=writer, log_every=args.log_every
            )
            
            if config.experiment.max_iter is not None and global_step > config.experiment.max_iter:
                break
    finally:
        # Close TensorBoard writer
        if writer is not None:
            writer.close()
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Final global step: {global_step}")
    print(f"Checkpoints saved in: {output_dir}")
    if not args.no_tensorboard:
        print(f"TensorBoard logs in: {os.path.join(output_dir, 'tensorboard')}")
    print("=" * 60)


if __name__ == '__main__':
    main()
