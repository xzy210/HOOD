#!/usr/bin/env python3
"""
Training script for HOOD using Mesh format animation sequences.

This script trains the network using mesh format pose sequences instead of SMPL parameters.
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
from Tests.mesh_training_utils import MeshDatasetWrapper, MeshRunner


def setup_paths(config, mesh_sequence=None, garment=None):
    """
    Set up paths for mesh sequence and garment.
    
    For paths from command line: convert to absolute paths
    For paths from config file: keep as-is (may be relative to data_root)
    
    Args:
        config: OmegaConf configuration object
        mesh_sequence: Path to mesh sequence file (optional override)
        garment: Path to garment template file (.pkl or .obj, optional override)
    """
    data_config = config.dataloader.dataset.from_any_pose
    
    # Handle mesh sequence path (should be .pkl for mesh format)
    # If provided via command line, convert to absolute path
    if mesh_sequence is not None:
        data_config.pose_sequence_path = os.path.abspath(mesh_sequence)
    
    # Handle garment path (supports both .pkl and .obj formats)
    # If provided via command line, convert to absolute path
    if garment is not None:
        data_config.garment_template_path = os.path.abspath(garment)
    
    # Handle obstacle_dict_file if present
    if hasattr(data_config, 'obstacle_dict_file') and data_config.obstacle_dict_file is not None:
        # Keep obstacle file path as-is from config
        pass
    
    return config


def main():
    # Setup environment
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    torch.set_num_threads(1)
    
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Train HOOD with Mesh format animation sequence'
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
        '--mesh_sequence',
        type=str,
        default=None,
        help='Path to mesh sequence .pkl file (overrides config)'
    )
    parser.add_argument(
        '--garment',
        type=str,
        default=None,
        help='Path to garment template file (.pkl or .obj, overrides config)'
    )
    args = parser.parse_args()
    
    # Load configuration
    print("=" * 60)
    print("HOOD Mesh Training")
    print("=" * 60)
    print(f"\nLoading config from: {args.config}")
    
    # Make config path absolute
    config_path = os.path.abspath(args.config)
    
    # Check if config file exists
    if not os.path.exists(config_path):
        print(f"\n[ERROR] Config file not found: {config_path}")
        sys.exit(1)
    
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
    modules['runner'] = load_module('runners', config.runner)
    
    # Load criterions
    modules['criterions'] = {}
    conf_criterions = config.criterions
    for criterion_name in conf_criterions:
        criterion_module = load_module('criterions', config.criterions, criterion_name)
        modules['criterions'][criterion_name] = criterion_module
    
    # Load dataset module
    dataset_module = load_module('datasets', config.dataloader.dataset)
    modules['dataset'] = dataset_module
    
    # Set up paths
    config = setup_paths(
        config,
        mesh_sequence=args.mesh_sequence,
        garment=args.garment
    )
    
    # Override checkpoint path if provided
    if args.checkpoint is not None:
        config.experiment.checkpoint_path = os.path.abspath(args.checkpoint)
        print(f"Checkpoint: {config.experiment.checkpoint_path}")
    
    # Print configuration info
    print(f"\nPose sequence type: {config.dataloader.dataset.from_any_pose.pose_sequence_type}")
    print(f"Mesh sequence: {config.dataloader.dataset.from_any_pose.pose_sequence_path}")
    print(f"Garment template: {config.dataloader.dataset.from_any_pose.garment_template_path}")
    print(f"Output directory: {config.experiment.output_dir}")
    
    # Helper function to check if file exists
    def check_file_exists(path, description):
        """Check if file exists, handling both relative and absolute paths."""
        if os.path.isabs(path):
            if not os.path.exists(path):
                return False, path
        else:
            # Try in data_root first
            from utils.defaults import DEFAULTS
            data_root_path = os.path.join(DEFAULTS.data_root, path)
            if os.path.exists(data_root_path):
                return True, data_root_path
            # Try current directory
            if os.path.exists(path):
                return True, path
            return False, path
        return True, path
    
    # Verify mesh sequence file exists
    mesh_seq_path = config.dataloader.dataset.from_any_pose.pose_sequence_path
    mesh_exists, mesh_actual_path = check_file_exists(mesh_seq_path, "Mesh sequence")
    if not mesh_exists:
        print(f"\n[ERROR] Mesh sequence file not found: {mesh_seq_path}")
        print("Checked locations:")
        if not os.path.isabs(mesh_seq_path):
            print(f"  - {os.path.join(DEFAULTS.data_root, mesh_seq_path)} (data_root)")
            print(f"  - {mesh_seq_path} (current directory)")
        print("\nTo generate sample data, run:")
        print("  python Tests/generate_sample_mesh_data.py")
        sys.exit(1)
    else:
        print(f"Mesh sequence found at: {mesh_actual_path}")
    
    # Verify garment template file exists
    garment_path = config.dataloader.dataset.from_any_pose.garment_template_path
    garment_exists, garment_actual_path = check_file_exists(garment_path, "Garment template")
    if not garment_exists:
        print(f"\n[ERROR] Garment template file not found: {garment_path}")
        print("Checked locations:")
        if not os.path.isabs(garment_path):
            print(f"  - {os.path.join(DEFAULTS.data_root, garment_path)} (data_root)")
            print(f"  - {garment_path} (current directory)")
        print("\nNote: Garment can be either .pkl or .obj format")
        print("\nTo generate sample data, run:")
        print("  python Tests/generate_sample_mesh_data.py")
        sys.exit(1)
    else:
        print(f"Garment template found at: {garment_actual_path}")
        garment_ext = os.path.splitext(garment_actual_path)[1].lower()
        print(f"Garment format: {garment_ext} (supported formats: .pkl, .obj)")
    
    # Create modules
    print("\nCreating training modules...")
    dataloader_m, runner, training_module, aux_modules = create_modules(modules, config)
    
    # Wrap dataset with MeshDatasetWrapper to handle mesh data format
    print("Wrapping dataset with MeshDatasetWrapper for proper temporal offsets...")
    original_dataset = dataloader_m.dataset
    dataloader_m.dataset = MeshDatasetWrapper(original_dataset)
    
    # Replace the runner's training module with MeshRunner
    # This handles the collect_sample method properly for mesh data
    print("Using MeshRunner for proper mesh data handling in training...")
    mesh_training_module = MeshRunner(
        training_module.model,
        training_module.criterion_dict,
        training_module.mcfg
    )
    # Copy over any other necessary attributes
    mesh_training_module.cloth_obj = training_module.cloth_obj
    training_module = mesh_training_module
    
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
    
    # Training loop
    global_step = config.step_start
    
    torch.manual_seed(57)
    np.random.seed(57)
    
    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60)
    
    for i in range(config.experiment.n_epochs):
        print(f"\nEpoch {i + 1}/{config.experiment.n_epochs}")
        print("-" * 60)
        
        dataloader = dataloader_m.create_dataloader()
        global_step = runner.run_epoch(
            training_module, aux_modules, dataloader, i, config,
            global_step=global_step
        )
        
        if config.experiment.max_iter is not None and global_step > config.experiment.max_iter:
            break
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Final global step: {global_step}")
    print(f"Checkpoints saved in: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
