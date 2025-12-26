#!/usr/bin/env python3
"""
Custom utilities for mesh-based training that work with the existing HOOD framework
without modifying core library files.

This module provides:
1. MeshDatasetWrapper: Wraps the dataset to properly format mesh sequence data
2. MeshRunner: Custom runner that handles mesh data format in collect_sample
3. mesh_run_epoch: Custom training loop with loss logging and TensorBoard support
"""

import os
import torch
import numpy as np
from datetime import datetime
from torch import nn
from torch.utils.data import DataLoader
from torch_geometric.data import HeteroData, Batch
from typing import Dict, Optional
from omegaconf.dictconfig import DictConfig
from tqdm import tqdm
from huepy import yellow

# Optional TensorBoard support
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    SummaryWriter = None
    TENSORBOARD_AVAILABLE = False

from runners.from_any_pose import Runner as BaseRunner
from runners.utils.collector import SampleCollector
from runners.utils.collision import CollisionPreprocessor
from runners.utils.material import RandomMaterial
from utils.cloth_and_material import FaceNormals, ClothMatAug
from utils.common import move2device, add_field_to_pyg_batch, save_checkpoint
from utils.defaults import DEFAULTS


class MeshDatasetWrapper:
    """
    Wrapper for the mesh dataset that properly formats the data for training.
    
    The core issue is that BareMeshBodyBuilder stores all frames in pos/prev_pos/target_pos
    as [V, N, 3], but the training pipeline expects either:
    1. Single frame data [V, 3] with lookup field for future frames, or
    2. Properly offset wholeseq data that can be processed by sequence2sample
    
    This wrapper converts the data to format 2, which is compatible with
    the wholeseq validation flow but also works for training when we
    call sequence2sample in collect_sample.
    """
    
    def __init__(self, base_dataset):
        """
        Args:
            base_dataset: The original dataset from from_any_pose.py
        """
        self.base_dataset = base_dataset
        
        # Load a sample to determine sequence length
        # The obstacle.pos is [V, N, 3] where N is number of frames
        sample = base_dataset[0]
        if hasattr(sample['obstacle'], 'pos') and sample['obstacle'].pos.dim() == 3:
            n_frames = sample['obstacle'].pos.shape[1]
            # We need at least 3 frames for temporal offset (prev, current, target)
            # After offset: prev_pos uses frames [0, N-3], pos uses [1, N-2], target_pos uses [2, N-1]
            # So the valid sequence length is N - 2
            self._len = max(1, n_frames - 2)
        else:
            self._len = 1
        
        print(f"[MeshDatasetWrapper] Sequence has {n_frames} frames, {self._len} valid training frames per epoch")
    
    def __len__(self):
        return self._len
    
    def __getitem__(self, item: int) -> HeteroData:
        """
        Load and transform the sample to have proper temporal offsets.
        
        Original format from BareMeshBodyBuilder:
            obstacle.prev_pos = pos = target_pos = [V, N, 3] (all same)
        
        Transformed format (wholeseq style):
            obstacle.prev_pos = all_verts[:, :-2, :]  # frames [0, N-3]
            obstacle.pos = all_verts[:, 1:-1, :]      # frames [1, N-2]
            obstacle.target_pos = all_verts[:, 2:, :] # frames [2, N-1]
        
        This allows sequence2sample to correctly extract single frames with
        proper temporal relationships.
        """
        sample = self.base_dataset[item]
        
        # Transform obstacle (body) data if it's in wholeseq format
        if hasattr(sample['obstacle'], 'pos') and sample['obstacle'].pos.dim() == 3:
            all_verts = sample['obstacle'].pos  # [V, N, 3]
            N = all_verts.shape[1]
            
            if N >= 3:
                # Apply proper temporal offsets
                sample['obstacle'].prev_pos = all_verts[:, :-2, :].clone()
                sample['obstacle'].pos = all_verts[:, 1:-1, :].clone()
                sample['obstacle'].target_pos = all_verts[:, 2:, :].clone()
            else:
                # If sequence is too short, just duplicate
                sample['obstacle'].prev_pos = all_verts.clone()
                sample['obstacle'].pos = all_verts.clone()
                sample['obstacle'].target_pos = all_verts.clone()
        
        return sample


class MeshRunner(BaseRunner):
    """
    Custom runner that properly handles mesh sequence data in training.
    
    The key difference from BaseRunner is that collect_sample:
    1. Calls sequence2sample to extract single frames from [V, N, 3] data
    2. Removes the lookup2target call since we don't have lookup field
    """
    
    def __init__(self, model: nn.Module, criterion_dict: Dict[str, nn.Module], mcfg: DictConfig):
        # Call grandparent's __init__ to avoid BaseRunner's initialization
        nn.Module.__init__(self)
        
        self.model = model
        self.criterion_dict = criterion_dict
        self.mcfg = mcfg

        self.cloth_obj = ClothMatAug(None, always_overwrite_mass=True)
        self.normals_f = FaceNormals()

        self.sample_collector = SampleCollector(mcfg, no_target=True)
        self.collision_solver = CollisionPreprocessor(mcfg)
        self.random_material = RandomMaterial(mcfg.material)
    
    def collect_sample(self, sample, idx, prev_out_dict=None, random_ts=False):
        """
        Collects a sample from the sequence, given the previous output and the index of the current step.
        
        This version is modified to work with mesh sequence data:
        1. First calls sequence2sample to extract single frame from [V, N, 3] format
        2. Does NOT call lookup2target since mesh data doesn't have lookup field
        
        :param sample: pytorch geometric batch from the dataloader
        :param idx: index of the current step
        :param prev_out_dict: previous output of the model
        :param random_ts: if True, the time step is randomly chosen between the initial and the regular time step
        :return: the sample for the current step
        """
        sample_step = sample.clone()
        
        # Extract single frame from sequence data [V, N, 3] -> [V, 3]
        # This is the key difference: we call sequence2sample like wholeseq validation does
        sample_step = self.sample_collector.sequence2sample(sample_step, idx)
        
        # Copy fields from the previous step (pred_pos -> pos, pos->prev_pos)
        sample_step = self.sample_collector.copy_from_prev(sample_step, prev_out_dict)
        ts = self.mcfg.regular_ts

        # NOTE: We removed lookup2target call here because:
        # 1. Mesh data doesn't have lookup field
        # 2. sequence2sample already extracted the correct target_pos from the sequence
        
        # In the first step, the obstacle and positions of the pinned vertices are static
        if idx == 0:
            is_init = np.random.rand() > 0.5
            sample_step = self.sample_collector.pos2target(sample_step)
            if is_init or not random_ts:
                sample_step = self.sample_collector.pos2prev(sample_step)
                ts = self.mcfg.initial_ts
        # For the second frame, we set velocity to zero
        elif idx == 1:
            sample_step = self.sample_collector.pos2prev(sample_step)

        sample_step = self.sample_collector.add_velocity(sample_step, prev_out_dict)
        sample_step = self.sample_collector.add_timestep(sample_step, ts)
        return sample_step


def wrap_dataset(dataset):
    """
    Convenience function to wrap a dataset with MeshDatasetWrapper.
    
    Args:
        dataset: Original dataset from from_any_pose.py
    
    Returns:
        MeshDatasetWrapper instance
    """
    return MeshDatasetWrapper(dataset)


def create_mesh_runner(model, criterion_dict, mcfg):
    """
    Convenience function to create a MeshRunner.
    
    Args:
        model: The neural network model
        criterion_dict: Dictionary of loss functions
        mcfg: Module config
    
    Returns:
        MeshRunner instance
    """
    return MeshRunner(model, criterion_dict, mcfg)


def mesh_run_epoch(training_module: MeshRunner, aux_modules: dict, dataloader: DataLoader,
                   n_epoch: int, cfg: DictConfig, global_step=None, writer=None,
                   log_every: int = 100):
    """
    Custom run_epoch function with loss logging and TensorBoard support.
    
    Args:
        training_module: The MeshRunner training module
        aux_modules: Dictionary containing optimizer and scheduler
        dataloader: Training data loader
        n_epoch: Current epoch number
        cfg: Configuration object
        global_step: Current global step (optional)
        writer: TensorBoard SummaryWriter (optional)
        log_every: Print loss log every N steps (default: 100)
    
    Returns:
        global_step: Updated global step after this epoch
    """
    global_step = global_step or len(dataloader) * n_epoch
    
    optimizer = aux_modules['optimizer']
    scheduler = aux_modules['scheduler']
    
    # Setup checkpoints directory
    if hasattr(cfg, 'run_dir'):
        checkpoints_dir = os.path.join(cfg.run_dir, 'checkpoints')
    else:
        now = datetime.now()
        dt_string = now.strftime("%Y%m%d_%H%M%S")
        cfg.run_dir = os.path.join(DEFAULTS.experiment_root, dt_string)
        checkpoints_dir = os.path.join(cfg.run_dir, 'checkpoints')
    
    print(yellow(f'run_epoch started, checkpoints will be saved in {checkpoints_dir}'))
    
    # Progress bar with loss display
    prbar = tqdm(dataloader, desc=f"Epoch {n_epoch + 1}")
    
    # Accumulate losses for logging
    loss_accumulator = {}
    loss_count = 0
    
    for sample in prbar:
        global_step += 1
        if cfg.experiment.max_iter is not None and global_step > cfg.experiment.max_iter:
            break
        
        sample = move2device(sample, cfg.device)
        
        # Add `iter` field to the sample
        B = sample.num_graphs
        sample = add_field_to_pyg_batch(sample, 'iter', [global_step] * B, 'cloth', reference_key=None)
        
        # Number of autoregressive steps
        roll_steps = 1 + (global_step // training_module.mcfg.increase_roll_every)
        roll_steps = min(roll_steps, training_module.mcfg.roll_max)
        
        # Forward pass and optimization
        optimizer_to_pass = optimizer if global_step >= training_module.mcfg.warmup_steps else None
        scheduler_to_pass = scheduler if global_step >= training_module.mcfg.warmup_steps else None
        ld_to_write = training_module(sample, roll_steps=roll_steps, optimizer=optimizer_to_pass,
                                      scheduler=scheduler_to_pass)
        
        # Accumulate losses
        for key, value in ld_to_write.items():
            if key not in loss_accumulator:
                loss_accumulator[key] = 0.0
            loss_accumulator[key] += value
        loss_count += 1
        
        # Update progress bar with current loss
        if 'total' in ld_to_write:
            prbar.set_postfix({'loss': f"{ld_to_write['total']:.6f}", 'step': global_step})
        
        # Log to TensorBoard and console every log_every steps
        if global_step % log_every == 0:
            # Calculate average losses
            avg_losses = {k: v / loss_count for k, v in loss_accumulator.items()}
            
            # Print to console
            loss_str = " | ".join([f"{k}: {v:.6f}" for k, v in avg_losses.items()])
            print(f"\n[Step {global_step}] {loss_str}")
            
            # Write to TensorBoard
            if writer is not None:
                for key, value in avg_losses.items():
                    writer.add_scalar(f'loss/{key}', value, global_step)
                
                # Also log learning rate
                if scheduler is not None:
                    current_lr = optimizer.param_groups[0]['lr']
                    writer.add_scalar('train/learning_rate', current_lr, global_step)
            
            # Reset accumulator
            loss_accumulator = {}
            loss_count = 0
        
        # Save checkpoint
        if global_step % cfg.experiment.save_checkpoint_every == 0:
            os.makedirs(checkpoints_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoints_dir, f"step_{global_step:010d}.pth")
            save_checkpoint(training_module, aux_modules, cfg, checkpoint_path)
            print(f"\n[Step {global_step}] Checkpoint saved: {checkpoint_path}")
    
    return global_step


def create_tensorboard_writer(log_dir: str, experiment_name: str = None):
    """
    Create a TensorBoard SummaryWriter.
    
    Args:
        log_dir: Base directory for logs
        experiment_name: Optional experiment name (uses timestamp if not provided)
    
    Returns:
        SummaryWriter instance, or None if TensorBoard is not available
    """
    if not TENSORBOARD_AVAILABLE:
        print("[TensorBoard] WARNING: tensorboard is not installed. Install with 'pip install tensorboard'")
        print("[TensorBoard] Training will continue without TensorBoard logging.")
        return None
    
    if experiment_name is None:
        experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    tensorboard_dir = os.path.join(log_dir, 'tensorboard', experiment_name)
    os.makedirs(tensorboard_dir, exist_ok=True)
    
    writer = SummaryWriter(log_dir=tensorboard_dir)
    print(f"[TensorBoard] Logs will be saved to: {tensorboard_dir}")
    print(f"[TensorBoard] Run 'tensorboard --logdir={os.path.dirname(tensorboard_dir)}' to visualize")
    
    return writer
