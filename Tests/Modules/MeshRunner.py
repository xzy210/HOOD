#!/usr/bin/env python3
"""
Custom MeshRunner for mesh-based training that works with the existing HOOD framework
without modifying core library files.

This module provides:
1. MaterialConfig: Material parameters configuration
2. OptimConfig: Optimizer configuration
3. Config: Main runner configuration
4. MeshRunner: Custom runner that handles mesh data format in collect_sample
5. create_mesh_runner: Convenience function to create a MeshRunner
6. run_epoch: Custom training loop with loss logging and TensorBoard support
"""

import os
import numpy as np
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, Optional

from torch import nn
from torch.utils.data import DataLoader
from omegaconf import II
from omegaconf.dictconfig import DictConfig
from tqdm import tqdm
from huepy import yellow

from utils.common import move2device, add_field_to_pyg_batch, save_checkpoint
from utils.defaults import DEFAULTS

from runners.from_any_pose import Runner as BaseRunner
from runners.utils.collector import SampleCollector
from runners.utils.collision import CollisionPreprocessor
from runners.utils.material import RandomMaterial
from utils.cloth_and_material import FaceNormals, ClothMatAug


@dataclass
class MaterialConfig:
    """Material parameters configuration for cloth simulation."""
    density_min: float = 0.20022            # minimal density to sample from (used to compute nodal masses)
    density_max: float = 0.20022            # maximal density to sample from (used to compute nodal masses)
    lame_mu_min: float = 23600.0            # minimal shear modulus to sample from
    lame_mu_max: float = 23600.0            # maximal shear modulus to sample from
    lame_lambda_min: float = 44400          # minimal Lame's lambda to sample from
    lame_lambda_max: float = 44400          # maximal Lame's lambda to sample from
    bending_coeff_min: float = 3.96e-05     # minimal bending coefficient to sample from
    bending_coeff_max: float = 3.96e-05     # maximal bending coefficient to sample from
    bending_multiplier: float = 1.          # multiplier for bending coefficient

    density_override: Optional[float] = None        # if set, overrides the sampled density (used in validation)
    lame_mu_override: Optional[float] = None        # if set, overrides the sampled shear modulus (used in validation)
    lame_lambda_override: Optional[float] = None    # if set, overrides the sampled Lame's lambda (used in validation)
    bending_coeff_override: Optional[float] = None  # if set, overrides the sampled bending coefficient (used in validation)


@dataclass
class OptimConfig:
    """Optimizer configuration."""
    lr: float = 1e-4                # initial learning rate
    decay_rate: float = 1e-1        # decay multiplier for the scheduler
    decay_min: float = 0            # minimal decay
    decay_steps: int = 5_000_000    # number of steps for one decay step
    step_start: int = 0             # step to start from (used to resume training)


@dataclass
class Config:
    """Main configuration for MeshRunner."""
    optimizer: OptimConfig = None
    material: MaterialConfig = None
    warmup_steps: int = 100                 # number of steps to warm up the normalization statistics
    increase_roll_every: int = 5000         # we start from predicting only one step, then increase the number of steps each `increase_roll_every` steps
    roll_max: int = 5                       # maximum number of steps to predict
    push_eps: float = 2e-3                  # threshold for collision solver, we apply it once before the first step
    grad_clip: Optional[float] = 1.         # if set, clips the gradient norm to this value
    overwrite_pos_every_step: bool = False  # if true, the canonical poses of each garment are not cached

    # In the paper, the difference between the initial and regular time steps is explained with alpha coefficient in the inertia loss term
    initial_ts: float = 1 / 3   # time between the first two steps in training, used to allow the model to faster reach static equilibrium
    regular_ts: float = 1 / 30  # time between the regular steps in training and validation

    device: str = 'cuda:0'
    
    def __post_init__(self):
        if self.optimizer is None:
            self.optimizer = OptimConfig()
        if self.material is None:
            self.material = MaterialConfig()


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


def create_mesh_runner(model, criterion_dict, mcfg):
    """
    Convenience function to create a MeshRunner.
    
    Args:
        model: The neural network model
        criterion_dict: Dictionary of loss functions
        mcfg: Module config (Config instance or DictConfig)
    
    Returns:
        MeshRunner instance
    """
    return MeshRunner(model, criterion_dict, mcfg)


def run_epoch(training_module: MeshRunner, aux_modules: dict, dataloader: DataLoader,
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
