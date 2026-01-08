"""
Mesh Runner Module for HOOD training.

This runner handles training with pre-computed mesh sequences (body + cloth PKL files).
It inherits from postcvpr.Runner to reuse the core training logic, but uses MeshDataset
for data loading.

Key differences from postcvpr:
- Uses MeshDataset instead of SMPL-based dataset
- Data comes from mesh sequence PKL files
- Limited data augmentation (only position noise)
"""

import os
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from omegaconf.dictconfig import DictConfig
from omegaconf import II
from tqdm import tqdm
from huepy import yellow

from utils.common import move2device, add_field_to_pyg_batch, save_checkpoint
from utils.defaults import DEFAULTS

from runners.postcvpr import Runner as PostcvprRunner
from runners.utils.collector import SampleCollector
from runners.utils.collision import CollisionPreprocessor
from runners.utils.material import RandomMaterial
from utils.cloth_and_material import FaceNormals, ClothMatAug


@dataclass
class MaterialConfig:
    """Material parameters configuration for cloth simulation."""
    density_min: float = 0.20022
    density_max: float = 0.20022
    lame_mu_min: float = 23600.0
    lame_mu_max: float = 23600.0
    lame_lambda_min: float = 44400
    lame_lambda_max: float = 44400
    bending_coeff_min: float = 3.96e-05
    bending_coeff_max: float = 3.96e-05
    bending_multiplier: float = 1.

    density_override: Optional[float] = None
    lame_mu_override: Optional[float] = None
    lame_lambda_override: Optional[float] = None
    bending_coeff_override: Optional[float] = None


@dataclass
class OptimConfig:
    """Optimizer configuration."""
    lr: float = 1e-4
    decay_rate: float = 1e-1
    decay_min: float = 0
    decay_steps: int = 5_000_000
    step_start: int = 0


@dataclass
class Config:
    """Main configuration for MeshRunner."""
    optimizer: OptimConfig = OptimConfig()
    material: MaterialConfig = MaterialConfig()
    warmup_steps: int = 100
    increase_roll_every: int = 5000
    roll_max: int = 5
    push_eps: float = 2e-3
    grad_clip: Optional[float] = 1.
    overwrite_pos_every_step: bool = False

    initial_ts: float = 1 / 3
    regular_ts: float = 1 / 30

    device: str = II('device')


class Runner(PostcvprRunner):
    """
    Runner for mesh sequence training.
    
    Inherits from postcvpr.Runner to reuse:
    - forward() training loop
    - valid_rollout() validation
    - collect_sample() with lookup2target
    - criterion_pass()
    - optimizer_step()
    
    The key insight is that MeshDataset now produces samples with the same
    structure as postcvpr dataset (single frame + lookup table), so we can
    reuse all the postcvpr training logic.
    """
    
    def __init__(self, model: nn.Module, criterion_dict: Dict[str, nn.Module], mcfg: DictConfig):
        """
        Initialize MeshRunner.
        
        Uses SampleCollector with no_target=False (same as postcvpr) because
        MeshDataset provides lookup table for target extraction.
        """
        # Call nn.Module.__init__ directly
        nn.Module.__init__(self)
        
        self.model = model
        self.criterion_dict = criterion_dict
        self.mcfg = mcfg

        self.cloth_obj = ClothMatAug(None, always_overwrite_mass=True)
        self.normals_f = FaceNormals()

        # Use standard SampleCollector (no_target=False) since we have lookup table
        self.sample_collector = SampleCollector(mcfg, no_target=False)
        self.collision_solver = CollisionPreprocessor(mcfg)
        self.random_material = RandomMaterial(mcfg.material)
    
    # Inherit all other methods from PostcvprRunner:
    # - valid_rollout()
    # - _rollout()
    # - collect_sample_wholeseq()
    # - set_random_material()
    # - add_cloth_obj()
    # - criterion_pass()
    # - collect_sample()  <- uses lookup2target
    # - optimizer_step()
    # - forward()


def create_optimizer(training_module: Runner, mcfg: DictConfig):
    """Create optimizer and scheduler for training."""
    optimizer = Adam(training_module.parameters(), lr=mcfg.lr)

    def sched_fun(step):
        decay = mcfg.decay_rate ** (step // mcfg.decay_steps) + 1e-2
        decay = max(decay, mcfg.decay_min)
        return decay

    scheduler = LambdaLR(optimizer, sched_fun)
    scheduler.last_epoch = mcfg.step_start

    return optimizer, scheduler


def run_epoch(training_module: Runner, aux_modules: dict, dataloader: DataLoader,
              n_epoch: int, cfg: DictConfig, global_step=None, writer=None,
              log_every: int = 100):
    """
    Run one training epoch with logging support.
    
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
    
    # Progress bar
    if cfg.experiment.max_iter is not None:
        num_iter = min(len(dataloader), cfg.experiment.max_iter - global_step)
    else:
        num_iter = len(dataloader)
    prbar = tqdm(dataloader, desc=f"Epoch {n_epoch + 1}", total=num_iter)
    
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
        
        # Update progress bar
        if 'total' in ld_to_write:
            prbar.set_postfix({'loss': f"{ld_to_write['total']:.6f}", 'step': global_step})
        
        # Log every N steps
        if global_step % log_every == 0 and loss_count > 0:
            avg_losses = {k: v / loss_count for k, v in loss_accumulator.items()}
            
            # Console output
            loss_str = " | ".join([f"{k}: {v:.6f}" for k, v in avg_losses.items()])
            print(f"\n[Step {global_step}] {loss_str}")
            
            # TensorBoard
            if writer is not None:
                for key, value in avg_losses.items():
                    writer.add_scalar(f'loss/{key}', value, global_step)
                if scheduler is not None:
                    current_lr = optimizer.param_groups[0]['lr']
                    writer.add_scalar('train/learning_rate', current_lr, global_step)
            
            loss_accumulator = {}
            loss_count = 0
        
        # Save checkpoint
        if global_step % cfg.experiment.save_checkpoint_every == 0:
            os.makedirs(checkpoints_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoints_dir, f"step_{global_step:010d}.pth")
            save_checkpoint(training_module, aux_modules, cfg, checkpoint_path)
            print(f"\n[Step {global_step}] Checkpoint saved: {checkpoint_path}")
    
    return global_step
