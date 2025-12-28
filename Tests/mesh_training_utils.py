#!/usr/bin/env python3
"""
Custom utilities for mesh-based training that work with the existing HOOD framework
without modifying core library files.

This module provides:
1. MeshDatasetWrapper: Wraps the dataset to properly format mesh sequence data
2. DualMeshDatasetWrapper: Dataset wrapper for body + tshirt mesh sequence training
3. MeshRunner: Custom runner that handles mesh data format in collect_sample
4. mesh_run_epoch: Custom training loop with loss logging and TensorBoard support
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
from utils.cloth_and_material import FaceNormals, ClothMatAug, load_obj
from utils.common import move2device, add_field_to_pyg_batch, save_checkpoint, pickle_load, triangles_to_edges
from utils.mesh_creation import obj2template
from utils.coarse import make_coarse_edges
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


class DualMeshDatasetWrapper:
    """
    Dataset wrapper for training with body mesh sequence + tshirt mesh sequence.
    
    This wrapper:
    1. Randomly selects starting frames from the sequence (like train.py)
    2. Initializes cloth position from tshirt animation sequence instead of static mesh
    3. Supports both body and tshirt mesh pkl files
    
    Expected pkl format:
    {
        'vertices': np.ndarray of shape (num_frames, num_vertices, 3),
        'faces': np.ndarray of shape (num_faces, 3),
        'num_frames': int,
        'num_vertices': int
    }
    """
    
    def __init__(self, body_sequence_path: str, tshirt_sequence_path: str, 
                 garment_template_path: str, n_coarse_levels: int = 4,
                 lookup_steps: int = 5):
        """
        Args:
            body_sequence_path: Path to body mesh sequence pkl file
            tshirt_sequence_path: Path to tshirt mesh sequence pkl file
            garment_template_path: Path to static tshirt obj file (for rest_pos and topology)
            n_coarse_levels: Number of coarse levels for mesh hierarchy
            lookup_steps: Number of frames to look ahead (for compatibility with training)
        """
        self.body_sequence_path = body_sequence_path
        self.tshirt_sequence_path = tshirt_sequence_path
        self.garment_template_path = garment_template_path
        self.n_coarse_levels = n_coarse_levels
        self.lookup_steps = lookup_steps
        
        # Load body sequence
        print(f"[DualMeshDatasetWrapper] Loading body sequence: {body_sequence_path}")
        body_data = pickle_load(body_sequence_path)
        self.body_vertices = body_data['vertices']  # (N, V_body, 3)
        self.body_faces = body_data['faces']  # (F_body, 3)
        self.n_body_frames = body_data['num_frames']
        
        # Load tshirt sequence
        print(f"[DualMeshDatasetWrapper] Loading tshirt sequence: {tshirt_sequence_path}")
        tshirt_data = pickle_load(tshirt_sequence_path)
        self.tshirt_vertices = tshirt_data['vertices']  # (N, V_tshirt, 3)
        self.tshirt_faces = tshirt_data['faces']  # (F_tshirt, 3)
        self.n_tshirt_frames = tshirt_data['num_frames']
        
        # Load garment template for rest_pos and coarse edges
        print(f"[DualMeshDatasetWrapper] Loading garment template: {garment_template_path}")
        if garment_template_path.endswith('.obj'):
            self.garment_dict = obj2template(garment_template_path)
        else:
            self.garment_dict = pickle_load(garment_template_path)
        
        # Verify frame counts match
        assert self.n_body_frames == self.n_tshirt_frames, \
            f"Body frames ({self.n_body_frames}) must match tshirt frames ({self.n_tshirt_frames})"
        
        self.n_frames = self.n_body_frames
        
        # Calculate available frames for random selection
        # Similar to postcvpr.py: _lens = N - 7 (for lookup_steps + 2 temporal offset)
        self.available_frames = max(1, self.n_frames - (lookup_steps + 2))
        
        print(f"[DualMeshDatasetWrapper] Total frames: {self.n_frames}, "
              f"Available training frames: {self.available_frames}")
    
    def __len__(self):
        return self.available_frames
    
    def _build_sample(self, start_frame: int) -> HeteroData:
        """
        Build a training sample starting from the given frame index.
        
        Args:
            start_frame: Starting frame index in the sequence
            
        Returns:
            HeteroData sample with:
            - obstacle (body): vertices from body sequence starting at start_frame
            - cloth (tshirt): vertices from tshirt sequence starting at start_frame,
                              with rest_pos from static garment template
        """
        sample = HeteroData()
        
        # Calculate frame range
        # We need: prev_pos, pos, target_pos + lookup_steps frames
        end_frame = min(start_frame + self.lookup_steps + 3, self.n_frames)
        n_sample_frames = end_frame - start_frame
        
        # ============ Build obstacle (body) data ============
        # Extract body vertices for this frame range [V, N, 3]
        body_verts = torch.FloatTensor(
            self.body_vertices[start_frame:end_frame]  # (N_frames, V, 3)
        ).permute(1, 0, 2)  # -> (V, N_frames, 3)
        
        if n_sample_frames >= 3:
            # Apply temporal offsets like train.py
            sample['obstacle'].prev_pos = body_verts[:, :-2, :].clone()
            sample['obstacle'].pos = body_verts[:, 1:-1, :].clone()
            sample['obstacle'].target_pos = body_verts[:, 2:, :].clone()
        else:
            sample['obstacle'].prev_pos = body_verts.clone()
            sample['obstacle'].pos = body_verts.clone()
            sample['obstacle'].target_pos = body_verts.clone()
        
        # Add body faces
        sample['obstacle'].faces_batch = torch.LongTensor(self.body_faces.astype(np.int64)).T
        
        # Add body vertex type (all regular: 1)
        n_body_verts = body_verts.shape[0]
        sample['obstacle'].vertex_type = torch.ones(n_body_verts, 1).long()
        sample['obstacle'].vertex_level = torch.zeros(n_body_verts, 1).long()
        
        # ============ Build cloth (tshirt) data ============
        # Extract tshirt vertices for this frame range [V, N, 3]
        # KEY DIFFERENCE: cloth position initialized from tshirt animation sequence!
        tshirt_verts = torch.FloatTensor(
            self.tshirt_vertices[start_frame:end_frame]  # (N_frames, V, 3)
        ).permute(1, 0, 2)  # -> (V, N_frames, 3)
        
        if n_sample_frames >= 3:
            # Apply temporal offsets
            sample['cloth'].prev_pos = tshirt_verts[:, :-2, :].clone()
            sample['cloth'].pos = tshirt_verts[:, 1:-1, :].clone()
            sample['cloth'].target_pos = tshirt_verts[:, 2:, :].clone()
        else:
            sample['cloth'].prev_pos = tshirt_verts.clone()
            sample['cloth'].pos = tshirt_verts.clone()
            sample['cloth'].target_pos = tshirt_verts.clone()
        
        # Use static garment template for rest_pos (canonical pose)
        sample['cloth'].rest_pos = torch.FloatTensor(self.garment_dict['vertices'])
        
        # Add cloth faces and edges
        cloth_faces = torch.LongTensor(self.garment_dict['faces'])
        sample['cloth'].faces_batch = cloth_faces.T
        edges = triangles_to_edges(cloth_faces.unsqueeze(0))
        sample['cloth', 'mesh_edge', 'cloth'].edge_index = edges
        
        # Add cloth vertex type (all normal: 0)
        n_cloth_verts = tshirt_verts.shape[0]
        sample['cloth'].vertex_type = torch.zeros(n_cloth_verts, 1).long()
        
        # ============ Add coarse edges for cloth ============
        if self.n_coarse_levels > 0:
            faces_np = self.garment_dict['faces']
            
            # Get or compute center nodes
            if 'center' not in self.garment_dict:
                from utils.mesh_creation import add_coarse_edges
                self.garment_dict = add_coarse_edges(self.garment_dict, self.n_coarse_levels)
            
            center_nodes = self.garment_dict['center']
            center = np.random.choice(center_nodes)
            
            # Get or compute coarse edges
            if 'coarse_edges' not in self.garment_dict:
                self.garment_dict['coarse_edges'] = {}
            
            if center in self.garment_dict['coarse_edges']:
                coarse_edges_dict = self.garment_dict['coarse_edges'][center]
            else:
                coarse_edges_dict = make_coarse_edges(faces_np, center, n_levels=self.n_coarse_levels)
                self.garment_dict['coarse_edges'][center] = coarse_edges_dict
            
            # Add coarse edges to sample
            vertex_level = np.zeros((n_cloth_verts, 1)).astype(np.int64)
            for i in range(self.n_coarse_levels):
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
        sample['sequence_name'] = os.path.basename(self.body_sequence_path)
        sample['garment_name'] = os.path.basename(self.garment_template_path)
        sample['start_frame'] = start_frame
        
        return sample
    
    def __getitem__(self, item: int) -> HeteroData:
        """
        Get a training sample. The item index is used to select a random starting frame.
        
        This mimics train.py's behavior of randomly sampling frames from the sequence.
        The DataLoader will shuffle indices, so each epoch sees different frame orderings.
        """
        # item directly maps to frame index (DataLoader handles shuffling)
        start_frame = item
        return self._build_sample(start_frame)


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
