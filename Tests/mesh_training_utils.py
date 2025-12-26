#!/usr/bin/env python3
"""
Custom utilities for mesh-based training that work with the existing HOOD framework
without modifying core library files.

This module provides:
1. MeshDatasetWrapper: Wraps the dataset to properly format mesh sequence data
2. MeshRunner: Custom runner that handles mesh data format in collect_sample
"""

import torch
import numpy as np
from torch import nn
from torch_geometric.data import HeteroData, Batch
from typing import Dict, Optional
from omegaconf.dictconfig import DictConfig

from runners.from_any_pose import Runner as BaseRunner
from runners.utils.collector import SampleCollector
from runners.utils.collision import CollisionPreprocessor
from runners.utils.material import RandomMaterial
from utils.cloth_and_material import FaceNormals, ClothMatAug
from utils.common import move2device, add_field_to_pyg_batch


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
        self._len = len(base_dataset)
    
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
