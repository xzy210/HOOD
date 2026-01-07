#!/usr/bin/env python3
"""
Custom dataset wrapper for mesh-based training that works with the existing HOOD framework
without modifying core library files.

This module provides:
1. DualMeshDatasetWrapper: Dataset wrapper for body + tshirt mesh sequence training
"""

import os
import torch
import numpy as np
from torch_geometric.data import HeteroData

from utils.common import pickle_load, triangles_to_edges
from utils.mesh_creation import obj2template
from utils.coarse import make_coarse_edges


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
