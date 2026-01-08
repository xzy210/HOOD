"""
Mesh Dataset Module for HOOD training.

This module provides dataset classes for training with pre-computed mesh sequences
(body mesh + cloth mesh PKL files). It mirrors the postcvpr dataset design but works
with direct mesh data instead of SMPL parameters.

Key differences from postcvpr:
- No SMPL-based pose/shape transformations
- No body shape augmentation
- Limited data augmentation (only position noise)
- Data comes from mesh sequence PKL files instead of SMPL sequences
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, List

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

from utils.common import NodeType, triangles_to_edges, pickle_load
from utils.coarse import make_coarse_edges
from utils.defaults import DEFAULTS
from utils.mesh_creation import obj2template, add_coarse_edges


@dataclass
class Config:
    """Configuration for MeshDataset.
    
    Supports two modes:
    1. Single sequence mode: specify body_sequence_path and cloth_sequence_path directly
    2. Multi-sequence mode: specify data_root and split_path (CSV file)
    """
    # Single sequence mode paths (backward compatible)
    body_sequence_path: Optional[str] = None
    cloth_sequence_path: Optional[str] = None
    cloth_template_path: Optional[str] = None
    
    # Multi-sequence mode paths
    data_root: Optional[str] = None  # Root directory for sequence files
    split_path: Optional[str] = None  # Path to CSV datasplit file
    garment_dict_file: Optional[str] = None  # Path to garment templates dict
    
    # Common settings
    n_coarse_levels: int = 4
    lookup_steps: int = 5
    noise_scale: float = 3e-3  # Noise scale for position augmentation
    wholeseq: bool = False  # If True, load whole sequence (for validation)
    pinned_verts: bool = False  # Whether to use pinned vertices


class MeshVertexBuilder:
    """
    Helper class to build mesh vertices from sequence data.
    Similar to postcvpr.VertexBuilder but works with direct mesh data.
    """
    
    def __init__(self, mcfg: Config):
        self.mcfg = mcfg
    
    def pad_lookup(self, lookup: np.ndarray) -> np.ndarray:
        """
        Pad the lookup sequence to the required number of steps.
        
        Args:
            lookup: [N, V, 3] vertex positions for N frames
            
        Returns:
            Padded lookup array with exactly mcfg.lookup_steps frames
        """
        n_lookup = lookup.shape[0]
        n_topad = self.mcfg.lookup_steps - n_lookup
        
        if n_topad <= 0:
            return lookup[:self.mcfg.lookup_steps]
        
        # Pad by repeating the last frame
        padlist = [lookup] + [lookup[-1:]] * n_topad
        lookup = np.concatenate(padlist, axis=0)
        return lookup
    
    def pos2tensor(self, pos: np.ndarray) -> torch.Tensor:
        """
        Convert a numpy array of vertices to a tensor and permute axes.
        
        Args:
            pos: [N, V, 3] vertex positions
            
        Returns:
            Tensor of shape [V, N, 3] or [V, 3] depending on wholeseq mode
        """
        pos = torch.tensor(pos, dtype=torch.float32).permute(1, 0, 2)  # [V, N, 3]
        if not self.mcfg.wholeseq and pos.shape[1] == 1:
            pos = pos[:, 0]  # [V, 3]
        return pos
    
    def build_vertices(self, vertices_seq: np.ndarray, idx: int, N_steps: int) -> Dict[str, torch.Tensor]:
        """
        Build vertex position tensors from sequence data.
        
        Args:
            vertices_seq: [N, V, 3] full vertex sequence
            idx: Starting frame index (not used if wholeseq)
            N_steps: Total number of frames in sequence
            
        Returns:
            Dictionary with prev_pos, pos, target_pos, and optionally lookup tensors
        """
        pos_dict = {}
        
        if self.mcfg.wholeseq:
            # Load whole sequence for validation
            pos_dict['prev_pos'] = vertices_seq[:-2]  # [N-2, V, 3]
            pos_dict['pos'] = vertices_seq[1:-1]  # [N-2, V, 3]
            pos_dict['target_pos'] = vertices_seq[2:]  # [N-2, V, 3]
        else:
            # Load single frame with lookup for training
            n_lookup = 1
            if self.mcfg.lookup_steps > 0:
                n_lookup = min(self.mcfg.lookup_steps, N_steps - idx - 2)
            
            # Extract frames: prev, pos, target, lookup...
            end_idx = idx + 2 + n_lookup
            all_vertices = vertices_seq[idx:end_idx]  # [2+n_lookup, V, 3]
            
            pos_dict['prev_pos'] = all_vertices[:1]  # [1, V, 3]
            pos_dict['pos'] = all_vertices[1:2]  # [1, V, 3]
            pos_dict['target_pos'] = all_vertices[2:3]  # [1, V, 3]
            
            # Build lookup table
            lookup = all_vertices[2:]  # [n_lookup, V, 3]
            lookup = self.pad_lookup(lookup)
            pos_dict['lookup'] = lookup
        
        # Convert all to tensors with correct shape
        result = {}
        for k, v in pos_dict.items():
            result[k] = self.pos2tensor(v)
        
        return result


class MeshNoiseMaker:
    """
    Helper class to add position noise to cloth vertices.
    Only applies noise to NORMAL vertices (vertex_type == 0), not HANDLE vertices.
    """
    
    def __init__(self, mcfg: Config):
        self.mcfg = mcfg
    
    def add_noise(self, sample: HeteroData) -> HeteroData:
        """
        Add gaussian noise to cloth pos and prev_pos.
        
        Args:
            sample: HeteroData with cloth node data
            
        Returns:
            Modified sample with noise added to positions
        """
        if self.mcfg.noise_scale == 0:
            return sample
        
        world_pos = sample['cloth'].pos
        vertex_type = sample['cloth'].vertex_type
        
        if len(vertex_type.shape) == 1:
            vertex_type = vertex_type[..., None]
        
        # Generate noise
        noise = np.random.normal(scale=self.mcfg.noise_scale, size=world_pos.shape).astype(np.float32)
        noise_prev = np.random.normal(scale=self.mcfg.noise_scale, size=world_pos.shape).astype(np.float32)
        
        noise = torch.tensor(noise)
        noise_prev = torch.tensor(noise_prev)
        
        # Create mask for NORMAL vertices only (not HANDLE)
        mask = (vertex_type == NodeType.NORMAL)
        if len(mask.shape) == 2 and len(noise.shape) == 3:
            mask = mask.unsqueeze(-1)
        
        noise = noise * mask
        noise_prev = noise_prev * mask
        
        sample['cloth'].pos = sample['cloth'].pos + noise
        sample['cloth'].prev_pos = sample['cloth'].prev_pos + noise_prev
        
        return sample


class MeshLoader:
    """
    Loader class for building HeteroData samples from mesh sequence files.
    
    Supports loading from:
    - Direct PKL sequence files (body + cloth)
    - Garment dictionary for template data
    """
    
    def __init__(self, mcfg: Config, garments_dict: Optional[Dict] = None, body_faces: Optional[np.ndarray] = None):
        """
        Args:
            mcfg: Configuration
            garments_dict: Dictionary with garment template data (for multi-sequence mode)
            body_faces: Body mesh faces (shared across sequences)
        """
        self.mcfg = mcfg
        self.garments_dict = garments_dict or {}
        self.body_faces = body_faces
        
        self.vertex_builder = MeshVertexBuilder(mcfg)
        self.noise_maker = MeshNoiseMaker(mcfg)
        
        # Cache for loaded sequences (to avoid repeated disk reads)
        self._sequence_cache = {}
    
    def _load_sequence(self, path: str) -> Dict:
        """Load and cache a sequence PKL file."""
        if path not in self._sequence_cache:
            self._sequence_cache[path] = pickle_load(path)
        return self._sequence_cache[path]
    
    def _resolve_path(self, path: str) -> str:
        """Resolve path relative to data_root or aux_data if needed."""
        if os.path.exists(path):
            return path
        
        # Try relative to data_root
        if self.mcfg.data_root:
            full_path = os.path.join(self.mcfg.data_root, path)
            if os.path.exists(full_path):
                return full_path
        
        # Try relative to DEFAULTS.data_root
        full_path = os.path.join(DEFAULTS.data_root, path)
        if os.path.exists(full_path):
            return full_path
        
        # Try relative to aux_data
        full_path = os.path.join(DEFAULTS.aux_data, path)
        if os.path.exists(full_path):
            return full_path
        
        return path
    
    def _get_garment_dict(self, garment_name: str) -> Dict:
        """Get garment template data from garments_dict or load from file."""
        if garment_name in self.garments_dict:
            return self.garments_dict[garment_name]
        
        # For single-sequence mode, load template from config path
        if self.mcfg.cloth_template_path:
            template_path = self._resolve_path(self.mcfg.cloth_template_path)
            if template_path.endswith('.obj'):
                garment_dict = obj2template(template_path)
            else:
                garment_dict = pickle_load(template_path)
            
            # Add coarse edges if needed
            if self.mcfg.n_coarse_levels > 0 and 'center' not in garment_dict:
                garment_dict = add_coarse_edges(garment_dict, self.mcfg.n_coarse_levels)
            
            self.garments_dict[garment_name] = garment_dict
            return garment_dict
        
        raise ValueError(f"Cannot find garment template for '{garment_name}'")
    
    def _build_cloth(self, sample: HeteroData, cloth_vertices: np.ndarray, 
                     idx: int, garment_name: str) -> HeteroData:
        """Build cloth node data."""
        N_steps = cloth_vertices.shape[0]
        garment_dict = self._get_garment_dict(garment_name)
        
        # Build position tensors
        pos_dict = self.vertex_builder.build_vertices(cloth_vertices, idx, N_steps)
        for k, v in pos_dict.items():
            setattr(sample['cloth'], k, v)
        
        # Add rest position from template
        rest_pos = garment_dict.get('rest_pos', garment_dict.get('vertices'))
        sample['cloth'].rest_pos = torch.tensor(rest_pos, dtype=torch.float32)
        
        # Add faces and edges
        faces = torch.tensor(garment_dict['faces'], dtype=torch.long)
        sample['cloth'].faces_batch = faces.T
        edges = triangles_to_edges(faces.unsqueeze(0))
        sample['cloth', 'mesh_edge', 'cloth'].edge_index = edges
        
        # Add vertex type
        n_verts = cloth_vertices.shape[1]
        if self.mcfg.pinned_verts and 'node_type' in garment_dict:
            vertex_type = garment_dict['node_type'].astype(np.int64)
        else:
            vertex_type = np.zeros((n_verts, 1), dtype=np.int64)
        sample['cloth'].vertex_type = torch.tensor(vertex_type)
        
        # Add coarse edges
        sample = self._add_coarse_edges(sample, garment_dict, n_verts)
        
        # Add noise (only for training, not wholeseq validation)
        if not self.mcfg.wholeseq:
            sample = self.noise_maker.add_noise(sample)
        
        return sample
    
    def _add_coarse_edges(self, sample: HeteroData, garment_dict: Dict, n_verts: int) -> HeteroData:
        """Add coarse edges for multi-scale GNN."""
        if self.mcfg.n_coarse_levels == 0:
            sample['cloth'].vertex_level = torch.zeros(n_verts, 1, dtype=torch.long)
            return sample
        
        faces = garment_dict['faces']
        
        # Ensure coarse edges data exists
        if 'center' not in garment_dict:
            garment_dict = add_coarse_edges(garment_dict, self.mcfg.n_coarse_levels)
        
        # Randomly choose center node
        center_nodes = garment_dict['center']
        center = np.random.choice(center_nodes)
        
        # Get or compute coarse edges
        if 'coarse_edges' not in garment_dict:
            garment_dict['coarse_edges'] = {}
        
        if center in garment_dict['coarse_edges']:
            coarse_edges_dict = garment_dict['coarse_edges'][center]
        else:
            coarse_edges_dict = make_coarse_edges(faces, center, n_levels=self.mcfg.n_coarse_levels)
            garment_dict['coarse_edges'][center] = coarse_edges_dict
        
        # Add coarse edges to sample
        vertex_level = np.zeros((n_verts, 1), dtype=np.int64)
        for i in range(self.mcfg.n_coarse_levels):
            key = f'coarse_edge{i}'
            edges_coarse = coarse_edges_dict[i].astype(np.int64)
            # Make bidirectional
            edges_coarse = np.concatenate([edges_coarse, edges_coarse[:, [1, 0]]], axis=0)
            sample['cloth', key, 'cloth'].edge_index = torch.tensor(edges_coarse.T)
            
            # Update vertex level
            nodes_unique = np.unique(edges_coarse.reshape(-1))
            vertex_level[nodes_unique] = i + 1
        
        sample['cloth'].vertex_level = torch.tensor(vertex_level)
        return sample
    
    def _build_obstacle(self, sample: HeteroData, body_vertices: np.ndarray, 
                        idx: int, body_faces: np.ndarray) -> HeteroData:
        """Build obstacle (body) node data."""
        N_steps = body_vertices.shape[0]
        
        # Build position tensors
        pos_dict = self.vertex_builder.build_vertices(body_vertices, idx, N_steps)
        for k, v in pos_dict.items():
            setattr(sample['obstacle'], k, v)
        
        # Add faces
        sample['obstacle'].faces_batch = torch.tensor(body_faces.astype(np.int64)).T
        
        # Add vertex type (all regular: 1)
        n_verts = body_vertices.shape[1]
        sample['obstacle'].vertex_type = torch.ones(n_verts, 1, dtype=torch.long)
        sample['obstacle'].vertex_level = torch.zeros(n_verts, 1, dtype=torch.long)
        
        return sample
    
    def load_sample(self, body_path: str, cloth_path: str, idx: int, 
                    garment_name: str) -> HeteroData:
        """
        Build HeteroData sample from sequence files.
        
        Args:
            body_path: Path to body sequence PKL
            cloth_path: Path to cloth sequence PKL
            idx: Frame index (not used if wholeseq)
            garment_name: Name of garment for template lookup
            
        Returns:
            HeteroData sample with cloth and obstacle data
        """
        # Load sequences
        body_path = self._resolve_path(body_path)
        cloth_path = self._resolve_path(cloth_path)
        
        body_data = self._load_sequence(body_path)
        cloth_data = self._load_sequence(cloth_path)
        
        body_vertices = body_data['vertices']  # [N, V_body, 3]
        cloth_vertices = cloth_data['vertices']  # [N, V_cloth, 3]
        body_faces = body_data['faces']
        
        # Verify frame counts match
        assert body_vertices.shape[0] == cloth_vertices.shape[0], \
            f"Frame count mismatch: body={body_vertices.shape[0]}, cloth={cloth_vertices.shape[0]}"
        
        # Build sample
        sample = HeteroData()
        sample = self._build_cloth(sample, cloth_vertices, idx, garment_name)
        sample = self._build_obstacle(sample, body_vertices, idx, body_faces)
        
        return sample
    
    def load_sample_from_split(self, fname: str, idx: int, garment_name: str) -> HeteroData:
        """
        Load sample using datasplit convention.
        
        Args:
            fname: Sequence identifier (without extension)
            idx: Frame index
            garment_name: Garment name for template lookup
            
        Returns:
            HeteroData sample
        """
        # Construct paths based on datasplit convention
        # Expect: data_root/body_sequence/{fname}.pkl and data_root/{garment}_sequence/{fname}.pkl
        body_path = os.path.join(self.mcfg.data_root, 'body_sequence', f'{fname}.pkl')
        cloth_path = os.path.join(self.mcfg.data_root, f'{garment_name}_sequence', f'{fname}.pkl')
        
        return self.load_sample(body_path, cloth_path, idx, garment_name)


def load_garments_dict_for_mesh(garment_dict_file: str) -> Dict:
    """Load garment templates dictionary."""
    if garment_dict_file is None:
        return {}
    
    garment_dict_path = os.path.join(DEFAULTS.aux_data, garment_dict_file)
    if not os.path.exists(garment_dict_path):
        return {}
    
    return pickle_load(garment_dict_path)


def create_loader(mcfg: Config) -> MeshLoader:
    """Create a MeshLoader instance from config."""
    garments_dict = {}
    
    # Load garment templates if specified
    if mcfg.garment_dict_file:
        garments_dict = load_garments_dict_for_mesh(mcfg.garment_dict_file)
    
    # Resolve data_root
    if mcfg.data_root:
        if not os.path.isabs(mcfg.data_root):
            mcfg.data_root = os.path.join(DEFAULTS.data_root, mcfg.data_root)
    
    return MeshLoader(mcfg, garments_dict)


def create(mcfg: Config):
    """
    Create MeshDataset from config.
    
    Supports two modes:
    1. Single sequence: body_sequence_path + cloth_sequence_path specified
    2. Multi-sequence: data_root + split_path specified
    """
    loader = create_loader(mcfg)
    
    # Determine mode
    if mcfg.split_path is not None:
        # Multi-sequence mode with datasplit CSV
        split_path = os.path.join(DEFAULTS.aux_data, mcfg.split_path)
        if not os.path.exists(split_path):
            split_path = mcfg.split_path
        datasplit = pd.read_csv(split_path, dtype='str')
        return MeshDataset(loader, mcfg, datasplit=datasplit, wholeseq=mcfg.wholeseq)
    
    elif mcfg.body_sequence_path and mcfg.cloth_sequence_path:
        # Single sequence mode
        return MeshDataset(loader, mcfg, 
                          body_path=mcfg.body_sequence_path,
                          cloth_path=mcfg.cloth_sequence_path,
                          wholeseq=mcfg.wholeseq)
    
    else:
        raise ValueError("Must specify either (split_path + data_root) or (body_sequence_path + cloth_sequence_path)")


class MeshDataset:
    """
    Dataset for mesh sequence training.
    
    Supports two modes:
    1. Single sequence mode: One body+cloth sequence pair
    2. Multi-sequence mode: Multiple sequences via datasplit CSV
    
    In training mode (wholeseq=False):
    - Returns single frames with lookup table
    - __len__ returns total available frames across all sequences
    
    In validation mode (wholeseq=True):
    - Returns whole sequences
    - __len__ returns number of sequences
    """
    
    def __init__(self, loader: MeshLoader, mcfg: Config,
                 datasplit: Optional[pd.DataFrame] = None,
                 body_path: Optional[str] = None,
                 cloth_path: Optional[str] = None,
                 wholeseq: bool = False):
        """
        Args:
            loader: MeshLoader instance
            mcfg: Configuration
            datasplit: DataFrame with columns [id, garment, length] for multi-sequence mode
            body_path: Body sequence path for single-sequence mode
            cloth_path: Cloth sequence path for single-sequence mode
            wholeseq: If True, return whole sequences (validation mode)
        """
        self.loader = loader
        self.mcfg = mcfg
        self.wholeseq = wholeseq
        
        # Determine mode and setup
        if datasplit is not None:
            self._setup_multi_sequence(datasplit)
        elif body_path and cloth_path:
            self._setup_single_sequence(body_path, cloth_path)
        else:
            raise ValueError("Must provide either datasplit or (body_path, cloth_path)")
    
    def _setup_single_sequence(self, body_path: str, cloth_path: str):
        """Setup for single sequence mode."""
        self.mode = 'single'
        
        # Resolve paths
        self.body_path = self.loader._resolve_path(body_path)
        self.cloth_path = self.loader._resolve_path(cloth_path)
        
        # Load sequence to get frame count
        body_data = pickle_load(self.body_path)
        self.n_frames = body_data['num_frames']
        
        # Extract garment name from path
        cloth_basename = os.path.basename(cloth_path)
        self.garment_name = cloth_basename.split('_')[0] if '_' in cloth_basename else 'cloth'
        
        # Calculate available frames
        # Need: idx, idx+1 (pos), idx+2 (target), + lookup_steps
        self.available_frames = max(1, self.n_frames - self.mcfg.lookup_steps - 2)
        
        if self.wholeseq:
            self._len = 1
        else:
            self._len = self.available_frames
    
    def _setup_multi_sequence(self, datasplit: pd.DataFrame):
        """Setup for multi-sequence mode with datasplit CSV."""
        self.mode = 'multi'
        self.datasplit = datasplit
        
        if self.wholeseq:
            self._len = len(datasplit)
        else:
            # Calculate cumulative lengths for frame indexing
            all_lens = datasplit['length'].astype(int).tolist()
            # Available frames per sequence: length - lookup_steps - 2
            self.all_lens = [max(1, x - self.mcfg.lookup_steps - 2) for x in all_lens]
            self._len = sum(self.all_lens)
            
            # Build cumulative lengths for fast indexing
            self.cumulative_lens = []
            cumsum = 0
            for l in self.all_lens:
                self.cumulative_lens.append(cumsum)
                cumsum += l
    
    def _find_sequence_and_frame(self, global_idx: int) -> Tuple[int, int]:
        """
        Convert global frame index to (sequence_idx, frame_idx).
        
        Uses binary search for efficiency with many sequences.
        """
        # Linear search (simple, works for reasonable number of sequences)
        seq_idx = 0
        local_idx = global_idx
        
        while seq_idx < len(self.all_lens) and local_idx >= self.all_lens[seq_idx]:
            local_idx -= self.all_lens[seq_idx]
            seq_idx += 1
        
        return seq_idx, local_idx
    
    def __len__(self) -> int:
        return self._len
    
    def __getitem__(self, item: int) -> HeteroData:
        """Get a sample by index."""
        if self.mode == 'single':
            return self._get_single_sequence_item(item)
        else:
            return self._get_multi_sequence_item(item)
    
    def _get_single_sequence_item(self, item: int) -> HeteroData:
        """Get item for single sequence mode."""
        if self.wholeseq:
            idx = 0
        else:
            idx = item
        
        sample = self.loader.load_sample(
            self.body_path, 
            self.cloth_path, 
            idx, 
            self.garment_name
        )
        
        sample['sequence_name'] = os.path.basename(self.body_path)
        sample['garment_name'] = self.garment_name
        
        return sample
    
    def _get_multi_sequence_item(self, item: int) -> HeteroData:
        """Get item for multi-sequence mode."""
        if self.wholeseq:
            seq_idx = item
            frame_idx = 0
        else:
            seq_idx, frame_idx = self._find_sequence_and_frame(item)
        
        # Get sequence info from datasplit
        row = self.datasplit.iloc[seq_idx]
        fname = row['id']
        garment_name = row['garment']
        
        sample = self.loader.load_sample_from_split(fname, frame_idx, garment_name)
        
        sample['sequence_name'] = fname
        sample['garment_name'] = garment_name
        
        return sample


# Legacy alias for backward compatibility
DualMeshDatasetWrapper = MeshDataset
