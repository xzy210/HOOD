#!/usr/bin/env python3
"""
Convert SMPL format animation sequence to Mesh format for HOOD training/inference.

Input format: SMPL parameters .pkl file
    {
        'body_pose': np.ndarray [N, 69],      # Body pose parameters
        'global_orient': np.ndarray [N, 3],   # Global orientation
        'transl': np.ndarray [N, 3],          # Translation
        'betas': np.ndarray [10]              # Shape parameters
    }

Output format: Mesh sequence .pkl file
    {
        'verts': np.ndarray [N, V, 3],       # Vertex positions for each frame
        'faces': np.ndarray [F, 3]           # Face indices (triangle mesh)
    }
"""

import os
import sys
import argparse
import pickle
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import smplx


def load_smpl_sequence(pkl_path: str) -> Dict:
    """
    Load SMPL parameter sequence from .pkl file.
    
    Args:
        pkl_path: Path to the SMPL .pkl file
        
    Returns:
        Dictionary containing SMPL parameters
    """
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    # Validate required fields
    required_keys = {'body_pose', 'global_orient', 'transl', 'betas'}
    missing_keys = required_keys - set(data.keys())
    if missing_keys:
        raise ValueError(f"Missing required SMPL parameters: {missing_keys}")
    
    return data


def smpl_to_mesh(
    body_pose: np.ndarray,
    global_orient: np.ndarray,
    transl: np.ndarray,
    betas: np.ndarray,
    smpl_model: smplx.SMPL
) -> tuple:
    """
    Convert SMPL parameters to mesh vertices.
    
    Args:
        body_pose: Body pose parameters [N, 69]
        global_orient: Global orientation [N, 3]
        transl: Translation [N, 3]
        betas: Shape parameters [10]
        smpl_model: SMPL model instance
        
    Returns:
        vertices: Mesh vertices [N, V, 3]
        faces: Face indices [F, 3]
    """
    N = body_pose.shape[0]
    
    # Ensure betas has the correct shape [N, 10]
    if len(betas.shape) == 1:
        betas = np.tile(betas, (N, 1))
    
    # Convert to tensors
    input_dict = {
        'body_pose': torch.FloatTensor(body_pose),
        'global_orient': torch.FloatTensor(global_orient),
        'transl': torch.FloatTensor(transl),
        'betas': torch.FloatTensor(betas)
    }
    
    print(f"Converting SMPL parameters to mesh ({N} frames)...")
    
    with torch.no_grad():
        smpl_output = smpl_model(**input_dict)
    
    # Extract vertices [N, V, 3]
    vertices = smpl_output.vertices.numpy().astype(np.float32)
    
    # Extract faces [F, 3]
    faces = smpl_model.faces.astype(np.int64)
    
    print(f"Generated mesh: {vertices.shape[0]} frames, {vertices.shape[1]} vertices, {faces.shape[0]} faces")
    
    return vertices, faces


def save_mesh_sequence(output_path: str, vertices: np.ndarray, faces: np.ndarray):
    """
    Save mesh sequence to .pkl file.
    
    Args:
        output_path: Output .pkl file path
        vertices: Vertex positions [N, V, 3]
        faces: Face indices [F, 3]
    """
    mesh_data = {
        'verts': vertices,
        'faces': faces
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(mesh_data, f)
    
    print(f"Saved mesh sequence to: {output_path}")


def convert_smpl_to_mesh(
    input_path: str,
    output_path: str,
    smpl_model_path: str,
    gender: str = 'neutral'
):
    """
    Convert SMPL sequence file to Mesh sequence file.
    
    Args:
        input_path: Input SMPL .pkl file path
        output_path: Output Mesh .pkl file path
        smpl_model_path: Path to SMPL model directory
        gender: SMPL model gender ('male', 'female', 'neutral')
    """
    print("=" * 60)
    print("SMPL to Mesh Converter")
    print("=" * 60)
    
    # Check input file
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    print(f"\n[1/4] Loading SMPL parameters from: {input_path}")
    smpl_data = load_smpl_sequence(input_path)
    
    # Print SMPL data info
    print(f"  - body_pose: {smpl_data['body_pose'].shape}")
    print(f"  - global_orient: {smpl_data['global_orient'].shape}")
    print(f"  - transl: {smpl_data['transl'].shape}")
    print(f"  - betas: {smpl_data['betas'].shape}")
    
    # Determine SMPL model path
    smpl_model_dir = Path(smpl_model_path)
    if not smpl_model_dir.exists():
        # Try to find in common locations
        possible_locations = [
            Path("aux_data/smpl"),
            Path("hood_data/aux_data/smpl"),
            Path("../aux_data/smpl"),
        ]
        for loc in possible_locations:
            if loc.exists():
                smpl_model_dir = loc
                break
    
    if not smpl_model_dir.exists():
        raise FileNotFoundError(f"SMPL model directory not found: {smpl_model_path}")
    
    # Map gender to model filename
    gender_map = {
        'male': 'SMPL_MALE.pkl',
        'female': 'SMPL_FEMALE.pkl',
        'neutral': 'SMPL_NEUTRAL.pkl'
    }
    
    model_file = gender_map.get(gender.lower(), 'SMPL_NEUTRAL.pkl')
    smpl_model_file = smpl_model_dir / model_file
    
    if not smpl_model_file.exists():
        raise FileNotFoundError(f"SMPL model file not found: {smpl_model_file}")
    
    print(f"\n[2/4] Loading SMPL model: {model_file}")
    smpl_model = smplx.SMPL(model_path=str(smpl_model_dir), gender=gender)
    
    print(f"\n[3/4] Converting SMPL parameters to mesh...")
    vertices, faces = smpl_to_mesh(
        smpl_data['body_pose'],
        smpl_data['global_orient'],
        smpl_data['transl'],
        smpl_data['betas'],
        smpl_model
    )
    
    print(f"\n[4/4] Saving mesh sequence to: {output_path}")
    
    # Create output directory if needed
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    save_mesh_sequence(output_path, vertices, faces)
    
    print("\n" + "=" * 60)
    print("Conversion completed successfully!")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Convert SMPL animation sequence to Mesh format for HOOD'
    )
    
    parser.add_argument(
        'input',
        help='Input SMPL .pkl file path'
    )
    
    parser.add_argument(
        'output',
        help='Output Mesh .pkl file path'
    )
    
    parser.add_argument(
        '--smpl_model',
        default='aux_data/smpl',
        help='Path to SMPL model directory (default: aux_data/smpl)'
    )
    
    parser.add_argument(
        '--gender',
        choices=['male', 'female', 'neutral'],
        default='neutral',
        help='SMPL model gender (default: neutral)'
    )
    
    args = parser.parse_args()
    
    try:
        convert_smpl_to_mesh(
            input_path=args.input,
            output_path=args.output,
            smpl_model_path=args.smpl_model,
            gender=args.gender
        )
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
