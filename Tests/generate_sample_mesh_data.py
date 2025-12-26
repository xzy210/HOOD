#!/usr/bin/env python3
"""
Generate sample mesh sequence and garment template for testing mesh training.

This script creates dummy data files that can be used to test the mesh training pipeline.
"""

import os
import sys
import pickle
import numpy as np
from pathlib import Path


def create_sample_mesh_sequence(output_path: str, n_frames: int = 100, n_verts: int = 6890):
    """
    Create a sample mesh sequence file.
    
    Args:
        output_path: Output .pkl file path
        n_frames: Number of frames in the sequence
        n_verts: Number of vertices (SMPL body has 6890 vertices)
    """
    print(f"Creating sample mesh sequence: {output_path}")
    print(f"  - Frames: {n_frames}")
    print(f"  - Vertices: {n_verts}")
    
    # Generate random vertex positions
    # Create a simple animation: oscillating sphere-like shape
    verts = np.random.randn(n_frames, n_verts, 3).astype(np.float32)
    
    # Normalize vertices
    verts = verts / np.linalg.norm(verts, axis=2, keepdims=True)
    
    # Add simple animation: scale oscillation
    t = np.linspace(0, 2 * np.pi, n_frames)
    scale = 1.0 + 0.1 * np.sin(t)
    verts = verts * scale[:, np.newaxis, np.newaxis]
    
    # Generate sphere topology (triangular mesh)
    # Simple icosphere-like face indices
    n_faces = n_verts * 2
    faces = []
    for i in range(n_faces):
        # Generate triangle faces (this is a simplified example)
        v0 = i % n_verts
        v1 = (i + 1) % n_verts
        v2 = (i + 2) % n_verts
        faces.append([v0, v1, v2])
    
    faces = np.array(faces[:min(n_faces, 13776)], dtype=np.int64)  # SMPL has 13776 faces
    print(f"  - Faces: {faces.shape[0]}")
    
    # Save to file
    data = {
        'verts': verts,
        'faces': faces
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"  ✓ Saved to: {output_path}")


def create_sample_garment_template(output_path: str, n_verts: int = 2000):
    """
    Create a sample garment template file.
    
    Args:
        output_path: Output .pkl file path
        n_verts: Number of vertices in the garment mesh
    """
    print(f"\nCreating sample garment template: {output_path}")
    print(f"  - Vertices: {n_verts}")
    
    # Generate garment mesh (e.g., a t-shirt)
    # Create vertices in a t-shirt like shape
    verts = np.random.randn(n_verts, 3).astype(np.float32)
    
    # Shape into t-shirt: widen the shoulders
    y = verts[:, 1]
    shoulder_factor = np.exp(-0.5 * (y - 0.3)**2 / 0.05)
    verts[:, 0] *= (1 + 0.5 * shoulder_factor)  # Widen x at shoulders
    
    # Generate triangular faces for garment
    n_faces = int(n_verts * 1.5)
    faces = []
    for i in range(n_faces):
        v0 = i % n_verts
        v1 = (i + 1) % n_verts
        v2 = (i + 2) % n_verts
        faces.append([v0, v1, v2])
    
    faces = np.array(faces[:n_faces], dtype=np.int64)
    print(f"  - Faces: {faces.shape[0]}")
    
    # Save to file
    data = {
        'verts': verts,
        'faces': faces
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"  ✓ Saved to: {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate sample mesh data for testing'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='Tests',
        help='Output directory (default: Tests)'
    )
    parser.add_argument(
        '--n_frames',
        type=int,
        default=100,
        help='Number of frames in mesh sequence (default: 100)'
    )
    parser.add_argument(
        '--n_verts',
        type=int,
        default=6890,
        help='Number of vertices in mesh sequence (default: 6890, same as SMPL)'
    )
    parser.add_argument(
        '--garment_verts',
        type=int,
        default=2000,
        help='Number of vertices in garment (default: 2000)'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Sample Mesh Data Generator")
    print("=" * 60)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create sample mesh sequence
    mesh_sequence_path = output_dir / 'mesh_sequence.pkl'
    create_sample_mesh_sequence(
        str(mesh_sequence_path),
        n_frames=args.n_frames,
        n_verts=args.n_verts
    )
    
    # Create sample garment template
    garment_path = output_dir / 'garment_template.pkl'
    create_sample_garment_template(
        str(garment_path),
        n_verts=args.garment_verts
    )
    
    print("\n" + "=" * 60)
    print("Sample data generation completed!")
    print("=" * 60)
    print(f"\nYou can now run training with:")
    print(f"  python Tests/train_mesh.py")
    print(f"\nOr specify your own data:")
    print(f"  python Tests/train_mesh.py \\")
    print(f"      --mesh_sequence your_mesh_sequence.pkl \\")
    print(f"      --garment your_garment_template.pkl")


if __name__ == '__main__':
    main()
