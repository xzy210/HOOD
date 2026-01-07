#!/usr/bin/env python3
"""
Generate sample mesh sequence and garment template for testing mesh training.

This script creates:
1. A square cloth (garment) mesh - 100x100 grid
2. A static sphere mesh animation sequence - 300 frames
3. Initial cloth position is 0.5m above the sphere
"""

import os
import sys
import pickle
import numpy as np
from pathlib import Path


def create_sphere_mesh(radius: float = 0.3, n_subdivisions: int = 16):
    """
    Create a UV sphere mesh.
    
    Args:
        radius: Sphere radius
        n_subdivisions: Number of subdivisions for latitude and longitude
    
    Returns:
        verts: (N, 3) vertex positions
        faces: (F, 3) face indices
    """
    n_lat = n_subdivisions  # latitude divisions
    n_lon = n_subdivisions * 2  # longitude divisions
    
    verts = []
    
    # Add top pole
    verts.append([0, radius, 0])
    
    # Add middle vertices
    for i in range(1, n_lat):
        lat = np.pi * i / n_lat
        y = radius * np.cos(lat)
        r = radius * np.sin(lat)
        
        for j in range(n_lon):
            lon = 2 * np.pi * j / n_lon
            x = r * np.cos(lon)
            z = r * np.sin(lon)
            verts.append([x, y, z])
    
    # Add bottom pole
    verts.append([0, -radius, 0])
    
    verts = np.array(verts, dtype=np.float32)
    
    # Generate faces
    faces = []
    
    # Top cap faces
    # NOTE: Face winding order must be counter-clockwise (CCW) when viewed from outside (+Y direction)
    # to ensure face normals point outward (toward +Y)
    for j in range(n_lon):
        next_j = (j + 1) % n_lon
        # Changed from [0, 1+j, 1+next_j] to [0, 1+next_j, 1+j] for correct outward normal
        faces.append([0, 1 + next_j, 1 + j])
    
    # Middle faces
    # NOTE: Face winding order must be counter-clockwise (CCW) when viewed from outside
    # to ensure face normals point outward (away from sphere center)
    for i in range(n_lat - 2):
        for j in range(n_lon):
            next_j = (j + 1) % n_lon
            v0 = 1 + i * n_lon + j
            v1 = 1 + i * n_lon + next_j
            v2 = 1 + (i + 1) * n_lon + j
            v3 = 1 + (i + 1) * n_lon + next_j
            
            # Changed from [v0, v2, v1] to [v0, v1, v2] for CCW winding
            faces.append([v0, v1, v2])
            faces.append([v1, v3, v2])
    
    # Bottom cap faces
    # NOTE: Face winding order must be counter-clockwise (CCW) when viewed from outside (-Y direction)
    # to ensure face normals point outward (toward -Y)
    bottom_pole = len(verts) - 1
    last_ring_start = 1 + (n_lat - 2) * n_lon
    for j in range(n_lon):
        next_j = (j + 1) % n_lon
        # Changed from [pole, j+1, j] to [pole, j, j+1] for CCW winding when viewed from -Y
        faces.append([bottom_pole, last_ring_start + j, last_ring_start + next_j])
    
    faces = np.array(faces, dtype=np.int64)
    
    return verts, faces


def create_sample_mesh_sequence(output_path: str, n_frames: int = 300, sphere_radius: float = 0.3):
    """
    Create a sample mesh sequence file with a static sphere.
    
    Args:
        output_path: Output .pkl file path
        n_frames: Number of frames in the sequence
        sphere_radius: Radius of the sphere
    """
    print(f"Creating sample mesh sequence (static sphere): {output_path}")
    print(f"  - Frames: {n_frames}")
    print(f"  - Sphere radius: {sphere_radius}")
    
    # Create sphere mesh
    verts, faces = create_sphere_mesh(radius=sphere_radius, n_subdivisions=16)
    
    print(f"  - Vertices: {verts.shape[0]}")
    print(f"  - Faces: {faces.shape[0]}")
    
    # Create static animation (same mesh for all frames)
    # Shape: (n_frames, n_verts, 3)
    verts_sequence = np.tile(verts[np.newaxis, :, :], (n_frames, 1, 1))
    
    # Save to file
    data = {
        'verts': verts_sequence.astype(np.float32),
        'faces': faces
    }
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"  ✓ Saved to: {output_path}")
    return sphere_radius


def create_square_cloth_obj(output_path: str, grid_size: int = 100, cloth_size: float = 1.0, 
                            height_above_sphere: float = 0.5, sphere_radius: float = 0.3):
    """
    Create a square cloth mesh as OBJ file.
    
    Args:
        output_path: Output .obj file path
        grid_size: Number of vertices per side (100 = 100x100 grid)
        cloth_size: Physical size of the cloth (in meters)
        height_above_sphere: Height of cloth above the sphere top
        sphere_radius: Radius of the sphere (to calculate cloth position)
    """
    print(f"\nCreating square cloth garment (OBJ): {output_path}")
    print(f"  - Grid size: {grid_size}x{grid_size}")
    print(f"  - Cloth size: {cloth_size}m x {cloth_size}m")
    
    n_verts = grid_size * grid_size
    print(f"  - Total vertices: {n_verts}")
    
    # Calculate cloth center height (above sphere)
    # Sphere top is at y = sphere_radius, cloth should be height_above_sphere above that
    cloth_y = sphere_radius + height_above_sphere
    print(f"  - Cloth height (Y): {cloth_y}m (sphere top: {sphere_radius}m + offset: {height_above_sphere}m)")
    
    # Generate vertices in a grid
    # Centered at origin (x, z), at height cloth_y
    verts = []
    half_size = cloth_size / 2
    
    for i in range(grid_size):
        for j in range(grid_size):
            x = -half_size + (i / (grid_size - 1)) * cloth_size
            z = -half_size + (j / (grid_size - 1)) * cloth_size
            y = cloth_y
            verts.append([x, y, z])
    
    verts = np.array(verts, dtype=np.float32)
    
    # Generate triangular faces for the grid
    faces = []
    for i in range(grid_size - 1):
        for j in range(grid_size - 1):
            # Vertex indices for this quad
            v00 = i * grid_size + j
            v01 = i * grid_size + (j + 1)
            v10 = (i + 1) * grid_size + j
            v11 = (i + 1) * grid_size + (j + 1)
            
            # Two triangles per quad
            faces.append([v00, v10, v01])
            faces.append([v01, v10, v11])
    
    faces = np.array(faces, dtype=np.int64)
    print(f"  - Total faces: {faces.shape[0]}")
    
    # Write OBJ file
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(f"# Square cloth mesh {grid_size}x{grid_size}\n")
        f.write(f"# Vertices: {len(verts)}, Faces: {len(faces)}\n")
        f.write(f"# Cloth size: {cloth_size}m x {cloth_size}m\n")
        f.write(f"# Position: centered at origin, height Y={cloth_y}m\n\n")
        
        # Write vertices
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        
        f.write("\n")
        
        # Write faces (OBJ uses 1-based indexing)
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
    
    print(f"  ✓ Saved to: {output_path}")


def create_square_cloth_pkl(output_path: str, grid_size: int = 100, cloth_size: float = 1.0,
                            height_above_sphere: float = 0.5, sphere_radius: float = 0.3):
    """
    Create a square cloth mesh as PKL file.
    
    Args:
        output_path: Output .pkl file path
        grid_size: Number of vertices per side (100 = 100x100 grid)
        cloth_size: Physical size of the cloth (in meters)
        height_above_sphere: Height of cloth above the sphere top
        sphere_radius: Radius of the sphere (to calculate cloth position)
    """
    print(f"\nCreating square cloth garment (PKL): {output_path}")
    print(f"  - Grid size: {grid_size}x{grid_size}")
    print(f"  - Cloth size: {cloth_size}m x {cloth_size}m")
    
    n_verts = grid_size * grid_size
    print(f"  - Total vertices: {n_verts}")
    
    # Calculate cloth center height (above sphere)
    cloth_y = sphere_radius + height_above_sphere
    print(f"  - Cloth height (Y): {cloth_y}m (sphere top: {sphere_radius}m + offset: {height_above_sphere}m)")
    
    # Generate vertices in a grid
    verts = []
    half_size = cloth_size / 2
    
    for i in range(grid_size):
        for j in range(grid_size):
            x = -half_size + (i / (grid_size - 1)) * cloth_size
            z = -half_size + (j / (grid_size - 1)) * cloth_size
            y = cloth_y
            verts.append([x, y, z])
    
    verts = np.array(verts, dtype=np.float32)
    
    # Generate triangular faces for the grid
    faces = []
    for i in range(grid_size - 1):
        for j in range(grid_size - 1):
            v00 = i * grid_size + j
            v01 = i * grid_size + (j + 1)
            v10 = (i + 1) * grid_size + j
            v11 = (i + 1) * grid_size + (j + 1)
            
            faces.append([v00, v10, v01])
            faces.append([v01, v10, v11])
    
    faces = np.array(faces, dtype=np.int64)
    print(f"  - Total faces: {faces.shape[0]}")
    
    # Save to file
    data = {
        'verts': verts,
        'faces': faces
    }
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"  ✓ Saved to: {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate sample mesh data for testing (square cloth + static sphere)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='Tests/data',
        help='Output directory (default: Tests/data)'
    )
    parser.add_argument(
        '--n_frames',
        type=int,
        default=300,
        help='Number of frames in mesh sequence (default: 300)'
    )
    parser.add_argument(
        '--grid_size',
        type=int,
        default=60,
        help='Cloth grid size (default: 60, creates 60x60 grid)'
    )
    parser.add_argument(
        '--cloth_size',
        type=float,
        default=1.0,
        help='Physical size of cloth in meters (default: 1.0)'
    )
    parser.add_argument(
        '--sphere_radius',
        type=float,
        default=0.3,
        help='Sphere radius in meters (default: 0.3)'
    )
    parser.add_argument(
        '--height_above_sphere',
        type=float,
        default=0.5,
        help='Height of cloth above sphere top in meters (default: 0.5)'
    )
    parser.add_argument(
        '--format',
        type=str,
        choices=['obj', 'pkl', 'both'],
        default='both',
        help='Output format for garment (default: both)'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Sample Mesh Data Generator")
    print("  - Garment: Square cloth (grid mesh)")
    print("  - Animation: Static sphere")
    print("=" * 60)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create sample mesh sequence (static sphere)
    mesh_sequence_path = output_dir / 'sphere_sequence.pkl'
    sphere_radius = create_sample_mesh_sequence(
        str(mesh_sequence_path),
        n_frames=args.n_frames,
        sphere_radius=args.sphere_radius
    )
    
    # Create square cloth garment
    if args.format in ['obj', 'both']:
        garment_obj_path = output_dir / 'square_cloth.obj'
        create_square_cloth_obj(
            str(garment_obj_path),
            grid_size=args.grid_size,
            cloth_size=args.cloth_size,
            height_above_sphere=args.height_above_sphere,
            sphere_radius=sphere_radius
        )
    
    # if args.format in ['pkl', 'both']:
    #     garment_pkl_path = output_dir / 'square_cloth.pkl'
    #     create_square_cloth_pkl(
    #         str(garment_pkl_path),
    #         grid_size=args.grid_size,
    #         cloth_size=args.cloth_size,
    #         height_above_sphere=args.height_above_sphere,
    #         sphere_radius=sphere_radius
    #     )
    
    print("\n" + "=" * 60)
    print("Sample data generation completed!")
    print("=" * 60)
    print(f"\nGenerated files:")
    print(f"  - Sphere animation: {output_dir / 'sphere_sequence.pkl'}")
    if args.format in ['obj', 'both']:
        print(f"  - Square cloth (OBJ): {output_dir / 'square_cloth.obj'}")
    if args.format in ['pkl', 'both']:
        print(f"  - Square cloth (PKL): {output_dir / 'square_cloth.pkl'}")
    
    print(f"\nScene setup:")
    print(f"  - Sphere: radius={args.sphere_radius}m, centered at origin")
    print(f"  - Cloth: {args.grid_size}x{args.grid_size} grid, {args.cloth_size}m x {args.cloth_size}m")
    print(f"  - Cloth position: {args.height_above_sphere}m above sphere top")
    print(f"  - Cloth Y coordinate: {args.sphere_radius + args.height_above_sphere}m")
    
    print(f"\nYou can now run training with:")
    print(f"  python Tests/train_mesh.py \\")
    print(f"      --mesh_sequence {output_dir / 'sphere_sequence.pkl'} \\")
    print(f"      --garment {output_dir / 'square_cloth.obj'}")


if __name__ == '__main__':
    main()
