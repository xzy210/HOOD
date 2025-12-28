"""
Batch convert SMPL parameter sequences to mesh format animation sequences.

This script processes all SMPL parameter pkl files and generates:
1. Body mesh sequences: SMPL body meshes for each frame
2. T-shirt mesh sequences: T-shirt meshes transformed using LBS skinning

Input: E:\Projects4\HOOD\hood_data\vto_dataset\smpl_parameters\*.pkl
Output: 
  - E:\Projects4\HOOD\Tests\data\vto_dataset_mesh\body_sequence\{seq_name}.pkl
  - E:\Projects4\HOOD\Tests\data\vto_dataset_mesh\tshirt_sequence\{seq_name}.pkl

Each pkl file contains:
  - 'vertices': np.ndarray of shape (num_frames, num_vertices, 3)
  - 'faces': np.ndarray of shape (num_faces, 3)
  - 'num_frames': int
  - 'num_vertices': int
  - 'source_file': str (original SMPL parameter file name)
"""

import os
import sys
import pickle
import numpy as np
import torch
from sklearn import neighbors
from tqdm import tqdm
import argparse
from glob import glob

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import smplx
from utils.cloth_and_material import load_obj
from utils import lbs as my_lbs
from utils.defaults import DEFAULTS


def load_smpl_sequence(pkl_path: str) -> dict:
    """
    Load SMPL sequence parameters from a pickle file.
    """
    with open(pkl_path, 'rb') as f:
        sequence = pickle.load(f)
    return sequence


def save_mesh_sequence_pkl(output_path: str, vertices: np.ndarray, faces: np.ndarray, 
                           source_file: str) -> None:
    """
    Save mesh sequence to a pkl file.
    
    :param output_path: path to save the pkl file
    :param vertices: np.ndarray of shape (num_frames, num_vertices, 3)
    :param faces: np.ndarray of shape (num_faces, 3)
    :param source_file: original SMPL parameter file name
    """
    data = {
        'vertices': vertices,
        'faces': faces,
        'num_frames': vertices.shape[0],
        'num_vertices': vertices.shape[1],
        'source_file': source_file
    }
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)


def create_lbs_weights_for_garment(garment_vertices: np.ndarray, 
                                    smpl_model: smplx.SMPL,
                                    n_samples: int = 0) -> dict:
    """
    Create LBS skinning weights for garment vertices based on nearest SMPL vertices.
    """
    # Get SMPL rest pose vertices
    smplx_v_rest_pose = smpl_model().vertices[0].detach().cpu().numpy()
    
    # Build KD-tree for nearest neighbor search
    smpl_tree = neighbors.KDTree(smplx_v_rest_pose)
    distances, nn_list = smpl_tree.query(garment_vertices)
    nn_inds = nn_list[..., 0]
    
    if n_samples == 0:
        garment_shapedirs = smpl_model.shapedirs[nn_inds].numpy()
        garment_posedirs = smpl_model.posedirs.reshape(207, -1, 3)[:, nn_inds].reshape(207, -1).numpy()
        garment_lbs_weights = smpl_model.lbs_weights[nn_inds].numpy()
    else:
        garment_shapedirs = 0
        garment_posedirs = 0
        garment_lbs_weights = 0
        
        for i in range(n_samples):
            noise = np.random.randn(*garment_vertices.shape)
            points_sampled = noise * (distances ** 0.5) + garment_vertices
            
            _, nn_list_sampled = smpl_tree.query(points_sampled)
            nn_inds_sampled = nn_list_sampled[..., 0]
            
            garment_shapedirs += smpl_model.shapedirs[nn_inds_sampled].numpy()
            garment_posedirs += smpl_model.posedirs.reshape(207, -1, 3)[:, nn_inds_sampled].reshape(207, -1).numpy()
            garment_lbs_weights += smpl_model.lbs_weights[nn_inds_sampled].numpy()
        
        garment_shapedirs = garment_shapedirs / n_samples
        garment_posedirs = garment_posedirs / n_samples
        garment_lbs_weights = garment_lbs_weights / n_samples
    
    lbs_dict = {
        'v': torch.FloatTensor(garment_vertices),
        'shapedirs': torch.FloatTensor(garment_shapedirs),
        'posedirs': torch.FloatTensor(garment_posedirs),
        'lbs_weights': torch.FloatTensor(garment_lbs_weights)
    }
    
    return lbs_dict


def transform_garment_with_lbs(garment_lbs_dict: dict,
                                smpl_model: smplx.SMPL,
                                betas: np.ndarray,
                                body_pose: np.ndarray,
                                global_orient: np.ndarray,
                                transl: np.ndarray = None) -> np.ndarray:
    """
    Transform garment vertices using LBS skinning based on SMPL parameters.
    """
    betas = torch.FloatTensor(betas)
    body_pose = torch.FloatTensor(body_pose)
    global_orient = torch.FloatTensor(global_orient)
    
    if len(betas.shape) == 1:
        betas = betas.unsqueeze(0)
    if len(body_pose.shape) == 1:
        body_pose = body_pose.unsqueeze(0)
    if len(global_orient.shape) == 1:
        global_orient = global_orient.unsqueeze(0)
    
    full_pose = torch.cat([global_orient, body_pose], dim=1)
    
    J_transformed, joint_transforms = my_lbs.get_transformed_joints(
        betas, full_pose,
        smpl_model.v_template,
        smpl_model.shapedirs,
        smpl_model.J_regressor,
        smpl_model.parents
    )
    
    with torch.no_grad():
        v_garment, _ = my_lbs.pose_garment(
            betas, full_pose,
            garment_lbs_dict['v'],
            garment_lbs_dict['shapedirs'],
            garment_lbs_dict['posedirs'],
            garment_lbs_dict['lbs_weights'],
            J_transformed, joint_transforms
        )
    
    if transl is not None:
        transl = torch.FloatTensor(transl)
        if len(transl.shape) == 1:
            transl = transl.unsqueeze(0)
        v_garment = v_garment + transl[:, None]
    
    return v_garment[0].numpy()


def generate_body_mesh(smpl_model: smplx.SMPL,
                       betas: np.ndarray,
                       body_pose: np.ndarray,
                       global_orient: np.ndarray,
                       transl: np.ndarray = None) -> np.ndarray:
    """
    Generate SMPL body mesh vertices for given parameters.
    """
    betas_t = torch.FloatTensor(betas)
    body_pose_t = torch.FloatTensor(body_pose)
    global_orient_t = torch.FloatTensor(global_orient)
    
    if len(betas_t.shape) == 1:
        betas_t = betas_t.unsqueeze(0)
    if len(body_pose_t.shape) == 1:
        body_pose_t = body_pose_t.unsqueeze(0)
    if len(global_orient_t.shape) == 1:
        global_orient_t = global_orient_t.unsqueeze(0)
    
    transl_t = None
    if transl is not None:
        transl_t = torch.FloatTensor(transl)
        if len(transl_t.shape) == 1:
            transl_t = transl_t.unsqueeze(0)
    
    with torch.no_grad():
        output = smpl_model(
            betas=betas_t,
            body_pose=body_pose_t,
            global_orient=global_orient_t,
            transl=transl_t
        )
    
    return output.vertices[0].numpy()


def get_sequence_name(pkl_filename: str) -> str:
    """
    Extract sequence name from pkl filename.
    e.g., "tshirt_shape00_01_01.pkl" -> "tshirt_shape00_01_01"
    """
    return os.path.splitext(pkl_filename)[0]


def process_single_sequence(pkl_path: str,
                            smpl_model: smplx.SMPL,
                            garment_lbs_dict: dict,
                            garment_faces: np.ndarray,
                            body_output_dir: str,
                            tshirt_output_dir: str,
                            skip_existing: bool = True) -> bool:
    """
    Process a single SMPL sequence and generate body and tshirt mesh sequences.
    
    :param pkl_path: path to the SMPL parameter pkl file
    :param smpl_model: SMPL model instance
    :param garment_lbs_dict: LBS parameters for garment
    :param garment_faces: garment face indices
    :param body_output_dir: output directory for body meshes
    :param tshirt_output_dir: output directory for tshirt meshes
    :param skip_existing: skip if output already exists
    :return: True if successful, False otherwise
    """
    seq_name = get_sequence_name(os.path.basename(pkl_path))
    
    body_pkl_path = os.path.join(body_output_dir, f"{seq_name}.pkl")
    tshirt_pkl_path = os.path.join(tshirt_output_dir, f"{seq_name}.pkl")
    
    # Check if already processed
    if skip_existing and os.path.exists(body_pkl_path) and os.path.exists(tshirt_pkl_path):
        return True
    
    try:
        # Load SMPL sequence
        sequence = load_smpl_sequence(pkl_path)
        
        # Get number of frames
        num_frames = sequence['body_pose'].shape[0]
        betas = sequence['betas']
        has_transl = 'transl' in sequence
        
        # Get SMPL faces (only need once)
        smpl_faces = smpl_model.faces
        
        # Collect all frames
        body_vertices_list = []
        tshirt_vertices_list = []
        
        # Process each frame
        for frame_idx in range(num_frames):
            body_pose = sequence['body_pose'][frame_idx]
            global_orient = sequence['global_orient'][frame_idx]
            transl = sequence['transl'][frame_idx] if has_transl else None
            
            # Generate body mesh
            body_verts = generate_body_mesh(
                smpl_model, betas, body_pose, global_orient, transl
            )
            body_vertices_list.append(body_verts)
            
            # Generate tshirt mesh
            tshirt_verts = transform_garment_with_lbs(
                garment_lbs_dict, smpl_model, betas, body_pose, global_orient, transl
            )
            tshirt_vertices_list.append(tshirt_verts)
        
        # Stack all frames into arrays
        body_vertices_array = np.stack(body_vertices_list, axis=0)  # (num_frames, num_verts, 3)
        tshirt_vertices_array = np.stack(tshirt_vertices_list, axis=0)  # (num_frames, num_verts, 3)
        
        # Save as pkl files
        source_file = os.path.basename(pkl_path)
        save_mesh_sequence_pkl(body_pkl_path, body_vertices_array, smpl_faces, source_file)
        save_mesh_sequence_pkl(tshirt_pkl_path, tshirt_vertices_array, garment_faces, source_file)
        
        return True
        
    except Exception as e:
        print(f"\nError processing {seq_name}: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Batch convert SMPL sequences to mesh sequences')
    parser.add_argument('--input_dir', type=str, 
                        default=r'E:\Projects4\HOOD\hood_data\vto_dataset\smpl_parameters',
                        help='Input directory containing SMPL pkl files')
    parser.add_argument('--body_output_dir', type=str,
                        default=r'E:\Projects4\HOOD\Tests\data\vto_dataset_mesh\body_sequence',
                        help='Output directory for body mesh sequences')
    parser.add_argument('--tshirt_output_dir', type=str,
                        default=r'E:\Projects4\HOOD\Tests\data\vto_dataset_mesh\tshirt_sequence',
                        help='Output directory for tshirt mesh sequences')
    parser.add_argument('--tshirt_obj', type=str,
                        default=r'E:\Projects4\HOOD\Tests\data\tshirt.obj',
                        help='Path to the template tshirt obj file')
    parser.add_argument('--skip_existing', action='store_true', default=True,
                        help='Skip sequences that are already processed')
    parser.add_argument('--limit', type=int, default=0,
                        help='Limit number of sequences to process (0 for all)')
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.body_output_dir, exist_ok=True)
    os.makedirs(args.tshirt_output_dir, exist_ok=True)
    
    print("="*60)
    print("Batch SMPL to Mesh Sequence Converter")
    print("="*60)
    print(f"Input directory: {args.input_dir}")
    print(f"Body output directory: {args.body_output_dir}")
    print(f"T-shirt output directory: {args.tshirt_output_dir}")
    print(f"T-shirt template: {args.tshirt_obj}")
    print("="*60)
    
    # Load SMPL model
    smpl_model_path = os.path.join(DEFAULTS.aux_data, 'smpl', 'SMPL_FEMALE.pkl')
    print(f"\nLoading SMPL model from: {smpl_model_path}")
    smpl_model = smplx.SMPL(smpl_model_path)
    print("SMPL model loaded successfully")
    
    # Load tshirt template
    print(f"\nLoading tshirt template from: {args.tshirt_obj}")
    garment_vertices, garment_faces = load_obj(args.tshirt_obj, tex_coords=False)
    print(f"Tshirt loaded: {garment_vertices.shape[0]} vertices, {garment_faces.shape[0]} faces")
    
    # Create LBS weights for garment (only need to do once)
    print("\nCreating LBS weights for tshirt (this may take a moment)...")
    garment_lbs_dict = create_lbs_weights_for_garment(garment_vertices, smpl_model, n_samples=0)
    print("LBS weights created successfully")
    
    # Get all pkl files
    pkl_files = sorted(glob(os.path.join(args.input_dir, "*.pkl")))
    total_files = len(pkl_files)
    print(f"\nFound {total_files} SMPL sequence files")
    
    if args.limit > 0:
        pkl_files = pkl_files[:args.limit]
        print(f"Processing limited to first {args.limit} files")
    
    # Process all sequences
    print("\n" + "="*60)
    print("Starting batch processing...")
    print("="*60 + "\n")
    
    success_count = 0
    skip_count = 0
    fail_count = 0
    
    for pkl_path in tqdm(pkl_files, desc="Processing sequences"):
        seq_name = get_sequence_name(os.path.basename(pkl_path))
        
        # Check if already processed
        body_pkl_path = os.path.join(args.body_output_dir, f"{seq_name}.pkl")
        tshirt_pkl_path = os.path.join(args.tshirt_output_dir, f"{seq_name}.pkl")
        
        if args.skip_existing and os.path.exists(body_pkl_path) and os.path.exists(tshirt_pkl_path):
            skip_count += 1
            continue
        
        success = process_single_sequence(
            pkl_path,
            smpl_model,
            garment_lbs_dict,
            garment_faces,
            args.body_output_dir,
            args.tshirt_output_dir,
            skip_existing=args.skip_existing
        )
        
        if success:
            success_count += 1
        else:
            fail_count += 1
    
    # Summary
    print("\n" + "="*60)
    print("Processing Complete!")
    print("="*60)
    print(f"Total sequences: {len(pkl_files)}")
    print(f"Successfully processed: {success_count}")
    print(f"Skipped (already exists): {skip_count}")
    print(f"Failed: {fail_count}")
    print(f"\nBody sequences saved to: {args.body_output_dir}")
    print(f"T-shirt sequences saved to: {args.tshirt_output_dir}")


if __name__ == '__main__':
    main()
