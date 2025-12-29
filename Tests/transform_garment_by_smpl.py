"""
Transform static garment mesh using SMPL parameters and LBS skinning.

This script demonstrates how to transform a static T-pose garment to a specific pose
using Linear Blend Skinning (LBS) weights derived from the nearest SMPL vertices.
"""

import os
import sys
import pickle
import numpy as np
import torch
from sklearn import neighbors

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import smplx
from utils.cloth_and_material import load_obj, save_obj
from utils import lbs as my_lbs
from utils.defaults import DEFAULTS


def load_smpl_sequence(pkl_path: str) -> dict:
    """
    Load SMPL sequence parameters from a pickle file.
    
    :param pkl_path: path to the pickle file
    :return: dictionary containing SMPL parameters
    """
    with open(pkl_path, 'rb') as f:
        sequence = pickle.load(f)
    return sequence


def create_lbs_weights_for_garment(garment_vertices: np.ndarray, 
                                    smpl_model: smplx.SMPL,
                                    n_samples: int = 0) -> dict:
    """
    Create LBS skinning weights for garment vertices based on nearest SMPL vertices.
    
    :param garment_vertices: [Vx3] garment template vertices
    :param smpl_model: SMPL model instance
    :param n_samples: number of samples for diffused weights (0 for nearest neighbor only)
    :return: dictionary containing LBS parameters (v, shapedirs, posedirs, lbs_weights)
    """
    # Get SMPL rest pose vertices
    smplx_v_rest_pose = smpl_model().vertices[0].detach().cpu().numpy()
    
    # Build KD-tree for nearest neighbor search
    smpl_tree = neighbors.KDTree(smplx_v_rest_pose)
    distances, nn_list = smpl_tree.query(garment_vertices)
    nn_inds = nn_list[..., 0]
    
    if n_samples == 0:
        # Take weights of the closest SMPL vertex
        garment_shapedirs = smpl_model.shapedirs[nn_inds].numpy()
        garment_posedirs = smpl_model.posedirs.reshape(207, -1, 3)[:, nn_inds].reshape(207, -1).numpy()
        garment_lbs_weights = smpl_model.lbs_weights[nn_inds].numpy()
    else:
        # Use diffused sampling approach (for loose garments)
        garment_shapedirs = 0
        garment_posedirs = 0
        garment_lbs_weights = 0
        
        for i in range(n_samples):
            # Sample random points around garment vertices
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
    
    :param garment_lbs_dict: dictionary containing garment LBS parameters
    :param smpl_model: SMPL model instance
    :param betas: [10] or [1x10] shape parameters
    :param body_pose: [69] or [1x69] body pose parameters
    :param global_orient: [3] or [1x3] global orientation
    :param transl: [3] or [1x3] translation (optional)
    :return: [Vx3] transformed garment vertices
    """
    # Convert to torch tensors and ensure batch dimension
    betas = torch.FloatTensor(betas)
    body_pose = torch.FloatTensor(body_pose)
    global_orient = torch.FloatTensor(global_orient)
    
    if len(betas.shape) == 1:
        betas = betas.unsqueeze(0)
    if len(body_pose.shape) == 1:
        body_pose = body_pose.unsqueeze(0)
    if len(global_orient.shape) == 1:
        global_orient = global_orient.unsqueeze(0)
    
    # Combine global_orient and body_pose to get full_pose [1x72]
    full_pose = torch.cat([global_orient, body_pose], dim=1)
    
    # Get transformed joints from SMPL model
    J_transformed, joint_transforms = my_lbs.get_transformed_joints(
        betas, full_pose,
        smpl_model.v_template,
        smpl_model.shapedirs,
        smpl_model.J_regressor,
        smpl_model.parents
    )
    
    # Apply LBS to garment
    with torch.no_grad():
        v_garment, _ = my_lbs.pose_garment(
            betas, full_pose,
            garment_lbs_dict['v'],
            garment_lbs_dict['shapedirs'],
            garment_lbs_dict['posedirs'],
            garment_lbs_dict['lbs_weights'],
            J_transformed, joint_transforms
        )
    
    # Add translation if provided
    if transl is not None:
        transl = torch.FloatTensor(transl)
        if len(transl.shape) == 1:
            transl = transl.unsqueeze(0)
        v_garment = v_garment + transl[:, None]
    
    return v_garment[0].numpy()


def main():
    """
    Main function to transform tshirt.obj using first frame of 01_01.pkl SMPL parameters.
    """
    # Paths
    data_dir = os.path.join(project_root, 'Tests', 'data')
    garment_obj_path = os.path.join(data_dir, 'tshirt.obj')
    smpl_sequence_path = os.path.join(project_root, 'hood_data','vto_dataset','smpl_parameters','tshirt_shape00_01_01.pkl')
    output_obj_path = os.path.join(data_dir, 'tshirt_transformed.obj')
    
    # SMPL model path
    smpl_model_path = os.path.join(DEFAULTS.aux_data, 'smpl', 'SMPL_FEMALE.pkl')
    
    print(f"Loading garment from: {garment_obj_path}")
    print(f"Loading SMPL sequence from: {smpl_sequence_path}")
    print(f"Using SMPL model: {smpl_model_path}")
    
    # Load garment mesh
    garment_vertices, garment_faces = load_obj(garment_obj_path, tex_coords=False)
    print(f"Garment loaded: {garment_vertices.shape[0]} vertices, {garment_faces.shape[0]} faces")
    
    # Load SMPL sequence
    sequence = load_smpl_sequence(smpl_sequence_path)
    print(f"SMPL sequence loaded: {sequence['body_pose'].shape[0]} frames")
    
    # Print available keys
    print(f"Available keys in sequence: {list(sequence.keys())}")
    
    # Load SMPL model
    smpl_model = smplx.SMPL(smpl_model_path)
    print("SMPL model loaded")
    
    # Create LBS weights for garment
    # For tight-fitting garments like t-shirt, use n_samples=0
    print("Creating LBS weights for garment (this may take a moment)...")
    garment_lbs_dict = create_lbs_weights_for_garment(
        garment_vertices, smpl_model, n_samples=0
    )
    print("LBS weights created")
    
    # Get first frame SMPL parameters
    frame_idx = 0
    betas = sequence['betas']
    body_pose = sequence['body_pose'][frame_idx]
    global_orient = sequence['global_orient'][frame_idx]
    transl = sequence['transl'][frame_idx] if 'transl' in sequence else None
    
    print(f"\nFirst frame SMPL parameters:")
    print(f"  betas shape: {betas.shape}")
    print(f"  body_pose shape: {body_pose.shape}")
    print(f"  global_orient shape: {global_orient.shape}")
    if transl is not None:
        print(f"  transl shape: {transl.shape}")
    
    # Transform garment
    print("\nTransforming garment with LBS...")
    transformed_vertices = transform_garment_with_lbs(
        garment_lbs_dict,
        smpl_model,
        betas,
        body_pose,
        global_orient,
        transl
    )
    print(f"Transformed garment: {transformed_vertices.shape[0]} vertices")
    
    # Save transformed garment
    save_obj(output_obj_path, transformed_vertices, garment_faces)
    print(f"\nTransformed garment saved to: {output_obj_path}")
    
    # Print some statistics
    print(f"\n=== Transformation Statistics ===")
    print(f"Original vertex range:")
    print(f"  X: [{garment_vertices[:, 0].min():.4f}, {garment_vertices[:, 0].max():.4f}]")
    print(f"  Y: [{garment_vertices[:, 1].min():.4f}, {garment_vertices[:, 1].max():.4f}]")
    print(f"  Z: [{garment_vertices[:, 2].min():.4f}, {garment_vertices[:, 2].max():.4f}]")
    print(f"Transformed vertex range:")
    print(f"  X: [{transformed_vertices[:, 0].min():.4f}, {transformed_vertices[:, 0].max():.4f}]")
    print(f"  Y: [{transformed_vertices[:, 1].min():.4f}, {transformed_vertices[:, 1].max():.4f}]")
    print(f"  Z: [{transformed_vertices[:, 2].min():.4f}, {transformed_vertices[:, 2].max():.4f}]")


if __name__ == '__main__':
    main()
