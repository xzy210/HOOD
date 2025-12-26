# Mesh Format Training

This directory contains scripts for training HOOD using mesh format animation sequences.

## Files

- `train_mesh.py` - Main training script for mesh format data
- `train_mesh_config.yaml` - Configuration file for mesh training
- `generate_sample_mesh_data.py` - Helper script to generate sample test data
- `convert_smpl_to_mesh.py` - Script to convert SMPL format to Mesh format

## Quick Start

### 1. Generate Sample Data

First, generate sample mesh data for testing:

```bash
python Tests/generate_sample_mesh_data.py
```

This will create:
- `Tests/mesh_sequence.pkl` - Sample mesh animation sequence
- `Tests/garment_template.pkl` - Sample garment template

### 2. Start Training

Train with the generated sample data:

```bash
python Tests/train_mesh.py
```

### 3. Use Your Own Data

#### Using Configuration File

Edit `train_mesh_config.yaml` and set your data paths:

```yaml
dataset:
  from_any_pose:
    pose_sequence_type: "mesh"
    pose_sequence_path: 'path/to/your/mesh_sequence.pkl'
    garment_template_path: 'path/to/your/garment_template.pkl'
```

Then run:

```bash
python Tests/train_mesh.py --config Tests/train_mesh_config.yaml
```

#### Using Command Line Arguments

Specify paths directly:

```bash
python Tests/train_mesh.py \
    --mesh_sequence /path/to/your/mesh_sequence.pkl \
    --garment /path/to/your/garment_template.pkl
```

Note: The `--garment` parameter supports both `.pkl` and `.obj` formats.

### 4. Resume Training

To resume from a checkpoint:

```bash
python Tests/train_mesh.py --checkpoint Tests/checkpoints/checkpoint.pth
```

## Data Format

### Mesh Sequence Format (.pkl)

```python
{
    'verts': np.ndarray,  # [N, V, 3] - N frames, V vertices, 3 coordinates (x, y, z)
    'faces': np.ndarray   # [F, 3]    - F triangular faces
}
```

Requirements:
- All frames must have the same number of vertices
- Vertices should be in meters
- Coordinates: x (horizontal), y (vertical), z (depth)
- Face indices are 0-based

### Garment Template Format (.pkl or .obj)

The garment template can be in either format:

**Format 1: .pkl file**
```python
{
    'verts': np.ndarray,  # [V, 3] - V vertices in rest pose
    'faces': np.ndarray   # [F, 3] - F triangular faces
}
```

**Format 2: .obj file**
Standard OBJ mesh format. The loader will automatically convert it to the required format.

Note: When using `.obj` format for the first time, it may take longer to load as the system builds coarse edges. For repeated training, consider converting to `.pkl` format using `utils.mesh_creation.obj2template()`.

### Convert SMPL to Mesh

If you have SMPL format data, convert it using:

```bash
python Tests/convert_smpl_to_mesh.py \
    input_smpl_sequence.pkl \
    output_mesh_sequence.pkl \
    --gender neutral
```

## Training Outputs

All training outputs are saved in `Tests/checkpoints/`:

- Checkpoints saved every 2000 iterations
- Checkpoint files named with iteration number
- Configuration files and logs included

## Configuration Options

Edit `train_mesh_config.yaml` to customize:

- `experiment.n_epochs` - Number of training epochs
- `experiment.save_checkpoint_every` - Checkpoint save frequency
- `dataloader.num_workers` - Data loading workers (set to 0 if issues occur)
- `dataloader.dataset.from_any_pose.n_coarse_levels` - Number of coarse levels for collision detection
- `runner.from_any_pose.material` - Material parameters for simulation

## Example Workflow

```bash
# Step 1: Convert your FBX/SMPL data to mesh format
python Tests/convert_smpl_to_mesh.py your_data.pkl your_data_mesh.pkl

# Step 2: Prepare garment template
# Option A: Use .obj file directly
python Tests/train_mesh.py \
    --mesh_sequence your_data_mesh.pkl \
    --garment your_garment_template.obj

# Option B: Convert .obj to .pkl first (faster for repeated training)
# See utils/mesh_creation.obj2template() function
python Tests/train_mesh.py \
    --mesh_sequence your_data_mesh.pkl \
    --garment your_garment_template.pkl

# Step 3: Monitor training
# Checkpoints saved in Tests/checkpoints/
```

## Troubleshooting

### File not found errors
- Make sure data file paths are correct
- Use absolute paths if relative paths don't work
- Run from project root directory

### Out of memory errors
- Reduce batch size (currently fixed at 1)
- Use smaller mesh sequences
- Check GPU memory usage

### Multi-processing issues
- Set `num_workers: 0` in config
- This is already set by default

## Notes

- Mesh format is equivalent to SMPL format for training and inference
- SMPL model file is still required by the codebase but not used in mesh mode
- All training results are saved in the `Tests/checkpoints/` directory
