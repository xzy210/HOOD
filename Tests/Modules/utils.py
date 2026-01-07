#!/usr/bin/env python3
"""
Utility functions for mesh-based training.
"""

import os
from datetime import datetime

# Optional TensorBoard support
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    SummaryWriter = None
    TENSORBOARD_AVAILABLE = False


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
