import os
import numpy as np
import torch
from datetime import datetime

from utils.arguments import load_params, create_modules
from utils.defaults import DEFAULTS

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
    """
    if not TENSORBOARD_AVAILABLE:
        print("[TensorBoard] WARNING: tensorboard is not installed. Install with 'pip install tensorboard'")
        print("[TensorBoard] Training will continue without TensorBoard logging.")
        return None
    
    if experiment_name is None:
        experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    tensorboard_dir = os.path.join(log_dir, 'tensorboard')
    os.makedirs(tensorboard_dir, exist_ok=True)
    
    writer = SummaryWriter(log_dir=tensorboard_dir)
    print(f"[TensorBoard] Logs will be saved to: {tensorboard_dir}")
    print(f"[TensorBoard] Run 'tensorboard --logdir={os.path.dirname(tensorboard_dir)}' to visualize")
    
    return writer


def main():
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    torch.set_num_threads(1)
    
    # Load params with default config 'mesh' if not specified via CLI
    # This allows running `python train_mesh.py` directly
    modules, config = load_params(config_name='mesh')
    dataloader_m, runner, training_module, aux_modules = create_modules(modules, config)

    if config.experiment.checkpoint_path is not None and os.path.exists(config.experiment.checkpoint_path):
        sd = torch.load(config.experiment.checkpoint_path)

        if 'training_module' in sd:
            training_module.load_state_dict(sd['training_module'])

            for k, v in aux_modules.items():
                if k in sd:
                    print(f'{k} LOADED!')
                    v.load_state_dict(sd[k])
        else:
            training_module.load_state_dict(sd)
        print('LOADED:', config.experiment.checkpoint_path)

    if config.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)

    # Setup run directory and TensorBoard
    now = datetime.now()
    dt_string = now.strftime("%Y%m%d_%H%M%S")
    exp_name = config.experiment.name if config.experiment.name else dt_string
    
    # Use DEFAULTS.experiment_root (usually 'checkpoints')
    # We create a subdirectory for this experiment
    run_dir = os.path.join(DEFAULTS.experiment_root, exp_name, dt_string)
    config.run_dir = run_dir
    
    writer = create_tensorboard_writer(run_dir, exp_name)

    global_step = config.step_start

    torch.manual_seed(57)
    np.random.seed(57)
    
    for i in range(config.experiment.n_epochs):
        dataloader = dataloader_m.create_dataloader()
        
        # Call run_epoch with writer
        # runner is the module (runners/mesh.py)
        global_step = runner.run_epoch(training_module, aux_modules, dataloader, i, config,
                                       global_step=global_step, writer=writer)

        if config.experiment.max_iter is not None and global_step > config.experiment.max_iter:
            break
            
    if writer is not None:
        writer.close()


if __name__ == '__main__':
    main()
