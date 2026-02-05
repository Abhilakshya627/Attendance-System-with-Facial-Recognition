"""
General utilities for model training.
Includes model checkpointing, logging, and common helper functions.
"""

import torch
import json
from pathlib import Path
from typing import Dict, Any, Optional
import logging
from datetime import datetime


def setup_logging(log_dir: str, experiment_name: str = None) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        log_dir: Directory to save logs
        experiment_name: Name of the experiment
        
    Returns:
        Configured logger
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    if experiment_name is None:
        experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    log_file = log_dir / f"{experiment_name}.log"
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_loss: float,
    checkpoint_dir: str,
    model_name: str = "model"
):
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state to save
        epoch: Current epoch number
        best_loss: Best loss achieved
        checkpoint_dir: Directory to save checkpoint
        model_name: Name of the model
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_loss': best_loss
    }
    
    checkpoint_path = checkpoint_dir / f"{model_name}_epoch_{epoch}.pt"
    torch.save(checkpoint, checkpoint_path)
    
    # Also save best model
    best_path = checkpoint_dir / f"{model_name}_best.pt"
    if epoch == 0 or best_loss < checkpoint['best_loss']:
        torch.save(checkpoint, best_path)
    
    return checkpoint_path


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    checkpoint_path: str,
    device: str = 'cpu'
) -> Dict[str, Any]:
    """
    Load model checkpoint.
    
    Args:
        model: Model to load state into
        optimizer: Optimizer to load state into (optional)
        checkpoint_path: Path to checkpoint file
        device: Device to load model to
        
    Returns:
        Dictionary with checkpoint metadata
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return {
        'epoch': checkpoint['epoch'],
        'best_loss': checkpoint['best_loss']
    }


def save_training_config(config: Dict[str, Any], output_dir: str):
    """
    Save training configuration to JSON.
    
    Args:
        config: Configuration dictionary
        output_dir: Output directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config_path = output_dir / "training_config.json"
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)


def load_training_config(config_path: str) -> Dict[str, Any]:
    """
    Load training configuration from JSON.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config


def get_device() -> str:
    """
    Get available device (GPU or CPU).
    
    Returns:
        Device string ('cuda' or 'cpu')
    """
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count total trainable parameters in model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Total number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_optimizer(
    model: torch.nn.Module,
    lr: float = 0.001,
    optimizer_type: str = 'adam'
) -> torch.optim.Optimizer:
    """
    Create optimizer for model training.
    
    Args:
        model: Model to optimize
        lr: Learning rate
        optimizer_type: Type of optimizer ('adam', 'sgd', 'adamw')
        
    Returns:
        Optimizer instance
    """
    if optimizer_type.lower() == 'adam':
        return torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_type.lower() == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=lr)
    elif optimizer_type.lower() == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str = 'cosine',
    num_epochs: int = 100
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Create learning rate scheduler.
    
    Args:
        optimizer: Optimizer to schedule
        scheduler_type: Type of scheduler ('cosine', 'step', 'exponential')
        num_epochs: Total number of training epochs
        
    Returns:
        Learning rate scheduler
    """
    if scheduler_type.lower() == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    elif scheduler_type.lower() == 'step':
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    elif scheduler_type.lower() == 'exponential':
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


def adjust_learning_rate(
    optimizer: torch.optim.Optimizer,
    epoch: int,
    initial_lr: float,
    warmup_epochs: int = 5
):
    """
    Adjust learning rate with warmup.
    
    Args:
        optimizer: Optimizer
        epoch: Current epoch
        initial_lr: Initial learning rate
        warmup_epochs: Number of warmup epochs
    """
    if epoch < warmup_epochs:
        # Linear warmup
        lr = initial_lr * (epoch + 1) / warmup_epochs
    else:
        # No additional adjustment after warmup
        lr = initial_lr
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
