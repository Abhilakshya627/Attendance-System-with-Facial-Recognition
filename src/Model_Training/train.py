"""
Main training orchestrator for face detection system.
Coordinates training of RetinaFace and YOLO detectors with optional fusion.
"""

import torch
import yaml
import argparse
from pathlib import Path
from typing import Dict, Optional
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from utils import (
    create_data_loaders,
    setup_logging,
    save_training_config,
    get_device
)
from RetinaFace.trainer import RetinaFaceTrainer
from YOLO.trainer import YOLOTrainer


class TrainingOrchestrator:
    """
    Orchestrates training of face detection models.
    Manages RetinaFace, YOLO, and detection fusion.
    """
    
    def __init__(self, config_path: str, output_dir: str = './outputs'):
        """
        Initialize training orchestrator.
        
        Args:
            config_path: Path to training configuration file
            output_dir: Output directory for all results
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = setup_logging(str(self.output_dir), 'training_orchestrator')
        
        # Save configuration
        save_training_config(self.config, str(self.output_dir))
        
        # Device setup
        self.device = get_device() if self.config.get('device', {}).get('cuda', True) else 'cpu'
        self.logger.info(f"Using device: {self.device}")
        
        # Initialize data loaders
        self.logger.info("Creating data loaders...")
        self.train_loader, self.val_loader = self._create_dataloaders()
        self.logger.info(f"Train batches: {len(self.train_loader)}, Val batches: {len(self.val_loader)}")
        
        # Initialize trainers
        self.retinaface_trainer = None
        self.yolo_trainer = None
        
        if self.config.get('retinaface', {}).get('enabled', True):
            self.logger.info("Initializing RetinaFace trainer...")
            self.retinaface_trainer = RetinaFaceTrainer(
                config=self.config.get('retinaface', {}),
                output_dir=self.config.get('retinaface', {}).get('output_dir', './outputs/retinaface'),
                device=self.device
            )
        
        if self.config.get('yolo', {}).get('enabled', True):
            self.logger.info("Initializing YOLO trainer...")
            self.yolo_trainer = YOLOTrainer(
                config=self.config.get('yolo', {}),
                output_dir=self.config.get('yolo', {}).get('output_dir', './outputs/yolo'),
                device=self.device
            )
    
    def _create_dataloaders(self):
        """Create training and validation data loaders."""
        data_config = self.config.get('data', {})
        train_annotation = data_config.get('train_annotation')
        val_annotation = data_config.get('val_annotation')
        
        # Get the directory for images (same as WIDER_train/WIDER_train/images)
        data_root = Path(data_config.get('data_root', 'data/WiderFace'))
        img_dir = data_root / "WIDER_train" / "WIDER_train" / "images"
        
        train_loader, val_loader = create_data_loaders(
            train_annotation=train_annotation,
            val_annotation=val_annotation,
            img_dir=str(img_dir),
            batch_size=data_config.get('batch_size', 16),
            num_workers=data_config.get('num_workers', 4),
            img_size=data_config.get('img_size', 640),
            augment=data_config.get('augmentation', True)
        )
        
        return train_loader, val_loader
    
    def train(self):
        """
        Execute training pipeline.
        Trains RetinaFace and YOLO models sequentially or in parallel.
        """
        self.logger.info("=" * 80)
        self.logger.info("Starting Face Detection Training Pipeline")
        self.logger.info("=" * 80)
        
        results = {}
        
        # Train RetinaFace
        if self.retinaface_trainer is not None:
            self.logger.info("\n" + "=" * 80)
            self.logger.info("Training RetinaFace Model")
            self.logger.info("=" * 80)
            
            try:
                best_loss = self.retinaface_trainer.train(
                    self.train_loader,
                    self.val_loader
                )
                results['retinaface'] = {'best_loss': best_loss}
                self.logger.info(f"RetinaFace training completed with best loss: {best_loss:.4f}")
            except Exception as e:
                self.logger.error(f"RetinaFace training failed: {str(e)}")
                results['retinaface'] = {'error': str(e)}
        
        # Train YOLO
        if self.yolo_trainer is not None:
            self.logger.info("\n" + "=" * 80)
            self.logger.info("Training YOLO Model")
            self.logger.info("=" * 80)
            
            try:
                best_loss = self.yolo_trainer.train(
                    self.train_loader,
                    self.val_loader
                )
                results['yolo'] = {'best_loss': best_loss}
                self.logger.info(f"YOLO training completed with best loss: {best_loss:.4f}")
            except Exception as e:
                self.logger.error(f"YOLO training failed: {str(e)}")
                results['yolo'] = {'error': str(e)}
        
        # Summary
        self.logger.info("\n" + "=" * 80)
        self.logger.info("Training Pipeline Summary")
        self.logger.info("=" * 80)
        for model_name, result in results.items():
            if 'error' in result:
                self.logger.info(f"{model_name}: FAILED - {result['error']}")
            else:
                self.logger.info(f"{model_name}: Success - Best Loss: {result['best_loss']:.4f}")
        
        self.logger.info("\nAll training completed!")
        return results
    
    def evaluate(self):
        """
        Evaluate trained models on validation set.
        """
        self.logger.info("\n" + "=" * 80)
        self.logger.info("Model Evaluation")
        self.logger.info("=" * 80)
        
        if self.retinaface_trainer is not None:
            self.logger.info("\nEvaluating RetinaFace...")
            val_metrics = self.retinaface_trainer.validate(self.val_loader)
            self.logger.info(f"RetinaFace Val Loss: {val_metrics['loss']:.4f}")
        
        if self.yolo_trainer is not None:
            self.logger.info("\nEvaluating YOLO...")
            val_metrics = self.yolo_trainer.validate(self.val_loader)
            self.logger.info(f"YOLO Val Loss: {val_metrics['loss']:.4f}")


def main():
    """Main entry point for training."""
    parser = argparse.ArgumentParser(
        description='Train face detection models (RetinaFace and YOLO)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='src/Model_Training/configs/training_config.yaml',
        help='Path to training configuration file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./outputs',
        help='Output directory for training results'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume training from'
    )
    
    args = parser.parse_args()
    
    # Create orchestrator and train
    orchestrator = TrainingOrchestrator(args.config, args.output_dir)
    results = orchestrator.train()
    
    # Evaluate
    orchestrator.evaluate()
    
    return results


if __name__ == '__main__':
    main()
