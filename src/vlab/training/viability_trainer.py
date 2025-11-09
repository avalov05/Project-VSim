"""
Training Pipeline for Viability Prediction Model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
import json
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from torch.cuda.amp import GradScaler, autocast

from ..viability.predictor import ViabilityPredictor
from ..viability.features import FeatureExtractor
from ..core.config import VLabConfig

logger = logging.getLogger(__name__)

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        bce_loss = nn.functional.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class ViabilityDataset(Dataset):
    """Dataset for viability prediction"""
    
    def __init__(self, annotations: List[Dict[str, Any]], labels: List[float]):
        self.annotations = annotations
        self.labels = labels
        self.feature_extractor = FeatureExtractor(None)
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        label = self.labels[idx]
        
        # Extract features
        features = self.feature_extractor.extract(annotation)
        
        return {
            'sequence_features': torch.tensor(features['sequence_features'], dtype=torch.float32),
            'structural_features': torch.tensor(features['structural_features'], dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.float32)
        }

class ViabilityTrainer:
    """Train viability prediction model with advanced features for maximum quality"""
    
    def __init__(self, config: VLabConfig, 
                 input_dim: int = 1024, 
                 hidden_dim: int = 512, 
                 num_layers: int = 8,
                 num_heads: int = 16,
                 use_focal_loss: bool = True,
                 use_mixed_precision: bool = True):
        self.config = config
        self.device = torch.device(f"cuda:{config.gpu_id}" if config.use_gpu and torch.cuda.is_available() else "cpu")
        self.use_mixed_precision = use_mixed_precision and torch.cuda.is_available()
        self.scaler = GradScaler() if self.use_mixed_precision else None
        
        # Enhanced model architecture
        self.model = ViabilityPredictor(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads
        ).to(self.device)
        
        self.feature_extractor = FeatureExtractor(config)
        self.use_focal_loss = use_focal_loss
        
        # Store architecture parameters for saving
        self.model_params = {
            'input_dim': input_dim,
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'num_heads': num_heads
        }
        
        # Training history
        self.train_history = {
            'loss': [],
            'accuracy': [],
            'f1': [],
            'auc': []
        }
        self.val_history = {
            'loss': [],
            'accuracy': [],
            'f1': [],
            'auc': []
        }
    
    def train(self, train_annotations: List[Dict[str, Any]], 
             train_labels: List[float],
             val_annotations: List[Dict[str, Any]] = None,
             val_labels: List[float] = None,
             epochs: int = 100,
             batch_size: int = 32,
             learning_rate: float = 1e-4,
             weight_decay: float = 1e-5,
             warmup_epochs: int = 5,
             early_stopping_patience: int = 15,
             gradient_clip_val: float = 1.0):
        """Train the model with advanced features for maximum quality"""
        logger.info("Starting training with enhanced configuration...")
        logger.info(f"Model architecture: input_dim=1024, hidden_dim=512, num_layers=8, num_heads=16")
        logger.info(f"Training settings: epochs={epochs}, batch_size={batch_size}, lr={learning_rate}")
        logger.info(f"Advanced features: focal_loss={self.use_focal_loss}, mixed_precision={self.use_mixed_precision}")
        
        # Create datasets
        train_dataset = ViabilityDataset(train_annotations, train_labels)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=2,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        if val_annotations:
            val_dataset = ViabilityDataset(val_annotations, val_labels)
            val_loader = DataLoader(
                val_dataset, 
                batch_size=batch_size, 
                shuffle=False,
                num_workers=2,
                pin_memory=True if torch.cuda.is_available() else False
            )
        else:
            val_loader = None
        
        # Loss function
        if self.use_focal_loss:
            criterion = FocalLoss(alpha=0.25, gamma=2.0)
        else:
            criterion = nn.BCELoss()
        
        # Optimizer with weight decay
        optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Learning rate scheduler with warmup and cosine annealing
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            else:
                return 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (epochs - warmup_epochs)))
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        # Early stopping
        best_val_loss = float('inf')
        best_val_auc = 0.0
        patience_counter = 0
        no_improvement_epochs = 0
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_preds = []
            train_targets = []
            
            for batch in train_loader:
                sequence_features = batch['sequence_features'].to(self.device, non_blocking=True)
                structural_features = batch['structural_features'].to(self.device, non_blocking=True)
                labels = batch['label'].to(self.device, non_blocking=True)
                
                optimizer.zero_grad()
                
                if self.use_mixed_precision:
                    with autocast():
                        outputs = self.model(sequence_features, structural_features)
                        loss = criterion(outputs['viability'], labels)
                    
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip_val)
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    outputs = self.model(sequence_features, structural_features)
                    loss = criterion(outputs['viability'], labels)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip_val)
                    optimizer.step()
                
                train_loss += loss.item()
                train_preds.extend(outputs['viability'].detach().cpu().numpy())
                train_targets.extend(labels.cpu().numpy())
            
            train_loss /= len(train_loader)
            train_preds = np.array(train_preds)
            train_targets = np.array(train_targets)
            train_preds_binary = (train_preds > 0.5).astype(int)
            
            # Calculate metrics
            train_acc = accuracy_score(train_targets, train_preds_binary)
            train_f1 = f1_score(train_targets, train_preds_binary, zero_division=0)
            train_auc = roc_auc_score(train_targets, train_preds) if len(np.unique(train_targets)) > 1 else 0.0
            
            self.train_history['loss'].append(train_loss)
            self.train_history['accuracy'].append(train_acc)
            self.train_history['f1'].append(train_f1)
            self.train_history['auc'].append(train_auc)
            
            # Update learning rate
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            
            # Validation phase
            if val_loader:
                self.model.eval()
                val_loss = 0.0
                val_preds = []
                val_targets = []
                
                with torch.no_grad():
                    for batch in val_loader:
                        sequence_features = batch['sequence_features'].to(self.device, non_blocking=True)
                        structural_features = batch['structural_features'].to(self.device, non_blocking=True)
                        labels = batch['label'].to(self.device, non_blocking=True)
                        
                        if self.use_mixed_precision:
                            with autocast():
                                outputs = self.model(sequence_features, structural_features)
                                loss = criterion(outputs['viability'], labels)
                        else:
                            outputs = self.model(sequence_features, structural_features)
                            loss = criterion(outputs['viability'], labels)
                        
                        val_loss += loss.item()
                        val_preds.extend(outputs['viability'].cpu().numpy())
                        val_targets.extend(labels.cpu().numpy())
                
                val_loss /= len(val_loader)
                val_preds = np.array(val_preds)
                val_targets = np.array(val_targets)
                val_preds_binary = (val_preds > 0.5).astype(int)
                
                # Calculate metrics
                val_acc = accuracy_score(val_targets, val_preds_binary)
                val_precision = precision_score(val_targets, val_preds_binary, zero_division=0)
                val_recall = recall_score(val_targets, val_preds_binary, zero_division=0)
                val_f1 = f1_score(val_targets, val_preds_binary, zero_division=0)
                val_auc = roc_auc_score(val_targets, val_preds) if len(np.unique(val_targets)) > 1 else 0.0
                
                self.val_history['loss'].append(val_loss)
                self.val_history['accuracy'].append(val_acc)
                self.val_history['f1'].append(val_f1)
                self.val_history['auc'].append(val_auc)
                
                # Log metrics
                logger.info(
                    f"Epoch {epoch+1}/{epochs} | LR: {current_lr:.2e} | "
                    f"Train: Loss={train_loss:.4f}, Acc={train_acc:.4f}, F1={train_f1:.4f}, AUC={train_auc:.4f} | "
                    f"Val: Loss={val_loss:.4f}, Acc={val_acc:.4f}, F1={val_f1:.4f}, AUC={val_auc:.4f}, "
                    f"P={val_precision:.4f}, R={val_recall:.4f}"
                )
                
                # Save best model based on validation AUC (better metric than loss)
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_val_loss = val_loss
                    self.save_model(self.config.models_dir / "viability_model_best.pth")
                    patience_counter = 0
                    logger.info(f"  → New best model saved! (AUC: {val_auc:.4f})")
                else:
                    patience_counter += 1
                    no_improvement_epochs += 1
                
                # Early stopping
                if early_stopping_patience > 0 and no_improvement_epochs >= early_stopping_patience:
                    logger.info(f"Early stopping triggered after {epoch+1} epochs (no improvement for {no_improvement_epochs} epochs)")
                    break
            else:
                logger.info(
                    f"Epoch {epoch+1}/{epochs} | LR: {current_lr:.2e} | "
                    f"Train: Loss={train_loss:.4f}, Acc={train_acc:.4f}, F1={train_f1:.4f}, AUC={train_auc:.4f}"
                )
        
        logger.info("Training complete!")
        if val_loader:
            logger.info(f"Best validation AUC: {best_val_auc:.4f}")
            logger.info(f"Best validation loss: {best_val_loss:.4f}")
        
        return self.model
    
    def save_model(self, path: Path):
        """Save model with architecture parameters"""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_class': 'ViabilityPredictor',
            'model_params': self.model_params,
            'train_history': self.train_history,
            'val_history': self.val_history
        }, path)
        logger.info(f"Model saved to: {path}")
    
    def load_model(self, path: Path):
        """Load model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        logger.info(f"Model loaded from: {path}")

def collect_training_data(data_dir: Path) -> tuple:
    """
    Collect training data from data directory
    
    Expected structure:
    data/
      train/
        viable/
          *.fasta
        non_viable/
          *.fasta
      val/
        viable/
          *.fasta
        non_viable/
          *.fasta
    """
    from ..annotation.annotator import GenomeAnnotator
    from ..core.config import VLabConfig
    
    config = VLabConfig()
    annotator = GenomeAnnotator(config)
    
    train_annotations = []
    train_labels = []
    val_annotations = []
    val_labels = []
    
    # Collect training data
    viable_dir = data_dir / "train" / "viable"
    non_viable_dir = data_dir / "train" / "non_viable"
    
    if viable_dir.exists():
        for fasta_file in viable_dir.glob("*.fasta"):
            try:
                annotation = annotator.annotate(fasta_file)
                train_annotations.append(annotation)
                train_labels.append(1.0)  # Viable
            except Exception as e:
                logger.warning(f"Error processing {fasta_file}: {e}")
    
    if non_viable_dir.exists():
        for fasta_file in non_viable_dir.glob("*.fasta"):
            try:
                annotation = annotator.annotate(fasta_file)
                train_annotations.append(annotation)
                train_labels.append(0.0)  # Non-viable
            except Exception as e:
                logger.warning(f"Error processing {fasta_file}: {e}")
    
    # Collect validation data
    val_viable_dir = data_dir / "val" / "viable"
    val_non_viable_dir = data_dir / "val" / "non_viable"
    
    if val_viable_dir.exists():
        for fasta_file in val_viable_dir.glob("*.fasta"):
            try:
                annotation = annotator.annotate(fasta_file)
                val_annotations.append(annotation)
                val_labels.append(1.0)
            except Exception as e:
                logger.warning(f"Error processing {fasta_file}: {e}")
    
    if val_non_viable_dir.exists():
        for fasta_file in val_non_viable_dir.glob("*.fasta"):
            try:
                annotation = annotator.annotate(fasta_file)
                val_annotations.append(annotation)
                val_labels.append(0.0)
            except Exception as e:
                logger.warning(f"Error processing {fasta_file}: {e}")
    
    logger.info(f"Collected {len(train_annotations)} training samples and {len(val_annotations)} validation samples")
    
    return (train_annotations, train_labels, val_annotations, val_labels)

def main():
    """Main training script"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train viability prediction model')
    parser.add_argument('--data_dir', type=str, default='data/training', help='Training data directory')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    config = VLabConfig.from_file(Path(args.config)) if Path(args.config).exists() else VLabConfig()
    
    # Collect data
    data_dir = Path(args.data_dir)
    train_annotations, train_labels, val_annotations, val_labels = collect_training_data(data_dir)
    
    if not train_annotations:
        logger.error("No training data found!")
        return
    
    # Train
    trainer = ViabilityTrainer(config)
    trainer.train(
        train_annotations, train_labels,
        val_annotations if val_annotations else None,
        val_labels if val_labels else None,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
    
    # Save final model
    trainer.save_model(config.models_dir / "viability_model_final.pth")

if __name__ == '__main__':
    main()

