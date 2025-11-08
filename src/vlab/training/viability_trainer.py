"""
Training Pipeline for Viability Prediction Model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Dict, Any
import logging
import json

from ..viability.predictor import ViabilityPredictor
from ..viability.features import FeatureExtractor
from ..core.config import VLabConfig

logger = logging.getLogger(__name__)

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
    """Train viability prediction model"""
    
    def __init__(self, config: VLabConfig):
        self.config = config
        self.device = torch.device(f"cuda:{config.gpu_id}" if config.use_gpu and torch.cuda.is_available() else "cpu")
        self.model = ViabilityPredictor().to(self.device)
        self.feature_extractor = FeatureExtractor(config)
    
    def train(self, train_annotations: List[Dict[str, Any]], 
             train_labels: List[float],
             val_annotations: List[Dict[str, Any]] = None,
             val_labels: List[float] = None,
             epochs: int = 50,
             batch_size: int = 32,
             learning_rate: float = 1e-4):
        """Train the model"""
        logger.info("Starting training...")
        
        # Create datasets
        train_dataset = ViabilityDataset(train_annotations, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        if val_annotations:
            val_dataset = ViabilityDataset(val_annotations, val_labels)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        else:
            val_loader = None
        
        # Loss and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            
            for batch in train_loader:
                sequence_features = batch['sequence_features'].to(self.device)
                structural_features = batch['structural_features'].to(self.device)
                labels = batch['label'].to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(sequence_features, structural_features)
                loss = criterion(outputs['viability'], labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation
            if val_loader:
                self.model.eval()
                val_loss = 0.0
                
                with torch.no_grad():
                    for batch in val_loader:
                        sequence_features = batch['sequence_features'].to(self.device)
                        structural_features = batch['structural_features'].to(self.device)
                        labels = batch['label'].to(self.device)
                        
                        outputs = self.model(sequence_features, structural_features)
                        loss = criterion(outputs['viability'], labels)
                        val_loss += loss.item()
                
                val_loss /= len(val_loader)
                scheduler.step(val_loss)
                
                logger.info(f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_model(self.config.models_dir / "viability_model_best.pth")
            else:
                logger.info(f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.4f}")
        
        logger.info("Training complete!")
        return self.model
    
    def save_model(self, path: Path):
        """Save model"""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_class': 'ViabilityPredictor'
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
    
    # Setup
    setup_logger(level=logging.INFO)
    config = VLabConfig.from_file(Path(args.config))
    
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

