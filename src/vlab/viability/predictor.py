"""
Viability Predictor - Determines if a viral genome will synthesize a functional virus
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

from ..core.config import VLabConfig
from ..viability.features import FeatureExtractor

logger = logging.getLogger(__name__)

class ViabilityPredictor(nn.Module):
    """
    Deep learning model for predicting viral genome viability
    Architecture: Transformer-based encoder + MLP classifier
    Enhanced version with larger capacity for maximum quality
    """
    
    def __init__(self, input_dim: int = 1024, hidden_dim: int = 512, num_layers: int = 8, num_heads: int = 16):
        super().__init__()
        
        # Transformer encoder for sequence features (larger and deeper)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,  # Larger feedforward
            dropout=0.1,
            batch_first=True,
            activation='gelu'  # GELU activation for better performance
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(input_dim)
        
        # Enhanced feature fusion with residual connections
        self.feature_fusion = nn.Sequential(
            nn.Linear(input_dim + 128, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # Deeper classifier with residual-style connections
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # Enhanced confidence estimator
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, sequence_features: torch.Tensor, 
                structural_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            sequence_features: [batch_size, seq_len, input_dim]
            structural_features: [batch_size, 128]
        
        Returns:
            Dictionary with 'viability' and 'confidence' scores
        """
        # Encode sequence
        encoded = self.encoder(sequence_features)
        
        # Apply layer normalization
        encoded = self.layer_norm(encoded)
        
        # Multi-head attention pooling (better than mean pooling)
        # Use both mean and max pooling for richer representation
        mean_pooled = encoded.mean(dim=1)  # [batch_size, input_dim]
        max_pooled = encoded.max(dim=1)[0]  # [batch_size, input_dim]
        pooled = (mean_pooled + max_pooled) / 2  # Combined pooling
        
        # Fuse features
        fused = torch.cat([pooled, structural_features], dim=1)
        fused_features = self.feature_fusion(fused)
        
        # Predict viability
        viability = self.classifier(fused_features)
        
        # Estimate confidence
        confidence = self.confidence_head(fused_features)
        
        return {
            'viability': viability.squeeze(-1),
            'confidence': confidence.squeeze(-1)
        }

class ViabilityPredictorWrapper:
    """Wrapper for viability prediction with model loading and inference"""
    
    def __init__(self, config: VLabConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.feature_extractor = FeatureExtractor(config)
        self.model = None
        self.device = torch.device(f"cuda:{config.gpu_id}" if config.use_gpu and torch.cuda.is_available() else "cpu")
        
        # Load model if available
        self._load_model()
    
    def _load_model(self):
        """Load trained model with enhanced architecture support"""
        model_path = self.config.viability_model_path
        
        if model_path and Path(model_path).exists():
            try:
                self.logger.info(f"Loading viability model from: {model_path}")
                checkpoint = torch.load(model_path, map_location=self.device)
                
                # Try to load architecture params from checkpoint, otherwise use defaults
                # Enhanced model uses: input_dim=1024, hidden_dim=512, num_layers=8, num_heads=16
                # Old model uses: input_dim=512, hidden_dim=256, num_layers=4, num_heads=8
                model_params = checkpoint.get('model_params', {})
                input_dim = model_params.get('input_dim', 1024)  # Default to enhanced
                hidden_dim = model_params.get('hidden_dim', 512)
                num_layers = model_params.get('num_layers', 8)
                num_heads = model_params.get('num_heads', 16)
                
                # Initialize model with correct architecture
                self.model = ViabilityPredictor(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                    num_heads=num_heads
                )
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.to(self.device)
                self.model.eval()
                
                self.logger.info(f"Viability model loaded successfully (arch: dim={input_dim}, layers={num_layers})")
            except Exception as e:
                self.logger.warning(f"Failed to load viability model: {e}. Using rule-based fallback.")
                self.model = None
        else:
            self.logger.warning("No viability model found. Using rule-based fallback.")
            self.model = None
    
    def predict(self, annotation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict viability of viral genome
        
        Args:
            annotation: Genome annotation results
        
        Returns:
            Dictionary with viability prediction results
        """
        self.logger.info("Predicting genome viability...")
        
        # Extract features
        features = self.feature_extractor.extract(annotation)
        
        if self.model is not None:
            # Use ML model
            return self._predict_ml(features, annotation)
        else:
            # Use rule-based prediction
            return self._predict_rule_based(annotation)
    
    def _predict_ml(self, features: Dict[str, Any], annotation: Dict[str, Any]) -> Dict[str, Any]:
        """Predict using ML model"""
        with torch.no_grad():
            # Prepare inputs
            sequence_features = torch.tensor(
                features['sequence_features'], 
                dtype=torch.float32
            ).unsqueeze(0).to(self.device)
            
            structural_features = torch.tensor(
                features['structural_features'],
                dtype=torch.float32
            ).unsqueeze(0).to(self.device)
            
            # Predict
            outputs = self.model(sequence_features, structural_features)
            
            viability_score = outputs['viability'].item()
            confidence = outputs['confidence'].item()
        
        # Determine viability
        is_viable = viability_score >= self.config.viability_threshold
        is_confident = confidence >= self.config.confidence_threshold
        
        return {
            'score': float(viability_score),
            'confidence': float(confidence),
            'is_viable': bool(is_viable),
            'is_confident': bool(is_confident),
            'method': 'ml_model',
            'features': {
                'gene_completeness': features.get('gene_completeness', 0.0),
                'codon_usage': features.get('codon_usage_bias', 0.0),
                'regulatory_elements': features.get('regulatory_elements_score', 0.0)
            },
            'reasons': self._get_viability_reasons(annotation, viability_score, is_viable)
        }
    
    def _predict_rule_based(self, annotation: Dict[str, Any]) -> Dict[str, Any]:
        """Rule-based viability prediction (fallback)"""
        reasons = []
        score = 0.0
        
        # Check essential genes
        essential_genes = self._check_essential_genes(annotation)
        if essential_genes['complete']:
            score += 0.4
            reasons.append("✓ Essential genes present")
        else:
            reasons.append(f"✗ Missing essential genes: {essential_genes['missing']}")
        
        # Check gene completeness
        gene_completeness = annotation.get('gene_completeness', {})
        completeness_score = gene_completeness.get('score', 0.0)
        score += completeness_score * 0.3
        if completeness_score > 0.8:
            reasons.append("✓ High gene completeness")
        
        # Check codon usage
        codon_usage = annotation.get('codon_usage', {})
        if codon_usage.get('bias_score', 0.0) > 0.5:
            score += 0.2
            reasons.append("✓ Reasonable codon usage")
        
        # Check regulatory elements
        regulatory = annotation.get('regulatory_elements', {})
        if regulatory.get('score', 0.0) > 0.6:
            score += 0.1
            reasons.append("✓ Regulatory elements detected")
        
        # Normalize score
        score = min(score, 1.0)
        is_viable = score >= self.config.viability_threshold
        
        return {
            'score': float(score),
            'confidence': float(min(score * 1.2, 1.0)),  # Lower confidence for rule-based
            'is_viable': bool(is_viable),
            'is_confident': False,
            'method': 'rule_based',
            'reasons': reasons
        }
    
    def _check_essential_genes(self, annotation: Dict[str, Any]) -> Dict[str, Any]:
        """Check for essential viral genes"""
        predicted_genes = annotation.get('genes', [])
        gene_functions = {gene.get('function', '').lower() for gene in predicted_genes}
        
        # Essential genes vary by virus type
        virus_type = annotation.get('virus_type', 'unknown').lower()
        
        essential_required = set()
        if 'rna' in virus_type:
            essential_required.add('polymerase')
            essential_required.add('replicase')
        if 'dna' in virus_type:
            essential_required.add('polymerase')
        
        # Always need capsid
        essential_required.add('capsid')
        essential_required.add('nucleocapsid')
        
        # Check for envelope
        is_enveloped = annotation.get('is_enveloped', False)
        if is_enveloped:
            essential_required.add('envelope')
            essential_required.add('glycoprotein')
        
        # Find missing genes
        missing = []
        for required in essential_required:
            found = any(required in func for func in gene_functions)
            if not found:
                missing.append(required)
        
        return {
            'complete': len(missing) == 0,
            'missing': missing,
            'found': list(gene_functions)
        }
    
    def _get_viability_reasons(self, annotation: Dict[str, Any], 
                               score: float, is_viable: bool) -> List[str]:
        """Get human-readable reasons for viability prediction"""
        reasons = []
        
        if is_viable:
            reasons.append(f"Genome predicted as VIABLE (score: {score:.3f})")
            reasons.append("All essential viral components appear to be present")
        else:
            reasons.append(f"Genome predicted as NON-VIABLE (score: {score:.3f})")
            reasons.append("One or more essential components may be missing or non-functional")
        
        # Add specific checks
        essential_check = self._check_essential_genes(annotation)
        if not essential_check['complete']:
            reasons.append(f"Missing genes: {', '.join(essential_check['missing'])}")
        
        return reasons

# For backward compatibility
ViabilityPredictor = ViabilityPredictorWrapper

