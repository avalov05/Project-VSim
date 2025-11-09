"""
Feature Extraction for Viability Prediction
"""

import numpy as np
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class FeatureExtractor:
    """Extract features from genome annotation for viability prediction"""
    
    def __init__(self, config=None):
        self.config = config
    
    def extract(self, annotation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract features from annotation
        
        Returns:
            Dictionary with 'sequence_features' and 'structural_features'
        """
        # Sequence-level features
        sequence_features = self._extract_sequence_features(annotation)
        
        # Structural features
        structural_features = self._extract_structural_features(annotation)
        
        return {
            'sequence_features': sequence_features,
            'structural_features': structural_features,
            'gene_completeness': annotation.get('gene_completeness', {}).get('score', 0.0),
            'codon_usage_bias': annotation.get('codon_usage', {}).get('bias_score', 0.0),
            'regulatory_elements_score': annotation.get('regulatory_elements', {}).get('score', 0.0)
        }
    
    def _extract_sequence_features(self, annotation: Dict[str, Any]) -> np.ndarray:
        """Extract sequence-level features (enhanced for larger model)"""
        genome = annotation.get('genome_sequence', '')
        
        if not genome:
            return np.zeros((100, 1024))  # Default empty features (increased to 1024)
        
        # Enhanced K-mer frequencies (multiple k values for richer features)
        kmer_features_3 = self._kmer_features(genome, k=3, max_len=100)
        kmer_features_4 = self._kmer_features(genome, k=4, max_len=100)
        
        # Nucleotide composition
        comp_features = self._composition_features(genome)
        
        # Codon usage
        codon_features = self._codon_features(annotation)
        
        # Combine features
        features = np.concatenate([
            kmer_features_3,
            kmer_features_4,
            comp_features,
            codon_features
        ], axis=1)
        
        # Pad or truncate to fixed size (increased dimensions)
        target_len = 100
        target_dim = 1024  # Increased from 512 to 1024
        
        if features.shape[0] > target_len:
            features = features[:target_len]
        else:
            padding = np.zeros((target_len - features.shape[0], features.shape[1]))
            features = np.vstack([features, padding])
        
        if features.shape[1] < target_dim:
            padding = np.zeros((features.shape[0], target_dim - features.shape[1]))
            features = np.hstack([features, padding])
        elif features.shape[1] > target_dim:
            features = features[:, :target_dim]
        
        return features.astype(np.float32)
    
    def _extract_structural_features(self, annotation: Dict[str, Any]) -> np.ndarray:
        """Extract structural/functional features"""
        features = []
        
        # Gene completeness
        gene_completeness = annotation.get('gene_completeness', {})
        features.append(gene_completeness.get('score', 0.0))
        features.append(len(annotation.get('genes', [])))
        
        # Codon usage
        codon_usage = annotation.get('codon_usage', {})
        features.append(codon_usage.get('bias_score', 0.0))
        features.append(codon_usage.get('cai', 0.0))
        
        # Regulatory elements
        regulatory = annotation.get('regulatory_elements', {})
        features.append(regulatory.get('score', 0.0))
        features.append(len(regulatory.get('promoters', [])))
        features.append(len(regulatory.get('rbs', [])))
        
        # Genome properties
        features.append(annotation.get('gc_content', 0.5))
        features.append(annotation.get('genome_length', 0) / 10000.0)  # Normalized
        features.append(1.0 if annotation.get('is_enveloped', False) else 0.0)
        features.append(1.0 if annotation.get('is_circular', False) else 0.0)
        
        # Essential genes presence
        essential_check = self._check_essential_genes(annotation)
        features.append(1.0 if essential_check['complete'] else 0.0)
        features.append(len(essential_check['missing']) / 10.0)  # Normalized
        
        # Pad to 128 dimensions
        while len(features) < 128:
            features.append(0.0)
        
        return np.array(features[:128], dtype=np.float32)
    
    def _kmer_features(self, sequence: str, k: int = 3, max_len: int = 100) -> np.ndarray:
        """Extract k-mer frequency features"""
        kmers = {}
        for i in range(len(sequence) - k + 1):
            kmer = sequence[i:i+k].upper()
            kmers[kmer] = kmers.get(kmer, 0) + 1
        
        # Normalize
        total = sum(kmers.values()) or 1
        kmers = {k: v/total for k, v in kmers.items()}
        
        # Create feature vector (all possible 3-mers for DNA = 64)
        all_kmers = [''.join([b1, b2, b3]) 
                     for b1 in 'ATCG' for b2 in 'ATCG' for b3 in 'ATCG']
        
        features = [kmers.get(kmer, 0.0) for kmer in all_kmers]
        
        # Reshape to (max_len, feature_dim)
        feature_dim = len(features)
        features = np.array(features).reshape(1, feature_dim)
        features = np.repeat(features, max_len, axis=0)
        
        return features.astype(np.float32)
    
    def _composition_features(self, sequence: str) -> np.ndarray:
        """Extract nucleotide composition features"""
        if not sequence:
            return np.zeros((100, 4))
        
        seq_upper = sequence.upper()
        total = len(seq_upper) or 1
        
        comp = {
            'A': seq_upper.count('A') / total,
            'T': seq_upper.count('T') / total,
            'C': seq_upper.count('C') / total,
            'G': seq_upper.count('G') / total
        }
        
        features = np.array([[comp['A'], comp['T'], comp['C'], comp['G']]])
        features = np.repeat(features, 100, axis=0)
        
        return features.astype(np.float32)
    
    def _codon_features(self, annotation: Dict[str, Any]) -> np.ndarray:
        """Extract codon usage features"""
        codon_usage = annotation.get('codon_usage', {})
        codon_freq = codon_usage.get('frequencies', {})
        
        # 64 possible codons
        all_codons = [''.join([b1, b2, b3]) 
                      for b1 in 'ATCG' for b2 in 'ATCG' for b3 in 'ATCG']
        
        features = [codon_freq.get(codon, 0.0) for codon in all_codons]
        features = np.array(features).reshape(1, 64)
        features = np.repeat(features, 100, axis=0)
        
        return features.astype(np.float32)
    
    def _check_essential_genes(self, annotation: Dict[str, Any]) -> Dict[str, Any]:
        """Check for essential genes (helper method)"""
        predicted_genes = annotation.get('genes', [])
        gene_functions = {gene.get('function', '').lower() for gene in predicted_genes}
        
        essential_required = {'polymerase', 'replicase', 'capsid', 'nucleocapsid'}
        
        if annotation.get('is_enveloped', False):
            essential_required.add('envelope')
            essential_required.add('glycoprotein')
        
        missing = [req for req in essential_required 
                  if not any(req in func for func in gene_functions)]
        
        return {
            'complete': len(missing) == 0,
            'missing': missing
        }

