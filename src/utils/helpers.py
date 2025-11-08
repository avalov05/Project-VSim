"""
Utility Functions
"""

from typing import List, Dict, Any
import numpy as np

def validate_genome_sequence(sequence: str) -> bool:
    """Validate genome sequence"""
    valid_bases = set('ATCGN')
    sequence_upper = sequence.upper()
    return all(base in valid_bases for base in sequence_upper)

def calculate_statistics(values: List[float]) -> Dict[str, float]:
    """Calculate basic statistics"""
    if not values:
        return {
            'mean': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0,
            'median': 0.0
        }
    
    return {
        'mean': float(np.mean(values)),
        'std': float(np.std(values)),
        'min': float(np.min(values)),
        'max': float(np.max(values)),
        'median': float(np.median(values))
    }

def format_confidence_interval(mean: float, std: float, n: int = 1) -> str:
    """Format confidence interval"""
    if n == 1:
        return f"{mean:.3f} ± {std:.3f}"
    
    # 95% confidence interval
    ci = 1.96 * std / np.sqrt(n)
    return f"{mean:.3f} (95% CI: {mean - ci:.3f} - {mean + ci:.3f})"

def normalize_score(score: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Normalize score to [0, 1] range"""
    return max(min_val, min(max_val, score))

def combine_scores(scores: List[float], weights: List[float] = None) -> float:
    """Combine multiple scores with optional weights"""
    if not scores:
        return 0.0
    
    if weights is None:
        weights = [1.0] * len(scores)
    
    if len(weights) != len(scores):
        weights = [1.0] * len(scores)
    
    weighted_sum = sum(s * w for s, w in zip(scores, weights))
    total_weight = sum(weights)
    
    return weighted_sum / total_weight if total_weight > 0 else 0.0

