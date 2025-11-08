"""
Infection Simulator - Simulates viral infection dynamics
"""

import logging
from typing import Dict, Any
import numpy as np

logger = logging.getLogger(__name__)

class InfectionSimulator:
    """Simulate viral infection dynamics"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def simulate(self, annotation: Dict[str, Any], 
                structures: Dict[str, Any],
                assembly: Dict[str, Any],
                host_prediction: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate infection dynamics"""
        self.logger.info("Simulating infection dynamics...")
        
        # Estimate replication parameters
        replication_params = self._estimate_replication_params(annotation, structures)
        
        # Simulate life cycle
        lifecycle = self._simulate_lifecycle(replication_params)
        
        # Calculate metrics
        metrics = self._calculate_metrics(lifecycle)
        
        return {
            'replication_params': replication_params,
            'lifecycle': lifecycle,
            'metrics': metrics,
            'burst_size': metrics.get('burst_size', 0),
            'replication_time': metrics.get('replication_time', 0.0)
        }
    
    def _estimate_replication_params(self, annotation: Dict[str, Any],
                                    structures: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate replication parameters"""
        genome_length = annotation.get('genome_length', 10000)
        virus_type = annotation.get('virus_type', 'unknown').lower()
        
        # Estimate replication rate based on virus type
        if 'rna' in virus_type:
            replication_rate = 0.1  # per hour
            translation_rate = 0.5  # proteins per hour
        else:
            replication_rate = 0.05  # per hour
            translation_rate = 0.3  # proteins per hour
        
        # Estimate assembly efficiency
        assembly_efficiency = 0.7
        
        # Estimate burst size
        if genome_length < 5000:
            burst_size = 1000
        elif genome_length < 15000:
            burst_size = 500
        else:
            burst_size = 100
        
        return {
            'replication_rate': replication_rate,
            'translation_rate': translation_rate,
            'assembly_efficiency': assembly_efficiency,
            'burst_size': burst_size,
            'latent_period': 6.0  # hours
        }
    
    def _simulate_lifecycle(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate viral life cycle"""
        time_points = np.arange(0, self.config.simulation_time, self.config.time_step)
        
        # Initial conditions
        viral_rna = [1.0]  # Starting with 1 genome
        proteins = [0.0]
        virions = [0.0]
        
        replication_rate = params['replication_rate']
        translation_rate = params['translation_rate']
        assembly_efficiency = params['assembly_efficiency']
        burst_size = params['burst_size']
        latent_period = params['latent_period']
        
        for t in time_points[1:]:
            # Replication phase
            if t < latent_period:
                # Latent period - slow replication
                drna = viral_rna[-1] * replication_rate * 0.1 * self.config.time_step
                dprotein = viral_rna[-1] * translation_rate * 0.1 * self.config.time_step
            else:
                # Active replication
                drna = viral_rna[-1] * replication_rate * self.config.time_step
                dprotein = viral_rna[-1] * translation_rate * self.config.time_step
            
            viral_rna.append(viral_rna[-1] + drna)
            proteins.append(proteins[-1] + dprotein)
            
            # Assembly (only after enough proteins)
            if proteins[-1] > 100 and t > latent_period:
                dvirions = min(proteins[-1] / 10, burst_size) * assembly_efficiency * self.config.time_step
                virions.append(virions[-1] + dvirions)
            else:
                virions.append(virions[-1])
        
        return {
            'time_points': time_points.tolist(),
            'viral_rna': viral_rna,
            'proteins': proteins,
            'virions': virions
        }
    
    def _calculate_metrics(self, lifecycle: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate infection metrics"""
        virions = lifecycle['virions']
        time_points = lifecycle['time_points']
        
        # Burst size
        burst_size = max(virions) if virions else 0
        
        # Replication time (time to reach 50% of max)
        max_virions = max(virions) if virions else 1
        half_max = max_virions / 2
        
        replication_time = 0.0
        for i, v in enumerate(virions):
            if v >= half_max:
                replication_time = time_points[i]
                break
        
        # Peak viral load
        peak_load = max(virions)
        
        return {
            'burst_size': int(burst_size),
            'replication_time': replication_time,
            'peak_viral_load': peak_load,
            'latent_period': 6.0  # Estimated
        }

