"""
Environmental Dynamics Module
Phase 3: Environmental stability and survival analysis
"""

import logging
from typing import Dict, List
import numpy as np
from scipy import stats

class EnvironmentalAnalyzer:
    """Analyze environmental dynamics and stability"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger('VSim.EnvironmentalAnalyzer')
        self.temp_range = config.get('environmental.temperature_range', [4, 40])
        self.ph_range = config.get('environmental.ph_range', [5.0, 9.0])
    
    def analyze(self, genome_results: Dict, structure_results: Dict) -> Dict:
        """Perform comprehensive environmental analysis"""
        self.logger.info("Starting environmental dynamics analysis...")
        
        results = {
            'thermal_stability': self._analyze_thermal_stability(genome_results, structure_results),
            'ph_stability': self._analyze_ph_stability(genome_results, structure_results),
            'environmental_survival': self._analyze_survival(genome_results, structure_results),
            'resistance_factors': self._analyze_resistance(genome_results, structure_results),
            'transmission_potential': self._analyze_transmission(genome_results, structure_results)
        }
        
        self.logger.info("Environmental analysis complete")
        return results
    
    def _analyze_thermal_stability(self, genome_results: Dict, structure_results: Dict) -> Dict:
        """Analyze thermal stability"""
        gc_content = genome_results.get('gc_content', 0.5)
        proteins = genome_results.get('proteins', [])
        
        # Higher GC content generally increases thermal stability
        base_stability = 0.5 + (gc_content - 0.5) * 0.3
        
        # Analyze protein stability
        avg_hydrophobicity = 0.0
        if structure_results.get('structures'):
            hydrophobicity_values = [
                s['surface_properties']['hydrophobicity_index'] 
                for s in structure_results['structures']
            ]
            avg_hydrophobicity = np.mean(hydrophobicity_values) if hydrophobicity_values else 0.5
        
        # More hydrophobic proteins tend to be more stable
        stability_adjustment = (avg_hydrophobicity - 0.5) * 0.2
        
        thermal_stability = base_stability + stability_adjustment
        thermal_stability = max(0.3, min(0.95, thermal_stability))
        
        # Estimate melting temperature
        # Simplified formula based on GC content
        tm = 64.9 + 0.41 * (gc_content * 100) - 675 / genome_results.get('length', 1000)
        
        return {
            'stability_score': round(thermal_stability, 3),
            'melting_temperature': round(tm, 2),
            'optimal_temperature': round(37 + (thermal_stability - 0.5) * 20, 2),
            'temperature_range': [round(tm - 20, 2), round(tm + 20, 2)]
        }
    
    def _analyze_ph_stability(self, genome_results: Dict, structure_results: Dict) -> Dict:
        """Analyze pH stability"""
        proteins = genome_results.get('proteins', [])
        
        if not proteins:
            return {
                'stability_score': 0.5,
                'optimal_ph': 7.0,
                'ph_range': [6.0, 8.0]
            }
        
        # Calculate average isoelectric point
        pI_values = [p.get('isoelectric_point', 7.0) for p in proteins]
        avg_pI = np.mean(pI_values)
        
        # Estimate pH stability range around pI
        ph_stability_range = [max(4.0, avg_pI - 2.0), min(10.0, avg_pI + 2.0)]
        
        # Calculate stability score
        ph_range_width = ph_stability_range[1] - ph_stability_range[0]
        stability_score = min(ph_range_width / 6.0, 1.0)
        
        return {
            'stability_score': round(stability_score, 3),
            'optimal_ph': round(avg_pI, 2),
            'ph_range': [round(x, 2) for x in ph_stability_range],
            'isoelectric_point': round(avg_pI, 2)
        }
    
    def _analyze_survival(self, genome_results: Dict, structure_results: Dict) -> Dict:
        """Analyze survival time without host"""
        thermal = self._analyze_thermal_stability(genome_results, structure_results)
        ph = self._analyze_ph_stability(genome_results, structure_results)
        
        # Base survival time (hours)
        base_survival = 24.0
        
        # Adjust based on stability
        thermal_factor = thermal['stability_score']
        ph_factor = ph['stability_score']
        
        # More stable = longer survival
        survival_time = base_survival * (1 + thermal_factor * 2 + ph_factor * 1.5)
        
        # Adjust for genome size (larger genomes may be more fragile)
        length_factor = 1.0 - min(genome_results.get('length', 10000) / 500000, 0.3)
        survival_time *= length_factor
        
        # Estimate survival in different conditions
        survival_conditions = {
            'room_temperature': round(survival_time, 2),
            'refrigerated': round(survival_time * 3, 2),
            'frozen': round(survival_time * 100, 2),
            'desiccated': round(survival_time * 0.5, 2),
            'aqueous_solution': round(survival_time * 1.2, 2)
        }
        
        return {
            'average_survival_time_hours': round(survival_time, 2),
            'survival_conditions': survival_conditions,
            'stability_class': self._classify_stability(survival_time)
        }
    
    def _classify_stability(self, survival_time: float) -> str:
        """Classify stability"""
        if survival_time < 12:
            return "Labile"
        elif survival_time < 48:
            return "Moderate"
        elif survival_time < 168:  # 1 week
            return "Stable"
        else:
            return "Very Stable"
    
    def _analyze_resistance(self, genome_results: Dict, structure_results: Dict) -> Dict:
        """Analyze resistance to environmental factors"""
        results = {
            'uv_resistance': self._estimate_uv_resistance(genome_results),
            'chemical_resistance': self._estimate_chemical_resistance(structure_results),
            'enzyme_resistance': self._estimate_enzyme_resistance(structure_results)
        }
        
        return results
    
    def _estimate_uv_resistance(self, genome_results: Dict) -> float:
        """Estimate UV resistance"""
        # GC content affects UV resistance
        gc_content = genome_results.get('gc_content', 0.5)
        
        # Higher GC content = more UV resistant (due to G-C base pairs)
        resistance = 0.3 + gc_content * 0.4
        
        return round(min(max(resistance, 0.1), 0.9), 3)
    
    def _estimate_chemical_resistance(self, structure_results: Dict) -> float:
        """Estimate chemical resistance"""
        structures = structure_results.get('structures', [])
        
        if not structures:
            return 0.5
        
        # More compact structures = more resistant
        avg_confidence = np.mean([s['confidence'] for s in structures]) if structures else 0.7
        
        # Higher confidence structures likely more stable
        resistance = 0.4 + avg_confidence * 0.4
        
        return round(min(max(resistance, 0.2), 0.9), 3)
    
    def _estimate_enzyme_resistance(self, structure_results: Dict) -> float:
        """Estimate enzyme resistance"""
        structures = structure_results.get('structures', [])
        
        if not structures:
            return 0.5
        
        # Analyze surface properties
        avg_hydrophobicity = 0.5
        if structures:
            hydrophobicity_values = [
                s['surface_properties']['hydrophobicity_index']
                for s in structures
            ]
            avg_hydrophobicity = np.mean(hydrophobicity_values) if hydrophobicity_values else 0.5
        
        # More hydrophobic = less accessible to enzymes
        resistance = 0.3 + avg_hydrophobicity * 0.5
        
        return round(min(max(resistance, 0.2), 0.85), 3)
    
    def _analyze_transmission(self, genome_results: Dict, structure_results: Dict) -> Dict:
        """Analyze transmission potential"""
        survival = self._analyze_survival(genome_results, structure_results)
        resistance = self._analyze_resistance(genome_results, structure_results)
        
        # Combine factors
        stability_score = survival['average_survival_time_hours'] / 168.0  # Normalize to 1 week
        resistance_score = (resistance['uv_resistance'] + 
                          resistance['chemical_resistance'] + 
                          resistance['enzyme_resistance']) / 3.0
        
        transmission_potential = (stability_score * 0.6 + resistance_score * 0.4)
        transmission_potential = min(max(transmission_potential, 0.0), 1.0)
        
        return {
            'transmission_potential': round(transmission_potential, 3),
            'airborne_potential': round(transmission_potential * 0.8, 3),
            'surface_potential': round(transmission_potential * 1.2, 3),
            'waterborne_potential': round(transmission_potential * 0.9, 3)
        }

