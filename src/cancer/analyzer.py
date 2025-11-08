"""
Cancer Cell Analysis Module
Phase 5: Oncolytic potential and cancer cell targeting
"""

import logging
from typing import Dict, List
import numpy as np

class CancerAnalyzer:
    """Analyze oncolytic potential and cancer cell targeting"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger('VSim.CancerAnalyzer')
        self.cancer_cell_types = config.get('cancer_analysis.cancer_cell_types', 
                                           ['HeLa', 'MCF-7', 'A549', 'HepG2'])
        self.selectivity_threshold = config.get('cancer_analysis.selectivity_threshold', 0.8)
    
    def analyze(self, genome_results: Dict, structure_results: Dict, 
                cell_results: Dict) -> Dict:
        """Perform comprehensive cancer cell analysis"""
        self.logger.info("Starting cancer cell analysis...")
        
        results = {
            'oncolytic_potential': self._assess_oncolytic_potential(genome_results, structure_results, cell_results),
            'cancer_cell_targeting': self._analyze_cancer_targeting(structure_results, cell_results),
            'selectivity': self._calculate_selectivity(cell_results),
            'efficacy_prediction': self._predict_efficacy(genome_results, structure_results, cell_results),
            'safety_assessment': self._assess_safety(cell_results)
        }
        
        self.logger.info("Cancer cell analysis complete")
        return results
    
    def _assess_oncolytic_potential(self, genome_results: Dict, structure_results: Dict,
                                   cell_results: Dict) -> Dict:
        """Assess oncolytic virus potential"""
        entry_efficiency = cell_results.get('cell_entry_mechanisms', {}).get('entry_efficiency', 0.5)
        receptor_binding = cell_results.get('receptor_binding', {}).get('binding_score', 0.5)
        
        # Oncolytic viruses need efficient entry
        oncolytic_score = entry_efficiency * 0.6 + receptor_binding * 0.4
        
        # Check for replication factors
        genome_length = genome_results.get('length', 10000)
        protein_count = len(genome_results.get('proteins', []))
        
        # Larger genomes with more proteins might indicate replication capability
        replication_score = min(protein_count / 10.0, 1.0)
        
        overall_potential = oncolytic_score * 0.7 + replication_score * 0.3
        
        return {
            'oncolytic_score': round(overall_potential, 3),
            'potential_classification': self._classify_oncolytic_potential(overall_potential),
            'replication_capability': round(replication_score, 3),
            'entry_efficiency': round(entry_efficiency, 3)
        }
    
    def _classify_oncolytic_potential(self, score: float) -> str:
        """Classify oncolytic potential"""
        if score < 0.4:
            return "Low"
        elif score < 0.6:
            return "Moderate"
        elif score < 0.8:
            return "High"
        else:
            return "Very High"
    
    def _analyze_cancer_targeting(self, structure_results: Dict, cell_results: Dict) -> Dict:
        """Analyze cancer cell targeting"""
        structures = structure_results.get('structures', [])
        receptor_binding = cell_results.get('receptor_binding', {})
        
        targeting_results = {}
        
        for cell_type in self.cancer_cell_types:
            # Simplified targeting prediction
            # In production, would use actual receptor expression data
            
            binding_score = receptor_binding.get('binding_score', 0.5)
            
            # Different cell types might have different receptor profiles
            cell_specific_score = binding_score
            
            # Adjust based on cell type characteristics
            if cell_type == 'HeLa':
                cell_specific_score *= 1.1  # Often more permissive
            elif cell_type == 'MCF-7':
                cell_specific_score *= 0.95
            
            targeting_results[cell_type] = {
                'targeting_score': round(min(cell_specific_score, 1.0), 3),
                'receptor_affinity': round(binding_score, 3),
                'entry_probability': round(cell_specific_score * 0.8, 3)
            }
        
        # Calculate average targeting
        avg_targeting = np.mean([r['targeting_score'] for r in targeting_results.values()])
        
        return {
            'cell_type_analysis': targeting_results,
            'average_targeting': round(avg_targeting, 3),
            'targeting_efficiency': round(avg_targeting * 0.9, 3)
        }
    
    def _calculate_selectivity(self, cell_results: Dict) -> Dict:
        """Calculate selectivity for cancer vs normal cells"""
        host_specificity = cell_results.get('host_specificity', {})
        specificity_score = host_specificity.get('specificity_score', 0.5)
        
        # High specificity might indicate selectivity
        # But need to balance - too specific might miss targets
        
        # Simplified selectivity calculation
        # In production, would compare cancer vs normal cell receptor profiles
        selectivity = 0.5 + (specificity_score - 0.5) * 0.5
        
        return {
            'selectivity_score': round(selectivity, 3),
            'cancer_vs_normal': round(selectivity * 1.2, 3),  # Preferentially targets cancer
            'selectivity_class': 'High' if selectivity > self.selectivity_threshold else 'Moderate' if selectivity > 0.5 else 'Low'
        }
    
    def _predict_efficacy(self, genome_results: Dict, structure_results: Dict,
                         cell_results: Dict) -> Dict:
        """Predict therapeutic efficacy"""
        oncolytic = self._assess_oncolytic_potential(genome_results, structure_results, cell_results)
        targeting = self._analyze_cancer_targeting(structure_results, cell_results)
        selectivity = self._calculate_selectivity(cell_results)
        
        # Combine factors
        efficacy = (
            oncolytic['oncolytic_score'] * 0.4 +
            targeting['average_targeting'] * 0.3 +
            selectivity['selectivity_score'] * 0.3
        )
        
        # Estimate therapeutic window
        therapeutic_window = selectivity['cancer_vs_normal'] / max(0.1, 1 - selectivity['selectivity_score'])
        
        return {
            'efficacy_score': round(efficacy, 3),
            'therapeutic_window': round(therapeutic_window, 2),
            'dose_requirement': self._estimate_dose(efficacy),
            'treatment_duration': self._estimate_treatment_duration(efficacy)
        }
    
    def _estimate_dose(self, efficacy: float) -> str:
        """Estimate required dose"""
        if efficacy > 0.8:
            return "Low"
        elif efficacy > 0.6:
            return "Moderate"
        elif efficacy > 0.4:
            return "High"
        else:
            return "Very High"
    
    def _estimate_treatment_duration(self, efficacy: float) -> str:
        """Estimate treatment duration"""
        if efficacy > 0.8:
            return "1-2 weeks"
        elif efficacy > 0.6:
            return "2-4 weeks"
        elif efficacy > 0.4:
            return "4-8 weeks"
        else:
            return "8+ weeks"
    
    def _assess_safety(self, cell_results: Dict) -> Dict:
        """Assess safety profile"""
        host_specificity = cell_results.get('host_specificity', {})
        tropism = cell_results.get('tropism', {})
        
        # More specific = potentially safer
        specificity = host_specificity.get('specificity_score', 0.5)
        
        # Limited tropism = safer
        tropism_score = tropism.get('tropism_score', 0.5)
        
        safety_score = specificity * 0.6 + (1 - tropism_score) * 0.4
        
        return {
            'safety_score': round(safety_score, 3),
            'off_target_risk': round(1 - safety_score, 3),
            'safety_class': 'High' if safety_score > 0.7 else 'Moderate' if safety_score > 0.5 else 'Low',
            'recommended_monitoring': self._recommend_monitoring(safety_score)
        }
    
    def _recommend_monitoring(self, safety_score: float) -> List[str]:
        """Recommend safety monitoring"""
        monitoring = []
        
        if safety_score < 0.7:
            monitoring.append('Continuous vital signs monitoring')
            monitoring.append('Regular blood tests')
        
        if safety_score < 0.5:
            monitoring.append('ICU monitoring')
            monitoring.append('Liver function tests')
            monitoring.append('Immune system monitoring')
        
        if not monitoring:
            monitoring.append('Standard clinical monitoring')
        
        return monitoring

