"""
Cell Interaction Analysis Module
Phase 4: Receptor binding, cell entry, and host specificity
"""

import logging
from typing import Dict, List, Tuple
import numpy as np

class CellInteractionAnalyzer:
    """Analyze cell interactions and host specificity"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger('VSim.CellInteractionAnalyzer')
        self.binding_threshold = config.get('cell_interaction.binding_affinity_threshold', -7.0)
    
    def analyze(self, genome_results: Dict, structure_results: Dict) -> Dict:
        """Perform comprehensive cell interaction analysis"""
        self.logger.info("Starting cell interaction analysis...")
        
        results = {
            'receptor_binding': self._analyze_receptor_binding(structure_results),
            'cell_entry_mechanisms': self._analyze_entry_mechanisms(genome_results, structure_results),
            'host_specificity': self._analyze_host_specificity(genome_results, structure_results),
            'tropism': self._analyze_tropism(structure_results),
            'binding_affinity': self._calculate_binding_affinities(structure_results)
        }
        
        self.logger.info("Cell interaction analysis complete")
        return results
    
    def _analyze_receptor_binding(self, structure_results: Dict) -> Dict:
        """Analyze receptor binding potential"""
        structures = structure_results.get('structures', [])
        
        if not structures:
            return {
                'potential_receptors': [],
                'binding_score': 0.5,
                'receptor_diversity': 0.0
            }
        
        # Analyze binding sites
        all_binding_sites = []
        for structure in structures:
            binding_sites = structure.get('binding_sites', [])
            all_binding_sites.extend(binding_sites)
        
        # Predict potential receptors based on surface properties
        potential_receptors = []
        
        for structure in structures:
            surface_props = structure.get('surface_properties', {})
            charge = surface_props.get('charge_density', 0.0)
            hydrophobicity = surface_props.get('hydrophobicity_index', 0.5)
            
            # Predict receptor types based on properties
            if abs(charge) > 0.1:
                potential_receptors.append({
                    'type': 'charged_receptor',
                    'confidence': 0.6,
                    'mechanism': 'electrostatic_interaction'
                })
            
            if hydrophobicity > 0.6:
                potential_receptors.append({
                    'type': 'hydrophobic_receptor',
                    'confidence': 0.7,
                    'mechanism': 'hydrophobic_interaction'
                })
        
        # Common viral receptors
        common_receptors = [
            'ACE2', 'CD4', 'ICAM-1', 'Sialic acid', 'Heparan sulfate'
        ]
        
        binding_score = min(len(all_binding_sites) / 5.0, 1.0) if all_binding_sites else 0.3
        
        return {
            'potential_receptors': potential_receptors[:10],  # Top 10
            'common_receptors': common_receptors,
            'binding_score': round(binding_score, 3),
            'receptor_diversity': round(len(set(r['type'] for r in potential_receptors)) / 5.0, 3),
            'binding_sites_count': len(all_binding_sites)
        }
    
    def _analyze_entry_mechanisms(self, genome_results: Dict, structure_results: Dict) -> Dict:
        """Analyze cell entry mechanisms"""
        structures = structure_results.get('structures', [])
        
        entry_mechanisms = []
        
        # Look for fusion proteins
        for structure in structures:
            protein_seq = structure.get('sequence', '')
            
            # Check for fusion motifs (simplified)
            if 'G' * 3 in protein_seq or len(protein_seq) > 500:
                entry_mechanisms.append({
                    'type': 'membrane_fusion',
                    'confidence': 0.7,
                    'proteins': [structure.get('protein_id', 0)]
                })
        
        # Check for receptor-mediated entry
        receptor_binding = self._analyze_receptor_binding(structure_results)
        if receptor_binding['binding_score'] > 0.5:
            entry_mechanisms.append({
                'type': 'receptor_mediated',
                'confidence': receptor_binding['binding_score'],
                'receptors': receptor_binding.get('potential_receptors', [])
            })
        
        # Check for endocytosis (common mechanism)
        if len(structures) > 3:
            entry_mechanisms.append({
                'type': 'endocytosis',
                'confidence': 0.8,
                'mechanism': 'clathrin_mediated'
            })
        
        return {
            'mechanisms': entry_mechanisms,
            'primary_mechanism': entry_mechanisms[0]['type'] if entry_mechanisms else 'unknown',
            'entry_efficiency': self._calculate_entry_efficiency(entry_mechanisms)
        }
    
    def _calculate_entry_efficiency(self, mechanisms: List[Dict]) -> float:
        """Calculate cell entry efficiency"""
        if not mechanisms:
            return 0.3
        
        # More mechanisms = higher efficiency
        efficiency = 0.4 + len(mechanisms) * 0.15
        
        # Higher confidence mechanisms boost efficiency
        avg_confidence = np.mean([m['confidence'] for m in mechanisms])
        efficiency += avg_confidence * 0.2
        
        return round(min(max(efficiency, 0.2), 0.95), 3)
    
    def _analyze_host_specificity(self, genome_results: Dict, structure_results: Dict) -> Dict:
        """Analyze host specificity"""
        receptor_binding = self._analyze_receptor_binding(structure_results)
        entry_mechanisms = self._analyze_entry_mechanisms(genome_results, structure_results)
        
        # Calculate specificity score
        binding_score = receptor_binding['binding_score']
        entry_efficiency = entry_mechanisms['entry_efficiency']
        
        # More specific = higher binding but potentially lower efficiency
        specificity = binding_score * 0.7 + (1 - entry_efficiency) * 0.3
        
        # Predict host range
        if specificity > 0.7:
            host_range = 'Narrow'
        elif specificity > 0.4:
            host_range = 'Moderate'
        else:
            host_range = 'Broad'
        
        # Predict potential hosts
        potential_hosts = []
        if entry_efficiency > 0.6:
            potential_hosts.extend(['Human', 'Mouse', 'Primate'])
        
        return {
            'specificity_score': round(specificity, 3),
            'host_range': host_range,
            'potential_hosts': potential_hosts,
            'binding_affinity': binding_score
        }
    
    def _analyze_tropism(self, structure_results: Dict) -> Dict:
        """Analyze tissue tropism"""
        structures = structure_results.get('structures', [])
        
        if not structures:
            return {
                'tropism_score': 0.5,
                'preferred_tissues': []
            }
        
        # Analyze surface properties for tissue preference
        avg_charge = 0.0
        if structures:
            charges = [s['surface_properties'].get('charge_density', 0.0) 
                      for s in structures if 'surface_properties' in s]
            avg_charge = np.mean(charges) if charges else 0.0
        
        # Predict tissues based on properties
        preferred_tissues = []
        
        if abs(avg_charge) > 0.1:
            preferred_tissues.append('Epithelial cells')
        
        if len(structures) > 5:
            preferred_tissues.append('Immune cells')
        
        preferred_tissues.extend(['Respiratory tract', 'Gastrointestinal tract'])
        
        return {
            'tropism_score': round(min(len(preferred_tissues) / 5.0, 1.0), 3),
            'preferred_tissues': preferred_tissues,
            'tissue_specificity': 'Moderate' if len(preferred_tissues) > 2 else 'Broad'
        }
    
    def _calculate_binding_affinities(self, structure_results: Dict) -> List[Dict]:
        """Calculate binding affinities"""
        structures = structure_results.get('structures', [])
        
        affinities = []
        
        for structure in structures:
            surface_props = structure.get('surface_properties', {})
            charge = surface_props.get('charge_density', 0.0)
            hydrophobicity = surface_props.get('hydrophobicity_index', 0.5)
            
            # Simplified binding affinity calculation
            # More negative = stronger binding
            base_affinity = -8.0  # kcal/mol
            charge_effect = abs(charge) * 2.0
            hydrophobicity_effect = hydrophobicity * 1.5
            
            affinity = base_affinity - charge_effect - hydrophobicity_effect
            
            affinities.append({
                'protein_id': structure.get('protein_id', 0),
                'binding_affinity_kcal_mol': round(affinity, 2),
                'binding_strength': 'Strong' if affinity < -10 else 'Moderate' if affinity < -7 else 'Weak'
            })
        
        return affinities

