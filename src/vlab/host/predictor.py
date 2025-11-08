"""
Host Predictor - Predicts host species and tropism
"""

import logging
from typing import Dict, Any, List
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

class HostPredictor:
    """Predict host species and cell tropism"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.device = torch.device(f"cuda:{config.gpu_id}" if config.use_gpu and torch.cuda.is_available() else "cpu")
    
    def predict(self, annotation: Dict[str, Any], structures: Dict[str, Any]) -> Dict[str, Any]:
        """Predict host and tropism"""
        self.logger.info("Predicting host species and tropism...")
        
        # Predict host species
        hosts = self._predict_hosts(annotation, structures)
        
        # Predict receptors
        receptors = self._predict_receptors(annotation, structures)
        
        # Predict tropism
        tropism = self._predict_tropism(receptors)
        
        # Predict infectivity
        infectivity = self._predict_infectivity(annotation, hosts, receptors)
        
        return {
            'hosts': hosts,
            'receptors': receptors,
            'tropism': tropism,
            'infectivity': infectivity,
            'human_infection_risk': infectivity.get('human_risk', 0.0)
        }
    
    def _predict_hosts(self, annotation: Dict[str, Any], structures: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Predict likely host species"""
        # Simplified prediction based on virus type and features
        virus_type = annotation.get('virus_type', 'unknown').lower()
        
        hosts = []
        
        # Heuristic-based prediction
        if 'corona' in virus_type:
            hosts.append({'species': 'Human', 'probability': 0.8})
            hosts.append({'species': 'Bat', 'probability': 0.6})
            hosts.append({'species': 'Pangolin', 'probability': 0.3})
        elif 'filo' in virus_type or 'ebola' in virus_type.lower():
            hosts.append({'species': 'Human', 'probability': 0.7})
            hosts.append({'species': 'Primate', 'probability': 0.8})
            hosts.append({'species': 'Bat', 'probability': 0.5})
        else:
            # Default predictions
            hosts.append({'species': 'Unknown', 'probability': 0.5})
        
        return hosts
    
    def _predict_receptors(self, annotation: Dict[str, Any], structures: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Predict cell surface receptors"""
        # Simplified receptor prediction
        receptors = []
        
        virus_type = annotation.get('virus_type', 'unknown').lower()
        
        if 'corona' in virus_type:
            receptors.append({
                'receptor': 'ACE2',
                'binding_affinity': 0.8,
                'cell_types': ['Lung epithelial', 'Intestinal epithelial']
            })
        elif 'hiv' in virus_type or 'retro' in virus_type:
            receptors.append({
                'receptor': 'CD4',
                'binding_affinity': 0.9,
                'cell_types': ['T cells', 'Macrophages']
            })
        else:
            receptors.append({
                'receptor': 'Unknown',
                'binding_affinity': 0.5,
                'cell_types': ['Unknown']
            })
        
        return receptors
    
    def _predict_tropism(self, receptors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Predict cell and tissue tropism"""
        cell_types = set()
        for receptor in receptors:
            cell_types.update(receptor.get('cell_types', []))
        
        # Determine tissue tropism
        tissues = []
        if any('Lung' in ct for ct in cell_types):
            tissues.append('Respiratory')
        if any('Intestinal' in ct for ct in cell_types):
            tissues.append('Gastrointestinal')
        if any('T cell' in ct for ct in cell_types):
            tissues.append('Lymphoid')
        
        return {
            'cell_types': list(cell_types),
            'tissues': tissues if tissues else ['Unknown'],
            'broad_tropism': len(cell_types) > 3
        }
    
    def _predict_infectivity(self, annotation: Dict[str, Any], 
                           hosts: List[Dict[str, Any]],
                           receptors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Predict human infectivity risk"""
        # Check if human is in predicted hosts
        human_host = next((h for h in hosts if 'Human' in h.get('species', '')), None)
        human_prob = human_host.get('probability', 0.0) if human_host else 0.0
        
        # Receptor binding strength
        max_affinity = max((r.get('binding_affinity', 0.0) for r in receptors), default=0.0)
        
        # Combined risk score
        risk_score = (human_prob * 0.6 + max_affinity * 0.4)
        
        # Risk categories
        if risk_score > 0.7:
            risk_level = 'High'
        elif risk_score > 0.4:
            risk_level = 'Medium'
        else:
            risk_level = 'Low'
        
        return {
            'human_risk': risk_score,
            'risk_level': risk_level,
            'human_host_probability': human_prob,
            'receptor_affinity': max_affinity
        }

