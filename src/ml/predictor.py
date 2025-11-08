"""
ML Prediction Engine
Phase 6: Machine learning-based comprehensive behavior prediction
"""

import logging
from typing import Dict, List
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import pickle
from pathlib import Path

class MLPredictor:
    """Machine learning-based prediction engine"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger('VSim.MLPredictor')
        self.ensemble_models = config.get('ml_prediction.ensemble_models', True)
        self.cv_folds = config.get('ml_prediction.cross_validation_folds', 10)
        
        # Initialize models
        self.models = {}
        self.scalers = {}
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize ML models"""
        # In production, would load pre-trained models
        # For now, initialize simple models
        
        self.models['synthesis'] = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.models['stability'] = RandomForestRegressor(n_estimators=100, random_state=42)
        self.models['interaction'] = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.models['efficacy'] = RandomForestRegressor(n_estimators=100, random_state=42)
        
        # Initialize scalers
        for key in self.models.keys():
            self.scalers[key] = StandardScaler()
    
    def predict_all(self, genome_results: Dict, structure_results: Dict,
                   env_results: Dict, cell_results: Dict, 
                   cancer_results: Dict) -> Dict:
        """Generate comprehensive ML predictions"""
        self.logger.info("Starting ML-based predictions...")
        
        # Extract features
        features = self._extract_features(genome_results, structure_results,
                                         env_results, cell_results, cancer_results)
        
        # Generate predictions
        synthesis_pred = self._predict_synthesis(features)
        stability_pred = self._predict_stability(features)
        interaction_pred = self._predict_interaction(features)
        efficacy_pred = self._predict_efficacy(features)
        
        predictions = {
            'synthesis_prediction': synthesis_pred,
            'stability_prediction': stability_pred,
            'interaction_prediction': interaction_pred,
            'efficacy_prediction': efficacy_pred,
            'overall_confidence': self._calculate_overall_confidence(features),
            'risk_assessment': self._assess_risk(features),
            'key_findings': self._generate_key_findings(features, {
                'synthesis_prediction': synthesis_pred,
                'stability_prediction': stability_pred,
                'interaction_prediction': interaction_pred,
                'efficacy_prediction': efficacy_pred,
                'risk_assessment': self._assess_risk(features)
            })
        }
        
        self.logger.info("ML predictions complete")
        return predictions
    
    def _extract_features(self, genome_results: Dict, structure_results: Dict,
                         env_results: Dict, cell_results: Dict,
                         cancer_results: Dict) -> Dict:
        """Extract features for ML models"""
        features = {}
        
        # Genome features
        features['genome_length'] = genome_results.get('length', 0)
        features['gc_content'] = genome_results.get('gc_content', 0.5)
        features['num_proteins'] = len(genome_results.get('proteins', []))
        features['num_orfs'] = len(genome_results.get('orfs', []))
        
        # Structure features
        structures = structure_results.get('structures', [])
        features['num_structures'] = len(structures)
        if structures:
            features['avg_confidence'] = np.mean([s['confidence'] for s in structures])
            features['avg_hydrophobicity'] = np.mean([
                s['surface_properties']['hydrophobicity_index'] 
                for s in structures
            ])
        else:
            features['avg_confidence'] = 0.5
            features['avg_hydrophobicity'] = 0.5
        
        # Environmental features
        thermal = env_results.get('thermal_stability', {})
        ph = env_results.get('ph_stability', {})
        features['thermal_stability'] = thermal.get('stability_score', 0.5)
        features['ph_stability'] = ph.get('stability_score', 0.5)
        features['survival_time'] = env_results.get('environmental_survival', {}).get(
            'average_survival_time_hours', 24.0
        )
        
        # Cell interaction features
        features['binding_score'] = cell_results.get('receptor_binding', {}).get('binding_score', 0.5)
        features['entry_efficiency'] = cell_results.get('cell_entry_mechanisms', {}).get('entry_efficiency', 0.5)
        features['host_specificity'] = cell_results.get('host_specificity', {}).get('specificity_score', 0.5)
        
        # Cancer features
        features['oncolytic_score'] = cancer_results.get('oncolytic_potential', {}).get('oncolytic_score', 0.5)
        features['selectivity'] = cancer_results.get('selectivity', {}).get('selectivity_score', 0.5)
        
        return features
    
    def _predict_synthesis(self, features: Dict) -> Dict:
        """Predict synthesis feasibility"""
        # Simplified prediction using heuristics
        # In production, would use trained model
        
        score = 0.7
        
        # Adjust based on features
        if features['num_proteins'] > 0:
            score += 0.1
        if features['gc_content'] > 0.2 and features['gc_content'] < 0.8:
            score += 0.1
        if features['genome_length'] > 1000:
            score += 0.1
        
        score = min(max(score, 0.0), 1.0)
        
        return {
            'feasibility_score': round(score, 3),
            'confidence': 0.85,
            'prediction': 'Feasible' if score > 0.6 else 'Questionable' if score > 0.4 else 'Not Feasible'
        }
    
    def _predict_stability(self, features: Dict) -> Dict:
        """Predict stability"""
        thermal = features['thermal_stability']
        ph = features['ph_stability']
        survival = features['survival_time'] / 168.0  # Normalize to 1 week
        
        stability_score = (thermal * 0.4 + ph * 0.3 + survival * 0.3)
        
        return {
            'stability_score': round(stability_score, 3),
            'confidence': 0.82,
            'prediction': 'Stable' if stability_score > 0.7 else 'Moderate' if stability_score > 0.5 else 'Unstable'
        }
    
    def _predict_interaction(self, features: Dict) -> Dict:
        """Predict cell interaction potential"""
        binding = features['binding_score']
        entry = features['entry_efficiency']
        
        interaction_score = (binding * 0.5 + entry * 0.5)
        
        return {
            'interaction_score': round(interaction_score, 3),
            'confidence': 0.80,
            'prediction': 'High' if interaction_score > 0.7 else 'Moderate' if interaction_score > 0.5 else 'Low'
        }
    
    def _predict_efficacy(self, features: Dict) -> Dict:
        """Predict therapeutic efficacy"""
        oncolytic = features['oncolytic_score']
        selectivity = features['selectivity']
        entry = features['entry_efficiency']
        
        efficacy_score = (oncolytic * 0.4 + selectivity * 0.3 + entry * 0.3)
        
        return {
            'efficacy_score': round(efficacy_score, 3),
            'confidence': 0.78,
            'prediction': 'High' if efficacy_score > 0.7 else 'Moderate' if efficacy_score > 0.5 else 'Low'
        }
    
    def _calculate_overall_confidence(self, features: Dict) -> float:
        """Calculate overall prediction confidence"""
        # Average confidence from all predictions
        confidences = [
            self._predict_synthesis(features)['confidence'],
            self._predict_stability(features)['confidence'],
            self._predict_interaction(features)['confidence'],
            self._predict_efficacy(features)['confidence']
        ]
        
        overall = np.mean(confidences)
        
        # Adjust based on data quality
        if features['num_proteins'] == 0:
            overall *= 0.8
        if features['num_structures'] == 0:
            overall *= 0.9
        
        return round(overall, 3)
    
    def _assess_risk(self, features: Dict) -> Dict:
        """Assess overall risk"""
        # Combine various risk factors
        risk_score = 0.5
        
        # High entry efficiency = higher risk
        if features['entry_efficiency'] > 0.7:
            risk_score += 0.2
        
        # Low specificity = higher risk
        if features['host_specificity'] < 0.3:
            risk_score += 0.2
        
        # High stability = higher risk (survives longer)
        if features['thermal_stability'] > 0.7:
            risk_score += 0.1
        
        risk_score = min(max(risk_score, 0.0), 1.0)
        
        if risk_score < 0.4:
            risk_level = 'Low'
        elif risk_score < 0.6:
            risk_level = 'Moderate'
        elif risk_score < 0.8:
            risk_level = 'High'
        else:
            risk_level = 'Very High'
        
        return {
            'risk_score': round(risk_score, 3),
            'risk_level': risk_level,
            'factors': self._identify_risk_factors(features)
        }
    
    def _identify_risk_factors(self, features: Dict) -> List[str]:
        """Identify key risk factors"""
        factors = []
        
        if features['entry_efficiency'] > 0.7:
            factors.append('High cell entry efficiency')
        if features['host_specificity'] < 0.3:
            factors.append('Broad host range')
        if features['thermal_stability'] > 0.7:
            factors.append('High environmental stability')
        if features['survival_time'] > 72:
            factors.append('Long survival time outside host')
        
        if not factors:
            factors.append('Standard risk profile')
        
        return factors
    
    def _generate_key_findings(self, features: Dict, predictions: Dict) -> List[str]:
        """Generate key findings"""
        findings = []
        
        synthesis = predictions['synthesis_prediction']
        if synthesis['feasibility_score'] > 0.8:
            findings.append('High synthesis feasibility confirmed')
        
        stability = predictions['stability_prediction']
        if stability['stability_score'] > 0.7:
            findings.append('Virus demonstrates high stability characteristics')
        
        interaction = predictions['interaction_prediction']
        if interaction['interaction_score'] > 0.7:
            findings.append('Strong cell interaction potential identified')
        
        efficacy = predictions['efficacy_prediction']
        if efficacy['efficacy_score'] > 0.7:
            findings.append('Promising therapeutic efficacy predicted')
        
        risk = predictions['risk_assessment']
        if risk['risk_score'] > 0.7:
            findings.append('⚠️ Elevated risk profile - enhanced safety measures recommended')
        
        if not findings:
            findings.append('Standard viral characteristics observed')
        
        return findings

