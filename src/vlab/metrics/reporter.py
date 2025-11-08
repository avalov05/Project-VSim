"""
Metrics Reporter - Calculates and reports comprehensive metrics
"""

import logging
from pathlib import Path
from typing import Dict, Any
import json

logger = logging.getLogger(__name__)

class MetricsReporter:
    """Calculate and report metrics"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def calculate(self, annotation: Dict[str, Any], viability: Dict[str, Any],
                 structures: Dict[str, Any], assembly: Dict[str, Any],
                 host_prediction: Dict[str, Any], infection: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive metrics"""
        self.logger.info("Calculating metrics...")
        
        return {
            'genomic': self._calculate_genomic_metrics(annotation),
            'structural': self._calculate_structural_metrics(assembly, structures),
            'viability': viability,
            'host': self._calculate_host_metrics(host_prediction),
            'infection': self._calculate_infection_metrics(infection),
            'safety': self._calculate_safety_metrics(viability, host_prediction)
        }
    
    def generate_report(self, annotation: Dict[str, Any], viability: Dict[str, Any],
                       structures: Dict[str, Any], assembly: Dict[str, Any],
                       host_prediction: Dict[str, Any], infection: Dict[str, Any],
                       metrics: Dict[str, Any]):
        """Generate comprehensive report"""
        self.logger.info("Generating report...")
        
        # Save JSON report
        report_data = {
            'annotation': annotation,
            'viability': viability,
            'structures': structures,
            'assembly': assembly,
            'host_prediction': host_prediction,
            'infection': infection,
            'metrics': metrics
        }
        
        json_file = self.config.output_dir / "report.json"
        with open(json_file, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        # Generate HTML report
        self._generate_html_report(report_data)
        
        self.logger.info(f"Report saved to: {self.config.output_dir}")
    
    def _calculate_genomic_metrics(self, annotation: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate genomic metrics"""
        return {
            'genome_length': annotation.get('genome_length', 0),
            'gc_content': annotation.get('gc_content', 0.0),
            'gene_count': annotation.get('gene_count', 0),
            'coding_density': annotation.get('gene_completeness', {}).get('score', 0.0),
            'codon_bias': annotation.get('codon_usage', {}).get('bias_score', 0.0)
        }
    
    def _calculate_structural_metrics(self, assembly: Dict[str, Any],
                                     structures: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate structural metrics"""
        geometry = assembly.get('geometry', {})
        return {
            'diameter_nm': geometry.get('diameter', 0.0),
            'shape': geometry.get('shape', 'unknown'),
            'symmetry': geometry.get('symmetry', 'unknown'),
            'num_subunits': assembly.get('capsid', {}).get('num_subunits', 0),
            'num_proteins': structures.get('total_structures', 0)
        }
    
    def _calculate_host_metrics(self, host_prediction: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate host-related metrics"""
        return {
            'predicted_hosts': host_prediction.get('hosts', []),
            'human_infection_risk': host_prediction.get('human_infection_risk', 0.0),
            'receptors': host_prediction.get('receptors', []),
            'tropism': host_prediction.get('tropism', {})
        }
    
    def _calculate_infection_metrics(self, infection: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate infection metrics"""
        return infection.get('metrics', {})
    
    def _calculate_safety_metrics(self, viability: Dict[str, Any],
                                 host_prediction: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate safety metrics"""
        human_risk = host_prediction.get('human_infection_risk', 0.0)
        is_viable = viability.get('is_viable', False)
        
        # Overall safety score (lower is safer)
        safety_score = human_risk if is_viable else 0.0
        
        if safety_score > 0.7:
            risk_level = 'Very High'
        elif safety_score > 0.5:
            risk_level = 'High'
        elif safety_score > 0.3:
            risk_level = 'Medium'
        else:
            risk_level = 'Low'
        
        return {
            'safety_score': safety_score,
            'risk_level': risk_level,
            'viability': is_viable,
            'human_risk': human_risk,
            'recommendations': self._get_safety_recommendations(safety_score, is_viable)
        }
    
    def _get_safety_recommendations(self, safety_score: float, is_viable: bool) -> list:
        """Get safety recommendations"""
        recommendations = []
        
        if not is_viable:
            recommendations.append("Genome predicted as non-viable - low risk")
        elif safety_score > 0.7:
            recommendations.append("⚠️ HIGH RISK: Enhanced containment required")
            recommendations.append("⚠️ Human infection risk is significant")
            recommendations.append("⚠️ Use BSL-3 or higher containment")
        elif safety_score > 0.5:
            recommendations.append("Medium risk - standard precautions recommended")
        else:
            recommendations.append("Low risk - standard laboratory practices")
        
        return recommendations
    
    def _generate_html_report(self, report_data: Dict[str, Any]):
        """Generate HTML report"""
        html_file = self.config.output_dir / "report.html"
        
        # Simple HTML template
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>VLab Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; }}
        h1 {{ color: #2c3e50; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        .metric {{ padding: 10px; margin: 10px 0; background: #ecf0f1; border-radius: 5px; }}
        .viable {{ color: #27ae60; font-weight: bold; }}
        .non-viable {{ color: #e74c3c; font-weight: bold; }}
        .high-risk {{ color: #e74c3c; }}
        .medium-risk {{ color: #f39c12; }}
        .low-risk {{ color: #27ae60; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Virtual In Silico Virus Laboratory - Analysis Report</h1>
        
        <h2>Viability Prediction</h2>
        <div class="metric">
            <p><strong>Score:</strong> {report_data['viability'].get('score', 0.0):.3f}</p>
            <p><strong>Viable:</strong> 
                <span class="{'viable' if report_data['viability'].get('is_viable') else 'non-viable'}">
                    {'YES' if report_data['viability'].get('is_viable') else 'NO'}
                </span>
            </p>
            <p><strong>Confidence:</strong> {report_data['viability'].get('confidence', 0.0):.3f}</p>
        </div>
        
        <h2>Safety Assessment</h2>
        <div class="metric">
            <p><strong>Human Infection Risk:</strong> 
                <span class="{self._get_risk_class(report_data['metrics']['safety']['safety_score'])}">
                    {report_data['metrics']['safety']['risk_level']}
                </span>
            </p>
            <p><strong>Safety Score:</strong> {report_data['metrics']['safety']['safety_score']:.3f}</p>
            <ul>
                {''.join(f'<li>{rec}</li>' for rec in report_data['metrics']['safety'].get('recommendations', []))}
            </ul>
        </div>
        
        <h2>Structural Properties</h2>
        <div class="metric">
            <p><strong>Shape:</strong> {report_data['assembly'].get('geometry', {}).get('shape', 'unknown')}</p>
            <p><strong>Diameter:</strong> {report_data['assembly'].get('geometry', {}).get('diameter', 0.0):.1f} nm</p>
            <p><strong>3D Model:</strong> <a href="{report_data['assembly'].get('pdb_file', '#')}">Download PDB</a></p>
        </div>
        
        <h2>Host Prediction</h2>
        <div class="metric">
            <p><strong>Predicted Hosts:</strong></p>
            <ul>
                {''.join(f"<li>{h.get('species', 'Unknown')} (probability: {h.get('probability', 0.0):.2f})</li>" 
                         for h in report_data['host_prediction'].get('hosts', []))}
            </ul>
        </div>
        
        <h2>Infection Dynamics</h2>
        <div class="metric">
            <p><strong>Burst Size:</strong> {report_data['infection'].get('burst_size', 0)}</p>
            <p><strong>Replication Time:</strong> {report_data['infection'].get('replication_time', 0.0):.1f} hours</p>
        </div>
    </div>
</body>
</html>
        """
        
        with open(html_file, 'w') as f:
            f.write(html)
        
        self.logger.info(f"HTML report saved to: {html_file}")
    
    def _get_risk_class(self, score: float) -> str:
        """Get CSS class for risk level"""
        if score > 0.7:
            return 'high-risk'
        elif score > 0.3:
            return 'medium-risk'
        else:
            return 'low-risk'

