"""
Test Suite for VSim
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from src.core.config import Config
from src.genome.analyzer import GenomeAnalyzer
from src.structure.predictor import StructurePredictor
from src.environmental.analyzer import EnvironmentalAnalyzer
from src.cell_interaction.analyzer import CellInteractionAnalyzer
from src.cancer.analyzer import CancerAnalyzer
from src.ml.predictor import MLPredictor

@pytest.fixture
def config():
    """Load test configuration"""
    return Config('config.yaml')

@pytest.fixture
def sample_genome_path():
    """Path to sample genome"""
    return 'data/raw/sample_genome.fasta'

def test_config_load(config):
    """Test configuration loading"""
    assert config is not None
    assert config.get('genome_analysis.min_orf_length') > 0

def test_genome_analyzer(config, sample_genome_path):
    """Test genome analyzer"""
    analyzer = GenomeAnalyzer(config)
    genome_data = analyzer.load_genome(sample_genome_path)
    assert genome_data is not None
    assert 'sequences' in genome_data
    
    results = analyzer.analyze(genome_data)
    assert 'synthesis_feasibility' in results
    assert 'orfs' in results
    assert 'proteins' in results

def test_structure_predictor(config):
    """Test structure predictor"""
    predictor = StructurePredictor(config)
    
    # Create mock genome results
    genome_results = {
        'proteins': [
            {
                'sequence': 'MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWYHVYGVNPTQSDNLRQFVSVFHHPVLGQAS'
            }
        ]
    }
    
    results = predictor.predict(genome_results)
    assert 'structures' in results
    assert 'prediction_method' in results

def test_environmental_analyzer(config):
    """Test environmental analyzer"""
    analyzer = EnvironmentalAnalyzer(config)
    
    genome_results = {
        'length': 10000,
        'gc_content': 0.5,
        'proteins': [{'sequence': 'MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWYHVYGVNPTQSDNLRQFVSVFHHPVLGQAS'}]
    }
    
    structure_results = {
        'structures': [{
            'surface_properties': {
                'hydrophobicity_index': 0.5,
                'charge_density': 0.1
            }
        }]
    }
    
    results = analyzer.analyze(genome_results, structure_results)
    assert 'thermal_stability' in results
    assert 'ph_stability' in results
    assert 'environmental_survival' in results

def test_cell_interaction_analyzer(config):
    """Test cell interaction analyzer"""
    analyzer = CellInteractionAnalyzer(config)
    
    genome_results = {'length': 10000, 'proteins': []}
    structure_results = {
        'structures': [{
            'surface_properties': {
                'hydrophobicity_index': 0.5,
                'charge_density': 0.1
            },
            'binding_sites': []
        }]
    }
    
    results = analyzer.analyze(genome_results, structure_results)
    assert 'receptor_binding' in results
    assert 'cell_entry_mechanisms' in results
    assert 'host_specificity' in results

def test_cancer_analyzer(config):
    """Test cancer analyzer"""
    analyzer = CancerAnalyzer(config)
    
    genome_results = {'length': 10000, 'proteins': []}
    structure_results = {'structures': []}
    cell_results = {
        'cell_entry_mechanisms': {'entry_efficiency': 0.6},
        'receptor_binding': {'binding_score': 0.6}
    }
    
    results = analyzer.analyze(genome_results, structure_results, cell_results)
    assert 'oncolytic_potential' in results
    assert 'cancer_cell_targeting' in results
    assert 'selectivity' in results

def test_ml_predictor(config):
    """Test ML predictor"""
    predictor = MLPredictor(config)
    
    genome_results = {
        'length': 10000,
        'gc_content': 0.5,
        'proteins': [],
        'orfs': []
    }
    
    structure_results = {'structures': []}
    env_results = {
        'thermal_stability': {'stability_score': 0.6},
        'ph_stability': {'stability_score': 0.6},
        'environmental_survival': {'average_survival_time_hours': 24}
    }
    
    cell_results = {
        'receptor_binding': {'binding_score': 0.6},
        'cell_entry_mechanisms': {'entry_efficiency': 0.6},
        'host_specificity': {'specificity_score': 0.6}
    }
    
    cancer_results = {
        'oncolytic_potential': {'oncolytic_score': 0.6},
        'selectivity': {'selectivity_score': 0.6}
    }
    
    results = predictor.predict_all(
        genome_results, structure_results, env_results,
        cell_results, cancer_results
    )
    
    assert 'overall_confidence' in results
    assert 'risk_assessment' in results
    assert 'key_findings' in results

def test_integration(config, sample_genome_path):
    """Integration test"""
    genome_analyzer = GenomeAnalyzer(config)
    structure_predictor = StructurePredictor(config)
    env_analyzer = EnvironmentalAnalyzer(config)
    cell_analyzer = CellInteractionAnalyzer(config)
    cancer_analyzer = CancerAnalyzer(config)
    ml_predictor = MLPredictor(config)
    
    # Load and analyze
    genome_data = genome_analyzer.load_genome(sample_genome_path)
    genome_results = genome_analyzer.analyze(genome_data)
    structure_results = structure_predictor.predict(genome_results)
    env_results = env_analyzer.analyze(genome_results, structure_results)
    cell_results = cell_analyzer.analyze(genome_results, structure_results)
    cancer_results = cancer_analyzer.analyze(genome_results, structure_results, cell_results)
    ml_results = ml_predictor.predict_all(
        genome_results, structure_results, env_results,
        cell_results, cancer_results
    )
    
    # Verify all results exist
    assert genome_results is not None
    assert structure_results is not None
    assert env_results is not None
    assert cell_results is not None
    assert cancer_results is not None
    assert ml_results is not None

