"""
Main Pipeline for Virtual In Silico Virus Laboratory
"""

import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass

from .config import VLabConfig
from ..annotation.annotator import GenomeAnnotator
from ..viability.predictor import ViabilityPredictorWrapper as ViabilityPredictor
from ..structure.folding import StructurePredictor
from ..assembly.simulator import AssemblySimulator
from ..host.predictor import HostPredictor
from ..infection.simulator import InfectionSimulator
from ..metrics.reporter import MetricsReporter

logger = logging.getLogger(__name__)

@dataclass
class VLabResults:
    """Results from VLab pipeline"""
    genome_path: Path
    annotation: Dict[str, Any]
    viability: Dict[str, Any]
    structures: Dict[str, Any]
    assembly: Dict[str, Any]
    host_prediction: Dict[str, Any]
    infection: Dict[str, Any]
    metrics: Dict[str, Any]
    runtime: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'genome_path': str(self.genome_path),
            'annotation': self.annotation,
            'viability': self.viability,
            'structures': self.structures,
            'assembly': self.assembly,
            'host_prediction': self.host_prediction,
            'infection': self.infection,
            'metrics': self.metrics,
            'runtime': self.runtime
        }

class VLabPipeline:
    """Main pipeline for viral genome analysis"""
    
    def __init__(self, config: VLabConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.annotator = GenomeAnnotator(config)
        self.viability_predictor = ViabilityPredictor(config)
        self.structure_predictor = StructurePredictor(config)
        self.assembly_simulator = AssemblySimulator(config)
        self.host_predictor = HostPredictor(config)
        self.infection_simulator = InfectionSimulator(config)
        self.metrics_reporter = MetricsReporter(config)
        
        # Create output directory
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
    
    def analyze(self, genome_path: Path, output_dir: Optional[Path] = None) -> VLabResults:
        """
        Run complete analysis pipeline on viral genome
        
        Args:
            genome_path: Path to viral genome FASTA file
            output_dir: Optional output directory (overrides config)
        
        Returns:
            VLabResults object with all analysis results
        """
        start_time = time.time()
        
        if output_dir:
            self.config.output_dir = Path(output_dir)
            self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Starting VLab analysis for: {genome_path}")
        self.logger.info(f"Output directory: {self.config.output_dir}")
        
        try:
            # Step 1: Genome Annotation
            self.logger.info("Step 1/7: Genome Annotation")
            annotation = self.annotator.annotate(genome_path)
            self._save_checkpoint("annotation", annotation)
            
            # Step 2: Viability Prediction
            self.logger.info("Step 2/7: Viability Prediction")
            viability = self.viability_predictor.predict(annotation)
            self._save_checkpoint("viability", viability)
            
            # Early exit if not viable
            if viability['score'] < self.config.viability_threshold:
                self.logger.warning(
                    f"Genome predicted as non-viable (score: {viability['score']:.3f}). "
                    "Continuing with reduced analysis..."
                )
            
            # Step 3: Structure Prediction
            self.logger.info("Step 3/7: Protein Structure Prediction")
            structures = self.structure_predictor.predict(annotation)
            self._save_checkpoint("structures", structures)
            
            # Step 4: Virion Assembly
            self.logger.info("Step 4/7: Virion Assembly Simulation")
            assembly = self.assembly_simulator.assemble(annotation, structures)
            self._save_checkpoint("assembly", assembly)
            
            # Step 5: Host Prediction
            self.logger.info("Step 5/7: Host and Tropism Prediction")
            host_prediction = self.host_predictor.predict(annotation, structures)
            self._save_checkpoint("host_prediction", host_prediction)
            
            # Step 6: Infection Simulation
            if self.config.simulate_infection:
                self.logger.info("Step 6/7: Infection Dynamics Simulation")
                infection = self.infection_simulator.simulate(
                    annotation, structures, assembly, host_prediction
                )
                self._save_checkpoint("infection", infection)
            else:
                infection = {}
            
            # Step 7: Metrics and Reporting
            self.logger.info("Step 7/7: Metrics Calculation and Reporting")
            metrics = self.metrics_reporter.calculate(
                annotation, viability, structures, assembly, 
                host_prediction, infection
            )
            
            # Generate reports
            if self.config.generate_report:
                self.metrics_reporter.generate_report(
                    annotation, viability, structures, assembly,
                    host_prediction, infection, metrics
                )
            
            runtime = time.time() - start_time
            self.logger.info(f"Analysis complete in {runtime:.1f} seconds ({runtime/3600:.2f} hours)")
            
            return VLabResults(
                genome_path=genome_path,
                annotation=annotation,
                viability=viability,
                structures=structures,
                assembly=assembly,
                host_prediction=host_prediction,
                infection=infection,
                metrics=metrics,
                runtime=runtime
            )
            
        except Exception as e:
            self.logger.error(f"Error in analysis pipeline: {e}", exc_info=True)
            raise
    
    def _save_checkpoint(self, step: str, data: Dict[str, Any]):
        """Save checkpoint data"""
        if self.config.save_intermediates:
            checkpoint_dir = self.config.output_dir / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            import json
            checkpoint_file = checkpoint_dir / f"{step}.json"
            with open(checkpoint_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)

