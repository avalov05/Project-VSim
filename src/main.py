"""
VSim - Viral Simulation & Analysis Platform
Main Entry Point
"""

import sys
import os
import argparse
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.config import Config
from src.core.logger import setup_logger
from src.genome.analyzer import GenomeAnalyzer
from src.structure.predictor import StructurePredictor
from src.environmental.analyzer import EnvironmentalAnalyzer
from src.cell_interaction.analyzer import CellInteractionAnalyzer
from src.cancer.analyzer import CancerAnalyzer
from src.ml.predictor import MLPredictor
from src.core.report import ReportGenerator

def main():
    """Main entry point for VSim command-line interface"""
    parser = argparse.ArgumentParser(
        description='VSim - Viral Simulation & Analysis Platform',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 src/main.py data/raw/sample_genome.fasta
  python3 src/main.py genome.fasta --output my_results/
  python3 src/main.py /path/to/genome.fasta --verbose
        """
    )
    parser.add_argument('genome_file', type=str, help='Path to viral genome file (FASTA format)')
    parser.add_argument('--output', '-o', type=str, default='results', help='Output directory')
    parser.add_argument('--config', '-c', type=str, default='config.yaml', help='Configuration file')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Validate genome file early
    genome_path = Path(args.genome_file)
    if not genome_path.exists():
        # Try common locations
        if Path('data/raw/sample_genome.fasta').exists():
            print(f"\n⚠️  File '{args.genome_file}' not found.")
            print(f"💡 Tip: Use the sample genome: python3 src/main.py data/raw/sample_genome.fasta\n")
    
    # Load configuration
    config = Config(args.config)
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logger(config, log_level)
    
    logger.info("=" * 80)
    logger.info("VSim - Viral Simulation & Analysis Platform")
    logger.info("=" * 80)
    logger.info(f"Analyzing genome: {args.genome_file}")
    
    try:
        # Initialize analyzers
        logger.info("Initializing analysis modules...")
        genome_analyzer = GenomeAnalyzer(config)
        structure_predictor = StructurePredictor(config)
        env_analyzer = EnvironmentalAnalyzer(config)
        cell_analyzer = CellInteractionAnalyzer(config)
        cancer_analyzer = CancerAnalyzer(config)
        ml_predictor = MLPredictor(config)
        
        # Load genome
        logger.info("Loading genome sequence...")
        genome_data = genome_analyzer.load_genome(args.genome_file)
        
        # Phase 1: Genome Analysis
        logger.info("\n" + "="*80)
        logger.info("PHASE 1: GENOME ANALYSIS")
        logger.info("="*80)
        genome_results = genome_analyzer.analyze(genome_data)
        
        # Phase 2: Structure Prediction
        logger.info("\n" + "="*80)
        logger.info("PHASE 2: STRUCTURE PREDICTION")
        logger.info("="*80)
        structure_results = structure_predictor.predict(genome_results)
        
        # Phase 3: Environmental Analysis
        logger.info("\n" + "="*80)
        logger.info("PHASE 3: ENVIRONMENTAL DYNAMICS")
        logger.info("="*80)
        env_results = env_analyzer.analyze(genome_results, structure_results)
        
        # Phase 4: Cell Interaction Analysis
        logger.info("\n" + "="*80)
        logger.info("PHASE 4: CELL INTERACTION ANALYSIS")
        logger.info("="*80)
        cell_results = cell_analyzer.analyze(genome_results, structure_results)
        
        # Phase 5: Cancer Cell Analysis
        logger.info("\n" + "="*80)
        logger.info("PHASE 5: CANCER CELL ANALYSIS")
        logger.info("="*80)
        cancer_results = cancer_analyzer.analyze(genome_results, structure_results, cell_results)
        
        # Phase 6: ML Prediction
        logger.info("\n" + "="*80)
        logger.info("PHASE 6: ML-BASED PREDICTION")
        logger.info("="*80)
        ml_results = ml_predictor.predict_all(
            genome_results, structure_results, env_results, 
            cell_results, cancer_results
        )
        
        # Generate comprehensive report
        logger.info("\n" + "="*80)
        logger.info("GENERATING COMPREHENSIVE REPORT")
        logger.info("="*80)
        report_generator = ReportGenerator(config)
        report = report_generator.generate(
            genome_results, structure_results, env_results,
            cell_results, cancer_results, ml_results
        )
        
        # Save results
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        report.save(output_dir / "comprehensive_report.html")
        report.save_json(output_dir / "results.json")
        
        logger.info(f"\nResults saved to: {output_dir}")
        logger.info("Analysis complete!")
        
        # Print summary
        print("\n" + "="*80)
        print("ANALYSIS SUMMARY")
        print("="*80)
        print(f"Synthesis Feasibility: {genome_results['synthesis_feasibility']:.2%}")
        print(f"Confidence Level: {ml_results['overall_confidence']:.2%}")
        print(f"Report: {output_dir / 'comprehensive_report.html'}")
        print("="*80)
        
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()

