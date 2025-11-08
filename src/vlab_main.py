"""
Virtual In Silico Virus Laboratory - Main Entry Point
"""

import sys
import argparse
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.vlab import VLabPipeline, VLabConfig
from src.core.logger import setup_logger

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Virtual In Silico Virus Laboratory - Comprehensive Viral Genome Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis
  python3 src/vlab_main.py data/raw/genome.fasta

  # With custom output directory
  python3 src/vlab_main.py genome.fasta --output my_results/

  # With custom config
  python3 src/vlab_main.py genome.fasta --config my_config.yaml

  # Verbose output
  python3 src/vlab_main.py genome.fasta --verbose
        """
    )
    
    parser.add_argument('genome_file', type=str, help='Path to viral genome file (FASTA format)')
    parser.add_argument('--output', '-o', type=str, default='results', help='Output directory')
    parser.add_argument('--config', '-c', type=str, default='config.yaml', help='Configuration file')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logger(level=log_level)
    logger = logging.getLogger(__name__)
    
    # Validate genome file
    genome_path = Path(args.genome_file)
    if not genome_path.exists():
        logger.error(f"Genome file not found: {genome_path}")
        sys.exit(1)
    
    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        config = VLabConfig.from_file(config_path)
    else:
        logger.warning(f"Config file not found: {config_path}. Using defaults.")
        config = VLabConfig()
        config.save(config_path)
    
    # Override output directory if specified
    if args.output:
        config.output_dir = Path(args.output)
    
    # Create pipeline
    pipeline = VLabPipeline(config)
    
    try:
        # Run analysis
        results = pipeline.analyze(genome_path)
        
        # Print summary
        print("\n" + "="*70)
        print("VIRTUAL IN SILICO VIRUS LABORATORY - ANALYSIS COMPLETE")
        print("="*70)
        print(f"\nGenome: {genome_path.name}")
        print(f"Viability Score: {results.viability['score']:.3f}")
        print(f"Viable: {'YES' if results.viability['is_viable'] else 'NO'}")
        print(f"Human Infection Risk: {results.host_prediction['human_infection_risk']:.3f}")
        print(f"3D Model: {results.assembly.get('pdb_file', 'N/A')}")
        print(f"Report: {config.output_dir / 'report.html'}")
        print(f"Runtime: {results.runtime/3600:.2f} hours")
        print("="*70 + "\n")
        
    except Exception as e:
        logger.error(f"Error during analysis: {e}", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    main()

