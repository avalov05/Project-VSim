#!/usr/bin/env python3
"""
Collect Training Data for VLab Viability Model
Harvests all viable viral genomes and generates non-viable genomes
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.vlab.data.collector import DataCollector

def main():
    """Main data collection script"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Collect training data for VLab viability model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic collection (defaults)
  python3 collect_training_data.py

  # Large-scale collection
  python3 collect_training_data.py --max_viable 100000 --num_synthetic 20000

  # With NCBI API key (faster)
  python3 collect_training_data.py --api_key YOUR_API_KEY

  # Quick test (small dataset)
  python3 collect_training_data.py --max_viable 1000 --num_synthetic 500
        """
    )
    
    parser.add_argument('--email', type=str, default="anton.valov05@gmail.com",
                       help='Your email for NCBI (required by NCBI)')
    parser.add_argument('--api_key', type=str, default="d7e5c7978697a8c4284af0fc71ce1a2b9808",
                       help='NCBI API key (optional, speeds up downloads)')
    parser.add_argument('--max_viable', type=int, default=None,
                       help='Maximum number of viable genomes to collect (default: auto-calculated for balanced dataset)')
    parser.add_argument('--num_synthetic', type=int, default=None,
                       help='Number of synthetic non-viable genomes (default: auto-calculated)')
    parser.add_argument('--num_mutated', type=int, default=None,
                       help='Number of mutated non-viable genomes (default: auto-calculated)')
    parser.add_argument('--total_target', type=int, default=500000,
                       help='Total target genomes (default: 500000, creates balanced 50/50 viable/non-viable dataset)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Create logs directory (before logging setup)
    Path('logs').mkdir(exist_ok=True)
    
    # Setup logging with unbuffered output
    log_level = logging.DEBUG if args.verbose else logging.INFO
    # Create stream handler with unbuffered output
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(log_level)
    stream_handler.setStream(sys.stdout)
    
    file_handler = logging.FileHandler('logs/data_collection.log')
    file_handler.setLevel(log_level)
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[stream_handler, file_handler],
        force=True  # Override any existing config
    )
    
    # Ensure unbuffered output
    sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None
    
    logger = logging.getLogger(__name__)
    
    # Check email
    if args.email == "vlab@example.com":
        logger.warning("Using default email. Please provide your email with --email")
        logger.warning("NCBI requires a valid email address")
    
    logger.info("="*70)
    logger.info("VIRTUAL IN SILICO VIRUS LABORATORY - DATA COLLECTION")
    logger.info("="*70)
    logger.info(f"Email: {args.email}")
    logger.info(f"API Key: {'Provided' if args.api_key else 'Not provided'}")
    logger.info(f"Target Total: {args.total_target:,} genomes")
    logger.info(f"Max viable genomes: {args.max_viable if args.max_viable is not None else 'Auto-calculated'}")
    logger.info(f"Synthetic non-viable: {args.num_synthetic if args.num_synthetic is not None else 'Auto-calculated'}")
    logger.info(f"Mutated non-viable: {args.num_mutated if args.num_mutated is not None else 'Auto-calculated'}")
    logger.info("="*70)
    
    try:
        # Create collector
        collector = DataCollector(email=args.email, api_key=args.api_key)
        
        # Collect data
        results = asyncio.run(collector.collect_all_data(
            max_viable=args.max_viable,
            num_synthetic_non_viable=args.num_synthetic,
            num_mutated_non_viable=args.num_mutated,
            total_target=args.total_target
        ))
        
        # Get statistics
        stats = collector.get_data_statistics()
        
        # Print summary
        print("\n" + "="*70)
        print("DATA COLLECTION COMPLETE!")
        print("="*70)
        print(f"Training Data:")
        print(f"  Viable: {stats['train_viable']:,}")
        print(f"  Non-viable: {stats['train_non_viable']:,}")
        print(f"  Total: {stats['total_train']:,}")
        print(f"\nValidation Data:")
        print(f"  Viable: {stats['val_viable']:,}")
        print(f"  Non-viable: {stats['val_non_viable']:,}")
        print(f"  Total: {stats['total_val']:,}")
        print(f"\nGrand Total: {stats['total']:,} genomes")
        if results.get('target'):
            print(f"Target: {results['target']:,} genomes")
            if results.get('reached_target'):
                print("[SUCCESS] Target reached!")
            else:
                print(f"[WARNING] Target not reached (collected {results['total']:,} of {results['target']:,})")
        print(f"\nData location: data/training/")
        print("="*70)
        print("\nNext steps:")
        print("1. Train the viability model:")
        print("   python3 src/vlab/training/viability_trainer.py --data_dir data/training")
        print("2. Use the trained model in VLab")
        print("="*70 + "\n")
        
    except KeyboardInterrupt:
        logger.info("\nData collection interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Data collection failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    main()

