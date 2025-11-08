"""
Enhanced Data Collector for VLab
Main interface for collecting training data
"""

import logging
from pathlib import Path
from typing import List, Dict, Any
import asyncio

from .viral_harvester import ViralGenomeHarvester

logger = logging.getLogger(__name__)

class DataCollector:
    """Main data collection interface"""
    
    def __init__(self, email="vlab@example.com", api_key=None):
        self.email = email
        self.api_key = api_key
        self.harvester = ViralGenomeHarvester(email=email, api_key=api_key)
    
    async def collect_all_data(self, 
                              max_viable=None,
                              num_synthetic_non_viable=None,
                              num_mutated_non_viable=None,
                              total_target=500000):
        """
        Collect all training data
        
        Args:
            max_viable: Maximum number of viable genomes to collect (None = auto-calculate)
            num_synthetic_non_viable: Number of synthetic non-viable genomes (None = auto-calculate)
            num_mutated_non_viable: Number of mutated non-viable genomes (None = auto-calculate)
            total_target: Total target genomes (creates balanced 50/50 viable/non-viable dataset)
        """
        logger.info("Starting comprehensive data collection...")
        
        # If total_target is specified, calculate distribution to reach target
        # Goal: Balanced dataset (50% viable, 50% non-viable) for best training
        if total_target:
            # Calculate optimal distribution: 50% viable, 50% non-viable
            target_viable = int(total_target * 0.5)
            target_non_viable = total_target - target_viable
            
            # Set defaults if not provided
            if max_viable is None:
                max_viable = target_viable
            elif max_viable < target_viable:
                logger.info(f"Adjusting max_viable from {max_viable:,} to {target_viable:,} to reach balanced dataset")
                max_viable = target_viable
            
            # Split non-viable between synthetic and mutated (70% synthetic, 30% mutated)
            if num_synthetic_non_viable is None:
                num_synthetic_non_viable = int(target_non_viable * 0.7)
            if num_mutated_non_viable is None:
                num_mutated_non_viable = int(target_non_viable * 0.3)
            
            logger.info(f"Target: {total_target:,} total genomes with balanced distribution:")
            logger.info(f"  Viable: {max_viable:,} (50%)")
            logger.info(f"  Synthetic non-viable: {num_synthetic_non_viable:,} (35%)")
            logger.info(f"  Mutated non-viable: {num_mutated_non_viable:,} (15%)")
        else:
            # Use provided values or defaults
            if max_viable is None:
                max_viable = 50000
            if num_synthetic_non_viable is None:
                num_synthetic_non_viable = 10000
            if num_mutated_non_viable is None:
                num_mutated_non_viable = 5000
        
        # Step 1: Harvest viable genomes
        logger.info("="*70)
        logger.info("STEP 1: Harvesting viable viral genomes from NCBI")
        logger.info("="*70)
        viable_sequences = await self.harvester.harvest_all_viable_genomes(max_sequences=max_viable)
        
        # Step 2: Generate synthetic non-viable
        logger.info("="*70)
        logger.info("STEP 2: Generating synthetic non-viable genomes")
        logger.info("="*70)
        self.harvester.generate_non_viable_genomes(
            count=num_synthetic_non_viable,
            methods=['random', 'fragmented', 'no_start', 'no_stop', 'invalid_codons', 'too_short', 'missing_genes']
        )
        
        # Step 3: Generate mutated non-viable
        if viable_sequences and num_mutated_non_viable > 0:
            logger.info("="*70)
            logger.info("STEP 3: Generating mutated non-viable genomes")
            logger.info("="*70)
            self.harvester.generate_from_viable(viable_sequences, count=num_mutated_non_viable)
        
        # Summary
        logger.info("="*70)
        logger.info("DATA COLLECTION COMPLETE!")
        logger.info("="*70)
        logger.info(f"✓ Viable genomes: {len(viable_sequences)}")
        logger.info(f"✓ Synthetic non-viable: {num_synthetic_non_viable}")
        logger.info(f"✓ Mutated non-viable: {num_mutated_non_viable}")
        logger.info(f"✓ Total training samples: {len(viable_sequences) + num_synthetic_non_viable + num_mutated_non_viable}")
        logger.info(f"✓ Data location: {self.harvester.data_dir}")
        
        total_collected = len(viable_sequences) + num_synthetic_non_viable + num_mutated_non_viable
        
        return {
            'viable': len(viable_sequences),
            'synthetic_non_viable': num_synthetic_non_viable,
            'mutated_non_viable': num_mutated_non_viable,
            'total': total_collected,
            'target': total_target if total_target else None,
            'reached_target': total_collected >= total_target if total_target else None
        }
    
    def collect_viable_genomes(self, output_dir: Path, max_sequences=10000):
        """Collect viable genomes only"""
        async def _collect():
            sequences = await self.harvester.harvest_all_viable_genomes(max_sequences=max_sequences)
            return sequences
        
        return asyncio.run(_collect())
    
    def generate_synthetic_non_viable(self, count: int, output_dir: Path):
        """Generate synthetic non-viable genomes"""
        return self.harvester.generate_non_viable_genomes(count=count)
    
    def get_data_statistics(self) -> Dict[str, Any]:
        """Get statistics about collected data"""
        data_dir = self.harvester.data_dir
        
        stats = {
            'train_viable': len(list((data_dir / "train" / "viable").glob("*.fasta"))),
            'train_non_viable': len(list((data_dir / "train" / "non_viable").glob("*.fasta"))),
            'val_viable': len(list((data_dir / "val" / "viable").glob("*.fasta"))),
            'val_non_viable': len(list((data_dir / "val" / "non_viable").glob("*.fasta"))),
        }
        
        stats['total_train'] = stats['train_viable'] + stats['train_non_viable']
        stats['total_val'] = stats['val_viable'] + stats['val_non_viable']
        stats['total'] = stats['total_train'] + stats['total_val']
        
        return stats

def main():
    """Main data collection script"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Collect training data for VLab')
    parser.add_argument('--email', type=str, default="vlab@example.com", help='NCBI email')
    parser.add_argument('--api_key', type=str, default=None, help='NCBI API key (optional)')
    parser.add_argument('--max_viable', type=int, default=50000, help='Max viable genomes')
    parser.add_argument('--num_synthetic', type=int, default=10000, help='Number of synthetic non-viable')
    parser.add_argument('--num_mutated', type=int, default=5000, help='Number of mutated non-viable')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Collect data
    collector = DataCollector(email=args.email, api_key=args.api_key)
    
    results = asyncio.run(collector.collect_all_data(
        max_viable=args.max_viable,
        num_synthetic_non_viable=args.num_synthetic,
        num_mutated_non_viable=args.num_mutated
    ))
    
    # Print statistics
    stats = collector.get_data_statistics()
    print("\n" + "="*70)
    print("DATA COLLECTION STATISTICS")
    print("="*70)
    for key, value in stats.items():
        print(f"{key}: {value}")
    print("="*70)

if __name__ == '__main__':
    main()
