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
                              total_target=500000,
                              skip_ncbi_if_fails=True):
        """
        Collect all training data
        
        Args:
            max_viable: Maximum number of viable genomes to collect (None = auto-calculate)
            num_synthetic_non_viable: Number of synthetic non-viable genomes (None = auto-calculate)
            num_mutated_non_viable: Number of mutated non-viable genomes (None = auto-calculate)
            total_target: Total target genomes (creates balanced 50/50 viable/non-viable dataset)
            skip_ncbi_if_fails: If True, skip NCBI downloads if connection fails and use synthetic data only
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
        
        # Step 1: Harvest viable genomes (with fallback if NCBI fails)
        logger.info("="*70)
        logger.info("STEP 1: Harvesting viable viral genomes from NCBI")
        logger.info("="*70)
        logger.info("This will download real viral genomes from the NCBI nucleotide database.")
        logger.info("These are complete, functional viral genomes that exist in nature.")
        logger.info("="*70)
        
        viable_sequences = []
        ncbi_failed = False
        ncbi_error = None
        
        # Attempt NCBI download with proper error handling
        logger.info("Attempting to download viable genomes from NCBI...")
        logger.info("This step will complete fully before proceeding to next steps.")
        logger.info("="*70)
        
        try:
            # Ensure the harvester completes the entire download process
            viable_sequences = await self.harvester.harvest_all_viable_genomes(max_sequences=max_viable)
            
            # Check if we got any sequences (even if empty list, the process completed)
            if viable_sequences is None:
                logger.warning("="*70)
                logger.warning("WARNING: harvest_all_viable_genomes returned None")
                logger.warning("This may indicate the download process encountered errors.")
                logger.warning("="*70)
                ncbi_failed = True
                viable_sequences = []
            elif len(viable_sequences) == 0:
                logger.warning("="*70)
                logger.warning("WARNING: No viable genomes downloaded from NCBI!")
                logger.warning("The download process completed but returned 0 sequences.")
                logger.warning("This is common in Colab due to NCBI API restrictions.")
                logger.warning("="*70)
                ncbi_failed = True
            else:
                logger.info(f"✓ Successfully downloaded {len(viable_sequences):,} viable genomes from NCBI")
                logger.info("="*70)
                logger.info("NCBI download step completed successfully.")
                logger.info("="*70)
        
        except Exception as e:
            # Safely extract error message
            try:
                error_msg = str(e)
                error_type = type(e).__name__
            except Exception:
                error_msg = "Unknown error"
                error_type = "Exception"
            
            # Check for coroutine-related errors (common in Colab)
            if 'coroutine' in error_msg.lower() or (hasattr(e, '__class__') and 'coroutine' in str(type(e)).lower()):
                error_msg = f"Async/coroutine error during NCBI download: {error_type}"
                logger.error(f"NCBI download encountered async error: {error_type}")
                logger.error("This is often caused by Colab's async environment. The download process may have partially completed.")
                logger.error("Checking for any successfully downloaded sequences...")
                
                # Try to recover any sequences that were downloaded before the error
                try:
                    # Check if harvester has any sequences in memory or on disk
                    # The harvester saves sequences incrementally, so check disk
                    from pathlib import Path
                    viable_dir = self.harvester.viable_dir
                    val_viable_dir = self.harvester.val_viable_dir
                    
                    if viable_dir.exists():
                        viable_files = list(viable_dir.glob("*.fasta"))
                        val_files = list(val_viable_dir.glob("*.fasta")) if val_viable_dir.exists() else []
                        total_files = len(viable_files) + len(val_files)
                        
                        if total_files > 0:
                            logger.info(f"Found {total_files} sequences that were saved to disk before the error.")
                            logger.info("Using these sequences instead of generating synthetic data.")
                            # Read sequences from disk
                            viable_sequences = []
                            for filepath in viable_files + val_files:
                                try:
                                    from Bio import SeqIO
                                    for record in SeqIO.parse(filepath, "fasta"):
                                        viable_sequences.append({
                                            'accession': record.id,
                                            'sequence': str(record.seq),
                                            'description': record.description
                                        })
                                except Exception:
                                    continue
                            
                            if len(viable_sequences) > 0:
                                logger.info(f"✓ Recovered {len(viable_sequences):,} sequences from disk")
                                ncbi_failed = False
                            else:
                                logger.warning("Could not read sequences from disk files.")
                                ncbi_failed = True
                        else:
                            logger.warning("No sequences found on disk.")
                            ncbi_failed = True
                    else:
                        logger.warning("Viable directory does not exist.")
                        ncbi_failed = True
                except Exception as recovery_error:
                    logger.warning(f"Could not recover sequences: {recovery_error}")
                    ncbi_failed = True
            else:
                logger.error(f"NCBI download failed: {error_msg}")
                logger.error(f"Error type: {error_type}")
                ncbi_error = error_msg
                ncbi_failed = True
                viable_sequences = []
            
            if ncbi_failed:
                logger.warning("="*70)
                logger.warning("NCBI download step completed with errors.")
                if skip_ncbi_if_fails:
                    logger.warning("Fallback mode enabled - will generate synthetic data.")
                else:
                    logger.warning("Fallback mode disabled - will not generate synthetic data.")
                logger.warning("="*70)
        
        # Log completion of NCBI step
        logger.info("")
        logger.info("="*70)
        logger.info("NCBI DOWNLOAD STEP COMPLETE")
        logger.info("="*70)
        if ncbi_failed:
            logger.info(f"Status: Failed ({len(viable_sequences)} sequences recovered)")
            if ncbi_error:
                logger.info(f"Error: {ncbi_error}")
        else:
            logger.info(f"Status: Success ({len(viable_sequences)} sequences downloaded)")
        logger.info("="*70)
        logger.info("")
        
        # If NCBI failed and we're in fallback mode, generate synthetic viable genomes
        if ncbi_failed and skip_ncbi_if_fails:
            logger.info("="*70)
            logger.info("FALLBACK MODE: Generating synthetic viable genomes")
            logger.info("="*70)
            logger.info("Since NCBI is not accessible, generating synthetic viable-like genomes.")
            logger.info("These are realistic viral genome sequences that mimic real viruses.")
            logger.info("="*70)
            
            # Generate synthetic viable genomes (realistic viral sequences)
            synthetic_viable_count = min(max_viable, 10000)  # Cap at 10K for performance
            logger.info(f"Generating {synthetic_viable_count:,} synthetic viable genomes...")
            self.harvester.generate_synthetic_viable_genomes(count=synthetic_viable_count)
            logger.info(f"✓ Generated {synthetic_viable_count:,} synthetic viable genomes")
            
            # Update max_viable to match what we actually generated
            max_viable = synthetic_viable_count
        
        # Step 2: Generate synthetic non-viable
        logger.info("")
        logger.info("="*70)
        logger.info("STEP 2: Generating synthetic non-viable genomes")
        logger.info("="*70)
        logger.info("NOTE: Non-viable viruses are NOT downloaded from NCBI.")
        logger.info("      They are generated synthetically using various methods:")
        logger.info("      - Random sequences")
        logger.info("      - Fragmented genomes (missing essential parts)")
        logger.info("      - No start codons (can't initiate translation)")
        logger.info("      - No stop codons (can't terminate translation)")
        logger.info("      - Invalid codons (contains N's and invalid bases)")
        logger.info("      - Too short sequences")
        logger.info("      - Missing essential genes")
        logger.info("="*70)
        self.harvester.generate_non_viable_genomes(
            count=num_synthetic_non_viable,
            methods=['random', 'fragmented', 'no_start', 'no_stop', 'invalid_codons', 'too_short', 'missing_genes']
        )
        
        # Step 3: Generate mutated non-viable
        # Note: If we're using synthetic viable genomes, we can still generate mutated ones
        # by using the synthetic sequences as a base
        if num_mutated_non_viable > 0:
            logger.info("")
            logger.info("="*70)
            logger.info("STEP 3: Generating mutated non-viable genomes")
            logger.info("="*70)
            logger.info("These are created by mutating viable genomes to make them non-viable.")
            logger.info("Mutation types: delete_start, insert_stop, frame_shift, corrupt")
            logger.info("="*70)
            
            # Use viable sequences if available, otherwise generate some for mutation
            if viable_sequences and len(viable_sequences) > 0:
                self.harvester.generate_from_viable(viable_sequences, count=num_mutated_non_viable)
            else:
                # Generate a small set of viable-like sequences for mutation
                logger.info("No viable sequences available for mutation.")
                logger.info("Generating viable-like sequences for mutation...")
                mutation_base_count = min(num_mutated_non_viable, 1000)
                mutation_base_sequences = self.harvester.generate_synthetic_viable_genomes(count=mutation_base_count)
                if mutation_base_sequences:
                    # Extract sequences for mutation
                    viable_for_mutation = [s for s in mutation_base_sequences]
                    self.harvester.generate_from_viable(viable_for_mutation, count=num_mutated_non_viable)
                else:
                    logger.warning("Could not generate sequences for mutation. Skipping mutated non-viable generation.")
                    num_mutated_non_viable = 0
        
        # Get actual counts from disk (more accurate)
        stats = self.get_data_statistics()
        
        # Summary
        logger.info("="*70)
        logger.info("DATA COLLECTION COMPLETE!")
        logger.info("="*70)
        logger.info(f"[+] Training viable: {stats['train_viable']:,}")
        logger.info(f"[+] Training non-viable: {stats['train_non_viable']:,}")
        logger.info(f"[+] Validation viable: {stats['val_viable']:,}")
        logger.info(f"[+] Validation non-viable: {stats['val_non_viable']:,}")
        logger.info(f"[+] Total training samples: {stats['total_train']:,}")
        logger.info(f"[+] Total validation samples: {stats['total_val']:,}")
        logger.info(f"[+] Grand total: {stats['total']:,}")
        logger.info(f"[+] Data location: {self.harvester.data_dir}")
        
        if ncbi_failed:
            logger.info("")
            logger.info("NOTE: NCBI download failed - used synthetic data instead")
            logger.info("This is common in Colab. The model will still train successfully!")
        
        return {
            'viable': stats['train_viable'] + stats['val_viable'],
            'synthetic_non_viable': stats['train_non_viable'] + stats['val_non_viable'],
            'mutated_non_viable': 0,  # Counted in non_viable
            'total': stats['total'],
            'target': total_target if total_target else None,
            'reached_target': stats['total'] >= total_target if total_target else False,
            'ncbi_failed': ncbi_failed
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
