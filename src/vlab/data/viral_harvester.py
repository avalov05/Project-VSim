"""
Comprehensive Viral Genome Harvester for VLab
Collects all viable viral genomes and generates non-viable genomes for training
"""

import asyncio
import aiohttp
from aiohttp import TCPConnector
import backoff
from Bio import Entrez, SeqIO
import xml.etree.ElementTree as ET
import pandas as pd
import json
import io
import os
import time
import ssl
import urllib.request as urlrequest
import gzip
import pickle
import logging
import sys
from datetime import datetime
from tqdm import tqdm
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import random
from typing import List, Dict, Any, Set
import numpy as np

try:
    import certifi
except Exception:
    certifi = None

logger = logging.getLogger(__name__)

class ViralGenomeHarvester:
    """Harvest all viable viral genomes from NCBI and other sources"""
    
    def __init__(self, email="vlab@example.com", api_key=None, max_concurrent=8):
        self.email = email
        self.api_key = api_key
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
        Entrez.email = email
        if api_key:
            Entrez.api_key = api_key
        
        # SSL setup
        ca_bundle = os.getenv("SSL_CERT_FILE") or os.getenv("REQUESTS_CA_BUNDLE")
        if not ca_bundle and certifi is not None:
            ca_bundle = certifi.where()
        if ca_bundle and os.path.exists(ca_bundle):
            ctx = ssl.create_default_context()
            ctx.load_verify_locations(cafile=ca_bundle)
            opener = urlrequest.build_opener(urlrequest.HTTPSHandler(context=ctx))
            urlrequest.install_opener(opener)
        
        self.ssl_context = ssl._create_unverified_context()
        
        # Data directories
        self.data_dir = Path("data/training")
        self.viable_dir = self.data_dir / "train" / "viable"
        self.non_viable_dir = self.data_dir / "train" / "non_viable"
        self.val_viable_dir = self.data_dir / "val" / "viable"
        self.val_non_viable_dir = self.data_dir / "val" / "non_viable"
        
        for dir in [self.viable_dir, self.non_viable_dir, self.val_viable_dir, self.val_non_viable_dir]:
            dir.mkdir(parents=True, exist_ok=True)
        
        # Checkpointing
        self.checkpoint_file = self.data_dir / "harvest_checkpoint.json"
        self.completed_ids: Set[str] = set()
        self._load_checkpoint()
    
    def _load_checkpoint(self):
        """Load checkpoint"""
        try:
            if self.checkpoint_file.exists():
                with open(self.checkpoint_file, 'r') as f:
                    checkpoint = json.load(f)
                    self.completed_ids = set(checkpoint.get('completed_ids', []))
                logger.info(f"Resumed with {len(self.completed_ids)} completed sequences")
        except Exception as e:
            logger.warning(f"Checkpoint load failed: {e}")
    
    def _save_checkpoint(self):
        """Save checkpoint"""
        try:
            checkpoint = {
                'completed_ids': list(self.completed_ids),
                'last_update': datetime.now().isoformat()
            }
            with open(self.checkpoint_file, 'w') as f:
                json.dump(checkpoint, f)
        except Exception as e:
            logger.warning(f"Checkpoint save failed: {e}")
    
    async def harvest_all_viable_genomes(self, max_sequences=250000):
        """Harvest all viable viral genomes from NCBI"""
        logger.info(f"Starting harvest of up to {max_sequences} viable viral genomes")
        
        # Comprehensive search terms for viable genomes
        search_terms = [
            # Tier 1: Highest quality - RefSeq complete genomes
            "viruses[ORGN] AND complete genome[TITLE] AND refseq[FILTER]",
            
            # Tier 2: Complete genomes by major families
            "viruses[ORGN] AND complete genome[TITLE] AND (Adenoviridae[ORGN] OR Herpesviridae[ORGN] OR Poxviridae[ORGN] OR Retroviridae[ORGN] OR Flaviviridae[ORGN] OR Coronaviridae[ORGN] OR Picornaviridae[ORGN] OR Parvoviridae[ORGN] OR Rhabdoviridae[ORGN] OR Paramyxoviridae[ORGN])",
            
            # Tier 3: Broad complete genomes
            "viruses[ORGN] AND complete genome[TITLE]",
            
            # Tier 4: Representative genomes
            "viruses[ORGN] AND representative genome[TITLE]",
            
            # Tier 5: Reference genomes
            "viruses[ORGN] AND reference genome[TITLE]",
        ]
        
        all_ids = set()
        
        for search_term in search_terms:
            if len(all_ids) >= max_sequences:
                break
            
            try:
                logger.info(f"Searching: {search_term[:80]}...")
                
                # Use WebEnv for pagination
                search_handle = Entrez.esearch(
                    db="nucleotide",
                    term=search_term,
                    usehistory="y",
                    retmax=1
                )
                search_results = Entrez.read(search_handle)
                search_handle.close()
                
                webenv = search_results["WebEnv"]
                query_key = search_results["QueryKey"]
                total_count = int(search_results["Count"])
                
                logger.info(f"Found {total_count:,} sequences")
                
                if total_count == 0:
                    continue
                
                # Fetch IDs in batches (NCBI limit: 10,000 per request)
                remaining_slots = max_sequences - len(all_ids)
                ids_to_fetch = min(total_count, remaining_slots)  # No per-search limit - collect all available
                
                fetched = 0
                batch_size = 10000
                
                while fetched < ids_to_fetch:
                    current_batch = min(batch_size, ids_to_fetch - fetched)
                    
                    fetch_handle = Entrez.efetch(
                        db="nucleotide",
                        rettype="acc",
                        retmode="text",
                        retstart=fetched,
                        retmax=current_batch,
                        webenv=webenv,
                        query_key=query_key
                    )
                    
                    id_text = fetch_handle.read()
                    fetch_handle.close()
                    
                    new_ids = [id.strip() for id in id_text.split('\n') if id.strip()]
                    all_ids.update(new_ids)
                    fetched += len(new_ids)
                    
                    logger.info(f"  Fetched {fetched:,}/{ids_to_fetch:,} IDs")
                    await asyncio.sleep(0.5)
            
            except Exception as e:
                logger.error(f"Search failed: {e}")
                continue
        
        # Filter out completed IDs
        remaining_ids = [id for id in all_ids if id not in self.completed_ids]
        logger.info(f"Downloading {len(remaining_ids):,} new sequences")
        
        # Download sequences
        sequences = await self._harvest_sequences_async(remaining_ids[:max_sequences])
        
        # Save viable genomes
        self._save_viable_genomes(sequences)
        
        return sequences
    
    @backoff.on_exception(backoff.expo, (aiohttp.ClientError, asyncio.TimeoutError), max_tries=10)
    async def _fetch_batch_async(self, session, batch_ids, batch_num):
        """Fetch a batch of sequences"""
        async with self.semaphore:
            url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
            params = {
                'db': 'nucleotide',
                'id': ','.join(batch_ids),
                'rettype': 'fasta',
                'retmode': 'text',
            }
            if self.api_key:
                params['api_key'] = self.api_key
            
            timeout = aiohttp.ClientTimeout(total=600)
            
            try:
                async with session.get(url, params=params, timeout=timeout) as response:
                    if response.status == 200:
                        data = await response.text()
                        return self._parse_fasta_batch(data, batch_ids)
                    elif response.status == 429:
                        await asyncio.sleep(60)
                        raise Exception("HTTP 429")
                    else:
                        await asyncio.sleep(10)
                        raise Exception(f"HTTP {response.status}")
            except Exception as e:
                logger.warning(f"Batch {batch_num} failed: {e}")
                raise
    
    async def _harvest_sequences_async(self, id_list, batch_size=200):
        """Async download of sequences"""
        remaining_ids = [id for id in id_list if id not in self.completed_ids]
        
        logger.info(f"Downloading {len(remaining_ids)} sequences")
        
        connector = TCPConnector(limit=self.max_concurrent, limit_per_host=50, ssl=self.ssl_context)
        sequences = []
        
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = []
            for i in range(0, len(remaining_ids), batch_size):
                batch_ids = remaining_ids[i:i + batch_size]
                task = self._fetch_batch_async(session, batch_ids, i//batch_size + 1)
                tasks.append(task)
                
                if i > 0 and (i // batch_size) % 10 == 0:
                    await asyncio.sleep(5)
            
            for future in asyncio.as_completed(tasks):
                try:
                    batch_result = await future
                    sequences.extend(batch_result)
                    
                    # Update checkpoint
                    for seq in batch_result:
                        acc = seq.get('accession')
                        if acc:
                            self.completed_ids.add(acc)
                    
                    if len(sequences) % 1000 == 0:
                        self._save_checkpoint()
                        logger.info(f"Downloaded {len(sequences)} sequences")
                
                except Exception as e:
                    logger.warning(f"Batch failed: {e}")
                    continue
        
        self._save_checkpoint()
        return sequences
    
    def _parse_fasta_batch(self, fasta_data, batch_ids):
        """Parse FASTA data"""
        sequences = []
        try:
            for record in SeqIO.parse(io.StringIO(fasta_data), "fasta"):
                seq_data = {
                    'accession': record.id.split('.')[0],
                    'id': record.id,
                    'description': record.description,
                    'sequence': str(record.seq).upper(),
                    'length': len(record.seq),
                    'source': 'ncbi',
                    'timestamp': datetime.now().isoformat(),
                    'is_viable': True  # All harvested genomes are viable
                }
                sequences.append(seq_data)
        except Exception as e:
            logger.warning(f"Failed to parse batch: {e}")
        return sequences
    
    def _save_viable_genomes(self, sequences):
        """Save viable genomes to training directories"""
        logger.info(f"Saving {len(sequences)} viable genomes")
        
        # Split into train/val (90/10)
        random.shuffle(sequences)
        split_idx = int(len(sequences) * 0.9)
        train_sequences = sequences[:split_idx]
        val_sequences = sequences[split_idx:]
        
        # Save training data
        for i, seq in enumerate(train_sequences):
            filename = f"{seq['accession']}.fasta"
            filepath = self.viable_dir / filename
            with open(filepath, 'w') as f:
                f.write(f">{seq['id']} {seq['description']}\n")
                f.write(seq['sequence'] + '\n')
        
        # Save validation data
        for i, seq in enumerate(val_sequences):
            filename = f"{seq['accession']}.fasta"
            filepath = self.val_viable_dir / filename
            with open(filepath, 'w') as f:
                f.write(f">{seq['id']} {seq['description']}\n")
                f.write(seq['sequence'] + '\n')
        
        logger.info(f"Saved {len(train_sequences)} training and {len(val_sequences)} validation viable genomes")
    
    def generate_non_viable_genomes(self, count=10000, methods=['random', 'fragmented', 'no_start', 'no_stop', 'invalid_codons']):
        """Generate non-viable genomes for training"""
        logger.info(f"Generating {count} non-viable genomes using methods: {methods}")
        
        non_viable_sequences = []
        
        for i in range(count):
            method = random.choice(methods)
            seq = self._generate_non_viable_sequence(method, i)
            non_viable_sequences.append(seq)
        
        # Split into train/val
        random.shuffle(non_viable_sequences)
        split_idx = int(len(non_viable_sequences) * 0.9)
        train_sequences = non_viable_sequences[:split_idx]
        val_sequences = non_viable_sequences[split_idx:]
        
        # Save training data
        for seq in train_sequences:
            filename = f"non_viable_{seq['id']}.fasta"
            filepath = self.non_viable_dir / filename
            with open(filepath, 'w') as f:
                f.write(f">{seq['id']} {seq['description']}\n")
                f.write(seq['sequence'] + '\n')
        
        # Save validation data
        for seq in val_sequences:
            filename = f"non_viable_{seq['id']}.fasta"
            filepath = self.val_non_viable_dir / filename
            with open(filepath, 'w') as f:
                f.write(f">{seq['id']} {seq['description']}\n")
                f.write(seq['sequence'] + '\n')
        
        logger.info(f"Generated {len(train_sequences)} training and {len(val_sequences)} validation non-viable genomes")
        return non_viable_sequences
    
    def _generate_non_viable_sequence(self, method: str, index: int) -> Dict[str, Any]:
        """Generate a non-viable sequence using specified method"""
        bases = 'ATCG'
        
        if method == 'random':
            # Completely random sequence - no structure
            length = random.randint(1000, 50000)
            sequence = ''.join(random.choice(bases) for _ in range(length))
            description = f"synthetic_non_viable_random_{index}"
        
        elif method == 'fragmented':
            # Fragmented genome - missing essential parts
            length = random.randint(5000, 30000)
            sequence = ''.join(random.choice(bases) for _ in range(length))
            # Remove start codons
            sequence = sequence.replace('ATG', 'XXX')
            description = f"synthetic_non_viable_fragmented_{index}"
        
        elif method == 'no_start':
            # No start codons
            length = random.randint(3000, 20000)
            sequence = ''.join(random.choice(bases) for _ in range(length))
            # Ensure no ATG
            sequence = sequence.replace('ATG', 'AAA')
            description = f"synthetic_non_viable_no_start_{index}"
        
        elif method == 'no_stop':
            # No stop codons - genes can't terminate
            length = random.randint(3000, 20000)
            sequence = ''.join(random.choice(bases) for _ in range(length))
            # Remove stop codons
            sequence = sequence.replace('TAA', 'AAA')
            sequence = sequence.replace('TAG', 'AAA')
            sequence = sequence.replace('TGA', 'AAA')
            description = f"synthetic_non_viable_no_stop_{index}"
        
        elif method == 'invalid_codons':
            # Invalid codon usage - won't translate properly
            length = random.randint(3000, 20000)
            sequence = ''.join(random.choice(bases) for _ in range(length))
            # Add N's and invalid bases
            invalid_positions = random.sample(range(len(sequence)), len(sequence) // 20)
            seq_list = list(sequence)
            for pos in invalid_positions:
                seq_list[pos] = 'N'
            sequence = ''.join(seq_list)
            description = f"synthetic_non_viable_invalid_codons_{index}"
        
        elif method == 'too_short':
            # Too short to be functional
            length = random.randint(100, 999)
            sequence = ''.join(random.choice(bases) for _ in range(length))
            description = f"synthetic_non_viable_too_short_{index}"
        
        elif method == 'missing_genes':
            # Missing essential genes - only partial sequences
            length = random.randint(1000, 5000)
            sequence = ''.join(random.choice(bases) for _ in range(length))
            # Make it look like fragments
            sequence = 'N' * 100 + sequence + 'N' * 100
            description = f"synthetic_non_viable_missing_genes_{index}"
        
        else:
            # Default: random
            length = random.randint(1000, 50000)
            sequence = ''.join(random.choice(bases) for _ in range(length))
            description = f"synthetic_non_viable_{method}_{index}"
        
        return {
            'id': description,
            'accession': f"SYNTH_{index:08d}",
            'description': description,
            'sequence': sequence,
            'length': len(sequence),
            'source': 'synthetic',
            'timestamp': datetime.now().isoformat(),
            'is_viable': False,
            'non_viable_method': method
        }
    
    def generate_from_viable(self, viable_sequences: List[Dict[str, Any]], count: int):
        """Generate non-viable genomes by mutating viable ones"""
        logger.info(f"Generating {count} non-viable genomes from {len(viable_sequences)} viable genomes")
        
        non_viable = []
        
        for i in range(count):
            # Pick a random viable sequence
            source = random.choice(viable_sequences)
            seq = source['sequence']
            
            # Apply mutations to make it non-viable
            mutation_type = random.choice(['delete_start', 'insert_stop', 'frame_shift', 'corrupt'])
            
            if mutation_type == 'delete_start':
                # Remove all start codons
                mutated = seq.replace('ATG', 'AAA')
            elif mutation_type == 'insert_stop':
                # Insert stop codons early
                if len(seq) > 100:
                    pos = random.randint(50, min(500, len(seq) - 50))
                    mutated = seq[:pos] + 'TAA' + seq[pos+3:]
                else:
                    mutated = seq
            elif mutation_type == 'frame_shift':
                # Insert or delete bases to shift reading frame
                bases = 'ATCG'
                if len(seq) > 100:
                    pos = random.randint(50, len(seq) - 50)
                    if random.random() > 0.5:
                        # Insert
                        mutated = seq[:pos] + random.choice(bases) + seq[pos:]
                    else:
                        # Delete
                        mutated = seq[:pos] + seq[pos+1:]
                else:
                    mutated = seq
            elif mutation_type == 'corrupt':
                # Randomly corrupt sequence
                bases = 'ATCG'
                corrupted_positions = random.sample(range(len(seq)), min(len(seq) // 10, 1000))
                seq_list = list(seq)
                for pos in corrupted_positions:
                    seq_list[pos] = random.choice(bases)
                mutated = ''.join(seq_list)
            else:
                mutated = seq
            
            non_viable.append({
                'id': f"mutated_{source['accession']}_{i}",
                'accession': f"MUT_{source['accession']}_{i}",
                'description': f"mutated_non_viable_{mutation_type}_{source['accession']}",
                'sequence': mutated,
                'length': len(mutated),
                'source': 'mutated',
                'timestamp': datetime.now().isoformat(),
                'is_viable': False,
                'non_viable_method': mutation_type,
                'source_accession': source['accession']
            })
        
        # Save
        random.shuffle(non_viable)
        split_idx = int(len(non_viable) * 0.9)
        train_sequences = non_viable[:split_idx]
        val_sequences = non_viable[split_idx:]
        
        for seq in train_sequences:
            filename = f"non_viable_{seq['accession']}.fasta"
            filepath = self.non_viable_dir / filename
            with open(filepath, 'w') as f:
                f.write(f">{seq['id']} {seq['description']}\n")
                f.write(seq['sequence'] + '\n')
        
        for seq in val_sequences:
            filename = f"non_viable_{seq['accession']}.fasta"
            filepath = self.val_non_viable_dir / filename
            with open(filepath, 'w') as f:
                f.write(f">{seq['id']} {seq['description']}\n")
                f.write(seq['sequence'] + '\n')
        
        logger.info(f"Generated {len(train_sequences)} training and {len(val_sequences)} validation mutated non-viable genomes")
        return non_viable

async def main():
    """Main data collection function"""
    harvester = ViralGenomeHarvester(
        email="vlab@example.com",  # Change to your email
        api_key=None,  # Optional: add your NCBI API key
        max_concurrent=8
    )
    
    # Step 1: Harvest all viable genomes
    logger.info("="*70)
    logger.info("STEP 1: Harvesting viable viral genomes")
    logger.info("="*70)
    viable_sequences = await harvester.harvest_all_viable_genomes(max_sequences=50000)
    
    # Step 2: Generate synthetic non-viable genomes
    logger.info("="*70)
    logger.info("STEP 2: Generating synthetic non-viable genomes")
    logger.info("="*70)
    harvester.generate_non_viable_genomes(
        count=10000,
        methods=['random', 'fragmented', 'no_start', 'no_stop', 'invalid_codons', 'too_short', 'missing_genes']
    )
    
    # Step 3: Generate non-viable from viable (mutated)
    logger.info("="*70)
    logger.info("STEP 3: Generating mutated non-viable genomes")
    logger.info("="*70)
    harvester.generate_from_viable(viable_sequences, count=5000)
    
    logger.info("="*70)
    logger.info("DATA COLLECTION COMPLETE!")
    logger.info("="*70)
    logger.info(f"Viable genomes: {len(viable_sequences)}")
    logger.info(f"Training data ready in: {harvester.data_dir}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())

