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
import hashlib
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
        
        # Configure Entrez with email and API key (required by NCBI)
        Entrez.email = email
        if api_key:
            Entrez.api_key = api_key
        else:
            logger.warning("No API key provided. NCBI requests will be slower (3 requests/second vs 10 requests/second with API key).")
            logger.warning("Get an API key from: https://www.ncbi.nlm.nih.gov/account/settings/")
        
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
        """Load checkpoint and verify files exist"""
        try:
            if self.checkpoint_file.exists():
                with open(self.checkpoint_file, 'r') as f:
                    checkpoint = json.load(f)
                    checkpoint_ids = set(checkpoint.get('completed_ids', []))
                
                # Verify files actually exist on disk
                logger.info(f"Loading checkpoint: {len(checkpoint_ids):,} IDs marked as completed")
                logger.info("Verifying files exist on disk...")
                
                verified_ids = set()
                missing_count = 0
                
                # Check both train and val directories
                for acc_id in checkpoint_ids:
                    train_file = self.viable_dir / f"{acc_id}.fasta"
                    val_file = self.val_viable_dir / f"{acc_id}.fasta"
                    if train_file.exists() or val_file.exists():
                        verified_ids.add(acc_id)
                    else:
                        missing_count += 1
                        # Remove from checkpoint if file doesn't exist
                        if missing_count <= 10:  # Log first 10
                            logger.debug(f"File missing for {acc_id}, will re-download")
                
                self.completed_ids = verified_ids
                
                if missing_count > 0:
                    logger.warning(f"Found {missing_count:,} IDs in checkpoint without files. Will re-download.")
                    # Update checkpoint to remove missing IDs
                    self._save_checkpoint()
                else:
                    logger.info(f"All {len(verified_ids):,} checkpoint IDs verified on disk")
                    
                logger.info(f"Resumed with {len(self.completed_ids):,} verified completed sequences")
        except Exception as e:
            logger.warning(f"Checkpoint load failed: {e}")
            self.completed_ids = set()
    
    def _save_checkpoint(self):
        """Save checkpoint atomically"""
        try:
            checkpoint = {
                'completed_ids': list(self.completed_ids),
                'last_update': datetime.now().isoformat()
            }
            # Atomic write: write to temp file first, then rename
            temp_file = self.checkpoint_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(checkpoint, f)
            # Atomic rename (works on Windows too)
            temp_file.replace(self.checkpoint_file)
        except Exception as e:
            logger.warning(f"Checkpoint save failed: {e}")
    
    async def _test_ncbi_connection(self):
        """Test if NCBI Entrez API is accessible"""
        try:
            logger.info("Testing NCBI connection with simple query...")
            # Very simple test query (no usehistory to avoid WebEnv issues)
            test_handle = Entrez.esearch(
                db="nucleotide",
                term="NC_045512",  # SARS-CoV-2 reference genome
                retmax=1,
                usehistory="n"  # Don't use history for simple test
            )
            test_results = Entrez.read(test_handle)
            test_handle.close()
            
            # Check if we got valid results
            if "Count" not in test_results:
                logger.warning("⚠ NCBI connection test: Invalid response format")
                return False
                
            count = int(test_results.get("Count", 0))
            if count > 0:
                logger.info(f"✓ NCBI connection test successful (found {count} results)")
                return True
            else:
                logger.warning("⚠ NCBI connection test returned 0 results")
                return False
        except Exception as e:
            logger.error(f"✗ NCBI connection test failed: {e}")
            logger.error(f"  Error type: {type(e).__name__}")
            logger.error("  NCBI Entrez API may not be accessible from this environment")
            logger.error("  Possible causes:")
            logger.error("    - Network firewall blocking NCBI")
            logger.error("    - NCBI blocking this IP address")
            logger.error("    - NCBI server issues")
            logger.error("    - Rate limiting")
            return False
    
    async def _search_with_retry(self, search_term, max_retries=3, delay=5):
        """Search with retry logic and better error handling"""
        original_search_term = search_term
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Searching (attempt {attempt+1}/{max_retries}): {search_term[:80]}...")
                
                # Add delay between attempts (longer delay for retries)
                if attempt > 0:
                    wait_time = delay * (2 ** attempt)  # Exponential backoff: 10s, 20s, 40s
                    logger.info(f"  Waiting {wait_time}s before retry (rate limiting protection)...")
                    await asyncio.sleep(wait_time)
                else:
                    # Longer initial delay to avoid rate limits
                    await asyncio.sleep(2)
                
                # Try without usehistory first (more reliable in Colab)
                # NCBI API in Colab sometimes doesn't return WebEnv properly
                try:
                    logger.debug(f"  Trying search without usehistory (more reliable)...")
                    search_handle = Entrez.esearch(
                        db="nucleotide",
                        term=search_term,
                        usehistory="n",  # Don't use history - more reliable
                        retmax=100  # Get first 100 IDs directly (can be increased)
                    )
                    search_results = Entrez.read(search_handle)
                    search_handle.close()
                    
                    total_count = int(search_results.get("Count", 0))
                    
                    if total_count == 0:
                        logger.warning(f"  ⚠ Search returned 0 results")
                        return None, None, 0
                    
                    # Get IDs directly from IdList
                    if "IdList" in search_results and len(search_results["IdList"]) > 0:
                        id_list = search_results["IdList"]
                        logger.info(f"  ✓ Found {total_count:,} sequences (got {len(id_list)} IDs directly)")
                        # Return IDs directly - we'll handle this in the calling function
                        return "DIRECT", id_list, total_count
                    else:
                        logger.warning(f"  ⚠ No IDs in response despite Count={total_count}")
                        # Try with usehistory as fallback
                        logger.debug(f"  Trying with usehistory as fallback...")
                        raise KeyError("IdList")
                        
                except (KeyError, Exception) as e:
                    # If direct mode fails, try with usehistory
                    if "IdList" in str(e) or "WebEnv" in str(e) or isinstance(e, KeyError):
                        try:
                            logger.debug(f"  Trying search with usehistory (fallback)...")
                            search_handle = Entrez.esearch(
                                db="nucleotide",
                                term=search_term,
                                usehistory="y",
                                retmax=1
                            )
                            search_results = Entrez.read(search_handle)
                            search_handle.close()
                            
                            # Check if we got valid results with WebEnv
                            if "WebEnv" not in search_results:
                                logger.warning(f"  ⚠ WebEnv not in response even with usehistory=y")
                                raise KeyError("WebEnv not available")
                            
                            webenv = search_results["WebEnv"]
                            query_key = search_results["QueryKey"]
                            total_count = int(search_results.get("Count", 0))
                            
                            if total_count == 0:
                                logger.warning(f"  ⚠ Search returned 0 results")
                                return None, None, 0
                            
                            logger.info(f"  ✓ Found {total_count:,} sequences (using WebEnv)")
                            return webenv, query_key, total_count
                        except Exception as e2:
                            # Both methods failed
                            logger.warning(f"  ⚠ Both direct and usehistory methods failed: {e2}")
                            raise e2  # Re-raise to be handled by outer exception handler
                    else:
                        # Some other error - re-raise
                        raise
                
            except Exception as e:
                error_msg = str(e)
                error_type = type(e).__name__
                
                if "400" in error_msg or "Bad Request" in error_msg:
                    logger.warning(f"  ⚠ HTTP 400 Bad Request: {error_msg}")
                    
                    # Try progressively simpler queries
                    if attempt < max_retries - 1:
                        if "AND" in search_term and "[ORGN]" in search_term:
                            # Try removing AND clauses
                            if "[TITLE]" in search_term:
                                search_term = search_term.replace("[TITLE]", "")
                                logger.info(f"    Retrying with simplified query (removed [TITLE]): {search_term[:80]}...")
                            elif "complete genome" in search_term.lower() or "genome" in search_term.lower():
                                # Try even simpler - just viruses
                                search_term = "viruses[ORGN]"
                                logger.info(f"    Retrying with simplest query: {search_term[:80]}...")
                            else:
                                # Last resort - very basic query
                                search_term = "viruses[ORGN]"
                                logger.info(f"    Retrying with basic query: {search_term[:80]}...")
                    else:
                        logger.error(f"    All retry attempts failed for: {original_search_term[:80]}...")
                elif "429" in error_msg or "Rate" in error_msg:
                    logger.warning(f"  ⚠ Rate limited: {error_msg}")
                    wait_time = 60 * (attempt + 1)  # Wait 60s, 120s, 180s
                    logger.warning(f"    Waiting {wait_time}s before retry...")
                    await asyncio.sleep(wait_time)
                elif "WebEnv" in error_msg or error_type == "KeyError":
                    logger.warning(f"  ⚠ Search response error: {error_msg}")
                    logger.warning(f"    This may indicate NCBI API format issues or connection problems")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(5)  # Wait before retry
                    else:
                        logger.error(f"    Search failed after {max_retries} attempts due to response format issues")
                else:
                    logger.warning(f"  ⚠ Search error ({error_type}): {error_msg}")
                
                if attempt == max_retries - 1:
                    logger.error(f"  ✗ Search failed after {max_retries} attempts")
                    logger.error(f"    Error: {error_type}: {error_msg}")
                    logger.error(f"    Final search term tried: {search_term[:80]}...")
                    return None, None, 0
        
        return None, None, 0
    
    async def harvest_all_viable_genomes(self, max_sequences=250000):
        """Harvest all viable viral genomes from NCBI"""
        logger.info(f"Starting harvest of up to {max_sequences} viable viral genomes")
        logger.info("="*70)
        logger.info("NOTE: NCBI API may return HTTP 400 errors due to:")
        logger.info("  - Rate limiting (temporary - will retry automatically)")
        logger.info("  - Server load (temporary - will retry automatically)")
        logger.info("  - Query complexity (will automatically simplify queries)")
        logger.info("  - Network/firewall issues (Colab IPs may be restricted)")
        logger.info("The system will retry with exponential backoff and simplified queries.")
        logger.info("Some searches may fail, but the system will continue with successful ones.")
        logger.info("="*70)
        
        # Test NCBI connection first
        logger.info("\nTesting NCBI connection...")
        connection_ok = await self._test_ncbi_connection()
        if not connection_ok:
            logger.warning("="*70)
            logger.warning("WARNING: NCBI connection test failed!")
            logger.warning("NCBI Entrez API may not be accessible from Colab.")
            logger.warning("The system will continue but may not be able to download genomes.")
            logger.warning("="*70)
            logger.warning("SOLUTIONS:")
            logger.warning("  1. Wait a few minutes and retry (temporary NCBI issues)")
            logger.warning("  2. Use a smaller dataset (TOTAL_TARGET = 10000)")
            logger.warning("  3. The system will generate synthetic data regardless")
            logger.warning("="*70)
        else:
            logger.info("✓ NCBI connection test passed - proceeding with searches")
        
        # Very simple search terms to maximize success rate
        # Start with the simplest possible queries
        search_terms = [
            # Tier 1: Simplest possible - just viruses
            "viruses[ORGN]",
            
            # Tier 2: Add genome keyword (broader)
            "viruses[ORGN] AND genome",
        ]
        
        # Only add more complex queries if simple ones work
        # These will be tried if the simple ones succeed
        advanced_search_terms = [
            "viruses[ORGN] AND complete",
            "viruses[ORGN] AND representative",
        ]
        
        all_ids = set()
        successful_searches = 0
        
        # Try simple searches first
        for search_term in search_terms:
            if len(all_ids) >= max_sequences:
                break
            
            # Search with retry
            webenv, query_key, total_count = await self._search_with_retry(search_term)
            
            if webenv is None or total_count == 0:
                logger.warning(f"Skipping search term (failed or no results): {search_term[:80]}...")
                continue
            
            successful_searches += 1
            
            try:
                # Handle direct ID list (when WebEnv not available)
                if webenv == "DIRECT" and isinstance(query_key, list):
                    # We got IDs directly from the search
                    new_ids = [str(id).strip() for id in query_key if id and str(id).strip()]
                    all_ids.update(new_ids)
                    logger.info(f"  Fetched {len(new_ids):,} IDs directly (limited batch)")
                    continue
                
                # Fetch IDs in batches using WebEnv (NCBI limit: 10,000 per request)
                remaining_slots = max_sequences - len(all_ids)
                ids_to_fetch = min(total_count, remaining_slots)
                
                fetched = 0
                batch_size = 10000
                
                while fetched < ids_to_fetch:
                    current_batch = min(batch_size, ids_to_fetch - fetched)
                    
                    try:
                        fetch_handle = Entrez.efetch(
                            db="nucleotide",
                            rettype="acc",
                            retmode="text",
                            retstart=fetched,
                            retmax=current_batch,
                            webenv=webenv,
                            query_key=query_key,
                            api_key=self.api_key if self.api_key else None,
                            email=self.email
                        )
                        
                        id_text = fetch_handle.read()
                        fetch_handle.close()
                        
                        new_ids = [id.strip() for id in id_text.split('\n') if id.strip()]
                        all_ids.update(new_ids)
                        fetched += len(new_ids)
                        
                        logger.info(f"  Fetched {fetched:,}/{ids_to_fetch:,} IDs")
                        await asyncio.sleep(0.5)  # Rate limiting
                        
                    except Exception as e:
                        logger.warning(f"  Error fetching IDs batch: {e}")
                        await asyncio.sleep(5)  # Wait before retry
                        if fetched > 0:
                            break  # If we got some IDs, continue to next search term
                        else:
                            raise  # If we got nothing, raise to skip this search term
            
            except Exception as e:
                logger.error(f"Failed to fetch IDs for search term: {e}")
                continue
            
            # Wait between search terms to avoid rate limiting
            await asyncio.sleep(1)
        
        # Log summary of search results
        if len(all_ids) == 0:
            logger.warning("="*70)
            logger.warning("WARNING: No genome IDs found from NCBI searches!")
            logger.warning("This could be due to:")
            logger.warning("  1. All searches failed (HTTP 400 errors)")
            logger.warning("  2. NCBI rate limiting (wait and retry)")
            logger.warning("  3. Network connectivity issues")
            logger.warning("="*70)
            logger.warning("The system will continue with synthetic non-viable genome generation.")
            logger.warning("You may want to:")
            logger.warning("  - Wait a few minutes and retry")
            logger.warning("  - Check your internet connection")
            logger.warning("  - Verify your NCBI API key is valid")
            logger.warning("  - Reduce TOTAL_TARGET to a smaller number for testing")
            logger.warning("="*70)
        else:
            logger.info(f"✓ Successfully found {len(all_ids):,} genome IDs from NCBI")
        
        # Filter out completed IDs - also check if files exist on disk
        # This ensures we don't try to download files that already exist
        verified_completed = set(self.completed_ids)
        for acc_id in all_ids:
            if acc_id in verified_completed:
                continue
            # Check if file exists in either train or val directory
            train_file = self.viable_dir / f"{acc_id}.fasta"
            val_file = self.val_viable_dir / f"{acc_id}.fasta"
            if train_file.exists() or val_file.exists():
                verified_completed.add(acc_id)
        
        remaining_ids = [id for id in all_ids if id not in verified_completed]
        self.completed_ids = verified_completed  # Update checkpoint with verified IDs
        
        logger.info("="*70)
        logger.info("SEARCH SUMMARY")
        logger.info("="*70)
        logger.info(f"Total unique sequence IDs found: {len(all_ids):,}")
        logger.info(f"Already downloaded (verified on disk): {len(verified_completed):,}")
        logger.info(f"New sequences to download: {len(remaining_ids):,}")
        logger.info(f"Target: {max_sequences:,} sequences")
        logger.info("="*70)
        
        # Limit to max_sequences if needed
        if len(remaining_ids) > max_sequences:
            logger.info(f"Limiting download to {max_sequences:,} sequences (first {max_sequences:,} of {len(remaining_ids):,})")
            remaining_ids = remaining_ids[:max_sequences]
        
        # Download sequences (they are saved immediately during download)
        sequences = await self._harvest_sequences_async(remaining_ids)
        
        # Note: Sequences are already saved to disk during download via _save_sequence_immediate
        # We don't need to call _save_viable_genomes here since saving happens incrementally
        logger.info("All sequences have been saved to disk during download.")
        
        return sequences
    
    @backoff.on_exception(
        backoff.expo,
        (aiohttp.ClientError, asyncio.TimeoutError, Exception),
        max_tries=15,
        max_time=300,  # 5 minutes max retry time
        base=2,
        factor=1.5
    )
    async def _fetch_batch_async(self, session, batch_ids, batch_num):
        """Fetch a batch of sequences with improved error handling"""
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
            
            # Increased timeout for large batches
            timeout = aiohttp.ClientTimeout(total=900, connect=60)
            
            try:
                batch_start = time.time()
                async with session.get(url, params=params, timeout=timeout, ssl=self.ssl_context) as response:
                    if response.status == 200:
                        data = await response.text()
                        batch_time = time.time() - batch_start
                        parsed = self._parse_fasta_batch(data, batch_ids)
                        logger.debug(f"Batch {batch_num}: Downloaded {len(parsed)} sequences in {batch_time:.2f}s")
                        return parsed
                    elif response.status == 429:
                        # Rate limited - wait longer
                        wait_time = 120  # 2 minutes
                        logger.warning(f"Batch {batch_num}: Rate limited (429). Waiting {wait_time} seconds...")
                        await asyncio.sleep(wait_time)
                        raise aiohttp.ClientResponseError(
                            request_info=response.request_info,
                            history=response.history,
                            status=429
                        )
                    elif response.status in [400, 502, 503, 504]:
                        # Server errors - retry with exponential backoff
                        wait_time = 30
                        logger.warning(f"Batch {batch_num}: HTTP {response.status} error. Will retry after {wait_time}s...")
                        await asyncio.sleep(wait_time)
                        raise aiohttp.ClientResponseError(
                            request_info=response.request_info,
                            history=response.history,
                            status=response.status
                        )
                    else:
                        # Other errors
                        logger.warning(f"Batch {batch_num}: HTTP {response.status} error. Retrying...")
                        await asyncio.sleep(20)
                        raise aiohttp.ClientResponseError(
                            request_info=response.request_info,
                            history=response.history,
                            status=response.status
                        )
            except (aiohttp.ClientPayloadError, aiohttp.ServerDisconnectedError) as e:
                # Network errors - retry
                logger.debug(f"Batch {batch_num}: Network error {type(e).__name__}. Will retry...")
                await asyncio.sleep(5)
                raise
            except Exception as e:
                logger.debug(f"Batch {batch_num} error: {e} (will retry)")
                await asyncio.sleep(10)
                raise
    
    def _save_sequence_immediate(self, seq: Dict[str, Any], is_training: bool = None):
        """Save a single sequence immediately to disk"""
        try:
            acc = seq.get('accession')
            if not acc:
                return False
            
            # Determine if training or validation (90/10 split based on hash)
            if is_training is None:
                # Use hash of accession for consistent split
                hash_val = int(hashlib.md5(acc.encode()).hexdigest(), 16)
                is_training = (hash_val % 10) < 9  # 90% training
            
            target_dir = self.viable_dir if is_training else self.val_viable_dir
            filename = f"{acc}.fasta"
            filepath = target_dir / filename
            
            # Skip if already exists
            if filepath.exists():
                return True
            
            # Save immediately
            with open(filepath, 'w') as f:
                f.write(f">{seq['id']} {seq['description']}\n")
                f.write(seq['sequence'] + '\n')
            
            return True
        except Exception as e:
            logger.debug(f"Failed to save sequence {seq.get('accession', 'unknown')}: {e}")
            return False
    
    async def _harvest_sequences_async(self, id_list, batch_size=200):
        """Async download of sequences with immediate saving and retry logic"""
        remaining_ids = [id for id in id_list if id not in self.completed_ids]
        total_to_download = len(remaining_ids)
        
        logger.info("="*70)
        logger.info(f"DOWNLOADING VIRAL GENOMES FROM NCBI")
        logger.info("="*70)
        logger.info(f"Total sequences to download: {total_to_download:,}")
        logger.info(f"Batch size: {batch_size} sequences per batch")
        logger.info(f"Concurrent downloads: {self.max_concurrent}")
        logger.info(f"Sequences will be saved IMMEDIATELY after each batch")
        logger.info(f"Starting download...")
        logger.info("="*70)
        
        connector = TCPConnector(limit=self.max_concurrent, limit_per_host=50, ssl=self.ssl_context)
        sequences = []
        failed_batches = []
        failed_batch_ids = {}  # Track which IDs failed in which batch
        saved_count = 0
        start_time = time.time()
        last_checkpoint_time = start_time
        
        # Create progress bar with better settings for real-time updates
        pbar = tqdm(total=total_to_download, desc="Downloading genomes", unit="seq", 
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
                   miniters=1, mininterval=0.5, file=sys.stdout)
        
        async with aiohttp.ClientSession(connector=connector) as session:
            # Create batch tracking - use a list of tuples to preserve order and mapping
            batch_info = []  # List of (batch_num, batch_ids, task)
            total_batches = (len(remaining_ids) + batch_size - 1) // batch_size
            logger.info(f"Creating {total_batches} download batches...")
            sys.stdout.flush()
            
            for i in range(0, len(remaining_ids), batch_size):
                batch_ids = remaining_ids[i:i + batch_size]
                batch_num = i//batch_size + 1
                # Create a task from the coroutine
                coro = self._fetch_batch_async(session, batch_ids, batch_num)
                task = asyncio.create_task(coro)
                batch_info.append((batch_num, batch_ids, task))
            
            logger.info(f"All batches queued. Starting parallel downloads...")
            print(f"\n{'='*70}", flush=True)
            print(f"DOWNLOAD STATUS: Waiting for first batch to complete...", flush=True)
            print(f"{'='*70}\n", flush=True)
            sys.stdout.flush()
            completed_batches = 0
            last_log_count = 0
            first_batch_done = False
            last_update_time = time.time()
            last_save_time = time.time()
            
            # Process completed batches using as_completed with proper task tracking
            # Create a mapping from task to batch info for easy lookup
            task_to_batch = {task: (bn, bid) for bn, bid, task in batch_info}
            
            try:
                # Process tasks as they complete
                for completed_future in asyncio.as_completed([task for _, _, task in batch_info]):
                    batch_num = None
                    batch_ids = None
                    batch_result = None
                    task_error = None
                    
                    try:
                        # Await the completed future to get the result
                        batch_result = await completed_future
                        
                        # Find which batch this task belongs to by checking which task completed
                        # We need to find the task in our batch_info list
                        for bn, bid, task in batch_info:
                            if task.done():
                                try:
                                    # Check if this is the completed task
                                    if task == completed_future or (hasattr(completed_future, 'result') and task.result() == batch_result):
                                        batch_num = bn
                                        batch_ids = bid
                                        break
                                    # Alternative: check by exception status
                                    if task.exception() is None and isinstance(batch_result, list):
                                        # This task completed successfully - verify it matches
                                        batch_num = bn
                                        batch_ids = bid
                                        break
                                except Exception:
                                    continue
                        
                        # If we still don't have batch info, try to match by finding the task
                        if batch_num is None:
                            for bn, bid, task in batch_info:
                                if task.done() and not task.cancelled():
                                    try:
                                        if task.exception() is None:
                                            result = task.result()
                                            if result == batch_result or (isinstance(result, list) and isinstance(batch_result, list) and len(result) == len(batch_result)):
                                                batch_num = bn
                                                batch_ids = bid
                                                batch_result = result
                                                break
                                    except Exception:
                                        pass
                        
                        # If batch_result is not a list, something went wrong
                        if not isinstance(batch_result, list):
                            logger.warning(f"Batch result is not a list: {type(batch_result)}")
                            # Try to get it from the task directly
                            for bn, bid, task in batch_info:
                                if task.done():
                                    try:
                                        result = task.result()
                                        if isinstance(result, list):
                                            batch_num = bn
                                            batch_ids = bid
                                            batch_result = result
                                            break
                                    except Exception as te:
                                        if batch_num is None:
                                            batch_num = bn
                                            batch_ids = bid
                                            task_error = str(te)
                                        continue
                        
                        # If we still have no batch info, log and continue
                        if batch_num is None or batch_result is None:
                            logger.warning("Could not match completed task to batch info")
                            # Try to process any available results
                            for bn, bid, task in batch_info:
                                if task.done() and not task.cancelled():
                                    try:
                                        result = task.result()
                                        if isinstance(result, list) and len(result) > 0:
                                            batch_num = bn
                                            batch_ids = bid
                                            batch_result = result
                                            break
                                    except Exception:
                                        continue
                            
                            if batch_result is None:
                                continue
                        
                        # Process the batch result
                        if len(batch_result) == 0:
                            continue
                        
                        batch_size_actual = len(batch_result)
                        sequences.extend(batch_result)
                        completed_batches += 1
                        
                        # Save each sequence immediately to disk (skips if already exists)
                        batch_saved = 0
                        batch_existed = 0
                        for seq in batch_result:
                            if not isinstance(seq, dict):
                                continue
                            acc = seq.get('accession')
                            if not acc:
                                continue
                            
                            # Check if already exists before trying to save
                            target_dir = self.viable_dir if (int(hashlib.md5(acc.encode()).hexdigest(), 16) % 10) < 9 else self.val_viable_dir
                            filepath = target_dir / f"{acc}.fasta"
                            
                            if filepath.exists():
                                # Already exists - just mark as completed
                                batch_existed += 1
                                self.completed_ids.add(acc)
                            else:
                                # Save new file
                                if self._save_sequence_immediate(seq):
                                    batch_saved += 1
                                    saved_count += 1
                                    self.completed_ids.add(acc)
                        
                        # Immediate feedback on first batch
                        if not first_batch_done:
                            pbar.write(f"\n[STARTED] First batch downloaded! {batch_size_actual} sequences received.")
                            if batch_saved > 0:
                                pbar.write(f"[INFO] {batch_saved} new sequences saved, {batch_existed} already existed.")
                            else:
                                pbar.write(f"[INFO] All {batch_existed} sequences already existed (skipped saving).")
                            pbar.write(f"[INFO] Downloads are now in progress. New sequences saved immediately to disk.")
                            sys.stdout.flush()
                            first_batch_done = True
                        
                        # Update progress bar immediately
                        pbar.update(batch_size_actual)
                        pbar.refresh()  # Force refresh
                        
                        # Detailed logging every 100 sequences OR every 5 seconds (whichever comes first)
                        current_time = time.time()
                        time_since_update = current_time - last_update_time
                        should_log = (len(sequences) - last_log_count >= 100) or (time_since_update >= 5)
                        
                        if should_log:
                            elapsed = time.time() - start_time
                            rate = len(sequences) / elapsed if elapsed > 0 else 0
                            remaining = (total_to_download - len(sequences)) / rate if rate > 0 else 0
                            
                            # Write to progress bar (appears above it)
                            pbar.write(f"\n[PROGRESS] Downloaded: {len(sequences):,}/{total_to_download:,} "
                                      f"({len(sequences)*100/total_to_download:.2f}%) | "
                                      f"Saved: {saved_count:,} | "
                                      f"Rate: {rate:.1f} seq/s | "
                                      f"ETA: {remaining/60:.1f} min | "
                                      f"Batches: {completed_batches}/{total_batches}")
                            sys.stdout.flush()
                            last_log_count = len(sequences)
                            last_update_time = current_time
                        
                        # Save checkpoint every 100 sequences or every 30 seconds (more frequent)
                        current_time = time.time()
                        if (saved_count % 100 == 0 and saved_count > 0) or (current_time - last_checkpoint_time > 30):
                            self._save_checkpoint()
                            last_checkpoint_time = current_time
                            if current_time - last_save_time > 60:  # Only log every minute
                                pbar.write(f"[CHECKPOINT] Saved progress: {saved_count:,} sequences saved to disk")
                                sys.stdout.flush()
                                last_save_time = current_time
                    
                    except Exception as e:
                        # Safely extract error message, handling coroutines
                        try:
                            error_msg = str(e)
                            error_type = type(e).__name__
                        except Exception:
                            error_msg = "Unknown error"
                            error_type = "Exception"
                        
                        # Check if error contains coroutine (indicates async issue)
                        if 'coroutine' in error_msg.lower() or (hasattr(e, '__class__') and 'coroutine' in str(type(e)).lower()):
                            error_msg = f"Async operation error: {error_type}"
                            logger.warning(f"Coroutine-related error detected: {error_type}")
                        
                        # Find the failed batch by checking all tasks
                        batch_num_failed = batch_num
                        batch_ids_failed = batch_ids
                        
                        if batch_num_failed is None:
                            # Find any task that has an exception
                            for bn, bid, task in batch_info:
                                if task.done():
                                    try:
                                        task.exception()  # This will raise if there's an exception
                                    except Exception:
                                        batch_num_failed = bn
                                        batch_ids_failed = bid
                                        break
                        
                        if batch_num_failed is not None:
                            failed_batches.append((batch_num_failed, batch_ids_failed or [], error_msg))
                            if batch_num_failed not in failed_batch_ids:
                                failed_batch_ids[batch_num_failed] = batch_ids_failed or []
                            pbar.write(f"[WARNING] Batch {batch_num_failed} failed: {error_msg}. Will retry later.")
                            sys.stdout.flush()
                            logger.warning(f"Batch {batch_num_failed} failed: {error_msg}")
            
            except Exception as outer_e:
                # Handle errors in the as_completed iteration itself
                try:
                    error_msg = str(outer_e)
                    error_type = type(outer_e).__name__
                except Exception:
                    error_msg = "Unknown error in batch processing"
                    error_type = "Exception"
                
                # Check for coroutine in error - this is the key issue
                if 'coroutine' in error_msg.lower() or (hasattr(outer_e, '__class__') and 'coroutine' in str(type(outer_e)).lower()):
                    error_msg = f"Async iteration error: {error_type}"
                    logger.error(f"Critical async error in batch processing: {error_msg}")
                    logger.error("Attempting to process batches using alternative method...")
                    
                    # Alternative: wait for all tasks and process results
                    logger.info("Waiting for all batches to complete...")
                    for bn, bid, task in batch_info:
                        try:
                            if not task.done():
                                await asyncio.wait_for(task, timeout=300)  # Wait up to 5 minutes per batch
                        except asyncio.TimeoutError:
                            logger.warning(f"Batch {bn} timed out")
                            failed_batches.append((bn, bid, "Timeout"))
                            failed_batch_ids[bn] = bid
                        except Exception as te:
                            error_str = str(te)
                            if 'coroutine' not in error_str.lower():
                                logger.warning(f"Batch {bn} error: {error_str}")
                            failed_batches.append((bn, bid, error_str))
                            failed_batch_ids[bn] = bid
                    
                    # Now process completed tasks
                    for bn, bid, task in batch_info:
                        if task.done() and not task.cancelled():
                            try:
                                result = task.result()
                                if isinstance(result, list) and len(result) > 0:
                                    sequences.extend(result)
                                    completed_batches += 1
                                    # Save sequences
                                    for seq in result:
                                        acc = seq.get('accession') if isinstance(seq, dict) else None
                                        if acc:
                                            if self._save_sequence_immediate(seq):
                                                saved_count += 1
                                                self.completed_ids.add(acc)
                                    pbar.update(len(result))
                            except Exception as te:
                                error_str = str(te)
                                if 'coroutine' not in error_str.lower():
                                    failed_batches.append((bn, bid, error_str))
                                    failed_batch_ids[bn] = bid
                else:
                    logger.error(f"Error in as_completed loop: {error_msg}", exc_info=True)
                    # Mark all remaining batches as failed for retry
                    for bn, bid, task in batch_info:
                        if not task.done():
                            failed_batches.append((bn, bid, error_msg))
                            failed_batch_ids[bn] = bid
        
        # Retry failed batches with exponential backoff (session still open)
        if failed_batches:
            logger.info("="*70)
            logger.info(f"RETRYING {len(failed_batches)} FAILED BATCHES")
            logger.info("="*70)
            pbar.write(f"\n[RETRY] Retrying {len(failed_batches)} failed batches...")
            sys.stdout.flush()
            
            for retry_round in range(3):  # Up to 3 retry rounds
                if not failed_batches:
                    break
                
                logger.info(f"Retry round {retry_round + 1}/3: {len(failed_batches)} batches remaining")
                pbar.write(f"[RETRY] Round {retry_round + 1}: Retrying {len(failed_batches)} batches...")
                sys.stdout.flush()
                
                # Wait before retry (except first round)
                if retry_round > 0:
                    wait_time = 60 * (retry_round + 1)  # 2 min, 3 min
                    logger.info(f"Waiting {wait_time} seconds before retry...")
                    await asyncio.sleep(wait_time)
                
                retry_tasks = {}
                retry_batch_info = []  # List of (batch_num, batch_ids, task) for retries
                remaining_failed = []
                
                for batch_num, batch_ids, error_msg in failed_batches:
                    # Filter out IDs that were already downloaded
                    remaining_batch_ids = [id for id in batch_ids if id not in self.completed_ids]
                    if not remaining_batch_ids:
                        continue  # All IDs in this batch were already downloaded
                    
                    # Create a task from the coroutine
                    coro = self._fetch_batch_async(session, remaining_batch_ids, f"R{batch_num}")
                    task = asyncio.create_task(coro)
                    retry_tasks[task] = (batch_num, remaining_batch_ids)
                    retry_batch_info.append((batch_num, remaining_batch_ids, task))
                
                # Process retry batches
                if retry_tasks:
                    try:
                        for completed_future in asyncio.as_completed(retry_tasks.keys()):
                            try:
                                batch_result = await completed_future
                                
                                # Find which batch this belongs to
                                batch_num = None
                                batch_ids = None
                                for bn, bid, task in retry_batch_info:
                                    if task.done():
                                        try:
                                            if task.result() == batch_result or task == completed_future:
                                                batch_num = bn
                                                batch_ids = bid
                                                break
                                        except Exception:
                                            continue
                                
                                # Fallback: find by checking retry_tasks
                                if batch_num is None:
                                    for task, (bn, bid) in list(retry_tasks.items()):
                                        if task.done():
                                            try:
                                                if task.result() == batch_result:
                                                    batch_num = bn
                                                    batch_ids = bid
                                                    break
                                            except Exception:
                                                continue
                                
                                if batch_num is None or not isinstance(batch_result, list):
                                    # Try to get from task directly
                                    for bn, bid, task in retry_batch_info:
                                        if task.done():
                                            try:
                                                result = task.result()
                                                if isinstance(result, list):
                                                    batch_num = bn
                                                    batch_ids = bid
                                                    batch_result = result
                                                    break
                                            except Exception:
                                                continue
                                
                                if batch_num is None or not isinstance(batch_result, list) or len(batch_result) == 0:
                                    continue
                                
                                batch_saved = 0
                                for seq in batch_result:
                                    if isinstance(seq, dict) and self._save_sequence_immediate(seq):
                                        batch_saved += 1
                                        saved_count += 1
                                        acc = seq.get('accession')
                                        if acc:
                                            self.completed_ids.add(acc)
                                sequences.extend(batch_result)
                                pbar.update(len(batch_result))
                                pbar.write(f"[SUCCESS] Batch {batch_num} retry successful: {batch_saved} sequences saved")
                                sys.stdout.flush()
                                self._save_checkpoint()
                            except Exception as e:
                                error_msg = str(e)
                                if 'coroutine' in error_msg.lower():
                                    error_msg = f"Async error: {type(e).__name__}"
                                # Find batch number
                                batch_num_failed = None
                                batch_ids_failed = None
                                for bn, bid, task in retry_batch_info:
                                    if task.done():
                                        try:
                                            task.exception()
                                        except Exception:
                                            batch_num_failed = bn
                                            batch_ids_failed = bid
                                            break
                                if batch_num_failed is None and retry_batch_info:
                                    batch_num_failed, batch_ids_failed, _ = retry_batch_info[0]
                                
                                if batch_num_failed is not None:
                                    remaining_failed.append((batch_num_failed, batch_ids_failed or [], error_msg))
                                    pbar.write(f"[WARNING] Batch {batch_num_failed} retry failed: {error_msg}")
                                    sys.stdout.flush()
                    except Exception as retry_outer_e:
                        error_msg = str(retry_outer_e)
                        if 'coroutine' in error_msg.lower():
                            logger.error("Async error in retry processing - waiting for all retry tasks...")
                            # Wait for all retry tasks
                            for bn, bid, task in retry_batch_info:
                                try:
                                    if not task.done():
                                        await asyncio.wait_for(task, timeout=300)
                                except Exception:
                                    pass
                                if task.done():
                                    try:
                                        result = task.result()
                                        if isinstance(result, list):
                                            sequences.extend(result)
                                            for seq in result:
                                                if isinstance(seq, dict) and self._save_sequence_immediate(seq):
                                                    saved_count += 1
                                                    acc = seq.get('accession')
                                                    if acc:
                                                        self.completed_ids.add(acc)
                                            pbar.update(len(result))
                                    except Exception as te:
                                        error_str = str(te)
                                        if 'coroutine' not in error_str.lower():
                                            remaining_failed.append((bn, bid, error_str))
                        else:
                            logger.error(f"Error in retry loop: {error_msg}")
                            # Mark remaining as failed
                            for bn, bid, task in retry_batch_info:
                                if not task.done():
                                    remaining_failed.append((bn, bid, error_msg))
                
                failed_batches = remaining_failed
        
        pbar.close()
        self._save_checkpoint()
        
        elapsed_total = time.time() - start_time
        rate_avg = saved_count / elapsed_total if elapsed_total > 0 else 0
        
        logger.info("="*70)
        logger.info(f"DOWNLOAD COMPLETE!")
        logger.info(f"Successfully downloaded: {len(sequences):,} sequences")
        logger.info(f"Saved to disk: {saved_count:,} sequences")
        logger.info(f"Failed batches (after retries): {len(failed_batches)}")
        if failed_batches:
            logger.warning(f"Some batches failed after retries. Run again to retry failed IDs.")
        logger.info(f"Total time: {elapsed_total/60:.2f} minutes")
        logger.info(f"Average rate: {rate_avg:.2f} sequences/second")
        logger.info("="*70)
        
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
        logger.info("="*70)
        logger.info(f"SAVING VIABLE GENOMES TO DISK")
        logger.info("="*70)
        logger.info(f"Total sequences to save: {len(sequences):,}")
        
        # Split into train/val (90/10)
        random.shuffle(sequences)
        split_idx = int(len(sequences) * 0.9)
        train_sequences = sequences[:split_idx]
        val_sequences = sequences[split_idx:]
        
        logger.info(f"Split: {len(train_sequences):,} training (90%) | {len(val_sequences):,} validation (10%)")
        logger.info(f"Saving training sequences to: {self.viable_dir}")
        
        # Save training data with progress bar
        pbar_train = tqdm(total=len(train_sequences), desc="Saving training genomes", unit="seq")
        for i, seq in enumerate(train_sequences):
            filename = f"{seq['accession']}.fasta"
            filepath = self.viable_dir / filename
            with open(filepath, 'w') as f:
                f.write(f">{seq['id']} {seq['description']}\n")
                f.write(seq['sequence'] + '\n')
            pbar_train.update(1)
            if (i + 1) % 1000 == 0:
                logger.info(f"  Saved {i + 1:,}/{len(train_sequences):,} training sequences")
        pbar_train.close()
        
        # Save validation data with progress bar
        logger.info(f"Saving validation sequences to: {self.val_viable_dir}")
        pbar_val = tqdm(total=len(val_sequences), desc="Saving validation genomes", unit="seq")
        for i, seq in enumerate(val_sequences):
            filename = f"{seq['accession']}.fasta"
            filepath = self.val_viable_dir / filename
            with open(filepath, 'w') as f:
                f.write(f">{seq['id']} {seq['description']}\n")
                f.write(seq['sequence'] + '\n')
            pbar_val.update(1)
            if (i + 1) % 500 == 0:
                logger.info(f"  Saved {i + 1:,}/{len(val_sequences):,} validation sequences")
        pbar_val.close()
        
        logger.info("="*70)
        logger.info(f"[SUCCESS] Saved {len(train_sequences):,} training and {len(val_sequences):,} validation viable genomes")
        logger.info("="*70)
    
    def generate_synthetic_viable_genomes(self, count=10000):
        """Generate synthetic viable-like genomes (realistic viral sequences)"""
        logger.info("="*70)
        logger.info(f"GENERATING SYNTHETIC VIABLE GENOMES")
        logger.info("="*70)
        logger.info(f"Total to generate: {count:,}")
        logger.info("These are realistic viral genome sequences with:")
        logger.info("  - Proper start/stop codons")
        logger.info("  - Valid codon usage")
        logger.info("  - Realistic gene structures")
        logger.info("  - Appropriate GC content")
        logger.info("="*70)
        
        sequences = []
        start_time = time.time()
        
        pbar = tqdm(total=count, desc="Generating synthetic viable", unit="seq",
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        
        for i in range(count):
            # Generate realistic viral genome
            # Typical viral genome length: 3K-300K bp
            length = random.randint(3000, 50000)
            
            # Generate sequence with realistic patterns
            sequence = self._generate_realistic_viral_sequence(length, i)
            
            # Determine train/val split (90/10)
            hash_val = int(hashlib.md5(f"SYNTH_VIABLE_{i}".encode()).hexdigest(), 16)
            is_training = (hash_val % 10) < 9
            
            target_dir = self.viable_dir if is_training else self.val_viable_dir
            filename = f"SYNTH_VIABLE_{i:08d}.fasta"
            filepath = target_dir / filename
            
            # Save immediately
            with open(filepath, 'w') as f:
                f.write(f">SYNTH_VIABLE_{i:08d} synthetic viable viral genome {i}\n")
                f.write(sequence + '\n')
            
            sequences.append({
                'id': f"SYNTH_VIABLE_{i:08d}",
                'accession': f"SYNTH_VIABLE_{i:08d}",
                'description': f"synthetic viable viral genome {i}",
                'sequence': sequence,
                'length': len(sequence),
                'source': 'synthetic_viable',
                'is_viable': True
            })
            
            pbar.update(1)
            if (i + 1) % 1000 == 0:
                logger.info(f"  Generated {i + 1:,}/{count:,} synthetic viable genomes")
        
        pbar.close()
        
        # Split into train/val
        split_idx = int(len(sequences) * 0.9)
        train_sequences = sequences[:split_idx]
        val_sequences = sequences[split_idx:]
        
        elapsed_total = time.time() - start_time
        logger.info("="*70)
        logger.info(f"[SUCCESS] Generated {len(train_sequences):,} training and {len(val_sequences):,} validation synthetic viable genomes")
        logger.info(f"Total time: {elapsed_total:.2f} seconds ({elapsed_total/60:.2f} minutes)")
        logger.info("="*70)
        
        return sequences
    
    def _generate_realistic_viral_sequence(self, length: int, seed: int) -> str:
        """Generate a realistic viral genome sequence"""
        random.seed(seed)  # For reproducibility
        
        # Realistic viral genome characteristics
        gc_content = random.uniform(0.35, 0.65)  # Typical viral GC content
        at_content = 1.0 - gc_content
        
        # Generate sequence with realistic codon usage
        sequence = []
        bases = ['A', 'T', 'C', 'G']
        base_probs = [
            at_content / 2,  # A
            at_content / 2,  # T
            gc_content / 2,  # C
            gc_content / 2   # G
        ]
        
        for _ in range(length):
            sequence.append(random.choices(bases, weights=base_probs)[0])
        
        seq = ''.join(sequence)
        
        # Add realistic features:
        # 1. Start codons (ATG) at reasonable intervals
        start_codon_positions = list(range(0, length, random.randint(500, 2000)))
        for pos in start_codon_positions[:min(10, len(start_codon_positions))]:
            if pos + 2 < length:
                seq = seq[:pos] + 'ATG' + seq[pos+3:]
        
        # 2. Stop codons (TAA, TAG, TGA) at reasonable intervals
        stop_codons = ['TAA', 'TAG', 'TGA']
        stop_positions = list(range(300, length-300, random.randint(800, 2500)))
        for pos in stop_positions[:min(10, len(stop_positions))]:
            if pos + 2 < length:
                seq = seq[:pos] + random.choice(stop_codons) + seq[pos+3:]
        
        return seq
    
    def generate_non_viable_genomes(self, count=10000, methods=['random', 'fragmented', 'no_start', 'no_stop', 'invalid_codons']):
        """Generate non-viable genomes for training"""
        logger.info("="*70)
        logger.info(f"GENERATING SYNTHETIC NON-VIABLE GENOMES")
        logger.info("="*70)
        logger.info(f"Total to generate: {count:,}")
        logger.info(f"Generation methods: {', '.join(methods)}")
        logger.info("="*70)
        logger.info("NOTE: Non-viable viruses are NOT downloaded from NCBI.")
        logger.info("      They are generated synthetically using various methods")
        logger.info("      to create sequences that cannot form viable viruses.")
        logger.info("="*70)
        
        non_viable_sequences = []
        method_counts = {method: 0 for method in methods}
        start_time = time.time()
        
        # Generate sequences with progress bar
        pbar = tqdm(total=count, desc="Generating non-viable genomes", unit="seq",
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        
        for i in range(count):
            method = random.choice(methods)
            method_counts[method] += 1
            seq = self._generate_non_viable_sequence(method, i)
            non_viable_sequences.append(seq)
            pbar.update(1)
            
            if (i + 1) % 5000 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                pbar.write(f"[PROGRESS] Generated {i + 1:,}/{count:,} ({i*100/count:.1f}%) | "
                          f"Rate: {rate:.1f} seq/s")
        
        pbar.close()
        
        # Log method distribution
        logger.info("="*70)
        logger.info("Generation method distribution:")
        for method, count_method in method_counts.items():
            percentage = (count_method / count) * 100
            logger.info(f"  {method}: {count_method:,} ({percentage:.1f}%)")
        logger.info("="*70)
        
        # Split into train/val
        logger.info("Splitting into training (90%) and validation (10%) sets...")
        random.shuffle(non_viable_sequences)
        split_idx = int(len(non_viable_sequences) * 0.9)
        train_sequences = non_viable_sequences[:split_idx]
        val_sequences = non_viable_sequences[split_idx:]
        
        logger.info(f"Saving {len(train_sequences):,} training sequences to: {self.non_viable_dir}")
        # Save training data with progress bar
        pbar_save = tqdm(total=len(train_sequences), desc="Saving training non-viable", unit="seq")
        for i, seq in enumerate(train_sequences):
            filename = f"non_viable_{seq['id']}.fasta"
            filepath = self.non_viable_dir / filename
            with open(filepath, 'w') as f:
                f.write(f">{seq['id']} {seq['description']}\n")
                f.write(seq['sequence'] + '\n')
            pbar_save.update(1)
            if (i + 1) % 5000 == 0:
                logger.info(f"  Saved {i + 1:,}/{len(train_sequences):,} training sequences")
        pbar_save.close()
        
        logger.info(f"Saving {len(val_sequences):,} validation sequences to: {self.val_non_viable_dir}")
        # Save validation data with progress bar
        pbar_save_val = tqdm(total=len(val_sequences), desc="Saving validation non-viable", unit="seq")
        for i, seq in enumerate(val_sequences):
            filename = f"non_viable_{seq['id']}.fasta"
            filepath = self.val_non_viable_dir / filename
            with open(filepath, 'w') as f:
                f.write(f">{seq['id']} {seq['description']}\n")
                f.write(seq['sequence'] + '\n')
            pbar_save_val.update(1)
            if (i + 1) % 2000 == 0:
                logger.info(f"  Saved {i + 1:,}/{len(val_sequences):,} validation sequences")
        pbar_save_val.close()
        
        elapsed_total = time.time() - start_time
        logger.info("="*70)
        logger.info(f"[SUCCESS] Generated {len(train_sequences):,} training and {len(val_sequences):,} validation non-viable genomes")
        logger.info(f"Total time: {elapsed_total:.2f} seconds ({elapsed_total/60:.2f} minutes)")
        logger.info("="*70)
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
        logger.info("="*70)
        logger.info(f"GENERATING MUTATED NON-VIABLE GENOMES")
        logger.info("="*70)
        logger.info(f"Total to generate: {count:,}")
        logger.info(f"Source: {len(viable_sequences):,} viable genomes")
        logger.info("="*70)
        logger.info("NOTE: These are generated by mutating viable genomes to make them non-viable.")
        logger.info("      Mutations include: deleting start codons, inserting premature stops,")
        logger.info("      frame shifts, and random corruption.")
        logger.info("="*70)
        
        if not viable_sequences:
            logger.warning("No viable sequences available for mutation!")
            return []
        
        non_viable = []
        mutation_types = ['delete_start', 'insert_stop', 'frame_shift', 'corrupt']
        mutation_counts = {mut: 0 for mut in mutation_types}
        start_time = time.time()
        
        # Generate with progress bar
        pbar = tqdm(total=count, desc="Mutating viable genomes", unit="seq",
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        
        for i in range(count):
            # Pick a random viable sequence
            source = random.choice(viable_sequences)
            seq = source['sequence']
            
            # Apply mutations to make it non-viable
            mutation_type = random.choice(mutation_types)
            mutation_counts[mutation_type] += 1
            
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
            
            pbar.update(1)
            if (i + 1) % 5000 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                pbar.write(f"[PROGRESS] Generated {i + 1:,}/{count:,} ({i*100/count:.1f}%) | "
                          f"Rate: {rate:.1f} seq/s")
        
        pbar.close()
        
        # Log mutation distribution
        logger.info("="*70)
        logger.info("Mutation type distribution:")
        for mut_type, count_mut in mutation_counts.items():
            percentage = (count_mut / count) * 100
            logger.info(f"  {mut_type}: {count_mut:,} ({percentage:.1f}%)")
        logger.info("="*70)
        
        # Save
        logger.info("Splitting into training (90%) and validation (10%) sets...")
        random.shuffle(non_viable)
        split_idx = int(len(non_viable) * 0.9)
        train_sequences = non_viable[:split_idx]
        val_sequences = non_viable[split_idx:]
        
        logger.info(f"Saving {len(train_sequences):,} training sequences...")
        pbar_save = tqdm(total=len(train_sequences), desc="Saving mutated training", unit="seq")
        for i, seq in enumerate(train_sequences):
            filename = f"non_viable_{seq['accession']}.fasta"
            filepath = self.non_viable_dir / filename
            with open(filepath, 'w') as f:
                f.write(f">{seq['id']} {seq['description']}\n")
                f.write(seq['sequence'] + '\n')
            pbar_save.update(1)
            if (i + 1) % 2000 == 0:
                logger.info(f"  Saved {i + 1:,}/{len(train_sequences):,} training sequences")
        pbar_save.close()
        
        logger.info(f"Saving {len(val_sequences):,} validation sequences...")
        pbar_save_val = tqdm(total=len(val_sequences), desc="Saving mutated validation", unit="seq")
        for i, seq in enumerate(val_sequences):
            filename = f"non_viable_{seq['accession']}.fasta"
            filepath = self.val_non_viable_dir / filename
            with open(filepath, 'w') as f:
                f.write(f">{seq['id']} {seq['description']}\n")
                f.write(seq['sequence'] + '\n')
            pbar_save_val.update(1)
            if (i + 1) % 1000 == 0:
                logger.info(f"  Saved {i + 1:,}/{len(val_sequences):,} validation sequences")
        pbar_save_val.close()
        
        elapsed_total = time.time() - start_time
        logger.info("="*70)
        logger.info(f"[SUCCESS] Generated {len(train_sequences):,} training and {len(val_sequences):,} validation mutated non-viable genomes")
        logger.info(f"Total time: {elapsed_total:.2f} seconds ({elapsed_total/60:.2f} minutes)")
        logger.info("="*70)
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

