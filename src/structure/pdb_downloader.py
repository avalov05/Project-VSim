"""
SARS-CoV-2 Protein Structure Downloader
Downloads real protein structures from PDB for accurate virus modeling
"""

import logging
import requests
from pathlib import Path
from typing import Optional, Dict
import json

class SARSCOV2StructureDownloader:
    """Download real SARS-CoV-2 protein structures from PDB"""
    
    # Known SARS-CoV-2 protein structures from PDB
    SARS_COV2_PDB_STRUCTURES = {
        'spike': {
            'pdb_id': '6VYB',
            'chains': ['A', 'B', 'C'],
            'description': 'Spike protein trimer',
            'url': 'https://files.rcsb.org/download/6VYB.pdb'
        },
        'nucleocapsid': {
            'pdb_id': '6VYO',
            'chains': ['A'],
            'description': 'Nucleocapsid protein',
            'url': 'https://files.rcsb.org/download/6VYO.pdb'
        },
        'membrane': {
            'pdb_id': '7MY6',
            'chains': ['A'],
            'description': 'Membrane protein',
            'url': 'https://files.rcsb.org/download/7MY6.pdb'
        },
        'envelope': {
            'pdb_id': '7K3G',
            'chains': ['A'],
            'description': 'Envelope protein',
            'url': 'https://files.rcsb.org/download/7K3G.pdb'
        },
        'nsp12': {
            'pdb_id': '6NUR',
            'chains': ['A'],
            'description': 'RNA-dependent RNA polymerase',
            'url': 'https://files.rcsb.org/download/6NUR.pdb'
        }
    }
    
    def __init__(self):
        self.logger = logging.getLogger('VSim.SARSCOV2Downloader')
        self.cache_dir = Path('data/structures/pdb')
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def download_structure(self, pdb_id: str) -> Optional[Path]:
        """Download PDB structure from RCSB"""
        cache_file = self.cache_dir / f'{pdb_id}.pdb'
        
        if cache_file.exists():
            self.logger.info(f"Using cached structure: {pdb_id}")
            return cache_file
        
        try:
            self.logger.info(f"Downloading structure from PDB: {pdb_id}")
            url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            with open(cache_file, 'w') as f:
                f.write(response.text)
            
            self.logger.info(f"Downloaded {pdb_id}: {len(response.text)} bytes")
            return cache_file
        except Exception as e:
            self.logger.warning(f"Failed to download {pdb_id}: {e}")
            return None
    
    def get_spike_protein(self) -> Optional[Path]:
        """Get spike protein structure"""
        return self.download_structure(self.SARS_COV2_PDB_STRUCTURES['spike']['pdb_id'])
    
    def get_nucleocapsid(self) -> Optional[Path]:
        """Get nucleocapsid protein structure"""
        return self.download_structure(self.SARS_COV2_PDB_STRUCTURES['nucleocapsid']['pdb_id'])
    
    def get_membrane_protein(self) -> Optional[Path]:
        """Get membrane protein structure"""
        return self.download_structure(self.SARS_COV2_PDB_STRUCTURES['membrane']['pdb_id'])
    
    def get_all_structures(self) -> Dict[str, Optional[Path]]:
        """Download all known SARS-CoV-2 structures"""
        structures = {}
        for name, info in self.SARS_COV2_PDB_STRUCTURES.items():
            structures[name] = self.download_structure(info['pdb_id'])
        return structures

