"""
Enhanced Virus 3D Model Generator
Creates detailed 3D visualization with individual protein structures
"""

import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import numpy as np
import requests
import json
from Bio import PDB
from Bio.PDB import PDBIO, PDBParser

# Import real structure tools
try:
    from src.structure.virus_assembler import RealisticVirusAssembler
    from src.structure.generic_assembler import GenericVirusAssembler
    REAL_STRUCTURES_AVAILABLE = True
except ImportError as e:
    REAL_STRUCTURES_AVAILABLE = False
    try:
        _logger = logging.getLogger('VSim.VirusModelGenerator')
        _logger.warning(f"Real structure tools not available: {e}")
    except:
        pass

class VirusModelGenerator:
    """Generate detailed 3D model of complete virus particle with protein structures"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger('VSim.VirusModelGenerator')
    
    def generate_virus_model(self, genome_results: Dict, structure_results: Dict) -> Dict:
        """Generate detailed 3D model with individual protein structures"""
        self.logger.info("Generating detailed 3D virus particle model with protein structures...")
        
        genome_length = genome_results.get('length', 0)
        protein_count = len(genome_results.get('proteins', []))
        proteins = genome_results.get('proteins', [])
        
        # Estimate capsid size (geometry will emerge from physics)
        estimated_diameter = self._estimate_virus_diameter(genome_length, protein_count)
        # Note: capsid_shape is now ignored - geometry emerges from physics
        capsid_shape = "emergent"  # Placeholder - actual shape detected after assembly
        
        # Determine virus type based on genome characteristics
        virus_type = self._classify_virus_type(genome_results)
        
        # Get protein structures (predicted or real)
        protein_structures = self._get_protein_structures(proteins, structure_results)
        
        # Create realistic virus assembly through pure physics simulation
        self.logger.info(f"Assembling {virus_type} virus with {len(protein_structures)} unique proteins...")
        self.logger.info("Using physics-based assembly - geometry will emerge naturally")
        
        # Generate realistic virus structure
        virus_model = self._create_realistic_virus_assembly(
            estimated_diameter, capsid_shape, protein_structures, proteins, virus_type
        )
        
        # Save comprehensive model
        model_path = self._save_realistic_virus_model(virus_model, genome_results, virus_type)
        
        return {
            'model_file': str(model_path) if model_path else None,
            'estimated_diameter_nm': estimated_diameter,
            'capsid_shape': capsid_shape,
            'protein_count': protein_count,
            'genome_length': genome_length,
            'virus_type': virus_type,
            'protein_structures': protein_structures,
            'visualization_data': self._create_detailed_visualization_data(
                virus_model, estimated_diameter, capsid_shape, protein_structures
            )
        }
    
    def _classify_virus_type(self, genome_results: Dict) -> str:
        """Classify virus type based on genome characteristics"""
        genome_length = genome_results.get('length', 0)
        gc_content = genome_results.get('gc_content', 0.5)
        proteins = genome_results.get('proteins', [])
        
        # Analyze protein characteristics
        protein_lengths = [len(p.get('sequence', '')) for p in proteins]
        avg_protein_length = np.mean(protein_lengths) if protein_lengths else 0
        max_protein_length = max(protein_lengths) if protein_lengths else 0
        
        # Classify based on genome size and characteristics (generic classification)
        if genome_length > 25000:
            # Large RNA virus
            if max_protein_length > 1200:
                return 'large_rna_virus_with_polyprotein'
            else:
                return 'large_rna_virus'
        elif genome_length > 10000:
            if gc_content > 0.6:
                return 'high_gc_rna_virus'
            else:
                return 'medium_rna_virus'
        elif genome_length > 5000:
            return 'small_rna_virus'
        else:
            return 'very_small_virus'
    
    def _get_protein_structures(self, proteins: List[Dict], structure_results: Dict) -> List[Dict]:
        """Get real protein structures from PDB or generate predictions"""
        structures = []
        
        # Check if we have predicted structures
        predicted_structures = structure_results.get('structures', [])
        
        for i, protein in enumerate(proteins):
            protein_seq = protein.get('sequence', '')
            protein_id = protein.get('orf_info', {}).get('start', i)
            
            structure_info = {
                'protein_id': protein_id,
                'sequence': protein_seq,
                'length': len(protein_seq),
                'pdb_id': None,
                'pdb_file': None,
                'structure_type': None
            }
            
            # Try to get from predicted structures
            if i < len(predicted_structures):
                pred_struct = predicted_structures[i]
                if pred_struct.get('pdb_file') and Path(pred_struct['pdb_file']).exists():
                    structure_info['pdb_file'] = pred_struct['pdb_file']
                    structure_info['structure_type'] = 'predicted'
                    structure_info['confidence'] = pred_struct.get('confidence', 0.8)
                    structures.append(structure_info)
                    continue
            
            # Try to find structures in PDB database based on sequence similarity
            # (This would require BLAST or similar - for now, skip automatic PDB lookup)
            # Users can manually provide PDB IDs if known structures exist
            
            # Generate simplified structure if no real structure available
            if not structure_info.get('pdb_file'):
                structure_info['pdb_file'] = self._generate_protein_structure_model(
                    protein_seq, protein_id
                )
                structure_info['structure_type'] = 'modeled'
            
            structures.append(structure_info)
        
        return structures
    
    def _try_download_pdb_structure(self, sequence: str) -> Optional[str]:
        """Try to find PDB structure for protein sequence (generic search)"""
        # This would require BLAST or sequence similarity search against PDB
        # For now, return None - structures will be predicted instead
        # Future: Implement BLAST search against PDB database
        return None
    
    def _generate_protein_structure_model(self, sequence: str, protein_id: int) -> Optional[str]:
        """Generate detailed protein structure model"""
        try:
            output_dir = Path('results/structures/proteins')
            output_dir.mkdir(parents=True, exist_ok=True)
            
            import hashlib
            seq_hash = hashlib.md5(sequence.encode()).hexdigest()[:8]
            pdb_file = output_dir / f'protein_{protein_id}_{seq_hash}.pdb'
            
            # Generate detailed structure based on sequence
            structure_atoms = self._generate_detailed_protein_structure(sequence)
            
            with open(pdb_file, 'w') as f:
                f.write(f"REMARK   VSim Protein Structure Model\n")
                f.write(f"REMARK   Protein ID: {protein_id}\n")
                f.write(f"REMARK   Sequence Length: {len(sequence)}\n")
                
                for atom in structure_atoms:
                    f.write(
                        f"ATOM  {atom['serial']:5d}  {atom['atom']:2s}  "
                        f"{atom['residue']:3s} {atom['chain']:1s}{atom['resid']:4d}    "
                        f"{atom['x']:8.3f}{atom['y']:8.3f}{atom['z']:8.3f}"
                        f"{atom['occupancy']:6.2f}{atom['temp_factor']:6.2f}\n"
                    )
                
                f.write("END\n")
            
            return str(pdb_file)
        except Exception as e:
            self.logger.warning(f"Failed to generate protein structure: {e}")
            return None
    
    def _generate_detailed_protein_structure(self, sequence: str) -> List[Dict]:
        """Generate detailed protein structure with secondary structure elements"""
        atoms = []
        
        # Predict secondary structure
        helix_regions, sheet_regions = self._predict_secondary_structure_regions(sequence)
        
        # Generate 3D coordinates based on secondary structure
        residue_positions = self._generate_residue_positions(
            sequence, helix_regions, sheet_regions
        )
        
        # Convert to atom coordinates with full backbone
        atom_serial = 1
        for i, (residue, pos) in enumerate(zip(sequence, residue_positions)):
            residue_num = i + 1
            residue_code = self._get_residue_code(residue)
            
            # Add full backbone atoms
            # CA (alpha carbon)
            atoms.append({
                'serial': atom_serial,
                'atom': 'CA',
                'residue': residue_code,
                'chain': 'A',
                'resid': residue_num,
                'x': pos[0], 'y': pos[1], 'z': pos[2],
                'occupancy': 1.0,
                'temp_factor': 30.0
            })
            atom_serial += 1
            
            # N (nitrogen)
            if i > 0:
                prev_pos = residue_positions[i-1]
                n_dir = (pos - prev_pos) / np.linalg.norm(pos - prev_pos) if np.linalg.norm(pos - prev_pos) > 0 else np.array([1, 0, 0])
                n_pos = pos - n_dir * 1.46  # N-CA bond length
                atoms.append({
                    'serial': atom_serial,
                    'atom': 'N',
                    'residue': residue_code,
                    'chain': 'A',
                    'resid': residue_num,
                    'x': n_pos[0], 'y': n_pos[1], 'z': n_pos[2],
                    'occupancy': 1.0,
                    'temp_factor': 30.0
                })
                atom_serial += 1
            
            # C (carbonyl carbon)
            if i < len(sequence) - 1:
                next_pos = residue_positions[i+1]
                c_dir = (next_pos - pos) / np.linalg.norm(next_pos - pos) if np.linalg.norm(next_pos - pos) > 0 else np.array([1, 0, 0])
                c_pos = pos + c_dir * 1.52  # CA-C bond length
                atoms.append({
                    'serial': atom_serial,
                    'atom': 'C',
                    'residue': residue_code,
                    'chain': 'A',
                    'resid': residue_num,
                    'x': c_pos[0], 'y': c_pos[1], 'z': c_pos[2],
                    'occupancy': 1.0,
                    'temp_factor': 30.0
                })
                atom_serial += 1
            
            # O (carbonyl oxygen)
            if i < len(sequence) - 1:
                next_pos = residue_positions[i+1]
                c_dir = (next_pos - pos) / np.linalg.norm(next_pos - pos) if np.linalg.norm(next_pos - pos) > 0 else np.array([1, 0, 0])
                c_pos = pos + c_dir * 1.52
                # Oxygen perpendicular to CA-C bond
                perp = np.cross(c_dir, np.array([0, 0, 1]))
                if np.linalg.norm(perp) < 0.1:
                    perp = np.cross(c_dir, np.array([0, 1, 0]))
                perp = perp / np.linalg.norm(perp) if np.linalg.norm(perp) > 0 else np.array([0, 1, 0])
                o_pos = c_pos + perp * 2.2  # C=O bond length
                atoms.append({
                    'serial': atom_serial,
                    'atom': 'O',
                    'residue': residue_code,
                    'chain': 'A',
                    'resid': residue_num,
                    'x': o_pos[0], 'y': o_pos[1], 'z': o_pos[2],
                    'occupancy': 1.0,
                    'temp_factor': 30.0
                })
                atom_serial += 1
            
            # Side chain representation (simplified)
            side_chain_atoms = self._generate_side_chain_atoms(residue, pos, atom_serial, residue_num, residue_code)
            atoms.extend(side_chain_atoms)
            atom_serial += len(side_chain_atoms)
        
        return atoms
    
    def _generate_side_chain_atoms(self, residue: str, ca_pos: np.ndarray, start_serial: int, 
                                   residue_num: int, residue_code: str) -> List[Dict]:
        """Generate simplified side chain atoms"""
        atoms = []
        
        # Side chain direction (perpendicular to backbone)
        side_chain_offset = np.array([0.5, 1.0, 0.5])  # Simplified
        side_chain_offset = side_chain_offset / np.linalg.norm(side_chain_offset) * 1.5
        
        # Add CB (beta carbon) for most residues
        if residue.upper() != 'G':  # Glycine has no CB
            cb_pos = ca_pos + side_chain_offset
            atoms.append({
                'serial': start_serial,
                'atom': 'CB',
                'residue': residue_code,
                'chain': 'A',
                'resid': residue_num,
                'x': cb_pos[0], 'y': cb_pos[1], 'z': cb_pos[2],
                'occupancy': 1.0,
                'temp_factor': 35.0
            })
        
        # Add additional atoms for larger side chains
        if residue.upper() in ['R', 'K', 'E', 'D']:  # Charged residues
            sc_pos = ca_pos + side_chain_offset * 2.5
            atoms.append({
                'serial': start_serial + 1 if residue.upper() != 'G' else start_serial,
                'atom': 'CG' if residue.upper() in ['E', 'D'] else 'CZ',
                'residue': residue_code,
                'chain': 'A',
                'resid': residue_num,
                'x': sc_pos[0], 'y': sc_pos[1], 'z': sc_pos[2],
                'occupancy': 1.0,
                'temp_factor': 35.0
            })
        
        return atoms
    
    def _predict_secondary_structure_regions(self, sequence: str) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """Predict helix and sheet regions"""
        helix_formers = set('AEHKQR')
        sheet_formers = set('FILVWY')
        
        helix_regions = []
        sheet_regions = []
        
        # Simple sliding window approach
        window_size = 5
        i = 0
        while i < len(sequence) - window_size:
            window = sequence[i:i+window_size]
            helix_score = sum(1 for aa in window if aa in helix_formers)
            sheet_score = sum(1 for aa in window if aa in sheet_formers)
            
            if helix_score >= 3:
                helix_regions.append((i, i + window_size))
                i += window_size
            elif sheet_score >= 3:
                sheet_regions.append((i, i + window_size))
                i += window_size
            else:
                i += 1
        
        return helix_regions, sheet_regions
    
    def _generate_residue_positions(self, sequence: str, helix_regions: List[Tuple[int, int]], 
                                   sheet_regions: List[Tuple[int, int]]) -> List[np.ndarray]:
        """Generate 3D positions for residues based on secondary structure"""
        positions = []
        
        # Initialize starting position
        x, y, z = 0.0, 0.0, 0.0
        direction = np.array([1.0, 0.0, 0.0])
        helix_offset = 0
        sheet_offset = 0
        
        for i, residue in enumerate(sequence):
            # Check if in helix region
            in_helix = any(start <= i < end for start, end in helix_regions)
            in_sheet = any(start <= i < end for start, end in sheet_regions)
            
            if in_helix:
                # Alpha helix: 3.6 residues per turn, 1.5A rise per residue
                # Find which helix we're in
                helix_idx = next((idx for idx, (s, e) in enumerate(helix_regions) if s <= i < e), 0)
                local_pos = i - helix_regions[helix_idx][0]
                
                angle = (local_pos * 2 * np.pi) / 3.6
                radius = 2.3  # Helix radius in Angstroms
                x = helix_offset * 20 + radius * np.cos(angle)
                y = radius * np.sin(angle)
                z = local_pos * 1.5
                
                # Update helix offset for next helix
                if i == helix_regions[helix_idx][1] - 1:
                    helix_offset += 1
                    
            elif in_sheet:
                # Beta sheet: extended conformation, ~3.5A per residue
                sheet_idx = next((idx for idx, (s, e) in enumerate(sheet_regions) if s <= i < e), 0)
                local_pos = i - sheet_regions[sheet_idx][0]
                
                # Alternate sheet position (pleated sheet)
                z_offset = (local_pos % 2) * 1.0  # Alternate up/down
                x = sheet_offset * 25 + local_pos * 3.5
                y = z_offset
                z = 0.0
                
                if i == sheet_regions[sheet_idx][1] - 1:
                    sheet_offset += 1
            else:
                # Random coil: follow previous direction with variation
                if i > 0:
                    prev_pos = positions[i-1]
                    direction = np.array([x, y, z]) - prev_pos
                    if np.linalg.norm(direction) > 0:
                        direction = direction / np.linalg.norm(direction)
                    else:
                        direction = np.array([1.0, 0.0, 0.0])
                
                # Add natural variation
                variation = np.random.normal(0, 0.2, 3)
                direction = direction + variation
                direction = direction / np.linalg.norm(direction) if np.linalg.norm(direction) > 0 else np.array([1.0, 0.0, 0.0])
                
                x += direction[0] * 3.8
                y += direction[1] * 3.8
                z += direction[2] * 3.8
            
            positions.append(np.array([x, y, z]))
        
        return positions
    
    def _get_residue_code(self, residue: str) -> str:
        """Get 3-letter residue code"""
        residue_codes = {
            'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS',
            'Q': 'GLN', 'E': 'GLU', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
            'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO',
            'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL'
        }
        return residue_codes.get(residue.upper(), 'UNK')
    
    def _create_realistic_virus_assembly(self, diameter: float, shape: str,
                                         protein_structures: List[Dict],
                                         proteins: List[Dict], virus_type: str) -> Dict:
        """Create realistic virus assembly through physics simulation - geometry emerges naturally"""
        # Use generic assembler (now uses physics-based assembly)
        try:
            assembler = GenericVirusAssembler()
            model_path = assembler.assemble_virus(
                protein_structures=protein_structures,
                virus_type=virus_type,
                diameter_nm=diameter,
                shape=shape  # Ignored - geometry emerges from physics
            )
            
            # Read detected geometry from PDB REMARK lines
            detected_shape = "emergent"
            detected_symmetry = "none"
            
            if model_path and Path(model_path).exists():
                with open(model_path, 'r') as f:
                    for line in f:
                        if line.startswith("REMARK") and "Emergent Shape" in line:
                            detected_shape = line.split(":")[-1].strip()
                        elif line.startswith("REMARK") and "Emergent Symmetry" in line:
                            detected_symmetry = line.split(":")[-1].strip()
            
            return {
                'model_file': str(model_path),
                'diameter': diameter,
                'shape': detected_shape,  # Use detected shape, not predicted
                'symmetry': detected_symmetry,
                'virus_type': virus_type,
                'proteins': protein_structures,
                'assembly_method': 'physics_based'
            }
        except Exception as e:
            self.logger.warning(f"Physics-based assembly failed: {e}, using fallback")
            return self._create_detailed_virus_structure(diameter, shape, protein_structures, proteins)
    
    def _save_realistic_virus_model(self, virus_model: Dict, genome_results: Dict, virus_type: str) -> Optional[Path]:
        """Save realistic virus model"""
        model_file = virus_model.get('model_file')
        if model_file and Path(model_file).exists():
            return Path(model_file)
        
        # Fallback to detailed model saving
        return self._save_detailed_virus_model(virus_model, genome_results)
    
    def _create_detailed_virus_structure(self, diameter: float, shape: str,
                                        protein_structures: List[Dict],
                                        proteins: List[Dict]) -> Dict:
        """Create detailed virus structure with protein positions"""
        radius = diameter / 2.0
        
        # Create icosahedral capsid positions
        if shape == "icosahedral":
            capsid_positions = self._create_icosahedral_capsid_positions(radius, len(protein_structures))
        else:
            capsid_positions = self._create_spherical_capsid_positions(radius, len(protein_structures))
        
        # Assign proteins to capsid positions
        virus_structure = {
            'capsid': {
                'diameter': diameter,
                'radius': radius,
                'shape': shape,
                'positions': capsid_positions
            },
            'proteins': []
        }
        
        # Position each protein on the capsid
        for i, (protein, pos) in enumerate(zip(protein_structures, capsid_positions)):
            virus_structure['proteins'].append({
                'protein_id': protein['protein_id'],
                'position': pos.tolist() if isinstance(pos, np.ndarray) else pos,
                'structure_file': protein.get('pdb_file'),
                'sequence_length': protein.get('length', 0),
                'structure_type': protein.get('structure_type', 'modeled')
            })
        
        return virus_structure
    
    def _create_icosahedral_capsid_positions(self, radius: float, num_proteins: int) -> List[np.ndarray]:
        """Create positions for proteins on icosahedral capsid"""
        positions = []
        
        # Icosahedron has 12 vertices, 20 faces
        # For virus capsid, proteins are arranged in a T=1, T=3, etc. icosahedral symmetry
        
        # Create icosahedral symmetry positions
        phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        
        # Basic icosahedron vertices
        vertices = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
        ])
        
        # Normalize and scale
        vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True) * radius
        
        # Add more positions for larger capsids (T=3 symmetry for ~180 proteins)
        if num_proteins > 12:
            # Create T=3 icosahedral positions (60 asymmetric units * 3 = 180 positions)
            additional_positions = []
            for i in range(12, min(num_proteins, 180)):
                # Distribute on icosahedral faces
                face_idx = i % 20
                # Position on face
                u = np.random.random()
                v = np.random.random()
                if u + v > 1:
                    u = 1 - u
                    v = 1 - v
                
                # Get face vertices
                v1 = vertices[face_idx % 12]
                v2 = vertices[(face_idx + 1) % 12]
                v3 = vertices[(face_idx + 2) % 12]
                
                pos = u * v1 + v * v2 + (1 - u - v) * v3
                pos = pos / np.linalg.norm(pos) * radius
                additional_positions.append(pos)
            
            positions = list(vertices) + additional_positions
        else:
            positions = list(vertices[:num_proteins])
        
        return positions
    
    def _create_spherical_capsid_positions(self, radius: float, num_proteins: int) -> List[np.ndarray]:
        """Create positions for proteins on spherical capsid"""
        positions = []
        
        # Use Fibonacci sphere distribution for even spacing
        for i in range(num_proteins):
            theta = np.pi * i / num_proteins
            phi = 2 * np.pi * i * 0.618033988749895  # Golden angle
            
            x = radius * np.sin(theta) * np.cos(phi)
            y = radius * np.sin(theta) * np.sin(phi)
            z = radius * np.cos(theta)
            
            positions.append(np.array([x, y, z]))
        
        return positions
    
    def _estimate_virus_diameter(self, genome_length: int, protein_count: int) -> float:
        """Estimate virus diameter in nanometers"""
        # Estimate diameter based on genome length (generic formula)
        if genome_length > 25000:
            # Large RNA virus
            base_diameter = 80 + min((genome_length - 25000) / 5000 * 40, 40)
        elif genome_length > 10000:
            base_diameter = 40 + ((genome_length - 10000) / 15000) * 40
        else:
            base_diameter = 20 + (genome_length / 10000) * 20
        
        # Adjust based on protein count
        if protein_count > 0:
            protein_factor = min(protein_count / 50.0, 1.3)
            base_diameter *= protein_factor
        
        return round(base_diameter, 1)
    
    def _predict_capsid_shape(self, genome_length: int, protein_count: int) -> str:
        """Predict viral capsid shape"""
        if genome_length > 25000:
            return "icosahedral"  # Large RNA viruses typically icosahedral
        elif genome_length > 10000:
            return "icosahedral"
        else:
            return "icosahedral"
    
    def _save_detailed_virus_model(self, virus_model: Dict, genome_results: Dict) -> Optional[Path]:
        """Save detailed virus model as multi-PDB file with all proteins"""
        try:
            output_dir = Path('results/structures')
            output_dir.mkdir(parents=True, exist_ok=True)
            
            import hashlib
            genome_hash = hashlib.md5(str(genome_results.get('length', 0)).encode()).hexdigest()[:8]
            pdb_file = output_dir / f'virus_particle_detailed_{genome_hash}.pdb'
            
            with open(pdb_file, 'w') as f:
                f.write("REMARK   VSim Detailed Virus Particle Model\n")
                f.write("REMARK   Individual protein structures arranged in icosahedral capsid\n")
                f.write(f"REMARK   Capsid diameter: {virus_model['capsid']['diameter']} nm\n")
                f.write(f"REMARK   Shape: {virus_model['capsid']['shape']}\n")
                f.write(f"REMARK   Number of proteins: {len(virus_model['proteins'])}\n\n")
                
                # Write each protein structure at its capsid position
                atom_serial = 1
                for chain_idx, protein in enumerate(virus_model['proteins']):
                    chain_id = chr(65 + (chain_idx % 26))  # A-Z
                    position = protein['position']
                    
                    # Load protein structure if available
                    if protein.get('structure_file') and Path(protein['structure_file']).exists():
                        # Read and translate protein structure
                        try:
                            with open(protein['structure_file'], 'r') as pf:
                                pdb_lines = pf.readlines()
                            
                            # Translate protein to capsid position
                            for line in pdb_lines:
                                if line.startswith('ATOM'):
                                    # Parse atom coordinates
                                    try:
                                        x = float(line[30:38].strip())
                                        y = float(line[38:46].strip())
                                        z = float(line[46:54].strip())
                                        
                                        # Translate to capsid position
                                        x += position[0]
                                        y += position[1]
                                        z += position[2]
                                        
                                        # Update chain ID and atom serial
                                        new_line = line[:6] + f"{atom_serial:5d}" + line[11:21] + \
                                                  chain_id + line[22:30] + \
                                                  f"{x:8.3f}{y:8.3f}{z:8.3f}" + line[54:]
                                        f.write(new_line)
                                        atom_serial += 1
                                    except (ValueError, IndexError):
                                        pass
                        except Exception as e:
                            self.logger.warning(f"Could not load protein structure: {e}")
                            # Generate simplified representation
                            for i in range(10):
                                angle = (i / 10.0) * 2 * np.pi
                                x = position[0] + 5.0 * np.cos(angle)
                                y = position[1] + 5.0 * np.sin(angle)
                                z = position[2]
                                
                                f.write(
                                    f"ATOM  {atom_serial:5d}  CA  PRO {chain_id:1s}{protein['protein_id']:4d}    "
                                    f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 50.00\n"
                                )
                                atom_serial += 1
                    
                    # Generate simplified structure if no file available
                    else:
                        # Create simplified representation at position
                        for i in range(min(protein.get('sequence_length', 10), 50)):
                            angle = (i / max(protein.get('sequence_length', 10), 1)) * 2 * np.pi
                            x = position[0] + 5.0 * np.cos(angle)
                            y = position[1] + 5.0 * np.sin(angle)
                            z = position[2] + i * 0.5
                            
                            f.write(
                                f"ATOM  {atom_serial:5d}  CA  PRO {chain_id:1s}{protein['protein_id']:4d}    "
                                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 50.00\n"
                            )
                            atom_serial += 1
                
                f.write("END\n")
            
            self.logger.info(f"Saved detailed virus model with {len(virus_model['proteins'])} proteins to {pdb_file}")
            return pdb_file
        except Exception as e:
            self.logger.warning(f"Failed to save detailed virus model: {e}")
            return None
    
    def _create_detailed_visualization_data(self, virus_model: Dict, diameter: float, 
                                           shape: str, protein_structures: List[Dict]) -> Dict:
        """Create data for detailed JavaScript visualization"""
        return {
            'diameter': float(diameter),
            'shape': shape,
            'radius': float(diameter / 2.0),
            'protein_count': len(protein_structures),
            'proteins': [
                {
                    'id': p['protein_id'],
                    'position': virus_model['proteins'][i]['position'] if i < len(virus_model['proteins']) else [0, 0, 0],
                    'structure_file': p.get('pdb_file'),
                    'length': p.get('length', 0)
                }
                for i, p in enumerate(protein_structures)
            ]
        }
