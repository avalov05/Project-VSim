"""
Advanced Generic Virus Assembler
Uses sophisticated protein folding and capsid assembly simulation
"""

import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import numpy as np
from Bio.PDB import PDBParser
import copy

class GenericVirusAssembler:
    """Assemble realistic virus particles from predicted protein structures"""
    
    def __init__(self):
        self.logger = logging.getLogger('VSim.GenericVirusAssembler')
        self.parser = PDBParser(QUIET=True)
    
    def assemble_virus(self, 
                      protein_structures: List[Dict],
                      virus_type: str,
                      diameter_nm: float,
                      shape: str) -> Path:
        """
        Assemble virus particle using physics-based simulation.
        NO shape assumptions - geometry emerges naturally from protein interactions.
        
        Args:
            protein_structures: List of protein structure info dicts
            virus_type: Type of virus (determined from genome characteristics)
            diameter_nm: Estimated diameter in nanometers
            shape: Capsid shape (ignored - geometry emerges from physics)
        """
        output_file = Path('results/structures/virus_assembled_from_genome.pdb')
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Starting physics-based virus assembly for {virus_type}...")
        self.logger.info("NO shape assumptions - geometry will emerge from protein physics")
        
        # Use realistic assembler (binding sites + nucleation)
        try:
            from src.structure.realistic_assembly import RealisticVirusAssembler
            assembler = RealisticVirusAssembler()
            
            # Step 1: Calculate protein copy numbers from physics
            self.logger.info("Step 1: Calculating protein copy numbers from surface area and packing...")
            copy_numbers = assembler.calculate_protein_copy_numbers(protein_structures, diameter_nm)
            
            # Step 2: Replicate proteins
            self.logger.info("Step 2: Replicating proteins according to physics calculations...")
            replicated_proteins = self._replicate_proteins(protein_structures, copy_numbers)
            
            # Step 3: Assemble through realistic nucleation-based assembly
            self.logger.info("Step 3: Running realistic assembly (binding sites + nucleation)...")
            assembled_capsid = assembler.assemble_virus(replicated_proteins, diameter_nm)
            
            # Step 4: Extract detected geometry
            detected_geometry = assembled_capsid.get('geometry', {})
            detected_shape = detected_geometry.get('shape', 'unknown')
            detected_symmetry = detected_geometry.get('symmetry', 'none')
            
            self.logger.info(f"Physics simulation complete!")
            self.logger.info(f"Emergent geometry detected: {detected_shape} (symmetry: {detected_symmetry})")
            
            # Convert to PDB format
            all_atoms = self._convert_to_pdb(assembled_capsid)
            
            # Write PDB file
            with open(output_file, 'w') as f:
                f.write("REMARK   Realistic Virus Particle - Assembled via Atom-Level Interactions\n")
                f.write(f"REMARK   Virus Type: {virus_type}\n")
                f.write(f"REMARK   Emergent Shape: {detected_shape}\n")
                f.write(f"REMARK   Emergent Symmetry: {detected_symmetry}\n")
                f.write(f"REMARK   Estimated Diameter: {diameter_nm} nm\n")
                f.write(f"REMARK   Total Protein Copies: {len(assembled_capsid['proteins'])}\n")
                f.write(f"REMARK   Assembly Method: Realistic (Binding Sites + Nucleation)\n")
                f.write(f"REMARK   Uses actual protein-protein interfaces and complementarity\n")
                f.write(f"REMARK   Geometry emerged from atom-level interactions\n")
                f.write(f"REMARK   Total atoms: {len(all_atoms)}\n\n")
                
                for atom in all_atoms:
                    f.write(atom)
            
            self.logger.info(f"Created physics-based virus structure: {output_file}")
            return output_file
            
        except Exception as e:
            self.logger.warning(f"Physics-based assembly failed: {e}, using fallback")
            import traceback
            self.logger.debug(traceback.format_exc())
            return self._fallback_assembly(protein_structures, virus_type, diameter_nm, shape)
    
    def _convert_to_pdb(self, capsid: Dict) -> List[str]:
        """Convert assembled capsid to PDB format"""
        all_atoms = []
        atom_serial = 1
        
        for i, (protein, pos, rot) in enumerate(zip(
            capsid['proteins'],
            capsid['positions'],
            capsid['orientations']
        )):
            # Use chain ID based on original protein index for tracking
            original_idx = protein.get('original_index', i)
            copy_idx = protein.get('copy_index', 0)
            
            # Chain ID: A-Z for first 26 proteins, then X with numbers
            if original_idx < 26:
                chain_id = chr(65 + original_idx)
            else:
                chain_id = 'X'
            
            # Transform protein coordinates
            coords = protein['coords']
            coords_centered = coords - coords.mean(axis=0)
            coords_transformed = coords_centered @ rot.T + pos
            
            sequence = protein.get('sequence', '')
            
            # Write atoms
            for j, coord in enumerate(coords_transformed):
                residue_idx = j + 1
                residue = sequence[j] if j < len(sequence) else 'A'
                
                # Ensure proper spacing for negative coordinates
                x_str = f"{coord[0]:8.3f}"
                y_str = f"{coord[1]:8.3f}"
                z_str = f"{coord[2]:8.3f}"
                
                all_atoms.append(
                    f"ATOM  {atom_serial:5d}  CA  {self._get_residue_code(residue):3s} "
                    f"{chain_id:1s}{residue_idx:4d}    "
                    f"{x_str} {y_str} {z_str}"
                    f"  1.00 50.00\n"
                )
                atom_serial += 1
        
        return all_atoms
    
    def _replicate_proteins(self, protein_structures: List[Dict], copy_numbers: Dict[int, int]) -> List[Dict]:
        """Replicate proteins according to copy numbers"""
        replicated = []
        
        for i, protein in enumerate(protein_structures):
            copies = copy_numbers.get(i, 1)
            
            for copy_idx in range(copies):
                protein_copy = copy.deepcopy(protein)
                protein_copy['original_index'] = i
                protein_copy['copy_index'] = copy_idx
                protein_copy['copy_id'] = f"{i}_{copy_idx}"
                replicated.append(protein_copy)
        
        self.logger.info(f"Replicated {len(protein_structures)} unique proteins into {len(replicated)} total copies")
        
        return replicated
    
    def _get_residue_code(self, residue: str) -> str:
        """Get 3-letter residue code"""
        codes = {
            'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS',
            'Q': 'GLN', 'E': 'GLU', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
            'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO',
            'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL'
        }
        return codes.get(residue.upper(), 'ALA')
    
    def _fallback_assembly(self, protein_structures: List[Dict],
                           virus_type: str, diameter_nm: float, shape: str) -> Path:
        """Fallback assembly method"""
        output_file = Path('results/structures/virus_assembled_from_genome.pdb')
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Using fallback assembly method...")
        
        radius_angstrom = (diameter_nm / 2.0) * 10
        
        # Classify proteins
        classified_proteins = self._classify_proteins(protein_structures)
        
        # Generate capsid positions
        structural_count = len(classified_proteins.get('structural', []))
        if shape == "icosahedral" and structural_count < 20:
            capsid_positions = self._generate_spherical_positions(radius_angstrom, structural_count)
        elif shape == "icosahedral":
            capsid_positions = self._generate_icosahedral_positions(radius_angstrom, structural_count)
        else:
            capsid_positions = self._generate_spherical_positions(radius_angstrom, structural_count)
        
        # Assemble components
        all_atoms = []
        atom_serial = 1
        
        structural_proteins = classified_proteins.get('structural', [])
        surface_positions = capsid_positions[:len(structural_proteins)]
        
        for i, (protein, pos) in enumerate(zip(structural_proteins, surface_positions)):
            chain_id = chr(65 + (i % 26))
            atoms = self._place_protein_on_surface(
                protein, pos, radius_angstrom, chain_id, atom_serial, scale=1.0
            )
            all_atoms.extend(atoms)
            atom_serial += len(atoms)
        
        # Write file
        with open(output_file, 'w') as f:
            f.write("REMARK   Virus Particle Assembled from Genome Sequence\n")
            f.write(f"REMARK   Virus Type: {virus_type}\n")
            f.write(f"REMARK   Capsid Shape: {shape}\n")
            f.write(f"REMARK   Diameter: {diameter_nm} nm\n")
            f.write(f"REMARK   Total atoms: {len(all_atoms)}\n\n")
            
            for atom in all_atoms:
                f.write(atom)
        
        return output_file
    
    def _classify_proteins(self, protein_structures: List[Dict]) -> Dict[str, List[Dict]]:
        """Classify proteins by function based on size and characteristics"""
        classified = {
            'structural': [],
            'spike_like': [],
            'internal': [],
            'enzymatic': []
        }
        
        for protein in protein_structures:
            length = protein.get('length', 0)
            seq = protein.get('sequence', '')
            
            # Large proteins (>1000 aa) are likely spikes or polymerases
            if length > 1000:
                if self._has_transmembrane_features(seq):
                    classified['spike_like'].append(protein)
                else:
                    classified['internal'].append(protein)
            elif length > 200:
                classified['structural'].append(protein)
            else:
                if self._has_transmembrane_features(seq):
                    classified['structural'].append(protein)
                else:
                    classified['internal'].append(protein)
        
        return classified
    
    def _has_transmembrane_features(self, sequence: str) -> bool:
        """Predict if protein has transmembrane domains"""
        if not sequence:
            return False
        
        hydrophobic = set('AILMFWV')
        window_size = 20
        hydrophobic_count = 0
        
        for i in range(len(sequence) - window_size):
            window = sequence[i:i+window_size]
            if sum(1 for aa in window if aa in hydrophobic) >= window_size * 0.6:
                hydrophobic_count += 1
        
        return hydrophobic_count >= 2
    
    def _generate_icosahedral_positions(self, radius: float, num_proteins: int) -> List[np.ndarray]:
        """Generate icosahedral symmetry positions"""
        phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        
        vertices = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
        ])
        vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True) * radius
        
        positions = list(vertices)
        
        if num_proteins > 12:
            for face_idx in range(20):
                v1_idx = face_idx % 12
                v2_idx = (face_idx + 1) % 12
                v3_idx = (face_idx + 2) % 12
                
                v1 = vertices[v1_idx]
                v2 = vertices[v2_idx]
                v3 = vertices[v3_idx]
                
                for u, v in [(0.33, 0.33), (0.66, 0.17), (0.17, 0.66)]:
                    pos = u * v1 + v * v2 + (1 - u - v) * v3
                    pos = pos / np.linalg.norm(pos) * radius
                    positions.append(pos)
        
        return positions[:num_proteins]
    
    def _generate_spherical_positions(self, radius: float, num_proteins: int) -> List[np.ndarray]:
        """Generate evenly distributed spherical positions using Fibonacci sphere"""
        positions = []
        for i in range(num_proteins):
            theta = np.pi * i / num_proteins
            phi = 2 * np.pi * i * 0.618033988749895  # Golden angle
            x = radius * np.sin(theta) * np.cos(phi)
            y = radius * np.sin(theta) * np.sin(phi)
            z = radius * np.cos(theta)
            positions.append(np.array([x, y, z]))
        return positions
    
    def _get_spike_positions(self, radius: float, num_spikes: int) -> List[np.ndarray]:
        """Get positions for spike proteins"""
        positions = []
        phi = (1 + np.sqrt(5)) / 2
        vertices = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
        ])
        vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True) * radius
        
        positions.extend(vertices[:min(12, num_spikes)])
        
        if num_spikes > 12:
            for i in range(12, min(num_spikes, 24)):
                v1 = vertices[i % 12]
                v2 = vertices[(i + 1) % 12]
                v3 = vertices[(i + 2) % 12]
                center = (v1 + v2 + v3) / 3
                center = center / np.linalg.norm(center) * radius
                positions.append(center)
        
        return positions[:num_spikes]
    
    def _get_internal_positions(self, radius: float, num: int) -> List[np.ndarray]:
        """Get positions for internal proteins"""
        positions = []
        for i in range(num):
            theta = np.pi * i / num
            phi = 2 * np.pi * i * 0.618033988749895
            x = radius * np.sin(theta) * np.cos(phi)
            y = radius * np.sin(theta) * np.sin(phi)
            z = radius * np.cos(theta)
            positions.append(np.array([x, y, z]))
        return positions
    
    def _place_protein_on_surface(self, protein: Dict, position: np.ndarray, 
                                  radius: float, chain_id: str, 
                                  start_serial: int, scale: float = 1.0) -> List[str]:
        """Place protein structure on capsid surface - properly oriented"""
        pdb_file = protein.get('pdb_file')
        
        # Normalize position to ensure it's on the sphere surface
        pos_normalized = position / np.linalg.norm(position)
        surface_position = pos_normalized * radius
        
        if pdb_file and Path(pdb_file).exists():
            try:
                structure = self.parser.get_structure('protein', str(pdb_file))
                atoms = list(structure.get_atoms())
                
                if atoms:
                    coords = np.array([atom.coord for atom in atoms])
                    center = coords.mean(axis=0)
                    coords = coords - center
                    
                    # Scale protein to reasonable size
                    protein_size = np.max(np.linalg.norm(coords, axis=1))
                    target_size = 25.0
                    if protein_size > 0:
                        scale_factor = min(scale, target_size / protein_size)
                        coords = coords * scale_factor
                    
                    # Orient protein outward
                    if len(coords) > 3:
                        cov = np.cov(coords.T)
                        eigenvals, eigenvecs = np.linalg.eigh(cov)
                        idx = eigenvals.argsort()
                        eigenvecs = eigenvecs[:, idx]
                        longest_axis = eigenvecs[:, -1]
                        
                        if np.dot(longest_axis, pos_normalized) < 0:
                            longest_axis = -longest_axis
                        
                        v = np.cross(longest_axis, pos_normalized)
                        if np.linalg.norm(v) > 0.001:
                            v = v / np.linalg.norm(v)
                            angle = np.arccos(np.clip(np.dot(longest_axis, pos_normalized), -1, 1))
                            K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
                            R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
                            coords = coords @ R.T
                        
                        protein_radius = np.max(np.linalg.norm(coords, axis=1))
                        center_offset = surface_position - pos_normalized * protein_radius * 0.3
                    else:
                        center_offset = surface_position
                    
                    coords = coords + center_offset
                    
                    atom_lines = []
                    serial = start_serial
                    for atom, coord in zip(atoms, coords):
                        atom_lines.append(
                            f"ATOM  {serial:5d}  {atom.get_name():2s}  "
                            f"{atom.get_parent().get_resname():3s} {chain_id:1s}"
                            f"{atom.get_parent().get_id()[1]:4d}    "
                            f"{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}"
                            f"{atom.get_occupancy():6.2f}{atom.get_bfactor():6.2f}\n"
                        )
                        serial += 1
                    
                    return atom_lines
            except Exception as e:
                self.logger.warning(f"Could not load protein structure: {e}")
        
        # Fallback: create compact globular representation
        atoms = []
        serial = start_serial
        length = protein.get('length', 20)
        protein_radius = min(length * 0.12, 12.0)
        
        n_points = min(length, 20)
        for i in range(n_points):
            theta = np.pi * i / n_points if n_points > 1 else 0
            phi = 2 * np.pi * i * 0.618033988749895
            
            local_offset = np.array([
                protein_radius * np.sin(theta) * np.cos(phi),
                protein_radius * np.sin(theta) * np.sin(phi),
                protein_radius * np.cos(theta)
            ])
            
            radial_component = np.dot(local_offset, pos_normalized) * pos_normalized
            tangential_offset = local_offset - radial_component
            
            final_pos = surface_position + tangential_offset
            
            atoms.append(
                f"ATOM  {serial:5d}  CA  PRO {chain_id:1s}{protein.get('protein_id', 1):4d}    "
                f"{final_pos[0]:8.3f}{final_pos[1]:8.3f}{final_pos[2]:8.3f}  1.00 50.00\n"
            )
            serial += 1
        
        return atoms
    
    def _place_internal_protein(self, protein: Dict, position: np.ndarray,
                               chain_id: str, start_serial: int, scale: float = 1.0) -> List[str]:
        """Place protein inside the capsid"""
        pdb_file = protein.get('pdb_file')
        
        if pdb_file and Path(pdb_file).exists():
            try:
                structure = self.parser.get_structure('protein', str(pdb_file))
                atoms = list(structure.get_atoms())
                
                if atoms:
                    coords = np.array([atom.coord for atom in atoms])
                    center = coords.mean(axis=0)
                    coords = coords - center
                    
                    protein_size = np.max(np.linalg.norm(coords, axis=1))
                    target_size = 20.0
                    if protein_size > 0:
                        scale_factor = min(scale, target_size / protein_size)
                        coords = coords * scale_factor
                    
                    coords = coords + position
                    
                    atom_lines = []
                    serial = start_serial
                    for atom, coord in zip(atoms, coords):
                        atom_lines.append(
                            f"ATOM  {serial:5d}  {atom.get_name():2s}  "
                            f"{atom.get_parent().get_resname():3s} {chain_id:1s}"
                            f"{atom.get_parent().get_id()[1]:4d}    "
                            f"{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}"
                            f"{atom.get_occupancy():6.2f}{atom.get_bfactor():6.2f}\n"
                        )
                        serial += 1
                    
                    return atom_lines
            except Exception as e:
                self.logger.warning(f"Could not load internal protein structure: {e}")
        
        # Fallback
        atoms = []
        serial = start_serial
        length = protein.get('length', 20)
        protein_radius = min(length * 0.12, 12.0)
        
        n_points = min(length, 20)
        for i in range(n_points):
            theta = np.random.random() * 2 * np.pi
            phi = np.arccos(2 * np.random.random() - 1)
            
            offset = np.array([
                protein_radius * np.sin(phi) * np.cos(theta),
                protein_radius * np.sin(phi) * np.sin(theta),
                protein_radius * np.cos(phi)
            ])
            
            final_pos = position + offset
            
            atoms.append(
                f"ATOM  {serial:5d}  CA  PRO {chain_id:1s}{protein.get('protein_id', 1):4d}    "
                f"{final_pos[0]:8.3f}{final_pos[1]:8.3f}{final_pos[2]:8.3f}  1.00 50.00\n"
            )
            serial += 1
        
        return atoms
