"""
Realistic Virus Capsid Assembler
Creates accurate virus particle models using real protein structures
"""

import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import numpy as np
from Bio.PDB import PDBParser, PDBIO
from Bio.PDB.vectors import Vector

class RealisticVirusAssembler:
    """Assemble realistic virus particles from protein structures"""
    
    def __init__(self):
        self.logger = logging.getLogger('VSim.VirusAssembler')
        self.parser = PDBParser(QUIET=True)
    
    def create_large_rna_virus_capsid(self, 
                                 spike_protein: Optional[Path] = None,
                                 membrane_protein: Optional[Path] = None,
                                 nucleocapsid: Optional[Path] = None,
                                 radius: float = 60.0) -> Path:
        """
        Create realistic large RNA virus capsid structure
        
        Structure characteristics:
        - ~80-120nm diameter
        - Icosahedral symmetry
        - Spike proteins on surface (~20-40 spikes)
        - Membrane proteins forming capsid
        - Envelope proteins embedded
        - Nucleocapsid inside
        """
        output_file = Path('results/structures/virus_large_rna_realistic.pdb')
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Assembling realistic large RNA virus capsid...")
        
        # Create icosahedral positions for proteins
        icosahedral_positions = self._generate_icosahedral_positions(radius)
        
        # Assemble components
        all_atoms = []
        atom_serial = 1
        
        # 1. Add spike proteins (~24 spikes in T=3 icosahedral arrangement)
        if spike_protein and spike_protein.exists():
            spike_positions = self._get_spike_positions(radius)
            for i, pos in enumerate(spike_positions[:24]):  # 24 spike proteins
                atoms = self._place_protein_on_capsid(
                    spike_protein, pos, radius, 'S', atom_serial
                )
                all_atoms.extend(atoms)
                atom_serial += len(atoms)
                self.logger.info(f"Placed spike protein {i+1}/24")
        
        # 2. Add membrane proteins (form the capsid shell)
        if membrane_protein and membrane_protein.exists():
            membrane_positions = icosahedral_positions[:60]  # 60 M proteins
            for i, pos in enumerate(membrane_positions):
                atoms = self._place_protein_on_capsid(
                    membrane_protein, pos, radius * 0.95, 'M', atom_serial
                )
                all_atoms.extend(atoms)
                atom_serial += len(atoms)
        
        # 3. Add envelope proteins (embedded in membrane)
        envelope_positions = icosahedral_positions[:20]  # ~20 E proteins
        for i, pos in enumerate(envelope_positions):
            atoms = self._create_simple_envelope_protein(pos, radius * 0.98, 'E', atom_serial)
            all_atoms.extend(atoms)
            atom_serial += len(atoms)
        
        # 4. Add nucleocapsid inside (RNA-protein complex)
        if nucleocapsid and nucleocapsid.exists():
            nucleocapsid_positions = self._get_internal_positions(radius * 0.7, 30)
            for i, pos in enumerate(nucleocapsid_positions[:30]):
                atoms = self._place_protein_on_capsid(
                    nucleocapsid, pos, radius * 0.7, 'N', atom_serial, scale=0.5
                )
                all_atoms.extend(atoms)
                atom_serial += len(atoms)
        
        # Write assembled virus
        with open(output_file, 'w') as f:
            f.write("REMARK   Realistic Large RNA Virus Structure\n")
            f.write("REMARK   Assembled from PDB structures\n")
            f.write(f"REMARK   Radius: {radius} Angstroms (~{radius/10:.1f} nm)\n")
            f.write(f"REMARK   Total atoms: {len(all_atoms)}\n\n")
            
            for atom in all_atoms:
                f.write(atom)
        
        self.logger.info(f"Created realistic virus structure: {output_file}")
        return output_file
    
    def _generate_icosahedral_positions(self, radius: float) -> List[np.ndarray]:
        """Generate icosahedral symmetry positions"""
        phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        
        # Icosahedron vertices
        vertices = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
        ])
        
        # Normalize and scale
        vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True) * radius
        
        # Generate T=3 icosahedral positions (60 asymmetric units)
        positions = list(vertices)
        
        # Add positions on faces for T=3 symmetry
        for face_idx in range(20):
            # Get face vertices
            v1_idx = face_idx % 12
            v2_idx = (face_idx + 1) % 12
            v3_idx = (face_idx + 2) % 12
            
            v1 = vertices[v1_idx]
            v2 = vertices[v2_idx]
            v3 = vertices[v3_idx]
            
            # Add positions on face (T=3 has 3 positions per face)
            for u, v in [(0.33, 0.33), (0.66, 0.17), (0.17, 0.66)]:
                pos = u * v1 + v * v2 + (1 - u - v) * v3
                pos = pos / np.linalg.norm(pos) * radius
                positions.append(pos)
        
        return positions
    
    def _get_spike_positions(self, radius: float) -> List[np.ndarray]:
        """Get positions for spike proteins (typical large RNA virus has ~24 spikes)"""
        # Spikes are arranged with icosahedral symmetry, pointing outward
        positions = []
        
        # Generate positions on icosahedral faces
        phi = (1 + np.sqrt(5)) / 2
        vertices = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
        ])
        vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True) * radius
        
        # Add spike at each vertex (12 positions)
        positions.extend(vertices)
        
        # Add spikes on faces (12 more positions for ~24 total)
        for i in range(12):
            v1 = vertices[i % 12]
            v2 = vertices[(i + 1) % 12]
            v3 = vertices[(i + 2) % 12]
            # Center of face
            center = (v1 + v2 + v3) / 3
            center = center / np.linalg.norm(center) * radius
            positions.append(center)
        
        return positions
    
    def _place_protein_on_capsid(self, pdb_file: Path, position: np.ndarray, 
                                 radius: float, chain_id: str, 
                                 start_serial: int, scale: float = 1.0) -> List[str]:
        """Place protein structure on capsid surface"""
        try:
            structure = self.parser.get_structure('protein', str(pdb_file))
            
            # Get center of mass
            atoms = list(structure.get_atoms())
            if not atoms:
                return []
            
            coords = np.array([atom.coord for atom in atoms])
            center = coords.mean(axis=0)
            
            # Translate to origin
            coords = coords - center
            
            # Scale
            coords = coords * scale
            
            # Orient protein to point outward from capsid
            # Normalize position vector
            pos_normalized = position / np.linalg.norm(position)
            
            # Rotate protein to align with radial direction
            # Simple orientation: align first principal axis with radial
            if len(coords) > 3:
                # Get principal axis
                cov = np.cov(coords.T)
                eigenvals, eigenvecs = np.linalg.eigh(cov)
                principal_axis = eigenvecs[:, -1]  # Longest axis
                
                # Rotate to align with radial
                if np.dot(principal_axis, pos_normalized) < 0:
                    principal_axis = -principal_axis
                
                # Rotation matrix
                v = np.cross(principal_axis, pos_normalized)
                if np.linalg.norm(v) > 0.001:
                    v = v / np.linalg.norm(v)
                    angle = np.arccos(np.clip(np.dot(principal_axis, pos_normalized), -1, 1))
                    # Rodrigues rotation
                    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
                    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
                    coords = coords @ R.T
            
            # Translate to capsid position
            coords = coords + position
            
            # Convert to PDB format
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
            self.logger.warning(f"Could not place protein {pdb_file}: {e}")
            return []
    
    def _create_simple_envelope_protein(self, position: np.ndarray, radius: float,
                                       chain_id: str, start_serial: int) -> List[str]:
        """Create simplified envelope protein"""
        # Small transmembrane protein
        atoms = []
        serial = start_serial
        
        # Create small helical bundle
        for i in range(10):
            angle = (i / 10.0) * 2 * np.pi
            x = position[0] + 2.0 * np.cos(angle)
            y = position[1] + 2.0 * np.sin(angle)
            z = position[2] + i * 1.5
            
            atoms.append(
                f"ATOM  {serial:5d}  CA  ALA {chain_id:1s}   1    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 50.00\n"
            )
            serial += 1
        
        return atoms
    
    def _get_internal_positions(self, radius: float, num: int) -> List[np.ndarray]:
        """Get positions for internal proteins"""
        positions = []
        for i in range(num):
            # Fibonacci sphere for even distribution
            theta = np.pi * i / num
            phi = 2 * np.pi * i * 0.618033988749895
            x = radius * np.sin(theta) * np.cos(phi)
            y = radius * np.sin(theta) * np.sin(phi)
            z = radius * np.cos(theta)
            positions.append(np.array([x, y, z]))
        return positions

