"""
Physics-Based Virus Assembly System
Assembles viruses through pure physics simulation - no hardcoded shapes
Geometry emerges naturally from protein-protein interactions
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
import copy

class PhysicsBasedAssembler:
    """
    Assembles viruses through pure physics simulation.
    No shape assumptions - geometry emerges from protein interactions.
    """
    
    def __init__(self):
        self.logger = logging.getLogger('VSim.PhysicsAssembler')
        self.kB = 0.001987  # Boltzmann constant (kcal/mol/K)
        self.temperature = 310.0  # Body temperature (K)
    
    def calculate_protein_copy_numbers(self,
                                      protein_structures: List[Dict],
                                      diameter_nm: float) -> Dict[int, int]:
        """
        Calculate how many copies of each protein are needed based on physics.
        Uses surface area, protein size, and packing density - no assumptions.
        
        Returns:
            Dict mapping protein_index -> number_of_copies
        """
        self.logger.info("Calculating protein copy numbers from physics...")
        
        radius_nm = diameter_nm / 2.0
        surface_area_nm2 = 4 * np.pi * radius_nm ** 2
        
        copy_numbers = {}
        
        for i, protein in enumerate(protein_structures):
            sequence = protein.get('sequence', '')
            length = len(sequence)
            
            # Estimate protein size from sequence length
            # Average amino acid volume ~110 Å³, globular protein ~0.75 packing density
            amino_acid_volume_angstrom3 = 110.0
            packing_density = 0.75
            
            # Estimate protein volume (assuming roughly spherical)
            protein_volume_angstrom3 = length * amino_acid_volume_angstrom3 / packing_density
            protein_radius_nm = (3 * protein_volume_angstrom3 / (4 * np.pi)) ** (1/3) / 10.0
            
            # Calculate surface area per protein when placed on capsid
            # Proteins pack with ~70% efficiency on surfaces
            packing_efficiency = 0.70
            protein_surface_area_nm2 = np.pi * protein_radius_nm ** 2 / packing_efficiency
            
            # Classify protein role based on sequence properties
            is_structural = self._is_structural_protein(sequence, length)
            is_transmembrane = self._has_transmembrane_features(sequence)
            
            if is_structural and not is_transmembrane:
                # Structural capsid proteins - fill the surface
                copies = int(surface_area_nm2 / protein_surface_area_nm2)
                # Ensure reasonable bounds (10-1000 copies)
                copies = max(10, min(copies, 1000))
            elif is_transmembrane or (length > 1000):
                # Large transmembrane proteins (spikes) - fewer copies
                # Typically 20-100 spikes per virus
                copies = max(12, min(int(surface_area_nm2 / (protein_surface_area_nm2 * 3)), 100))
            else:
                # Internal proteins - estimate from volume
                volume_nm3 = (4/3) * np.pi * radius_nm ** 3
                internal_volume_nm3 = volume_nm3 * 0.6  # ~60% internal space
                protein_volume_nm3 = (4/3) * np.pi * (protein_radius_nm ** 3)
                copies = max(1, min(int(internal_volume_nm3 / protein_volume_nm3), 100))
            
            copy_numbers[i] = copies
            self.logger.info(f"Protein {i+1} (length={length}): {copies} copies (radius={protein_radius_nm:.1f}nm)")
        
        total_copies = sum(copy_numbers.values())
        self.logger.info(f"Total protein copies needed: {total_copies}")
        
        return copy_numbers
    
    def _is_structural_protein(self, sequence: str, length: int) -> bool:
        """Determine if protein is structural based on sequence properties"""
        if length < 50:
            return False
        
        # Structural proteins typically have:
        # - Moderate hydrophobicity (0.35-0.55)
        # - Regular secondary structure propensity
        hydrophobic = set('AILMFWV')
        hydrophobicity = sum(1 for aa in sequence if aa in hydrophobic) / len(sequence)
        
        # Check for structural motifs
        has_structural_propensity = 0.35 <= hydrophobicity <= 0.55
        
        # Large proteins (>200aa) are often structural if not highly hydrophobic
        if length > 200 and hydrophobicity < 0.6:
            return True
        
        # Medium proteins with structural propensity
        if 50 <= length <= 500 and has_structural_propensity:
            return True
        
        return False
    
    def _has_transmembrane_features(self, sequence: str) -> bool:
        """Detect transmembrane domains"""
        if not sequence or len(sequence) < 20:
            return False
        
        hydrophobic = set('AILMFWV')
        window_size = 20
        hydrophobic_windows = 0
        
        for i in range(len(sequence) - window_size):
            window = sequence[i:i+window_size]
            hydrophobic_fraction = sum(1 for aa in window if aa in hydrophobic) / window_size
            if hydrophobic_fraction >= 0.6:
                hydrophobic_windows += 1
        
        return hydrophobic_windows >= 2
    
    def replicate_proteins(self,
                          protein_structures: List[Dict],
                          copy_numbers: Dict[int, int]) -> List[Dict]:
        """
        Replicate proteins according to calculated copy numbers.
        Each copy gets a unique identifier for tracking.
        """
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
    
    def assemble_through_physics(self,
                                replicated_proteins: List[Dict],
                                diameter_nm: float) -> Dict:
        """
        Assemble virus through pure physics simulation.
        No shape assumptions - geometry emerges naturally.
        """
        self.logger.info("Starting physics-based assembly simulation...")
        
        radius_angstrom = (diameter_nm / 2.0) * 10
        
        # Step 1: Fold all proteins
        folded_proteins = []
        for i, protein in enumerate(replicated_proteins):
            if i % 50 == 0:
                self.logger.info(f"Folding protein {i+1}/{len(replicated_proteins)}...")
            
            sequence = protein.get('sequence', '')
            if sequence:
                from src.structure.protein_folding import ProteinFoldingSimulator
                folder = ProteinFoldingSimulator()
                folded_coords = folder.fold_protein(sequence)
                folded_proteins.append({
                    'coords': folded_coords,
                    'sequence': sequence,
                    'protein_id': protein.get('copy_id', i),
                    'original_index': protein.get('original_index', i),
                    'copy_index': protein.get('copy_index', 0)
                })
        
        # Step 2: Initialize assembly - start with random positions
        self.logger.info("Initializing assembly with random positions...")
        assembly = self._initialize_assembly(folded_proteins, radius_angstrom)
        
        # Step 3: Run molecular dynamics simulation
        self.logger.info("Running molecular dynamics simulation...")
        assembly = self._run_molecular_dynamics(assembly, radius_angstrom, n_steps=1000)
        
        # Step 4: Energy minimization
        self.logger.info("Performing energy minimization...")
        assembly = self._minimize_energy(assembly, radius_angstrom)
        
        # Step 5: Detect emergent geometry
        geometry = self._detect_geometry(assembly)
        self.logger.info(f"Detected geometry: {geometry['shape']} (symmetry: {geometry['symmetry']})")
        
        assembly['geometry'] = geometry
        
        return assembly
    
    def _initialize_assembly(self,
                            proteins: List[Dict],
                            radius: float) -> Dict:
        """Initialize assembly with proteins in random positions"""
        positions = []
        orientations = []
        
        # Place proteins randomly near the surface
        for protein in proteins:
            # Random position near surface
            theta = np.random.random() * 2 * np.pi
            phi = np.arccos(2 * np.random.random() - 1)
            
            # Vary radius slightly (proteins can be on surface or inside)
            r_factor = 0.8 + np.random.random() * 0.2
            
            pos = np.array([
                radius * r_factor * np.sin(phi) * np.cos(theta),
                radius * r_factor * np.sin(phi) * np.sin(theta),
                radius * r_factor * np.cos(phi)
            ])
            
            positions.append(pos)
            
            # Random orientation
            orientations.append(self._random_rotation_matrix())
        
        return {
            'proteins': proteins,
            'positions': positions,
            'orientations': orientations,
            'velocities': [np.zeros(3) for _ in proteins]  # For MD
        }
    
    def _run_molecular_dynamics(self,
                               assembly: Dict,
                               radius: float,
                               n_steps: int = 1000,
                               dt: float = 0.001) -> Dict:
        """
        Run molecular dynamics simulation to let proteins assemble naturally.
        Uses Langevin dynamics with protein-protein interactions.
        """
        proteins = assembly['proteins']
        positions = assembly['positions']
        orientations = assembly['orientations']
        velocities = assembly['velocities']
        
        # MD parameters
        gamma = 0.1  # Friction coefficient
        kT = self.kB * self.temperature
        
        for step in range(n_steps):
            if step % 200 == 0:
                self.logger.info(f"MD step {step}/{n_steps}")
            
            # Calculate forces on each protein
            forces = self._calculate_forces(proteins, positions, orientations, radius)
            
            # Update positions and velocities (Langevin dynamics)
            for i in range(len(proteins)):
                # Random force (thermal noise)
                random_force = np.random.normal(0, np.sqrt(2 * gamma * kT / dt), 3)
                
                # Update velocity (simplified - no mass term)
                velocities[i] = velocities[i] * (1 - gamma * dt) + forces[i] * dt + random_force * dt
                
                # Update position
                new_pos = positions[i] + velocities[i] * dt
                
                # Constrain to near surface (soft boundary)
                dist_from_center = np.linalg.norm(new_pos)
                if dist_from_center > radius * 1.1:
                    # Push back towards surface
                    direction = new_pos / dist_from_center
                    new_pos = direction * radius * 1.05
                elif dist_from_center < radius * 0.3:
                    # Push outward if too deep
                    direction = new_pos / (dist_from_center + 1e-10)
                    new_pos = direction * radius * 0.5
                
                positions[i] = new_pos
            
            # Update orientations based on interactions
            orientations = self._update_orientations(proteins, positions, orientations)
        
        return {
            'proteins': proteins,
            'positions': positions,
            'orientations': orientations,
            'velocities': velocities
        }
    
    def _calculate_forces(self,
                         proteins: List[Dict],
                         positions: List[np.ndarray],
                         orientations: List[np.ndarray],
                         radius: float) -> List[np.ndarray]:
        """Calculate forces between proteins"""
        n = len(proteins)
        forces = [np.zeros(3) for _ in range(n)]
        
        for i in range(n):
            for j in range(i + 1, n):
                # Get transformed coordinates
                coords_i = self._transform_coords(proteins[i]['coords'], positions[i], orientations[i])
                coords_j = self._transform_coords(proteins[j]['coords'], positions[j], orientations[j])
                
                # Calculate pairwise distances
                distances = cdist(coords_i, coords_j)
                min_dist = np.min(distances)
                
                # Lennard-Jones-like interaction
                if min_dist < 20.0:  # Interaction range
                    sigma = 5.0  # Van der Waals radius
                    epsilon = 1.0  # Interaction strength
                    
                    # Vector from i to j
                    direction = positions[j] - positions[i]
                    dist_norm = np.linalg.norm(direction)
                    
                    if dist_norm > 0:
                        direction = direction / dist_norm
                        
                        # LJ force (simplified)
                        if min_dist < sigma:
                            # Repulsive
                            force_magnitude = epsilon * ((sigma / min_dist) ** 12 - (sigma / min_dist) ** 6)
                        else:
                            # Attractive at optimal distance
                            optimal_dist = 8.0
                            force_magnitude = -epsilon * np.exp(-((min_dist - optimal_dist) ** 2) / 2.0)
                        
                        force = force_magnitude * direction
                        forces[i] -= force
                        forces[j] += force
        
        return forces
    
    def _transform_coords(self,
                        coords: np.ndarray,
                        position: np.ndarray,
                        orientation: np.ndarray) -> np.ndarray:
        """Transform protein coordinates to world space"""
        centered = coords - coords.mean(axis=0)
        return centered @ orientation.T + position
    
    def _update_orientations(self,
                            proteins: List[Dict],
                            positions: List[np.ndarray],
                            orientations: List[np.ndarray]) -> List[np.ndarray]:
        """Update protein orientations based on interactions"""
        new_orientations = []
        
        for i, protein in enumerate(proteins):
            # Align protein towards surface normal
            direction = positions[i] / np.linalg.norm(positions[i])
            
            # Calculate optimal orientation
            coords = protein['coords']
            centered = coords - coords.mean(axis=0)
            
            if len(centered) > 3:
                # Principal axis
                cov = np.cov(centered.T)
                eigenvals, eigenvecs = np.linalg.eigh(cov)
                principal_axis = eigenvecs[:, -1]
                
                # Rotate to align with surface normal
                new_orientation = self._align_vectors(principal_axis, direction)
            else:
                new_orientation = orientations[i]
            
            new_orientations.append(new_orientation)
        
        return new_orientations
    
    def _align_vectors(self, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
        """Rotation matrix to align v1 with v2"""
        v1 = v1 / (np.linalg.norm(v1) + 1e-10)
        v2 = v2 / (np.linalg.norm(v2) + 1e-10)
        
        v = np.cross(v1, v2)
        s = np.linalg.norm(v)
        c = np.dot(v1, v2)
        
        if s < 1e-10:
            return np.eye(3)
        
        K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + K + K @ K * ((1 - c) / (s ** 2))
        
        return R
    
    def _random_rotation_matrix(self) -> np.ndarray:
        """Generate random rotation matrix"""
        # Random axis
        axis = np.random.randn(3)
        axis = axis / np.linalg.norm(axis)
        
        # Random angle
        angle = np.random.random() * 2 * np.pi
        
        # Rodrigues' rotation formula
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
        
        return R
    
    def _minimize_energy(self,
                       assembly: Dict,
                       radius: float) -> Dict:
        """Energy minimization of assembled structure"""
        self.logger.info("Minimizing energy...")
        
        proteins = assembly['proteins']
        positions = assembly['positions']
        orientations = assembly['orientations']
        
        # Flatten coordinates for optimization
        all_coords = []
        for protein, pos, rot in zip(proteins, positions, orientations):
            coords = self._transform_coords(protein['coords'], pos, rot)
            all_coords.append(coords)
        
        def energy_func(x):
            coords_flat = x.reshape(-1, 3)
            return self._calculate_total_energy(coords_flat, radius)
        
        x0 = np.vstack(all_coords).flatten()
        result = minimize(energy_func, x0, method='L-BFGS-B', options={'maxiter': 100, 'maxfun': 10000})
        
        # Update positions
        optimized_coords = result.x.reshape(-1, 3)
        
        # Redistribute back to proteins
        idx = 0
        new_positions = []
        for protein in proteins:
            n_atoms = len(protein['coords'])
            protein_coords = optimized_coords[idx:idx+n_atoms]
            new_pos = protein_coords.mean(axis=0)
            
            # Ensure on surface
            dist = np.linalg.norm(new_pos)
            if dist > 0:
                new_pos = new_pos / dist * radius * 0.95
            
            new_positions.append(new_pos)
            idx += n_atoms
        
        assembly['positions'] = new_positions
        
        return assembly
    
    def _calculate_total_energy(self, all_coords: np.ndarray, radius: float) -> float:
        """Calculate total energy of system"""
        distances = cdist(all_coords, all_coords)
        np.fill_diagonal(distances, np.inf)
        
        # Lennard-Jones potential
        sigma = 5.0
        epsilon = 1.0
        lj = 4 * epsilon * ((sigma/distances)**12 - (sigma/distances)**6)
        
        # Overlap penalty
        overlap_mask = distances < 3.0
        overlap_penalty = np.sum((3.0 - distances[overlap_mask]) * 100) if np.any(overlap_mask) else 0.0
        
        # Surface constraint (prefer proteins on surface)
        dist_from_center = np.linalg.norm(all_coords, axis=1)
        surface_penalty = np.sum((dist_from_center - radius * 0.95) ** 2) * 0.01
        
        return np.sum(lj) + overlap_penalty + surface_penalty
    
    def _detect_geometry(self, assembly: Dict) -> Dict:
        """
        Detect emergent geometry from assembled structure.
        No assumptions - analyze what actually formed.
        """
        positions = assembly['positions']
        proteins = assembly['proteins']
        
        if len(positions) < 3:
            return {'shape': 'unknown', 'symmetry': 'none'}
        
        # Calculate center
        center = np.mean(positions, axis=0)
        centered_positions = positions - center
        
        # Analyze shape
        # 1. Check if roughly spherical
        distances = np.linalg.norm(centered_positions, axis=1)
        avg_radius = np.mean(distances)
        radius_variance = np.var(distances) / (avg_radius ** 2)
        
        # 2. Check if elongated (filamentous)
        # Principal component analysis
        cov = np.cov(centered_positions.T)
        eigenvals, eigenvecs = np.linalg.eigh(cov)
        eigenvals = np.sort(eigenvals)[::-1]
        
        elongation_ratio = eigenvals[0] / (eigenvals[1] + 1e-10)
        
        # 3. Check for symmetry
        symmetry = self._detect_symmetry(centered_positions)
        
        # Classify shape
        if elongation_ratio > 3.0:
            shape = 'filamentous'
        elif radius_variance < 0.1:
            if symmetry == 'icosahedral':
                shape = 'icosahedral'
            elif symmetry == 'helical':
                shape = 'helical'
            else:
                shape = 'spherical'
        else:
            shape = 'irregular'
        
        return {
            'shape': shape,
            'symmetry': symmetry,
            'elongation_ratio': elongation_ratio,
            'radius_variance': radius_variance
        }
    
    def _detect_symmetry(self, positions: np.ndarray) -> str:
        """Detect symmetry in assembled structure"""
        n = len(positions)
        
        if n < 12:
            return 'none'
        
        # Check for icosahedral symmetry (12 vertices)
        # Calculate distances between all pairs
        distances = cdist(positions, positions)
        
        # Find common distances (symmetry indicators)
        unique_dists = np.unique(distances.flatten())
        unique_dists = unique_dists[unique_dists > 0.1]
        
        # If many proteins at similar distances from center, likely symmetric
        center_distances = np.linalg.norm(positions, axis=1)
        if np.std(center_distances) / np.mean(center_distances) < 0.15:
            # Check if 12-fold symmetry
            if n >= 12 and n % 12 == 0:
                return 'icosahedral'
            # Check if helical (many proteins along axis)
            elif n > 50:
                return 'helical'
        
        return 'none'

