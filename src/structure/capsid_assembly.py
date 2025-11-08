"""
Advanced Virus Capsid Assembly Simulator
Simulates realistic virus assembly using physics-based models
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
import itertools

class VirusAssemblySimulator:
    """Simulate realistic virus capsid assembly"""
    
    def __init__(self):
        self.logger = logging.getLogger('VSim.VirusAssembly')
    
    def assemble_capsid(self, 
                       protein_structures: List[Dict],
                       target_diameter_nm: float,
                       capsid_shape: str) -> Dict:
        """
        Simulate capsid assembly process
        
        Returns:
            Dict with assembled virus structure
        """
        self.logger.info(f"Simulating capsid assembly for {len(protein_structures)} proteins...")
        
        radius_angstrom = (target_diameter_nm / 2.0) * 10
        
        # Step 1: Fold individual proteins
        folded_proteins = []
        for i, protein in enumerate(protein_structures):
            self.logger.info(f"Folding protein {i+1}/{len(protein_structures)}...")
            sequence = protein.get('sequence', '')
            if sequence:
                from src.structure.protein_folding import ProteinFoldingSimulator
                folder = ProteinFoldingSimulator()
                folded_coords = folder.fold_protein(sequence)
                folded_proteins.append({
                    'coords': folded_coords,
                    'sequence': sequence,
                    'protein_id': protein.get('protein_id', i)
                })
        
        # Step 2: Determine assembly pathway
        assembly_pathway = self._determine_assembly_pathway(folded_proteins, capsid_shape)
        
        # Step 3: Simulate assembly process
        assembled_capsid = self._simulate_assembly(folded_proteins, assembly_pathway, radius_angstrom)
        
        # Step 4: Energy minimization of complete capsid
        optimized_capsid = self._minimize_capsid_energy(assembled_capsid)
        
        self.logger.info("Capsid assembly simulation complete!")
        
        return optimized_capsid
    
    def _determine_assembly_pathway(self, proteins: List[Dict], shape: str) -> Dict:
        """Determine optimal assembly pathway"""
        # Classify proteins
        structural = []
        spike = []
        internal = []
        
        for protein in proteins:
            length = len(protein.get('sequence', ''))
            if length > 1000:
                spike.append(protein)
            elif length > 200:
                structural.append(protein)
            else:
                internal.append(protein)
        
        return {
            'structural': structural,
            'spike': spike,
            'internal': internal,
            'shape': shape
        }
    
    def _simulate_assembly(self, proteins: List[Dict], pathway: Dict, radius: float) -> Dict:
        """Simulate step-by-step assembly"""
        self.logger.info("Simulating assembly process...")
        
        # Start with nucleation
        nucleated = self._nucleate_capsid(pathway['structural'], radius)
        
        # Add proteins one by one
        assembled = self._grow_capsid(nucleated, pathway['structural'], radius)
        
        # Add spike proteins
        if pathway['spike']:
            assembled = self._add_spike_proteins(assembled, pathway['spike'], radius)
        
        # Add internal proteins
        if pathway['internal']:
            assembled = self._add_internal_proteins(assembled, pathway['internal'], radius)
        
        return assembled
    
    def _nucleate_capsid(self, proteins: List[Dict], radius: float) -> Dict:
        """Form initial nucleation complex"""
        if not proteins:
            return {'proteins': [], 'positions': [], 'orientations': [], 'added': 0}
        
        # Start with 3-5 proteins forming initial complex
        n_nucleus = min(5, len(proteins))
        nucleus_proteins = proteins[:n_nucleus]
        
        positions = []
        orientations = []
        
        # Place first protein at origin
        positions.append(np.array([radius, 0, 0]))
        orientations.append(np.eye(3))
        
        # Add proteins around it to form initial complex
        for i in range(1, n_nucleus):
            # Use spherical distribution for speed
            angle = 2 * np.pi * i / n_nucleus
            theta = np.pi / 2
            pos = np.array([
                radius * np.sin(theta) * np.cos(angle),
                radius * np.sin(theta) * np.sin(angle),
                radius * np.cos(theta)
            ])
            positions.append(pos)
            orientations.append(np.eye(3))
        
        return {
            'proteins': nucleus_proteins,
            'positions': positions,
            'orientations': orientations,
            'added': n_nucleus
        }
    
    def _grow_capsid(self, nucleus: Dict, all_proteins: List[Dict], radius: float) -> Dict:
        """Grow capsid by adding proteins"""
        added = nucleus['added']
        positions = list(nucleus['positions'])
        orientations = list(nucleus['orientations'])
        added_proteins = list(nucleus['proteins'])
        
        remaining = all_proteins[added:]
        
        # Use spherical distribution for remaining proteins
        remaining_count = len(remaining)
        if remaining_count > 0:
            remaining_positions = self._generate_spherical_positions(radius, remaining_count)
            
            for protein, pos in zip(remaining, remaining_positions):
                # Orient protein outward
                direction = pos / np.linalg.norm(pos)
                rot_matrix = self._align_to_axis(protein['coords'], direction)
                
                positions.append(pos)
                orientations.append(rot_matrix)
                added_proteins.append(protein)
                added += 1
                
                if added % 10 == 0:
                    self.logger.info(f"Assembled {added}/{len(all_proteins)} proteins...")
        
        return {
            'proteins': added_proteins,
            'positions': positions,
            'orientations': orientations
        }
    
    def _find_optimal_docking(self, protein1_coords: np.ndarray, 
                             protein2_coords: np.ndarray,
                             protein1_pos: np.ndarray,
                             target_radius: float) -> Tuple[np.ndarray, np.ndarray]:
        """Find optimal docking position and orientation"""
        # Calculate protein sizes
        size1 = np.max(np.linalg.norm(protein1_coords - protein1_coords.mean(axis=0), axis=1))
        size2 = np.max(np.linalg.norm(protein2_coords - protein2_coords.mean(axis=0), axis=1))
        
        # Try multiple orientations
        best_score = -np.inf
        best_pos = None
        best_rot = None
        
        # Generate candidate positions around protein1
        for angle in np.linspace(0, 2*np.pi, 12):
            for theta in np.linspace(0, np.pi, 6):
                # Candidate position
                offset = np.array([
                    np.sin(theta) * np.cos(angle),
                    np.sin(theta) * np.sin(angle),
                    np.cos(theta)
                ]) * (size1 + size2 + 5.0)  # Close contact distance
                
                candidate_pos = protein1_pos + offset
                
                # Normalize to target radius
                candidate_pos = candidate_pos / np.linalg.norm(candidate_pos) * target_radius
                
                # Try different orientations
                for rot_angle in np.linspace(0, 2*np.pi, 8):
                    rot_matrix = self._rotation_matrix(rot_angle, offset)
                    
                    # Calculate interaction energy
                    score = self._calculate_interaction_energy(
                        protein1_coords,
                        protein2_coords,
                        protein1_pos,
                        candidate_pos,
                        np.eye(3),
                        rot_matrix
                    )
                    
                    if score > best_score:
                        best_score = score
                        best_pos = candidate_pos
                        best_rot = rot_matrix
        
        return best_pos, best_rot
    
    def _find_best_docking_site(self, existing_proteins: List[Dict],
                                new_protein: Dict,
                                existing_positions: List[np.ndarray],
                                existing_orientations: List[np.ndarray],
                                target_radius: float) -> Tuple[np.ndarray, np.ndarray, float]:
        """Find best site for new protein"""
        best_score = -np.inf
        best_pos = None
        best_rot = None
        
        # Try docking to each existing protein
        for i, existing_protein in enumerate(existing_proteins):
            pos, rot = self._find_optimal_docking(
                existing_protein['coords'],
                new_protein['coords'],
                existing_positions[i],
                target_radius
            )
            
            # Check for clashes with other proteins
            clash_score = self._check_clashes(
                new_protein['coords'],
                pos,
                existing_proteins,
                existing_positions,
                existing_orientations
            )
            
            score = -clash_score  # Lower clash = better
            
            if score > best_score:
                best_score = score
                best_pos = pos
                best_rot = rot
        
        return best_pos, best_rot, best_score
    
    def _calculate_interaction_energy(self, coords1: np.ndarray, coords2: np.ndarray,
                                     pos1: np.ndarray, pos2: np.ndarray,
                                     rot1: np.ndarray, rot2: np.ndarray) -> float:
        """Calculate interaction energy between two proteins"""
        # Transform coordinates
        coords1_transformed = (coords1 - coords1.mean(axis=0)) @ rot1.T + pos1
        coords2_transformed = (coords2 - coords2.mean(axis=0)) @ rot2.T + pos2
        
        # Calculate distances
        distances = cdist(coords1_transformed, coords2_transformed)
        
        # Lennard-Jones potential
        sigma = 4.0
        epsilon = 0.5
        lj = 4 * epsilon * ((sigma/distances)**12 - (sigma/distances)**6)
        
        # Prefer optimal contact distance (5-8 A)
        optimal_dist = 6.0
        contact_energy = -np.exp(-((distances - optimal_dist)**2) / 2.0)
        
        total_energy = np.sum(lj) + np.sum(contact_energy)
        
        return total_energy
    
    def _check_clashes(self, new_coords: np.ndarray, new_pos: np.ndarray,
                      existing_proteins: List[Dict], existing_positions: List[np.ndarray],
                      existing_orientations: List[np.ndarray]) -> float:
        """Check for clashes with existing proteins"""
        clash_score = 0.0
        
        new_coords_transformed = new_coords - new_coords.mean(axis=0) + new_pos
        
        for i, existing_protein in enumerate(existing_proteins):
            existing_coords = existing_protein['coords']
            existing_coords_transformed = (existing_coords - existing_coords.mean(axis=0)) @ existing_orientations[i].T + existing_positions[i]
            
            distances = cdist(new_coords_transformed, existing_coords_transformed)
            min_dist = np.min(distances)
            
            if min_dist < 3.0:  # Clash threshold
                clash_score += (3.0 - min_dist) * 100  # Penalty for clashes
        
        return clash_score
    
    def _rotation_matrix(self, angle: float, axis: np.ndarray) -> np.ndarray:
        """Generate rotation matrix"""
        axis = axis / (np.linalg.norm(axis) + 1e-10)
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        K = np.array([[0, -axis[2], axis[1]],
                     [axis[2], 0, -axis[0]],
                     [-axis[1], axis[0], 0]])
        R = np.eye(3) + sin_a * K + (1 - cos_a) * np.dot(K, K)
        return R
    
    def _add_spike_proteins(self, capsid: Dict, spike_proteins: List[Dict], radius: float) -> Dict:
        """Add spike proteins to capsid"""
        self.logger.info(f"Adding {len(spike_proteins)} spike proteins...")
        
        # Generate spike positions
        spike_positions = self._generate_spike_positions(radius, len(spike_proteins))
        
        for spike_protein, spike_pos in zip(spike_proteins, spike_positions):
            # Orient spike outward
            direction = spike_pos / np.linalg.norm(spike_pos)
            rot_matrix = self._align_to_axis(spike_protein['coords'], direction)
            
            capsid['proteins'].append(spike_protein)
            capsid['positions'].append(spike_pos)
            capsid['orientations'].append(rot_matrix)
        
        return capsid
    
    def _add_internal_proteins(self, capsid: Dict, internal_proteins: List[Dict], radius: float) -> Dict:
        """Add internal proteins"""
        self.logger.info(f"Adding {len(internal_proteins)} internal proteins...")
        
        internal_radius = radius * 0.6
        
        for internal_protein in internal_proteins:
            # Random position inside
            phi = np.random.random() * 2 * np.pi
            theta = np.arccos(2 * np.random.random() - 1)
            pos = np.array([
                internal_radius * np.sin(theta) * np.cos(phi),
                internal_radius * np.sin(theta) * np.sin(phi),
                internal_radius * np.cos(theta)
            ])
            
            capsid['proteins'].append(internal_protein)
            capsid['positions'].append(pos)
            capsid['orientations'].append(np.eye(3))
        
        return capsid
    
    def _generate_spike_positions(self, radius: float, n_spikes: int) -> List[np.ndarray]:
        """Generate positions for spike proteins"""
        positions = []
        
        # Use icosahedral symmetry for spike placement
        phi = (1 + np.sqrt(5)) / 2
        vertices = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
        ])
        vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True) * radius
        
        positions.extend(vertices[:min(12, n_spikes)])
        
        # Add more if needed
        if n_spikes > 12:
            for i in range(12, n_spikes):
                v1 = vertices[i % 12]
                v2 = vertices[(i + 1) % 12]
                v3 = vertices[(i + 2) % 12]
                center = (v1 + v2 + v3) / 3
                center = center / np.linalg.norm(center) * radius
                positions.append(center)
        
        return positions[:n_spikes]
    
    def _generate_spherical_positions(self, radius: float, num_proteins: int) -> List[np.ndarray]:
        """Generate evenly distributed spherical positions using Fibonacci sphere"""
        positions = []
        for i in range(num_proteins):
            theta = np.pi * i / num_proteins if num_proteins > 1 else 0
            phi = 2 * np.pi * i * 0.618033988749895  # Golden angle
            x = radius * np.sin(theta) * np.cos(phi)
            y = radius * np.sin(theta) * np.sin(phi)
            z = radius * np.cos(theta)
            positions.append(np.array([x, y, z]))
        return positions
    
    def _align_to_axis(self, coords: np.ndarray, axis: np.ndarray) -> np.ndarray:
        """Align protein's principal axis to given axis"""
        # Get principal axis
        centered = coords - coords.mean(axis=0)
        cov = np.cov(centered.T)
        eigenvals, eigenvecs = np.linalg.eigh(cov)
        principal_axis = eigenvecs[:, -1]
        
        # Rotate to align
        v = np.cross(principal_axis, axis)
        if np.linalg.norm(v) > 1e-10:
            v = v / np.linalg.norm(v)
            angle = np.arccos(np.clip(np.dot(principal_axis, axis), -1, 1))
            K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
            return R
        return np.eye(3)
    
    def _minimize_capsid_energy(self, capsid: Dict) -> Dict:
        """Energy minimization of complete capsid"""
        self.logger.info("Minimizing capsid energy...")
        
        # Flatten coordinates
        all_coords = []
        for protein, pos, rot in zip(capsid['proteins'], capsid['positions'], capsid['orientations']):
            coords_transformed = (protein['coords'] - protein['coords'].mean(axis=0)) @ rot.T + pos
            all_coords.append(coords_transformed)
        
        # Minimize
        def energy_func(x):
            coords_flat = x.reshape(-1, 3)
            return self._calculate_capsid_energy(coords_flat, capsid)
        
        x0 = np.vstack(all_coords).flatten()
        result = minimize(energy_func, x0, method='L-BFGS-B', options={'maxiter': 50})
        
        # Update positions
        optimized_coords = result.x.reshape(-1, 3)
        
        # Redistribute back to proteins
        idx = 0
        new_positions = []
        for protein in capsid['proteins']:
            n_atoms = len(protein['coords'])
            protein_coords = optimized_coords[idx:idx+n_atoms]
            new_pos = protein_coords.mean(axis=0)
            new_positions.append(new_pos)
            idx += n_atoms
        
        capsid['positions'] = new_positions
        
        return capsid
    
    def _calculate_capsid_energy(self, all_coords: np.ndarray, capsid: Dict) -> float:
        """Calculate total capsid energy"""
        distances = cdist(all_coords, all_coords)
        np.fill_diagonal(distances, np.inf)
        
        # Lennard-Jones potential
        sigma = 4.0
        epsilon = 0.5
        lj = 4 * epsilon * ((sigma/distances)**12 - (sigma/distances)**6)
        
        # Avoid overlaps
        overlap_mask = distances < 3.0
        if np.any(overlap_mask):
            overlap_penalty = np.sum((3.0 - distances[overlap_mask]) * 100)
        else:
            overlap_penalty = 0.0
        
        return np.sum(lj) + overlap_penalty

