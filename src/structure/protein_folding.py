"""
Advanced Protein Folding Simulator
Uses molecular dynamics principles to fold proteins realistically
"""

import logging
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path
from scipy.spatial.distance import cdist
from scipy.optimize import minimize

class ProteinFoldingSimulator:
    """Simulate protein folding using molecular dynamics principles"""
    
    # Amino acid properties
    AA_PROPERTIES = {
        'A': {'hydrophobic': True, 'size': 1.0, 'charge': 0},
        'R': {'hydrophobic': False, 'size': 2.0, 'charge': 1},
        'N': {'hydrophobic': False, 'size': 1.5, 'charge': 0},
        'D': {'hydrophobic': False, 'size': 1.5, 'charge': -1},
        'C': {'hydrophobic': True, 'size': 1.2, 'charge': 0},
        'Q': {'hydrophobic': False, 'size': 1.8, 'charge': 0},
        'E': {'hydrophobic': False, 'size': 1.8, 'charge': -1},
        'G': {'hydrophobic': False, 'size': 0.5, 'charge': 0},
        'H': {'hydrophobic': False, 'size': 2.0, 'charge': 1},
        'I': {'hydrophobic': True, 'size': 1.8, 'charge': 0},
        'L': {'hydrophobic': True, 'size': 1.8, 'charge': 0},
        'K': {'hydrophobic': False, 'size': 2.0, 'charge': 1},
        'M': {'hydrophobic': True, 'size': 1.8, 'charge': 0},
        'F': {'hydrophobic': True, 'size': 2.2, 'charge': 0},
        'P': {'hydrophobic': True, 'size': 1.5, 'charge': 0},
        'S': {'hydrophobic': False, 'size': 1.2, 'charge': 0},
        'T': {'hydrophobic': False, 'size': 1.5, 'charge': 0},
        'W': {'hydrophobic': True, 'size': 2.5, 'charge': 0},
        'Y': {'hydrophobic': True, 'size': 2.2, 'charge': 0},
        'V': {'hydrophobic': True, 'size': 1.5, 'charge': 0},
    }
    
    def __init__(self):
        self.logger = logging.getLogger('VSim.ProteinFolding')
    
    def fold_protein(self, sequence: str, max_iterations: int = 500) -> np.ndarray:
        """
        Fold protein using simplified molecular dynamics
        
        Returns: Nx3 array of CA atom positions
        """
        self.logger.info(f"Folding protein of length {len(sequence)}...")
        
        n_residues = len(sequence)
        
        # Initialize with extended chain
        coords = self._initialize_chain(n_residues)
        
        # Run MD simulation
        for iteration in range(max_iterations):
            # Calculate forces
            forces = self._calculate_forces(coords, sequence)
            
            # Update positions (Verlet integration)
            dt = 0.01
            coords = coords + forces * dt * dt
            
            # Apply constraints
            coords = self._apply_constraints(coords, sequence)
            
            if iteration % 100 == 0:
                energy = self._calculate_energy(coords, sequence)
                self.logger.debug(f"Iteration {iteration}: Energy = {energy:.2f}")
        
        # Energy minimization
        coords = self._minimize_energy(coords, sequence)
        
        self.logger.info(f"Folding complete. Final structure: {coords.shape}")
        return coords
    
    def _initialize_chain(self, n_residues: int) -> np.ndarray:
        """Initialize protein chain in extended conformation"""
        coords = np.zeros((n_residues, 3))
        
        # Create extended chain along z-axis
        for i in range(n_residues):
            coords[i] = [0, 0, i * 3.8]  # 3.8 A per residue in extended chain
        
        return coords
    
    def _calculate_forces(self, coords: np.ndarray, sequence: str) -> np.ndarray:
        """Calculate forces on each atom"""
        n = len(coords)
        forces = np.zeros_like(coords)
        
        # Bond forces (harmonic potential)
        for i in range(n - 1):
            bond_vec = coords[i+1] - coords[i]
            bond_length = np.linalg.norm(bond_vec)
            target_length = 3.8  # CA-CA distance
            force_magnitude = (bond_length - target_length) * 100  # Spring constant
            direction = bond_vec / (bond_length + 1e-10)
            forces[i] += force_magnitude * direction
            forces[i+1] -= force_magnitude * direction
        
        # Angle forces (maintain protein geometry)
        for i in range(n - 2):
            vec1 = coords[i+1] - coords[i]
            vec2 = coords[i+2] - coords[i+1]
            cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-10)
            target_angle = np.cos(np.radians(110))  # CA-CA-CA angle
            angle_error = cos_angle - target_angle
            # Apply restoring force
            perp = np.cross(vec1, vec2)
            if np.linalg.norm(perp) > 1e-10:
                perp = perp / np.linalg.norm(perp)
                forces[i] += perp * angle_error * 50
                forces[i+2] += perp * angle_error * 50
        
        # Non-bonded interactions (Lennard-Jones + electrostatics)
        distances = cdist(coords, coords)
        np.fill_diagonal(distances, np.inf)  # Exclude self
        
        for i in range(n):
            for j in range(i + 1, n):
                dist = distances[i, j]
                if dist < 20:  # Cutoff
                    # Lennard-Jones potential
                    sigma = 4.0  # Van der Waals radius
                    epsilon = 0.5
                    lj_force = 24 * epsilon * (2 * (sigma/dist)**12 - (sigma/dist)**6) / dist
                    
                    # Electrostatic (charge-charge)
                    aa_i = sequence[i] if i < len(sequence) else 'A'
                    aa_j = sequence[j] if j < len(sequence) else 'A'
                    charge_i = self.AA_PROPERTIES.get(aa_i, {}).get('charge', 0)
                    charge_j = self.AA_PROPERTIES.get(aa_j, {}).get('charge', 0)
                    electrostatic = charge_i * charge_j * 332 / (dist * dist)  # kcal/mol/A
                    
                    # Hydrophobic interactions
                    hydro_i = self.AA_PROPERTIES.get(aa_i, {}).get('hydrophobic', False)
                    hydro_j = self.AA_PROPERTIES.get(aa_j, {}).get('hydrophobic', False)
                    if hydro_i and hydro_j:
                        hydrophobic = -0.5 * np.exp(-dist/5.0)  # Attractive
                    else:
                        hydrophobic = 0
                    
                    # Total force
                    force_magnitude = lj_force + electrostatic + hydrophobic
                    direction = (coords[j] - coords[i]) / dist
                    forces[i] -= force_magnitude * direction
                    forces[j] += force_magnitude * direction
        
        return forces
    
    def _apply_constraints(self, coords: np.ndarray, sequence: str) -> np.ndarray:
        """Apply geometric constraints"""
        n = len(coords)
        
        # Maintain bond lengths
        for i in range(n - 1):
            bond_vec = coords[i+1] - coords[i]
            bond_length = np.linalg.norm(bond_vec)
            if bond_length > 0:
                coords[i+1] = coords[i] + bond_vec / bond_length * 3.8
        
        return coords
    
    def _calculate_energy(self, coords: np.ndarray, sequence: str) -> float:
        """Calculate total energy"""
        n = len(coords)
        energy = 0.0
        
        # Bond energy
        for i in range(n - 1):
            bond_length = np.linalg.norm(coords[i+1] - coords[i])
            energy += 0.5 * 100 * (bond_length - 3.8)**2
        
        # Non-bonded energy
        distances = cdist(coords, coords)
        np.fill_diagonal(distances, np.inf)
        
        for i in range(n):
            for j in range(i + 1, n):
                dist = distances[i, j]
                if dist < 20:
                    sigma = 4.0
                    epsilon = 0.5
                    lj = 4 * epsilon * ((sigma/dist)**12 - (sigma/dist)**6)
                    
                    aa_i = sequence[i] if i < len(sequence) else 'A'
                    aa_j = sequence[j] if j < len(sequence) else 'A'
                    charge_i = self.AA_PROPERTIES.get(aa_i, {}).get('charge', 0)
                    charge_j = self.AA_PROPERTIES.get(aa_j, {}).get('charge', 0)
                    elec = charge_i * charge_j * 332 / dist
                    
                    hydro_i = self.AA_PROPERTIES.get(aa_i, {}).get('hydrophobic', False)
                    hydro_j = self.AA_PROPERTIES.get(aa_j, {}).get('hydrophobic', False)
                    if hydro_i and hydro_j:
                        hydro = -0.5 * np.exp(-dist/5.0)
                    else:
                        hydro = 0
                    
                    energy += lj + elec + hydro
        
        return energy
    
    def _minimize_energy(self, coords: np.ndarray, sequence: str) -> np.ndarray:
        """Energy minimization using scipy"""
        def energy_func(x):
            coords_flat = x.reshape(-1, 3)
            return self._calculate_energy(coords_flat, sequence)
        
        x0 = coords.flatten()
        result = minimize(energy_func, x0, method='L-BFGS-B', options={'maxiter': 100})
        
        return result.x.reshape(-1, 3)

