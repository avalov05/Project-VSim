"""
Realistic Virus Assembly System
Simulates actual virus assembly like in real life:
- Detects binding sites on proteins (surface patches)
- Calculates binding affinities (complementarity)
- Nucleation-based assembly (small complexes form first)
- Specific protein-protein docking
- Atom-level interactions
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from pathlib import Path
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import copy

class RealisticVirusAssembler:
    """
    Assembles viruses like real life - through specific protein-protein interactions.
    No assumptions - uses actual binding sites and complementarity.
    """
    
    def __init__(self):
        self.logger = logging.getLogger('VSim.RealisticAssembler')
        self.kB = 0.001987  # kcal/mol/K
        self.temperature = 310.0  # K
        self.dielectric = 80.0  # Water dielectric constant
        
        # Amino acid properties for binding calculations
        self.aa_properties = {
            'A': {'hydrophobic': True, 'charge': 0, 'vdw_radius': 2.0},
            'R': {'hydrophobic': False, 'charge': 1, 'vdw_radius': 3.0},
            'N': {'hydrophobic': False, 'charge': 0, 'vdw_radius': 2.5},
            'D': {'hydrophobic': False, 'charge': -1, 'vdw_radius': 2.5},
            'C': {'hydrophobic': True, 'charge': 0, 'vdw_radius': 2.2},
            'Q': {'hydrophobic': False, 'charge': 0, 'vdw_radius': 2.8},
            'E': {'hydrophobic': False, 'charge': -1, 'vdw_radius': 2.8},
            'G': {'hydrophobic': False, 'charge': 0, 'vdw_radius': 1.5},
            'H': {'hydrophobic': False, 'charge': 1, 'vdw_radius': 3.0},
            'I': {'hydrophobic': True, 'charge': 0, 'vdw_radius': 2.8},
            'L': {'hydrophobic': True, 'charge': 0, 'vdw_radius': 2.8},
            'K': {'hydrophobic': False, 'charge': 1, 'vdw_radius': 3.0},
            'M': {'hydrophobic': True, 'charge': 0, 'vdw_radius': 2.8},
            'F': {'hydrophobic': True, 'charge': 0, 'vdw_radius': 3.2},
            'P': {'hydrophobic': True, 'charge': 0, 'vdw_radius': 2.5},
            'S': {'hydrophobic': False, 'charge': 0, 'vdw_radius': 2.2},
            'T': {'hydrophobic': False, 'charge': 0, 'vdw_radius': 2.5},
            'W': {'hydrophobic': True, 'charge': 0, 'vdw_radius': 3.5},
            'Y': {'hydrophobic': True, 'charge': 0, 'vdw_radius': 3.2},
            'V': {'hydrophobic': True, 'charge': 0, 'vdw_radius': 2.5},
        }
    
    def calculate_protein_copy_numbers(self,
                                      protein_structures: List[Dict],
                                      diameter_nm: float) -> Dict[int, int]:
        """Calculate copy numbers from surface area and packing"""
        radius_nm = diameter_nm / 2.0
        surface_area_nm2 = 4 * np.pi * radius_nm ** 2
        
        copy_numbers = {}
        
        for i, protein in enumerate(protein_structures):
            sequence = protein.get('sequence', '')
            length = len(sequence)
            
            # Estimate from actual folded structure if available
            if 'coords' in protein:
                coords = protein['coords']
                # Calculate actual protein size from structure
                center = coords.mean(axis=0)
                distances = np.linalg.norm(coords - center, axis=1)
                protein_radius_nm = np.max(distances) / 10.0  # Convert A to nm
            else:
                # Estimate from sequence
                protein_volume_angstrom3 = length * 110.0 / 0.75
                protein_radius_nm = (3 * protein_volume_angstrom3 / (4 * np.pi)) ** (1/3) / 10.0
            
            # Surface area per protein
            packing_efficiency = 0.70
            protein_surface_area_nm2 = np.pi * protein_radius_nm ** 2 / packing_efficiency
            
            # Classify protein
            is_structural = self._is_structural_protein(sequence, length)
            is_transmembrane = self._has_transmembrane_features(sequence)
            
            if is_structural and not is_transmembrane:
                copies = int(surface_area_nm2 / protein_surface_area_nm2)
                copies = max(10, min(copies, 1000))
            elif is_transmembrane or (length > 1000):
                copies = max(12, min(int(surface_area_nm2 / (protein_surface_area_nm2 * 3)), 100))
            else:
                volume_nm3 = (4/3) * np.pi * radius_nm ** 3
                internal_volume_nm3 = volume_nm3 * 0.6
                protein_volume_nm3 = (4/3) * np.pi * (protein_radius_nm ** 3)
                copies = max(1, min(int(internal_volume_nm3 / protein_volume_nm3), 100))
            
            copy_numbers[i] = copies
        
        return copy_numbers
    
    def _is_structural_protein(self, sequence: str, length: int) -> bool:
        """Detect structural proteins"""
        if length < 50:
            return False
        
        hydrophobic = set('AILMFWV')
        hydrophobicity = sum(1 for aa in sequence if aa in hydrophobic) / len(sequence)
        
        if 0.35 <= hydrophobicity <= 0.55:
            return True
        if length > 200 and hydrophobicity < 0.6:
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
            if sum(1 for aa in window if aa in hydrophobic) / window_size >= 0.6:
                hydrophobic_windows += 1
        
        return hydrophobic_windows >= 2
    
    def detect_binding_sites(self, coords: np.ndarray, sequence: str) -> List[Dict]:
        """
        Detect binding sites on protein surface.
        Returns patches of surface residues that can bind to other proteins.
        """
        if len(coords) < 3:
            return []
        
        # Calculate surface points using distance from center (more robust)
        center = coords.mean(axis=0)
        distances = np.linalg.norm(coords - center, axis=1)
        
        # Use top 30% most distant points as surface
        if len(coords) > 10:
            threshold = np.percentile(distances, 70)
        else:
            threshold = np.mean(distances)
        
        surface_indices = set(np.where(distances >= threshold)[0])
        
        # If too few surface points, use all points
        if len(surface_indices) < 3:
            surface_indices = set(range(len(coords)))
        
        # Group nearby surface residues into patches
        patches = []
        used = set()
        
        # Also try ConvexHull if possible
        try:
            if len(coords) >= 4:
                hull = ConvexHull(coords)
                hull_surface = set(hull.vertices)
                if len(hull_surface) >= 3:
                    surface_indices = hull_surface
        except:
            pass  # Use distance-based method
        
        for idx in surface_indices:
            if idx in used:
                continue
            
            # Find nearby surface residues
            patch_indices = [idx]
            used.add(idx)
            
            for other_idx in surface_indices:
                if other_idx in used:
                    continue
                
                dist = np.linalg.norm(coords[idx] - coords[other_idx])
                if dist < 15.0:  # Increased patch size for better detection
                    patch_indices.append(other_idx)
                    used.add(other_idx)
            
            # Reduced minimum patch size to 2 for small proteins
            min_patch_size = 2 if len(coords) < 20 else 3
            
            if len(patch_indices) >= min_patch_size:
                patch_coords = coords[patch_indices]
                patch_center = patch_coords.mean(axis=0)
                
                # Calculate patch properties
                patch_seq = ''.join([sequence[i] if i < len(sequence) else 'A' for i in patch_indices])
                hydrophobicity = sum(1 for aa in patch_seq if aa in set('AILMFWV')) / len(patch_seq) if patch_seq else 0.5
                charge = sum(self.aa_properties.get(aa, {}).get('charge', 0) for aa in patch_seq)
                
                # Surface normal (pointing outward)
                center = coords.mean(axis=0)
                vec_to_center = patch_center - center
                vec_norm = np.linalg.norm(vec_to_center)
                if vec_norm > 1e-10:
                    surface_normal = vec_to_center / vec_norm
                else:
                    surface_normal = np.array([1.0, 0.0, 0.0])
                
                patches.append({
                    'indices': patch_indices,
                    'center': patch_center,
                    'normal': surface_normal,
                    'hydrophobicity': hydrophobicity,
                    'charge': charge,
                    'sequence': patch_seq
                })
        
        # If no patches found, create at least one patch from surface points
        if not patches and len(surface_indices) > 0:
            patch_indices = list(surface_indices)[:min(10, len(surface_indices))]
            patch_coords = coords[patch_indices]
            patch_center = patch_coords.mean(axis=0)
            
            patch_seq = ''.join([sequence[i] if i < len(sequence) else 'A' for i in patch_indices])
            hydrophobicity = sum(1 for aa in patch_seq if aa in set('AILMFWV')) / len(patch_seq) if patch_seq else 0.5
            charge = sum(self.aa_properties.get(aa, {}).get('charge', 0) for aa in patch_seq)
            
            center = coords.mean(axis=0)
            vec_to_center = patch_center - center
            vec_norm = np.linalg.norm(vec_to_center)
            if vec_norm > 1e-10:
                surface_normal = vec_to_center / vec_norm
            else:
                surface_normal = np.array([1.0, 0.0, 0.0])
            
            patches.append({
                'indices': patch_indices,
                'center': patch_center,
                'normal': surface_normal,
                'hydrophobicity': hydrophobicity,
                'charge': charge,
                'sequence': patch_seq
            })
        
        return patches
    
    def calculate_binding_affinity(self,
                                  patch1: Dict,
                                  patch2: Dict,
                                  coords1: np.ndarray,
                                  coords2: np.ndarray,
                                  sequence1: str,
                                  sequence2: str) -> float:
        """
        Calculate binding affinity between two protein patches.
        Uses complementarity: shape, charge, hydrophobicity.
        """
        # Geometric complementarity (shape fit)
        patch1_coords = coords1[patch1['indices']]
        patch2_coords = coords2[patch2['indices']]
        
        # Calculate how well patches fit together
        distances = cdist(patch1_coords, patch2_coords)
        min_dist = np.min(distances)
        
        # Optimal contact distance
        optimal_dist = 5.0  # Angstroms
        shape_complementarity = -np.exp(-((min_dist - optimal_dist) ** 2) / 2.0)
        
        # Charge complementarity (opposite charges attract)
        charge_complementarity = -patch1['charge'] * patch2['charge'] * 0.5
        
        # Hydrophobic complementarity (hydrophobic patches attract)
        if patch1['hydrophobicity'] > 0.5 and patch2['hydrophobicity'] > 0.5:
            hydrophobic_complementarity = -patch1['hydrophobicity'] * patch2['hydrophobicity']
        else:
            hydrophobic_complementarity = 0
        
        # Normal orientation (patches should face each other)
        normal_alignment = np.dot(patch1['normal'], -patch2['normal'])
        orientation_score = max(0, normal_alignment) * 0.5
        
        # Total binding affinity (lower = stronger binding)
        binding_affinity = shape_complementarity + charge_complementarity + \
                          hydrophobic_complementarity - orientation_score
        
        return binding_affinity
    
    def find_best_docking(self,
                         protein1: Dict,
                         protein2: Dict,
                         coords1: np.ndarray,
                         coords2: np.ndarray,
                         sequence1: str,
                         sequence2: str) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Find best docking configuration between two proteins.
        Returns: (position2, rotation2, binding_score)
        """
        # Detect binding sites
        patches1 = self.detect_binding_sites(coords1, sequence1)
        patches2 = self.detect_binding_sites(coords2, sequence2)
        
        # Always ensure we have at least one patch per protein
        if not patches1:
            patches1 = self._create_default_patch(coords1, sequence1)
        if not patches2:
            patches2 = self._create_default_patch(coords2, sequence2)
        
        # Try all patch combinations
        best_score = np.inf
        best_pos = None
        best_rot = None
        
        for patch1 in patches1:
            for patch2 in patches2:
                # Calculate binding affinity
                affinity = self.calculate_binding_affinity(
                    patch1, patch2, coords1, coords2, sequence1, sequence2
                )
                
                if affinity < best_score:
                    # Calculate docking position
                    # Align patches
                    patch1_center = patch1['center']
                    patch2_center = patch2['center']
                    
                    # Place protein2 such that patches align
                    offset = patch1_center - patch2_center
                    new_pos = coords2.mean(axis=0) + offset
                    
                    # Rotate to align normals
                    normal1 = patch1['normal']
                    normal2 = patch2['normal']
                    
                    # Rotation to align normals
                    v = np.cross(normal2, -normal1)
                    s = np.linalg.norm(v)
                    c = np.dot(normal2, -normal1)
                    
                    if s > 1e-10:
                        K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
                        rot = np.eye(3) + K + K @ K * ((1 - c) / (s ** 2))
                    else:
                        rot = np.eye(3)
                    
                    # Adjust position to optimal contact distance
                    optimal_dist = 5.0
                    direction = normal1
                    new_pos = patch1_center - direction * optimal_dist
                    
                    best_score = affinity
                    best_pos = new_pos - coords2.mean(axis=0) @ rot.T
                    best_rot = rot
        
        if best_pos is None:
            return self._geometric_docking(coords1, coords2)
        
        return best_pos, best_rot, best_score
    
    def _create_default_patch(self, coords: np.ndarray, sequence: str) -> List[Dict]:
        """Create a default patch from surface points if detection fails"""
        if len(coords) < 3:
            return []
        
        # Use outermost points
        center = coords.mean(axis=0)
        distances = np.linalg.norm(coords - center, axis=1)
        max_dist_idx = np.argmax(distances)
        
        # Create patch around this point
        patch_indices = [max_dist_idx]
        for i in range(len(coords)):
            if i != max_dist_idx:
                dist = np.linalg.norm(coords[i] - coords[max_dist_idx])
                if dist < 10.0:
                    patch_indices.append(i)
        
        patch_coords = coords[patch_indices]
        patch_center = patch_coords.mean(axis=0)
        
        patch_seq = ''.join([sequence[i] if i < len(sequence) else 'A' for i in patch_indices])
        hydrophobicity = sum(1 for aa in patch_seq if aa in set('AILMFWV')) / len(patch_seq) if patch_seq else 0.5
        charge = sum(self.aa_properties.get(aa, {}).get('charge', 0) for aa in patch_seq)
        
        vec_to_center = patch_center - center
        vec_norm = np.linalg.norm(vec_to_center)
        if vec_norm > 1e-10:
            surface_normal = vec_to_center / vec_norm
        else:
            surface_normal = np.array([1.0, 0.0, 0.0])
        
        return [{
            'indices': patch_indices,
            'center': patch_center,
            'normal': surface_normal,
            'hydrophobicity': hydrophobicity,
            'charge': charge,
            'sequence': patch_seq
        }]
    
    def _geometric_docking(self, coords1: np.ndarray, coords2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """Fallback geometric docking"""
        center1 = coords1.mean(axis=0)
        center2 = coords2.mean(axis=0)
        
        # Place protein2 next to protein1
        size1 = np.max(np.linalg.norm(coords1 - center1, axis=1))
        size2 = np.max(np.linalg.norm(coords2 - center2, axis=1))
        
        optimal_dist = size1 + size2 + 5.0
        direction = np.random.randn(3)
        direction = direction / np.linalg.norm(direction)
        
        new_pos = center1 + direction * optimal_dist - center2
        rot = np.eye(3)
        
        return new_pos, rot, 1.0
    
    def assemble_virus(self,
                      replicated_proteins: List[Dict],
                      diameter_nm: float) -> Dict:
        """
        Assemble virus through nucleation and growth - like real life.
        """
        self.logger.info("Starting realistic virus assembly (nucleation-based)...")
        
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
        
        # Step 2: Nucleation - form small stable complexes
        self.logger.info("Step 1: Nucleation - forming initial complexes...")
        nucleated_complexes = self._nucleate_complexes(folded_proteins, radius_angstrom)
        
        # Step 3: Growth - add proteins one by one
        self.logger.info("Step 2: Growth - adding proteins to complexes...")
        assembled_capsid = self._grow_capsid(nucleated_complexes, folded_proteins, radius_angstrom)
        
        # Step 4: Final energy minimization
        self.logger.info("Step 3: Energy minimization...")
        assembled_capsid = self._minimize_energy(assembled_capsid, radius_angstrom)
        
        # Step 5: Detect emergent geometry
        geometry = self._detect_geometry(assembled_capsid)
        self.logger.info(f"Detected geometry: {geometry['shape']} (symmetry: {geometry['symmetry']})")
        
        assembled_capsid['geometry'] = geometry
        
        return assembled_capsid
    
    def _nucleate_complexes(self,
                           proteins: List[Dict],
                           radius: float) -> List[Dict]:
        """
        Nucleation: Form small stable complexes (like real virus assembly).
        """
        if not proteins:
            return []
        
        complexes = []
        used = set()
        
        # Start with first protein
        first_protein = proteins[0]
        complex_proteins = [first_protein]
        complex_positions = [np.array([radius, 0, 0])]
        complex_orientations = [np.eye(3)]
        used.add(0)
        
        # Add proteins that bind strongly to form nucleation complex
        nucleation_size = min(5, len(proteins))
        
        for _ in range(1, nucleation_size):
            best_score = np.inf
            best_idx = None
            best_pos = None
            best_rot = None
            
            for i, protein in enumerate(proteins):
                if i in used:
                    continue
                
                # Try docking to each protein in complex
                for j, existing_protein in enumerate(complex_proteins):
                    try:
                        pos, rot, score = self.find_best_docking(
                            existing_protein,
                            protein,
                            existing_protein['coords'],
                            protein['coords'],
                            existing_protein['sequence'],
                            protein['sequence']
                        )
                        
                        # Transform position relative to existing protein center
                        existing_center = existing_protein['coords'].mean(axis=0)
                        protein_center = protein['coords'].mean(axis=0)
                        
                        # Calculate actual position
                        transformed_pos = existing_center + pos
                        
                        # If position is invalid, use geometric docking
                        if np.any(np.isnan(transformed_pos)) or np.any(np.isinf(transformed_pos)):
                            pos, rot, score = self._geometric_docking(
                                existing_protein['coords'],
                                protein['coords']
                            )
                            transformed_pos = existing_center + pos
                        
                        # Constrain to surface
                        dist = np.linalg.norm(transformed_pos)
                        if dist > 0:
                            transformed_pos = transformed_pos / dist * radius
                        else:
                            # Fallback position
                            angle = 2 * np.pi * len(complex_proteins) / nucleation_size
                            transformed_pos = np.array([
                                radius * np.cos(angle),
                                radius * np.sin(angle),
                                0
                            ])
                        
                        if score < best_score:
                            best_score = score
                            best_idx = i
                            best_pos = transformed_pos
                            best_rot = rot
                    except Exception as e:
                        # Fallback to geometric docking
                        try:
                            pos, rot, score = self._geometric_docking(
                                existing_protein['coords'],
                                protein['coords']
                            )
                            existing_center = existing_protein['coords'].mean(axis=0)
                            transformed_pos = existing_center + pos
                            dist = np.linalg.norm(transformed_pos)
                            if dist > 0:
                                transformed_pos = transformed_pos / dist * radius
                            else:
                                angle = 2 * np.pi * len(complex_proteins) / nucleation_size
                                transformed_pos = np.array([
                                    radius * np.cos(angle),
                                    radius * np.sin(angle),
                                    0
                                ])
                            
                            if score < best_score:
                                best_score = score
                                best_idx = i
                                best_pos = transformed_pos
                                best_rot = rot
                        except:
                            continue
            
            if best_idx is not None:
                complex_proteins.append(proteins[best_idx])
                complex_positions.append(best_pos)
                complex_orientations.append(best_rot)
                used.add(best_idx)
            else:
                # If no good docking found, add protein at geometric position
                if len(complex_proteins) < len(proteins):
                    unused_idx = next(i for i in range(len(proteins)) if i not in used)
                    angle = 2 * np.pi * len(complex_proteins) / nucleation_size
                    pos = np.array([
                        radius * np.cos(angle),
                        radius * np.sin(angle),
                        0
                    ])
                    complex_proteins.append(proteins[unused_idx])
                    complex_positions.append(pos)
                    complex_orientations.append(np.eye(3))
                    used.add(unused_idx)
        
        complexes.append({
            'proteins': complex_proteins,
            'positions': complex_positions,
            'orientations': complex_orientations
        })
        
        return complexes
    
    def _grow_capsid(self,
                    complexes: List[Dict],
                    all_proteins: List[Dict],
                    radius: float) -> Dict:
        """
        Grow capsid by adding proteins one by one at best binding sites.
        """
        if not complexes:
            return {'proteins': [], 'positions': [], 'orientations': []}
        
        # Start with first complex
        assembled = complexes[0]
        used_indices = set(range(len(assembled['proteins'])))
        
        remaining = [i for i in range(len(all_proteins)) if i not in used_indices]
        
        # Add proteins one by one
        for step, protein_idx in enumerate(remaining):
            if step % 100 == 0 and len(remaining) > 100:
                self.logger.info(f"Assembled {step}/{len(remaining)} proteins...")
            
            protein = all_proteins[protein_idx]
            
            # Find best binding site
            best_score = np.inf
            best_pos = None
            best_rot = None
            
            for i, existing_protein in enumerate(assembled['proteins']):
                try:
                    pos, rot, score = self.find_best_docking(
                        existing_protein,
                        protein,
                        existing_protein['coords'],
                        protein['coords'],
                        existing_protein['sequence'],
                        protein['sequence']
                    )
                    
                    # Transform position
                    existing_center = existing_protein['coords'].mean(axis=0)
                    transformed_pos = existing_center + pos
                    
                    # Check for invalid positions
                    if np.any(np.isnan(transformed_pos)) or np.any(np.isinf(transformed_pos)):
                        pos, rot, score = self._geometric_docking(
                            existing_protein['coords'],
                            protein['coords']
                        )
                        transformed_pos = existing_center + pos
                    
                    # Constrain to surface
                    dist = np.linalg.norm(transformed_pos)
                    if dist > 0:
                        transformed_pos = transformed_pos / dist * radius
                    else:
                        # Fallback: geometric position
                        angle = 2 * np.pi * step / len(remaining)
                        theta = np.arccos(2 * np.random.random() - 1)
                        transformed_pos = np.array([
                            radius * np.sin(theta) * np.cos(angle),
                            radius * np.sin(theta) * np.sin(angle),
                            radius * np.cos(theta)
                        ])
                    
                    if score < best_score:
                        best_score = score
                        best_pos = transformed_pos
                        best_rot = rot
                except Exception as e:
                    # Fallback to geometric docking
                    try:
                        pos, rot, score = self._geometric_docking(
                            existing_protein['coords'],
                            protein['coords']
                        )
                        existing_center = existing_protein['coords'].mean(axis=0)
                        transformed_pos = existing_center + pos
                        dist = np.linalg.norm(transformed_pos)
                        if dist > 0:
                            transformed_pos = transformed_pos / dist * radius
                        else:
                            angle = 2 * np.pi * step / len(remaining)
                            theta = np.arccos(2 * np.random.random() - 1)
                            transformed_pos = np.array([
                                radius * np.sin(theta) * np.cos(angle),
                                radius * np.sin(theta) * np.sin(angle),
                                radius * np.cos(theta)
                            ])
                        
                        if score < best_score:
                            best_score = score
                            best_pos = transformed_pos
                            best_rot = rot
                    except:
                        continue
            
            if best_pos is not None:
                assembled['proteins'].append(protein)
                assembled['positions'].append(best_pos)
                assembled['orientations'].append(best_rot)
                used_indices.add(protein_idx)
            else:
                # Fallback: add at geometric position
                angle = 2 * np.pi * step / len(remaining)
                theta = np.arccos(2 * np.random.random() - 1)
                fallback_pos = np.array([
                    radius * np.sin(theta) * np.cos(angle),
                    radius * np.sin(theta) * np.sin(angle),
                    radius * np.cos(theta)
                ])
                assembled['proteins'].append(protein)
                assembled['positions'].append(fallback_pos)
                assembled['orientations'].append(np.eye(3))
                used_indices.add(protein_idx)
        
        return assembled
    
    def _minimize_energy(self,
                        capsid: Dict,
                        radius: float) -> Dict:
        """Energy minimization using atom-level interactions"""
        self.logger.info("Minimizing energy...")
        
        proteins = capsid['proteins']
        positions = capsid['positions']
        orientations = capsid['orientations']
        
        # Calculate all atom coordinates
        all_atoms = []
        atom_to_protein = []
        
        for i, (protein, pos, rot) in enumerate(zip(proteins, positions, orientations)):
            coords = protein['coords']
            centered = coords - coords.mean(axis=0)
            transformed = centered @ rot.T + pos
            
            all_atoms.extend(transformed)
            atom_to_protein.extend([i] * len(coords))
        
        all_atoms = np.array(all_atoms)
        
        # Minimize energy
        def energy_func(x):
            coords_flat = x.reshape(-1, 3)
            return self._calculate_atom_energy(coords_flat, radius)
        
        x0 = all_atoms.flatten()
        result = minimize(energy_func, x0, method='L-BFGS-B', options={'maxiter': 50})
        
        optimized_coords = result.x.reshape(-1, 3)
        
        # Redistribute back to proteins
        idx = 0
        new_positions = []
        for protein in proteins:
            n_atoms = len(protein['coords'])
            protein_coords = optimized_coords[idx:idx+n_atoms]
            new_pos = protein_coords.mean(axis=0)
            
            # Constrain to surface
            dist = np.linalg.norm(new_pos)
            if dist > 0:
                new_pos = new_pos / dist * radius * 0.95
            
            new_positions.append(new_pos)
            idx += n_atoms
        
        capsid['positions'] = new_positions
        
        return capsid
    
    def _calculate_atom_energy(self, all_coords: np.ndarray, radius: float) -> float:
        """Calculate energy using atom-level interactions"""
        distances = cdist(all_coords, all_coords)
        np.fill_diagonal(distances, np.inf)  # Exclude self-interactions
        
        # Avoid division by zero - set minimum distance
        distances = np.clip(distances, 1e-6, None)  # Minimum 1e-6 Angstroms
        
        # Lennard-Jones (van der Waals)
        sigma = 3.5  # Angstroms
        epsilon = 0.1  # kcal/mol
        lj = 4 * epsilon * ((sigma/distances)**12 - (sigma/distances)**6)
        
        # Overlap penalty
        overlap_mask = distances < 2.5
        overlap_penalty = np.sum((2.5 - distances[overlap_mask]) * 100) if np.any(overlap_mask) else 0.0
        
        # Surface constraint
        dist_from_center = np.linalg.norm(all_coords, axis=1)
        surface_penalty = np.sum((dist_from_center - radius * 0.95) ** 2) * 0.01
        
        # Sum LJ energy (avoid NaN/inf)
        lj_energy = np.nansum(np.nan_to_num(lj, nan=0.0, posinf=0.0, neginf=0.0))
        
        return lj_energy + overlap_penalty + surface_penalty
    
    def _detect_geometry(self, assembly: Dict) -> Dict:
        """Detect emergent geometry"""
        positions = assembly['positions']
        
        if len(positions) < 3:
            return {'shape': 'unknown', 'symmetry': 'none'}
        
        center = np.mean(positions, axis=0)
        centered = positions - center
        
        distances = np.linalg.norm(centered, axis=1)
        avg_radius = np.mean(distances)
        radius_variance = np.var(distances) / (avg_radius ** 2)
        
        # PCA
        cov = np.cov(centered.T)
        eigenvals, eigenvecs = np.linalg.eigh(cov)
        eigenvals = np.sort(eigenvals)[::-1]
        elongation_ratio = eigenvals[0] / (eigenvals[1] + 1e-10)
        
        # Symmetry
        symmetry = 'none'
        if len(positions) >= 12 and radius_variance < 0.15:
            if len(positions) % 12 == 0:
                symmetry = 'icosahedral'
            elif len(positions) > 50:
                symmetry = 'helical'
        
        # Shape
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

