"""
Assembly Simulator - Simulates virion assembly and generates 3D models
"""

import logging
from pathlib import Path
from typing import Dict, Any, List
import numpy as np

logger = logging.getLogger(__name__)

class AssemblySimulator:
    """Simulate virion assembly"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def assemble(self, annotation: Dict[str, Any], structures: Dict[str, Any]) -> Dict[str, Any]:
        """Assemble complete virion"""
        self.logger.info("Simulating virion assembly...")
        
        # Predict geometry
        geometry = self._predict_geometry(annotation)
        
        # Assemble capsid
        capsid = self._assemble_capsid(annotation, structures, geometry)
        
        # Add envelope if needed
        if annotation.get('is_enveloped', False):
            virion = self._add_envelope(capsid, annotation, structures)
        else:
            virion = capsid
        
        # Save 3D model
        pdb_file = self._save_virion_model(virion, annotation)
        
        return {
            'geometry': geometry,
            'capsid': capsid,
            'virion': virion,
            'pdb_file': str(pdb_file),
            'method': self.config.assembly_method,
            'diameter_nm': geometry.get('diameter', 0.0),
            'shape': geometry.get('shape', 'unknown')
        }
    
    def _predict_geometry(self, annotation: Dict[str, Any]) -> Dict[str, Any]:
        """Predict virion geometry"""
        virus_type = annotation.get('virus_type', 'unknown')
        genome_length = annotation.get('genome_length', 0)
        is_enveloped = annotation.get('is_enveloped', False)
        
        # Predict shape
        if 'filamentous' in virus_type.lower() or 'filo' in virus_type.lower():
            shape = 'filamentous'
            diameter = 80.0  # nm
            length = genome_length / 100.0  # Rough estimate
        elif 'corona' in virus_type.lower():
            shape = 'spherical'
            diameter = 120.0  # nm
            length = diameter
        elif 'small' in virus_type.lower():
            shape = 'icosahedral'
            diameter = 30.0  # nm
            length = diameter
        else:
            shape = 'spherical'
            diameter = 100.0  # nm
            length = diameter
        
        # Predict symmetry
        if shape == 'icosahedral':
            symmetry = 'T=3'
        elif shape == 'filamentous':
            symmetry = 'helical'
        else:
            symmetry = 'none'
        
        return {
            'shape': shape,
            'diameter': diameter,
            'length': length,
            'symmetry': symmetry,
            'is_enveloped': is_enveloped
        }
    
    def _assemble_capsid(self, annotation: Dict[str, Any], 
                        structures: Dict[str, Any], 
                        geometry: Dict[str, Any]) -> Dict[str, Any]:
        """Assemble capsid structure"""
        structures_list = structures.get('structures', [])
        capsid_proteins = [s for s in structures_list 
                          if 'capsid' in s.get('gene_id', '').lower() or
                          'structural' in annotation.get('genes', [{}])[0].get('function', '').lower()]
        
        if not capsid_proteins:
            # Use first structure as capsid
            capsid_proteins = structures_list[:1]
        
        # Estimate number of subunits
        num_subunits = self._estimate_subunits(geometry)
        
        return {
            'proteins': capsid_proteins,
            'num_subunits': num_subunits,
            'geometry': geometry
        }
    
    def _estimate_subunits(self, geometry: Dict[str, Any]) -> int:
        """Estimate number of capsid subunits"""
        shape = geometry.get('shape', 'spherical')
        diameter = geometry.get('diameter', 100.0)
        
        if shape == 'icosahedral':
            # T=3 icosahedron has 180 subunits
            return 180
        elif shape == 'spherical':
            # Rough estimate based on diameter
            return int((diameter / 10) ** 2 * 10)
        else:
            # Filamentous - estimate based on length
            length = geometry.get('length', 1000.0)
            return int(length / 5.0)
    
    def _add_envelope(self, capsid: Dict[str, Any], 
                     annotation: Dict[str, Any],
                     structures: Dict[str, Any]) -> Dict[str, Any]:
        """Add envelope and spikes"""
        structures_list = structures.get('structures', [])
        spike_proteins = [s for s in structures_list 
                         if 'spike' in s.get('gene_id', '').lower() or
                         'envelope' in s.get('gene_id', '').lower()]
        
        # Estimate spike density
        num_spikes = int(capsid.get('num_subunits', 100) * 0.1)  # 10% spikes
        
        return {
            **capsid,
            'envelope': True,
            'spikes': spike_proteins,
            'num_spikes': num_spikes
        }
    
    def _save_virion_model(self, virion: Dict[str, Any], 
                          annotation: Dict[str, Any]) -> Path:
        """Save complete virion 3D model"""
        output_dir = self.config.output_dir / "structures"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        pdb_file = output_dir / "virus_assembled.pdb"
        
        # Generate PDB content
        pdb_lines = [
            "REMARK   VSim Virtual Lab - Complete Virion Model",
            f"REMARK   Virus Type: {annotation.get('virus_type', 'unknown')}",
            f"REMARK   Shape: {virion.get('geometry', {}).get('shape', 'unknown')}",
            f"REMARK   Diameter: {virion.get('geometry', {}).get('diameter', 0.0):.1f} nm",
            ""
        ]
        
        # Add capsid proteins
        atom_num = 1
        residue_num = 1
        chain_id = 'A'
        
        geometry = virion.get('geometry', {})
        shape = geometry.get('shape', 'spherical')
        diameter = geometry.get('diameter', 100.0)
        num_subunits = virion.get('num_subunits', 100)
        
        # Generate positions based on geometry
        positions = self._generate_positions(shape, diameter, num_subunits)
        
        for pos in positions:
            x, y, z = pos
            pdb_lines.append(
                f"ATOM  {atom_num:5d}  CA  MET {chain_id}{residue_num:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 50.00"
            )
            atom_num += 1
            residue_num += 1
            
            if residue_num > 9999:
                residue_num = 1
                chain_id = chr(ord(chain_id) + 1)
        
        pdb_lines.append("END")
        
        with open(pdb_file, 'w') as f:
            f.write("\n".join(pdb_lines))
        
        self.logger.info(f"Virion model saved to: {pdb_file}")
        return pdb_file
    
    def _generate_positions(self, shape: str, diameter: float, num_points: int) -> List[tuple]:
        """Generate 3D positions for virion structure"""
        positions = []
        
        if shape == 'spherical' or shape == 'icosahedral':
            # Generate points on sphere
            for i in range(num_points):
                theta = 2 * np.pi * i / num_points
                phi = np.arccos(2 * (i / num_points) - 1)
                radius = diameter / 2.0
                
                x = radius * np.sin(phi) * np.cos(theta)
                y = radius * np.sin(phi) * np.sin(theta)
                z = radius * np.cos(phi)
                
                positions.append((x, y, z))
        
        elif shape == 'filamentous':
            # Generate points along filament
            length = diameter * 10  # Filament length
            for i in range(num_points):
                z = (i / num_points) * length - length / 2
                angle = 2 * np.pi * i / 20  # Helical twist
                radius = diameter / 2.0
                
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                
                positions.append((x, y, z))
        
        else:
            # Default: spherical
            for i in range(num_points):
                theta = 2 * np.pi * i / num_points
                phi = np.arccos(2 * (i / num_points) - 1)
                radius = diameter / 2.0
                
                x = radius * np.sin(phi) * np.cos(theta)
                y = radius * np.sin(phi) * np.sin(theta)
                z = radius * np.cos(phi)
                
                positions.append((x, y, z))
        
        return positions

