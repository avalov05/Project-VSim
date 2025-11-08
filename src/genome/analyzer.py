"""
Genome Analysis Module
Phase 1: Comprehensive viral genome analysis and synthesis feasibility assessment
"""

from Bio import SeqIO
from Bio.Seq import Seq
try:
    from Bio.SeqUtils import GC
except ImportError:
    # Fallback for newer BioPython versions
    def GC(seq):
        """Calculate GC content"""
        seq_str = str(seq).upper()
        gc_count = seq_str.count('G') + seq_str.count('C')
        total = len(seq_str) - seq_str.count('N')  # Exclude ambiguous bases
        return (gc_count / total * 100) if total > 0 else 0.0
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import numpy as np
from collections import Counter

class GenomeAnalyzer:
    """Comprehensive viral genome analyzer"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger('VSim.GenomeAnalyzer')
        self.min_orf_length = config.get('genome_analysis.min_orf_length', 100)
        self.genetic_code = config.get('genome_analysis.genetic_code', 1)
        self.confidence_threshold = config.get('genome_analysis.confidence_threshold', 0.95)
    
    def load_genome(self, filepath: str) -> Dict:
        """Load genome from FASTA file"""
        file_path = Path(filepath)
        
        # Check if file exists
        if not file_path.exists():
            # Try relative to current directory
            abs_path = Path.cwd() / filepath
            if abs_path.exists():
                file_path = abs_path
            else:
                # Try in data/raw directory
                data_path = Path.cwd() / 'data' / 'raw' / filepath
                if data_path.exists():
                    file_path = data_path
                else:
                    raise FileNotFoundError(
                        f"Genome file not found: {filepath}\n"
                        f"Tried locations:\n"
                        f"  - {filepath}\n"
                        f"  - {abs_path}\n"
                        f"  - {data_path}\n"
                        f"\nPlease provide a valid FASTA file path.\n"
                        f"Example: python3 src/main.py data/raw/sample_genome.fasta"
                    )
        
        self.logger.info(f"Loading genome from {file_path}")
        
        # Validate file extension
        if file_path.suffix.lower() not in ['.fasta', '.fa', '.fas', '.fsa', '.fna']:
            self.logger.warning(f"File extension '{file_path.suffix}' may not be FASTA format")
        
        sequences = []
        try:
            for record in SeqIO.parse(str(file_path), "fasta"):
                sequences.append({
                    'id': record.id,
                    'description': record.description,
                    'sequence': str(record.seq).upper(),
                    'length': len(record.seq)
                })
        except Exception as e:
            raise ValueError(
                f"Error parsing FASTA file {file_path}: {str(e)}\n"
                f"Please ensure the file is in valid FASTA format."
            ) from e
        
        if not sequences:
            raise ValueError(
                f"No sequences found in {file_path}\n"
                f"Please ensure the file contains valid FASTA sequences."
            )
        
        genome_data = {
            'sequences': sequences,
            'total_length': sum(s['length'] for s in sequences),
            'num_segments': len(sequences),
            'file_path': str(file_path)
        }
        
        self.logger.info(f"Loaded {len(sequences)} sequence(s), total length: {genome_data['total_length']} bp")
        return genome_data
    
    def analyze(self, genome_data: Dict) -> Dict:
        """Perform comprehensive genome analysis"""
        self.logger.info("Starting comprehensive genome analysis...")
        
        results = {
            'length': genome_data['total_length'],
            'num_segments': genome_data['num_segments'],
            'sequences': []
        }
        
        # Analyze each sequence
        all_orfs = []
        all_proteins = []
        
        for seq_data in genome_data['sequences']:
            seq_analysis = self._analyze_sequence(seq_data)
            results['sequences'].append(seq_analysis)
            all_orfs.extend(seq_analysis['orfs'])
            all_proteins.extend(seq_analysis['proteins'])
        
        # Aggregate results
        results['orfs'] = all_orfs
        results['proteins'] = all_proteins
        results['gc_content'] = self._calculate_gc_content(genome_data)
        results['nucleotide_composition'] = self._analyze_composition(genome_data)
        results['synthesis_feasibility'] = self._assess_synthesis_feasibility(results)
        results['genome_structure'] = self._predict_genome_structure(results)
        results['regulatory_elements'] = self._detect_regulatory_elements(genome_data)
        
        self.logger.info(f"Analysis complete: {len(all_orfs)} ORFs, {len(all_proteins)} proteins")
        return results
    
    def _analyze_sequence(self, seq_data: Dict) -> Dict:
        """Analyze individual sequence"""
        sequence = seq_data['sequence']
        seq_length = len(sequence)
        
        gc_value = GC(sequence)
        # Convert percentage to fraction if needed
        gc_content = gc_value / 100.0 if gc_value > 1 else gc_value
        
        analysis = {
            'id': seq_data['id'],
            'length': seq_length,
            'gc_content': gc_content,
            'orfs': [],
            'proteins': []
        }
        
        # Find ORFs in all reading frames
        for frame in range(3):
            orfs = self._find_orfs(sequence, frame)
            analysis['orfs'].extend(orfs)
        
        # Also check reverse complement
        reverse_seq = str(Seq(sequence).reverse_complement())
        for frame in range(3):
            orfs = self._find_orfs(reverse_seq, frame, reverse=True)
            analysis['orfs'].extend(orfs)
        
        # Filter overlapping ORFs - keep longest ORF when overlapping
        analysis['orfs'] = self._filter_overlapping_orfs(analysis['orfs'])
        
        # Translate ORFs to proteins
        for orf in analysis['orfs']:
            protein = self._translate_orf(orf, sequence)
            if protein:
                analysis['proteins'].append(protein)
        
        return analysis
    
    def _filter_overlapping_orfs(self, orfs: List[Dict]) -> List[Dict]:
        """Filter overlapping ORFs, keeping the longest one"""
        if not orfs:
            return []
        
        # Sort by length (longest first), then by start position, prioritize forward strand
        sorted_orfs = sorted(orfs, key=lambda x: (-x['length'], x['reverse'], x['start']))
        
        filtered = []
        for orf in sorted_orfs:
            # Check if this ORF overlaps with any already added ORF
            overlaps = False
            for existing in filtered:
                # Check if ORFs overlap significantly (>50% overlap)
                overlap_start = max(orf['start'], existing['start'])
                overlap_end = min(orf['end'], existing['end'])
                overlap_length = max(0, overlap_end - overlap_start)
                
                # Calculate overlap percentage
                orf_length = orf['end'] - orf['start']
                existing_length = existing['end'] - existing['start']
                overlap_pct = overlap_length / min(orf_length, existing_length) if min(orf_length, existing_length) > 0 else 0
                
                # If more than 50% overlap and on same strand, skip this one
                if overlap_pct > 0.5 and orf['reverse'] == existing['reverse']:
                    overlaps = True
                    break
            
            if not overlaps:
                filtered.append(orf)
        
        return filtered
    
    def _find_orfs(self, sequence: str, frame: int, reverse: bool = False) -> List[Dict]:
        """Find Open Reading Frames (ORFs)"""
        orfs = []
        start_codons = ['ATG']
        stop_codons = ['TAA', 'TAG', 'TGA']
        
        i = frame
        while i < len(sequence) - 2:
            codon = sequence[i:i+3]
            
            if codon in start_codons:
                # Found start codon, look for stop codon
                j = i + 3
                while j < len(sequence) - 2:
                    codon = sequence[j:j+3]
                    if codon in stop_codons:
                        orf_length = j - i + 3
                        if orf_length >= self.min_orf_length:
                            orf_seq = sequence[i:j+3]
                            orfs.append({
                                'start': i + 1,  # 1-indexed
                                'end': j + 3,
                                'length': orf_length,
                                'frame': frame,
                                'reverse': reverse,
                                'sequence': orf_seq
                            })
                        break
                    j += 3
                i = j + 3
            else:
                i += 3
        
        return orfs
    
    def _translate_orf(self, orf: Dict, sequence: str) -> Optional[Dict]:
        """Translate ORF to protein sequence"""
        try:
            seq = Seq(orf['sequence'])
            if orf['reverse']:
                seq = seq.reverse_complement()
            
            protein_seq = seq.translate(to_stop=True, table=self.genetic_code)
            
            if len(protein_seq) >= self.min_orf_length // 3:
                return {
                    'sequence': str(protein_seq),
                    'length': len(protein_seq),
                    'orf_info': orf,
                    'molecular_weight': self._calculate_molecular_weight(str(protein_seq)),
                    'isoelectric_point': self._estimate_isoelectric_point(str(protein_seq))
                }
        except Exception as e:
            self.logger.warning(f"Translation error: {e}")
        
        return None
    
    def _calculate_molecular_weight(self, protein_seq: str) -> float:
        """Calculate approximate molecular weight"""
        # Amino acid molecular weights (Da)
        aa_weights = {
            'A': 89.09, 'R': 174.20, 'N': 132.12, 'D': 133.10, 'C': 121.16,
            'Q': 146.15, 'E': 147.13, 'G': 75.07, 'H': 155.16, 'I': 131.17,
            'L': 131.17, 'K': 146.19, 'M': 149.21, 'F': 165.19, 'P': 115.13,
            'S': 105.09, 'T': 119.12, 'W': 204.23, 'Y': 181.19, 'V': 117.15
        }
        
        mw = sum(aa_weights.get(aa, 110) for aa in protein_seq.upper())
        mw += 18.02  # Water molecule
        return mw
    
    def _estimate_isoelectric_point(self, protein_seq: str) -> float:
        """Estimate isoelectric point"""
        # Simplified pI calculation based on charged residues
        positive = protein_seq.count('R') + protein_seq.count('K') + protein_seq.count('H')
        negative = protein_seq.count('D') + protein_seq.count('E')
        
        # Rough estimation
        if positive == 0 and negative == 0:
            return 7.0
        
        ratio = positive / (positive + negative) if (positive + negative) > 0 else 0.5
        pI = 4.0 + (ratio * 6.0)  # Approximate range 4-10
        return round(pI, 2)
    
    def _calculate_gc_content(self, genome_data: Dict) -> float:
        """Calculate overall GC content"""
        total_gc = 0
        total_length = 0
        
        for seq_data in genome_data['sequences']:
            gc = GC(seq_data['sequence'])
            # GC returns percentage, convert to fraction
            gc_fraction = gc / 100.0 if gc > 1 else gc
            length = len(seq_data['sequence'])
            total_gc += gc_fraction * length
            total_length += length
        
        return total_gc / total_length if total_length > 0 else 0.0
    
    def _analyze_composition(self, genome_data: Dict) -> Dict:
        """Analyze nucleotide composition"""
        all_seq = ''.join(seq_data['sequence'] for seq_data in genome_data['sequences'])
        counter = Counter(all_seq)
        total = len(all_seq)
        
        composition = {
            'A': counter.get('A', 0) / total if total > 0 else 0,
            'T': counter.get('T', 0) / total if total > 0 else 0,
            'G': counter.get('G', 0) / total if total > 0 else 0,
            'C': counter.get('C', 0) / total if total > 0 else 0,
        }
        
        return composition
    
    def _assess_synthesis_feasibility(self, results: Dict) -> float:
        """Advanced pharmaceutical-grade synthesis feasibility assessment"""
        scores = {}
        
        # 1. Genome length validation (weight: 0.15)
        length = results['length']
        if length < 1000:
            scores['length'] = 0.3  # Too short, likely incomplete
        elif length < 3000:
            scores['length'] = 0.8  # Small but reasonable
        elif length < 50000:
            scores['length'] = 1.0  # Ideal range
        elif length < 500000:
            scores['length'] = 0.9  # Large but possible
        else:
            scores['length'] = 0.6  # Very large, challenging
        
        # 2. ORF detection and quality (weight: 0.25)
        orf_count = len(results['orfs'])
        if orf_count == 0:
            scores['orfs'] = 0.1  # No ORFs found
        elif orf_count < 2:
            scores['orfs'] = 0.5  # Too few ORFs
        elif orf_count < 10:
            scores['orfs'] = 0.95  # Good range
        elif orf_count < 50:
            scores['orfs'] = 0.85  # Many ORFs
        else:
            scores['orfs'] = 0.7  # Too many, might be false positives
        
        # Check ORF quality (overlapping, nested, etc.)
        orf_quality = self._assess_orf_quality(results['orfs'])
        scores['orf_quality'] = orf_quality
        
        # 3. Protein prediction (weight: 0.20)
        protein_count = len(results['proteins'])
        if protein_count == 0:
            scores['proteins'] = 0.1
        elif protein_count < 2:
            scores['proteins'] = 0.6
        elif protein_count < 10:
            scores['proteins'] = 0.95
        else:
            scores['proteins'] = 0.85
        
        # Check protein quality
        protein_quality = self._assess_protein_quality(results['proteins'])
        scores['protein_quality'] = protein_quality
        
        # 4. GC content analysis (weight: 0.15)
        gc_content = results['gc_content']
        if gc_content < 0.20 or gc_content > 0.80:
            scores['gc_content'] = 0.6  # Extreme GC content
        elif gc_content < 0.30 or gc_content > 0.70:
            scores['gc_content'] = 0.8  # Moderate extremes
        else:
            scores['gc_content'] = 1.0  # Normal range
        
        # 5. Sequence quality and composition (weight: 0.15)
        composition_score = self._assess_sequence_composition(results)
        scores['composition'] = composition_score
        
        # 6. Regulatory elements (weight: 0.10)
        regulatory_score = self._assess_regulatory_elements(results.get('regulatory_elements', {}))
        scores['regulatory'] = regulatory_score
        
        # Weighted average
        weights = {
            'length': 0.15,
            'orfs': 0.15,
            'orf_quality': 0.10,
            'proteins': 0.10,
            'protein_quality': 0.10,
            'gc_content': 0.15,
            'composition': 0.15,
            'regulatory': 0.10
        }
        
        total_score = sum(scores.get(key, 0.5) * weights.get(key, 0) for key in weights.keys())
        total_weight = sum(weights.values())
        
        final_score = total_score / total_weight if total_weight > 0 else 0.5
        
        # Ensure score is between 0 and 1
        return min(max(final_score, 0.0), 1.0)
    
    def _assess_orf_quality(self, orfs: List[Dict]) -> float:
        """Assess quality of detected ORFs"""
        if not orfs:
            return 0.0
        
        scores = []
        
        for orf in orfs:
            score = 1.0
            
            # Check ORF length
            length = orf['length']
            if length < 100:
                score *= 0.5
            elif length < 300:
                score *= 0.8
            elif length > 10000:
                score *= 0.7
            
            # Check for internal stop codons (should not have them)
            seq = orf['sequence']
            if len(seq) > 3:
                # Check every 3rd position for stop codons
                for i in range(3, len(seq) - 3, 3):
                    codon = seq[i:i+3]
                    if codon in ['TAA', 'TAG', 'TGA']:
                        score *= 0.3  # Internal stop codon
            
            scores.append(score)
        
        return np.mean(scores) if scores else 0.5
    
    def _assess_protein_quality(self, proteins: List[Dict]) -> float:
        """Assess quality of predicted proteins"""
        if not proteins:
            return 0.0
        
        scores = []
        
        for protein in proteins:
            score = 1.0
            seq = protein['sequence']
            
            # Check length
            if len(seq) < 30:
                score *= 0.6
            elif len(seq) > 2000:
                score *= 0.7
            
            # Check for valid amino acids
            valid_aas = set('ACDEFGHIKLMNPQRSTVWY')
            invalid_count = sum(1 for aa in seq if aa not in valid_aas)
            if invalid_count > 0:
                score *= (1 - invalid_count / len(seq))
            
            # Check for unusual composition
            if seq.count('X') / len(seq) > 0.1:  # Too many unknown residues
                score *= 0.5
            
            scores.append(score)
        
        return np.mean(scores) if scores else 0.5
    
    def _assess_sequence_composition(self, results: Dict) -> float:
        """Assess sequence composition quality"""
        composition = results.get('nucleotide_composition', {})
        
        # Check for balanced composition
        at_content = composition.get('A', 0) + composition.get('T', 0)
        gc_content = composition.get('G', 0) + composition.get('C', 0)
        
        # Both should be present
        if at_content < 0.1 or gc_content < 0.1:
            return 0.4
        
        # Check for reasonable ratios
        if at_content / gc_content > 10 or gc_content / at_content > 10:
            return 0.6
        
        return 1.0
    
    def _assess_regulatory_elements(self, regulatory: Dict) -> float:
        """Assess regulatory elements"""
        score = 0.5  # Base score
        
        # Check for presence of regulatory elements
        promoters = regulatory.get('promoters', [])
        terminators = regulatory.get('terminators', [])
        
        if promoters:
            score += 0.2
        if terminators:
            score += 0.2
        
        # Check for ribosome binding sites
        if regulatory.get('ribosome_binding_sites'):
            score += 0.1
        
        return min(score, 1.0)
    
    def _predict_genome_structure(self, results: Dict) -> Dict:
        """Predict genome structure characteristics"""
        length = results['length']
        
        # Classify by size
        if length < 5000:
            structure_type = "Small RNA virus"
        elif length < 30000:
            structure_type = "Medium RNA virus"
        elif length < 200000:
            structure_type = "Large DNA virus"
        else:
            structure_type = "Very large DNA virus"
        
        return {
            'type': structure_type,
            'estimated_complexity': min(len(results['proteins']) / 10, 1.0),
            'segmented': results['num_segments'] > 1
        }
    
    def _detect_regulatory_elements(self, genome_data: Dict) -> Dict:
        """Detect regulatory elements"""
        # Simplified detection - would use more sophisticated methods in production
        elements = {
            'promoters': [],
            'terminators': [],
            'ribosome_binding_sites': []
        }
        
        # Look for common regulatory sequences
        for seq_data in genome_data['sequences']:
            sequence = seq_data['sequence']
            
            # TATA box (simplified)
            if 'TATAAA' in sequence or 'TATATA' in sequence:
                elements['promoters'].append('TATA-like')
            
            # PolyA signal (simplified)
            if 'AATAAA' in sequence:
                elements['terminators'].append('PolyA signal')
        
        return elements

