"""
Genome Annotator - Annotates viral genomes and predicts genes
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from Bio import SeqIO
from Bio.Seq import Seq
try:
    from Bio.SeqUtils import GC
except ImportError:
    # Fallback for calculating GC content
    def GC(seq):
        """Calculate GC content percentage"""
        if not seq:
            return 0.0
        gc_count = seq.upper().count('G') + seq.upper().count('C')
        return (gc_count / len(seq)) * 100 if len(seq) > 0 else 0.0

logger = logging.getLogger(__name__)

class GenomeAnnotator:
    """Annotate viral genomes"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def annotate(self, genome_path: Path) -> Dict[str, Any]:
        """
        Annotate viral genome
        
        Args:
            genome_path: Path to genome FASTA file
        
        Returns:
            Annotation dictionary
        """
        self.logger.info(f"Annotating genome: {genome_path}")
        
        # Read genome
        genome = self._read_genome(genome_path)
        
        # Basic statistics
        stats = self._calculate_statistics(genome)
        
        # Predict genes/ORFs
        genes = self._predict_genes(genome)
        
        # Predict gene functions
        gene_functions = self._predict_gene_functions(genes, genome)
        
        # Calculate gene completeness
        gene_completeness = self._calculate_gene_completeness(genes)
        
        # Codon usage
        codon_usage = self._analyze_codon_usage(genes, genome)
        
        # Regulatory elements
        regulatory = self._find_regulatory_elements(genome)
        
        # Virus type prediction
        virus_type = self._predict_virus_type(genome, genes)
        
        # Determine if enveloped
        is_enveloped = self._predict_envelope(genes, virus_type)
        
        return {
            'genome_path': str(genome_path),
            'genome_sequence': str(genome.seq),
            'genome_length': len(genome.seq),
            'gc_content': GC(genome.seq),
            'is_circular': self._is_circular(genome),
            'is_enveloped': is_enveloped,
            'virus_type': virus_type,
            'genes': gene_functions,
            'gene_count': len(genes),
            'gene_completeness': gene_completeness,
            'codon_usage': codon_usage,
            'regulatory_elements': regulatory,
            'statistics': stats
        }
    
    def _read_genome(self, genome_path: Path) -> SeqIO.SeqRecord:
        """Read genome from FASTA file"""
        try:
            records = list(SeqIO.parse(genome_path, "fasta"))
            if not records:
                raise ValueError("No sequences found in genome file")
            return records[0]
        except Exception as e:
            self.logger.error(f"Error reading genome: {e}")
            raise
    
    def _calculate_statistics(self, genome: SeqIO.SeqRecord) -> Dict[str, Any]:
        """Calculate basic statistics"""
        seq = genome.seq
        return {
            'length': len(seq),
            'gc_content': GC(seq),
            'at_content': 1 - GC(seq) / 100,
            'n_count': seq.count('N') + seq.count('n'),
            'ambiguous_bases': sum(seq.count(base) for base in 'RYSWKMBDHVN'),
        }
    
    def _predict_genes(self, genome: SeqIO.SeqRecord, min_length: int = 100) -> List[Dict[str, Any]]:
        """Predict ORFs/genes in all reading frames"""
        seq = str(genome.seq).upper()
        genes = []
        
        # Check all 6 reading frames (3 forward, 3 reverse)
        for frame in range(3):
            # Forward frames
            orfs = self._find_orfs(seq[frame:], frame, '+', min_length)
            genes.extend(orfs)
            
            # Reverse frames
            rev_seq = str(Seq(seq).reverse_complement())
            orfs = self._find_orfs(rev_seq[frame:], frame, '-', min_length)
            genes.extend(orfs)
        
        # Sort by position
        genes.sort(key=lambda x: x['start'])
        
        # Filter overlapping ORFs (keep longest)
        genes = self._filter_overlapping(genes)
        
        return genes
    
    def _find_orfs(self, sequence: str, frame: int, strand: str, min_length: int) -> List[Dict[str, Any]]:
        """Find ORFs in a sequence"""
        orfs = []
        start_codons = ['ATG']
        stop_codons = ['TAA', 'TAG', 'TGA']
        
        i = 0
        while i < len(sequence) - 2:
            codon = sequence[i:i+3]
            
            if codon in start_codons:
                # Found start, look for stop
                for j in range(i+3, len(sequence)-2, 3):
                    stop_codon = sequence[j:j+3]
                    if stop_codon in stop_codons:
                        length = j - i + 3
                        if length >= min_length:
                            orfs.append({
                                'start': i,
                                'end': j + 3,
                                'length': length,
                                'frame': frame,
                                'strand': strand,
                                'sequence': sequence[i:j+3]
                            })
                        i = j + 3
                        break
                else:
                    i += 3
            else:
                i += 1
        
        return orfs
    
    def _filter_overlapping(self, genes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter overlapping genes, keeping the longest"""
        if not genes:
            return []
        
        filtered = []
        sorted_genes = sorted(genes, key=lambda x: x['length'], reverse=True)
        used_positions = set()
        
        for gene in sorted_genes:
            start, end = gene['start'], gene['end']
            
            # Check if overlaps with existing genes
            overlaps = False
            for pos in range(start, end):
                if pos in used_positions:
                    overlaps = True
                    break
            
            if not overlaps:
                filtered.append(gene)
                for pos in range(start, end):
                    used_positions.add(pos)
        
        return sorted(filtered, key=lambda x: x['start'])
    
    def _predict_gene_functions(self, genes: List[Dict[str, Any]], 
                                genome: SeqIO.SeqRecord) -> List[Dict[str, Any]]:
        """Predict function of each gene"""
        annotated_genes = []
        
        for i, gene in enumerate(genes):
            seq = gene['sequence']
            
            # Predict function based on sequence features
            function = self._predict_function(seq, gene, i, len(genes))
            
            annotated_genes.append({
                **gene,
                'id': f"gene_{i+1}",
                'function': function,
                'protein_sequence': self._translate(seq),
                'protein_length': len(self._translate(seq))
            })
        
        return annotated_genes
    
    def _predict_function(self, sequence: str, gene: Dict[str, Any], 
                         index: int, total_genes: int) -> str:
        """Predict gene function from sequence"""
        # Simple heuristic-based prediction
        # In production, this would use BLAST, HMMs, or ML models
        
        seq_upper = sequence.upper()
        length = len(sequence)
        
        # Check for common motifs
        if 'GGGGGG' in seq_upper or 'CCCCCC' in seq_upper:
            return 'polymerase'
        
        # Length-based heuristics
        if length > 2000:
            return 'polymerase'
        elif length > 1000:
            if index == 0:
                return 'replicase'
            else:
                return 'capsid'
        elif length > 500:
            return 'envelope'
        elif 'ATG' * 10 in seq_upper:
            return 'structural'
        else:
            return 'unknown'
    
    def _translate(self, dna_seq: str) -> str:
        """Translate DNA to protein"""
        try:
            return str(Seq(dna_seq).translate())
        except:
            return ""
    
    def _calculate_gene_completeness(self, genes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate gene completeness score"""
        if not genes:
            return {'score': 0.0, 'complete_genes': 0, 'total_genes': 0}
        
        # Check if genes have start and stop codons
        complete = sum(1 for g in genes 
                      if g['sequence'].startswith('ATG') and 
                      g['sequence'][-3:] in ['TAA', 'TAG', 'TGA'])
        
        score = complete / len(genes) if genes else 0.0
        
        return {
            'score': score,
            'complete_genes': complete,
            'total_genes': len(genes),
            'fraction_complete': score
        }
    
    def _analyze_codon_usage(self, genes: List[Dict[str, Any]], 
                            genome: SeqIO.SeqRecord) -> Dict[str, Any]:
        """Analyze codon usage bias"""
        if not genes:
            return {'bias_score': 0.0, 'cai': 0.0, 'frequencies': {}}
        
        # Count codons
        codon_counts = {}
        total_codons = 0
        
        for gene in genes:
            seq = gene['sequence'].upper()
            for i in range(0, len(seq)-2, 3):
                codon = seq[i:i+3]
                if len(codon) == 3 and 'N' not in codon:
                    codon_counts[codon] = codon_counts.get(codon, 0) + 1
                    total_codons += 1
        
        # Calculate frequencies
        frequencies = {codon: count/total_codons 
                      for codon, count in codon_counts.items()} if total_codons > 0 else {}
        
        # Simple bias score (entropy-based)
        import math
        if frequencies:
            entropy = -sum(p * math.log2(p) for p in frequencies.values() if p > 0)
            max_entropy = math.log2(64)  # 64 possible codons
            bias_score = 1 - (entropy / max_entropy) if max_entropy > 0 else 0.0
        else:
            bias_score = 0.0
        
        # CAI (Codon Adaptation Index) - simplified
        cai = 0.5  # Placeholder
        
        return {
            'bias_score': bias_score,
            'cai': cai,
            'frequencies': frequencies,
            'total_codons': total_codons
        }
    
    def _find_regulatory_elements(self, genome: SeqIO.SeqRecord) -> Dict[str, Any]:
        """Find regulatory elements"""
        seq = str(genome.seq).upper()
        
        # Find promoters (simplified)
        promoters = []
        # TATA box
        if 'TATA' in seq[:1000]:
            promoters.append({'type': 'TATA', 'position': seq.find('TATA')})
        
        # Find RBS (ribosome binding sites)
        rbs = []
        if 'AGGAGG' in seq[:500] or 'GGAGG' in seq[:500]:
            rbs.append({'position': 0})  # Simplified
        
        score = min(len(promoters) * 0.3 + len(rbs) * 0.2, 1.0)
        
        return {
            'score': score,
            'promoters': promoters,
            'rbs': rbs
        }
    
    def _predict_virus_type(self, genome: SeqIO.SeqRecord, 
                           genes: List[Dict[str, Any]]) -> str:
        """Predict virus type"""
        seq = str(genome.seq).upper()
        length = len(seq)
        gc_content = GC(seq)
        
        # Simple heuristics
        if length < 5000:
            return 'small_rna_virus'
        elif length < 15000:
            if gc_content > 50:
                return 'dna_virus'
            else:
                return 'rna_virus'
        else:
            return 'large_rna_virus'
    
    def _predict_envelope(self, genes: List[Dict[str, Any]], 
                         virus_type: str) -> bool:
        """Predict if virus is enveloped"""
        # Check for envelope proteins
        functions = [g.get('function', '').lower() for g in genes]
        has_envelope = any('envelope' in f or 'glycoprotein' in f for f in functions)
        
        # Type-based heuristics
        if 'coronavirus' in virus_type.lower() or 'flavi' in virus_type.lower():
            return True
        elif 'picorna' in virus_type.lower() or 'norovirus' in virus_type.lower():
            return False
        
        return has_envelope
    
    def _is_circular(self, genome: SeqIO.SeqRecord) -> bool:
        """Check if genome is circular"""
        # Check if ends are similar (indicating circularization)
        seq = str(genome.seq)
        if len(seq) < 100:
            return False
        
        # Check for overlapping ends
        end1 = seq[-50:]
        end2 = seq[:50]
        
        # Simple check
        return end1[-20:] in end2 or end2[:20] in end1

