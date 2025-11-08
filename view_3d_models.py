#!/usr/bin/env python3
"""
Simple 3D Model Viewer for VSim Results
View your PDB files interactively using various methods
"""

import os
from pathlib import Path
import sys

def view_html_report():
    """Open the HTML report in your default browser"""
    report_path = Path('results/comprehensive_report.html')
    if report_path.exists():
        print(f"📊 Opening HTML report: {report_path}")
        if sys.platform == 'darwin':  # macOS
            os.system(f'open "{report_path}"')
        elif sys.platform == 'linux':
            os.system(f'xdg-open "{report_path}"')
        elif sys.platform == 'win32':
            os.system(f'start "{report_path}"')
        else:
            print(f"Open this file in your browser: {report_path.absolute()}")
        return True
    else:
        print(f"❌ HTML report not found at: {report_path}")
        return False

def list_pdb_files():
    """List all available PDB files"""
    structures_dir = Path('results/structures')
    
    if not structures_dir.exists():
        print(f"❌ Structures directory not found: {structures_dir}")
        return [], []
    
    virus_files = sorted(structures_dir.glob('virus_*.pdb'))
    protein_files = sorted((structures_dir / 'proteins').glob('*.pdb')) if (structures_dir / 'proteins').exists() else []
    
    print("\n" + "="*70)
    print("AVAILABLE 3D MODELS")
    print("="*70)
    
    if virus_files:
        print(f"\n🦠 Virus Particle Models ({len(virus_files)} files):")
        for f in virus_files:
            size_kb = f.stat().st_size / 1024
            print(f"   • {f.name} ({size_kb:.1f} KB)")
    else:
        print("\n⚠️  No virus particle models found")
    
    if protein_files:
        print(f"\n🧬 Protein Structures ({len(protein_files)} files):")
        for f in protein_files[:10]:  # Show first 10
            size_kb = f.stat().st_size / 1024
            print(f"   • {f.name} ({size_kb:.1f} KB)")
        if len(protein_files) > 10:
            print(f"   ... and {len(protein_files) - 10} more")
    else:
        print("\n⚠️  No protein structure files found")
    
    print("="*70 + "\n")
    
    return virus_files, protein_files

def view_with_py3dmol():
    """View PDB files using py3Dmol (interactive in Jupyter)"""
    try:
        import py3Dmol
        print("✓ py3Dmol available")
        
        virus_files, protein_files = list_pdb_files()
        
        if not virus_files and not protein_files:
            print("No PDB files to display")
            return
        
        print("\n📦 Viewing with py3Dmol:")
        print("   Note: This works best in Jupyter notebooks")
        print("   For standalone viewing, use the HTML report or online viewers\n")
        
        # Show first virus file
        if virus_files:
            pdb_file = virus_files[0]
            print(f"   Displaying: {pdb_file.name}")
            with open(pdb_file, 'r') as f:
                pdb_data = f.read()
            
            view = py3Dmol.view(width=800, height=600)
            view.addModel(pdb_data, 'pdb')
            view.setStyle({'cartoon': {'color': 'spectrum'}})
            view.setBackgroundColor('white')
            view.zoomTo()
            view.show()
            
    except ImportError:
        print("⚠️  py3Dmol not installed. Install with: pip install py3Dmol")
        print("   Or use one of the other viewing methods below")

def view_with_online_viewer():
    """Instructions for using online viewers"""
    print("\n" + "="*70)
    print("ONLINE 3D VIEWERS")
    print("="*70)
    print("\nYou can upload your PDB files to these online viewers:")
    print("\n1. Mol* Viewer (https://molstar.org/viewer/)")
    print("   • Drag and drop your PDB file")
    print("   • Interactive 3D visualization")
    print("   • No installation needed")
    
    print("\n2. 3Dmol.js Viewer (https://3dmol.csb.pitt.edu/)")
    print("   • Paste PDB content or upload file")
    print("   • Simple and fast")
    
    print("\n3. RCSB PDB Viewer (https://www.rcsb.org/3d-view)")
    print("   • Upload PDB file")
    print("   • Professional molecular viewer")
    
    print("\n" + "="*70)
    
    virus_files, protein_files = list_pdb_files()
    
    if virus_files:
        print(f"\n💡 Try uploading: {virus_files[0].absolute()}")
    elif protein_files:
        print(f"\n💡 Try uploading: {protein_files[0].absolute()}")

def view_with_nglview():
    """View with NGLview (requires Jupyter)"""
    try:
        import nglview
        print("✓ NGLview available")
        print("   Note: NGLview requires Jupyter notebook")
        print("   Use: import nglview; view = nglview.show_file('path/to/file.pdb'); view")
    except ImportError:
        print("⚠️  NGLview not installed. Install with: pip install nglview")
        print("   Note: Requires Jupyter notebook")

def main():
    """Main function"""
    print("\n" + "="*70)
    print("VSim 3D Model Viewer")
    print("="*70)
    
    # Check if we're in the right directory
    if not Path('results').exists():
        print("\n⚠️  'results' directory not found!")
        print("   Make sure you're running this from the Project-VSim directory")
        print(f"   Current directory: {Path.cwd()}")
        return
    
    # List available files
    virus_files, protein_files = list_pdb_files()
    
    if not virus_files and not protein_files:
        print("❌ No PDB files found. Run analysis first:")
        print("   python3 src/main.py data/raw/your_genome.fasta")
        return
    
    # Options
    print("\nVIEWING OPTIONS:")
    print("1. HTML Report (Recommended) - Interactive 3D visualization in browser")
    print("2. Online Viewers - Upload PDB files to web-based viewers")
    print("3. py3Dmol - Interactive viewer (requires Jupyter)")
    
    # Check if running interactively
    try:
        choice = input("\nChoose option (1/2/3) or press Enter for HTML report: ").strip()
    except EOFError:
        # Non-interactive mode, default to HTML report
        choice = '1'
    
    if choice == '2':
        view_with_online_viewer()
    elif choice == '3':
        view_with_py3dmol()
    else:
        # Default: HTML report
        if view_html_report():
            print("\n✅ HTML report opened in your browser!")
            print("   Scroll to 'Virus 3D Model' section to see the interactive 3D viewer")
            print("   You can rotate, zoom, and interact with the model")

if __name__ == '__main__':
    main()

