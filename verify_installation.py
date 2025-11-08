#!/usr/bin/env python3
"""
VSim Installation Verification Script
Checks if all components are properly installed and working
"""

import sys
from pathlib import Path

def check_import(module_name, package_name=None):
    """Check if a module can be imported"""
    try:
        __import__(module_name)
        print(f"✅ {package_name or module_name}")
        return True
    except ImportError as e:
        print(f"❌ {package_name or module_name}: {e}")
        return False

def check_file(filepath):
    """Check if a file exists"""
    path = Path(filepath)
    if path.exists():
        print(f"✅ {filepath}")
        return True
    else:
        print(f"❌ {filepath} - File not found")
        return False

def main():
    """Main verification function"""
    print("="*60)
    print("VSim Installation Verification")
    print("="*60)
    
    print("\n📦 Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 9:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} (3.9+ required)")
        return False
    
    print("\n📚 Checking core dependencies...")
    dependencies = [
        ('Bio', 'BioPython'),
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('scipy', 'SciPy'),
        ('sklearn', 'scikit-learn'),
        ('flask', 'Flask'),
    ]
    
    all_ok = True
    for module, name in dependencies:
        if not check_import(module, name):
            all_ok = False
    
    print("\n📁 Checking project structure...")
    directories = [
        'src',
        'src/core',
        'src/genome',
        'src/structure',
        'src/environmental',
        'src/cell_interaction',
        'src/cancer',
        'src/ml',
        'src/web',
        'data/raw',
        'docs',
        'tests'
    ]
    
    for directory in directories:
        if not check_file(directory):
            all_ok = False
    
    print("\n📄 Checking key files...")
    files = [
        'config.yaml',
        'requirements.txt',
        'README.md',
        'src/main.py',
        'src/core/config.py',
        'src/genome/analyzer.py',
        'data/raw/sample_genome.fasta'
    ]
    
    for file in files:
        if not check_file(file):
            all_ok = False
    
    print("\n🔍 Testing imports...")
    sys.path.insert(0, str(Path(__file__).parent / 'src'))
    
    modules_to_test = [
        ('src.core.config', 'Config'),
        ('src.genome.analyzer', 'GenomeAnalyzer'),
        ('src.structure.predictor', 'StructurePredictor'),
        ('src.environmental.analyzer', 'EnvironmentalAnalyzer'),
        ('src.cell_interaction.analyzer', 'CellInteractionAnalyzer'),
        ('src.cancer.analyzer', 'CancerAnalyzer'),
        ('src.ml.predictor', 'MLPredictor'),
    ]
    
    for module_path, class_name in modules_to_test:
        try:
            module = __import__(module_path, fromlist=[class_name])
            cls = getattr(module, class_name)
            print(f"✅ {module_path}.{class_name}")
        except Exception as e:
            print(f"❌ {module_path}.{class_name}: {e}")
            all_ok = False
    
    print("\n" + "="*60)
    if all_ok:
        print("✅ All checks passed! VSim is ready to use.")
        print("\nNext steps:")
        print("1. Run: python3 src/main.py data/raw/sample_genome.fasta")
        print("2. Or start web interface: python3 src/web/app.py")
        return True
    else:
        print("❌ Some checks failed. Please install missing dependencies:")
        print("   pip install -r requirements.txt")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

