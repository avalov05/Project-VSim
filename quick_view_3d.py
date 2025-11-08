#!/usr/bin/env python3
"""
Quick 3D Model Viewer Generator
Creates a standalone HTML viewer for existing PDB files without regenerating the model
"""

import os
from pathlib import Path
import sys

def create_viewer_html(pdb_file_path, output_html_path):
    """Create a standalone HTML viewer for a PDB file"""
    
    pdb_file = Path(pdb_file_path)
    if not pdb_file.exists():
        print(f"❌ PDB file not found: {pdb_file}")
        return False
    
    # Read PDB content
    print(f"📖 Reading PDB file: {pdb_file.name} ({pdb_file.stat().st_size / 1024:.1f} KB)")
    with open(pdb_file, 'r') as f:
        pdb_content = f.read()
    
    # Ensure END statement exists
    pdb_content = pdb_content.rstrip()
    if not pdb_content.endswith('END'):
        pdb_content += '\nEND\n'
    
    # Escape for HTML/JavaScript
    import html
    pdb_content_escaped = html.escape(pdb_content)
    
    # Create HTML viewer
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>VSim 3D Model Viewer - {pdb_file.name}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            margin-top: 0;
        }}
        .viewer-container {{
            margin: 20px 0;
            border: 2px solid #3498db;
            border-radius: 10px;
            padding: 20px;
            background: #f8f9fa;
        }}
        #viewer {{
            width: 100%;
            height: 700px;
            background: white;
            border-radius: 5px;
            border: 1px solid #ddd;
        }}
        .controls {{
            margin-top: 15px;
            padding: 15px;
            background: #e8f4f8;
            border-radius: 5px;
        }}
        button {{
            padding: 10px 20px;
            margin: 5px;
            background: #3498db;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
        }}
        button:hover {{
            background: #2980b9;
        }}
        .info {{
            margin-top: 15px;
            padding: 15px;
            background: #ecf0f1;
            border-radius: 5px;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🧬 VSim 3D Model Viewer</h1>
        <p><strong>Model:</strong> {pdb_file.name}</p>
        
        <div class="viewer-container">
            <div id="viewer"></div>
            
            <div class="controls">
                <strong>Controls:</strong>
                <button onclick="resetView()">Reset View</button>
                <button onclick="toggleStyle('cartoon')">Cartoon</button>
                <button onclick="toggleStyle('stick')">Stick</button>
                <button onclick="toggleStyle('sphere')">Sphere</button>
                <button onclick="toggleStyle('surface')">Surface</button>
                <button onclick="toggleRotation()">Toggle Rotation</button>
            </div>
        </div>
        
        <div class="info">
            <p><strong>Tips:</strong></p>
            <ul>
                <li>🖱️ <strong>Drag</strong> to rotate the model</li>
                <li>🔍 <strong>Scroll</strong> to zoom in/out</li>
                <li>🎨 Use style buttons to change visualization</li>
                <li>📥 <a href="{pdb_file.name}" download>Download PDB File</a></li>
            </ul>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/3dmol@2.1.0/build/3Dmol-min.js"></script>
    <script>
        let viewer;
        let rotationInterval = null;
        let currentStyle = 'cartoon';
        
        function initViewer() {{
            const viewerDiv = document.getElementById('viewer');
            
            if (typeof $3Dmol === 'undefined') {{
                setTimeout(initViewer, 100);
                return;
            }}
            
            try {{
                viewer = $3Dmol.createViewer(viewerDiv, {{
                    backgroundColor: "#f0f0f0",
                    defaultcolors: $3Dmol.elementColors.rasmol
                }});
                
                // Show loading
                const loadingDiv = document.createElement('div');
                loadingDiv.style.cssText = 'position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); font-size: 18px; color: #666;';
                loadingDiv.innerHTML = 'Loading 3D model...';
                viewerDiv.appendChild(loadingDiv);
                
                const pdbData = `{pdb_content_escaped}`;
                
                if (!pdbData || pdbData.trim().length === 0) {{
                    throw new Error('PDB data is empty');
                }}
                
                console.log('Loading PDB data, length:', pdbData.length);
                
                // Ensure END statement
                let cleanPdbData = pdbData.trim();
                if (!cleanPdbData.endsWith('END')) {{
                    cleanPdbData += '\\nEND\\n';
                }}
                
                viewer.addModel(cleanPdbData, "pdb");
                
                // Remove loading
                loadingDiv.remove();
                
                // Apply cartoon style by default
                viewer.setStyle({{}}, {{
                    cartoon: {{color: "spectrum"}}
                }});
                
                viewer.zoomTo();
                viewer.render();
                
                console.log('✅ Model loaded successfully!');
                
            }} catch (error) {{
                console.error('Error loading model:', error);
                viewerDiv.innerHTML = '<div style="padding: 40px; text-align: center; background: #fff3cd; border-radius: 5px;"><p style="color: #856404; font-weight: bold; font-size: 18px;">⚠️ Error loading 3D model</p><p style="color: #856404; margin-top: 10px;">' + error.message + '</p><p style="font-size: 0.9em; margin-top: 15px; color: #7f8c8d;">Check browser console (F12) for details</p></div>';
            }}
        }}
        
        function resetView() {{
            if (viewer) {{
                viewer.zoomTo();
                viewer.render();
            }}
        }}
        
        function toggleStyle(style) {{
            if (!viewer) return;
            
            currentStyle = style;
            viewer.setStyle({{}}, {{}});
            
            if (style === 'cartoon') {{
                viewer.setStyle({{}}, {{
                    cartoon: {{color: "spectrum"}}
                }});
            }} else if (style === 'stick') {{
                viewer.setStyle({{}}, {{
                    stick: {{radius: 0.3, color: "spectrum"}}
                }});
            }} else if (style === 'sphere') {{
                viewer.setStyle({{}}, {{
                    sphere: {{radius: 0.5, color: "spectrum"}}
                }});
            }} else if (style === 'surface') {{
                viewer.setStyle({{}}, {{
                    cartoon: {{color: "spectrum"}}
                }});
                viewer.addSurface($3Dmol.SurfaceType.VDW, {{
                    opacity: 0.7,
                    color: "spectrum"
                }});
            }}
            
            viewer.render();
        }}
        
        function toggleRotation() {{
            if (rotationInterval) {{
                clearInterval(rotationInterval);
                rotationInterval = null;
            }} else {{
                rotationInterval = setInterval(() => {{
                    if (viewer) {{
                        viewer.rotate(0.5, 'y');
                        viewer.render();
                    }}
                }}, 50);
            }}
        }}
        
        // Initialize when ready
        if (document.readyState === 'loading') {{
            document.addEventListener('DOMContentLoaded', initViewer);
        }} else {{
            initViewer();
        }}
    </script>
</body>
</html>
"""
    
    # Write HTML file
    output_path = Path(output_html_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    print(f"✅ Viewer created: {output_path}")
    return True

def main():
    """Main function"""
    print("\n" + "="*70)
    print("VSim Quick 3D Model Viewer Generator")
    print("="*70 + "\n")
    
    # Find available PDB files
    structures_dir = Path('results/structures')
    
    if not structures_dir.exists():
        print("❌ results/structures directory not found!")
        print("   Make sure you're in the Project-VSim directory")
        return
    
    virus_files = sorted(structures_dir.glob('virus_*.pdb'))
    
    if not virus_files:
        print("❌ No virus particle PDB files found!")
        return
    
    print("Available virus particle models:")
    for i, f in enumerate(virus_files, 1):
        size_kb = f.stat().st_size / 1024
        print(f"  {i}. {f.name} ({size_kb:.1f} KB)")
    
    # Use the most detailed model (largest file)
    best_file = max(virus_files, key=lambda f: f.stat().st_size)
    
    print(f"\n📦 Using: {best_file.name}")
    
    # Create viewer
    output_html = Path('results/viewer_3d.html')
    
    if create_viewer_html(best_file, output_html):
        print(f"\n✅ Success! Viewer created at: {output_html}")
        print(f"\n🌐 Opening in browser...")
        
        # Open in browser
        if sys.platform == 'darwin':  # macOS
            os.system(f'open "{output_html}"')
        elif sys.platform == 'linux':
            os.system(f'xdg-open "{output_html}"')
        elif sys.platform == 'win32':
            os.system(f'start "{output_html}"')
        else:
            print(f"   Open this file manually: {output_html.absolute()}")
        
        print("\n" + "="*70)
        print("✅ Done! The 3D model viewer should now be open in your browser.")
        print("="*70 + "\n")

if __name__ == '__main__':
    main()

