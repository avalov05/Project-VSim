"""
Report Generation Module
"""

import json
from pathlib import Path
from typing import Dict, Any
from datetime import datetime
import html as html_module

class ReportGenerator:
    """Generate comprehensive analysis reports"""
    
    def __init__(self, config):
        self.config = config
    
    def generate(self, genome_results: Dict, structure_results: Dict, 
                 env_results: Dict, cell_results: Dict, 
                 cancer_results: Dict, ml_results: Dict) -> 'Report':
        """Generate comprehensive report from all analysis results"""
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'genome_analysis': genome_results,
            'structure_prediction': structure_results,
            'environmental_dynamics': env_results,
            'cell_interactions': cell_results,
            'cancer_analysis': cancer_results,
            'ml_predictions': ml_results,
            'summary': self._generate_summary(genome_results, ml_results)
        }
        
        return Report(report_data, self.config)
    
    def _generate_summary(self, genome_results: Dict, ml_results: Dict) -> Dict:
        """Generate executive summary"""
        risk_assessment = ml_results.get('risk_assessment', {})
        if isinstance(risk_assessment, dict):
            risk_level = risk_assessment.get('risk_level', 'Unknown')
        else:
            risk_level = risk_assessment
        
        return {
            'synthesis_feasibility': genome_results.get('synthesis_feasibility', 0.0),
            'overall_confidence': ml_results.get('overall_confidence', 0.0),
            'risk_assessment': risk_level,
            'key_findings': ml_results.get('key_findings', [])
        }

class Report:
    """Report container"""
    
    def __init__(self, data: Dict[str, Any], config):
        self.data = data
        self.config = config
    
    def save(self, filepath: Path):
        """Save report as HTML"""
        html_content = self._generate_html()
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            f.write(html_content)
    
    def save_json(self, filepath: Path):
        """Save report as JSON"""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.data, f, indent=2, default=str)
    
    def _generate_html(self) -> str:
        """Generate HTML report"""
        data = self.data
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>VSim Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        .summary {{ background: #ecf0f1; padding: 20px; border-radius: 5px; margin: 20px 0; }}
        .metric {{ display: inline-block; margin: 10px 20px; padding: 15px; background: white; border-radius: 5px; }}
        .metric-label {{ font-weight: bold; color: #7f8c8d; }}
        .metric-value {{ font-size: 24px; color: #2c3e50; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #3498db; color: white; }}
        .confidence-high {{ color: #27ae60; font-weight: bold; }}
        .confidence-medium {{ color: #f39c12; font-weight: bold; }}
        .confidence-low {{ color: #e74c3c; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>VSim Analysis Report</h1>
        <p><strong>Generated:</strong> {data['timestamp']}</p>
        
        <div class="summary">
            <h2>Executive Summary</h2>
            <div class="metric">
                <div class="metric-label">Synthesis Feasibility</div>
                <div class="metric-value">{data['summary']['synthesis_feasibility']:.2%}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Overall Confidence</div>
                <div class="metric-value">{data['summary']['overall_confidence']:.2%}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Risk Assessment</div>
                <div class="metric-value">{data['summary'].get('risk_assessment', 'Unknown')}</div>
            </div>
        </div>
        
        <h2>Genome Analysis</h2>
        <table>
            <tr><th>Property</th><th>Value</th></tr>
            <tr><td>Genome Length</td><td>{data['genome_analysis'].get('length', 'N/A')} bp</td></tr>
            <tr><td>GC Content</td><td>{data['genome_analysis'].get('gc_content', 0):.2%}</td></tr>
            <tr><td>ORFs Detected</td><td>{len(data['genome_analysis'].get('orfs', []))}</td></tr>
            <tr><td>Proteins Predicted</td><td>{len(data['genome_analysis'].get('proteins', []))}</td></tr>
        </table>
        
        <h2>Virus 3D Model</h2>
        {self._generate_virus_model_viewer(data['structure_prediction'])}
        
        <h2>Structure Prediction</h2>
        <p>3D structures predicted for {len(data['structure_prediction'].get('structures', []))} proteins</p>
        
        <div class="structure-viewers">
            {self._generate_structure_viewers(data['structure_prediction'])}
        </div>
        
        <h2>Environmental Dynamics</h2>
        <table>
            <tr><th>Property</th><th>Value</th></tr>
            <tr><td>Thermal Stability</td><td>{data['environmental_dynamics'].get('thermal_stability', {}).get('stability_score', 'N/A')}</td></tr>
            <tr><td>pH Stability Range</td><td>{data['environmental_dynamics'].get('ph_stability', {}).get('ph_range', 'N/A')}</td></tr>
            <tr><td>Survival Time (no host)</td><td>{data['environmental_dynamics'].get('environmental_survival', {}).get('average_survival_time_hours', 'N/A')} hours</td></tr>
        </table>
        
        <h2>Cell Interactions</h2>
        <p>Analysis of receptor binding and cell entry mechanisms completed.</p>
        
        <h2>Cancer Cell Analysis</h2>
        <p>Oncolytic potential assessed for multiple cancer cell types.</p>
        
        <h2>ML Predictions</h2>
        <p>Machine learning models applied for comprehensive behavior prediction.</p>
        
        <h2>Key Findings</h2>
        <ul>
"""
        for finding in data['summary'].get('key_findings', []):
            html += f"            <li>{html_module.escape(str(finding))}</li>\n"
        
        html += """
        </ul>
    </div>
</body>
</html>
"""
        return html
    
    def _generate_structure_viewers(self, structure_results: Dict) -> str:
        """Generate 3D structure viewers"""
        structures = structure_results.get('structures', [])
        
        if not structures:
            return "<p>No structures available for visualization.</p>"
        
        viewers_html = ""
        
        for i, structure in enumerate(structures):
            pdb_file = structure.get('pdb_file')
            sequence = structure.get('sequence', '')
            protein_id = structure.get('protein_id', i)
            
            if pdb_file and Path(pdb_file).exists():
                # Read PDB content
                try:
                    with open(pdb_file, 'r') as f:
                        pdb_content = f.read()
                    
                    # Escape for HTML
                    pdb_content_escaped = html_module.escape(pdb_content)
                    
                    viewers_html += f"""
                    <div class="structure-viewer-container" style="margin: 20px 0; border: 1px solid #ddd; border-radius: 10px; padding: 20px;">
                        <h3>Protein {protein_id} - 3D Structure</h3>
                        <p><strong>Sequence Length:</strong> {len(sequence)} residues</p>
                        <p><strong>Confidence:</strong> {structure.get('confidence', 0):.2%}</p>
                        <p><strong>Method:</strong> {structure.get('method', 'N/A')}</p>
                        
                        <div id="viewer_{i}" style="width: 100%; height: 400px; background: #f0f0f0; border-radius: 5px;"></div>
                        
                        <script src="https://cdn.jsdelivr.net/npm/3dmol@2.1.0/build/3Dmol-min.js"></script>
                        <script>
                            (function() {{
                                var viewer = $3Dmol.createViewer(document.getElementById("viewer_{i}"), {{
                                    backgroundColor: "white"
                                }});
                                
                                var pdbData = `{pdb_content_escaped}`;
                                
                                viewer.addModel(pdbData, "pdb");
                                viewer.setStyle({{}}, {{
                                    cartoon: {{color: "spectrum"}},
                                    stick: {{radius: 0.3}}
                                }});
                                viewer.zoomTo();
                                viewer.render();
                            }})();
                        </script>
                        
                        <p style="margin-top: 10px;">
                            <a href="{pdb_file}" download>Download PDB File</a>
                        </p>
                    </div>
                    """
                except Exception as e:
                    viewers_html += f"""
                    <div class="structure-viewer-container" style="margin: 20px 0; padding: 20px; border: 1px solid #ddd;">
                        <h3>Protein {protein_id}</h3>
                        <p>Structure file available but visualization failed: {str(e)}</p>
                        <p><a href="{pdb_file}" download>Download PDB File</a></p>
                    </div>
                    """
            else:
                # No PDB file, show structure info
                viewers_html += f"""
                <div class="structure-viewer-container" style="margin: 20px 0; padding: 20px; border: 1px solid #ddd;">
                    <h3>Protein {protein_id}</h3>
                    <p><strong>Sequence:</strong> {sequence[:50]}...</p>
                    <p><strong>Length:</strong> {len(sequence)} residues</p>
                    <p><strong>Confidence:</strong> {structure.get('confidence', 0):.2%}</p>
                    <p><strong>Method:</strong> {structure.get('method', 'N/A')}</p>
                    <p><em>3D structure prediction available. Install ESMFold for real structure generation.</em></p>
                </div>
                """
        
        return viewers_html
    
    def _generate_virus_model_viewer(self, structure_results: Dict) -> str:
        """Generate 3D viewer for complete virus particle"""
        virus_particle = structure_results.get('virus_particle', {})
        
        # Find model file - check multiple locations
        model_file = None
        
        # 1. Prefer assembled from genome model
        assembled_model = Path('results/structures/virus_assembled_from_genome.pdb')
        if assembled_model.exists():
            model_file = str(assembled_model)
        
        # 2. Check virus_particle data
        if not model_file and virus_particle:
            model_file = virus_particle.get('model_file')
        
        # 3. Check for realistic large RNA virus model
        if not model_file or not Path(model_file).exists():
            realistic_model = Path('results/structures/virus_large_rna_realistic.pdb')
            if realistic_model.exists():
                model_file = str(realistic_model)
        
        # 4. Check for detailed models
        if not model_file or not Path(model_file).exists():
            structures_dir = Path('results/structures')
            detailed_models = list(structures_dir.glob('virus_particle_detailed_*.pdb'))
            if detailed_models:
                model_file = str(detailed_models[0])
        
        # Get metadata from virus_particle or extract from PDB file
        diameter = virus_particle.get('estimated_diameter_nm', 0) if virus_particle else 0
        shape = virus_particle.get('capsid_shape', 'unknown') if virus_particle else 'unknown'
        virus_type = virus_particle.get('virus_type', 'unknown') if virus_particle else 'unknown'
        genome_length = virus_particle.get('genome_length', 'N/A') if virus_particle else 'N/A'
        protein_count = virus_particle.get('protein_count', 'N/A') if virus_particle else 'N/A'
        symmetry = virus_particle.get('symmetry', '') if virus_particle else ''
        
        if not model_file:
            return "<p>Virus particle model not available. Generating model...</p>"
        
        if model_file and Path(model_file).exists():
            # Try to extract metadata from PDB REMARK lines if not available
            if diameter == 0 or shape == 'unknown' or virus_type == 'unknown':
                try:
                    with open(model_file, 'r') as f:
                        for line in f:
                            if line.startswith('REMARK'):
                                if 'Diameter:' in line and diameter == 0:
                                    try:
                                        diameter = float(line.split('Diameter:')[1].split('nm')[0].strip())
                                    except:
                                        pass
                                if ('Emergent Shape:' in line or 'Capsid Shape:' in line) and shape == 'unknown':
                                    try:
                                        if 'Emergent Shape:' in line:
                                            shape = line.split('Emergent Shape:')[1].strip()
                                        else:
                                            shape = line.split('Capsid Shape:')[1].strip()
                                    except:
                                        pass
                                if 'Emergent Symmetry:' in line and not symmetry:
                                    try:
                                        symmetry = line.split('Emergent Symmetry:')[1].strip()
                                    except:
                                        pass
                                if 'Virus Type:' in line and virus_type == 'unknown':
                                    try:
                                        virus_type = line.split('Virus Type:')[1].strip()
                                    except:
                                        pass
                except:
                    pass
            try:
                # Read PDB file - limit size for browser performance
                pdb_file_path = Path(model_file)
                file_size_mb = pdb_file_path.stat().st_size / (1024 * 1024)
                
                if file_size_mb > 10:
                    # File too large, create simplified version
                    self.logger.warning(f"PDB file too large ({file_size_mb:.1f}MB), creating simplified version")
                    simplified_file = self._create_simplified_pdb(model_file)
                    if simplified_file:
                        model_file = str(simplified_file)
                
                with open(model_file, 'r') as f:
                    pdb_content = f.read()
                
                # Ensure PDB file ends with END if it doesn't already
                pdb_content = pdb_content.rstrip()
                if not pdb_content.endswith('END'):
                    pdb_content += '\nEND\n'
                
                # Limit content size for browser
                lines = pdb_content.split('\n')
                if len(lines) > 50000:
                    # Take every Nth line to reduce size
                    step = len(lines) // 50000 + 1
                    pdb_content = '\n'.join(lines[::step])
                    # Ensure END is still present after reduction
                    if not pdb_content.rstrip().endswith('END'):
                        pdb_content = pdb_content.rstrip() + '\nEND\n'
                
                pdb_content_escaped = html_module.escape(pdb_content)
                
                return f"""
                <div class="virus-model-container" style="margin: 20px 0; border: 2px solid #3498db; border-radius: 10px; padding: 20px; background: #f8f9fa;">
                    <h3>🧬 Complete Virus Particle - 3D Model</h3>
                    <div style="display: flex; gap: 20px; margin-bottom: 15px;">
                        <div>
                            <p><strong>Estimated Diameter:</strong> {diameter} nm</p>
                            <p><strong>Emergent Shape:</strong> {shape.capitalize()}</p>
                            {f'<p><strong>Emergent Symmetry:</strong> {symmetry.capitalize()}</p>' if symmetry else ''}
                            <p><strong>Virus Type:</strong> {virus_type.replace('_', ' ').title()}</p>
                            <p><strong>Genome Length:</strong> {genome_length} bp</p>
                            <p><strong>Protein Count:</strong> {protein_count}</p>
                            <p><em style="color: #666; font-size: 0.9em;">Geometry emerged naturally from physics simulation</em></p>
                        </div>
                    </div>
                    
                    <div id="virus_viewer" style="width: 100%; height: 600px; background: white; border-radius: 5px; border: 1px solid #ddd; position: relative;"></div>
                    
                    <script src="https://cdn.jsdelivr.net/npm/3dmol@2.1.0/build/3Dmol-min.js"></script>
                    <script>
                        (function() {{
                            // Wait for DOM and 3Dmol.js to be ready
                            function initVirusViewer() {{
                                var viewerDiv = document.getElementById("virus_viewer");
                                if (!viewerDiv) {{
                                    setTimeout(initVirusViewer, 100);
                                    return;
                                }}
                                
                                if (typeof $3Dmol === 'undefined') {{
                                    setTimeout(initVirusViewer, 100);
                                    return;
                                }}
                                
                                try {{
                                    var viewer = $3Dmol.createViewer(viewerDiv, {{
                                        backgroundColor: "#f0f0f0",
                                        defaultcolors: $3Dmol.elementColors.rasmol
                                    }});
                                    
                                    // Show loading message
                                    var loadingDiv = document.createElement('div');
                                    loadingDiv.id = 'loading_message';
                                    loadingDiv.style.cssText = 'position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); font-size: 18px; color: #666; z-index: 10; pointer-events: none;';
                                    loadingDiv.innerHTML = 'Loading 3D model...';
                                    viewerDiv.appendChild(loadingDiv);
                                    
                                    var pdbData = `{pdb_content_escaped}`;
                                    
                                    if (!pdbData || pdbData.trim().length === 0) {{
                                        throw new Error('PDB data is empty');
                                    }}
                                    
                                    console.log('Loading PDB data, length:', pdbData.length);
                                    console.log('PDB data ends with:', pdbData.slice(-50));
                                    
                                    // Ensure END statement exists
                                    if (!pdbData.trim().endsWith('END')) {{
                                        pdbData = pdbData.trim() + '\\nEND\\n';
                                        console.log('Added END statement');
                                    }}
                                    
                                    viewer.addModel(pdbData, "pdb");
                                    
                                    // Remove loading message
                                    var loadingMsg = document.getElementById('loading_message');
                                    if (loadingMsg) loadingMsg.remove();
                                
                                // Style different protein chains differently
                                var chains = {{}};
                                var lines = pdbData.split('\\n');
                                lines.forEach(function(line) {{
                                    if (line.startsWith('ATOM')) {{
                                        var chain = line.charAt(21);
                                        if (!chains[chain]) chains[chain] = true;
                                    }}
                                }});
                                
                                var chainArray = Object.keys(chains).sort();
                                var colors = ['blue', 'red', 'green', 'orange', 'purple', 'cyan', 'yellow', 'magenta', 
                                             'pink', 'lime', 'teal', 'maroon', 'navy', 'olive', 'gray', 'silver'];
                                
                                // Color spikes (S chains) red, membrane (M) blue, envelope (E) green, nucleocapsid (N) yellow
                                chainArray.forEach(function(chain, idx) {{
                                    var color = colors[idx % colors.length];
                                    if (chain === 'S') color = 'red';      // Spike proteins
                                    else if (chain === 'M') color = 'blue'; // Membrane
                                    else if (chain === 'E') color = 'green'; // Envelope
                                    else if (chain === 'N') color = 'yellow'; // Nucleocapsid
                                    
                                    viewer.setStyle({{chain: chain}}, {{
                                        cartoon: {{
                                            color: color,
                                            thickness: 0.5,
                                            arrows: true
                                        }},
                                        stick: {{
                                            radius: 0.25, 
                                            color: color
                                        }}
                                    }});
                                }});
                                
                                // Default style
                                viewer.setStyle({{}}, {{
                                    cartoon: {{thickness: 0.4}},
                                    stick: {{radius: 0.15}}
                                }});
                                
                                viewer.zoomTo();
                                viewer.render();
                                
                                // Smooth rotation animation
                                var rotate = function() {{
                                    viewer.rotate(0.3, 'y');
                                    viewer.render();
                                }};
                                setInterval(rotate, 30);
                                
                                // Enable mouse controls
                                viewer.enableFog(true);
                                
                                // Add controls for visualization style
                                var styleSelect = document.createElement('select');
                                styleSelect.innerHTML = '<option value="cartoon">Cartoon</option><option value="stick">Stick</option><option value="sphere">Sphere</option><option value="surface">Surface</option>';
                                styleSelect.style.marginTop = '10px';
                                styleSelect.style.marginBottom = '10px';
                                styleSelect.style.padding = '5px';
                                styleSelect.onchange = function() {{
                                    var style = this.value;
                                    chainArray.forEach(function(chain) {{
                                        var idx = chainArray.indexOf(chain);
                                        var color = colors[idx % colors.length];
                                        if (chain === 'S') color = 'red';
                                        else if (chain === 'M') color = 'blue';
                                        else if (chain === 'E') color = 'green';
                                        else if (chain === 'N') color = 'yellow';
                                        
                                        if (style === 'cartoon') {{
                                            viewer.setStyle({{chain: chain}}, {{
                                                cartoon: {{color: color}},
                                                stick: {{radius: 0.2}}
                                            }});
                                        }} else if (style === 'stick') {{
                                            viewer.setStyle({{chain: chain}}, {{
                                                stick: {{radius: 0.3, color: color}}
                                            }});
                                        }} else if (style === 'sphere') {{
                                            viewer.setStyle({{chain: chain}}, {{
                                                sphere: {{radius: 1.0, color: color}}
                                            }});
                                        }} else if (style === 'surface') {{
                                            viewer.setStyle({{chain: chain}}, {{
                                                cartoon: {{color: color}}
                                            }});
                                        }}
                                    }});
                                    viewer.render();
                                }};
                                document.getElementById("virus_viewer").parentElement.appendChild(styleSelect);
                                
                                }} catch (error) {{
                                    console.error('Error loading 3D model:', error);
                                    console.error('Error details:', error.message, error.stack);
                                    var loadingMsg = document.getElementById('loading_message');
                                    if (loadingMsg) {{
                                        loadingMsg.innerHTML = 'Error loading 3D model: ' + error.message + '. Check browser console (F12) for details.';
                                        loadingMsg.style.color = 'red';
                                    }}
                                }}
                            }}
                            
                            // Start initialization when DOM is ready
                            if (document.readyState === 'loading') {{
                                document.addEventListener('DOMContentLoaded', initVirusViewer);
                            }} else {{
                                initVirusViewer();
                            }}
                        }})();
                    </script>
                    
                    <div style="margin-top: 10px; padding: 10px; background: #e8f4f8; border-radius: 5px;">
                        <p style="margin: 0; font-size: 0.9em;"><strong>Visualization Controls:</strong></p>
                        <ul style="margin: 5px 0; padding-left: 20px; font-size: 0.85em;">
                            <li>Drag to rotate</li>
                            <li>Scroll to zoom</li>
                            <li>Each color represents a different protein chain</li>
                            <li>Shows individual protein structures in capsid arrangement</li>
                        </ul>
                    </div>
                    
                    <div style="margin-top: 15px;">
                        <button onclick="document.getElementById('virus_viewer').style.height='500px'; document.getElementById('virus_viewer').style.width='100%';" style="padding: 8px 15px; margin-right: 10px; background: #3498db; color: white; border: none; border-radius: 5px; cursor: pointer;">Reset View</button>
                        <a href="{model_file}" download style="padding: 8px 15px; background: #27ae60; color: white; text-decoration: none; border-radius: 5px; display: inline-block;">Download Virus Model PDB</a>
                    </div>
                    <p style="margin-top: 10px; font-size: 0.9em; color: #7f8c8d;">
                        <em>This is a simplified 3D representation based on genome analysis. Actual virus structure may vary.</em>
                    </p>
                </div>
                """
            except Exception as e:
                return f"""
                <div class="virus-model-container" style="margin: 20px 0; padding: 20px; border: 1px solid #ddd;">
                    <h3>Virus Particle Model</h3>
                    <p>Error loading virus model: {str(e)}</p>
                    <p><strong>Estimated Diameter:</strong> {diameter} nm</p>
                    <p><strong>Capsid Shape:</strong> {shape}</p>
                </div>
                """
        else:
            return f"""
            <div class="virus-model-container" style="margin: 20px 0; padding: 20px; border: 1px solid #ddd;">
                <h3>Virus Particle Model</h3>
                <p><strong>Estimated Diameter:</strong> {diameter} nm</p>
                <p><strong>Capsid Shape:</strong> {shape}</p>
                <p><em>3D model generation in progress...</em></p>
            </div>
            """
    
