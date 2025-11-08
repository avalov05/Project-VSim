"""
Web Interface Module
Flask-based web application for VSim
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import logging
import json

from src.core.config import Config
from src.core.logger import setup_logger
from src.genome.analyzer import GenomeAnalyzer
from src.structure.predictor import StructurePredictor
from src.environmental.analyzer import EnvironmentalAnalyzer
from src.cell_interaction.analyzer import CellInteractionAnalyzer
from src.cancer.analyzer import CancerAnalyzer
from src.ml.predictor import MLPredictor
from src.core.report import ReportGenerator

def create_app(config_path='config.yaml'):
    """Create Flask application"""
    app = Flask(__name__)
    CORS(app)
    
    # Load configuration
    config = Config(config_path)
    logger = setup_logger(config)
    
    # Initialize analyzers
    genome_analyzer = GenomeAnalyzer(config)
    structure_predictor = StructurePredictor(config)
    env_analyzer = EnvironmentalAnalyzer(config)
    cell_analyzer = CellInteractionAnalyzer(config)
    cancer_analyzer = CancerAnalyzer(config)
    ml_predictor = MLPredictor(config)
    report_generator = ReportGenerator(config)
    
    @app.route('/')
    def index():
        """Main page"""
        return render_template('index.html')
    
    @app.route('/api/analyze', methods=['POST'])
    def analyze():
        """API endpoint for genome analysis"""
        try:
            if 'file' not in request.files:
                return jsonify({'error': 'No file provided'}), 400
            
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            
            # Save uploaded file
            upload_dir = Path('data/raw')
            upload_dir.mkdir(parents=True, exist_ok=True)
            filepath = upload_dir / file.filename
            file.save(filepath)
            
            logger.info(f"Processing uploaded file: {filepath}")
            
            # Load genome
            genome_data = genome_analyzer.load_genome(str(filepath))
            
            # Run analysis
            genome_results = genome_analyzer.analyze(genome_data)
            structure_results = structure_predictor.predict(genome_results)
            env_results = env_analyzer.analyze(genome_results, structure_results)
            cell_results = cell_analyzer.analyze(genome_results, structure_results)
            cancer_results = cancer_analyzer.analyze(genome_results, structure_results, cell_results)
            ml_results = ml_predictor.predict_all(
                genome_results, structure_results, env_results,
                cell_results, cancer_results
            )
            
            # Generate report
            report = report_generator.generate(
                genome_results, structure_results, env_results,
                cell_results, cancer_results, ml_results
            )
            
            # Save results
            results_dir = Path('results')
            results_dir.mkdir(parents=True, exist_ok=True)
            report.save(results_dir / 'report.html')
            report.save_json(results_dir / 'results.json')
            
            return jsonify({
                'status': 'success',
                'results': report.data,
                'report_url': '/api/report'
            })
        
        except Exception as e:
            logger.error(f"Error during analysis: {str(e)}", exc_info=True)
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/report')
    def get_report():
        """Get generated report"""
        report_path = Path('results/report.html')
        if report_path.exists():
            return send_file(report_path)
        return jsonify({'error': 'Report not found'}), 404
    
    @app.route('/api/results')
    def get_results():
        """Get results JSON"""
        results_path = Path('results/results.json')
        if results_path.exists():
            with open(results_path, 'r') as f:
                return jsonify(json.load(f))
        return jsonify({'error': 'Results not found'}), 404
    
    @app.route('/api/status')
    def status():
        """API status check"""
        return jsonify({
            'status': 'online',
            'version': '1.0.0'
        })
    
    return app

# For running directly
if __name__ == '__main__':
    app = create_app()
    config = Config()
    port = config.get('web_interface.port', 8080)
    host = config.get('web_interface.host', '0.0.0.0')
    
    app.run(host=host, port=port, debug=config.get('web_interface.debug', False))

