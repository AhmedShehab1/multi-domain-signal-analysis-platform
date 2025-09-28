#!/usr/bin/env python3
"""
Multi-Domain Signal Analysis Platform
Unified Flask API with blueprints for ECG analysis and Doppler sound generation
"""

from flask import Flask, render_template
from flask_cors import CORS

# Import blueprints
from blueprints.ecg import ecg_bp
from blueprints.doppler import doppler_bp

def create_app():
    """Create and configure the Flask application"""
    app = Flask(__name__)
    
    # Enable CORS for all domains
    CORS(app, resources={r"/*": {"origins": "*"}})
    
    # Register blueprints with API prefix
    app.register_blueprint(ecg_bp, url_prefix='/api/ecg')
    app.register_blueprint(doppler_bp, url_prefix='/api/doppler')
    
    # Main route
    @app.route('/')
    def index():
        """Main application index page"""
        return render_template('index.html')
    
    # API documentation route
    @app.route('/api')
    def api_info():
        """API documentation and status"""
        return {
            'status': 'operational',
            'version': '1.0.0',
            'endpoints': {
                '/api/ecg/classify': 'POST - Classify ECG signals',
                '/api/doppler/generate': 'POST - Generate Doppler effect sounds',
                '/api/doppler/info': 'GET - Doppler API information'
            }
        }
    
    return app

if __name__ == '__main__':
    app = create_app()
    print("🚀 Multi-Domain Signal Analysis Platform")
    print("========================================")
    print("Starting unified Flask server...")
    print("Open http://localhost:5001 in your browser")
    print("API documentation: http://localhost:5001/api")
    print()
    
    app.run(debug=True, host='0.0.0.0', port=5001)