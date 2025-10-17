#!/usr/bin/env python3
"""
Multi-Domain Signal Analysis Platform
Unified Flask API with blueprints for ECG analysis and Doppler sound generation
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os

# Import blueprints
from blueprints.ecg import ecg_bp
from blueprints.doppler import doppler_bp
from blueprints.eeg import eeg_bp, setup_task_cleanup

from blueprints.resample import resample_bp
def create_app():
    """Create and configure the Flask application"""
    app = Flask(__name__)
    
    app.url_map.strict_slashes = False
    
    # Set file upload limits
    app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB max upload
    app.config['MAX_FORM_MEMORY_SIZE'] = 500 * 1024 * 1024
    app.config['UPLOAD_EXTENSIONS'] = ['.edf', '.csv', '.txt', '.set', '.hdf5', '.h5', '.npy']
    app.config['UPLOAD_PATH'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tmp')
    os.makedirs(app.config['UPLOAD_PATH'], exist_ok=True)
    
    # Ensure the request parsing is handled properly
    app.config['PRESERVE_CONTEXT_ON_EXCEPTION'] = False
    
    # Enable CORS for all domains
    CORS(app, 
         resources={r"/*": {"origins": "*"}},
         supports_credentials=True,
         allow_headers=["Content-Type", "ngrok-skip-browser-warning", "X-Requested-With", "Transfer-Encoding"],
         methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
    
    # Register blueprints with API prefix
    app.register_blueprint(ecg_bp, url_prefix='/api/ecg')
    app.register_blueprint(doppler_bp, url_prefix='/api/doppler')
    app.register_blueprint(eeg_bp, url_prefix='/api/eeg')
    app.register_blueprint(resample_bp, url_prefix='/api/resample')
    
    # Call setup functions
    setup_task_cleanup()
    
    # Main route
    @app.route('/')
    def index():
        """Main application index page"""
        return render_template('index.html')
    
    # Alias for ECG classifier frontend
    @app.route('/ecg')
    def ecg_frontend():
        """ECG classification frontend"""
        return render_template('ecg_classifier.html')

    @app.route('/resample_audio')
    def resample():
        return render_template('resample_audio.html')
    
    # API documentation route
    @app.route('/api')
    def api_info():
        """API documentation and status"""
        return {
            'status': 'operational',
            'version': '1.0.0',
            'endpoints': {
                '/api/ecg/classify': 'POST - Classify ECG signals',
                '/api/ecg/info': 'GET - Get ECG model information',
                '/api/ecg/health': 'GET - Check ECG model health',
                '/api/doppler/generate': 'POST - Generate Doppler effect sounds',
                '/api/doppler/info': 'GET - Doppler API information',
                '/api/eeg/classify': 'POST - Upload and classify EEG recordings',
                '/api/eeg/stream/<task_id>': 'GET - Stream EEG classification progress',
                '/api/eeg/results/<task_id>': 'GET - Get EEG classification results'
            }
        }
    
    # Handle chunked transfers
    @app.before_request
    def handle_chunking():
        """Handle chunked transfers"""
        transfer_encoding = request.headers.get("Transfer-Encoding", None)
        if transfer_encoding == "chunked":
            request.environ["wsgi.input_terminated"] = True
    
    # Handle file size errors
    @app.errorhandler(413)
    def handle_file_too_large(_):
        """Handle file too large errors"""
        return jsonify({"status": "failed", "message": "File too large"}), 413
    
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