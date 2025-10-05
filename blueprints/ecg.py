"""ECG analysis blueprint"""

from flask import Blueprint, request, jsonify, render_template
import numpy as np
import h5py
import io
import os
import time

# Create blueprint
ecg_bp = Blueprint('ecg', __name__)

# Global classifier instances for reuse
_classifier_single = None
_classifier_ensemble = None

def get_single_classifier():
    """Get or initialize the single ECG classifier"""
    global _classifier_single
    if _classifier_single is None:
        from utils.ecg_inference import ECGClassifier
        model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                 'model', 'model.hdf5')
        _classifier_single = ECGClassifier(model_path=model_path, use_ensemble=False)
    return _classifier_single

def get_ensemble_classifier():
    """Get or initialize the ensemble ECG classifier"""
    global _classifier_ensemble
    if _classifier_ensemble is None:
        from utils.ecg_inference import ECGClassifier
        _classifier_ensemble = ECGClassifier(use_ensemble=True)
    return _classifier_ensemble

def load_ecg_from_file(file):
    """Load ECG data from various file formats"""
    filename = file.filename.lower()
    print(f"Loading ECG file: {filename}")
    
    try:
        if filename.endswith('.hdf5') or filename.endswith('.h5'):
            # HDF5 file
            file_content = file.read()
            with h5py.File(io.BytesIO(file_content), 'r') as f:
                # Print the file structure for debugging
                print("HDF5 file structure:")
                print_hdf5_structure(f)
                
                # Try common HDF5 structures
                # Special handling for ecg_tracings.hdf5

                if 'tracings' in f:
                    print("Found 'tracings' dataset")
                    if len(f['tracings'].shape) == 3:
                        print(f"Shape: {f['tracings'].shape} - Taking first sample")
                        ecg_data = f['tracings'][0]
                    else:
                        print(f"Shape: {f['tracings'].shape}")
                        ecg_data = f['tracings'][:]
                elif 'ecg' in f:
                    print("Found 'ecg' dataset")
                    print(f"Shape: {f['ecg'].shape}")
                    ecg_data = f['ecg'][:]
                else:
                    # Use first dataset found
                    key = list(f.keys())[0]
                    print(f"Using first dataset: '{key}'")
                    print(f"Shape: {f[key].shape}")
                    ecg_data = f[key][:]
                    
                if filename == 'ecg_tracings.hdf5':
                    print("Processing PTB-XL dataset format...")
                    with h5py.File(io.BytesIO(file_content), 'r') as f:
                        # This file likely has many samples
                        # Take just the first complete ECG reading
                        if 'tracings' in f:
                            print(f"Found tracings with shape {f['tracings'].shape}")
                            # Select just the first sample if it's a 3D array
                            if len(f['tracings'].shape) == 3:
                                ecg_data = f['tracings'][0]
                            else:
                                ecg_data = f['tracings'][:]
                print(f"Loaded ECG data with shape: {ecg_data.shape}")
        
        # ... rest of the code for other formats ...
        
    except Exception as e:
        import traceback
        print(f"Error loading ECG file: {str(e)}")
        print(traceback.format_exc())
        raise
    
    return ecg_data

def print_hdf5_structure(hdf5_file, indent=0):
    """Helper function to print HDF5 file structure"""
    for key in hdf5_file.keys():
        print(" " * indent + f"- {key}: ", end="")
        if isinstance(hdf5_file[key], h5py.Dataset):
            print(f"Dataset {hdf5_file[key].shape}, {hdf5_file[key].dtype}")
        else:
            print("Group")
            print_hdf5_structure(hdf5_file[key], indent + 4)

@ecg_bp.route('/')
def index():
    """ECG classification interface"""
    return render_template('ecg_classifier.html')

@ecg_bp.route('/classify', methods=['POST'])
def classify():
    """Endpoint to classify ECG data"""
    try:
        start_time = time.time()
        
        use_ensemble = request.form.get('useEnsemble', 'false').lower() == 'true'
        model_type = request.form.get('modelType', 'main')
        threshold = float(request.form.get('threshold', 0.5))
        
        # Load ECG data
        if 'file' in request.files:
            file = request.files['file']
            ecg_data = load_ecg_from_file(file)
        elif request.is_json:
            data = request.get_json()
            ecg_data = np.array(data['ecg_data'])
        else:
            return jsonify({'error': 'No data provided'}), 400
        
        # Validate basic shape (at least 12 leads)
        if len(ecg_data.shape) != 2 or ecg_data.shape[1] != 12:
            return jsonify({
                'error': f'Invalid ECG shape. Expected (n_samples, 12), got {ecg_data.shape}'
            }), 400
        
        # Make prediction
        if use_ensemble:
            classifier = get_ensemble_classifier()
            result = classifier.predict(ecg_data, threshold=threshold)
        elif model_type != 'main':
            classifier = get_single_classifier()
            result = classifier.predict_with_specific_model(ecg_data, model_type)
        else:
            classifier = get_single_classifier()
            result = classifier.predict(ecg_data, threshold=threshold)
        
        processing_time = time.time() - start_time
        
        return jsonify({
            'success': True,
            'result': result,
            'model_info': {
                'type': 'ensemble' if use_ensemble else model_type,
                'threshold': threshold,
                'processing_time_ms': round(processing_time * 1000, 2)
            }
        })
    
    except Exception as e:
        import traceback
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@ecg_bp.route('/info', methods=['GET'])
def info():
    """Get information about available models"""
    classifier = get_single_classifier()
    return jsonify({
        'available_models': {
            'main': 'Main model trained on full dataset',
            'date_order': 'Model trained with date-ordered split',
            'individual_patients': 'Model trained with individual patient split',
            'normal_order': 'Model trained with normal-ordered split'
        },
        'ensemble': {
            'available': True,
            'num_models': 11,
            'description': 'Average predictions from 11 models with different seeds'
        },
        'abnormalities': classifier.class_names,
        'input_requirements': {
            'shape': '(4096, 12) or (n_samples, 12) with auto-padding',
            'sampling_rate': '400 Hz',
            'duration': '~10 seconds',
            'leads_order': 'DI, DII, DIII, AVR, AVL, AVF, V1, V2, V3, V4, V5, V6',
            'scale': 'Volts (will be multiplied by 1000 internally)'
        }
    })

@ecg_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    global _classifier_single, _classifier_ensemble
    return jsonify({
        'status': 'healthy',
        'models_loaded': {
            'single': _classifier_single is not None,
            'ensemble': _classifier_ensemble is not None
        }
    })
