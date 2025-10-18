"""
Voice Gender Classification Blueprint
"""

from flask import Blueprint, request, jsonify
import torch
import torch.nn.functional as F
import os
import tempfile
from werkzeug.utils import secure_filename

# Create blueprint first so we can use it even if model loading fails
voice_gender_bp = Blueprint('voice_gender', __name__)

# Set default MODEL_LOADED and MODEL_INFO
MODEL_LOADED = False
MODEL_INFO = {
    'name': 'ECAPA-TDNN Voice Gender Classifier',
    'status': 'not loaded',
    'error': None
}

try:
    # Try to import safetensors first
    import safetensors
    from models.voice_gender import ECAPA_gender

    # Try to load the model
    model = ECAPA_gender.from_pretrained("JaesungHuh/voice-gender-classifier")
    model.eval()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    MODEL_LOADED = True
    MODEL_INFO = {
        'name': 'ECAPA-TDNN Voice Gender Classifier',
        'architecture': 'ECAPA-TDNN',
        'source': 'JaesungHuh/voice-gender-classifier',
        'classes': ['male', 'female'],
        'status': 'loaded',
        'device': str(device)
    }
except ImportError as e:
    MODEL_INFO['status'] = 'failed to load'
    MODEL_INFO['error'] = f"Missing dependency: {str(e)}. Please install safetensors package."
    print(f"Error loading voice gender model: {e}")
except Exception as e:
    MODEL_INFO['status'] = 'failed to load'
    MODEL_INFO['error'] = str(e)
    print(f"Error loading voice gender model: {e}")

@voice_gender_bp.route('/classify', methods=['POST'])
def classify():
    """
    Classify voice gender from audio file
    """
    if not MODEL_LOADED:
        return jsonify({
            'status': 'error',
            'message': 'Model not loaded',
            'error': MODEL_INFO.get('error', 'Unknown error'),
            'solution': 'Please install the safetensors package with "pip install safetensors"'
        }), 500
    
    if 'file' not in request.files:
        return jsonify({
            'status': 'error',
            'message': 'No file part'
        }), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({
            'status': 'error',
            'message': 'No selected file'
        }), 400
    
    # Check file extension (basic validation)
    allowed_extensions = ['.wav', '.mp3', '.flac', '.ogg']
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_extensions:
        return jsonify({
            'status': 'error',
            'message': f'Unsupported file format. Supported formats: {", ".join(allowed_extensions)}'
        }), 400
    
    try:
        # Save file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            file.save(tmp.name)
            temp_path = tmp.name
        
        # Load audio and get model prediction with confidence
        audio = model.load_audio(temp_path).to(device)
        
        with torch.no_grad():
            # Get raw logits from the model
            output = model(audio)
            
            # Apply softmax to get probabilities
            probs = F.softmax(output, dim=1)[0]
            
            # Get prediction and confidence
            pred_idx = torch.argmax(probs).item()
            confidence = probs[pred_idx].item() * 100  # Convert to percentage
            gender = model.pred2gender[pred_idx]
            
            # Get probabilities for each class
            class_probs = {
                'male': probs[0].item() * 100,
                'female': probs[1].item() * 100
            }
        
        # Clean up temporary file
        os.unlink(temp_path)
        
        return jsonify({
            'status': 'success',
            'gender': gender,
            'confidence': round(confidence, 2),  # Round to 2 decimal places
            'probabilities': {
                'male': round(class_probs['male'], 2),
                'female': round(class_probs['female'], 2)
            }
        })
    
    except Exception as e:
        # Clean up temporary file if it exists
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.unlink(temp_path)
        
        return jsonify({
            'status': 'error',
            'message': 'Error processing audio',
            'error': str(e)
        }), 500

@voice_gender_bp.route('/info', methods=['GET'])
def info():
    """Get information about the voice gender classification model"""
    return jsonify(MODEL_INFO)

@voice_gender_bp.route('/health', methods=['GET'])
def health():
    """Check if the model is loaded and ready"""
    if MODEL_LOADED:
        return jsonify({
            'status': 'healthy',
            'message': 'Voice gender classification model is loaded and ready'
        })
    else:
        return jsonify({
            'status': 'unhealthy',
            'message': 'Voice gender classification model is not loaded',
            'error': MODEL_INFO.get('error', 'Unknown error'),
            'solution': 'Please install the safetensors package with "pip install safetensors"'
        }), 503