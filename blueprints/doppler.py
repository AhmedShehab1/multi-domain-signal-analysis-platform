"""Doppler Effect Sound Generation Blueprint"""

import os
import tempfile
import atexit
import uuid
from flask import Blueprint, jsonify, request, send_file, render_template, current_app
import numpy as np
from werkzeug.utils import secure_filename
from utils.audio import generate_doppler_car_sound, generate_simple_tone, DopplerVelocityDetector
import sys

doppler_bp = Blueprint('doppler', __name__)

# Store temporary files to clean up later
temp_files = []

def cleanup_temp_files():
    """Clean up temporary files on exit"""
    for temp_file in temp_files:
        try:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
        except:
            pass

atexit.register(cleanup_temp_files)

@doppler_bp.route('/')
def doppler_index():
    """Serve the Doppler effect interface"""
    return render_template('doppler.html')

@doppler_bp.route('/analysis', methods=['POST'])
def analyze_doppler():
    """
    API endpoint to analyze audio files for Doppler effect
    
    Expects a WAV audio file in the 'file' field of form data
    Optional parameters:
    - base_frequency: float - The known base frequency (Hz)
    - speed_of_sound: float - Speed of sound in m/s
    - min_frequency: float - Minimum frequency to track (Hz)
    - max_frequency: float - Maximum frequency to track (Hz)
    
    Returns JSON with:
    - frequency: str - Observed frequency range
    - velocity: str - Calculated velocity range (in km/h)
    - details: dict - Additional details about the analysis
    """
    try:
        # Check if file is present in the request
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Validate file type
        if not file.filename.lower().endswith(('.wav', '.mp3', '.ogg', '.flac')):
            return jsonify({"error": "Only audio files (WAV, MP3, OGG, FLAC) are supported"}), 400
        
        # Get parameters from request
        base_frequency = float(request.form.get('base_frequency', 800.0))
        speed_of_sound = float(request.form.get('speed_of_sound', 343.0))
        
        # Frequency range for tracking (default: ±20% of base frequency)
        min_freq = float(request.form.get('min_frequency', base_frequency * 0.8))
        max_freq = float(request.form.get('max_frequency', base_frequency * 1.2))
        
        # Save file temporarily
        filename = secure_filename(file.filename)
        temp_dir = os.path.join(current_app.config.get('UPLOAD_PATH', 'tmp'))
        os.makedirs(temp_dir, exist_ok=True)
        
        temp_file_path = os.path.join(temp_dir, f"{uuid.uuid4()}_{filename}")
        file.save(temp_file_path)
        
        # Redirect stdout to capture printed output
        stdout_backup = sys.stdout
        sys.stdout = io.StringIO()
        
        try:
            # Initialize detector with specified speed of sound
            detector = DopplerVelocityDetector(speed_of_sound=speed_of_sound)
            
            # Load and process audio
            detector.load_audio(temp_file_path)
            detector.compute_spectrogram(window_size=16384, overlap=14000)
            detector.track_frequency(freq_min=min_freq, freq_max=max_freq, smoothing_window=5)
            
            # Calculate velocity using the provided base frequency
            detector.calculate_velocity(f0=base_frequency)
            
            # Get analysis output
            log_output = sys.stdout.getvalue()
            
            # Extract key results
            freq_min = np.nanmin(detector.observed_frequencies)
            freq_max = np.nanmax(detector.observed_frequencies)
            vel_min = np.nanmin(detector.velocities)
            vel_max = np.nanmax(detector.velocities)
            
            # Prepare the response
            result = {
                "success": True,
                "frequency": f"{freq_min:.1f} - {freq_max:.1f} Hz",
                "velocity": f"{vel_min * 3.6:.1f} to {vel_max * 3.6:.1f} km/h",
                "details": {
                    "frequency_min": float(freq_min),
                    "frequency_max": float(freq_max),
                    "velocity_min_ms": float(vel_min),
                    "velocity_max_ms": float(vel_max),
                    "velocity_min_kmh": float(vel_min * 3.6),
                    "velocity_max_kmh": float(vel_max * 3.6),
                    "base_frequency": float(base_frequency),
                    "speed_of_sound": float(speed_of_sound),
                    "audio_duration": len(detector.audio) / detector.fs,
                    "sample_rate": int(detector.fs),
                    "log": log_output
                }
            }
            
            return jsonify(result)
            
        finally:
            # Restore stdout
            sys.stdout = stdout_backup
            
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
                
    except Exception as e:
        import traceback
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500
    
@doppler_bp.route('/generate', methods=['POST'])
def generate_sound():
    """Generate and return a sound file with Doppler effect"""
    try:
        data = request.json
        frequency = float(data.get('frequency', 440))
        velocity = float(data.get('velocity', 30))
        duration = float(data.get('duration', 3.0))
        sound_type = data.get('type', 'doppler')  # 'doppler' or 'tone'
        
        # Validate inputs
        frequency = max(20, min(2000, frequency))  # Limit frequency range
        velocity = max(1, min(100, velocity))      # Limit velocity range
        duration = max(0.5, min(10, duration))     # Limit duration
        
        # Generate sound based on type
        if sound_type == 'tone':
            audio_data, sample_rate = generate_simple_tone(frequency, duration)
            filename = f'tone_{frequency}Hz.wav'
        else:
            audio_data, sample_rate = generate_doppler_car_sound(frequency, velocity, duration)
            filename = f'car_doppler_{frequency}Hz_{velocity}ms.wav'
        
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        temp_files.append(temp_file.name)  # Track for cleanup
        
        # Save the audio file (moved to audio utils)
        from scipy.io import wavfile
        audio_int16 = (audio_data * 32767).astype(np.int16)
        wavfile.write(temp_file.name, sample_rate, audio_int16)
        temp_file.close()
        
        # Clean up old temp files (keep only last 10)
        if len(temp_files) > 10:
            old_file = temp_files.pop(0)
            try:
                if os.path.exists(old_file):
                    os.unlink(old_file)
            except:
                pass
        
        return send_file(
            temp_file.name, 
            as_attachment=True, 
            download_name=filename,
            mimetype='audio/wav'
        )
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@doppler_bp.route('/info')
def doppler_info():
    """Return Doppler API information"""
    return jsonify({
        'status': 'running',
        'version': '1.0.0',
        'endpoints': {
            '/api/doppler/': 'Web interface',
            '/api/doppler/generate': 'POST - Generate sound with Doppler effect',
            '/api/doppler/info': 'GET - API information'
        },
        'parameters': {
            'frequency': 'Base frequency in Hz (20-2000)',
            'velocity': 'Car velocity in m/s (1-100)',
            'duration': 'Sound duration in seconds (0.5-10)',
            'type': 'Sound type: "doppler" or "tone"'
        }
    })