"""Doppler Effect Sound Generation Blueprint"""

import os
import tempfile
import atexit

from flask import Blueprint, jsonify, request, send_file, render_template
import numpy as np

from utils.audio import generate_doppler_car_sound, generate_simple_tone

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