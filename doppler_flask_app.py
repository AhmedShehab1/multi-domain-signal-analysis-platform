#!/usr/bin/env python3
"""
Doppler Effect Sound Generator Flask Web Application
Generates realistic car sounds with Doppler effect based on frequency and velocity
"""

import os
import tempfile
import numpy as np
from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
from scipy.io import wavfile
import threading
import atexit

app = Flask(__name__)
CORS(app)

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

def generate_doppler_car_sound(base_frequency=440, velocity=30, duration=3.0, sample_rate=44100):
    """
    Generate a car sound with Doppler effect
    
    Args:
        base_frequency (float): Base frequency of the car engine (Hz)
        velocity (float): Car velocity (m/s)
        duration (float): Duration of the sound (seconds)
        sample_rate (int): Audio sample rate
    
    Returns:
        tuple: (audio_data, sample_rate)
    """
    
    sound_speed = 343
    
    # Calculate Doppler-shifted frequency
    # Assuming observer is stationary and car is moving towards/away
    doppler_freq = base_frequency * sound_speed / (sound_speed + velocity)
    
    # Generate time array
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Create car engine sound (mix of frequencies + noise)
    engine_sound = (
        np.sin(2 * np.pi * doppler_freq * t) * 0.3 +
        np.sin(2 * np.pi * doppler_freq * 2 * t) * 0.2 +  # 2nd harmonic
        np.sin(2 * np.pi * doppler_freq * 3 * t) * 0.1 +  # 3rd harmonic
        np.random.normal(0, 0.05, len(t))  # Engine noise
    )
    
    # Apply envelope for realistic car pass-by effect
    envelope = np.exp(-((t - duration/2) / (duration/4))**2)
    engine_sound *= envelope
    
    # Normalize
    engine_sound = engine_sound / np.max(np.abs(engine_sound)) * 0.8
    
    return engine_sound, sample_rate

def generate_simple_tone(frequency=440, duration=2.0, sample_rate=44100):
    """Generate a simple sine wave tone"""
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio_data = 0.3 * np.sin(2 * np.pi * frequency * t)
    
    # Apply fade in/out to avoid clicks
    fade_samples = int(0.05 * sample_rate)  # 50ms fade
    audio_data[:fade_samples] *= np.linspace(0, 1, fade_samples)
    audio_data[-fade_samples:] *= np.linspace(1, 0, fade_samples)
    
    return audio_data, sample_rate

@app.route('/')
def index():
    """Serve the main web interface"""
    return render_template('index.html')

@app.route('/generate_sound', methods=['POST'])
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
        duration = max(0.5, min(10, duration))    # Limit duration
        
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
        
        # Convert to 16-bit integer format
        audio_int16 = (audio_data * 32767).astype(np.int16)
        
        # Write WAV file
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

@app.route('/api/info')
def api_info():
    """Return API information"""
    return jsonify({
        'status': 'running',
        'version': '1.0.0',
        'endpoints': {
            '/': 'Web interface',
            '/generate_sound': 'POST - Generate sound with Doppler effect',
            '/api/info': 'GET - API information'
        },
        'parameters': {
            'frequency': 'Base frequency in Hz (20-2000)',
            'velocity': 'Car velocity in m/s (1-100)',
            'duration': 'Sound duration in seconds (0.5-10)',
            'type': 'Sound type: "doppler" or "tone"'
        }
    })

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    print("🚗 Doppler Effect Sound Generator")
    print("================================")
    print("Starting Flask server...")
    print("Open http://localhost:5001 in your browser")
    print("API endpoint: POST /generate_sound")
    print("Parameters: frequency, velocity, duration, type")
    print()
    
    app.run(debug=True, host='0.0.0.0', port=5001)