"""Audio resampling service blueprint"""

import os
import io
import uuid
import numpy as np
from flask import Blueprint, request, jsonify, send_file, current_app
from werkzeug.utils import secure_filename
import traceback

# Audio processing imports
try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False
    print("Warning: soundfile not available, will use scipy for I/O")

try:
    import samplerate
    LIBSAMPLERATE_AVAILABLE = True
except ImportError:
    LIBSAMPLERATE_AVAILABLE = False
    print("Warning: samplerate not available, will use scipy for resampling")

from scipy import signal
from scipy.io import wavfile
import subprocess
import math

# Create blueprint
resample_bp = Blueprint('resample', __name__)

# Configuration
MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50 MB
ALLOWED_EXTENSIONS = {'.wav', '.mp3', '.ogg', '.flac', '.m4a', '.aac'}

class AudioResampler:
    """High-quality audio resampling with anti-aliasing"""
    
    @staticmethod
    def load_audio(file_path):
        """Load audio file using best available method"""
        if SOUNDFILE_AVAILABLE:
            try:
                data, sr = sf.read(file_path, dtype='float32')
                return data, sr
            except Exception as e:
                print(f"soundfile failed: {e}, trying scipy")
        
        # Fallback to scipy
        try:
            sr, data = wavfile.read(file_path)
            # Convert to float32
            if data.dtype == np.int16:
                data = data.astype(np.float32) / 32768.0
            elif data.dtype == np.int32:
                data = data.astype(np.float32) / 2147483648.0
            elif data.dtype == np.uint8:
                data = (data.astype(np.float32) - 128) / 128.0
            return data, sr
        except Exception as e:
            print(f"scipy failed: {e}, trying ffmpeg")
        
        # Fallback to ffmpeg
        return AudioResampler.load_with_ffmpeg(file_path)
    
    @staticmethod
    def load_with_ffmpeg(file_path):
        """Load audio using ffmpeg as last resort"""
        try:
            cmd = [
                'ffmpeg', '-i', file_path,
                '-f', 'f32le',  # 32-bit float PCM
                '-acodec', 'pcm_f32le',
                '-ar', '44100',  # Will be resampled later if needed
                '-ac', '2',  # Stereo
                '-'
            ]
            
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )
            
            audio_data = np.frombuffer(result.stdout, dtype=np.float32)
            # Reshape to stereo
            audio_data = audio_data.reshape(-1, 2)
            
            return audio_data, 44100
        except Exception as e:
            raise RuntimeError(f"All audio loading methods failed: {e}")
    
    @staticmethod
    def estimate_fmax(audio, sr, energy_threshold=0.995):
        """
        Estimate maximum frequency using cumulative energy method
        
        Args:
            audio: Audio data (can be multichannel)
            sr: Sample rate
            energy_threshold: Cumulative energy threshold (default 99.5%)
        
        Returns:
            fmax_estimate: Estimated maximum frequency in Hz
        """
        # Convert to mono if multichannel
        if len(audio.shape) > 1:
            audio_mono = np.mean(audio, axis=1)
        else:
            audio_mono = audio
        
        # Apply Hanning window
        window = np.hanning(len(audio_mono))
        audio_windowed = audio_mono * window
        
        # Compute FFT
        fft = np.fft.rfft(audio_windowed)
        power = np.abs(fft) ** 2
        
        # Frequency bins
        freqs = np.fft.rfftfreq(len(audio_mono), 1/sr)
        
        # Cumulative energy
        cumulative_energy = np.cumsum(power)
        total_energy = cumulative_energy[-1]
        
        # Find frequency where cumulative energy reaches threshold
        threshold_idx = np.argmax(cumulative_energy >= energy_threshold * total_energy)
        
        if threshold_idx == 0:
            # If threshold not reached, use 95% of Nyquist
            fmax_estimate = 0.95 * sr / 2
        else:
            fmax_estimate = freqs[threshold_idx]
        
        # Safety clamp
        fmax_estimate = min(fmax_estimate, sr / 2 * 0.95)
        
        return float(fmax_estimate)
    
    @staticmethod
    def resample_audio(audio, orig_sr, target_sr, method='auto'):
        """
        Resample audio with anti-aliasing
        
        Args:
            audio: Audio data (channels last if multichannel)
            orig_sr: Original sample rate
            target_sr: Target sample rate
            method: 'libsamplerate', 'scipy', 'ffmpeg', or 'auto'
        
        Returns:
            resampled_audio: Resampled audio data
        """
        if orig_sr == target_sr:
            return audio
        
        ratio = target_sr / orig_sr
        is_multichannel = len(audio.shape) > 1
        
        # Method 1: libsamplerate (best quality)
        if (method == 'auto' or method == 'libsamplerate') and LIBSAMPLERATE_AVAILABLE:
            try:
                if is_multichannel:
                    # Process each channel
                    resampled_channels = []
                    for ch in range(audio.shape[1]):
                        resampled = samplerate.resample(
                            audio[:, ch],
                            ratio,
                            converter_type='sinc_best'
                        )
                        resampled_channels.append(resampled)
                    return np.column_stack(resampled_channels)
                else:
                    return samplerate.resample(audio, ratio, converter_type='sinc_best')
            except Exception as e:
                print(f"libsamplerate failed: {e}, falling back to scipy")
        
        # Method 2: scipy.signal.resample_poly (good fallback)
        if method == 'auto' or method == 'scipy':
            try:
                # Calculate GCD for integer ratio
                from math import gcd
                
                # Convert to integer ratio
                up = target_sr
                down = orig_sr
                g = gcd(up, down)
                up //= g
                down //= g
                
                # Limit upsampling factor for performance
                max_up = 1000
                if up > max_up:
                    scale = max_up / up
                    up = max_up
                    down = int(down * scale)
                
                if is_multichannel:
                    resampled_channels = []
                    for ch in range(audio.shape[1]):
                        resampled = signal.resample_poly(
                            audio[:, ch],
                            up,
                            down,
                            window=('kaiser', 5.0)
                        )
                        resampled_channels.append(resampled)
                    return np.column_stack(resampled_channels)
                else:
                    return signal.resample_poly(
                        audio,
                        up,
                        down,
                        window=('kaiser', 5.0)
                    )
            except Exception as e:
                print(f"scipy resampling failed: {e}")
                if method == 'scipy':
                    raise
        
        # Method 3: ffmpeg fallback
        raise RuntimeError("No resampling method available")
    
    @staticmethod
    def save_audio(audio, sr, output_path, format='wav'):
        """Save audio file"""
        # Ensure audio is in valid range
        audio = np.clip(audio, -1.0, 1.0)
        
        if format == 'wav':
            if SOUNDFILE_AVAILABLE:
                sf.write(output_path, audio, sr, subtype='PCM_16')
            else:
                # Convert to int16
                audio_int16 = (audio * 32767).astype(np.int16)
                wavfile.write(output_path, sr, audio_int16)
        
        elif format == 'mp3':
            # Use ffmpeg for MP3 encoding
            temp_wav = output_path + '.temp.wav'
            
            if SOUNDFILE_AVAILABLE:
                sf.write(temp_wav, audio, sr, subtype='PCM_16')
            else:
                audio_int16 = (audio * 32767).astype(np.int16)
                wavfile.write(temp_wav, sr, audio_int16)
            
            # Convert to MP3
            cmd = [
                'ffmpeg', '-y', '-i', temp_wav,
                '-codec:a', 'libmp3lame',
                '-b:a', '192k',
                output_path
            ]
            subprocess.run(cmd, check=True, stderr=subprocess.DEVNULL)
            os.remove(temp_wav)
        
        else:
            raise ValueError(f"Unsupported format: {format}")

@resample_bp.route('/estimate_fmax', methods=['POST'])
def estimate_fmax():
    """
    Estimate maximum frequency content of uploaded audio
    
    Returns:
        JSON with:
        - orig_sr: Original sample rate
        - duration: Duration in seconds
        - channels: Number of channels
        - fmax_estimate: Estimated max frequency (Hz)
        - safe_min: Safe minimum sample rate (2.1 * fmax)
        - demo_min: Demo minimum sample rate
        - nyquist: Nyquist frequency
    """
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Validate file extension
        ext = os.path.splitext(file.filename.lower())[1]
        if ext not in ALLOWED_EXTENSIONS:
            return jsonify({
                "error": f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
            }), 400
        
        # Check file size
        file.seek(0, os.SEEK_END)
        size = file.tell()
        file.seek(0)
        
        if size > MAX_UPLOAD_SIZE:
            return jsonify({
                "error": f"File too large. Maximum size: {MAX_UPLOAD_SIZE / 1024 / 1024:.1f} MB"
            }), 400
        
        # Save temporarily
        temp_dir = current_app.config.get('UPLOAD_PATH', 'tmp')
        os.makedirs(temp_dir, exist_ok=True)
        
        temp_file = os.path.join(temp_dir, f"{uuid.uuid4()}{ext}")
        file.save(temp_file)
        
        try:
            # Load audio
            audio, sr = AudioResampler.load_audio(temp_file)
            
            # Get audio info
            duration = len(audio) / sr
            channels = audio.shape[1] if len(audio.shape) > 1 else 1
            
            # Estimate fmax
            fmax_estimate = AudioResampler.estimate_fmax(audio, sr)
            
            # Calculate safe bounds
            safe_min = math.ceil(2 * fmax_estimate * 1.05)
            demo_min = max(4000, int(0.5 * 2 * fmax_estimate))
            nyquist = sr / 2
            
            return jsonify({
                "success": True,
                "orig_sr": int(sr),
                "duration": float(duration),
                "channels": int(channels),
                "fmax_estimate": float(fmax_estimate),
                "safe_min": int(safe_min),
                "demo_min": int(demo_min),
                "nyquist": float(nyquist),
                "file_size_mb": float(size / 1024 / 1024)
            })
        
        finally:
            # Cleanup
            if os.path.exists(temp_file):
                os.remove(temp_file)
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500

@resample_bp.route('/', methods=['POST'])
def resample():
    """
    Resample uploaded audio to target sample rate
    
    Form parameters:
        - file: Audio file
        - target_sr: Target sample rate (integer)
        - mono: Convert to mono (optional, boolean)
        - format: Output format ('wav' or 'mp3', default 'wav')
    
    Returns:
        Resampled audio file
    """
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Get parameters
        try:
            target_sr = int(request.form.get('target_sr'))
        except (TypeError, ValueError):
            return jsonify({"error": "Invalid target_sr parameter"}), 400
        
        mono = request.form.get('mono', 'false').lower() == 'true'
        output_format = request.form.get('format', 'wav').lower()
        
        if output_format not in ['wav', 'mp3']:
            return jsonify({"error": "Format must be 'wav' or 'mp3'"}), 400
        
        if target_sr < 1000 or target_sr > 192000:
            return jsonify({"error": "target_sr must be between 1000 and 192000"}), 400
        
        # Validate file
        ext = os.path.splitext(file.filename.lower())[1]
        if ext not in ALLOWED_EXTENSIONS:
            return jsonify({"error": f"Unsupported file type"}), 400
        
        # Check size
        file.seek(0, os.SEEK_END)
        size = file.tell()
        file.seek(0)
        
        if size > MAX_UPLOAD_SIZE:
            return jsonify({"error": "File too large"}), 400
        
        # Save temporarily
        temp_dir = current_app.config.get('UPLOAD_PATH', 'tmp')
        os.makedirs(temp_dir, exist_ok=True)
        
        temp_input = os.path.join(temp_dir, f"{uuid.uuid4()}_input{ext}")
        temp_output = os.path.join(temp_dir, f"{uuid.uuid4()}_output.{output_format}")
        
        file.save(temp_input)
        
        try:
            # Load audio
            audio, orig_sr = AudioResampler.load_audio(temp_input)
            
            # Convert to mono if requested
            if mono and len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            
            # Resample
            resampled = AudioResampler.resample_audio(audio, orig_sr, target_sr)
            
            # Save output
            AudioResampler.save_audio(resampled, target_sr, temp_output, output_format)
            
            # Send file
            return send_file(
                temp_output,
                mimetype='audio/wav' if output_format == 'wav' else 'audio/mpeg',
                as_attachment=True,
                download_name=f"resampled_{target_sr}Hz.{output_format}"
            )
        
        finally:
            # Cleanup (note: temp_output is sent to client, will be cleaned up by Flask)
            if os.path.exists(temp_input):
                os.remove(temp_input)
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500

@resample_bp.route('/info', methods=['GET'])
def info():
    """Return service information"""
    return jsonify({
        "service": "Audio Resampling Service",
        "version": "1.0.0",
        "capabilities": {
            "libsamplerate": LIBSAMPLERATE_AVAILABLE,
            "soundfile": SOUNDFILE_AVAILABLE,
            "scipy": True
        },
        "max_upload_size_mb": MAX_UPLOAD_SIZE / 1024 / 1024,
        "supported_formats": list(ALLOWED_EXTENSIONS),
        "endpoints": {
            "/estimate_fmax": "POST - Estimate maximum frequency content",
            "/resample": "POST - Resample audio to target sample rate",
            "/info": "GET - Service information"
        }
    })