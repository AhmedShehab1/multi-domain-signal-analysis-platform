"""ECG Aliasing Test Blueprint"""

from flask import Blueprint, request, jsonify, send_file
import numpy as np
import io
import math

from utils.ecg_aliasing import (
    generate_multichannel_ecg,
    save_ecg_to_csv,
    load_ecg_from_csv,
    estimate_fmax,
    resample_ecg,
    energy_above_cutoff
)

ecg_aliasing_bp = Blueprint('ecg_aliasing', __name__)

@ecg_aliasing_bp.route('/generate/3ch', methods=['POST', 'GET'])
def generate_3ch_ecg():
    """
    Generate synthetic 3-channel ECG and return as CSV
    
    Query parameters:
        - fs: Sampling frequency (default: 1000 Hz)
        - duration: Duration in seconds (default: 10)
        - hr_bpm: Heart rate in BPM (default: 70)
        - qrs_width_ms: QRS width in ms (default: 80)
    """
    try:
        # Get parameters
        fs = int(request.args.get('fs', 1000))
        duration = float(request.args.get('duration', 10.0))
        hr_bpm = int(request.args.get('hr_bpm', 70))
        qrs_width_ms = int(request.args.get('qrs_width_ms', 80))
        
        # Validate parameters
        if fs < 100 or fs > 10000:
            return jsonify({"error": "fs must be between 100 and 10000 Hz"}), 400
        if duration < 1 or duration > 60:
            return jsonify({"error": "duration must be between 1 and 60 seconds"}), 400
        if hr_bpm < 40 or hr_bpm > 200:
            return jsonify({"error": "hr_bpm must be between 40 and 200"}), 400
        
        # Generate ECG
        data, fs_out = generate_multichannel_ecg(
            fs=fs,
            duration_s=duration,
            hr_bpm=hr_bpm,
            channels=3,
            qrs_width_ms=qrs_width_ms
        )
        
        # Save to CSV string
        csv_content = save_ecg_to_csv(None, data, fs_out, time_col=True, header_comment=True)
        
        # Create file-like object
        output = io.BytesIO()
        output.write(csv_content.encode('utf-8'))
        output.seek(0)
        
        return send_file(
            output,
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'synthetic_ecg_3ch_{fs}Hz.csv'
        )
    
    except Exception as e:
        import traceback
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500

@ecg_aliasing_bp.route('/generate/12ch', methods=['POST', 'GET'])
def generate_12ch_ecg():
    """
    Generate synthetic 12-channel ECG and return as CSV
    
    Query parameters:
        - fs: Sampling frequency (default: 1000 Hz)
        - duration: Duration in seconds (default: 10)
        - hr_bpm: Heart rate in BPM (default: 70)
        - qrs_width_ms: QRS width in ms (default: 80)
    """
    try:
        # Get parameters
        fs = int(request.args.get('fs', 1000))
        duration = float(request.args.get('duration', 10.0))
        hr_bpm = int(request.args.get('hr_bpm', 70))
        qrs_width_ms = int(request.args.get('qrs_width_ms', 80))
        
        # Validate parameters
        if fs < 100 or fs > 10000:
            return jsonify({"error": "fs must be between 100 and 10000 Hz"}), 400
        if duration < 1 or duration > 60:
            return jsonify({"error": "duration must be between 1 and 60 seconds"}), 400
        if hr_bpm < 40 or hr_bpm > 200:
            return jsonify({"error": "hr_bpm must be between 40 and 200"}), 400
        
        # Generate ECG
        data, fs_out = generate_multichannel_ecg(
            fs=fs,
            duration_s=duration,
            hr_bpm=hr_bpm,
            channels=12,
            qrs_width_ms=qrs_width_ms
        )
        
        # Save to CSV string
        csv_content = save_ecg_to_csv(None, data, fs_out, time_col=True, header_comment=True)
        
        # Create file-like object
        output = io.BytesIO()
        output.write(csv_content.encode('utf-8'))
        output.seek(0)
        
        return send_file(
            output,
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'synthetic_ecg_12ch_{fs}Hz.csv'
        )
    
    except Exception as e:
        import traceback
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500

@ecg_aliasing_bp.route('/analyze', methods=['POST'])
def analyze_ecg():
    """
    Analyze uploaded ECG CSV file
    
    Returns:
        - orig_sr: Original sampling rate
        - duration: Duration in seconds
        - channels: Number of channels
        - fmax_estimate: Estimated maximum frequency
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
        
        # Read CSV content
        csv_content = file.read().decode('utf-8')
        
        # Load ECG data
        data, fs, meta = load_ecg_from_csv(csv_content)
        
        if fs is None:
            return jsonify({"error": "Could not determine sampling rate from CSV"}), 400
        
        # Get info
        duration = len(data) / fs
        channels = data.shape[1]
        
        # Estimate fmax
        fmax = estimate_fmax(data, fs)
        
        # Calculate bounds
        safe_min = math.ceil(2 * fmax * 1.05)
        demo_min = max(50, int(0.5 * 2 * fmax))
        nyquist = fs / 2
        
        return jsonify({
            "success": True,
            "orig_sr": int(fs),
            "duration": float(duration),
            "channels": int(channels),
            "fmax_estimate": float(fmax),
            "safe_min": int(safe_min),
            "demo_min": int(demo_min),
            "nyquist": float(nyquist),
            "channel_names": meta.get('columns', [])
        })
    
    except Exception as e:
        import traceback
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500

@ecg_aliasing_bp.route('/resample', methods=['POST'])
def resample_ecg_endpoint():
    """
    Resample ECG CSV file
    
    Form parameters:
        - file: CSV file
        - target_sr: Target sampling rate
        - mode: 'safe' or 'demo' (default: 'safe')
    
    Returns:
        Resampled CSV file
    """
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Get parameters
        target_sr = int(request.form.get('target_sr'))
        mode = request.form.get('mode', 'safe')
        
        if mode not in ['safe', 'demo']:
            return jsonify({"error": "mode must be 'safe' or 'demo'"}), 400
        
        if target_sr < 50 or target_sr > 10000:
            return jsonify({"error": "target_sr must be between 50 and 10000"}), 400
        
        # Read CSV
        csv_content = file.read().decode('utf-8')
        data, orig_sr, meta = load_ecg_from_csv(csv_content)
        
        if orig_sr is None:
            return jsonify({"error": "Could not determine sampling rate"}), 400
        
        # Resample
        resampled_data, new_sr = resample_ecg(data, orig_sr, target_sr, mode=mode)
        
        # Generate CSV
        csv_output = save_ecg_to_csv(None, resampled_data, new_sr, time_col=True, header_comment=True)
        
        # Create file
        output = io.BytesIO()
        output.write(csv_output.encode('utf-8'))
        output.seek(0)
        
        mode_suffix = "_aliased" if mode == 'demo' else "_safe"
        return send_file(
            output,
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'ecg_resampled_{target_sr}Hz{mode_suffix}.csv'
        )
    
    except Exception as e:
        import traceback
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500

@ecg_aliasing_bp.route('/info', methods=['GET'])
def info():
    """Service information"""
    return jsonify({
        "service": "ECG Aliasing Test Service",
        "version": "1.0.0",
        "description": "Generate synthetic ECG signals and demonstrate aliasing effects",
        "endpoints": {
            "/generate/3ch": "GET - Generate 3-channel synthetic ECG CSV",
            "/generate/12ch": "GET - Generate 12-channel synthetic ECG CSV",
            "/analyze": "POST - Analyze ECG CSV and estimate Fmax",
            "/resample": "POST - Resample ECG with safe or demo (aliasing) mode",
            "/info": "GET - Service information"
        },
        "parameters": {
            "generate": {
                "fs": "Sampling frequency (100-10000 Hz, default: 1000)",
                "duration": "Duration in seconds (1-60, default: 10)",
                "hr_bpm": "Heart rate (40-200 BPM, default: 70)",
                "qrs_width_ms": "QRS width in ms (default: 80)"
            },
            "resample": {
                "target_sr": "Target sampling rate (50-10000 Hz)",
                "mode": "'safe' (anti-alias) or 'demo' (allow aliasing)"
            }
        },
        "recommended_ranges": {
            "typical_ecg_rates": [250, 500, 1000],
            "clinical_fmax": "~100-250 Hz (QRS features)",
            "safe_downsample_rule": "target_sr >= 2.1 * fmax"
        }
    })