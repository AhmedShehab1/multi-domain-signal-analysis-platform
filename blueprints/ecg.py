"""ECG Signal Analysis Blueprint"""

import io
import json
from functools import lru_cache

import numpy as np
from flask import Blueprint, jsonify, request, render_template

from utils.ecg_utils import (
    CLASS_NAMES,
    download_model,
    load_model,
    preprocess_ecg_data,
)

ecg_bp = Blueprint('ecg', __name__)


@ecg_bp.route('/')
def ecg_demo():
    """Serve the ECG demo interface"""
    return render_template('ecg_demo.html')


@ecg_bp.route('/demo')
def ecg_demo_alias():
    """Alias for the ECG demo interface"""
    return render_template('ecg_demo.html')

@lru_cache(maxsize=1)
def get_model():
    """Load and cache the classifier model."""
    download_model()
    model = load_model()
    if model is None:
        raise RuntimeError("Unable to load the ECG classification model.")
    return model

def _parse_signal_from_json(payload):
    if "signal" not in payload:
        raise ValueError("JSON body must include a `signal` field containing an array of samples.")

    signal = payload["signal"]
    if not isinstance(signal, (list, tuple)) or isinstance(signal, (str, bytes, dict)):
        raise ValueError("`signal` must be a JSON array (list) of numeric values.")

    try:
        array = np.asarray(signal, dtype=np.float32)
    except ValueError as exc:
        raise ValueError(f"Unable to convert signal values to float: {exc}") from exc

    if array.ndim != 1:
        raise ValueError("`signal` must describe a one-dimensional array of samples.")

    if array.size == 0:
        raise ValueError("`signal` must contain at least one sample.")

    return array

def _parse_signal_from_file(storage):
    filename = storage.filename or "uploaded file"
    try:
        raw = storage.read()
        storage.seek(0)
    except Exception as exc:
        raise ValueError(f"Failed to read uploaded file {filename}: {exc}") from exc

    if not raw:
        raise ValueError(f"Uploaded file {filename} is empty.")

    # Try JSON first (useful for drag-drop JSON files)
    try:
        payload = json.loads(raw.decode("utf-8"))
        if isinstance(payload, dict):
            return _parse_signal_from_json(payload)
    except Exception:
        pass

    # Fall back to CSV/plain-text numeric values
    try:
        array = np.loadtxt(io.BytesIO(raw), delimiter=",", dtype=np.float32)
        print(array)
    except Exception as exc:
        raise ValueError(
            "Unable to parse uploaded file. Provide a CSV/plain-text column of numbers "
            "or a JSON file containing {'signal': [...]}"
        ) from exc

    if not isinstance(array, np.ndarray):
        array = np.asarray([array], dtype=np.float32)

    if array.ndim > 1:
        array = array.reshape(-1)

    if array.size == 0:
        raise ValueError("Uploaded file does not contain any samples.")

    return array

def _extract_signal():
    if request.is_json:
        payload = request.get_json(silent=True) or {}
        if not isinstance(payload, dict):
            raise ValueError("JSON body must be an object containing a `signal` field.")
        return _parse_signal_from_json(payload)

    if request.files:
        storage = request.files.get("file")
        if storage is None:
            raise ValueError("Multipart requests must include a `file` field with the ECG content.")
        return _parse_signal_from_file(storage)

    raise ValueError(
        "Unsupported payload. Send JSON {\"signal\": [...]} or multipart/form-data with a `file`."
    )

@ecg_bp.route('/classify', methods=['POST'])
def classify_ecg():
    """Classify a single ECG waveform and return class probabilities."""
    try:
        signal = _extract_signal()
        processed = preprocess_ecg_data(signal)
        model = get_model()
        predictions = model.predict(processed, verbose=0)[0]
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 500

    class_index = int(np.argmax(predictions))
    confidence = float(predictions[class_index])

    response = {
        "prediction": {
            "class_index": class_index,
            "class_label": CLASS_NAMES[class_index],
            "confidence": confidence,
        },
        "probabilities": [
            {
                "class_index": idx,
                "class_label": CLASS_NAMES[idx],
                "score": float(score),
            }
            for idx, score in enumerate(predictions)
        ],
        "input_summary": {
            "sample_count": int(signal.size),
        },
    }

    return jsonify(response)
