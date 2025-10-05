"""ECG utilities and model handling"""

import os
import numpy as np
import requests
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

# Classification labels
CLASS_NAMES = ['Normal', 'Supraventricular', 'Ventricular', 'Fusion', 'Unclassifiable']

def download_model():
    """Download the model.h5 file from the repository"""
    url = "https://huggingface.co/Nonbangkok/MITBIH-ECG-Arrhythmia-Classification/resolve/main/model.h5"
    
    if not os.path.exists("model.h5"):
        print("Downloading model...")
        response = requests.get(url)
        with open("model.h5", "wb") as f:
            f.write(response.content)
        print("Model downloaded successfully!")
    else:
        print("Model already exists!")

def load_model():
    """Load the pre-trained model"""
    try:
        model = keras.models.load_model("model.h5")
        print("Model loaded successfully!")
        print(f"Model input shape: {model.input_shape}")
        print(f"Model output shape: {model.output_shape}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def preprocess_ecg_data(ecg_signal, target_length=187):
    """
    Preprocess ECG signal for the model
    
    Args:
        ecg_signal: 1D array of ECG values
        target_length: Expected input length for the model (typically 187 for MIT-BIH)
    
    Returns:
        Preprocessed ECG data ready for model input
    """
    signal = np.asarray(ecg_signal, dtype=np.float32)

    if signal.ndim == 1:
        signal = _resize_segment(signal, target_length)
        signal = _standardize_segment(signal)
        return signal.reshape(1, target_length, 1)

    if signal.ndim == 2:
        processed_segments = []
        for segment in signal:
            segment = _resize_segment(np.asarray(segment, dtype=np.float32), target_length)
            processed_segments.append(_standardize_segment(segment))
        stacked = np.stack(processed_segments)
        return stacked.reshape(stacked.shape[0], target_length, 1)

    raise ValueError("ECG signal must be a 1D or 2D array.")

def _resize_segment(segment: np.ndarray, target_length: int) -> np.ndarray:
    """Better resizing that preserves signal characteristics"""
    if segment.size == target_length:
        return segment
        
    if segment.size > target_length:
        # Center-crop around most important part (look for R peak)
        r_peak_idx = np.argmax(np.abs(segment))
        half_window = target_length // 2
        start = max(0, r_peak_idx - half_window)
        end = start + target_length
        if end > segment.size:
            start = max(0, segment.size - target_length)
            end = segment.size
        return segment[start:end]
    
    # For shorter segments, use edge padding instead of zero padding
    return np.pad(segment, (0, target_length - segment.size), mode='edge')

def _standardize_segment(segment: np.ndarray) -> np.ndarray:
    scaler = StandardScaler()
    return scaler.fit_transform(segment.reshape(-1, 1)).flatten()
