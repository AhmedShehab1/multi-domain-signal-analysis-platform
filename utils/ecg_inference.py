"""ECG classification model implementation"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import os

class ECGClassifier:
    def __init__(self, model_path='./model/model.hdf5', use_ensemble=False):
        """
        Initialize the ECG classifier with pre-trained weights
        
        Args:
            model_path: Path to the main model file or directory
            use_ensemble: If True, load all models from other_seeds for ensemble prediction
        """
        self.class_names = [
            '1st degree AV block (1dAVb)',
            'Right bundle branch block (RBBB)',
            'Left bundle branch block (LBBB)',
            'Sinus bradycardia (SB)',
            'Atrial fibrillation (AF)',
            'Sinus tachycardia (ST)'
        ]
        
        self.use_ensemble = use_ensemble
        
        if use_ensemble:
            self.models = self._load_ensemble_models()
            print(f"Loaded {len(self.models)} models for ensemble prediction")
        else:
            # Load model without compilation
            self.model = keras.models.load_model(model_path, compile=False)
            # Then compile with correct settings as per README
            self.model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam())
            print(f"Loaded single model from {model_path}")
    
    def _load_ensemble_models(self):
        """Load all models from other_seeds directory for ensemble prediction"""
        models = []
        
        # Get base directory from where this module is located
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Load main model
        main_model_path = os.path.join(base_dir, 'model', 'model.hdf5')
        model = keras.models.load_model(main_model_path, compile=False)
        model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam())
        models.append(model)
        
        # Load other seeds
        other_seeds_dir = os.path.join(base_dir, 'model', 'other_seeds')
        if os.path.exists(other_seeds_dir):
            for i in range(1, 11):
                model_file = os.path.join(other_seeds_dir, f'model_{i}.hdf5')
                if os.path.exists(model_file):
                    model = keras.models.load_model(model_file, compile=False)
                    model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam())
                    models.append(model)
        
        return models
    
    def preprocess_ecg(self, ecg_data):
        """
        Preprocess ECG data to match model input requirements
        
        Args:
            ecg_data: numpy array of shape (4096, 12) or (N, 4096, 12)
                    Should be in Volts, will be scaled to 1e-4V
        
        Returns:
            Preprocessed tensor ready for model input
        """
        # Debug original data shape and stats
        print(f"Original ECG data shape: {ecg_data.shape}")
        print(f"Data range: [{np.min(ecg_data):.6f}, {np.max(ecg_data):.6f}]")
        print(f"Data mean: {np.mean(ecg_data):.6f}")
        
        # Convert to numpy if needed
        if not isinstance(ecg_data, np.ndarray):
            ecg_data = np.array(ecg_data)
        
        # Add batch dimension if needed
        if len(ecg_data.shape) == 2:
            ecg_data = np.expand_dims(ecg_data, axis=0)
        
        # Check for NaNs or infinity values
        if np.isnan(ecg_data).any() or np.isinf(ecg_data).any():
            print("WARNING: Data contains NaNs or infinity values. Replacing with zeros.")
            ecg_data = np.nan_to_num(ecg_data)
        
        # Scale to 1e-4V (multiply by 1000 if in Volts)
        # If your data is already in millivolts, you may need to adjust this
        ecg_data = ecg_data * 1000.0
        
        # Pad or truncate to 4096 if needed
        if ecg_data.shape[1] < 4096:
            # Pad with zeros
            padding = np.zeros((ecg_data.shape[0], 4096 - ecg_data.shape[1], 12))
            ecg_data = np.concatenate([ecg_data, padding], axis=1)
        elif ecg_data.shape[1] > 4096:
            # Truncate
            ecg_data = ecg_data[:, :4096, :]
        
        # Ensure correct shape (N, 4096, 12)
        assert ecg_data.shape[1] == 4096, f"Expected 4096 samples, got {ecg_data.shape[1]}"
        assert ecg_data.shape[2] == 12, f"Expected 12 leads, got {ecg_data.shape[2]}"
        
        # Check for specific leads ordering in the README:
        # The model expects: {DI, DII, DIII, AVL, AVF, AVR, V1, V2, V3, V4, V5, V6}
        # If your data is in a different order, you need to reorder it
        # Example for reordering if needed:
        # ecg_data = ecg_data[:, :, [0, 1, 2, 4, 5, 3, 6, 7, 8, 9, 10, 11]]
        
        print(f"Preprocessed data shape: {ecg_data.shape}")
        print(f"Preprocessed data range: [{np.min(ecg_data):.6f}, {np.max(ecg_data):.6f}]")
        
        return ecg_data.astype(np.float32)
    
    def predict(self, ecg_data, threshold=0.5):
        """
        Make predictions on ECG data
        
        Args:
            ecg_data: numpy array of shape (4096, 12) or (N, 4096, 12)
            threshold: Probability threshold for classification (default: 0.5)
        
        Returns:
            Dictionary with predictions and probabilities
        """
        # Preprocess
        processed_data = self.preprocess_ecg(ecg_data)
        
        # Make prediction
        if self.use_ensemble:
            # Average predictions from all models
            all_predictions = []
            for model in self.models:
                pred = model.predict(processed_data, verbose=0)
                all_predictions.append(pred)
            predictions = np.mean(all_predictions, axis=0)
            prediction_std = np.std(all_predictions, axis=0)
        else:
            predictions = self.model.predict(processed_data, verbose=0)
            prediction_std = None
        
        # Format results
        results = []
        for i in range(predictions.shape[0]):
            sample_result = {
                'predictions': [],
                'summary': {},
                'raw_probabilities': predictions[i].tolist()
            }
            
            for j, class_name in enumerate(self.class_names):
                prob = float(predictions[i, j])
                pred_dict = {
                    'abnormality': class_name,
                    'probability': prob,
                    'percentage': f"{prob * 100:.2f}%",
                    'detected': prob > threshold
                }
                
                # Add uncertainty if ensemble
                if prediction_std is not None:
                    pred_dict['uncertainty'] = float(prediction_std[i, j])
                
                sample_result['predictions'].append(pred_dict)
            
            # Summary
            detected_abnormalities = [
                p['abnormality'] for p in sample_result['predictions'] 
                if p['detected']
            ]
            sample_result['summary'] = {
                'total_detected': len(detected_abnormalities),
                'abnormalities': detected_abnormalities,
                'is_normal': len(detected_abnormalities) == 0,
                'confidence': 'high' if (predictions[i].max() > 0.8 or predictions[i].max() < 0.2) else 'medium'
            }
            
            results.append(sample_result)
        
        return results[0] if len(results) == 1 else results
    
    def predict_with_specific_model(self, ecg_data, model_type='main'):
        """
        Make predictions using a specific model variant
        
        Args:
            ecg_data: ECG data array
            model_type: 'main', 'date_order', 'individual_patients', or 'normal_order'
        
        Returns:
            Prediction results
        """
        # Get base directory
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        model_paths = {
            'main': os.path.join(base_dir, 'model', 'model.hdf5'),
            'date_order': os.path.join(base_dir, 'model', 'other_splits', 'model_date_order.hdf5'),
            'individual_patients': os.path.join(base_dir, 'model', 'other_splits', 'model_individual_patients.hdf5'),
            'normal_order': os.path.join(base_dir, 'model', 'other_splits', 'model_normal_order.hdf5')
        }
        
        if model_type not in model_paths:
            raise ValueError(f"Invalid model_type. Choose from: {list(model_paths.keys())}")
        
        model = keras.models.load_model(model_paths[model_type])
        processed_data = self.preprocess_ecg(ecg_data)
        predictions = model.predict(processed_data, verbose=0)
        
        # Format the predictions
        return self._format_predictions(predictions)
    
    def _format_predictions(self, predictions, threshold=0.5):
        """Helper method to format predictions"""
        results = []
        for i in range(predictions.shape[0]):
            sample_result = {
                'predictions': [],
                'summary': {},
                'raw_probabilities': predictions[i].tolist()
            }
            
            for j, class_name in enumerate(self.class_names):
                prob = float(predictions[i, j])
                sample_result['predictions'].append({
                    'abnormality': class_name,
                    'probability': prob,
                    'percentage': f"{prob * 100:.2f}%",
                    'detected': prob > threshold
                })
            
            detected_abnormalities = [
                p['abnormality'] for p in sample_result['predictions'] 
                if p['detected']
            ]
            sample_result['summary'] = {
                'total_detected': len(detected_abnormalities),
                'abnormalities': detected_abnormalities,
                'is_normal': len(detected_abnormalities) == 0
            }
            
            results.append(sample_result)
        
        return results[0] if len(results) == 1 else results