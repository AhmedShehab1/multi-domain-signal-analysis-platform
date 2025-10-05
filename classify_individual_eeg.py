"""
Classify individual EEG files using LEAD pretrained model

This script allows you to:
1. Load raw EEG files (.set, .edf, .fif, etc.)
2. Preprocess them to match LEAD's requirements (19 channels, 128Hz, 1-second segments)
3. Run inference using pretrained LEAD model
4. Get AD/Healthy predictions

Usage:
    python classify_individual_eeg.py --input my_eeg_file.set --model P-11-F-5-Base
"""

import argparse
import os
import sys
import numpy as np
import torch
from pathlib import Path
from types import SimpleNamespace
# Import MNE for EEG processing
try:
    import mne
    from mne.io import read_raw_eeglab, read_raw_edf, read_raw_fif
except ImportError:
    print("❌ MNE library not found. Install with: pip install mne")
    sys.exit(1)

# Standard 19-channel montage (10-20 system)
STANDARD_CHANNELS = [
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
    'T7', 'C3', 'Cz', 'C4', 'T8',
    'P7', 'P3', 'Pz', 'P4', 'P8',
    'O1', 'O2'
]

# Alternative channel names (some datasets use T3/T4 instead of T7/T8)
CHANNEL_ALIASES = {
    'T3': 'T7', 'T4': 'T8', 'T5': 'P7', 'T6': 'P8',
    'FP1': 'Fp1', 'FP2': 'Fp2',  # Case variations
}

TARGET_SFREQ = 128  # Hz
SEGMENT_LENGTH = 1.0  # seconds
LOW_FREQ = 0.5  # Hz
HIGH_FREQ = 45.0  # Hz

def _build_inference_args():
    return SimpleNamespace(
        method='LEAD',
        task_name='finetune',
        model='LEAD',
        model_id='P-11-F-5-Base',
        is_training=0,
        seq_len=128,
        enc_in=19,
        c_out=2,
        d_model=128,
        n_heads=8,
        e_layers=12,
        d_layers=1,
        d_ff=256,
        moving_avg=25,
        factor=1,
        distil=True,
        dropout=0.1,
        embed='timeF',
        freq='h',
        activation='gelu',
        patch_len=32,
        stride=8,
        patch_len_list='4',
        up_dim_list='76',
        num_kernels=6,
        top_k=5,
        K=65536,
        momentum=0.999,
        temperature=0.07,
        mask_ratio=0.5,
        contrastive_loss='all',
        no_inter_attn=False,
        no_temporal_block=False,
        no_channel_block=False,
        output_attention=False,
        augmentations='flip,frequency,jitter,mask,channel,drop',
        num_class=2,
        p_hidden_dims=[128, 128],
        p_hidden_layers=2,
    )
def load_eeg_file(filepath):
    """Load EEG file (supports .set, .edf, .fif formats)"""
    filepath = Path(filepath)
    ext = filepath.suffix.lower()
    
    print(f"📂 Loading EEG file: {filepath.name}")
    print(f"   Format: {ext}")
    
    try:
        if ext == '.set':
            raw = read_raw_eeglab(filepath, preload=True)
        elif ext == '.edf':
            raw = mne.io.read_raw_edf(filepath, preload=True)
        elif ext == '.fif':
            raw = read_raw_fif(filepath, preload=True)
        else:
            print(f"❌ Unsupported file format: {ext}")
            print("   Supported formats: .set (EEGLAB), .edf (EDF), .fif (MNE)")
            sys.exit(1)
        
        print(f"✅ Loaded successfully")
        print(f"   Channels: {len(raw.ch_names)}")
        print(f"   Sampling rate: {raw.info['sfreq']} Hz")
        print(f"   Duration: {raw.times[-1]:.2f} seconds")
        
        return raw
    
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        sys.exit(1)


def standardize_channel_names(raw):
    """Standardize channel names to match 10-20 system"""
    # Create mapping for current channel names
    mapping = {}
    for ch in raw.ch_names:
        ch_upper = ch.upper().strip()
        # Check if it's an alias
        if ch_upper in CHANNEL_ALIASES:
            mapping[ch] = CHANNEL_ALIASES[ch_upper]
        # Check if it matches standard (case-insensitive)
        elif ch_upper in [c.upper() for c in STANDARD_CHANNELS]:
            # Find the correct case version
            for std_ch in STANDARD_CHANNELS:
                if ch_upper == std_ch.upper():
                    mapping[ch] = std_ch
                    break
    
    if mapping:
        raw.rename_channels(mapping)
        print(f"   Renamed {len(mapping)} channels to standard names")
    
    return raw


def align_to_19_channels(raw):
    """Align EEG data to standard 19 channels"""
    print("\n🔧 Aligning channels to 19-channel 10-20 system...")
    
    # Standardize channel names first
    raw = standardize_channel_names(raw)
    
    # Get EEG channels only
    raw.pick_types(eeg=True, exclude='bads')
    
    current_channels = raw.ch_names
    print(f"   Current EEG channels: {len(current_channels)}")
    
    # Check which standard channels are present
    present_channels = [ch for ch in STANDARD_CHANNELS if ch in current_channels]
    missing_channels = [ch for ch in STANDARD_CHANNELS if ch not in current_channels]
    
    print(f"   Present: {len(present_channels)}/19")
    print(f"   Missing: {len(missing_channels)}/19")
    
    if missing_channels:
        print(f"   Missing channels: {', '.join(missing_channels)}")
    
    # Set montage for interpolation
    try:
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage, match_case=False, on_missing='warn')
    except Exception as e:
        print(f"⚠️  Warning: Could not set montage: {e}")
    
    # Pick only the channels we have from the standard 19
    if present_channels:
        raw.pick_channels(present_channels, ordered=True)
    
    # Interpolate missing channels if we have enough reference channels
    if missing_channels and len(present_channels) >= 3:
        print(f"   Interpolating {len(missing_channels)} missing channels...")
        raw.info['bads'] = missing_channels
        raw = raw.interpolate_bads(reset_bads=True)
    
    # Reorder to match standard channel order
    raw.reorder_channels(STANDARD_CHANNELS)
    
    print(f"✅ Aligned to 19 channels")
    
    return raw


def preprocess_eeg(raw):
    """Preprocess EEG data to match LEAD requirements"""
    print("\n🔧 Preprocessing EEG data...")
    
    # 1. Resample to 128 Hz
    if raw.info['sfreq'] != TARGET_SFREQ:
        print(f"   Resampling: {raw.info['sfreq']} Hz → {TARGET_SFREQ} Hz")
        raw.resample(TARGET_SFREQ)
    
    # 2. Bandpass filter (0.5 - 45 Hz)
    print(f"   Filtering: {LOW_FREQ} - {HIGH_FREQ} Hz")
    raw.filter(LOW_FREQ, HIGH_FREQ, fir_design='firwin')
    
    # 3. Get data
    data = raw.get_data()  # Shape: (n_channels, n_timepoints)
    
    # 4. Segment into 1-second windows
    n_samples_per_segment = int(TARGET_SFREQ * SEGMENT_LENGTH)
    n_segments = data.shape[1] // n_samples_per_segment
    
    print(f"   Segmenting: {n_segments} segments of {SEGMENT_LENGTH}s")
    
    # Reshape into segments
    segments = []
    for i in range(n_segments):
        start_idx = i * n_samples_per_segment
        end_idx = start_idx + n_samples_per_segment
        segment = data[:, start_idx:end_idx].T  # Shape: (128, 19)
        segments.append(segment)
    
    segments = np.array(segments)  # Shape: (n_segments, 128, 19)
    
    # 5. Standardize each segment independently (per channel)
    print(f"   Standardizing {segments.shape[0]} segments...")
    for i in range(segments.shape[0]):
        for ch in range(segments.shape[2]):
            mean = segments[i, :, ch].mean()
            std = segments[i, :, ch].std()
            if std > 0:
                segments[i, :, ch] = (segments[i, :, ch] - mean) / std
    
    print(f"✅ Preprocessed data shape: {segments.shape}")
    print(f"   (n_segments={segments.shape[0]}, timestamps={segments.shape[1]}, channels={segments.shape[2]})")
    
    return segments


def find_seed_directories(checkpoint_dir):
    """Find all seed directories within checkpoint directory"""
    seed_dirs = []
    
    # Check if checkpoint_dir itself contains checkpoint.pth
    direct_checkpoint = os.path.join(checkpoint_dir, 'checkpoint.pth')
    if os.path.exists(direct_checkpoint):
        return [checkpoint_dir]
    
    # Look for seed directories (e.g., nh8_el12_dm128_df256_seed41)
    if os.path.exists(checkpoint_dir):
        for item in os.listdir(checkpoint_dir):
            item_path = os.path.join(checkpoint_dir, item)
            if os.path.isdir(item_path):
                checkpoint_file = os.path.join(item_path, 'checkpoint.pth')
                if os.path.exists(checkpoint_file):
                    seed_dirs.append(item_path)
    
    return sorted(seed_dirs)


def load_model(model_name, seed_dir, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Load a single pretrained LEAD model from a specific seed directory"""
    # Import model
    sys.path.insert(0, os.path.abspath('.'))
    try:
        from models.LEAD import Model as LEADModel
        
        # Initialize model
        class Args:
            seq_len = 128
            enc_in = 19
            c_out = 2
            d_model = 128
            n_heads = 8
            e_layers = 12
            d_ff = 256
            dropout = 0.1
            embed = 'timeF'
            freq = 'h'
            activation = 'gelu'
        
        args = _build_inference_args()
        model = LEADModel(args).to(device)
        
        # Load checkpoint
        checkpoint_file = os.path.join(seed_dir, 'checkpoint.pth')
        checkpoint = torch.load(checkpoint_file, map_location=device)
        if isinstance(checkpoint, dict):
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        cleaned_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace('module.', '', 1) if k.startswith('module.') else k
            cleaned_state_dict[new_key] = v
        model.load_state_dict(cleaned_state_dict, strict=False)
        
        model.eval()
        return model
    
    except ImportError as e:
        print(f"❌ Could not import LEAD model: {e}")
        print("   Make sure you're running this script from the LEAD project root")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error loading model from {seed_dir}: {e}")
        raise


def load_all_models(model_name, device='cuda' if torch.cuda.is_available() else 'cpu', use_ensemble=True):
    """Load all pretrained LEAD models (potentially multiple seeds for ensemble)"""
    print(f"\n🤖 Loading pretrained model: {model_name}")
    print(f"   Device: {device}")
    
    # Determine checkpoint path
    if model_name == 'P-11-F-5-Base':
        checkpoint_dir = './checkpoints/LEAD/finetune/LEAD/P-11-F-5-Base/'
    elif model_name == 'S-5-Sup':
        checkpoint_dir = './checkpoints/LEAD/supervised/LEAD/S-5-Sup/'
    elif model_name == 'P-14-Base':
        checkpoint_dir = './checkpoints/LEAD/pretrain_lead/LEAD/P-14-Base/'
    elif model_name == 'P-16-Base':
        checkpoint_dir = './checkpoints/LEAD/pretrain_lead/LEAD/P-16-Base/'
    else:
        print(f"❌ Unknown model: {model_name}")
        sys.exit(1)
    
    if not os.path.exists(checkpoint_dir):
        print(f"❌ Model checkpoint not found at: {checkpoint_dir}")
        print("   Please download pretrained models from:")
        print("   https://drive.google.com/drive/folders/1JDg0VxbML6pIrzxzm9GXiC6ixUIgujt9")
        sys.exit(1)
    
    # Find all seed directories
    seed_dirs = find_seed_directories(checkpoint_dir)
    
    if not seed_dirs:
        print(f"❌ No checkpoint files found in: {checkpoint_dir}")
        print("   Expected structure: {checkpoint_dir}/seedXX/checkpoint.pth")
        sys.exit(1)
    
    print(f"   Found {len(seed_dirs)} model checkpoint(s)")
    
    # Load models
    models = []
    for i, seed_dir in enumerate(seed_dirs):
        seed_name = os.path.basename(seed_dir)
        print(f"   Loading model {i+1}/{len(seed_dirs)}: {seed_name}")
        try:
            model = load_model(model_name, seed_dir, device)
            models.append(model)
        except Exception as e:
            print(f"   ⚠️  Warning: Failed to load model from {seed_name}: {e}")
            continue
    
    if not models:
        print(f"❌ Failed to load any models")
        sys.exit(1)
    
    print(f"✅ Successfully loaded {len(models)} model(s)")
    
    if len(models) > 1 and use_ensemble:
        print(f"   Using ensemble prediction (averaging {len(models)} models)")
    elif len(models) > 1:
        print(f"   Using single model (first seed only)")
        models = [models[0]]
    
    return models


def predict(models, segments, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Run inference on preprocessed segments using single model or ensemble"""
    n_models = len(models)
    
    if n_models == 1:
        print(f"\n🔮 Running inference on {segments.shape[0]} segments...")
    else:
        print(f"\n🔮 Running ensemble inference on {segments.shape[0]} segments ({n_models} models)...")
    
    # Convert to torch tensor
    X = torch.FloatTensor(segments).to(device)
    
    # Store probabilities from all models
    all_probabilities = []
    
    # Run inference with each model
    for model_idx, model in enumerate(models):
        if n_models > 1:
            print(f"   Model {model_idx + 1}/{n_models}...", end=' ')
        
        model_probabilities = []
        
        with torch.no_grad():
            # Process in batches to avoid memory issues
            batch_size = 128
            for i in range(0, len(X), batch_size):
                batch = X[i:i+batch_size]
                x_mark_enc = torch.zeros(batch.size(0), batch.size(1), 4, device=device)
                x_dec = batch.clone()
                x_mark_dec = torch.zeros(batch.size(0), batch.size(1), 4, device=device)

                outputs = model(batch, x_mark_enc, x_dec, x_mark_dec)
                probs = torch.softmax(outputs, dim=1)
                model_probabilities.extend(probs.cpu().numpy())
        
        all_probabilities.append(np.array(model_probabilities))
        
        if n_models > 1:
            print("✓")
    
    # Average probabilities across all models (ensemble)
    probabilities = np.mean(all_probabilities, axis=0)
    
    # Get final predictions from averaged probabilities
    predictions = np.argmax(probabilities, axis=1)
    
    if n_models == 1:
        print(f"✅ Inference completed")
    else:
        print(f"✅ Ensemble inference completed (averaged {n_models} models)")
    
    return predictions, probabilities


def analyze_results(predictions, probabilities):
    """Analyze and display results"""
    print("\n" + "="*70)
    print("📊 CLASSIFICATION RESULTS")
    print("="*70)
    
    # Sample-level results
    n_healthy = np.sum(predictions == 0)
    n_ad = np.sum(predictions == 1)
    total = len(predictions)
    
    print(f"\n📋 Sample-level predictions ({total} segments):")
    print(f"   Healthy (Class 0): {n_healthy} ({n_healthy/total*100:.1f}%)")
    print(f"   AD (Class 1):      {n_ad} ({n_ad/total*100:.1f}%)")
    
    # Subject-level prediction (majority voting)
    subject_prediction = 1 if n_ad > n_healthy else 0
    confidence = max(n_ad, n_healthy) / total * 100
    
    print(f"\n🎯 Subject-level prediction (majority voting):")
    if subject_prediction == 0:
        print(f"   Result: HEALTHY")
    else:
        print(f"   Result: ALZHEIMER'S DISEASE")
    print(f"   Confidence: {confidence:.1f}%")
    
    # Average probabilities
    avg_healthy_prob = probabilities[:, 0].mean()
    avg_ad_prob = probabilities[:, 1].mean()
    
    print(f"\n📈 Average probabilities across all segments:")
    print(f"   P(Healthy): {avg_healthy_prob:.3f}")
    print(f"   P(AD):      {avg_ad_prob:.3f}")
    
    # Confidence distribution
    print(f"\n📊 Confidence distribution:")
    high_conf = np.sum(np.max(probabilities, axis=1) > 0.8)
    med_conf = np.sum((np.max(probabilities, axis=1) > 0.6) & (np.max(probabilities, axis=1) <= 0.8))
    low_conf = np.sum(np.max(probabilities, axis=1) <= 0.6)
    
    print(f"   High confidence (>80%): {high_conf} segments ({high_conf/total*100:.1f}%)")
    print(f"   Medium confidence (60-80%): {med_conf} segments ({med_conf/total*100:.1f}%)")
    print(f"   Low confidence (<60%): {low_conf} segments ({low_conf/total*100:.1f}%)")
    
    return {
        'sample_level': {
            'predictions': predictions,
            'probabilities': probabilities,
            'n_healthy': n_healthy,
            'n_ad': n_ad
        },
        'subject_level': {
            'prediction': subject_prediction,
            'confidence': confidence,
            'avg_healthy_prob': avg_healthy_prob,
            'avg_ad_prob': avg_ad_prob
        }
    }


def save_results(results, output_path):
    """Save results to file"""
    print(f"\n💾 Saving results to: {output_path}")
    
    np.savez(output_path,
             sample_predictions=results['sample_level']['predictions'],
             sample_probabilities=results['sample_level']['probabilities'],
             subject_prediction=results['subject_level']['prediction'],
             subject_confidence=results['subject_level']['confidence'])
    
    # Also save a text summary
    txt_path = output_path.replace('.npz', '_summary.txt')
    with open(txt_path, 'w') as f:
        f.write("LEAD Model Classification Results\n")
        f.write("="*70 + "\n\n")
        f.write(f"Total segments: {len(results['sample_level']['predictions'])}\n")
        f.write(f"Healthy predictions: {results['sample_level']['n_healthy']}\n")
        f.write(f"AD predictions: {results['sample_level']['n_ad']}\n\n")
        f.write(f"Subject-level prediction: {'AD' if results['subject_level']['prediction'] == 1 else 'Healthy'}\n")
        f.write(f"Confidence: {results['subject_level']['confidence']:.1f}%\n")
    
    print(f"✅ Results saved")


def main():
    parser = argparse.ArgumentParser(
        description='Classify individual EEG files using LEAD pretrained model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Classify a single .set file (uses ensemble of all available seeds)
  python classify_individual_eeg.py --input my_eeg.set --model P-11-F-5-Base
  
  # Classify .edf file with custom output
  python classify_individual_eeg.py --input recording.edf --output results.npz
  
  # Use only first seed (faster, no ensemble)
  python classify_individual_eeg.py --input data.set --no-ensemble
  
  # Use CPU instead of GPU
  python classify_individual_eeg.py --input data.set --device cpu
        """
    )
    
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Input EEG file (.set, .edf, .fif)')
    parser.add_argument('--model', '-m', type=str, default='P-11-F-5-Base',
                        choices=['P-11-F-5-Base', 'S-5-Sup', 'P-14-Base', 'P-16-Base'],
                        help='Pretrained model to use (default: P-11-F-5-Base)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output file path (default: input_name_results.npz)')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='Device to use (default: auto)')
    parser.add_argument('--skip-interpolation', action='store_true',
                        help='Skip channel interpolation (only use available channels)')
    parser.add_argument('--no-ensemble', action='store_true',
                        help='Use only first model instead of ensemble (faster but less accurate)')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"❌ Input file not found: {args.input}")
        sys.exit(1)
    
    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    # Set output path
    if args.output is None:
        input_path = Path(args.input)
        args.output = str(input_path.parent / f"{input_path.stem}_results.npz")
    
    print("="*70)
    print("LEAD EEG Classification Pipeline")
    print("="*70)
    print(f"Input: {args.input}")
    print(f"Model: {args.model}")
    print(f"Device: {device}")
    print(f"Output: {args.output}")
    print("="*70)
    
    # Pipeline
    try:
        # 1. Load EEG file
        raw = load_eeg_file(args.input)
        
        # 2. Align to 19 channels
        if not args.skip_interpolation:
            raw = align_to_19_channels(raw)
        
        # 3. Preprocess
        segments = preprocess_eeg(raw)
        
        # 4. Load model(s)
        use_ensemble = not args.no_ensemble
        models = load_all_models(args.model, device, use_ensemble)
        
        # 5. Predict
        predictions, probabilities = predict(models, segments, device)
        
        # 6. Analyze results
        results = analyze_results(predictions, probabilities)
        
        # 7. Save results
        save_results(results, args.output)
        
        print("\n" + "="*70)
        print("✅ Classification completed successfully!")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ Error during classification: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
