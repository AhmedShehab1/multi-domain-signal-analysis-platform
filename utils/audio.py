"""Audio generation utility functions"""

import numpy as np

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