"""Audio generation utility functions"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import stft, find_peaks
import warnings
warnings.filterwarnings('ignore')


class DopplerVelocityDetector:
    """
    Analyzes audio to detect frequency changes and calculate velocity
    using the Doppler Effect.
    """
    
    def __init__(self, speed_of_sound=343.0):
        """
        Initialize the detector.
        
        Parameters:
        -----------
        speed_of_sound : float
            Speed of sound in m/s (default: 343 m/s for air at 20°C)
        """
        self.c = speed_of_sound
        self.fs = None  # Sampling rate
        self.audio = None  # Audio data
        self.time_vector = None
        self.freq_vector = None
        self.spectrogram = None
        self.observed_frequencies = None
        self.velocities = None

    def _interpolate_peak_frequency(self, spectrum, freq_vector, peak_idx):
        """Parabolic interpolation for sub-bin frequency resolution"""
        if peak_idx <= 0 or peak_idx >= len(spectrum)-1:
            return freq_vector[peak_idx]
        
        y0, y1, y2 = spectrum[peak_idx-1], spectrum[peak_idx], spectrum[peak_idx+1]
        bin_width = freq_vector[1] - freq_vector[0]
        
        # Quadratic interpolation
        p = 0.5 * (y0 - y2) / (y0 - 2*y1 + y2)
        
        # Interpolated peak is p bins away from peak_idx
        interp_freq = freq_vector[peak_idx] + p * bin_width
        
        return interp_freq
    
    def load_audio(self, filepath):
        """
        Load audio file from disk.
        
        Parameters:
        -----------
        filepath : str
            Path to WAV audio file
        """
        self.fs, audio_data = wavfile.read(filepath)
        
        # Convert to mono if stereo
        if len(audio_data.shape) > 1:
            self.audio = audio_data.mean(axis=1)
        else:
            self.audio = audio_data
            
        # Normalize to [-1, 1]
        self.audio = self.audio / np.max(np.abs(self.audio))
        
        print(f"✓ Audio loaded: {len(self.audio)/self.fs:.2f} seconds")
        print(f"  Sampling rate: {self.fs} Hz")
        
    def compute_spectrogram(self, window_size=16384, overlap=1536):
        """
        Compute Short-Time Fourier Transform (STFT) to get time-frequency representation.
        
        Parameters:
        -----------
        window_size : int
            Number of samples per window (larger = better frequency resolution)
        overlap : int
            Number of overlapping samples between windows
        
        Think of this like taking "snapshots" of the sound every few milliseconds!
        """
        nperseg = window_size
        noverlap = overlap
        
        # Compute STFT
        self.freq_vector, self.time_vector, Zxx = stft(
            self.audio, 
            fs=self.fs, 
            nperseg=nperseg, 
            noverlap=noverlap
        )
        
        # Convert to magnitude (we only care about "how strong" each frequency is)
        self.spectrogram = np.abs(Zxx)
        
        print(f"✓ Spectrogram computed:")
        print(f"  Time resolution: {self.time_vector[1] - self.time_vector[0]:.4f} s")
        print(f"  Frequency resolution: {self.freq_vector[1] - self.freq_vector[0]:.2f} Hz")
        
    def track_frequency(self, freq_min=100, freq_max=2000, smoothing_window=5):
        """
        Extract the dominant (peak) frequency at each time frame.
        """
        # Find indices corresponding to frequency range
        freq_mask = (self.freq_vector >= freq_min) & (self.freq_vector <= freq_max)
        
        observed_freqs = []
        
        for time_idx in range(self.spectrogram.shape[1]):
            # Get spectrum at this time frame
            spectrum = self.spectrogram[freq_mask, time_idx]
            
            # Find peaks in the spectrum
            peaks, properties = find_peaks(spectrum, height=np.max(spectrum)*0.05)  # Reduced threshold

            if len(peaks) > 0:
                strongest_peak_idx = peaks[np.argmax(properties['peak_heights'])]
                peak_freq = self._interpolate_peak_frequency(spectrum, self.freq_vector[freq_mask], strongest_peak_idx)
            else:
                peak_idx = np.argmax(spectrum)
                peak_freq = self._interpolate_peak_frequency(spectrum, self.freq_vector[freq_mask], peak_idx)

            observed_freqs.append(peak_freq)
        
        self.observed_frequencies = np.array(observed_freqs)
        
        # Apply moving average smoothing to reduce noise
        if smoothing_window > 1:
            self.observed_frequencies = self._moving_average(
                self.observed_frequencies, 
                smoothing_window
            )
        
        print(f"✓ Frequency tracking complete")
        print(f"  Observed frequency range: {np.nanmin(self.observed_frequencies):.1f} - "
            f"{np.nanmax(self.observed_frequencies):.1f} Hz")
        
    def estimate_base_frequency(self):
        """
        Estimate the actual emitted frequency (f0) of the source.
        
        Uses the harmonic mean of max and min observed frequencies.
        This works when the object passes by (moving toward then away).
        """
        f_max = np.nanmax(self.observed_frequencies)
        f_min = np.nanmin(self.observed_frequencies)
        
        # Harmonic mean formula for Doppler
        f0 = (2 * f_max * f_min) / (f_max + f_min)
        
        print(f"✓ Estimated base frequency (f₀): {f0:.1f} Hz")
        return f0
    
    def calculate_velocity(self, f0, direction_auto=True):
        """
        Calculate radial velocity using the Doppler formula.
        
        For moving source:
        f_obs = f0 * (c / (c - v_r))  # approaching
        f_obs = f0 * (c / (c + v_r))  # receding
        
        Rearranged:
        v_r = c * (f0 / f_obs - 1)
        
        Positive velocity = moving away
        Negative velocity = approaching
        """
        # Calculate velocity for each time frame (moving source formula)
        self.velocities = self.c * (f0 / self.observed_frequencies - 1)
        
        # Filter out unrealistic velocities (likely errors)
        self.velocities = np.where(np.abs(self.velocities) > self.c * 0.5, np.nan, self.velocities)
        
        print(f"✓ Velocity calculation complete")
        print(f"  Velocity range: {np.nanmin(self.velocities):.1f} to "
            f"{np.nanmax(self.velocities):.1f} m/s")
        
        # Convert to km/h for readability
        velocities_kmh = self.velocities * 3.6
        print(f"  In km/h: {np.nanmin(velocities_kmh):.1f} to "
            f"{np.nanmax(velocities_kmh):.1f} km/h")
        
    def _moving_average(self, data, window_size):
        """Helper function for smoothing data."""
        # Pad the data to handle edges
        padded = np.pad(data, (window_size//2, window_size//2), mode='edge')
        # Use convolution for efficient moving average
        kernel = np.ones(window_size) / window_size
        smoothed = np.convolve(padded, kernel, mode='valid')
        return smoothed[:len(data)]
    
    def plot_results(self, f0=None, expected_velocity=None, save_path=None):
        """
        Visualize the complete analysis.
        
        Creates a 4-panel plot showing:
        1. Raw audio waveform
        2. Spectrogram (time-frequency heatmap)
        3. Tracked frequency over time
        4. Calculated velocity over time
        
        Parameters:
        -----------
        f0 : float, optional
            Base frequency to display as reference line
        expected_velocity : float, optional
            Expected velocity to display as reference lines
        save_path : str, optional
            Path to save the figure
        """
        fig, axes = plt.subplots(4, 1, figsize=(14, 12))
        fig.suptitle('Doppler Effect Analysis', fontsize=16, fontweight='bold')
        
        # 1. Audio Waveform
        time_audio = np.arange(len(self.audio)) / self.fs
        axes[0].plot(time_audio, self.audio, color='steelblue', linewidth=0.5)
        axes[0].set_ylabel('Amplitude')
        axes[0].set_title('📊 Raw Audio Signal')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xlim([0, time_audio[-1]])
        
        # 2. Spectrogram
        im = axes[1].pcolormesh(
            self.time_vector, 
            self.freq_vector, 
            10 * np.log10(self.spectrogram + 1e-10),  # Convert to dB
            shading='gouraud',
            cmap='magma'
        )
        axes[1].set_ylabel('Frequency (Hz)')
        axes[1].set_title('🌈 Spectrogram (Time-Frequency Analysis)')
        axes[1].set_ylim([0, 2500])
        plt.colorbar(im, ax=axes[1], label='Power (dB)')
        
        # 3. Tracked Frequency
        axes[2].plot(self.time_vector, self.observed_frequencies, 
                    color='crimson', linewidth=2, label='Observed Frequency')
        if f0 is not None:
            axes[2].axhline(f0, color='green', linestyle='--', 
                        linewidth=2, label=f'Base Frequency f₀ = {f0:.1f} Hz')
        axes[2].set_ylabel('Frequency (Hz)')
        axes[2].set_title('🎵 Tracked Dominant Frequency')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        # 4. Velocity
        if self.velocities is not None:
            axes[3].plot(self.time_vector, self.velocities, 
                        color='darkorange', linewidth=2)
            axes[3].axhline(0, color='black', linestyle=':', linewidth=1)
            
            # Add expected velocity reference lines if provided
            if expected_velocity is not None:
                axes[3].axhline(expected_velocity, color='green', linestyle='--', 
                            linewidth=2, label=f'Expected: {expected_velocity} m/s')
                axes[3].axhline(-expected_velocity, color='green', linestyle='--', 
                            linewidth=2)
                axes[3].legend()
            
            axes[3].fill_between(self.time_vector, 0, self.velocities, 
                                alpha=0.3, color='orange')
            axes[3].set_ylabel('Velocity (m/s)')
            axes[3].set_xlabel('Time (seconds)')
            axes[3].set_title('🚗 Calculated Radial Velocity')
            axes[3].grid(True, alpha=0.3)
            
            # Add text annotations
            max_vel = np.nanmax(np.abs(self.velocities))
            axes[3].text(0.02, 0.95, f'Max Speed: {max_vel:.1f} m/s ({max_vel*3.6:.1f} km/h)',
                        transform=axes[3].transAxes, 
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                        verticalalignment='top')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Plot saved to {save_path}")
        
        plt.show()
    
    def export_results(self, output_file='doppler_results.csv'):
        """
        Export results to CSV file.
        """
        import pandas as pd
        
        df = pd.DataFrame({
            'Time (s)': self.time_vector,
            'Observed Frequency (Hz)': self.observed_frequencies,
            'Velocity (m/s)': self.velocities,
            'Velocity (km/h)': self.velocities * 3.6
        })
        
        df.to_csv(output_file, index=False)
        print(f"✓ Results exported to {output_file}")

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