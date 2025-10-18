"""
ECG Aliasing Test Utilities
Generates synthetic ECG signals and provides resampling with aliasing demonstrations
"""

import numpy as np
import pandas as pd
from scipy import signal
from fractions import Fraction
import math

try:
    import samplerate
    HAS_SAMPLERATE = True
except Exception:
    HAS_SAMPLERATE = False

# ---------------------------
# Synthetic ECG generator
# ---------------------------
def _gaussian(x, mu, sigma, amp=1.0):
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def generate_single_lead_ecg(fs=1000, duration_s=10.0, hr_bpm=60,
                             qrs_width_ms=80, p_width_ms=110, t_width_ms=160,
                             noise_std=0.01, jitter_ms=10):
    """
    Build a simple synthetic ECG: place P, QRS, T gaussians per beat.
    - fs: sampling rate (Hz)
    - duration_s: total seconds
    - hr_bpm: heart rate in bpm
    - qrs_width_ms: width of QRS complex (smaller -> more HF energy)
    """
    n = int(round(duration_s * fs))
    t = np.arange(n) / fs
    rr_s = 60.0 / hr_bpm
    # Beat times with small jitter
    beat_times = np.arange(0, duration_s, rr_s)
    beat_times += np.random.normal(0.0, jitter_ms / 1000.0, size=beat_times.shape)

    sig = np.zeros_like(t, dtype=float)
    for bt in beat_times:
        # center times for P, QRS, T relative to beat
        p_center = bt - 0.20 * rr_s
        qrs_center = bt
        t_center = bt + 0.25 * rr_s

        # widths in seconds
        qrs_sigma = (qrs_width_ms / 1000.0) / (2.0 * math.sqrt(2*math.log(2)))
        p_sigma = (p_width_ms / 1000.0) / (2.0 * math.sqrt(2*math.log(2)))
        t_sigma = (t_width_ms / 1000.0) / (2.0 * math.sqrt(2*math.log(2)))

        # amplitudes
        p_amp = 0.1
        qrs_amp = 1.0
        t_amp = 0.3

        sig += _gaussian(t, p_center, p_sigma, amp=p_amp)
        sig += _gaussian(t, qrs_center, qrs_sigma, amp=qrs_amp)
        sig += _gaussian(t, t_center, t_sigma, amp=t_amp)

    # add baseline wander (low freq) and gaussian noise
    baseline = 0.02 * np.sin(2 * np.pi * 0.33 * t)
    noise = np.random.normal(0.0, noise_std, size=sig.shape)
    sig = sig + baseline + noise

    # normalize to -1..1 roughly
    sig = sig / (np.max(np.abs(sig)) + 1e-12) * 0.9
    return sig, fs

def generate_multichannel_ecg(fs=1000, duration_s=10.0, hr_bpm=60,
                              channels=3, **kwargs):
    """
    Generate multichannel ECG as variations of the primary lead.
    For 3-channel: create small delays and amplitude scalings.
    For 12-channel: produce 12 variants by mixing/scaling/delays.
    Returns: data (n_samples, channels), fs
    """
    lead0, fs = generate_single_lead_ecg(fs=fs, duration_s=duration_s, hr_bpm=hr_bpm, **kwargs)
    n = len(lead0)
    if channels == 1:
        return lead0.reshape(-1,1), fs

    def mk_variant(delay_ms=0.0, scale=1.0, noise=0.005):
        shift = int(round(delay_ms * fs / 1000.0))
        if shift >= 0:
            v = np.concatenate([np.zeros(shift), lead0[:n-shift]])
        else:
            s = -shift
            v = np.concatenate([lead0[s:], np.zeros(s)])
        v = v * scale + np.random.normal(0.0, noise, size=v.shape)
        return v

    if channels == 3:
        ch1 = mk_variant(delay_ms=0.0, scale=1.0)
        ch2 = mk_variant(delay_ms=5.0, scale=0.9)
        ch3 = mk_variant(delay_ms=-3.0, scale=1.05)
        data = np.column_stack([ch1, ch2, ch3])
        return data, fs

    # channels == 12 (approximate variants)
    variants = []
    delays = [0, 3, -3, 6, -6, 9, -9, 12, -12, 15, -15, 18]
    scales = [1.0, 0.9, 0.95, 1.05, 0.85, 0.8, 1.1, 0.7, 1.2, 0.9, 0.95, 1.0]
    for d, s in zip(delays, scales):
        variants.append(mk_variant(delay_ms=d, scale=s, noise=0.006))
    data = np.column_stack(variants)
    return data, fs

# ---------------------------
# CSV IO helpers
# ---------------------------
def save_ecg_to_csv(path, data, fs, time_col=True, header_comment=True):
    """
    Save ECG numpy (n,channels) to CSV with optional time column.
    First line is a comment with sampling rate if header_comment True.
    Columns: time (if enabled), ch1,ch2,...
    """
    n = data.shape[0]
    channels = data.shape[1]
    df = pd.DataFrame(data, columns=[f"ch{i+1}" for i in range(channels)])
    if time_col:
        df.insert(0, "time", np.arange(n) / fs)
    
    import io as builtin_io
    output = builtin_io.StringIO()
    if header_comment:
        output.write(f"# SAMPLING_RATE={fs}\n")
    df.to_csv(output, index=False)
    return output.getvalue()

def load_ecg_from_csv(csv_content):
    """
    Load CSV and try to extract sampling rate:
    - If first line begins with '# SAMPLING_RATE=', parse it.
    - Else if 'time' column exists, compute sampling rate as 1/median(diff(time)).
    Returns (data (n,channels), fs, meta)
    """
    lines = csv_content.strip().split('\n')
    fs = None
    
    if lines[0].startswith("#"):
        if "SAMPLING_RATE=" in lines[0]:
            try:
                fs = float(lines[0].split("SAMPLING_RATE=")[1].strip())
            except Exception:
                fs = None
    
    # read with pandas
    import io as builtin_io
    df = pd.read_csv(builtin_io.StringIO(csv_content), comment='#')
    
    if 'time' in df.columns:
        time = df['time'].values
        diffs = np.diff(time)
        if len(diffs) > 0:
            fs_est = 1.0 / np.median(diffs)
            if fs is None:
                fs = fs_est
    
    # build numpy array of channels (exclude 'time' column)
    chcols = [c for c in df.columns if c != 'time']
    data = df[chcols].values.astype(float)
    return data, int(round(fs)) if fs is not None else None, {"columns": chcols}

# ---------------------------
# Fmax estimation
# ---------------------------
def estimate_fmax(audio, fs, energy_frac=0.995, max_fft_len=65536):
    """
    Estimate highest significant frequency (Hz) containing `energy_frac` of energy.
    audio: (n,) or (n,channels). Mix to mono.
    """
    if audio.ndim > 1:
        mono = audio.mean(axis=1)
    else:
        mono = audio
    N = len(mono)
    if N == 0:
        return 0.0
    L = min(N, max_fft_len)
    start = max(0, (N - L)//2)
    seg = mono[start:start+L] * np.hanning(L)
    X = np.fft.rfft(seg)
    power = np.abs(X)**2
    freqs = np.fft.rfftfreq(L, 1.0/fs)
    total = power.sum()
    if total <= 0:
        return 0.0
    cum = np.cumsum(power)
    idx = np.searchsorted(cum, energy_frac * total)
    idx = min(idx, len(freqs)-1)
    fmax = float(freqs[idx])
    # safety clamp
    fmax = min(fmax, 0.95 * (fs / 2.0))
    return fmax

# ---------------------------
# Resampling (safe/demo)
# ---------------------------
def _fix_length_1d(arr, target_len):
    if len(arr) > target_len:
        return arr[:target_len]
    if len(arr) < target_len:
        return np.concatenate([arr, np.zeros(target_len - len(arr), dtype=arr.dtype)])
    return arr

def resample_ecg(audio, orig_fs, target_fs, mode='safe', max_den=1000):
    """
    Resample ECG data.
    - audio: (n,) or (n,channels)
    - mode: 'safe' (anti-alias, default) or 'demo' (allow aliasing)
    Returns resampled array and new fs.
    """
    if orig_fs == target_fs:
        return audio, orig_fs

    ratio = float(target_fs) / float(orig_fs)
    out_len = int(round(audio.shape[0] * ratio))

    # Demo mode: integer decimation to produce clear aliasing
    if mode == 'demo':
        approx = orig_fs / target_fs
        nearest = int(round(approx))
        if abs(approx - nearest) < 1e-6 and nearest >= 1:
            # decimate by integer factor
            if audio.ndim == 1:
                out = audio[::nearest]
            else:
                out = audio[::nearest, :]
            return _fix_length_1d(out, out_len) if out.ndim==1 else out[:out_len,:], target_fs
        else:
            # non-integer ratio: use resample_poly but weaken the filter
            frac = Fraction(target_fs, orig_fs).limit_denominator(max_den)
            up, down = frac.numerator, frac.denominator
            if audio.ndim == 1:
                out = signal.resample_poly(audio, up, down, window=('kaiser', 0.1))
                out = _fix_length_1d(out, out_len)
            else:
                out = np.vstack([signal.resample_poly(audio[:,ch], up, down, window=('kaiser',0.1))
                                for ch in range(audio.shape[1])]).T
            return out, target_fs

    # SAFE mode: prefer libsamplerate if available
    if HAS_SAMPLERATE:
        try:
            conv = samplerate.Converter('sinc_best')
            out = conv.process(audio, ratio)
            if out.ndim == 1:
                out = _fix_length_1d(out, out_len)
            else:
                out = out[:out_len, :]
            return out, target_fs
        except Exception as e:
            print("samplerate failed, falling back:", e)

    # fallback: resample_poly with rational approx
    frac = Fraction(target_fs, orig_fs).limit_denominator(max_den)
    up, down = frac.numerator, frac.denominator
    if audio.ndim == 1:
        out = signal.resample_poly(audio, up, down)
        out = _fix_length_1d(out, out_len)
    else:
        out = np.vstack([signal.resample_poly(audio[:,ch], up, down) for ch in range(audio.shape[1])]).T
        out = out[:out_len, :]
    return out, target_fs

# ---------------------------
# Energy above a cutoff
# ---------------------------
def energy_above_cutoff(audio, fs, cutoff_hz, max_fft_len=65536):
    if audio.ndim > 1:
        mono = audio.mean(axis=1)
    else:
        mono = audio
    N = min(len(mono), max_fft_len)
    seg = mono[:N] * np.hanning(N)
    X = np.fft.rfft(seg)
    freqs = np.fft.rfftfreq(N, 1.0/fs)
    mask = freqs > cutoff_hz
    return np.sum(np.abs(X[mask])**2)