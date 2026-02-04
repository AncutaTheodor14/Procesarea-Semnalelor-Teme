import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from pathlib import Path
import time
import json
import torch.nn as nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_FFT = 510
HOP_LENGTH = 128
SAMPLE_RATE = 16000
OUTPUT_DIR = Path("experimental_results")
OUTPUT_DIR.mkdir(exist_ok=True)


def generate_test_signal(signal_type='chirp', duration=1.0):
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration))
    
    if signal_type == 'pure_tone':
        signal = np.sin(2 * np.pi * 440 * t)
    elif signal_type == 'chirp':
        f0, f1 = 200, 2000
        k = (f1 - f0) / duration
        phase = 2 * np.pi * (f0 * t + (k/2) * t**2)
        signal = np.sin(phase)
    elif signal_type == 'multi_tone':
        signal = (np.sin(2*np.pi*300*t) + 
                0.7*np.sin(2*np.pi*600*t) + 
                0.5*np.sin(2*np.pi*1200*t))
        signal /= signal.max()
    elif signal_type == 'training_like':
        np.random.seed(42)
        signal = np.zeros_like(t)
        num_components = np.random.randint(1, 4)
        for _ in range(num_components):
            amp = np.random.uniform(0.3, 1.0)
            freq = np.random.uniform(200, 2000)
            phase_offset = np.random.uniform(0, 2*np.pi)
            signal += amp * np.sin(2 * np.pi * freq * t + phase_offset)
        signal = signal / (np.abs(signal).max() + 1e-6)
    else:
        raise ValueError(f"Unknown signal type: {signal_type}")
    
    return signal

def compute_stft_magnitude(signal):
    signal_tensor = torch.tensor(signal, dtype=torch.float32)
    window = torch.hann_window(N_FFT)
    stft = torch.stft(signal_tensor, n_fft=N_FFT, hop_length=HOP_LENGTH, 
                    window=window, return_complex=True, center=True, pad_mode='reflect')
    magnitude = torch.abs(stft)
    return magnitude

def griffin_lim(magnitude, n_iters=100, verbose=False):
    window = torch.hann_window(N_FFT)
    
    phase = torch.zeros_like(magnitude)
    complex_spec = magnitude * torch.exp(1j * phase)
    
    for i in range(n_iters):
        waveform = torch.istft(complex_spec, n_fft=N_FFT, hop_length=HOP_LENGTH, 
                            window=window, center=True)
        
        new_complex = torch.stft(waveform, n_fft=N_FFT, hop_length=HOP_LENGTH, 
                                window=window, return_complex=True, center=True, pad_mode='reflect')
        
        min_time = min(new_complex.shape[1], magnitude.shape[1])
        new_complex = new_complex[:, :min_time]
        mag_crop = magnitude[:, :min_time]
        
        phase = torch.angle(new_complex)
        complex_spec = mag_crop * torch.exp(1j * phase)
        
        if verbose and (i+1) % 20 == 0:
            print(f"  GL Iteration {i+1}/{n_iters}")
    
    return torch.istft(complex_spec, n_fft=N_FFT, hop_length=HOP_LENGTH, window=window, center=True)

def fast_griffin_lim(magnitude, n_iters=30, momentum=0.9, verbose=False):
    window = torch.hann_window(N_FFT)
    
    phase = torch.zeros_like(magnitude)
    complex_spec = magnitude * torch.exp(1j * phase)
    prev_complex_spec = complex_spec.clone()
    
    for i in range(n_iters):
        waveform = torch.istft(complex_spec, n_fft=N_FFT, hop_length=HOP_LENGTH, 
                            window=window, center=True)
        
        new_complex = torch.stft(waveform, n_fft=N_FFT, hop_length=HOP_LENGTH, 
                                window=window, return_complex=True, center=True, pad_mode='reflect')
        
        min_time = min(new_complex.shape[1], magnitude.shape[1])
        new_complex = new_complex[:, :min_time]
        mag_crop = magnitude[:, :min_time]
        
        phase = torch.angle(new_complex)
        complex_spec_new = mag_crop * torch.exp(1j * phase)
        
        complex_spec_accelerated = complex_spec_new + momentum * (complex_spec_new - prev_complex_spec)
        prev_complex_spec = complex_spec_new
        complex_spec = complex_spec_accelerated
        
        if verbose and (i+1) % 10 == 0:
            print(f"  FGL Iteration {i+1}/{n_iters}")
    
    return torch.istft(complex_spec, n_fft=N_FFT, hop_length=HOP_LENGTH, window=window, center=True)

def align_signals(original, reconstructed):
    min_len = min(len(original), len(reconstructed))
    original = original[:min_len]
    reconstructed = reconstructed[:min_len]
    
    correlation = np.correlate(original, reconstructed, mode='full')
    lag = np.argmax(np.abs(correlation)) - (len(original) - 1)
    
    if lag > 0:
        aligned_reconstructed = np.pad(reconstructed, (lag, 0))[:min_len]
        aligned_original = original
    elif lag < 0:
        aligned_original = np.pad(original, (-lag, 0))[:min_len]
        aligned_reconstructed = reconstructed
    else:
        aligned_original = original
        aligned_reconstructed = reconstructed
    
    final_len = min(len(aligned_original), len(aligned_reconstructed))
    return aligned_original[:final_len], aligned_reconstructed[:final_len]

def compute_snr(original, reconstructed):
    aligned_orig, aligned_rec = align_signals(original, reconstructed)
    
    scale = np.dot(aligned_orig, aligned_rec) / (np.dot(aligned_rec, aligned_rec) + 1e-10)
    aligned_rec = aligned_rec * scale
    
    noise = aligned_orig - aligned_rec
    
    signal_power = np.mean(aligned_orig ** 2)
    noise_power = np.mean(noise ** 2)
    
    if noise_power < 1e-10:
        return 100.0
    
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

def compute_lsd(magnitude_target, magnitude_reconstructed):
    min_time = min(magnitude_target.shape[1], magnitude_reconstructed.shape[1])
    mag_t = magnitude_target[:, :min_time].numpy()
    mag_r = magnitude_reconstructed[:, :min_time].numpy()
    
    log_diff = np.log10(mag_t + 1e-10) - np.log10(mag_r + 1e-10)
    lsd = np.mean(np.sqrt(np.mean(log_diff ** 2, axis=0)))
    return lsd * 20

def compute_spectral_convergence(magnitude_target, magnitude_reconstructed):
    min_time = min(magnitude_target.shape[1], magnitude_reconstructed.shape[1])
    mag_t = magnitude_target[:, :min_time]
    mag_r = magnitude_reconstructed[:, :min_time]
    diff = mag_t - mag_r
    convergence = torch.norm(diff, p='fro') / torch.norm(mag_t, p='fro')
    return convergence.item()

class PhaseUNet(nn.Module):
    def __init__(self):
        super(PhaseUNet, self).__init__()
        self.e1 = nn.Conv2d(1, 64, 4, 2, 1) 
        self.e2 = nn.Sequential(nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2))
        self.e3 = nn.Sequential(nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2))
        self.e4 = nn.Sequential(nn.Conv2d(256, 512, 4, 2, 1), nn.BatchNorm2d(512), nn.LeakyReLU(0.2))
        
        self.d1 = nn.Sequential(nn.ConvTranspose2d(512, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU())
        self.d2 = nn.Sequential(nn.ConvTranspose2d(512, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU())
        self.d3 = nn.Sequential(nn.ConvTranspose2d(256, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU())
        
        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, 2, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        e1 = self.e1(x); e2 = self.e2(e1); e3 = self.e3(e2); e4 = self.e4(e3)
        d1 = self.d1(e4); d1 = torch.cat([d1, e3], 1)
        d2 = self.d2(d1); d2 = torch.cat([d2, e2], 1)
        d3 = self.d3(d2); d3 = torch.cat([d3, e1], 1)
        out = self.final(d3)
        return torch.nn.functional.normalize(out, p=2, dim=1)

def load_phasenet():
    model = PhaseUNet().to(DEVICE)
    model_path = Path(__file__).parent / "audio_phasenet.pth"
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
        model.eval()
        return model
    return None

def phasenet_reconstruct_proper(original_signal, model):
    signal_tensor = torch.tensor(original_signal, dtype=torch.float32)
    window = torch.hann_window(N_FFT)
    stft = torch.stft(signal_tensor, n_fft=N_FFT, hop_length=HOP_LENGTH, 
                    window=window, return_complex=True, center=True, pad_mode='reflect')
    
    stft_cropped = stft[:128, :128]
    if stft_cropped.shape[1] < 128:
        pad = 128 - stft_cropped.shape[1]
        stft_cropped = torch.nn.functional.pad(stft_cropped, (0, pad))
    
    mag = torch.abs(stft_cropped)
    mag_normalized = torch.log1p(mag)
    mag_normalized = mag_normalized / (mag_normalized.max() + 1e-6)
    
    mag_2d = mag_normalized.unsqueeze(0).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        phase_pred = model(mag_2d)
    
    cos_pred = phase_pred[0, 0].cpu()
    sin_pred = phase_pred[0, 1].cpu()
    
    real_part = mag * cos_pred
    imag_part = mag * sin_pred
    complex_spec_pred = torch.complex(real_part, imag_part)
    
    full_stft = torch.zeros_like(stft)
    h = min(128, full_stft.shape[0])
    w = min(128, full_stft.shape[1])
    full_stft[:h, :w] = complex_spec_pred[:h, :w]
    
    waveform = torch.istft(full_stft, n_fft=N_FFT, hop_length=HOP_LENGTH, window=window, center=True)
    return waveform, mag, stft_cropped

def phasenet_reconstruct(magnitude, model):
    h, w = magnitude.shape
    
    input_h = min(128, h)
    input_w = min(128, w)
    mag_input = magnitude[:input_h, :input_w].clone()
    
    if mag_input.shape[0] < 128:
        pad_h = 128 - mag_input.shape[0]
        mag_input = torch.nn.functional.pad(mag_input, (0, 0, 0, pad_h))
    if mag_input.shape[1] < 128:
        pad_w = 128 - mag_input.shape[1]
        mag_input = torch.nn.functional.pad(mag_input, (0, pad_w))
    
    mag_normalized = torch.log1p(mag_input)
    mag_normalized = mag_normalized / (mag_normalized.max() + 1e-6)
    
    mag_2d = mag_normalized.unsqueeze(0).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        phase_pred = model(mag_2d)
    
    cos_pred = phase_pred[0, 0, :input_h, :input_w]
    sin_pred = phase_pred[0, 1, :input_h, :input_w]
    
    full_cos = torch.zeros_like(magnitude)
    full_sin = torch.zeros_like(magnitude)
    full_cos[:input_h, :input_w] = cos_pred.cpu()
    full_sin[:input_h, :input_w] = sin_pred.cpu()
    
    real_part = magnitude * full_cos
    imag_part = magnitude * full_sin
    complex_spec = torch.complex(real_part, imag_part)
    
    window = torch.hann_window(N_FFT)
    waveform = torch.istft(complex_spec, n_fft=N_FFT, hop_length=HOP_LENGTH, window=window, center=True)
    return waveform

def experiment_1_convergence_comparison():
    original_signal = generate_test_signal('chirp')
    magnitude = compute_stft_magnitude(original_signal)
    
    gl_iters = [10, 20, 30, 50, 70, 100]
    gl_errors = []
    
    for n in gl_iters:
        reconstructed = griffin_lim(magnitude, n_iters=n)
        mag_reconstructed = compute_stft_magnitude(reconstructed.numpy())
        error = compute_spectral_convergence(magnitude, mag_reconstructed)
        gl_errors.append(error)
        print(f"  GL {n} iters: Error = {error:.4f}")
    
    fgl_iters = [5, 10, 15, 20, 25, 30]
    fgl_errors = []
    
    print("Testing Fast Griffin-Lim...")
    for n in fgl_iters:
        reconstructed = fast_griffin_lim(magnitude, n_iters=n)
        mag_reconstructed = compute_stft_magnitude(reconstructed.numpy())
        error = compute_spectral_convergence(magnitude, mag_reconstructed)
        fgl_errors.append(error)
        print(f"  FGL {n} iters: Error = {error:.4f}")
    
    plt.figure(figsize=(10, 6))
    plt.plot(gl_iters, gl_errors, 'o-', linewidth=2, markersize=8, label='Griffin-Lim')
    plt.plot(fgl_iters, fgl_errors, 's-', linewidth=2, markersize=8, label='Fast Griffin-Lim')
    plt.xlabel('Număr Iterații', fontsize=12)
    plt.ylabel('Spectral Convergence (Eroare)', fontsize=12)
    plt.title('Comparație Convergență: GL vs FGL', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = OUTPUT_DIR / "exp1_convergence_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()
    
    data = {
        'gl_iters': gl_iters,
        'gl_errors': gl_errors,
        'fgl_iters': fgl_iters,
        'fgl_errors': fgl_errors
    }
    with open(OUTPUT_DIR / "exp1_data.json", 'w') as f:
        json.dump(data, f, indent=2)


def experiment_2_execution_time():
    print("\n" + "="*60)
    print("EXPERIMENT 2: Execution Time Benchmark")
    print("="*60)
    
    original_signal = generate_test_signal('chirp')
    magnitude = compute_stft_magnitude(original_signal)
    
    methods = {
        'GL-30': lambda: griffin_lim(magnitude, n_iters=30),
        'GL-50': lambda: griffin_lim(magnitude, n_iters=50),
        'GL-100': lambda: griffin_lim(magnitude, n_iters=100),
        'FGL-30': lambda: fast_griffin_lim(magnitude, n_iters=30),
    }
    
    times = {}
    snr_values = {}
    lsd_values = {}
    sc_values = {}
    
    for name, method in methods.items():
        print(f"\nTesting {name}...")
        
        _ = method()
        
        start = time.time()
        result = method()
        elapsed = time.time() - start
        
        times[name] = elapsed
        reconstructed = result
        
        snr = compute_snr(original_signal, reconstructed.numpy())
        snr_values[name] = snr
        
        mag_reconstructed = compute_stft_magnitude(reconstructed.numpy())
        lsd = compute_lsd(magnitude, mag_reconstructed)
        sc = compute_spectral_convergence(magnitude, mag_reconstructed)
        
        lsd_values[name] = lsd
        sc_values[name] = sc
        
        print(f"  Time: {elapsed:.3f}s | SNR: {snr:.2f} dB | LSD: {lsd:.2f} dB | SC: {sc:.4f}")
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6'][:len(times)]
    
    ax1.bar(times.keys(), times.values(), color=colors)
    ax1.set_ylabel('Timp Execuție (secunde)', fontsize=12)
    ax1.set_title('Benchmark Timp de Execuție', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    ax2.bar(snr_values.keys(), snr_values.values(), color=colors)
    ax2.set_ylabel('SNR (dB)', fontsize=12)
    ax2.set_title('Calitate Audio (SNR ↑ = mai bine)', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    ax3.bar(lsd_values.keys(), lsd_values.values(), color=colors)
    ax3.set_ylabel('LSD (dB)', fontsize=12)
    ax3.set_title('Log-Spectral Distance (↓ = mai bine)', fontsize=14, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "exp2_benchmark.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {output_path}")
    plt.close()
    
    data = {
        'times': {k: float(v) for k, v in times.items()},
        'snr': {k: float(v) for k, v in snr_values.items()},
        'lsd': {k: float(v) for k, v in lsd_values.items()}
    }
    with open(OUTPUT_DIR / "exp2_data.json", 'w') as f:
        json.dump(data, f, indent=2)


def experiment_3_spectrogram_comparison():
    original_signal = generate_test_signal('chirp', duration=2.0)
    mag_original = compute_stft_magnitude(original_signal)
    
    reconstructed_gl = griffin_lim(mag_original, n_iters=100, verbose=True)
    mag_gl = compute_stft_magnitude(reconstructed_gl.numpy())
    
    reconstructed_fgl = fast_griffin_lim(mag_original, n_iters=30, verbose=True)
    mag_fgl = compute_stft_magnitude(reconstructed_fgl.numpy())
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes[0, 0].imshow(mag_original.numpy(), aspect='auto', origin='lower', cmap='inferno')
    axes[0, 0].set_title('Target Magnitude', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Frecvență (bins)', fontsize=10)
    
    axes[0, 1].imshow(mag_gl.numpy(), aspect='auto', origin='lower', cmap='inferno')
    axes[0, 1].set_title('GL-100 Reconstructed', fontsize=12, fontweight='bold')
    
    axes[0, 2].imshow(mag_fgl.numpy(), aspect='auto', origin='lower', cmap='inferno')
    axes[0, 2].set_title('FGL-30 Reconstructed', fontsize=12, fontweight='bold')
    
    time_axis_orig = np.linspace(0, len(original_signal)/SAMPLE_RATE, len(original_signal))
    time_axis_gl = np.linspace(0, len(reconstructed_gl)/SAMPLE_RATE, len(reconstructed_gl))
    time_axis_fgl = np.linspace(0, len(reconstructed_fgl)/SAMPLE_RATE, len(reconstructed_fgl))
    
    axes[1, 0].plot(time_axis_orig, original_signal, linewidth=0.5)
    axes[1, 0].set_title('Original Waveform', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Timp (s)', fontsize=10)
    axes[1, 0].set_ylabel('Amplitudine', fontsize=10)
    axes[1, 0].set_ylim(-1.2, 1.2)
    
    axes[1, 1].plot(time_axis_gl, reconstructed_gl.numpy(), linewidth=0.5, color='orange')
    axes[1, 1].set_title(f'GL-100 (SNR: {compute_snr(original_signal, reconstructed_gl.numpy()):.1f} dB)', fontsize=12)
    axes[1, 1].set_xlabel('Timp (s)', fontsize=10)
    axes[1, 1].set_ylim(-1.2, 1.2)
    
    axes[1, 2].plot(time_axis_fgl, reconstructed_fgl.numpy(), linewidth=0.5, color='green')
    axes[1, 2].set_title(f'FGL-30 (SNR: {compute_snr(original_signal, reconstructed_fgl.numpy()):.1f} dB)', fontsize=12)
    axes[1, 2].set_xlabel('Timp (s)', fontsize=10)
    axes[1, 2].set_ylim(-1.2, 1.2)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "exp3_spectrogram_waveform_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

def experiment_4_training_loss():
    try:
        with open("training_loss_history.json", 'r') as f:
            loss_history = json.load(f)
        
        plt.figure(figsize=(10, 6))
        plt.plot(loss_history, linewidth=2, color='#e74c3c')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('MSE Loss', fontsize=12)
        plt.title('Curba de Învățare (PhaseUNet)', fontsize=14, fontweight='bold')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        output_path = OUTPUT_DIR / "exp4_training_loss.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        plt.close()
        
    except FileNotFoundError:
        print("⚠ Training loss history not found. Skipping.")
        print("  Modifică train.py să salveze loss_history într-un JSON:")
        print("  with open('training_loss_history.json', 'w') as f:")
        print("      json.dump(loss_history, f)")

def experiment_5_frequency_error_analysis():
    test_freqs = [200, 500, 1000, 2000, 3000, 4000]
    gl_errors = []
    fgl_errors = []
    
    for freq in test_freqs:
        print(f"Testing frequency: {freq} Hz")
        
        t = np.linspace(0, 1, SAMPLE_RATE)
        signal = np.sin(2 * np.pi * freq * t)
        magnitude = compute_stft_magnitude(signal)
        
        reconstructed_gl = griffin_lim(magnitude, n_iters=50)
        mag_gl = compute_stft_magnitude(reconstructed_gl.numpy())
        error_gl = compute_spectral_convergence(magnitude, mag_gl)
        gl_errors.append(error_gl)
        
        reconstructed_fgl = fast_griffin_lim(magnitude, n_iters=30)
        mag_fgl = compute_stft_magnitude(reconstructed_fgl.numpy())
        error_fgl = compute_spectral_convergence(magnitude, mag_fgl)
        fgl_errors.append(error_fgl)
    
    plt.figure(figsize=(10, 6))
    plt.plot(test_freqs, gl_errors, 'o-', linewidth=2, markersize=8, label='GL-50')
    plt.plot(test_freqs, fgl_errors, 's-', linewidth=2, markersize=8, label='FGL-30')
    plt.xlabel('Frecvență (Hz)', fontsize=12)
    plt.ylabel('Spectral Convergence', fontsize=12)
    plt.title('Eroare în Funcție de Frecvență', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = OUTPUT_DIR / "exp5_frequency_error.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

def main():
    experiment_1_convergence_comparison()
    experiment_2_execution_time()
    experiment_3_spectrogram_comparison()
    experiment_4_training_loss()
    experiment_5_frequency_error_analysis()
    
    print("\n" + "="*60)
    print(" ✓ ALL EXPERIMENTS COMPLETED ".center(60))
    print("="*60)
    print(f"\nGrafice salvate în: {OUTPUT_DIR.absolute()}")
    print("\nAdaugă aceste fișiere în LaTeX:")
    print("  - exp1_convergence_comparison.png")
    print("  - exp2_benchmark.png")
    print("  - exp3_spectrogram_waveform_comparison.png")
    print("  - exp4_training_loss.png")
    print("  - exp5_frequency_error.png")

if __name__ == "__main__":
    main()