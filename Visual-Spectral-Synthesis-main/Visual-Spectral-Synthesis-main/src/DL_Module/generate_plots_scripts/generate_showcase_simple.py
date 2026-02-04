import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import scipy.io.wavfile as wav

N_FFT = 510
HOP_LENGTH = 128
SAMPLE_RATE = 16000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

try:
    from DL_Module.phasenet import PhaseUNet, process_image, griffin_lim_hybrid
    print("✓ Successfully imported from test_image_sound.py")
except ImportError:
    print("✗ Error: Cannot import from test_image_sound.py")
    print("Make sure test_image_sound.py is in the same directory!")
    sys.exit(1)

def compute_spectrogram(waveform):
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.cpu()
    else:
        waveform = torch.tensor(waveform)
    
    window = torch.hann_window(N_FFT)
    stft = torch.stft(waveform, n_fft=N_FFT, hop_length=HOP_LENGTH, 
                    window=window, return_complex=True)
    magnitude = torch.abs(stft)
    magnitude_db = 20 * torch.log10(magnitude + 1e-6)
    return magnitude_db.numpy()

def load_model(model_path="audio_phasenet.pth"):
    model = PhaseUNet().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model

def generate_showcase(image_path, output_name, model):
    print(f"\n{'='*70}")
    print(f"Generating showcase: {output_name}")
    print(f"{'='*70}")
    
    print("  [1/4] Processing image...")
    img_processed = process_image(image_path)
    
    print("  [2/4] Running PhaseUNet + Griffin-Lim...")
    mag_tensor = torch.tensor(img_processed).to(DEVICE)
    mag_input_4d = mag_tensor.unsqueeze(0).unsqueeze(0)
    
    with torch.no_grad():
        pred = model(mag_input_4d)
    
    cos_p = pred[0, 0]
    sin_p = pred[0, 1]
    complex_ai = mag_tensor * torch.complex(cos_p, sin_p)
    
    waveform = griffin_lim_hybrid(mag_tensor, complex_ai, n_iters=30)
    audio_np = waveform.cpu().numpy()
    
    max_val = np.max(np.abs(audio_np))
    if max_val > 0:
        audio_np = audio_np / max_val
    
    print("  [3/4] Computing spectrogram...")
    spectrogram = compute_spectrogram(audio_np)
    
    print("  [4/4] Creating figure...")
    fig = plt.figure(figsize=(16, 10))
    
    ax1 = plt.subplot(2, 2, 1)
    original_img = cv2.imread(str(image_path))
    if original_img is not None:
        original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        ax1.imshow(original_img_rgb)
    else:
        ax1.imshow(np.zeros((128, 128, 3)), cmap='gray')
    ax1.set_title('(a) Input Image', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    ax2 = plt.subplot(2, 2, 2)
    ax2.imshow(img_processed, cmap='gray', origin='upper')
    ax2.set_title('(b) Canny Edge Detection', fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    ax3 = plt.subplot(2, 2, 3)
    time_axis = np.linspace(0, len(audio_np)/SAMPLE_RATE, len(audio_np))
    ax3.plot(time_axis, audio_np, linewidth=0.5, color='#2c3e50')
    ax3.set_title('(c) Generated Waveform', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Time (seconds)', fontsize=11)
    ax3.set_ylabel('Amplitude', fontsize=11)
    ax3.set_ylim(-1.1, 1.1)
    ax3.grid(True, alpha=0.3)
    
    ax4 = plt.subplot(2, 2, 4)
    im = ax4.imshow(spectrogram[:128, :], aspect='auto', origin='lower', 
                    cmap='inferno', vmin=-60, vmax=0)
    ax4.set_title('(d) Reconstructed Spectrogram', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Time (frames)', fontsize=11)
    ax4.set_ylabel('Frequency (bins)', fontsize=11)
    
    cbar = plt.colorbar(im, ax=ax4, orientation='vertical')
    cbar.set_label('Magnitude (dB)', fontsize=10)
    
    plt.tight_layout()
    output_path = f"{output_name}_showcase.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {output_path}")
    plt.close()
    
    audio_path = f"{output_name}_audio.wav"
    wav.write(audio_path, SAMPLE_RATE, audio_np)
    print(f"✓ Saved: {audio_path}")
    
    return output_path


def main():
    if len(sys.argv) != 3:
        print("\nUSAGE:")
        print("  python generate_showcase_simple.py <name> <image_path>")
        print("\nEXAMPLES:")
        print("  python generate_showcase_simple.py blackhole test_images/blackhole.png")
        print("  python generate_showcase_simple.py diagonal test_images/diagonal.png")
        print("  python generate_showcase_simple.py fractal test_images/fractal.png")
        print("  python generate_showcase_simple.py spiral test_images/spiral.png")
        sys.exit(1)
    
    name = sys.argv[1]
    image_path = sys.argv[2]
    
    if not Path(image_path).exists():
        print(f"✗ Error: Image not found: {image_path}")
        sys.exit(1)
    
    print("Loading PhaseUNet model...")
    try:
        model = load_model()
        print("✓ Model loaded successfully!")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        sys.exit(1)
    
    output_file = generate_showcase(image_path, name, model)
    
    print("\n" + "="*70)
    print("✓ SHOWCASE GENERATION COMPLETE")
    print("="*70)
    print(f"\nGenerated file: {output_file}")
    print("\nUse in LaTeX:")
    print(f"  \\includegraphics[width=1.0\\textwidth]{{{output_file}}}")

if __name__ == "__main__":
    main()