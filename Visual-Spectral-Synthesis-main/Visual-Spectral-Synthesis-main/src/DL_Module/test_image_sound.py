import torch
import torch.nn as nn
import numpy as np
import cv2
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt

IMAGE_PATH = "test_images/fractal.png" 
MODEL_PATH = "audio_phasenet.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

N_FFT = 510
HOP_LENGTH = 128

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
        return out

def process_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error can not find image path: {img_path}")
        return None
    
    img = cv2.resize(img, (128, 128))
    
    edges = cv2.Canny(img, 100, 200)
    
    edges = cv2.flip(edges, 0)
    
    img_float = edges.astype(np.float32) / 255.0
    return img_float

def griffin_lim_hybrid(magnitude, init_phase_complex, n_iters=30):
    window = torch.hann_window(N_FFT).to(DEVICE)
    
    expected_freq = N_FFT // 2 + 1  # 256
    current_freq = magnitude.shape[0] # 128
    pad_freq = expected_freq - current_freq # 128
    
    if pad_freq > 0:
        magnitude_full = torch.nn.functional.pad(magnitude, (0, 0, 0, pad_freq))
        complex_spec = torch.nn.functional.pad(init_phase_complex, (0, 0, 0, pad_freq))
    else:
        magnitude_full = magnitude
        complex_spec = init_phase_complex

    for _ in range(n_iters):
        waveform = torch.istft(complex_spec, n_fft=N_FFT, hop_length=HOP_LENGTH, window=window)
        
        new_complex = torch.stft(waveform, n_fft=N_FFT, hop_length=HOP_LENGTH, window=window, return_complex=True)
        
        min_time = min(new_complex.shape[1], magnitude_full.shape[1])
        new_complex = new_complex[:, :min_time]
        magnitude_crop = magnitude_full[:, :min_time]
        
        phase = torch.angle(new_complex)
        
        complex_spec = magnitude_crop * torch.exp(1j * phase)
        
    return torch.istft(complex_spec, n_fft=N_FFT, hop_length=HOP_LENGTH, window=window)

def main():
    print(f"Processing image: {IMAGE_PATH}")
    img_input = process_image(IMAGE_PATH)
    if img_input is None: return

    model = PhaseUNet().to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    mag_tensor = torch.tensor(img_input).to(DEVICE)
    mag_input_4d = mag_tensor.unsqueeze(0).unsqueeze(0) 
    
    with torch.no_grad():
        pred = model(mag_input_4d)

    cos_p = pred[0, 0]
    sin_p = pred[0, 1]
    
    print("Generating sound (Hybrid Method)...")
    complex_ai = mag_tensor * torch.complex(cos_p, sin_p)
    
    waveform = griffin_lim_hybrid(mag_tensor, complex_ai, n_iters=30)
    
    audio_np = waveform.cpu().numpy()
    max_val = np.max(np.abs(audio_np))
    if max_val > 0:
        audio_np = audio_np / max_val
    
    out_name = "blackhole_sound.wav"
    wav.write(out_name, 16000, audio_np)
    print(f"Success! File saved: {out_name}")

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Image Contours")
    plt.imshow(img_input, origin='lower', cmap='inferno')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title("Generated Waveform")
    plt.plot(audio_np)
    plt.ylim(-1, 1)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()