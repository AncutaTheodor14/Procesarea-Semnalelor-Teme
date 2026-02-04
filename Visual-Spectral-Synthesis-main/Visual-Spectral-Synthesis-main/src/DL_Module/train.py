import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64     
EPOCHS = 100      
LR = 0.001          
DATASET_SIZE = 10000 
MODEL_PATH = "audio_phasenet.pth"

N_FFT = 510
HOP_LENGTH = 128
SAMPLE_RATE = 16000

class AudioPhaseDatasetUltimate(Dataset):
    def __init__(self, length=10000):
        print(f"Generating DATASET ({length} samples)...")
        print("Includes: Static Sinusoids + Chirps (Variable Frequencies).")
        self.data_cache = []
        
        t = np.linspace(0, 1, SAMPLE_RATE)
        
        for i in range(length):
            if i % 1000 == 0 and i > 0:
                print(f"  -> Generating {i}/{length}...")

            waveform = np.zeros_like(t)
            
            signal_type = np.random.choice([0, 1])
            num_components = np.random.randint(1, 4) 

            for _ in range(num_components):
                amp = np.random.uniform(0.1, 1.0)
                phase_offset = np.random.uniform(0, 2*np.pi)
                
                if signal_type == 0:
                    freq = np.random.uniform(100, 4000)
                    waveform += amp * np.sin(2 * np.pi * freq * t + phase_offset)
                else:
                    f_start = np.random.uniform(100, 3000)
                    f_end = np.random.uniform(100, 3000)
                    
                    k = (f_end - f_start) / 1.0
                    phase_inst = 2 * np.pi * (f_start * t + (k/2) * t**2)
                    waveform += amp * np.sin(phase_inst + phase_offset)
            
            waveform = torch.tensor(waveform, dtype=torch.float32)
            window = torch.hann_window(N_FFT)
            stft = torch.stft(waveform, n_fft=N_FFT, hop_length=HOP_LENGTH, window=window, return_complex=True)
            
            stft = stft[:128, :128]
            if stft.shape[1] < 128:
                pad = 128 - stft.shape[1]
                stft = torch.nn.functional.pad(stft, (0, pad))
                
            mag = torch.abs(stft)
            mag = torch.log1p(mag)
            mag = mag / (mag.max() + 1e-6)
            
            phase = torch.angle(stft)
            target = torch.stack([torch.cos(phase), torch.sin(phase)], dim=0)
            
            self.data_cache.append((mag.unsqueeze(0), target))
            
        print("Generation complete! Dataset ready.")

    def __len__(self):
        return len(self.data_cache)

    def __getitem__(self, idx):
        return self.data_cache[idx]

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

def train():
    print(f"=== TRAINING on {DEVICE} ===")
    
    dataset = AudioPhaseDatasetUltimate(length=DATASET_SIZE)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    model = PhaseUNet().to(DEVICE)
    start_epoch = 0
    
    if os.path.exists(MODEL_PATH):
        print(f"Loading existing model: {MODEL_PATH}")
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            print("Success! Continuing refinement.")
        except:
            print("Error loading. Starting from scratch.")
    
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    criterion = nn.MSELoss()
    
    loss_history = []
    
    model.train()
    print("-" * 50)
    
    try:
        for epoch in range(EPOCHS):
            epoch_loss = 0
            
            for i, (mag, target) in enumerate(loader):
                mag, target = mag.to(DEVICE), target.to(DEVICE)
                
                optimizer.zero_grad()
                pred = model(mag)
                loss = criterion(pred, target)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(loader)
            loss_history.append(avg_loss)
            
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch [{epoch+1}/{EPOCHS}] | Loss: {avg_loss:.5f} | LR: {current_lr:.6f}")
            
            scheduler.step(avg_loss)
            
            if (epoch + 1) % 5 == 0:
                torch.save(model.state_dict(), MODEL_PATH)

    except KeyboardInterrupt:
        print("\nManual stop. Saving...")

    torch.save(model.state_dict(), MODEL_PATH)
    print("Training terminated.")
    
    # Save loss history
    np.savetxt('loss_history.txt', loss_history)
    print(f"Loss history saved to loss_history.txt")
    
    # Convert loss to percentage
    loss_percent = [loss * 100 for loss in loss_history]
    
    plt.figure(figsize=(10, 6))
    plt.plot(loss_percent, linewidth=2)
    plt.yscale('log')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (%)', fontsize=12)
    plt.title('Loss Curve', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('training_loss.png', dpi=300, bbox_inches='tight')
    print(f"Loss graph saved to training_loss.png")
    plt.show()

if __name__ == "__main__":
    train()