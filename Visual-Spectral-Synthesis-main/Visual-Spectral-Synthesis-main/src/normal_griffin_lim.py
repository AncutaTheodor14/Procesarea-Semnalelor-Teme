import numpy as np
import fft_operations as fft_ops

def griffin_lim(spectrogram, nr_iteratii, n_fft, hop_length, window_type, init_complex=None):
    
    # --- Partea de inițializare (Hybrid sau Random) ---
    if init_complex is not None:
        print(" -> Inițializare Hibridă (GAN)...")
        faza_initiala = np.angle(init_complex)
        spectru_estimat = spectrogram * np.exp(1j * faza_initiala)
    else:
        print(" -> Inițializare Random (Standard)...")
        faza_curenta = 2*np.pi*np.random.rand(spectrogram.shape[0], spectrogram.shape[1]) - np.pi 
        spectru_estimat = spectrogram * np.exp(1j*faza_curenta)

    # --- Bucla Principală ---
    for i in range(nr_iteratii):
        # AICI ADĂUGĂM PRINT-UL
        print(f"   [Griffin-Lim] Procesăm iterația {i+1} din {nr_iteratii}...")

        semnal_estimat = fft_ops.istft(spectru_estimat, n_fft, hop_length, window_type)
        spectru_nou = fft_ops.stft(semnal_estimat, n_fft, hop_length, window_type)
        
        # Ajustare dimensiuni
        min_rows = min(spectrogram.shape[0], spectru_nou.shape[0])
        min_cols = min(spectrogram.shape[1], spectru_nou.shape[1])
        spectru_nou = spectru_nou[:min_rows, :min_cols]
        spectrogram_cut = spectrogram[:min_rows, :min_cols]

        faza_unitara = spectru_nou / (np.abs(spectru_nou)+ 1e-10)
        spectru_estimat = spectrogram_cut * faza_unitara
        
    semnal_final = fft_ops.istft(spectru_estimat, n_fft, hop_length, window_type)
    return semnal_final