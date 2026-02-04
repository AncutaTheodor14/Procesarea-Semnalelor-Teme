import numpy as np
import fft_operations as fft_ops

def fast_griffin_lim(spectrogram, nr_iteratii, n_fft, hop_length, window_type, alpha):
    #spectrogram - spectrograma pe care vrem sa o transformam in sunet
    #nr_iteratii - numarul de iteratii ca sa gasim faza cat mai exacta
    #initializam faza cu o valoare aleatoare
    faza_curenta = 2*np.pi*np.random.rand(spectrogram.shape[0], spectrogram.shape[1]) - np.pi #luam un unghi intre -pi si pi
    #construim spectrul intial cu faza intitializata mai sus si magnitudinile pe care le avem corecte de la inceput
    #deci acum spectrograma noastra contine si informatii despre faza, directie
    spectru_estimat = spectrogram * np.exp(1j*faza_curenta)
    spectru_anterior = spectru_estimat.copy()
    for i in range(nr_iteratii):
        #trecem in domeniu timp cu faza curenta
        semnal_estimat = fft_ops.istft(spectru_estimat, n_fft, hop_length, window_type)
        #trecem inapoi in domeniul frecventa(la spectrograma), dupa ce mai sus avem un semnal real cu o faza reala si deja istft a rezolvat
        #problema overlapului, adunand valorile care se suprapuneau. spectru_estimat nu putea decide ce face cu aceste 2 valori
        spectru_nou = fft_ops.stft(semnal_estimat, n_fft, hop_length, window_type)
        #ne asiguram ca de la erorile de rotunjire facand istft apoi stft nu avem vreo coloana in plus
        spectru_nou = spectru_nou[:spectrogram.shape[0], :spectrogram.shape[1]]
        #in loc sa folosim spectrul nou, il acceleram in directia in care ne indica spectru_nou fata de anterior
        spectru_accelerat = spectru_nou + alpha * (spectru_nou - spectru_anterior)
        spectru_anterior = spectru_nou.copy()
        #pastram faza care a venit din semnalul real, dar distrugem magnitudinea nou creata prin stft si o inlocuim cu spectrograma noastra initiala
        #impartim spectrul la maginitudinea lui si ramane doar directia (faza)
        faza_unitara = spectru_accelerat / (np.abs(spectru_accelerat)+ 1e-10)
        spectru_estimat = spectrogram * faza_unitara
    semnal_final = fft_ops.istft(spectru_estimat, n_fft, hop_length, window_type)
    return semnal_final