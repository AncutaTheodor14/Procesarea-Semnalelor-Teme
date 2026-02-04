import numpy as np

def stft(semnal, n_fft, hop_length, window_type):
#semnal - semnalul pe care vrem sa aplicam stft
#n_fft - numarul de esantioane in timp(lungimea ferestrei) astfel incat sa obtinem 256 de frecvente cand facem fft
#        (i-am dat resize la imagine la 256 x 256). calculul este: n_fft/2 + 1 = 256. fv maxima pe care o atingem
#        e maxim fv de esantionare / 2 conform nyquist. n_fft = (256-1) * 2 = 255 * 2= 510
#hop_length - cate esantioane sarim pentru a ajunge la urmatorul interval in timp pe care facem fft (daca sarim cu 510,
#             inseamna ca luam intervalele fara suprapunere si pierdem informatie)
    n = len(semnal) #lungimea totala a semnalului
    n_frames = (n - n_fft)//hop_length + 1#va trebui sa fie 256, deoarece imaginea are 256 x 256 deci trebuie sa aplicam 256 de fft-uri in 
                                         #total, dar facem calculul pentru generalitate
    n_fv_bins = n_fft //2 + 1 #aici am calculat inaltimea maxima pe axa y (frecvente)
    spectrograma_stft = np.zeros((n_fv_bins, n_frames), dtype = complex)
    if window_type == 'dreptunghiulara':
        fereastra = np.ones(n_fft)
    elif window_type == 'hanning':
        fereastra = np.hanning(n_fft)
    elif window_type == 'hamming':
        fereastra = np.hamming(n_fft)
    elif window_type == 'blackman':
        fereastra = np.blackman(n_fft)
    elif window_type == 'flattop':
        n1 = np.arange(n_fft)
        fereastra = 0.22 - 0.42 * np.cos(2 * np.pi * n1 / n_fft) + 0.28 * np.cos(4 * np.pi * n1 / n_fft) - 0.08 * np.cos(6 * np.pi * n1 / n_fft) + 0.007 * np.cos(8 * np.pi * n1 / n_fft)
    else:
        raise ValueError("Tip necunoscut de fereastra")
    for t in range(n_frames): #atatea intervale avem
        #calculez indecsii de inceput si final al bucatii curente si o extrag
        start = t * hop_length
        end = start + n_fft
        bucata_curenta = semnal[start:end] #aici trebuie facut padding sa ne asiguram ca inclusiv ultima bucata are lungimea n_fft
        if len(bucata_curenta) < n_fft:
            lipsa = n_fft - len(bucata_curenta)
            bucata_curenta = np.pad(bucata_curenta, (0, lipsa), 'constant')
        bucata_curenta_windowed = bucata_curenta * fereastra
        frecvente = np.fft.rfft(bucata_curenta_windowed, n=n_fft)
        spectrograma_stft[:, t] = frecvente
    return spectrograma_stft #returnez matricea care are ca linii frecventele si coloanele timpul (asemanator cu o spectrograma)

def istft(spectrograma, n_fft, hop_length, window_type):
    n_fv_bins, n_frames = spectrograma.shape
    if window_type == 'dreptunghiulara':
        fereastra = np.ones(n_fft)
    elif window_type == 'hanning':
        fereastra = np.hanning(n_fft)
    elif window_type == 'hamming':
        fereastra = np.hamming(n_fft)
    elif window_type == 'blackman':
        fereastra = np.blackman(n_fft)
    elif window_type == 'flattop':
        n1 = np.arange(n_fft)
        fereastra = 0.22 - 0.42 * np.cos(2 * np.pi * n1 / n_fft) + 0.28 * np.cos(4 * np.pi * n1 / n_fft) - 0.08 * np.cos(6 * np.pi * n1 / n_fft) + 0.007 * np.cos(8 * np.pi * n1 / n_fft)
    else:
        raise ValueError("Tip necunoscut de fereastra")
    lungime_semnal = (n_frames - 1) * hop_length + n_fft #cand incepe ultimul frame + lungimea lui
    semnal = np.zeros(lungime_semnal) #semnalul pe care il reconstruiesc alipind fasiile pe care aplic ifft
    norm_buffer = np.zeros(lungime_semnal) #semnalul nostru a fost inmultit cu fereastra ^ 2 (o data la stft si o data la istft)
    for t in range(n_frames):              #si pentru ca adunam amplitudinile in unele zone, anume unde se intalnesc marginile o sa avem o valoare mai mare decat trebuie deci luam norm_buffer ca sa retinem acolo unde se intampla si sa normalizam 
        spectru_coloana = spectrograma[:, t]
        bucata_timp = np.fft.irfft(spectru_coloana, n=n_fft)
        start = t * hop_length
        end = start + n_fft
        semnal[start:end]+=bucata_timp*fereastra #adunam pentru ca atunci cand facem overlap nu vrem sa pierdem informatie, vrem sa fie cat mai smooth, de aceea e nevoie de normalizare
        norm_buffer[start:end]+=fereastra ** 2 #aceasta bucata a fost deformata cu patratul ferestri (o data ca sa evitam spectral leakage si ca atunci cand facem ifft sa ne asiguram ca unde se termina fereastra precedenta incepe noua fereastra semnalul sa fie continuu)
    norm_buffer[norm_buffer <1e-10] = 1.0 #evitam impartirea la 0. e posibil la capetele semnalului unde nu se suprapun ferestre
    semnal_final = semnal/norm_buffer #normalizarea
    return semnal_final
