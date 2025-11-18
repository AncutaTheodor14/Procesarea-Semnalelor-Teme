import numpy as np

N = 5
coefP = np.random.randint(-15, 15, N+1)
coefQ = np.random.randint(-15, 15, N+1)

#inmultirea directa
rez = np.zeros(11, dtype=int)
for i in range(N+1):
    for j in range(N+1):
        rez[i+j] += coefP[i]*coefQ[j]
print(rez)

#fft
P_fft = np.fft.fft(coefP, 11)
Q_fft = np.fft.fft(coefQ, 11)
rez = np.fft.ifft(P_fft * Q_fft)
print(rez.real)