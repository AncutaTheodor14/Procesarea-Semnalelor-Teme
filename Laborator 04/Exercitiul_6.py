import math

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from scipy.io import wavfile

fs, x = wavfile.read("vocale.wav")
print(fs)
n = len(x)
print(n)
#audacity inregistreaza pe 2 canale
print(x.shape)
x = x[:, 0]

lungime_grup = n//100
pas = lungime_grup // 2
v = []
for i in range(0, n - lungime_grup, pas):
    v.append(x[i:i + lungime_grup])
v=np.array(v)
print(v.shape)

X = np.fft.fft(v, axis=1)
print(X.shape)

X = X.T
X = abs(X)

t = np.arange(X.shape[1])*pas/fs
f = np.linspace(0, fs/2, X.shape[0]//2)
extent = [0, t[-1], 0, f[-1]]
plt.imshow(X[:len(f)], extent = extent, aspect='auto', cmap='plasma', norm=LogNorm())
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.savefig('Exercitiul_6.pdf', format='pdf')
plt.show()