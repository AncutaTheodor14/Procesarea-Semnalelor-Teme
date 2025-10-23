import math

import numpy as np
from matplotlib import pyplot as plt

N = 64
M = np.zeros((N, N), dtype=complex)
for k in range(N):
    for i in range(N):
        M[k, i] = math.e**(1j* (-2) * np.pi * k * i/N)
fig, axs = plt.subplots(4)
fig.suptitle('Exercitiul_1')
for i in range(N):
    if i == 1:
        axs[i-1].plot(range(N), M[i].real)
        axs[i-1].plot(range(N), M[i].imag)
        axs[i].plot(range(N), M[-1].real)
        axs[i].plot(range(N), M[-1].imag)
    if i ==2:
        axs[i].plot(range(N), M[i].real)
        axs[i].plot(range(N), M[i].imag)
        axs[i+1].plot(range(N), M[-2].real)
        axs[i+1].plot(range(N), M[-2].imag)
plt.savefig('Exercitiul_1.pdf', format='pdf')
plt.show()

unitara = np.allclose(np.dot(M.conj().T, M), N*np.eye(N))
print(unitara)