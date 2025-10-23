import math

import numpy as np
from matplotlib import pyplot as plt


def sinusoidal_1(A, t, f, fi):
    return A * np.sin(2*np.pi*f*t + fi)

def sinusoidal_2(A, t, f, fi):
    return A * np.sin(2*np.pi*f*t + fi)

def sinusoidal_3(A, t, f, fi):
    return A * np.sin(2*np.pi*f*t + fi)

def x(t):
    return sinusoidal_1(1, t, 200, 0) + sinusoidal_2(2, t, 400, np.pi/4)+ sinusoidal_3(1/2, t, 600, np.pi/6)

t = np.arange(0, 0.04, 1/2000)
plt.plot(t, x(t))
plt.savefig('Exercitiul_3_1.pdf', format='pdf')
plt.show()

N = len(t)
x_val = x(t)
X = np.zeros(N, dtype=complex)
for k in range(N):
    for n in range(N):
        X[k] += x_val[n] * math.e**(1j* (-2) * np.pi * k * n/N)

frecventa = np.arange(N) * 2000/N
plt.stem(frecventa, np.abs(X))
plt.xlim(0,1000)
plt.savefig('Exercitiul_3_2.pdf', format='pdf')
plt.show()

