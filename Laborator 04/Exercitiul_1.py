import math
import time
import numpy as np
from matplotlib import pyplot as plt

N = [128, 256, 512, 1024, 2048, 4096, 8192]

def fft_rapid(x):
    n = len(x)
    if n<=1:
        return x
    x_par = fft_rapid(x[::2])
    x_impar = fft_rapid(x[1::2])
    factor = math.e**(1j* (-2) * np.pi * np.arange(n)/n)
    first = x_par + factor[:n//2] * x_impar
    second = x_par - factor[:n//2] * x_impar
    return np.concatenate([first,second])

a=np.zeros(len(N))
b=np.zeros(len(N))
c=np.zeros(len(N))
k1=0
for n in N:
    M = np.zeros((n, n), dtype=complex)
    for k in range(n):
        for i in range(n):
            M[k, i] = math.e**(1j* (-2) * np.pi * k * i/n)
    x = np.random.rand(n)
    start = time.time()
    rez = np.dot(M, x)
    end = time.time()
    #print(f'Cazul discret: {end-start}')
    a[k1]=end-start
    start = time.time()
    X = fft_rapid(x)
    end = time.time()
    #print(f'Cazul continuu: {end-start}')
    b[k1]=end-start
    start = time.time()
    X = np.fft.fft(x)
    end = time.time()
    #print(f'Librarie: {end-start}')
    c[k1]=end-start
    k1+=1
plt.plot(N, a, label = 'Cazul discret')
plt.plot(N,b, label = 'Cazul rapid')
plt.plot(N,c, label = 'fft_librarie')
plt.yscale('log')
plt.legend()
plt.savefig('Exercitiul_1.pdf', format='pdf')
plt.show()