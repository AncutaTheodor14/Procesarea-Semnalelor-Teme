import math

import numpy as np
from matplotlib import pyplot as plt

f = 5
def sinusoidal(t):
    return np.sin(2*np.pi*f*t)

fs = 1000
timp = np.arange(0,1+1/fs, 1/fs)
plt.plot(timp, sinusoidal(timp))
plt.savefig('Exercitiul_2_figura_1_1.pdf', format='pdf')
plt.show()

def sinusoidal_pe_cerc(t):
    return sinusoidal(t) * math.e**(1j* (-2) * np.pi * t)

plt.plot(sinusoidal_pe_cerc(timp).real, sinusoidal_pe_cerc(timp).imag)
plt.savefig('Exercitiul_2_figura_1_2.pdf', format='pdf')
plt.show()

def z(omega, t):
    return sinusoidal(t) * math.e**(1j* (-2) * np.pi * omega *t)

valori_omega = [f, f+1, f+4, f+6]
k = 0
for i in valori_omega:
    k+=1
    plt.plot(z(i, timp).real, z(i, timp).imag)
    plt.savefig(f'Exercitiul_2_figura_2_{k}.pdf', format='pdf')
    plt.show()