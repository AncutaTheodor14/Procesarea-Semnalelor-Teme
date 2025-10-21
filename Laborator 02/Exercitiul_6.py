import numpy as np
from matplotlib import pyplot as plt


def x(A, f, phi, t):
    return A * np.sin(2*np.pi*f*t + phi)

fs = 10000
t = np.arange(0, 0.03, 1/fs)
fig, axs = plt.subplots(3)
fig.suptitle('Exercitiul_6')
axs[0].plot(t, x(1,fs/2, 0, t)) #frecventa e mare deci sinusoida face schimbari sus jos des
axs[1].plot(t, x(1,fs/4, 0, t)) #sinusoida normala, ajunge la valorile 1, -1
axs[2].plot(t, x(1,0, 0, t)) #o sa avem valoarea constanta sin(0) la orice esantion la care luam valoarea functiei
plt.savefig('Exercitiul_6.pdf', format='pdf')
plt.show()