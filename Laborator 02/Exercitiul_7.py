import numpy as np
from matplotlib import pyplot as plt


def x(A, f, phi, t):
    return A * np.sin(2*np.pi*f*t + phi)

fs = 1000
t = np.arange(0, 0.03+ 1/fs, 1/fs)
f = 200
semnal = x(1, f, 0, t)
fig, axs = plt.subplots(3)
fig.suptitle('Exercitiul_7')
axs[0].plot(t, semnal)
axs[1].plot(t[::4], semnal[::4])
axs[2].plot(t[1::4], semnal[1::4])
plt.savefig('Exercitiul_7.pdf', format='pdf')
plt.show()
#pentru al doilea semnal, se observa ca se omite esantionarea in puncte importante, deci oscilatiile sunt mai rare
#daca incepem cu al doilea element, atunci avem acelasi semnal mutat mai la dreapta
