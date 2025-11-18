import numpy as np
from matplotlib import pyplot as plt

def dreptunghiulara(N):
    return np.ones(N)
def hanning(N):
    n = np.arange(N)
    return 0.5 * (1-np.cos(2*np.pi*n/N))

f = 100
def sinusoidal(t):
    return np.sin(2*np.pi*f*t)

N = 200
t = np.linspace(0,0.1, 200)
plt.plot(t, sinusoidal(t) * dreptunghiulara(N))
plt.show()

fig, axs = plt.subplots(3)
fig.suptitle('Exercitiul 5')
axs[0].plot(t, sinusoidal(t))
axs[1].plot(t, sinusoidal(t) * dreptunghiulara(N))
axs[2].plot(t, sinusoidal(t) * hanning(N))
plt.savefig('Exercitiul_5.pdf', format='pdf')
plt.show()