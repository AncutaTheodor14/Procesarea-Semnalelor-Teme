import numpy as np
from matplotlib import pyplot as plt

f = 240
t = np.arange(0, 0.03, 1/1000)
def sinusoidal(t):
    return np.sin(2 * np.pi * t * f)

def sawtooth(t):
    return 2* (f * t - np.floor(f*t + 0.5))

fig, axs = plt.subplots(3)
fig.suptitle('Exercitiul_4')
axs[0].plot(t, sinusoidal(t))
axs[1].plot(t, sawtooth(t))
axs[2].plot(t, sinusoidal(t)+sawtooth(t))
plt.savefig('Exercitiul_4.pdf', format='pdf')
plt.show()
