import numpy as np
from matplotlib import pyplot as plt

def sinusoidal(f, t):
    return np.sin(2*np.pi*f*t)

t = np.arange(0, 0.05, 1/5000)
t1 = np.arange(0, 0.05, 1/150)
f = 100

f1 = 400
f2 = 250

fig, axs = plt.subplots(4)
fig.suptitle('Exercitiul 2')
axs[0].plot(t, sinusoidal(f, t))
axs[1].plot(t, sinusoidal(f, t))
axs[1].scatter(t1, sinusoidal(f, t1), marker='o', color='r')
axs[2].plot(t, sinusoidal(f1, t))
axs[2].scatter(t1, sinusoidal(f, t1), marker='o', color='r')
axs[3].plot(t, sinusoidal(f2, t))
axs[3].scatter(t1, sinusoidal(f, t1), marker='o', color='r')
plt.savefig('Exercitiul_2.pdf', format='pdf')
plt.show()