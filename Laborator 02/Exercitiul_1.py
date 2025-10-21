import numpy as np
from matplotlib import pyplot as plt


def x(A, f, phi, t):
    return A * np.sin(2*np.pi*f*t + phi)
pas = 0.0005
t = np.arange(0, 0.1+pas, pas)
fig, axs = plt.subplots(2)
fig.suptitle('Exercitiul 1')
A = 2
f = 200
phi = 0
axs[0].plot(t, x(A, f, phi, t))
axs[1].plot(t, x(A, f, phi + np.pi/2, t))
plt.savefig('Exercitiul_1.pdf', format='pdf')
plt.show()