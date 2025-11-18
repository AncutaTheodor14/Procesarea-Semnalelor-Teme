import numpy as np
from matplotlib import pyplot as plt

N = 100
x = np.random.rand(N)
t = np.arange(N)
fig, axs = plt.subplots(4)
fig.suptitle('Exercitiul_2')
axs[0].plot(t, x)
x = np.convolve(x,x)
t = np.arange(len(x))
axs[1].plot(t, x)
x = np.convolve(x,x)
t = np.arange(len(x))
axs[2].plot(t, x)
x = np.convolve(x, x)
t = np.arange(len(x))
axs[3].plot(t, x)
plt.savefig('Exercitiul_2.pdf', format='pdf')
plt.show()

#converg spre distributie gaussiana
N = 100
x = np.zeros(100)
x[40:61] = 1
t = np.arange(100)
fig, axs = plt.subplots(4)
fig.suptitle('Exercitiul_2_rectangular')
axs[0].plot(t, x)
x = np.convolve(x,x)
t = np.arange(len(x))
axs[1].plot(t, x)
x = np.convolve(x,x)
t = np.arange(len(x))
axs[2].plot(t, x)
x = np.convolve(x, x)
t = np.arange(len(x))
axs[3].plot(t, x)
plt.savefig('Exercitiul_2_rectangular.pdf', format='pdf')
plt.show()