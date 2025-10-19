import numpy as np
from matplotlib import pyplot as plt

#a
f = 400
ts = 0.1/1600
def sinusoidal(n):
    return np.sin(2* np.pi * n * ts * f)
plt.plot(ts * np.arange(1600), sinusoidal(np.arange(1600)))
plt.savefig('Exercitiul_2_a.pdf', format='pdf')
plt.show()

#b
f = 800
ts = 1/16000
def sinusoidal(n):
    return np.sin(2* np.pi * n * ts * f)
plt.plot(ts*np.arange(3*16000), sinusoidal(np.arange(3*16000)))
plt.savefig('Exercitiul_2_b.pdf', format='pdf')
plt.show()

#c
f = 240
def sawtooth(t):
    return 2* (f * t - np.floor(f*t + 0.5))
timp = np.arange(0, 0.05, 1/1000)
plt.plot(timp, sawtooth(timp))
plt.savefig('Exercitiul_2_c.pdf', format='pdf')
plt.show()

#d
f = 300
def square(t):
    return np.sign(np.sin(2*np.pi*t*f))
timp = np.arange(0, 0.05, 1/3000)
plt.plot(timp, square(timp))
plt.savefig('Exercitiul_2_d.pdf', format='pdf')
plt.show()

#e
semnal2D = np.random.rand(128,128)
plt.imshow(semnal2D)
plt.savefig('Exercitiul_2_e.pdf', format='pdf')
plt.show()

#f
semnal2D = np.zeros((128,128))
for i in range(0,128,2):
    for j in range(128):
        semnal2D[i][j]=1
for i in range(128):
    for j in range(0,128, 2):
        semnal2D[i][j]=1
plt.imshow(semnal2D)
plt.savefig('Exercitiul_2_f.pdf', format='pdf')
plt.show()
