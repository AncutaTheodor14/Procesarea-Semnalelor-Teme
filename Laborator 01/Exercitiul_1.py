import numpy as np
from matplotlib import pyplot as plt

#exercitiul 1
#a
val_start = 0
val_end = 0.03
pas = 0.0005
axa_timp = np.arange(val_start, val_end + pas, pas)
print(axa_timp)
print(len(axa_timp))

#b
def x(t):
    return np.cos(520*np.pi *t + np.pi/3)

def y(t):
    return np.cos(280*np.pi *t - np.pi/3)

def z(t):
    return np.cos(120*np.pi *t + np.pi/3)

valori_x = x(axa_timp)
valori_y = y(axa_timp)
valori_z = z(axa_timp)

fig, axs = plt.subplots(3)
fig.suptitle('Exercitiul 1 b)')
axs[0].plot(axa_timp, valori_x)
axs[1].plot(axa_timp, valori_y)
axs[2].plot(axa_timp, valori_z)
plt.savefig('Exercitiul_1_b.pdf', format='pdf')
plt.show()

#c
T = 1/200
def x_discret(n):
    return x(n * T)
def y_discret(n):
    return y(n * T)
def z_discret(n):
    return z(n * T)
n=np.arange(0, val_end/T+1)
fig, axs = plt.subplots(3)
fig.suptitle('Exercitiul 1 c)')
axs[0].stem(n*T, x_discret(n))
axs[1].stem(n*T, y_discret(n))
axs[2].stem(n*T, z_discret(n))
axs[0].plot(n*T, x_discret(n))
axs[1].plot(n*T, y_discret(n))
axs[2].plot(n*T, z_discret(n))
plt.savefig('Exercitiul_1_c.pdf', format='pdf')
plt.show()
