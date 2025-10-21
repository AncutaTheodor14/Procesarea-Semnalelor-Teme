import numpy as np
from matplotlib import pyplot as plt


def x(A, f, phi, t):
    return A * np.sin(2*np.pi*f*t + phi)
pas = 0.00025
t = np.arange(0, 0.1+pas, pas)

z = np.random.normal(0, 1, len(t))
SNR = [0.1, 1, 10, 100]

A = 1
f = 50
phi = [np.pi/4, np.pi/2 + np.pi/4, np.pi + np.pi/4, 3*np.pi/2 + np.pi/4]
for inx, i in enumerate(phi):
    semnal = x(A, f, i, t)
    plt.plot(t, semnal)
plt.savefig('Exercitiul_2.pdf', format='pdf')
plt.show()

for j in range(len(SNR)):
    for inx, i in enumerate(phi):
        semnal = x(A, f, i, t)
        if inx == 0:
            copie = semnal.copy()
            gamma = np.sqrt(np.linalg.norm(semnal)**2 / (SNR[j] * (np.linalg.norm(z)**2)))
            copie = semnal + gamma*z
            plt.plot(t, copie)
    plt.savefig(f'Exercitiul_2_snr={SNR[j]}.pdf', format='pdf')
    plt.show()