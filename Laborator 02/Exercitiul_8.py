import numpy as np
from matplotlib import pyplot as plt


def sinus(x):
    return np.sin(x)
def prima_bisect(x):
    return x
def pade(x):
    return (x - 7*x*x*x/60)/(1+x*x/20)

t = np.arange(-np.pi/2, np.pi/2, 1/1000)
plt.plot(t, sinus(t))
plt.plot(t, prima_bisect(t))
plt.savefig('Exercitiul_8.pdf', format='pdf')
plt.show()

plt.plot(t, sinus(t)-prima_bisect(t))
plt.savefig('Exercitiul_8_eroarea.pdf', format='pdf')
plt.show()

plt.plot(t, sinus(t))
plt.plot(t, pade(t))
plt.savefig('Exercitiul_8_pade.pdf', format='pdf')
plt.show()

plt.plot(t, sinus(t)-pade(t))
plt.savefig('Exercitiul_8_pade_eroarea.pdf', format='pdf')
plt.show()

plt.plot(t, sinus(t)-prima_bisect(t))
plt.yscale('log')
plt.savefig('Exercitiul_8_yscale_log.pdf', format='pdf')
plt.show()

plt.plot(t, sinus(t)-pade(t))
plt.yscale('log')
plt.savefig('Exercitiul_8_pade_yscale_log.pdf', format='pdf')
plt.show()