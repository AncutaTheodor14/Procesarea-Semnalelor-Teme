import numpy as np
import scipy
from matplotlib import pyplot as plt

x = np.genfromtxt('Train.csv', delimiter=',')
x = x[1: ]

#a
x = x[:24*3]

#b
t = np.arange(0, 24*3)
plt.plot(t,x[:, 2], label='semnal original')
w = [5,9,13,17]
for i in w:
    nou = np.convolve(x[:,2], np.ones(i), 'valid') / i
    plt.plot(t[i-1:],nou, label=f'w={i}')
plt.legend()
plt.savefig('Exercitiul_6_b.pdf', format='pdf')
plt.show()
#filtrul face media ultimelor w puncte, w mare-> semnal mai neted

#c
# T = 1 ora = 60 *60 = 3600s perioada de esantionare, deci fv = 1/T = 1/3600
# fv nyquist = fv/2 = 1/7200
# aleg frecventa de taiere 12 ore, deci ce e sub 12 ore sa fie considerat zgomot, deci pastrez valorile de dimineata si seara
# frecventa taiere = 1/12 = 1/(12*3600)
# valoarea fv normalizate = 1/(12 * 3600) / 1/7200 = 2/12 =1/6

#d
Wn = 1/6
rp = 5
N = 5
b, a = scipy.signal.butter(N, Wn, btype='low')
b1, a1 = scipy.signal.cheby1(N, rp, Wn, btype='low')
w,h = scipy.signal.freqz(b, a)
plt.plot(w, 20*np.log10(abs(h)))
plt.savefig('butter.pdf', format='pdf')
plt.show()

w,h = scipy.signal.freqz(b1, a1)
plt.plot(w, 20*np.log10(abs(h)))
plt.savefig('cheby.pdf', format='pdf')
plt.show()

#e
x_butter = scipy.signal.filtfilt(b, a, x[:, 2])
x_cheby = scipy.signal.filtfilt(b1, a1, x[:, 2])
plt.plot(t, x[:, 2], label='original')
plt.plot(t, x_butter, label='butter')
plt.plot(t, x_cheby, label='cheby')
plt.legend()
plt.savefig('Exercitiul_6_e.pdf', format='pdf')
plt.show()
#aleg butterworth, e mai lin decat chebyshev, mai putine ondulatii si psatreaza trendul

#f
for val in [3, 5, 11]:
    b, a = scipy.signal.butter(val, Wn, btype='low')
    w, h = scipy.signal.freqz(b, a)
    x_butter = scipy.signal.filtfilt(b, a, x[:, 2])
    plt.plot(t, x[:, 2], label='original')
    plt.plot(t, x_butter, label='butter')
    plt.savefig(f'butter_N={val}.pdf', format='pdf')
    plt.show()

for val in [(2, 5), (5,3), (10, 5), (5, 11)]:
    rp = val[0]
    N = val[1]
    b, a = scipy.signal.cheby1(N, rp, Wn, btype='low')
    w, h = scipy.signal.freqz(b, a)
    x_cheby = scipy.signal.filtfilt(b, a, x[:, 2])
    plt.plot(t, x[:, 2], label='original')
    plt.plot(t, x_cheby, label='cheby')
    plt.savefig(f'cheby_N={N}_rp={rp}.pdf', format='pdf')
    plt.show()

#pentru cebysev, N=5, rp=2 pare cel mai aproape de realitate