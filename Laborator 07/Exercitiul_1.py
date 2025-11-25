import math

import scipy
import numpy as np
import matplotlib.pyplot as plt

X = scipy.datasets.face(gray=True)
plt.imshow(X, cmap=plt.cm.gray)
plt.show()

def Y(x):
    N1 = x.shape[0]
    N2 = x.shape[1]
    Y = np.zeros((N1,N2), dtype=complex)
    for m1 in range(N1):
        for m2 in range(N2):
            suma = 0.0j
            for n1 in range(N1):
                for n2 in range(N2):
                    suma += x[n1,n2] * np.exp(-2j * np.pi * (m1 * n1/N1 + m2*n2/N2))
            Y[m1,m2] = suma
    return Y

def x(n1, n2):
    return np.sin(2 * np.pi * n1 + 3 * np.pi * n2)
def x1(n1, n2):
    return np.sin(4*np.pi*n1) + np.cos(6*np.pi*n2)
n1 = np.arange(64)
n2 = np.arange(64)
N1, N2 = np.meshgrid(n1, n2, indexing='ij')
img = x(N1, N2)
plt.imshow(img, cmap='gray')
plt.savefig('Semnal_exemplu_1.pdf', format='pdf')
plt.show()

y = np.fft.fft2(img)
freq = 20*np.log10(abs(y)+1)
#faceam log din 0 si de aia dadea gri, am adunat 1
plt.imshow(freq, cmap='gray')
plt.savefig('Spectrul_exemplu_1.pdf', format='pdf')
plt.show()
#semnalul contine doar 2 frecvente puternice, simetrice, iar restul sunt nule
#de aia fundalul e negru complet, cele 2 fv puternice sunt 0 de aia vedem
#un singur punct (sin pi * (2*n1+3*n2))


img = x1(N1, N2)
plt.imshow(img, cmap='gray')
plt.savefig('Semnal_exemplu_2.pdf', format='pdf')
plt.show()

y = np.fft.fft2(img)
freq = 20*np.log10(abs(y)+1)
plt.imshow(freq, cmap='gray')
plt.savefig('Spectrul_exemplu_2.pdf', format='pdf')
plt.show()
#aici avem doar un punct, pentru ca semnalul nostru este 0+1 = 1 constant


Y = np.zeros((64,64), dtype=complex)
Y[0,5]=Y[0,64-5]=1
plt.imshow(np.abs(Y), cmap='gray')
plt.savefig('Spectru_exemplu_3.pdf', format='pdf')
plt.show()
#construiesc frecventele aici vedem 2 puncte, deci 2 fv inalte

semnal = np.fft.ifft2(Y)
plt.imshow(np.real(semnal), cmap='gray')
plt.savefig('Semnal_exemplu_3.pdf', format='pdf')
plt.show()
#observam ca variaza doar pe orizontala, pe verticala ramne la fel


Y = np.zeros((64,64), dtype=complex)
Y[5,0]=Y[64-5,0]=1
plt.imshow(np.abs(Y), cmap='gray')
plt.savefig('Spectru_exemplu_4.pdf', format='pdf')
plt.show()

semnal = np.fft.ifft2(Y)
plt.imshow(np.real(semnal), cmap='gray')
plt.savefig('Semnal_exemplu_4.pdf', format='pdf')
plt.show()
#ca mai sus, doar ca acum variaza pe verticala


Y = np.zeros((64,64), dtype=complex)
Y[5,5]=Y[64-5,64-5]=1
plt.imshow(np.abs(Y), cmap='gray')
plt.savefig('Spectru_exemplu_5.pdf', format='pdf')
plt.show()

semnal = np.fft.ifft2(Y)
plt.imshow(np.real(semnal), cmap='gray')
plt.savefig('Semnal_exemplu_5.pdf', format='pdf')
plt.show()
#aici vedem ca variaza si pe verticala si pe orizontala