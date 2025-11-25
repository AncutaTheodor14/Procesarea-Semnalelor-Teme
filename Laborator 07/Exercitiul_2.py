import numpy as np
import scipy
from matplotlib import pyplot as plt

X = scipy.datasets.face(gray=True)
plt.imshow(X, cmap=plt.cm.gray)
plt.show()

SNR = 115
y = np.fft.fft2(X)
freq_db = 20*np.log10(abs(y))
Y_cutoff = y.copy()
Y_cutoff[freq_db > SNR] = 0
X_cutoff = np.fft.ifft2(Y_cutoff)
X_cutoff = np.real(X_cutoff)    # avoid rounding erros in the complex domain,
                                # in practice use irfft2
plt.imshow(X_cutoff, cmap=plt.cm.gray)
plt.savefig('Exercitiul_2.pdf', format='pdf')
plt.show()