import numpy as np
import scipy
from matplotlib import pyplot as plt

X = scipy.datasets.face(gray=True)
plt.imshow(X, cmap=plt.cm.gray)
plt.show()

pixel_noise = 200

noise = np.random.randint(-pixel_noise, high=pixel_noise+1, size=X.shape)
X_noisy = X + noise
plt.imshow(X, cmap=plt.cm.gray)
plt.title('Original')
plt.show()
plt.imshow(X_noisy, cmap=plt.cm.gray)
plt.title('Noisy')
plt.show()

#SNR = 10 * log10 (P semnal/P zgomot)
#P zgomot = ||X - X_noisy||
pSemnal = np.sum(X**2)
pZgomot = np.sum((X - X_noisy)**2)
SNR = 10* np.log10(pSemnal/pZgomot)
print(SNR)

#elimin zgomotu, fac filtrare, zgomotu reprezinta frecvente mari
y = np.fft.fft2(X_noisy)
freq_db = 20*np.log10(abs(y)+ 1e-9)
Y_cutoff = y.copy()
Y_cutoff[freq_db > 150] = 0
X_cutoff = np.fft.ifft2(Y_cutoff)
X_cutoff = np.real(X_cutoff)

pnou = np.sum((X - X_cutoff)**2)
SNR = 10* np.log10(pSemnal/pnou)
print(SNR)

