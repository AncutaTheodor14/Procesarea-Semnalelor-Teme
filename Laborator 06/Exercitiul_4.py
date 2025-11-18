import numpy as np

n = 20
d=5
x = np.linspace(0, 2, 20)
y = x.copy()
y = np.roll(y, d)
print(x)
print(y)

x_nou = np.fft.ifft(np.fft.fft(x)*np.fft.fft(y))
print(x_nou.real)

x_nou_1 = np.fft.ifft(np.fft.fft(y)*np.fft.fft(x))
print(x_nou_1.real)

#rezultatele sunt egale ca inmultirea e comutativa in domeniul frecventa
