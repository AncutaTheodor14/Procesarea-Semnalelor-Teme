import numpy as np
import scipy
import sounddevice
from matplotlib import pyplot as plt

#a
f = 400
fs= 44100
ts = 1/fs#0.1/1600
def sinusoidal(n):
    return np.sin(2* np.pi * n * ts * f)
semnal = sinusoidal(np.arange(fs))#1600
sounddevice.play(semnal, fs)
sounddevice.wait()

#b
f = 800
fs = 44100
ts = 1/fs
def sinusoidal(n):
    return np.sin(2* np.pi * n * ts * f)
semnal = sinusoidal(np.arange(3*fs))
sounddevice.play(semnal, fs)
sounddevice.wait()


#c
f = 240
fs = 44100
def sawtooth(t):
    return 2* (f * t - np.floor(f*t + 0.5))
timp = np.arange(0, 0.5, 1/fs)
semnal = sawtooth(timp)
sounddevice.play(semnal, fs)
sounddevice.wait()

#d
f = 300
fs = 44100
def square(t):
    return np.sign(np.sin(2*np.pi*t*f))
timp = np.arange(0, 0.5, 1/fs)
semnal = square(timp)
sounddevice.play(semnal, fs)
sounddevice.wait()
rate = int(1e5)
scipy.io.wavfile.write('Semnal', rate, semnal)
rate, x = scipy.io.wavfile.read('Semnal')

