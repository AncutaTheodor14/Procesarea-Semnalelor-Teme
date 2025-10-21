import numpy as np
import sounddevice


def sawtooth(t,f):
    return 2* (f * t - np.floor(f*t + 0.5))

t = np.arange(0, 0.5, 1/44100)
semnal1 = sawtooth(t, 100)
semnal2 = sawtooth(t, 200)
semnalnou = np.concatenate([semnal1, semnal2])

fs = 44100
sounddevice.play(semnalnou, fs)
sounddevice.wait()

#sunetul se aude mai intens cand trecem de la 100 la 200Hz