import numpy as np
from matplotlib import pyplot as plt


def x(t, B):
    return np.sinc(B * t) * np.sinc(B * t)
def x_reconstruit(puncte, ts1, t, ts):
    sum = 0
    for i in range(len(ts1)):
        val = ts1[i]
        sum += puncte[i] * np.sinc((t-val)/ts)
    return sum


B = 1
t = np.arange(-3, 3+1/500, 1/500)
plt.plot(t, x(t,B))
plt.show()

#fs = 1, 1.5, 2, 4hz = 1/t, t=1,2/3,1/2,1/4
val = [1, 2/3, 1/2, 1/4]
for i in range(len(val)):
    ts1_pozitiv = np.arange(0, 3+val[i], val[i])
    ts1_negativ = -ts1_pozitiv
    ts1 = np.unique(np.concatenate((ts1_pozitiv, ts1_negativ)))
    plt.plot(t, x(t,B))
    puncte = x(ts1, B)
    plt.stem(ts1, puncte, markerfmt='ro', linefmt='r')
    plt.plot(t, x_reconstruit(puncte, ts1, t, val[i]), linestyle='--')
    plt.savefig(f'Exercitiul_1_fs={1/val[i]}.pdf', format='pdf')
    plt.title(f'fs = {1/val[i]}hz')
    plt.show()

#cu B=2 doar ultimul se suprapune