import numpy as np
from matplotlib import pyplot as plt

#a
N = 1000
def trend(t, a, b,c):
    return a*t*t + b*t + c
def sezon(t, f1, f2):
    return np.sin(2*np.pi*t*f1) + np.sin(2*np.pi*t*f2)
def variatii_mici(n):
    return np.random.normal(0, 0.2, n)

t = np.linspace(0, 2, N)
fig, axs = plt.subplots(4)
fig.suptitle('Exercitiul 1')
a = np.random.rand()
b = np.random.rand()
c = np.random.rand()
axs[0].plot(t, trend(t, a, b, c))
axs[1].plot(t, sezon(t, 10, 20))
axs[2].plot(t, variatii_mici(N))
f = trend(t, a, b, c) + sezon(t, 10, 20)+variatii_mici(N)
axs[3].plot(t, f)
plt.savefig('Exercitiul_1.pdf', format='pdf')
plt.show()

#b

val = np.correlate(f, f, "same")
print(val)
plt.plot(t, val)
plt.savefig('Exercitiul_1_b_numpy.pdf', format='pdf')
plt.show()

sum = 0
val = np.zeros(len(f))
for k in range(len(f)):
    sum=0
    for i in range(len(f)-k):
        sum += (f[i+k]*f[i])
    val[k] = sum
plt.plot(t, val)
plt.savefig('Exercitiul_1_b_functie.pdf', format='pdf')
plt.show()

#c
print(len(f))
p = 2
y = f[p:]
Y = np.zeros((len(f)-p, p))
for i in range(len(f)-p):
    for j in range(p):
        Y[i,j] = f[p+i-j-1]
print(len(y), len(Y))
x = np.dot(np.dot(np.linalg.inv(np.dot(Y.T, Y)), Y.T), y)
print(x)
print(np.dot(Y.T, Y))

y_pred = np.dot(Y, x)
plt.plot(t, f)
t_pred = t[p:]
plt.plot(t_pred, y_pred, linestyle='--')
plt.savefig('Exercitiul_1_c.pdf', format='pdf')
plt.show()

#d
p = [x for x in range(1,21)]
m = [100, 200, 300, 400, 500, 600, 700, 800]
minim = np.inf
val_p=0
val_m=0
for i in m:
    for j in p:
        f_train = f[:i]
        y = f_train[j:]
        Y = np.zeros((i-j, j))
        for i1 in range(i-j):
            for j1 in range(j):
                Y[i1,j1] = f_train[j+i1-j1-1]
        x = np.dot(np.dot(np.linalg.inv(np.dot(Y.T, Y)), Y.T), y)
        last_vals = f_train[i-j: i][::-1]
        pred = np.dot(x, last_vals)
        #calculez RMSE
        diferenta = np.sqrt(np.mean((f[i] - pred)**2))
        if diferenta < minim:
            minim = diferenta
            val_p = j
            val_m = i
print(minim, val_p, val_m)