import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import Lasso
from numpy import linalg as LA

# #exercitiul 1
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

#exercitiul 2
def model_AR(p, m):
    y_train = f[p:m]
    Y_train = np.zeros((m - p, p))
    for i in range(m - p):
        for j in range(p):
            Y_train[i, j] = f[p + i - j - 1]
    print(len(y_train), len(Y_train))
    x = np.dot(np.dot(np.linalg.inv(np.dot(Y_train.T, Y_train)), Y_train.T), y_train)
    print(x)
    print(np.dot(Y_train.T, Y_train))
    #predictia pe toata seria acum ca avem valorile x
    Y_total = np.zeros((N-p,p))
    for i in range(N-p):
        for j in range(p):
            Y_total[i,j] = f[p+i-j-1]
    y_pred_total = np.dot(Y_total, x)
    return y_pred_total, x

p = 2
m = 600
y_pred, x = model_AR(p, m)
plt.plot(t, f)
t_pred = t[p:]
plt.plot(t_pred, y_pred, linestyle='--')
plt.savefig('Exercitiul_2.pdf', format='pdf')
plt.show()

#exercitiul 3
#plecam cu un p mai mare si toate valorile x sunt 0 si incercam sa adaugam la fiecare pas cea mai buna coloana, cea
#care se coreleaza cel mai bine cu f, la inceput reziduu este y-0 = y si scopul e sa fie cat mai mic deci y prezis sa fie
#cat y
#repetam procesu de k ori
def model_AR_greedy(p, m, k):
    y_train = f[p:m]
    Y_train = np.zeros((m - p, p))
    for i in range(m - p):
        for j in range(p):
            Y_train[i, j] = f[p + i - j - 1]
    x_greedy = np.zeros(p)
    reziduu = y_train.copy()
    indici_activi = []
    for i in range(k):
        corelatii = np.dot(Y_train.T, reziduu)
        best_index = np.argmax(np.abs(corelatii))
        if best_index not in indici_activi:
            indici_activi.append(best_index)
        Y_partial = Y_train[:, indici_activi]
        x_partial = np.dot(np.dot(np.linalg.inv(np.dot(Y_partial.T, Y_partial)), Y_partial.T), y_train)
        reziduu = y_train - np.dot(Y_partial, x_partial)
        x_greedy[indici_activi] = x_partial
    return x_greedy, indici_activi

p = 50
m = 500
k = 10
x_greedy, indici_activi = model_AR_greedy(p, m, k)
print('Indicii care sunt != 0: ', indici_activi)
Y_total = np.zeros((N - p, p))
for i in range(N - p):
    for j in range(p):
        Y_total[i, j] = f[p + i - j - 1]
y_pred_total = np.dot(Y_total, x_greedy)
plt.plot(t, f)
plt.plot(t[p:], y_pred_total, linestyle='--')
plt.savefig('Exercitiul_3_greedy.pdf', format='pdf')
plt.show()

#folosind L1
p = 50
m = 500
l1 = Lasso(alpha=0.01) #alpha ne izce cat de sparse e solutia
y_train = f[p:m]
Y_train = np.zeros((m - p, p))
for i in range(m - p):
    for j in range(p):
        Y_train[i, j] = f[p + i - j - 1]
l1.fit(Y_train, y_train)
x_l1 = l1.coef_
val_x = []
for i in x_l1:
    if i !=0:
        val_x.append(i)
print('Indicii pastrati !=0: ', val_x)
Y_total = np.zeros((N - p, p))
for i in range(N - p):
    for j in range(p):
        Y_total[i, j] = f[p + i - j - 1]
y_pred = np.dot(Y_total, x_l1)
plt.plot(t, f)
plt.plot(t[p:], y_pred, linestyle='--')
plt.savefig('Exercitiul_3_l1.pdf', format='pdf')
plt.show()

#Exercitiul 4
def functie(coeficienti):
    coeficienti = coeficienti[::-1]
    N = len(coeficienti)
    C = np.zeros((N, N))
    for i in range(N):
        C[i, N-1] = coeficienti[i]
    for i in range(1, N):
        C[i, i-1] = 1
    val, vect = LA.eig(C)
    return val

#Exercitiul 5
coef = []
coef.extend(x)
radacini_polinom = functie(coef)
ok = 0
for i in radacini_polinom:
    val = np.abs(i)
    if val >= 1:
        ok = 1
        print('Seria de timp nu e stationara')
        break
if ok == 0:
    print('Serie de timp stationara')

coef = []
coef.extend(x_greedy)
radacini_polinom = functie(coef)
ok = 0
for i in radacini_polinom:
    val = np.abs(i)
    if val >= 1:
        ok = 1
        print('Seria de timp nu e stationara')
        break
if ok == 0:
    print('Serie de timp stationara')

coef = []
coef.extend(x_l1)
radacini_polinom = functie(coef)
ok = 0
for i in radacini_polinom:
    val = np.abs(i)
    if val >= 1:
        ok = 1
        print('Seria de timp nu e stationara')
        break
if ok == 0:
    print('Serie de timp stationara')
