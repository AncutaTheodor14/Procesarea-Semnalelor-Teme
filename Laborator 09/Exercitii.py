import numpy as np
from matplotlib import pyplot as plt

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

# #exercitiul 2
def mediere_exponentiala(alpha, x):
    s = np.zeros(N)
    s[0] = x[0]
    for i in range(1, N):
        s[i] = alpha*x[i] + (1-alpha)*s[i-1]
    return s

alpha_fixat = 0.3
noua_serie = mediere_exponentiala(alpha_fixat, f)
plt.plot(t, f)
plt.plot(t, noua_serie, linestyle='--')
plt.savefig('Exercitiul_2_mediere_exp_alfa_fix.pdf', format='pdf')
plt.show()

valori_alpha = np.linspace(0, 1, 100)
best = -1
serie1 = []
mini=np.inf
erori =[]
for i in valori_alpha:
    serie = mediere_exponentiala(i, f)
    sum = 0
    for j in range(0, N-1):
        sum+= ((serie[j]-f[j+1])**2)
    erori.append(sum)
    if sum <mini:
        mini=sum
        best=i
        serie1 = serie
print('Mediere exponentiala', best)
plt.plot(t, f)
plt.plot(t, serie1, linestyle='--')
plt.savefig('Exercitiul_2mediere_exp_alfa_calc.pdf', format='pdf')
plt.show()

plt.stem(valori_alpha[60:], erori[60:])
plt.scatter(best, mini, color='red', s=100, zorder=5)
plt.savefig('Exercitiul_2_best_alfa.pdf', format='pdf')
plt.show()

def mediere_exponentiala_dubla(alpha, x):
    s1 = mediere_exponentiala(alpha, x)
    s2 = mediere_exponentiala(alpha, s1)
    return s2

valori_alpha = np.linspace(0, 1, 100)
best = -1
serie1 = []
mini=np.inf
erori =[]
for i in valori_alpha:
    serie = mediere_exponentiala_dubla(i, f)
    sum = 0
    for j in range(0, N-1):
        sum+= ((serie[j]-f[j+1])**2)
    erori.append(sum)
    if sum <mini:
        mini=sum
        best=i
        serie1 = serie
print(best)
plt.plot(t, f)
plt.plot(t, serie1, linestyle='--')
plt.savefig('Exercitiul_2mediere_exp_dubla_alfa_calc.pdf', format='pdf')
plt.show()

plt.stem(valori_alpha, erori)
plt.scatter(best, mini, color='red', s=100, zorder=5)
plt.savefig('Exercitiul_2_dubla_best_alfa.pdf', format='pdf')
plt.show()

def mediere_exponentiala_tripla(alpha, x):
    s1 = mediere_exponentiala(alpha, x)
    s2 = mediere_exponentiala(alpha, s1)
    s3 = mediere_exponentiala(alpha, s2)
    return s3

valori_alpha = np.linspace(0, 1, 100)
best = -1
serie1 = []
mini=np.inf
erori =[]
for i in valori_alpha:
    serie = mediere_exponentiala_tripla(i, f)
    sum = 0
    for j in range(0, N-1):
        sum+= ((serie[j]-f[j+1])**2)
    erori.append(sum)
    if sum <mini:
        mini=sum
        best=i
        serie1 = serie
print(best)
plt.plot(t, f)
plt.plot(t, serie1, linestyle='--')
plt.savefig('Exercitiul_2mediere_exp_tripla_alfa_calc.pdf', format='pdf')
plt.show()

plt.stem(valori_alpha, erori)
plt.scatter(best, mini, color='red', s=100, zorder=5)
plt.savefig('Exercitiul_2_tripla_best_alfa.pdf', format='pdf')
plt.show()

# #rezultatele pentru toate 3 medierile sunt in jur de 1

#testam si pe o sinusoida
def f1(A, fv, phi, t):
    return A * np.sin(2*np.pi*fv*t + phi)

t = np.arange(0, 0.1, 0.0001)
fv=200
f1= f1(1, fv, 0, t)
plt.plot(t, f1)
plt.show()

valori_alpha = np.linspace(0, 1, 100)
best = -1
serie1 = []
mini=np.inf
erori =[]
for i in valori_alpha:
    serie = mediere_exponentiala(i, f1)
    sum = 0
    for j in range(0, N-1):
        sum+= ((serie[j]-f1[j+1])**2)
    erori.append(sum)
    if sum <mini:
        mini=sum
        best=i
        serie1 = serie
print('Mediere exponentiala', best)
plt.plot(t, f1)
plt.plot(t, serie1, linestyle='--')
plt.show()

alpha_fixat = 0.1
noua_serie = mediere_exponentiala(alpha_fixat, f1)
plt.plot(t, f)
plt.plot(t, noua_serie, linestyle='--')
plt.show()

#va functiona cand alfa e aproape de 1

#exercitiul 3
#folosim tot cele mai mici patrate, de data asta x este theta, Y e matricea formata din epsilon
q = 5
m = N
medie = np.mean(f)
sigma = np.std(f)
epsilon = np.random.normal(medie, sigma, N+m+q)
Y = np.zeros((N-q, q))
for i in range(N-q):
    for j in range(q):
        Y[i,j] = epsilon[q+i-j-1]
Z = f[q:N]-medie-epsilon[q:N]
theta=np.linalg.lstsq(Y, Z, rcond=None)[0]
print(theta)#[-0.18312275 -0.19767361 -0.16462424 -0.14936179 -0.17880456]

serie1 = list(f)
for i in range(N, N+m):
    serie1.append(medie+epsilon[i])
    for j in range(q):
        serie1[i] += theta[j] * epsilon[i-1-j]

serie1 = np.array(serie1)
plt.plot(np.arange(0, N, 1), f)
plt.plot(np.arange(0, N+m, 1), serie1, linestyle='--')
plt.savefig('Exercitiul_3.pdf', format='pdf')
plt.show()

#exercitiul 4
from statsmodels.tsa.arima.model import ARIMA

max_p = 10
max_q = 10
best=np.inf
best_p=-1
best_q=-1
for p in range(0, max_p+1, 2):
    for q in range(0, max_q+1, 2):
        model = ARIMA(f, order= (p,0,q)).fit()
        if model.aic <best:
            best=model.aic
            best_p=p
            best_q=q
print(best, best_p, best_q)
model_best = ARIMA(f, order = (best_p, 0, best_q)).fit()
pred = model_best.predict()
plt.plot(np.arange(0, N, 1), f)
plt.plot(np.arange(0, N, 1), pred)
plt.savefig('Exercitiul_4.pdf', format='pdf')
plt.show()

#calculez si eroarea acestui model folosind RMSE
rmse = np.sqrt(np.mean((f-pred)**2))
print(rmse)
