import numpy as np
from matplotlib import pyplot as plt

#exercitiul 1
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

#Exercitiul 2
def matrice_hankel(serie, N, L):
    K = N-L+1
    m = np.zeros((L, K))
    for i in range(K):
        m[:,i] = serie[i:i+L]
    return m
L = 100
X = matrice_hankel(f, N, L)
print(X)

#Exercitiul 3
XXt = np.dot(X, X.T)
val_proprii, vect_proprii = np.linalg.eig(XXt)
XtX = np.dot(X.T, X)
val_proprii_1, vect_proprii_1 = np.linalg.eig(XtX)
U, s, Vt = np.linalg.svd(X)

ind = np.argsort(val_proprii)[::-1]
val_proprii_sortate = val_proprii[ind]
vect_proprii_sortati = vect_proprii[:, ind]

ind1 = np.argsort(val_proprii_1)[::-1]
val_proprii_1_sortate = val_proprii_1[ind1]
vect_proprii_1_sortati = vect_proprii_1[:, ind1]

print(s[:5])
print(val_proprii_sortate[:5])
print(val_proprii_1_sortate[:5])
#obs ca val proprii are lui XXt = XtX = valorile singulare ^2
print(vect_proprii_sortati[0, :])
print(U)

#Exercitiul 4

def hankel(M):
    L, K = M.shape
    N_reconst = L+K-1
    new_series = np.zeros(N_reconst)
    count_matrix = np.zeros(N_reconst)
    for i in range(L):
        for j in range(K):
            new_series[i+j] +=M[i,j]
            count_matrix[i+j] += 1
    return new_series/count_matrix

r = 4 #alegem primele r componente sa reconstruim semnalu
X_reconst = np.zeros(X.shape)
for i in range(r):
    vect_u = U[:,i]
    vect_v = Vt[i, :]
    sigma = s[i]
    Xi = sigma * np.outer(vect_u, vect_v)
    X_reconst += Xi
semnal_final = hankel(X_reconst)

plt.plot(t, f, label='Semnal original')
plt.plot(t, semnal_final, label='Semnal reconstruit')
plt.legend()
plt.savefig('Exercitiul_4_reconst.pdf', format='pdf')
plt.show()