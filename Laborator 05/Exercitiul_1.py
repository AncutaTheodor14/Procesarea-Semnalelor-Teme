import numpy as np
from matplotlib import pyplot as plt

#a
#fs = 1/ts, unde ts e perioada de esantionare
#ts = 1h = 60min = 3600 s, deci fs = 1/3600

#b
#avem 18288 esantioane, esantionate la 3600 de secunde
#deci intervalul e [0, 18287 * 3600] = [0, 18287 * 3600]

#c
#ca sa avem semnal corect, trebuie sa esantionam de cel putin 2 ori mai repede decat frecventa fundamentala
#deci frecventa maxima trebuie sa fie cel mult fv de esantionare/2, deci f e maxim 1/7200

#d
x = np.genfromtxt('Train.csv', delimiter=',')
print(len(x))
x = x[1: , 2]
t = np.arange(0, 18288 * 3600, 3600)
plt.plot(t, x)
plt.show()

x = x-np.mean(x)
x[0]=0 #dupa ce scad media, x[0] nu o sa fie chiar 0 din cauza erorilor de aproximare
#o setez 0 deci cand plotez cu log o sa am log 0 = -inf deci ii da skip
X = np.fft.fft(x)
print(abs(X[0]))
N = len(x)
fs = 1/3600
f = fs * np.linspace(0, N/2, N//2) / N
plt.plot(f, np.abs(X)[:N//2])
plt.yscale('log')
plt.savefig('Exercitiul_1_d.pdf', format='pdf')
plt.show()

#e
#am setat media 0

#f
val_modul = np.abs(X)[:N//2]
idx_valori_maxime = np.argsort(val_modul)[-4:]
valori_maxime = val_modul[idx_valori_maxime]
frecvente_maxime = f[idx_valori_maxime]
print('Modulul transformatei, Frecventa')
for i in range(len(frecvente_maxime)):
    print(valori_maxime[i], frecvente_maxime[i], sep=', ')

# Modulul transformatei, Frecventa
# 461234.7172957773, 4.557220460096978e-08
# 495604.05653598375, 1.1575339968646325e-05
# 644139.1878762624, 3.038146973397985e-08
# 1222670.8667623012, 1.5190734866989925e-08

# fv = 1.519 * 10^(-8) = 1/T, deci T = 65,832,784 sec = 18,286 ore = 761 zile ->perioada totala in care s a facut masuratoarea
# fv = 3.03 * 10^(-8) = 1/T, deci T = 33,003,300 sec = 9,167 ore = 381 zile -> perioada este cam de un an
# fv = 1.15 * 10^(-5) = 1/T. deci T = 86,956 sec = 24 ore = 1 zi -> aproximativ zi de zi comportamentul e repetitiv
# fv = 4.55 * 10^(-8) = 1/T, deci T = 21,978,021 sec = 6,105 ore = 254 zile -> aproximativ la fel in anotimpul cald/rece
# fft descompune semnalul in sinusoide (ideea e de caracter repetitiv) cu anumite fv si ne spune cat de importanta e fiecare

#g
#esantionam din ora in ora, o luna inseamna 24*30 = 720 ore = 2592000 secunde
start_esantion = 1008 #1008 = 7*144 -> sa fie zi de luni
timp = 720
x_nou = x[start_esantion:start_esantion + timp]
t = np.arange(timp)
plt.plot(t/24, x_nou) #timpul in zile
plt.savefig('Exercitiul_1_g.pdf', format='pdf')
plt.show()

#h
#ideea mea ar fi sa luam esantioane pe distanta mai mare(spre exemplu 1 pe zi sau saptamana, deci sa grupam aceste
# esantioane luate din ora in ora punanad media)si identificam perioadele cu trafic specific (de ex in cele
#3 luni de vacanta de vara ma astept sa am trafic mai redus, apoi perioade foarte aglomerate Craciun, revelion, anul nou
#si punandu-le in ordine as putea sa mi dau seama orientativ in ce luna incepe inregistrarea
#problema apare ca pot aparea sarbatori specifice in diferite zone, sa fiu sigur ca am acoperit toate aceste evenimente
#speciale si trebuie sa masor pe minim 1 an, deci sa trec prin toate anotimpurile, altfel metoda nu functioneaza bine

#i
#incerc sa elimin frecvente care au perioada <7 zile, deci ca la ideea de mai sus
#deci f sa fie 1.15 * 10^(-5)/7 (m am inspirat de la subpunctul f)

f_lim = 1.15 * (10 ** -5)/7
for i in range(len(X)//2):
    f_i = i*1/(3600*len(X))
    if f_i > f_lim:
        X[i]=0
        X[-i]=0
x_nou = np.fft.ifft(X).real
t = np.arange(0, 18288 * 3600, 3600)
t_zile = t/(3600*24)
plt.plot(t_zile, x_nou)
plt.savefig('Exercitiul_1_i.pdf', format='pdf')
plt.show()
