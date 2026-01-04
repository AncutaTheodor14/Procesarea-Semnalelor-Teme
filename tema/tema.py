import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.datasets
from scipy.fft import dctn, idctn
from skimage import color

X = scipy.datasets.ascent() #matrice de pixeli
plt.imshow(X, cmap=plt.cm.gray)
plt.show()

#Exercitiul 1-> algoritmul jpeg
Q_down = 10
X_jpeg = X.copy()
X_jpeg = Q_down*np.round(X_jpeg/Q_down)
print(X_jpeg.shape)
N = X_jpeg.shape[0]

Q_jpeg = [[16, 11, 10, 16, 24, 40, 51, 61],
          [12, 12, 14, 19, 26, 28, 60, 55],
          [14, 13, 16, 24, 40, 57, 69, 56],
          [14, 17, 22, 29, 51, 87, 80, 62],
          [18, 22, 37, 56, 68, 109, 103, 77],
          [24, 35, 55, 64, 81, 104, 113, 92],
          [49, 64, 78, 87, 103, 121, 120, 101],
          [72, 92, 95, 98, 112, 100, 103, 99]]
X_reconst = np.zeros((N, N))
y_nnz = 0
y_jpeg_nnz = 0
for i in range(0, N//8):
    for j in range(0, N//8):
        bloc = X_jpeg[(i*8):(i+1)*8, (j*8):(j+1)*8]
        # plt.imshow(bloc, cmap=plt.cm.gray)
        # plt.show()
        y = dctn(bloc)
        # plt.imshow(y, cmap=plt.cm.gray)
        # plt.show()
        y_nnz += np.count_nonzero(y)
        y_bloc = Q_jpeg*np.round(y/Q_jpeg)
        y_jpeg_nnz += np.count_nonzero(y_bloc)
        bloc_reconst = idctn(y_bloc)
        # plt.imshow(bloc_reconst, cmap=plt.cm.gray)
        # plt.show()
        X_reconst[(i*8):(i+1)*8, (j*8):(j+1)*8] = bloc_reconst
plt.imshow(X_jpeg, cmap=plt.cm.gray)
plt.savefig('Exercitiul_1_poza_originala.pdf', format='pdf')
plt.show()
plt.imshow(X_reconst, cmap=plt.cm.gray)
plt.savefig('Exercitiul_1_poza_jpeg.pdf', format='pdf')
plt.show()
print('Componente în frecvență:' + str(y_nnz) +
      '\nComponente în frecvență după cuantizare: ' + str(y_jpeg_nnz))

#Exercitiul2
X_color = scipy.datasets.face()
plt.imshow(X_color)
plt.show()
print(X_color.shape)
N_color = X_color.shape[0]
M_color = X_color.shape[1]

X_ycbcr = color.rgb2ycbcr(X_color)
X_reconst = np.zeros((N_color, M_color, 3))
for canale in range(3):
    for i in range(0, N_color // 8):
        for j in range(0, M_color // 8):
            bloc = X_ycbcr[(i * 8):(i + 1) * 8, (j * 8):(j + 1) * 8, canale]
            y = dctn(bloc)
            y_bloc = Q_jpeg * np.round(y / Q_jpeg)
            bloc_reconst = idctn(y_bloc)
            X_reconst[(i * 8):(i + 1) * 8, (j * 8):(j + 1) * 8, canale] = bloc_reconst
X_reconst = color.ycbcr2rgb(X_reconst)
plt.imshow(X_color)
plt.savefig('Exercitiul_2_poza_originala.pdf', format='pdf')
plt.show()
plt.imshow(X_reconst)
plt.savefig('Exercitiul_2_poza_reconstruita.pdf', format='pdf')
plt.show()

#Exercitiul 3
def mse(imagine1, imagine2):
    return np.mean((imagine1 - imagine2)**2)

#folosesc algoritmul pentru imagini color de la ex 2
print('Valoare MSE cand aplic 1 data algoritmul ', mse(X_reconst, X_color)) #->15322.1229
MSE_limit = 15600
current_MSE = 0
X_color_1 = X_color.copy()
cnt = 0
k=8
#eroarea va creste pe masura ce eliminam mai multe frecvente, la inceput elimin cele mai mari frecvente care sunt rare
#deci MSE creste lent, apoi MSE creste rapid cand ajung la frecvente mai mici, prezente des in poza
X_ycbcr = color.rgb2ycbcr(X_color_1)
while current_MSE < MSE_limit and k>=0:
    X_reconst = np.zeros((N_color, M_color, 3))
    for canale in range(3):
        for i in range(0, N_color // 8):
            for j in range(0, M_color // 8):
                bloc = X_ycbcr[(i * 8):(i + 1) * 8, (j * 8):(j + 1) * 8, canale]
                y = dctn(bloc)
                y_bloc = Q_jpeg * np.round(y / Q_jpeg)
                y_bloc[k:, k:] = 0
                #y_bloc[:, k:] = 0
                bloc_reconst = idctn(y_bloc)
                X_reconst[(i * 8):(i + 1) * 8, (j * 8):(j + 1) * 8, canale] = bloc_reconst
    X_reconst = color.ycbcr2rgb(X_reconst)
    current_MSE = mse(X_reconst, X_color_1)
    cnt+=1
    k-=1
    print(f'Valoarea MSE dupa {cnt} operatii ', current_MSE)

#Exercitiul 4
cap = cv2.VideoCapture('video.mp4')
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('video_modificat.mp4', fourcc, fps, (frame_width, frame_height))
print(frame_width, frame_height, fps)
cnt = 0
skip = 1 #pentru ca video-ul e prea lung, iau cadre din 3 in 3
while True:
    ret, frame = cap.read()
    if ret:
        cnt+=1
        if cnt % skip != 0:
            continue
        print('Sunt la frame ul ', cnt)
        N_frame = frame.shape[0]
        M_frame = frame.shape[1]
        MSE_limit = 10
        current_MSE = 0
        k = 8
        X_ycbcr = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        while current_MSE < MSE_limit and k > 0:
            X_reconst = np.zeros((N_frame, M_frame, 3))
            for canale in range(3):
                for i in range(0, N_frame // 8):
                    for j in range(0, M_frame // 8):
                        bloc = X_ycbcr[(i * 8):(i + 1) * 8, (j * 8):(j + 1) * 8, canale].astype(np.float32)
                        y = dctn(bloc)
                        y_bloc = Q_jpeg * np.round(y / Q_jpeg)
                        y_bloc[k:, k:] = 0
                        bloc_reconst = idctn(y_bloc)
                        X_reconst[(i * 8):(i + 1) * 8, (j * 8):(j + 1) * 8, canale] = bloc_reconst
            #normalizez valorile pixelilor in intervalul coresp si ii fac din float in int
            X_reconst = np.clip(X_reconst, 0, 255).astype(np.uint8)
            X_reconst = cv2.cvtColor(X_reconst, cv2.COLOR_YCrCb2BGR)

            current_MSE = mse(X_reconst, frame)
            print('Valoare MSE ', current_MSE) #observ ca MSE creste cu cat elimin frecvente mai mici, deci mai comune, mai multe
            #range de la 2 la 21 aprox, o sa pun limita 10
            k-=1
        #normalizam
        out.write(X_reconst)
    else:
        break
#print('Numarul de framuri este: ', cnt)
cap.release()
out.release()