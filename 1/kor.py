from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt 
import mpl_toolkits.mplot3d
from matplotlib import cm
import sys
from numba import njit

@njit(fastmath=True, parallel=True)
def main(ci, ri):
        k = np.zeros((ci.shape[0]-ri.shape[0], ci.shape[1]-ri.shape[1]), dtype=np.float64)
        k1 = np.zeros((ci.shape[0]-ri.shape[0], ci.shape[1]-ri.shape[1]), dtype=np.float64)
        i0 = ri.shape[0]
        j0 = ri.shape[1]
        m_ci = ci.mean()
        m_ri = ri.mean()

        
        for di in range(0, ci.shape[0]-i0, 1):                              # Перебор всех строк исходного изображения
            for dj in range(0, ci.shape[1]-j0, 1):                          # Перебор всех столбцов одной строки исходного изображения
                for i in range(0, i0):                                      # Перебор всех строк эталона
                    for j in range(0, j0):                                  # Перебор всех столбцов одной строки эталона
                        k[di, dj] +=(ri[i,j]-m_ri)*(ci[i+di, j+dj]-m_ci)    # Вычисление корреляционной функции для di,dj элемента
                        k1[di, dj] += np.abs(ri[i, j] - ci[i+di, j+dj])           # Вычисление разностной функции для di,dj элемента
                        #print((ri[i, j] - ci[i+di, j+dj]))
                k1[di,dj] /= np.float64(i0*j0)
                k[di,dj] /= np.float64(i0*j0*np.sqrt(ci.var()*ri.var()))
        #print(np.float64(i0*j0))
        

        return (k,k1)
        
if __name__ == "__main__":
    symbol = ["2", "7", "9"]
    for s in symbol:
        img_ci = Image.open('./img/CI_{}.png'.format(s))
        img_ri = Image.open('./img/RI_{}.png'.format(s))
        ci = np.asarray(img_ci.convert('L'), dtype=np.float64)
        ri = np.asarray(img_ri.convert('L'), dtype=np.float64)
        k, k1 = main(ci, ri)


        maxi = np.argmax(k)
        max_index_j = int(maxi/k.shape[0])
        max_index_i = int(maxi%k.shape[0])
        maxi_ = np.argmin(k1)
        max_index_j_ = int(maxi_/k.shape[0])
        max_index_i_ = int(maxi_%k.shape[0])

        draw = ImageDraw.Draw(img_ci)
        draw.rectangle((max_index_i, max_index_j, max_index_i+ri.shape[0], max_index_j+ri.shape[1]), outline="red")
        draw.rectangle((max_index_i_, max_index_j_, max_index_i_+ri.shape[0], max_index_j_+ri.shape[1]), outline="blue")
        print("Символ {}:\n\tКорр. функция:       ({}; {})\n\tРазностная функция:  ({}; {})".format(s,max_index_i,max_index_j, max_index_i_,max_index_j_))

        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        cb_k = ax[0].imshow(k, cmap="gray")
        cb_k1 = ax[1].imshow(k1, cmap="gray")
        ax[0].set_title("Корреляционная функция")
        ax[1].set_title("Разностная функция")
        ax[2].imshow(np.array(img_ci))

        fig.colorbar(cb_k, ax=ax[0])
        fig.colorbar(cb_k1, ax=ax[1])

        plt.show()
