from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt 
import mpl_toolkits.mplot3d
from matplotlib import cm
import sys
from numba import njit

@njit(parallel=True)
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
    symbol = ["2"]
    for s in symbol:
        img_ci = Image.open('./img/CI_{}.png'.format(s))
        img_ri = Image.open('./img/RI_{}.png'.format(s))
        ci = np.asarray(img_ci.convert('L'), dtype=np.float64)
        ri = np.asarray(img_ri.convert('L'), dtype=np.float64)
        k, k1 = main(ci, ri)


        maxi = np.argmax(k)
        max_index_j = int(maxi/k.shape[0])
        max_index_i = int(maxi%k.shape[0])
        print(max_index_i,max_index_j)
        draw = ImageDraw.Draw(img_ci)
        draw.rectangle((max_index_i, max_index_j, max_index_i+ri.shape[0], max_index_j+ri.shape[1]), outline="red")

        img_ci.show()

        """
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        X = np.arange(ci.shape[0]-ri.shape[0])
        Y = np.arange(ci.shape[1]-ri.shape[1])
        (x ,y) = np.meshgrid(X,Y)

        surf = ax.plot_surface(x, y, k, cmap=cm.coolwarm)
        fig.colorbar(surf, shrink=0.5, aspect=10)

        fig2 = plt.figure()
        ax2 = fig2.add_subplot(projection='3d')
        X = np.arange(ci.shape[0]-ri.shape[0])
        Y = np.arange(ci.shape[1]-ri.shape[1])
        (x ,y) = np.meshgrid(X,Y)

        surf2 = ax2.plot_surface(x, y, k1, cmap=cm.coolwarm)
        fig2.colorbar(surf2, shrink=0.5, aspect=10)

        fig1 = plt.figure()
        ax1 = fig1.add_subplot(projection='3d')
        X1 = np.arange(ci.shape[0])
        Y1 = np.arange(ci.shape[1])
        (x1 ,y1) = np.meshgrid(X1,Y1)
        surf1 = ax1.plot_surface(x1,y1, ci,cmap=cm.coolwarm)
        """

        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        cb_k = ax[0].imshow(k, cmap="gray")
        cb_k1 = ax[1].imshow(k1, cmap="gray")
        ax[0].set_title("Корреляционная функция")
        ax[1].set_title("Разностная функция")

        fig.colorbar(cb_k, ax=ax[0])
        fig.colorbar(cb_k1, ax=ax[1])

        plt.show()
