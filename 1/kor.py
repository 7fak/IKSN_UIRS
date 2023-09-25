from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt 
import mpl_toolkits.mplot3d
from matplotlib import cm
import sys
#from numba import njit

#@njit(parallel=True)
#def main():
#    pass

if __name__ == "__main__":
    symbol = ["2"]
    for s in symbol:
        img_ci = Image.open('./img/CI_{}.png'.format(s))
        img_ri = Image.open('./img/RI_{}.png'.format(s))

        ci = np.asarray(img_ci.convert('L'))
        ri = np.asarray(img_ri.convert('L'))
        print(ci.dtype)

        print(ci.shape) # (256, 256)
        print(ri.shape) # (64, 64)

        k = np.zeros((ci.shape[0]-ri.shape[0], ci.shape[1]-ri.shape[1]), dtype=np.float64)
        k1 = np.zeros((ci.shape[0]-ri.shape[0], ci.shape[1]-ri.shape[1]), dtype=np.float64)
        i0 = ri.shape[0]
        j0 = ri.shape[1]
        m_ci = ci.mean()
        m_ri = ri.mean()

        
        for di in range(0, ci.shape[0]-i0, 1): #, int(ri.shape[0]/4)):
            for dj in range(0, ci.shape[1]-j0, 1): #, int(ri.shape[1]/4)):
                for i in range(0, i0):
                    for j in range(0, j0):
                        k[di, dj] +=(ri[i,j]-m_ri)*(ci[i+di, j+dj]-m_ci)
                        k1[di, dj] += (ri[i, j] - ci[i+di, j+dj])
                k1[di,dj] /= np.float64(i0*j0)
                k[di,dj] /= np.float64(i0*j0*np.sqrt(ci.var()*ri.var()))
                print("di,dj:", di,dj)
        

        
        maxi = np.argmax(k)
        max_index_j = int(maxi/k.shape[0])
        max_index_i = int(maxi%k.shape[0])
        print(max_index_i,max_index_j)
        draw = ImageDraw.Draw(img_ci)
        draw.rectangle((max_index_i, max_index_j, max_index_i+i0, max_index_j+j0), outline="red")
        #img_ci.show()
        

        #np.set_printoptions(threshold=sys.maxsize)
        #print(k)
        

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

        fig.colorbar(cb_k, ax=ax[0])
        fig.colorbar(cb_k1, ax=ax[1])

        plt.show()

    

