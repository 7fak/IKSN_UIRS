# 1) Считать картинку с экрана
# 2) По событию выделить эталон
# 3) Получать координаты эталона
import numpy as np
import cv2 as cv
from numba import njit
import matplotlib.pyplot as plt 


def autoCorrel(ri):
    tmp = np.zeros_like(ri)
    tmp1 = np.hstack((tmp,tmp ,tmp))
    tmp2 = np.hstack((tmp,ri,tmp))
    f = np.vstack((tmp1, tmp2, tmp1)).astype(float)
    for i in range(1,5):
        k = kor(f,ri, 1, i)
        minval, maxval, min_i, max_i = cv.minMaxLoc(k)
        k = (k-minval)/(maxval-minval)
        print(i, "Автокорреляция", minval, maxval)

        cv.imshow("{} Canvas".format(i), f.astype(np.uint8))
        cv.imshow("{} Samokorrel".format(i), k)

@njit(fastmath=True)
def raznostnaya(ci, ri, ci_k=3, ri_k=2):
        (M, N) = ci.shape
        (i0, j0) = ri.shape

        k1 = np.zeros((M-i0, N-j0), dtype=np.float64)

        for di in range(0, M-i0, ci_k):
            for dj in range(0, N-j0,ci_k):
                for i in range(0, i0, ri_k):
                    for j in range(0, j0, ri_k):
                        k1[di, dj] += np.abs(ri[i, j] - ci[i+di, j+dj])
        k1 /= i0*j0/ri_k/ri_k
        
        return k1

@njit(fastmath=True, parallel=True)
def kor(ci, ri, ci_k=3, ri_k=2):
        (M, N) = ci.shape
        (i0, j0) = ri.shape
        ci_m = ci.mean()
        ri_m = ri.mean()
        k1 = np.zeros((M-i0, N-j0), dtype=np.float64)

        for di in range(0, M-i0, ci_k):
            for dj in range(0, N-j0,ci_k):
                for i in range(0, i0, ri_k):
                    for j in range(0, j0, ri_k):
                        k1[di, dj] += (ri[i, j]-ri_m)*(ci[i+di, j+dj]-ci_m)
        k1 /= (i0*j0/ri_k/ri_k*np.sqrt(ri.var()*ci.var()))
        
        return k1

@njit(fastmath=True)
def kor__(ci, ri):
    (M, N) = ci.shape
    ci_m = ci.mean()
    ri_m = ri.mean()
    k1 = 0

    for i in range(0, M):
        for j in range(0, N):
            k1 += (ri[i, j]-ri_m)*(ci[i, j]-ci_m)
    k1 /= (M*N*np.sqrt(ri.var()*ci.var()))

    return k1


vid = cv.VideoCapture(0) 
vid.set(cv.CAP_PROP_FRAME_WIDTH, 320)
vid.set(cv.CAP_PROP_FRAME_HEIGHT, 240)


def main():

    imgs = [ cv.cvtColor(cv.imread("etalons/{}.jpg".format(i)), cv.COLOR_RGB2GRAY) for i in range(6) ]
    
    ri_width = 50
    ri_height = 50
    ri_exist = False


    while(vid.isOpened()):
        ret, frame = vid.read()
        frame_grey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        (height, width , chanels) = frame.shape


        # Место выделения нового эталона
        frame = cv.rectangle(frame,
                                (int((width+ri_width)/2), int((height+ri_height)/2)),
                                (int((width-ri_width)/2), int((height-ri_height)/2)),
                                (100,100,0),
                                1)
        
        # Приведение к типу для корректной работы
        frame_grey_ = frame_grey.astype(float)
        if ri_exist:
            # Вычисление корреляционной функции
            k = kor(frame_grey_, ri)
            minval, maxval, min_i, max_i = cv.minMaxLoc(k)
            k = (k-minval)/(maxval-minval)
            
            #Потеря эталона
            if maxval < 0.9:
                frame = cv.putText(frame, "! OBJECT LOST !".format(minval), (20, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (50,50,250), 3, 2)     
            
            # Прямоугольник найденного эталона
            frame = cv.rectangle(frame,
                                    max_i,
                                    (max_i[0]+ri_height, max_i[1]+ri_width),
                                    (0,50,127),
                                    2)
            # Направление от эталона к центру
            frame = cv.arrowedLine(frame,
                                (int(max_i[0]+ri_height/2), int(max_i[1]+ri_width/2)),
                                (int((width+ri_width)/2), int((height+ri_height)/2)),
                                (50,50,50),
                                2)
            
            frame = cv.putText(frame, "max(K) = {:.2f}".format(maxval), (20, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (180,10,10), 3, 2)
            cv.imshow("Korrelatsionnaya", np.uint8(k*255.0))
            cv.imshow("Kernel",ri)

        
        # Вывод на экран
        cv.imshow('Video capture', frame)
        
        
        # Обработчик нажатия на клавишу
        res = cv.waitKey(1)
        if res & 0xFF == ord('q') or res == 27: 
            break
        if res & 0xFF == ord('s'):
            ri_ = frame_grey[int((height-ri_height)/2):int((height+ri_height)/2), int((width-ri_width)/2):int((width+ri_width)/2)]
            ri = ri_.astype(float)
            ri_exist = True
            k_list = [kor__(ri, imgs[i]) for i in range(6)]
            print(k_list)
            print("Метка ", k_list.index(max(k_list)))
            # autoCorrel(ri)
            print("Обновлено")
    
    vid.release()
    cv.destroyAllWindows() 


if __name__ == "__main__":
    main()